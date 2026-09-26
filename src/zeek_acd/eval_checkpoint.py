"""Score a trained checkpoint against any labeled Zeek dataset.

Two modes, because they answer different questions:

``--mode flat`` (default) walks every record exactly once with no
mitigation dynamics, and reports the action the policy picks for each true
class. This is the one to read when comparing datasets: no connection is
suppressed, so every policy sees the identical set of flows.

``--mode env`` replays the dataset through the Markov game the agent was
trained in (blocking really does suppress a source's later flows), which
is the number that is comparable with training-time eval output -- but
step counts then differ between policies.

Works with both checkpoint kinds: the plain DQN from ``train_dqn.py`` and
the Minimax-DQN from ``train.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from .data import load_conn_log_labeled
from .ensemble import RawExtractor, load_ensemble
from .env import ACDMarkovGameEnv
from .evaluate import make_constant_policy, rollout, summarize
from .features import FeatureExtractor, RunningNormalizer
from .game import AttackerClass, DefenderAction, PayoffTable, classify_label
from .live.run_agent import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, action="append",
                    help="repeat (with one --normalizer each) to score the Q-averaged "
                         "ensemble of several plain-DQN checkpoints")
    p.add_argument("--normalizer", required=True, action="append")
    p.add_argument("--data", required=True, help="glob for conn.log.labeled files")
    p.add_argument("--max-records-per-file", type=int, default=None)
    p.add_argument("--mode", choices=["flat", "env"], default="flat")
    p.add_argument("--episodes", type=int, default=10, help="--mode env only")
    p.add_argument("--episode-length", type=int, default=256, help="--mode env only")
    p.add_argument("--payoff", type=str, default=None,
                    help="JSON file with a PayoffTable to score against instead of the "
                         "built-in one (see payoffs/low_fp.json)")
    p.add_argument("--baselines", action="store_true",
                    help="also score always-ALLOW / always-LOG_ALERT / always-BLOCK_SRC")
    return p.parse_args()


def flat_eval(records, fx, policy, payoff: PayoffTable) -> dict:
    """Every record once, no suppression: pure per-flow decision quality."""
    per_class_actions = defaultdict(Counter)
    per_class_reward = defaultdict(float)
    per_class_n = Counter()
    total_reward = 0.0

    for record in records:
        cls = int(classify_label(record.get("label"), record.get("detailed-label")))
        vec = fx.transform(record, training=False)
        action, _scores, _value = policy(vec)
        reward = payoff.reward(action, cls)
        per_class_actions[cls][action] += 1
        per_class_reward[cls] += reward
        per_class_n[cls] += 1
        total_reward += reward

    result = {"avg_reward": total_reward / max(len(records), 1),
              "total_steps": len(records), "per_class": {}}
    for cls in AttackerClass:
        n = per_class_n.get(int(cls), 0)
        if not n:
            continue
        actions = per_class_actions[int(cls)]
        result["per_class"][cls.name] = {
            "n": n,
            "action_rate": sum(k for a, k in actions.items()
                               if a != int(DefenderAction.ALLOW)) / n,
            "avg_reward": per_class_reward[int(cls)] / n,
            "actions": {DefenderAction(a).name: round(k / n, 3)
                        for a, k in actions.most_common()},
        }
    return result


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()

    records = load_conn_log_labeled(args.data, args.max_records_per_file)
    if len(args.checkpoint) > 1:
        policy, kind = load_ensemble(args.checkpoint, args.normalizer), \
            f"ensemble({len(args.checkpoint)})"
        fx = RawExtractor()
    else:
        with open(args.normalizer[0]) as f:
            fx = FeatureExtractor(RunningNormalizer.from_dict(json.load(f)))
        policy, kind = load_policy(args.checkpoint[0])
    payoff = (PayoffTable.from_dict(json.load(open(args.payoff))) if args.payoff
              else PayoffTable.default())
    print(f"[eval] {len(records)} records from {args.data}, {kind} checkpoint "
          f"{', '.join(args.checkpoint)}, mode={args.mode}")

    if args.mode == "flat":
        print(summarize(f"{kind} checkpoint", flat_eval(records, fx, policy, payoff)))
        if args.baselines:
            for name, action in (("always-allow", DefenderAction.ALLOW),
                                 ("always-log-alert", DefenderAction.LOG_ALERT),
                                 ("always-block-src", DefenderAction.BLOCK_SRC)):
                const = lambda vec, a=int(action): (a, [], 0.0)  # noqa: E731
                print(summarize(f"baseline {name}", flat_eval(records, fx, const, payoff)))
        return

    env = ACDMarkovGameEnv(records, fx, payoff=payoff,
                           episode_length=args.episode_length, training=False, seed=1000)
    print(summarize(f"{kind} checkpoint",
                    rollout(env, lambda obs: policy(obs)[0],
                            n_episodes=args.episodes, seed=0)))
    if args.baselines:
        for name, action in (("always-allow", DefenderAction.ALLOW),
                             ("always-log-alert", DefenderAction.LOG_ALERT),
                             ("always-block-src", DefenderAction.BLOCK_SRC)):
            print(summarize(f"baseline {name}",
                            rollout(env, make_constant_policy(action),
                                    n_episodes=args.episodes, seed=0)))


if __name__ == "__main__":
    main()
