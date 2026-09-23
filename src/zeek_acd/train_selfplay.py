"""Self-play training CLI: an attacker DQN and a defender DQN learn against
each other in the zero-sum game defined by ``selfplay.py``.

Two defender modes:
  --defender vanilla   (default) train a fresh belief-based defender DQN
                       jointly with the attacker.
  --defender minimax-frozen --defender-checkpoint PATH
                       load an already-trained Minimax-DQN defender, FREEZE
                       it, and train only the attacker against it. This
                       measures how exploitable an existing defender is --
                       the attacker's average reward is exactly how much it
                       can beat the frozen defender by.

The attacker's checkpoint can then be plugged back into ``train.py``-style
training as an adaptive opponent (future work) or inspected to see which
attack strategies best defeat the current defender.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import load_and_split_per_file, train_eval_split
from .dqn import DQNAgent, DQNConfig, DQNTransition, SimpleReplayBuffer
from .features import FeatureExtractor, RunningNormalizer
from .game import AttackerClass, DefenderAction, PayoffTable
from .minimax_dqn import MinimaxDQNAgent
from .selfplay import SelfPlayACDEnv
from .synthetic import generate_synthetic_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--max-records-per-file", type=int, default=None)
    p.add_argument("--synthetic-n", type=int, default=20000)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--episode-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--min-buffer", type=int, default=1000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--defender", choices=["vanilla", "minimax-frozen"], default="vanilla")
    p.add_argument("--defender-checkpoint", type=str, default=None,
                    help="required for --defender minimax-frozen")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_selfplay")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def epsilon_at(ep: int, args) -> float:
    frac = min(1.0, ep / max(1, args.eps_decay_episodes))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


class DefenderPolicy:
    """Uniform interface over the trainable vanilla defender and the frozen
    minimax defender."""

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def select_action(self, obs, epsilon, rng) -> int:
        raise NotImplementedError

    trainable: bool = False


class VanillaDefender(DefenderPolicy):
    trainable = True

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, obs):
        return self.agent.act(obs)

    def select_action(self, obs, epsilon, rng):
        return self.agent.select_action(obs, epsilon=epsilon, rng=rng)


class FrozenMinimaxDefender(DefenderPolicy):
    trainable = False

    def __init__(self, agent: MinimaxDQNAgent):
        self.agent = agent

    def act(self, obs):
        pi, _ = self.agent.defender_strategy(obs)
        return int(np.argmax(pi))

    def select_action(self, obs, epsilon, rng):
        return self.act(obs)  # frozen: no exploration needed


def evaluate(env: SelfPlayACDEnv, attacker: DQNAgent, defender: DefenderPolicy,
             n_episodes: int) -> dict:
    total_att_reward = 0.0
    steps = 0
    suppressed_steps = 0
    attacker_choices = Counter()
    per_class_seen = defaultdict(int)
    per_class_acted = defaultdict(int)

    for _ in range(n_episodes):
        a_obs = env.reset()
        done = False
        while not done:
            a_action = attacker.act(a_obs)
            attacker_choices[a_action] += 1
            d_obs, realized_class, suppressed = env.attacker_step(a_action)
            if suppressed:
                _, a_reward, a_obs, done = env.suppressed_step()
                suppressed_steps += 1
            else:
                d_action = defender.act(d_obs)
                _, a_reward, a_obs, done = env.defender_step(d_action, realized_class)
                per_class_seen[realized_class] += 1
                if d_action != int(DefenderAction.ALLOW):
                    per_class_acted[realized_class] += 1
            total_att_reward += a_reward
            steps += 1

    result = {
        "attacker_avg_reward": total_att_reward / max(steps, 1),
        "defender_avg_reward": -total_att_reward / max(steps, 1),
        "suppressed_fraction": suppressed_steps / max(steps, 1),
        "attacker_action_mix": {
            AttackerClass(a).name: round(attacker_choices.get(a, 0) / max(steps, 1), 3)
            for a in range(env.n_attacker_actions)
        },
        "defender_detection": {},
    }
    for c in range(env.n_attacker_actions):
        seen = per_class_seen.get(c, 0)
        if seen:
            result["defender_detection"][AttackerClass(c).name] = {
                "seen": seen,
                "action_rate": round(per_class_acted.get(c, 0) / seen, 3),
            }
    return result


def summarize(tag: str, r: dict) -> str:
    lines = [
        f"== {tag} ==",
        f"  attacker_avg_reward={r['attacker_avg_reward']:+.3f}  "
        f"defender_avg_reward={r['defender_avg_reward']:+.3f}  "
        f"suppressed={r['suppressed_fraction']:.2f}",
        f"  attacker chose: {r['attacker_action_mix']}",
    ]
    for cls, s in r["defender_detection"].items():
        lines.append(f"    defender vs {cls:16s} seen={s['seen']:5d} action_rate={s['action_rate']:.2f}")
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.data:
        train_records, eval_records = load_and_split_per_file(
            args.data, args.max_records_per_file, seed=args.seed
        )
    else:
        print(f"[selfplay] no --data; generating {args.synthetic_n} synthetic records")
        records = generate_synthetic_records(args.synthetic_n, seed=args.seed)
        train_records, eval_records = train_eval_split(records, seed=args.seed)
    print(f"[selfplay] {len(train_records)} train / {len(eval_records)} eval records")

    normalizer = RunningNormalizer()
    fx_train = FeatureExtractor(normalizer)
    fx_eval = FeatureExtractor(normalizer)
    payoff = PayoffTable.default()

    train_env = SelfPlayACDEnv(train_records, fx_train, payoff=payoff,
                               episode_length=args.episode_length, training=True, seed=args.seed)
    eval_env = SelfPlayACDEnv(eval_records, fx_eval, payoff=payoff,
                              episode_length=args.episode_length, training=False, seed=args.seed + 1000)

    attacker = DQNAgent(DQNConfig(
        state_dim=train_env.attacker_obs_dim, n_actions=train_env.n_attacker_actions,
        hidden=args.hidden, lr=args.lr, gamma=args.gamma))
    attacker_buf = SimpleReplayBuffer(args.buffer_size, train_env.attacker_obs_dim, seed=args.seed)

    if args.defender == "minimax-frozen":
        if not args.defender_checkpoint:
            raise SystemExit("--defender minimax-frozen requires --defender-checkpoint")
        defender = FrozenMinimaxDefender(MinimaxDQNAgent.load(args.defender_checkpoint))
        defender_agent = None
        defender_buf = None
        print(f"[selfplay] frozen minimax defender from {args.defender_checkpoint}; training attacker only")
    else:
        defender_agent = DQNAgent(DQNConfig(
            state_dim=train_env.defender_obs_dim, n_actions=train_env.n_defender_actions,
            hidden=args.hidden, lr=args.lr, gamma=args.gamma))
        defender = VanillaDefender(defender_agent)
        defender_buf = SimpleReplayBuffer(args.buffer_size, train_env.defender_obs_dim, seed=args.seed + 7)
        print("[selfplay] training both attacker and belief-based (vanilla) defender")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for ep in range(args.episodes):
        a_obs = train_env.reset()
        eps = epsilon_at(ep, args)
        done = False
        # pending defender transition, finalized at its next decision or ep end
        pending_def = None  # (state, action, reward)

        while not done:
            a_action = attacker.select_action(a_obs, epsilon=eps, rng=rng)
            d_obs, realized_class, suppressed = train_env.attacker_step(a_action)

            if suppressed:
                d_reward, a_reward, next_a_obs, done = train_env.suppressed_step()
            else:
                d_action = defender.select_action(d_obs, eps, rng)
                d_reward, a_reward, next_a_obs, done = train_env.defender_step(d_action, realized_class)
                if defender.trainable:
                    if pending_def is not None:
                        ps, pa, pr = pending_def
                        defender_buf.push(DQNTransition(ps, pa, pr, d_obs, False))
                    pending_def = (d_obs, d_action, d_reward)

            attacker_buf.push(DQNTransition(a_obs, a_action, a_reward, next_a_obs, done))
            a_obs = next_a_obs
            global_step += 1

            if len(attacker_buf) >= args.min_buffer:
                attacker.train_step(attacker_buf.sample(args.batch_size))
            if defender.trainable and defender_buf is not None and len(defender_buf) >= args.min_buffer:
                defender_agent.train_step(defender_buf.sample(args.batch_size))

        # finalize the last pending defender transition as terminal
        if defender.trainable and pending_def is not None:
            ps, pa, pr = pending_def
            zero = np.zeros(train_env.defender_obs_dim, dtype=np.float32)
            defender_buf.push(DQNTransition(ps, pa, pr, zero, True))

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            r = evaluate(eval_env, attacker, defender, args.eval_episodes)
            print(f"[selfplay] episode {ep+1} eps={eps:.3f}")
            print(summarize(f"eval @ episode {ep+1}", r))
            attacker.save(ckpt_dir / "attacker_latest.pt")
            if defender.trainable:
                defender_agent.save(ckpt_dir / "defender_latest.pt")
            with open(ckpt_dir / "normalizer_latest.json", "w") as f:
                json.dump(normalizer.to_dict(), f)

    print(f"[selfplay] done. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
