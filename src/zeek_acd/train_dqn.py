"""CLI entry point: trains a *plain* (non-minimax) DQN defender on the same
Zeek trace, same environment and same payoff table as ``train.py``.

Why this exists. The Minimax-DQN defender in ``train.py`` extracts its
action by solving a matrix game over *all* attacker-class columns, so it
hedges against every class at every state and converges to ``LOG_ALERT``
for everything (see "Known behavior" in the README). Worse, its
``Q(s, a_d, a_a)`` entries are only ever trained on the one column that
actually occurred, so each column ends up reproducing that column of the
payoff table with almost no state dependence -- the network never has to
tell the classes apart.

A plain DQN has no class axis: it learns ``Q(s, a_d)`` directly from the
realized reward, so the only way to score well is to *infer* the class
from the connection's features and act on that belief. This is the
"act on your own belief" alternative the README points at, and the
head-to-head baseline for the minimax run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_sources, train_eval_split
from .dqn import DQNAgent, DQNConfig, DQNTransition, SimpleReplayBuffer
from .env import ACDMarkovGameEnv
from .evaluate import make_constant_policy, rollout, summarize
from .features import FeatureExtractor, RunningNormalizer
from .game import DefenderAction, PayoffTable
from .synthetic import generate_synthetic_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, action="append", default=None,
                    help="glob for conn.log.labeled files, e.g. 'iot23/*/conn.log.labeled'; "
                         "repeat the flag to train on several datasets at once")
    p.add_argument("--repeat", type=int, action="append", default=None,
                    help="how many times to repeat the matching --data source's training "
                         "records (aligned by position, default 1); lets a small dataset "
                         "weigh as much as a large one")
    p.add_argument("--max-records-per-file", type=int, default=None,
                    help="keep only the first N rows of each --data file (balances scenarios)")
    p.add_argument("--synthetic-n", type=int, default=20000,
                    help="used only when --data is not given")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--min-buffer", type=int, default=1000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_dqn")
    p.add_argument("--payoff", type=str, default=None,
                    help="JSON file with a PayoffTable to use instead of the built-in one "
                         "(see payoffs/low_fp.json); this table is the main lever on the "
                         "false-positive / missed-detection trade-off")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def epsilon_at(ep: int, args: argparse.Namespace) -> float:
    frac = min(1.0, ep / max(1, args.eps_decay_episodes))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def make_dqn_policy(agent: DQNAgent):
    def policy(obs: np.ndarray) -> int:
        return agent.act(obs)
    return policy


def main() -> None:
    # Same reasoning as train.py: the net is small enough that torch's
    # multi-threaded CPU path is pure overhead.
    torch.set_num_threads(1)

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.data:
        train_records, eval_records = load_sources(
            args.data, args.repeat, args.max_records_per_file, seed=args.seed
        )
    else:
        print(f"[dqn] no --data given, generating {args.synthetic_n} synthetic records")
        records = generate_synthetic_records(args.synthetic_n, seed=args.seed)
        train_records, eval_records = train_eval_split(records, seed=args.seed)
    print(f"[dqn] {len(train_records)} train records, {len(eval_records)} eval records")

    normalizer = RunningNormalizer()
    fx_train = FeatureExtractor(normalizer)
    fx_eval = FeatureExtractor(normalizer)

    payoff = (PayoffTable.from_dict(json.load(open(args.payoff))) if args.payoff
              else PayoffTable.default())
    train_env = ACDMarkovGameEnv(
        train_records, fx_train, payoff=payoff,
        episode_length=args.episode_length, training=True, seed=args.seed,
    )
    eval_env = ACDMarkovGameEnv(
        eval_records, fx_eval, payoff=payoff,
        episode_length=args.episode_length, training=False, seed=args.seed + 1000,
    )

    agent = DQNAgent(DQNConfig(
        state_dim=fx_train.dim, n_actions=int(train_env.action_space.n),
        hidden=args.hidden, lr=args.lr, gamma=args.gamma, tau=args.tau,
    ))
    buffer = SimpleReplayBuffer(args.buffer_size, state_dim=fx_train.dim, seed=args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start = time.time()
    reward_history: list[float] = []

    for ep in range(args.episodes):
        obs, info = train_env.reset(seed=args.seed + ep)
        eps = epsilon_at(ep, args)
        done = False
        ep_reward = 0.0
        losses = []

        while not done:
            action = agent.select_action(obs, epsilon=eps, rng=rng)
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            buffer.push(DQNTransition(obs, action, reward, next_obs, done))
            obs = next_obs
            ep_reward += reward
            global_step += 1

            if len(buffer) >= args.min_buffer and global_step % args.train_every == 0:
                losses.append(agent.train_step(buffer.sample(args.batch_size)))

        reward_history.append(ep_reward)
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_r = np.mean(reward_history[-10:])
            avg_loss = np.mean(losses) if losses else float("nan")
            print(f"[dqn] ep={ep+1:5d} eps={eps:.3f} avg_reward(10)={avg_r:+.3f} "
                  f"loss={avg_loss:.4f} buffer={len(buffer)} elapsed={time.time()-start:.1f}s")

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            result = rollout(eval_env, make_dqn_policy(agent),
                             n_episodes=args.eval_episodes, seed=args.seed)
            print(summarize(f"eval @ episode {ep+1}", result))
            agent.save(ckpt_dir / f"agent_ep{ep+1}.pt")
            with open(ckpt_dir / f"normalizer_ep{ep+1}.json", "w") as f:
                json.dump(normalizer.to_dict(), f)
            with open(ckpt_dir / "payoff.json", "w") as f:
                json.dump(payoff.to_dict(), f)

    # Baselines on the same eval split, for context on the final numbers.
    for name, action in (("always-allow", DefenderAction.ALLOW),
                         ("always-log-alert", DefenderAction.LOG_ALERT),
                         ("always-block-src", DefenderAction.BLOCK_SRC)):
        print(summarize(f"baseline {name}",
                        rollout(eval_env, make_constant_policy(action),
                                n_episodes=args.eval_episodes, seed=args.seed)))

    agent.save(ckpt_dir / "agent_latest.pt")
    with open(ckpt_dir / "normalizer_latest.json", "w") as f:
        json.dump(normalizer.to_dict(), f)
    print(f"[dqn] done. latest checkpoint: {ckpt_dir / 'agent_latest.pt'}")


if __name__ == "__main__":
    main()
