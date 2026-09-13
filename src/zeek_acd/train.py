"""CLI entry point: trains the Minimax-DQN defender against a Zeek
connection trace (a real IoT-23 ``conn.log.labeled`` glob, or synthetic
data for a quick smoke test) and checkpoints it for offline evaluation or
live deployment (see ``live/run_agent.py``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_conn_log_labeled, train_eval_split
from .env import ACDMarkovGameEnv
from .evaluate import make_agent_policy, rollout, summarize
from .features import FeatureExtractor, RunningNormalizer
from .game import PayoffTable
from .minimax_dqn import MinimaxDQNAgent, MinimaxDQNConfig
from .replay_buffer import ReplayBuffer
from .synthetic import generate_synthetic_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default=None,
                    help="glob for conn.log.labeled files, e.g. 'iot23/*/conn.log.labeled'")
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
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def epsilon_at(ep: int, args: argparse.Namespace) -> float:
    frac = min(1.0, ep / max(1, args.eps_decay_episodes))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def main() -> None:
    # This network is tiny (a few hundred hidden units); torch's default
    # multi-threaded CPU backward/optimizer has fixed per-call overhead
    # that dwarfs the actual compute at this scale (~40ms vs ~2.5ms per
    # step measured single- vs multi-threaded), so force single-threaded.
    torch.set_num_threads(1)

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.data:
        records = load_conn_log_labeled(args.data)
    else:
        print(f"[train] no --data given, generating {args.synthetic_n} synthetic records "
              f"for a smoke test (use --data to train on real Zeek/IoT-23 logs)")
        records = generate_synthetic_records(args.synthetic_n, seed=args.seed)

    train_records, eval_records = train_eval_split(records, seed=args.seed)
    print(f"[train] {len(train_records)} train records, {len(eval_records)} eval records")

    normalizer = RunningNormalizer()
    fx_train = FeatureExtractor(normalizer)
    fx_eval = FeatureExtractor(normalizer)

    payoff = PayoffTable.default()
    train_env = ACDMarkovGameEnv(
        train_records, fx_train, payoff=payoff,
        episode_length=args.episode_length, training=True, seed=args.seed,
    )
    eval_env = ACDMarkovGameEnv(
        eval_records, fx_eval, payoff=payoff,
        episode_length=args.episode_length, training=False, seed=args.seed + 1000,
    )

    config = MinimaxDQNConfig(
        state_dim=fx_train.dim, hidden=args.hidden, lr=args.lr, gamma=args.gamma, tau=args.tau,
    )
    agent = MinimaxDQNAgent(config)
    buffer = ReplayBuffer(args.buffer_size, state_dim=fx_train.dim, seed=args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start = time.time()
    reward_history = []

    for ep in range(args.episodes):
        obs, info = train_env.reset(seed=args.seed + ep)
        eps = epsilon_at(ep, args)
        done = False
        ep_reward = 0.0
        losses = []

        while not done:
            action, _pi = agent.select_action(obs, epsilon=eps, rng=rng)
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, info["attacker_class"], reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            global_step += 1

            if len(buffer) >= args.min_buffer and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size)
                losses.append(agent.train_step(batch))

        reward_history.append(ep_reward)
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_r = np.mean(reward_history[-10:])
            avg_loss = np.mean(losses) if losses else float("nan")
            elapsed = time.time() - start
            print(f"[train] ep={ep+1:5d} eps={eps:.3f} avg_reward(10)={avg_r:+.3f} "
                  f"loss={avg_loss:.4f} buffer={len(buffer)} elapsed={elapsed:.1f}s")

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            policy = make_agent_policy(agent, greedy=True)
            result = rollout(eval_env, policy, n_episodes=args.eval_episodes, seed=args.seed)
            print(summarize(f"eval @ episode {ep+1}", result))
            ckpt_path = ckpt_dir / f"agent_ep{ep+1}.pt"
            agent.save(ckpt_path)
            with open(ckpt_dir / f"normalizer_ep{ep+1}.json", "w") as f:
                json.dump(normalizer.to_dict(), f)
            with open(ckpt_dir / f"payoff.json", "w") as f:
                json.dump(payoff.to_dict(), f)
            print(f"[train] checkpoint saved to {ckpt_path}")

    latest = ckpt_dir / "agent_latest.pt"
    agent.save(latest)
    with open(ckpt_dir / "normalizer_latest.json", "w") as f:
        json.dump(normalizer.to_dict(), f)
    print(f"[train] done. latest checkpoint: {latest}")


if __name__ == "__main__":
    main()
