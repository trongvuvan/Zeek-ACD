"""Self-play training CLI: an attacker DQN and a defender DQN learn against
each other in the zero-sum game defined by ``selfplay.py``.

Two defender modes:
  --defender vanilla   (default) train a fresh belief-based defender DQN
                       jointly with the attacker.
  --defender frozen --defender-checkpoint PATH
                       load an already-trained defender (either kind --
                       plain DQN from train_dqn.py or Minimax-DQN from
                       train.py), FREEZE it, and train only the attacker
                       against it. This measures how exploitable a
                       deployed defender is: the attacker's average reward
                       is exactly how much it can beat that defender by,
                       and its action mix shows which attack it settled on.
                       ``minimax-frozen`` is the old name for this and
                       still works.

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

from .data import load_sources, train_eval_split
from .dqn import DQNAgent, DQNConfig, DQNTransition, SimpleReplayBuffer
from .env import ACDMarkovGameEnv
from .features import FeatureExtractor, RunningNormalizer
from .game import AttackerClass, DefenderAction, PayoffTable
from .live.run_agent import load_policy
from .selfplay import EVASION_MODES, SelfPlayACDEnv
from .synthetic import generate_synthetic_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, action="append", default=None,
                    help="glob for conn.log.labeled files; repeat for several datasets")
    p.add_argument("--repeat", type=int, action="append", default=None,
                    help="how many times to repeat the matching --data source (by position)")
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
    p.add_argument("--defender", choices=["vanilla", "frozen", "minimax-frozen"],
                    default="vanilla")
    p.add_argument("--defender-normalizer", type=str, default=None,
                    help="normalizer JSON saved next to --defender-checkpoint; a frozen "
                         "defender must see features scaled the way it was trained, so "
                         "pass this whenever you freeze one")
    p.add_argument("--defender-checkpoint", type=str, default=None,
                    help="required for --defender frozen")
    p.add_argument("--defender-init", type=str, default=None,
                    help="with --defender vanilla: start the trainable defender from this "
                         "plain-DQN checkpoint instead of from scratch. Self-play only "
                         "teaches a defender to handle an adaptive attacker; starting from "
                         "one that already works on real traffic keeps what it knows "
                         "instead of relearning it against a synthetic opponent")
    p.add_argument("--defender-lr", type=float, default=None,
                    help="override the learning rate of a --defender-init defender "
                         "(fine-tuning a working defender wants a smaller step)")
    p.add_argument("--defender-gamma", type=float, default=None,
                    help="override the defender's gamma (the v8 defenders use 0)")
    p.add_argument("--defender-eps", type=float, default=None,
                    help="constant exploration rate for the trainable defender instead of "
                         "the shared decay schedule; a warm-started defender exploring at "
                         "eps=1.0 would hand the attacker a random opponent early on")
    p.add_argument("--real-frac", type=float, default=0.0,
                    help="with --defender vanilla: fraction of every defender batch drawn "
                         "from replayed REAL traffic (the --data trace stepped through "
                         "env.ACDMarkovGameEnv each episode) instead of self-play. Joint "
                         "training with 0 wrecked the defender on real traffic: the "
                         "synthetic stream is too far from the real distribution")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_selfplay")
    p.add_argument("--payoff", type=str, default=None,
                    help="JSON file with a PayoffTable (see payoffs/low_fp.json); must "
                         "match what a frozen defender was trained with")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def epsilon_at(ep: int, args) -> float:
    frac = min(1.0, ep / max(1, args.eps_decay_episodes))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


class DefenderPolicy:
    """Uniform interface over the trainable defender and a frozen one."""

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


class FrozenDefender(DefenderPolicy):
    """Any already-trained defender, held fixed. ``load_policy`` works out
    which kind the checkpoint holds, so the same flag covers both."""

    trainable = False

    def __init__(self, checkpoint: str):
        self.policy, self.kind = load_policy(checkpoint)

    def act(self, obs):
        return self.policy(obs)[0]

    def select_action(self, obs, epsilon, rng):
        return self.act(obs)  # frozen: no exploration needed


def evaluate(env: SelfPlayACDEnv, attacker: DQNAgent, defender: DefenderPolicy,
             n_episodes: int) -> dict:
    total_att_reward = 0.0
    steps = 0
    suppressed_steps = 0
    attacker_classes = Counter()
    attacker_modes = Counter()
    per_class_seen = defaultdict(int)
    per_class_acted = defaultdict(int)

    for _ in range(n_episodes):
        a_obs = env.reset()
        done = False
        while not done:
            a_action = attacker.act(a_obs)
            # The action is a (class, evasion mode) pair; report the two
            # separately, since "what it attacks with" and "how it hides"
            # are different questions.
            chosen_class, chosen_mode = env.decode_action(a_action)
            attacker_classes[chosen_class] += 1
            attacker_modes[chosen_mode] += 1
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
        "attacker_class_mix": {
            AttackerClass(c).name: round(attacker_classes.get(c, 0) / max(steps, 1), 3)
            for c in range(env.n_attacker_classes)
        },
        "attacker_evasion_mix": {
            mode: round(attacker_modes.get(mode, 0) / max(steps, 1), 3)
            for mode in EVASION_MODES
        },
        "defender_detection": {},
    }
    for c in range(env.n_attacker_classes):
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
        f"  attacker class:   {r['attacker_class_mix']}",
        f"  attacker evasion: {r['attacker_evasion_mix']}",
    ]
    for cls, s in r["defender_detection"].items():
        lines.append(f"    defender vs {cls:16s} seen={s['seen']:5d} action_rate={s['action_rate']:.2f}")
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.data:
        train_records, eval_records = load_sources(
            args.data, args.repeat, args.max_records_per_file, seed=args.seed
        )
    else:
        print(f"[selfplay] no --data; generating {args.synthetic_n} synthetic records")
        records = generate_synthetic_records(args.synthetic_n, seed=args.seed)
        train_records, eval_records = train_eval_split(records, seed=args.seed)
    print(f"[selfplay] {len(train_records)} train / {len(eval_records)} eval records")

    # A frozen defender was trained against a particular feature scaling;
    # refitting a fresh normalizer here would hand it inputs it has never
    # seen and make it look far more exploitable than it is.
    if args.defender_normalizer:
        with open(args.defender_normalizer) as f:
            normalizer = RunningNormalizer.from_dict(json.load(f))
        print(f"[selfplay] using frozen normalizer {args.defender_normalizer}")
    else:
        normalizer = RunningNormalizer()
    fx_train = FeatureExtractor(normalizer)
    fx_eval = FeatureExtractor(normalizer)
    payoff = (PayoffTable.from_dict(json.load(open(args.payoff))) if args.payoff
              else PayoffTable.default())

    # A loaded normalizer stays fixed: updating it here would rescale the
    # (frozen or warm-started) defender's inputs towards the synthetic stream.
    update_norm = not args.defender_normalizer
    train_env = SelfPlayACDEnv(train_records, fx_train, payoff=payoff,
                               episode_length=args.episode_length, training=update_norm,
                               seed=args.seed)
    eval_env = SelfPlayACDEnv(eval_records, fx_eval, payoff=payoff,
                              episode_length=args.episode_length, training=False, seed=args.seed + 1000)

    attacker = DQNAgent(DQNConfig(
        state_dim=train_env.attacker_obs_dim, n_actions=train_env.n_attacker_actions,
        hidden=args.hidden, lr=args.lr, gamma=args.gamma))
    attacker_buf = SimpleReplayBuffer(args.buffer_size, train_env.attacker_obs_dim, seed=args.seed)

    if args.defender in ("frozen", "minimax-frozen"):
        if not args.defender_checkpoint:
            raise SystemExit(f"--defender {args.defender} requires --defender-checkpoint")
        defender = FrozenDefender(args.defender_checkpoint)
        defender_agent = None
        defender_buf = None
        print(f"[selfplay] frozen {defender.kind} defender from {args.defender_checkpoint}; "
              f"training attacker only")
    else:
        if args.defender_init:
            defender_agent = DQNAgent.load(args.defender_init)
            if args.defender_gamma is not None:
                defender_agent.cfg.gamma = args.defender_gamma
            if args.defender_lr is not None:
                defender_agent.cfg.lr = args.defender_lr
                for g in defender_agent.optimizer.param_groups:
                    g["lr"] = args.defender_lr
            print(f"[selfplay] defender warm-started from {args.defender_init} "
                  f"(lr={defender_agent.cfg.lr}, gamma={defender_agent.cfg.gamma})")
        else:
            defender_agent = DQNAgent(DQNConfig(
                state_dim=train_env.defender_obs_dim, n_actions=train_env.n_defender_actions,
                hidden=args.hidden, lr=args.lr, gamma=args.gamma))
        defender = VanillaDefender(defender_agent)
        defender_buf = SimpleReplayBuffer(args.buffer_size, train_env.defender_obs_dim, seed=args.seed + 7)
        print("[selfplay] training both attacker and belief-based (vanilla) defender")

    real_env = real_buf = None
    if args.real_frac > 0 and defender_agent is not None:
        real_env = ACDMarkovGameEnv(train_records, FeatureExtractor(normalizer), payoff=payoff,
                                    episode_length=args.episode_length, training=update_norm,
                                    seed=args.seed + 13)
        real_buf = SimpleReplayBuffer(args.buffer_size, train_env.defender_obs_dim,
                                      seed=args.seed + 17)
        print(f"[selfplay] {args.real_frac:.0%} of each defender batch from real traffic")
    n_real = int(round(args.batch_size * args.real_frac))

    def defender_batch():
        if real_buf is None or len(real_buf) < args.min_buffer or n_real == 0:
            return defender_buf.sample(args.batch_size)
        if n_real >= args.batch_size:
            return real_buf.sample(args.batch_size)
        return tuple(np.concatenate(parts) for parts in
                     zip(real_buf.sample(n_real), defender_buf.sample(args.batch_size - n_real)))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for ep in range(args.episodes):
        eps = epsilon_at(ep, args)
        d_eps = eps if args.defender_eps is None else args.defender_eps

        if real_env is not None:
            # One episode of the real trace for the defender alone, so its
            # replay never loses touch with the distribution it is deployed on.
            obs, _ = real_env.reset(seed=args.seed + ep)
            real_done = False
            while not real_done:
                act = defender.select_action(obs, d_eps, rng)
                nxt, r, term, trunc, _ = real_env.step(act)
                real_done = term or trunc
                real_buf.push(DQNTransition(obs, act, r, nxt, real_done))
                obs = nxt

        a_obs = train_env.reset()
        done = False
        # pending defender transition, finalized at its next decision or ep end
        pending_def = None  # (state, action, reward)

        while not done:
            a_action = attacker.select_action(a_obs, epsilon=eps, rng=rng)
            d_obs, realized_class, suppressed = train_env.attacker_step(a_action)

            if suppressed:
                d_reward, a_reward, next_a_obs, done = train_env.suppressed_step()
            else:
                d_action = defender.select_action(d_obs, d_eps, rng)
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
                defender_agent.train_step(defender_batch())

        # finalize the last pending defender transition as terminal
        if defender.trainable and pending_def is not None:
            ps, pa, pr = pending_def
            zero = np.zeros(train_env.defender_obs_dim, dtype=np.float32)
            defender_buf.push(DQNTransition(ps, pa, pr, zero, True))

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            r = evaluate(eval_env, attacker, defender, args.eval_episodes)
            print(f"[selfplay] episode {ep+1} eps={eps:.3f}")
            print(summarize(f"eval @ episode {ep+1}", r))
            attacker.save(ckpt_dir / f"attacker_ep{ep+1}.pt")
            attacker.save(ckpt_dir / "attacker_latest.pt")
            if defender.trainable:
                defender_agent.save(ckpt_dir / f"defender_ep{ep+1}.pt")
                defender_agent.save(ckpt_dir / "defender_latest.pt")
            for name in (f"normalizer_ep{ep+1}.json", "normalizer_latest.json"):
                with open(ckpt_dir / name, "w") as f:
                    json.dump(normalizer.to_dict(), f)

    print(f"[selfplay] done. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
