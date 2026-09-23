# zeek-acd — Documentation

An Active Cyber Defense (ACD) agent that decides, per network connection,
what to do about it (allow it, alert on it, rate-limit it, block its
source, isolate the local host, or redirect it to a decoy). It's trained
with a DQN-style neural network as a **zero-sum stochastic game** between
a defender and an attacker, observing **Zeek** connection telemetry.

This document explains what every part of the code does and how to run
it, end to end. For a shorter orientation, see `../README.md`.

## Table of contents

1. [The idea, in plain terms](#1-the-idea-in-plain-terms)
2. [Setup](#2-setup)
3. [Quick start: smoke test on synthetic data](#3-quick-start-smoke-test-on-synthetic-data)
4. [Training on real data (IoT-23)](#4-training-on-real-data-iot-23)
5. [Evaluating a trained agent](#5-evaluating-a-trained-agent)
6. [Running against a live Zeek instance](#6-running-against-a-live-zeek-instance)
7. [Self-play: training a symmetric attacker](#7-self-play-training-a-symmetric-attacker)
8. [Architecture: file by file](#8-architecture-file-by-file)
9. [Configuration & tuning](#9-configuration--tuning)
10. [CLI reference](#10-cli-reference)
11. [Known limitations](#11-known-limitations)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. The idea, in plain terms

**Zeek** is a network monitor that turns raw traffic into structured logs
— `conn.log` records one line per connection (who talked to whom, for how
long, how many bytes, how the connection ended, etc.). That's the raw
material here: every decision the agent makes is based on one `conn.log`
record.

**DQN** (Deep Q-Network) is a reinforcement learning method: instead of
being told the right answer for each input, the agent tries actions,
observes a reward, and learns a function `Q(state, action)` estimating
how good each action is in each state. Here the "state" is a Zeek
connection's features, and the actions are the defensive responses listed
above.

**Stochastic game theory** enters because we don't just want the average
best response — a good defense should hold up even in the worst case, not
just against average/expected traffic. We frame the problem as a
**zero-sum Markov game**: the defender picks a defensive action, an
"attacker" is modeled as picking a traffic category (benign, port scan,
DoS, C2/botnet, other malicious), and the defender's reward is exactly
the negative of the attacker's. The algorithm that finds a solution to
this kind of game — **minimax-Q** (Littman, 1994) — is normally defined
for small tabular state spaces; this project replaces the table with a
neural network, i.e. **Minimax-DQN**, so it can generalize across the
huge space of possible connection feature vectors.

Putting it together: **ACD** (Active Cyber Defense) is the umbrella term
for a system that doesn't just detect attacks but automatically responds
to them. This project is one such system: Zeek supplies the telemetry,
Minimax-DQN supplies the decision policy, and a live connector turns that
policy's decisions into actual (or, by default, merely logged/proposed)
defensive actions.

## 2. Setup

Requires Python ≥ 3.10.

```bash
cd /root/zeek-acd
python3 -m venv .venv
.venv/bin/pip install -e .
```

This installs the package (`zeek_acd`) in editable mode along with its
dependencies: `numpy`, `scipy` (for the exact minimax linear program),
`gymnasium` (environment API), and `torch` (the Q-network). All commands
below assume you run them from `/root/zeek-acd` using `.venv/bin/python`
(or activate the venv with `source .venv/bin/activate` first and drop the
`.venv/bin/` prefix).

No GPU is required or used — the network is small enough that CPU
training is fast (see [§11](#11-troubleshooting) for why threading matters
here).

## 3. Quick start: smoke test on synthetic data

You don't need Zeek or a real dataset to check that the whole pipeline
works. `train.py` generates synthetic, Zeek-shaped connection records
when you don't pass `--data`:

```bash
.venv/bin/python -m zeek_acd.train --synthetic-n 12000 --episodes 300 \
  --episode-length 128 --eval-every 100 --checkpoint-dir checkpoints
```

This will:
1. Generate 12,000 synthetic `conn.log`-shaped records across 5 traffic
   classes (benign + 4 attack types).
2. Split them into train/eval segments.
3. Train a Minimax-DQN agent for 300 episodes (~35k environment steps).
4. Every 100 episodes, run a greedy evaluation rollout and print, per
   true traffic class: how often the agent acted (anything other than
   `ALLOW`) and the average reward — for `BENIGN` this "action rate" is
   your false-positive rate; for the attack classes it's your detection
   rate.
5. Save a checkpoint (`agent_ep<N>.pt` + `normalizer_ep<N>.json`) after
   every eval, plus `agent_latest.pt` / `normalizer_latest.json` /
   `payoff.json` at the end.

On a modern CPU core this takes roughly 10-15 minutes. Expect output like:

```
[train] ep=  300 eps=0.050 avg_reward(10)=-0.625 loss=0.0027 buffer=38400 elapsed=696.3s
== eval @ episode 300 == avg_reward=0.052 steps=1024
  BENIGN           n=  620  action_rate=1.00  avg_reward=-0.050
  RECON            n=  124  action_rate=1.00  avg_reward=+0.300
  DOS              n=   90  action_rate=1.00  avg_reward=+0.100
  C2               n=  100  action_rate=1.00  avg_reward=+0.200
  OTHER_MALICIOUS  n=   90  action_rate=1.00  avg_reward=+0.200
```

Don't be alarmed that `BENIGN` action_rate is `1.00` here — see
[§10](#10-known-limitations) for why the default minimax policy converges
to a uniformly-cautious `LOG_ALERT` and how to change that if you want the
agent to act on its own confidence instead.

## 4. Training on real data (IoT-23)

For real training data, use the
[Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23) dataset
— it ships Zeek-native `conn.log.labeled` files (standard Zeek TSV plus
`label`/`detailed-label` columns) from IoT malware captures: benign
traffic plus port scans, DDoS, C&C/botnet beaconing, etc.

```bash
# download/extract one or more scenario captures somewhere, e.g. iot23/
# each scenario directory has a bro/ (Zeek) folder containing conn.log.labeled

.venv/bin/python -m zeek_acd.train \
  --data 'iot23/*/bro/conn.log.labeled' \
  --episodes 800 --episode-length 256 \
  --checkpoint-dir checkpoints
```

`--data` takes a glob; every matching file is loaded and concatenated.
Everything else (evaluation, checkpointing) works the same as the
synthetic quick start.

## 5. Evaluating a trained agent

`evaluate.py` provides rollout-based evaluation and baseline policies to
compare against, for use in a script or notebook:

```python
from zeek_acd.data import load_conn_log_labeled, train_eval_split
from zeek_acd.env import ACDMarkovGameEnv
from zeek_acd.evaluate import make_agent_policy, make_constant_policy, rollout, summarize
from zeek_acd.features import FeatureExtractor, RunningNormalizer
from zeek_acd.game import DefenderAction, PayoffTable
from zeek_acd.minimax_dqn import MinimaxDQNAgent
import json

records = load_conn_log_labeled("iot23/*/bro/conn.log.labeled")
_, eval_records = train_eval_split(records)

normalizer = RunningNormalizer.from_dict(json.load(open("checkpoints/normalizer_latest.json")))
fx = FeatureExtractor(normalizer)
env = ACDMarkovGameEnv(eval_records, fx, payoff=PayoffTable.default(), training=False)

agent = MinimaxDQNAgent.load("checkpoints/agent_latest.pt")
print(summarize("trained agent", rollout(env, make_agent_policy(agent))))
print(summarize("always allow", rollout(env, make_constant_policy(DefenderAction.ALLOW))))
print(summarize("always block", rollout(env, make_constant_policy(DefenderAction.BLOCK_SRC))))
```

`rollout()` runs the policy through the environment (so `BLOCK_SRC`'s
suppression effect is included, not just a one-shot per-flow score) and
`summarize()` formats the per-class breakdown shown above.

## 6. Running against a live Zeek instance

Once you have a checkpoint, `live/run_agent.py` tails a running Zeek
instance's `conn.log` and scores each connection as it's written:

```bash
.venv/bin/python -m zeek_acd.live.run_agent \
  --checkpoint checkpoints/agent_latest.pt \
  --normalizer checkpoints/normalizer_latest.json \
  --log-path /path/to/zeek/logs/current/conn.log \
  --audit-log acd_audit.jsonl
```

It auto-detects TSV vs. JSON Zeek log format, follows log rotation, and
for each connection prints and records: the connection's `uid`, its
source, the chosen defender action, the full mixed strategy (so you can
see how confident the decision was), and the game's estimated value.

**Read this before you enable real enforcement.** By default
(`--executor log_only`, which is also what happens if you omit
`--executor`), the agent **never touches the network**. It only appends a
JSON line per decision to `--audit-log`. This is intentional and meant to
be the mode you run in first: review the audit log, check the
false-positive rate on your real traffic, and only then consider
enforcement.

`--executor nftables` adds a second layer of intent-checking: even with
`nftables` selected, actions are **still only logged, not executed**,
unless you *also* pass `--live-enforce`. You'll also need to pre-create
the nftables table/sets it expects — see the docstring at the top of
`live/action_executor.py` for the exact `nft` commands. Every source IP is
validated with `ipaddress.ip_address()` before it's placed in an `nft`
command argument list (passed as separate argv elements, never through a
shell), so malformed or hostile-looking log data can't be used to inject
commands — but validation only rules out injection, not false positives,
which is what the audit-log dry run is for.

## 7. Self-play: training a symmetric attacker

Everything above trains the *defender* against whatever traffic mix a
dataset happens to contain — the attacker's "move" at each step is just
read from the ground-truth label. Self-play makes the attacker a **second
learning agent** that actively tries to defeat the defender, which (a)
directly addresses the "attacker doesn't adapt" limitation and (b)
produces a defender that's robust to an adversary reacting to its policy,
not just to a fixed empirical traffic distribution.

**Nothing here touches a real network** — it's pure simulation, an
extension of the same zero-sum Markov game. The attacker doesn't run
scans or send packets; it *chooses a class of traffic to emit*
(`BENIGN`/`RECON`/`DOS`/`C2`/`OTHER_MALICIOUS`) and the environment draws
a real Zeek record of that class from the dataset to produce what the
defender sees. See `selfplay.py`'s module docstring for the exact game
formulation.

### The two modes

**Joint self-play (default)** — train a fresh belief-based defender and
the attacker together:

```bash
.venv/bin/python -m zeek_acd.train_selfplay \
  --synthetic-n 12000 --episodes 600 --episode-length 128 \
  --checkpoint-dir checkpoints_selfplay
```

The defender here is a plain (non-minimax) DQN that acts on its own
argmax belief — the "acts on its confidence" variant discussed in the
limitations section — because that pairs symmetrically with the attacker
and makes the cat-and-mouse legible. Outputs `attacker_latest.pt`,
`defender_latest.pt`, and `normalizer_latest.json`.

**Frozen-defender attack (exploitability test)** — take a defender you
already trained with `train.py` (a Minimax-DQN), freeze it, and train
*only* the attacker against it:

```bash
.venv/bin/python -m zeek_acd.train_selfplay \
  --defender minimax-frozen \
  --defender-checkpoint checkpoints/agent_latest.pt \
  --synthetic-n 12000 --episodes 400 \
  --checkpoint-dir checkpoints_attack
```

This is the most useful diagnostic: the attacker's average reward at
convergence is *exactly how much a learning adversary can beat your
deployed defender by*, and the printed "attacker chose" mix tells you
*which* attack types slip past it. A near-zero attacker reward means the
frozen defender is hard to exploit; a large positive one means it has a
blind spot the attacker found.

### Reading the eval output

Each eval prints, from the attacker's point of view:

```
== eval @ episode 400 ==
  attacker_avg_reward=+0.123  defender_avg_reward=-0.123  suppressed=0.31
  attacker chose: {'BENIGN': 0.22, 'RECON': 0.05, 'DOS': 0.4, 'C2': 0.28, 'OTHER_MALICIOUS': 0.05}
    defender vs BENIGN          seen= 900 action_rate=0.10
    defender vs DOS             seen=1200 action_rate=0.88
    ...
```

- `attacker_avg_reward` / `defender_avg_reward`: the zero-sum score. Watch
  which way it moves over training — that's who's currently winning.
- `suppressed`: fraction of steps the attacker was blocked/rate-limited
  (its traffic never landed). High = the defender is successfully cutting
  it off.
- `attacker chose`: the attacker's learned strategy mix. Emergent evasive
  behavior shows up here — e.g. leaning on whichever class the defender
  detects worst, or mixing in `BENIGN` to avoid getting blocked.
- `defender vs <class>` action rate: the defender's detection rate on the
  traffic the attacker actually decided to send.

### What to do with the trained attacker

The immediate payoff is the diagnostic above. Turning the learned attacker
into an opponent that the *minimax defender* trains against inside
`train.py` (closing the loop into full adversarial co-training) is the
natural next step and is noted as future work — the pieces (attacker
agent, self-play env) are all here, but `train.py` currently still reads
the attacker move from labels.

## 8. Architecture: file by file

```
src/zeek_acd/
  game.py             action spaces, attacker-class label mapping, payoff table
  zeeklog.py          Zeek TSV log parsing (shared by offline data and live tailing)
  features.py         conn.log record -> normalized feature vector
  env.py              Gymnasium environment implementing the zero-sum Markov game
  replay_buffer.py     uniform experience replay
  minimax_dqn.py        QNetwork + minimax-Q solvers + MinimaxDQNAgent
  data.py              loads real conn.log.labeled files, train/eval split
  synthetic.py         generates Zeek-shaped synthetic data for smoke-testing
  train.py             defender training CLI (attacker move read from labels)
  evaluate.py          rollout-based evaluation + baseline policies
  dqn.py               plain (single-agent) Double-DQN, used by the self-play attacker
  selfplay.py          two-sided self-play env (attacker + defender both learn)
  train_selfplay.py    self-play training CLI (joint, or attacker-vs-frozen-defender)
  live/
    zeek_tail.py        follows a live conn.log (TSV or JSON, rotation-aware)
    action_executor.py  log_only / nftables backends; always audit-logged
    run_agent.py         live inference CLI
```

### `game.py`

Defines the two action spaces as `IntEnum`s:

- `DefenderAction`: `ALLOW, LOG_ALERT, RATE_LIMIT, BLOCK_SRC, ISOLATE_HOST, DECEIVE`
- `AttackerClass`: `BENIGN, RECON, DOS, C2, OTHER_MALICIOUS`

`classify_label(label, detailed_label)` maps a dataset's ground-truth
label strings (IoT-23's convention) to an `AttackerClass` — this is what
the environment uses to compute the reward for a given connection; it's
never given to the network as an input feature (the network only sees
`features.py`'s output).

`PayoffTable` holds the defender's reward for every
`(defender_action, attacker_class)` pair. Since the game is zero-sum, the
attacker's reward is the negation of this value — there's no separate
attacker payoff table. `PayoffTable.default()` is a hand-picked, plausible
but **not calibrated** table (see [§8](#8-configuration--tuning)); this is
the main lever for adapting the agent's behavior to your actual cost of
false positives vs. missed attacks.

### `zeeklog.py`

A small parser for Zeek's standard ASCII TSV log format: it reads the
`#separator`/`#fields`/`#types`/... header lines Zeek writes at the top of
every log, then parses subsequent lines into `dict[str, str]` records
keyed by field name. It also understands IoT-23's convention of appending
two undeclared trailing columns (`label`, `detailed-label`) to
`conn.log.labeled` files. This same parser backs both `data.py` (static
files) and `live/zeek_tail.py` (a live, growing file).

### `features.py`

`raw_feature_vector(record)` turns one Zeek conn record into a fixed
60-dimensional numeric vector: continuous fields (duration, byte/packet
counts, a couple of engineered ratios/logs), one-hot encodings of
`proto`/`conn_state`/`service`, and per-character counts of the `history`
flag string (Zeek's compact encoding of a connection's TCP flag/event
sequence).

`RunningNormalizer` is a Welford's-algorithm online mean/variance tracker
applied only to the continuous features (one-hot indicators are left
alone). `FeatureExtractor` wraps a normalizer and a `transform(record,
training=...)` call: pass `training=True` during training so the
normalizer keeps adapting to the data it sees, and `training=False` at
evaluation/live time so a checkpoint's saved normalizer stays fixed. The
normalizer is serialized to/from JSON (`RunningNormalizer.to_dict/from_dict`)
so a saved `normalizer_*.json` travels with a checkpoint.

### `env.py` — `ACDMarkovGameEnv`

A `gymnasium.Env` subclass. State transitions are driven by replaying a
list of Zeek records (`self.records`) — the *order* of connections is
fixed/exogenous (it's whatever the dataset or live tail gives you), but
what makes this a genuine Markov game rather than a per-flow supervised
bandit is that the defender's action changes what happens next:

- `BLOCK_SRC` / `ISOLATE_HOST` add the connection's source IP to a
  permanent (for the rest of the episode) blocklist; every subsequent
  record from that source is skipped when advancing the environment,
  simulating the source's traffic actually being cut off.
- `RATE_LIMIT` adds the source to a temporary list; for the next
  `RATE_LIMIT_WINDOW` (20) records from that source, each has a 50%
  chance of being suppressed too.

`reset()` picks a random starting point in the record list and clears
this suppression state; `step(action)` computes the reward from
`PayoffTable` using the *true* label of the current record (via
`classify_label`), applies any new suppression, advances to the next
non-suppressed record (or truncates the episode if the trace runs out or
`episode_length` steps have elapsed), and returns the next observation.

### `replay_buffer.py`

A plain fixed-capacity ring buffer (`ReplayBuffer`) of
`(state, defender_action, attacker_class, reward, next_state, done)`
tuples, with uniform random sampling into a `Batch`. Nothing unusual here
— this is standard DQN-style experience replay, just also storing the
realized attacker class alongside each transition (needed to index into
the joint Q-value at training time).

### `minimax_dqn.py`

The algorithmic core.

- **`QNetwork`**: a 2-hidden-layer (128 units by default) MLP mapping a
  state vector to a `(n_defender_actions, n_attacker_classes)` matrix —
  i.e. `Q(s, a_d, a_a)` for every joint action pair in one forward pass.
  Because the game is zero-sum, this single network's output *is* both
  players' payoff information (the attacker's payoff is its negation), so
  there's no separate attacker network.

- **`solve_matrix_game(Q)`**: given one state's `Q` matrix, solves the
  exact linear program for the value and the defender's optimal mixed
  strategy of that (tiny, e.g. 6×5) zero-sum matrix game, using
  `scipy.optimize.linprog`. This is what's used for actual action
  selection — at rollout/deployment time you only need this once per
  environment step, so exactness is affordable.

- **`batch_solve_fictitious_play(Q_batch)`**: approximates the same thing
  for a whole *batch* of matrix games at once, using
  [fictitious play](https://en.wikipedia.org/wiki/Fictitious_play)
  (each player repeatedly best-responds to the other's empirical
  strategy; for zero-sum games the running averages provably converge to
  a Nash equilibrium and the game value). Implemented as pure numpy array
  ops across the whole batch dimension simultaneously — this is what
  makes computing a TD-bootstrap value for every sample in a training
  batch, every single gradient step, computationally practical. (An
  earlier version called the exact LP solver once per sample in a Python
  loop; it was measured to be the main training bottleneck and was
  replaced with this vectorized approximation — see
  [§11](#11-troubleshooting).)

- **`MinimaxDQNAgent`**: owns an online and a target `QNetwork` (standard
  DQN-style target network for stability), an Adam optimizer, and:
  - `defender_strategy(state)` / `select_action(state, epsilon)`: exact-LP
    action selection, with epsilon-greedy mixed into the equilibrium
    strategy for exploration during training.
  - `train_step(batch)`: computes the TD target
    `r + gamma * (1 - done) * V(s')` where `V(s')` comes from
    `batch_solve_fictitious_play` on the *target* network's next-state Q
    matrices, then does a standard smooth-L1 loss + Adam step on the
    *online* network's prediction for the realized `(a_d, a_a)` cell,
    followed by a Polyak/soft update of the target network.
  - `save`/`load`: (de)serializes both networks' weights plus the config
    to/from a `.pt` file via `torch.save`/`torch.load`.

### `data.py` / `synthetic.py`

`data.py` loads one or more `conn.log.labeled` files matching a glob into
a flat record list, and splits them into a contiguous training block and
a contiguous, temporally-distinct evaluation block (not a per-row
shuffle, so evaluation isn't leaking adjacent flows from the same session
into training).

`synthetic.py` generates records with the same shape as a real Zeek conn
log, sampled from five hand-picked statistical "profiles" (one per
`AttackerClass`), with a handful of persistently "hostile" source IPs so
suppression actions have a consistent effect to learn against. It exists
purely so you can validate the whole pipeline without first obtaining a
real dataset.

### `train.py`

The training CLI (see [§9](#9-cli-reference) for every flag). At a glance,
the loop is: for each episode, reset the environment, then repeatedly
select an action (epsilon-greedy, epsilon decaying linearly over
`--eps-decay-episodes` episodes from `--eps-start` to `--eps-end`), step
the environment, push the transition to the replay buffer, and — once the
buffer has at least `--min-buffer` transitions — sample a batch and call
`agent.train_step()`. Every `--eval-every` episodes it runs a greedy
evaluation rollout (via `evaluate.rollout`) and saves a checkpoint.

### `evaluate.py`

`rollout(env, policy, n_episodes)` runs a policy function
(`state -> action`) through the environment for several episodes and
returns overall and per-true-class average reward and "action rate" (the
fraction of steps where the policy chose something other than `ALLOW`).
`make_agent_policy(agent)` wraps a trained agent as a greedy policy;
`make_constant_policy(action)` gives you fixed baselines (always allow,
always block, ...) to compare against. `summarize()` formats a result
dict for printing.

### `live/zeek_tail.py`

`LiveZeekTail` follows a file the way `tail -F` does: it reads newly
appended lines in a loop (polling every 0.5s), and if the file's inode
changes (Zeek rotated the log), it transparently reopens and re-parses
the new header. It auto-detects TSV vs. JSON per line (Zeek can be
configured to emit either) unless you force one with `--format`.
`from_start()` also processes what's already in the file; the default
(`iter(tail)`) seeks to the end first and only yields new records.

### `live/action_executor.py`

`Decision` is a small record of one scored connection's outcome. Every
`ActionExecutor.execute(decision)` call **always** writes a JSONL entry to
the audit log first — logging is not something you opt into, all
executors do it. `LogOnlyExecutor` does nothing else, which is why it's
the safe default. `NftablesExecutor` additionally adds the source IP to
an nftables set via `nft add element ...` (as a `subprocess.run` argv
list, never a shell string), but only when constructed with
`dry_run=False`; otherwise it logs what command it *would* have run.
`build_executor(kind, ...)` is the factory `run_agent.py` uses.

### `live/run_agent.py`

The live inference CLI: loads a checkpoint and its matching normalizer,
opens a `LiveZeekTail` on the given log path, and for each record,
extracts features (`training=False`, so the saved normalizer's statistics
aren't perturbed by live traffic), computes the defender's exact-LP
strategy, takes the argmax action (deterministic — no exploration at
deployment time), and hands the decision to the configured executor.

### `dqn.py`

A plain single-agent Double-DQN, kept separate from `minimax_dqn.py`. Its
`MLP` outputs a flat vector of Q-values over one agent's own actions (not
a joint payoff matrix), `DQNAgent` does epsilon-greedy selection and a
Double-DQN update (online net picks the next action, target net values
it), and `SimpleReplayBuffer` is a generic `(s, a, r, s', done)` buffer.
Used for the self-play attacker, and for the belief-based defender in
joint self-play.

### `selfplay.py`

The two-sided game. `ClassIndexedRecords` buckets the dataset by attacker
class so the environment can draw a real Zeek record of any chosen class
on demand (with a fallback if a class is absent from the data).
`SelfPlayACDEnv` runs an episode in two phases per step: `attacker_step`
(the attacker emits a chosen class → env samples a record → defender
features, unless the attacker is currently suppressed) and `defender_step`
(the defender responds → zero-sum rewards, plus block/rate-limit
dynamics). The attacker's own observation is a small behavioral vector
(episode progress, whether it's suppressed, the defender's last action,
and rolling summaries of how often it's been acted on and how it's
scored) — deliberately *not* the Zeek features, so it must learn evasive
policy from consequences, not from seeing what the defender sees.

### `train_selfplay.py`

The self-play training CLI (see [§10](#10-cli-reference)). Runs both
agents through the env, storing each one's transitions in its own buffer
and training each on its own objective. Supports the two modes described
in [§7](#7-self-play-training-a-symmetric-attacker): joint (`--defender
vanilla`, both learn) and exploitability testing (`--defender
minimax-frozen`, load and freeze a `train.py` defender, train only the
attacker). The `DefenderPolicy` wrapper gives both defender types a
uniform `act`/`select_action` interface. The defender's transitions are
finalized with a small "pending transition" pattern because suppressed
steps mean its consecutive decision points aren't adjacent env steps.

## 9. Configuration & tuning

There's no YAML config file by design — the two things worth tuning are
exposed directly:

- **The payoff table** (`game.py::PayoffTable.default()`): edit the
  6×5 `table` list directly, or construct a `PayoffTable(table=...)` and
  pass it to `ACDMarkovGameEnv(..., payoff=your_table)` in a custom
  script. Rows are `DefenderAction` in enum order, columns are
  `AttackerClass` in enum order. This is genuinely just illustrative
  numbers right now — calibrate it to your actual cost of a false
  positive (an unnecessary `BLOCK_SRC` on real traffic) vs. cost of a
  missed attack before trusting results.
- **Hyperparameters**: all exposed as `train.py` flags — see
  [§10](#10-cli-reference). The ones most worth adjusting first: `--episodes`
  / `--episode-length` (more training data exposure),
  `--eps-decay-episodes` (how long the agent keeps exploring), and
  `--hidden` (network capacity, if you add richer features).

## 10. CLI reference

### `python -m zeek_acd.train`

| Flag | Default | Meaning |
|---|---|---|
| `--data` | *(none)* | Glob for `conn.log.labeled` files; omit to use synthetic data |
| `--synthetic-n` | `20000` | Synthetic record count (only used without `--data`) |
| `--episodes` | `400` | Number of training episodes |
| `--episode-length` | `256` | Max environment steps per episode |
| `--batch-size` | `64` | Replay batch size per training step |
| `--buffer-size` | `50000` | Replay buffer capacity |
| `--min-buffer` | `1000` | Minimum transitions before training starts |
| `--train-every` | `1` | Train every N environment steps |
| `--eps-start` / `--eps-end` | `1.0` / `0.05` | Epsilon-greedy exploration schedule endpoints |
| `--eps-decay-episodes` | `250` | Episodes over which epsilon linearly decays |
| `--lr` | `1e-3` | Adam learning rate |
| `--gamma` | `0.95` | Discount factor |
| `--tau` | `0.01` | Target network Polyak update rate |
| `--hidden` | `128` | Q-network hidden layer width |
| `--eval-every` | `25` | Episodes between eval rollouts + checkpoints |
| `--eval-episodes` | `10` | Episodes per eval rollout |
| `--checkpoint-dir` | `checkpoints` | Output directory |
| `--seed` | `0` | RNG seed |

### `python -m zeek_acd.live.run_agent`

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | *(required)* | Path to a `.pt` file from `train.py` |
| `--normalizer` | *(required)* | Path to the matching `normalizer_*.json` |
| `--log-path` | *(required)* | Path to the live `conn.log` to tail |
| `--format` | `auto` | `auto` / `tsv` / `json` |
| `--from-start` | off | Also process lines already in the file |
| `--audit-log` | `acd_audit.jsonl` | Where every decision is logged |
| `--executor` | `log_only` | `log_only` or `nftables` |
| `--live-enforce` | off | **Required in addition to `--executor nftables`** to actually run `nft` |
| `--nft-table` | `inet acd` | nftables table (must already exist) |
| `--nft-blocked-set` | `blocked` | nftables set for `BLOCK_SRC`/`ISOLATE_HOST` |
| `--nft-ratelimited-set` | `ratelimited` | nftables set for `RATE_LIMIT` |
| `--deterministic` | on | Argmax action instead of sampling the mixed strategy |

### `python -m zeek_acd.train_selfplay`

| Flag | Default | Meaning |
|---|---|---|
| `--data` | *(none)* | Glob for `conn.log.labeled` files; omit for synthetic |
| `--synthetic-n` | `20000` | Synthetic record count (only used without `--data`) |
| `--episodes` | `600` | Number of self-play episodes |
| `--episode-length` | `128` | Max steps per episode |
| `--batch-size` | `64` | Replay batch size |
| `--buffer-size` | `50000` | Per-agent replay buffer capacity |
| `--min-buffer` | `1000` | Min transitions before an agent starts training |
| `--eps-start` / `--eps-end` | `1.0` / `0.05` | Exploration schedule endpoints |
| `--eps-decay-episodes` | `400` | Episodes over which epsilon decays |
| `--lr` | `1e-3` | Adam learning rate (both agents) |
| `--gamma` | `0.95` | Discount factor |
| `--hidden` | `128` | Hidden layer width (both agents) |
| `--defender` | `vanilla` | `vanilla` (train both) or `minimax-frozen` (train attacker only) |
| `--defender-checkpoint` | *(none)* | Required for `minimax-frozen`: the frozen defender's `.pt` |
| `--eval-every` | `50` | Episodes between eval + checkpoints |
| `--eval-episodes` | `10` | Episodes per eval |
| `--checkpoint-dir` | `checkpoints_selfplay` | Output directory |
| `--seed` | `0` | RNG seed |

## 11. Known limitations

**The minimax policy is conservative by construction.** Probing a trained
network's raw `Q(s, a_d, true_class)` values shows it *does* learn the
right per-class action rankings (e.g., for a C2 state it correctly ranks
`ISOLATE_HOST` highest). But the deployed policy comes from solving the
minimax LP over **all** attacker-class columns for every state, including
the four columns the network has no real state-specific evidence for at
that state. So even when the network is confident a connection is C2, the
LP still hedges against "what if this is actually benign," and ends up
picking whatever's cheapest under that hypothetical too — typically
`LOG_ALERT`. This is a real property of minimax-Q (robustness to a
state-independent worst case), not a bug, but it means the agent
essentially never acts on its own class confidence. If you want that
instead: after training, act greedily on
`argmax_a mean_over_class Q(s, a, class)` (or just train a plain,
non-minimax DQN on the same reward) rather than using
`agent.defender_strategy()`'s minimax mixed strategy at deployment.

**No cross-flow state.** Each decision is based on a single connection's
features; there's no rolling per-source context (recent alert count,
beaconing interval, etc.). This likely limits detection of inherently
multi-flow patterns like C2 beaconing. A recurrent or windowed-aggregate
state representation is the natural extension.

**The attacker doesn't adapt — in `train.py`.** In the standard defender
training loop, the attacker's "action" is read directly from the dataset's
ground-truth label, so the defender trains against a fixed empirical
traffic mix rather than an adversary that reacts to its policy. The
self-play tooling in [§7](#7-self-play-training-a-symmetric-attacker)
addresses this — an attacker DQN that actively learns to exploit the
defender — but it's currently a separate loop (`train_selfplay.py`); it
isn't yet wired back into `train.py` as the opponent for the *minimax*
defender's own training. Closing that loop (full adversarial co-training)
is the remaining step.

**The payoff table is illustrative, not calibrated.** See [§9](#9-configuration--tuning).

## 12. Troubleshooting

**Training seems extremely slow (tens of seconds per episode or worse).**
This network is tiny (a couple hundred hidden units). PyTorch's default
multi-threaded CPU execution has fixed per-call overhead that, at this
scale, dwarfs the actual compute — measured at ~40ms/training-step
multi-threaded vs. ~2.5ms single-threaded on this project's network size.
Both `train.py` and `live/run_agent.py` already call
`torch.set_num_threads(1)` at startup for this reason; if you've copied
code out of this project into your own script, make sure to do the same.

**`FileNotFoundError: no files matched pattern` from `--data`.** Check
your glob actually matches the `conn.log.labeled` files' real location —
IoT-23 scenario archives typically nest them under a `bro/` subdirectory
per scenario, so a pattern like `iot23/*/conn.log.labeled` (missing
`bro/`) will silently match nothing.

**`nft: command not found` / permission errors with `--executor
nftables`.** You need `nftables` installed and the process needs
privileges to modify firewall rules (typically root, or an equivalent
capability). Also double check you pre-created the table/sets referenced
by `--nft-table`/`--nft-blocked-set`/`--nft-ratelimited-set` — see the
`NftablesExecutor` docstring in `live/action_executor.py`.
