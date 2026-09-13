# zeek-acd

An Active Cyber Defense (ACD) agent for network flows: a DQN-style value
network trained as a **zero-sum stochastic (Markov) game** between a
defender and an attacker, observing **Zeek** connection telemetry.

## How the pieces fit together

- **Game.** `src/zeek_acd/game.py` defines the two action spaces:
  - Defender: `ALLOW, LOG_ALERT, RATE_LIMIT, BLOCK_SRC, ISOLATE_HOST, DECEIVE`
  - Attacker (latent, used only to compute training reward): `BENIGN, RECON, DOS, C2, OTHER_MALICIOUS`
  - A payoff table `PayoffTable.default()` gives the defender's reward for
    every `(defender_action, attacker_class)` pair; the attacker's reward
    is its negation (zero-sum). Tune this table to reflect your actual
    cost of false positives vs. cost of a missed attack.

- **Algorithm.** `src/zeek_acd/minimax_dqn.py` implements **minimax-Q**
  (Littman, 1994) with a neural network in place of the tabular Q-function
  (a "Minimax-DQN"). Because the game is zero-sum, one network outputs the
  defender's payoff `Q(s, a_defender, a_attacker)` for every joint action;
  the state value used for TD bootstrapping is the value of the resulting
  matrix game, solved exactly each step via a small linear program
  (`scipy.optimize.linprog`). This gives the defender a policy that is
  robust to the worst case over attacker behavior consistent with the
  observed state, rather than just the empirical average.

- **Environment.** `src/zeek_acd/env.py` replays a trace of Zeek
  connection records. It's a genuine Markov game, not a per-flow
  supervised bandit: choosing `BLOCK_SRC`/`ISOLATE_HOST` suppresses all
  further connections from that source for the rest of the episode, and
  `RATE_LIMIT` probabilistically drops a window of follow-on connections
  — the defender's action changes which future states occur.

- **Features.** `src/zeek_acd/features.py` turns a raw `conn.log` record
  (duration, byte/packet counts, proto, `conn_state`, `service`, `history`
  flags) into a fixed-size vector, normalized online (Welford) so the same
  code path works for offline training and live inference without
  refitting on production traffic.

- **Data.** `src/zeek_acd/zeeklog.py` parses Zeek's standard ASCII TSV log
  format (`#separator`/`#fields`/`#types` headers), so the same parser
  works for static training files and for tailing a live log.

## Dataset: IoT-23

The trainer expects **Zeek-native, labeled** connection logs. The
[Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23) dataset
is a good fit: it ships as real `conn.log.labeled` files (Zeek TSV +
`label`/`detailed-label` columns) from IoT malware captures, covering
benign traffic plus scans, DDoS, C&C/botnet beaconing, etc. — exactly the
attacker classes modeled here.

```bash
# Get one or more scenario captures, e.g. from the IoT-23 mirror; each
# scenario directory contains a bro/ (Zeek) folder with conn.log.labeled.
# Then train against every scenario you've downloaded:
python -m zeek_acd.train --data 'iot23/*/conn.log.labeled' --episodes 800
```

Without `--data`, `train.py` generates synthetic Zeek-shaped records so
you can smoke-test the whole pipeline (env, LP solver, replay buffer,
network) before spending time on the real dataset:

```bash
python -m zeek_acd.train --synthetic-n 20000 --episodes 400
```

Training prints periodic eval rollouts broken out by true attacker class,
e.g. `action_rate` for `BENIGN` is your false-positive rate, and
`action_rate` for `RECON`/`DOS`/`C2`/`OTHER_MALICIOUS` is your detection
rate. Checkpoints (`agent_*.pt` + `normalizer_*.json`) land in
`--checkpoint-dir`.

## Offline evaluation

```python
from zeek_acd.minimax_dqn import MinimaxDQNAgent
from zeek_acd.evaluate import make_agent_policy, make_constant_policy, rollout, summarize
from zeek_acd.game import DefenderAction

agent = MinimaxDQNAgent.load("checkpoints/agent_latest.pt")
print(summarize("trained", rollout(eval_env, make_agent_policy(agent))))
print(summarize("always-allow baseline", rollout(eval_env, make_constant_policy(DefenderAction.ALLOW))))
print(summarize("always-block baseline", rollout(eval_env, make_constant_policy(DefenderAction.BLOCK_SRC))))
```

## Live Zeek integration

`src/zeek_acd/live/` tails a running Zeek instance's `conn.log` (TSV or
JSON, auto-detected) and scores each connection with the trained policy:

```bash
python -m zeek_acd.live.run_agent \
  --checkpoint checkpoints/agent_latest.pt \
  --normalizer checkpoints/normalizer_latest.json \
  --log-path /path/to/zeek/logs/current/conn.log \
  --audit-log acd_audit.jsonl
```

**Safety model — read before enabling enforcement:**

- The default executor (`--executor log_only`, the default) **never
  touches the network**. It only writes a JSONL audit record per decision
  (uid, source/dest, chosen action, full mixed strategy) so a human can
  review what the agent *would* do before anything is enforced.
- `--executor nftables` requires a pre-created nftables table/sets (see
  the docstring in `live/action_executor.py`) and, on top of that, the
  separate `--live-enforce` flag. Selecting `nftables` without
  `--live-enforce` still only logs what it would have run (dry run).
  Only pass `--live-enforce` once you've reviewed the audit log from a
  dry run and are satisfied with the false-positive rate — `BLOCK_SRC`
  and `ISOLATE_HOST` suppress a real source's traffic.
- Every source IP is validated (`ipaddress.ip_address`) before it's ever
  placed in an `nft` command argument list (no shell interpolation).

## A note on `/root/zeek`

There's a full Zeek source checkout at `/root/zeek` with its own
`AGENTS.md`/`AI_POLICY.md` — those govern *contributions to Zeek itself*
(PRs against the Zeek project) and are unrelated to this project, which
only consumes Zeek's standard log output and doesn't modify Zeek's source.
You still need a working `zeek` binary to produce live logs; see
`/root/zeek/doc/building-from-source.rst` in that checkout, or install a
packaged release instead of building from source.

## Project layout

```
src/zeek_acd/
  game.py            action spaces, attacker-class label mapping, payoff table
  features.py        conn.log record -> normalized feature vector
  zeeklog.py          Zeek TSV log parsing (shared by offline + live)
  env.py              Gymnasium env: the zero-sum Markov game
  minimax_dqn.py       QNetwork + minimax-Q LP solver + MinimaxDQNAgent
  replay_buffer.py
  data.py, synthetic.py
  train.py, evaluate.py
  live/
    zeek_tail.py        follows a live conn.log (TSV or JSON, rotation-aware)
    action_executor.py  log_only / nftables backends, always audit-logged
    run_agent.py         CLI wiring it all together
```

## Known behavior: the minimax policy is conservative by construction

Smoke-testing against synthetic data (see below) turned up something worth
understanding before you trust this on real traffic: the trained policy
tends to converge to `LOG_ALERT` for almost everything, rarely committing
to `BLOCK_SRC`/`ISOLATE_HOST` even for classes it clearly recognizes.

This is **not** a training failure. Probing the raw `Q(s, a_d, true_class)`
values shows the network has learned the right per-class rankings (e.g.
for a C2 state, `ISOLATE_HOST` scores highest, exactly as the payoff table
intends; for RECON, `DECEIVE` scores highest). The issue is downstream, in
how the deployed action is *extracted*: the LP takes the worst case over
**all** attacker-class columns for every state, including the four columns
the network has no real state-specific evidence for. So even when the
network is confident a state is C2, the minimax solver still asks "but
what if this is actually BENIGN?" and picks whatever's safe under that
hypothetical too — which is `LOG_ALERT`, almost by design of the payoff
table (it's the cheapest hedge against being wrong about anything).

This is a real property of minimax-Q, not an artifact: it buys robustness
to a state-independent worst case at the cost of never acting on the
network's own confidence. Whether that's desirable depends on your threat
model:

- If you actually want protection against an attacker that can make
  malicious traffic *look* benign in the specific features you're using,
  this conservatism is the point — don't fight it, tune the payoff table
  instead (e.g., make `LOG_ALERT` cheaper still relative to false
  BLOCK/ISOLATE costs so it isn't already the default safe harbor).
- If you actually want the agent to act on its own class belief once
  confident, this formulation is the wrong tool as-is. The straightforward
  fix is to drop the adversarial framing for action *selection* (keep it
  for the TD target if you like the regularization) and act greedily on
  `argmax_a E_class[Q(s,a,class)]` using the network's own implied belief,
  or just train a plain (non-minimax) DQN on the same reward and compare.

## Caveats / what a v2 should improve

- State is a single connection's features; it doesn't yet carry
  cross-flow context (e.g. a rolling count of an source's recent alerts).
  Adding a small recurrent or windowed-aggregate state would likely help
  the C2/beaconing case, which is inherently a multi-flow pattern.
- The attacker's "action" during training is read directly from the
  dataset's ground-truth label, not from an adaptive adversary — the
  robustness minimax-Q buys you is against worst-case behavior *consistent
  with a given state*, not against an attacker that observes and reacts to
  the deployed policy. A true self-play variant (both sides learn) is
  the natural next step if you need robustness against an adaptive
  attacker specifically.
- The payoff table in `game.py` is illustrative, not calibrated to any
  real cost model — treat it as the main lever to tune before trusting
  results.
