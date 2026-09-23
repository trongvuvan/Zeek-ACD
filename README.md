# zeek-acd

An Active Cyber Defense (ACD) agent for network flows: a DQN-style value
network trained as a **zero-sum stochastic (Markov) game** between a
defender and an attacker, observing **Zeek** connection telemetry.

**For the full walkthrough — what every module does, how to run each
stage, the CLI reference, and known limitations — see
[`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).** This README stays a
short orientation.

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

- **Cross-flow context.** `src/zeek_acd/context.py` adds what a single
  connection can't show: how much of the network's traffic this source
  accounts for, its fan-out across destinations and ports, its ratio of
  failed handshakes, how regular the gaps to this exact destination are
  (beaconing), and how rare the destination is. One streaming pass, shared
  by offline training and the live tail. These are deliberately *shares
  and ratios*, not counts -- see "Context features must be relative".

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
  context.py         cross-flow rolling statistics (offline + live)
  dqn.py              plain single-agent DQN (defender or self-play attacker)
  selfplay.py         two-sided environment where both players learn
  iocs.py             defanged IOC report -> matchable indicator set
  pcap_dataset.py      scenario pcaps / live logs -> labeled Zeek data
  data.py, synthetic.py
  train.py            Minimax-DQN trainer
  train_dqn.py         plain DQN trainer (the one that works on real data)
  train_selfplay.py    attacker + defender self-play
  eval_checkpoint.py   score any checkpoint on any labeled dataset
  audit_report.py      read back what a live run decided, vs. labels
  evaluate.py
  live/
    zeek_tail.py        follows a live conn.log (TSV or JSON, rotation-aware)
    action_executor.py  log_only / nftables backends, always audit-logged
    run_agent.py         CLI wiring it all together
```

## Two defenders: Minimax-DQN and plain DQN

`train.py` trains the Minimax-DQN described above; `train_dqn.py` trains a
plain (single-agent) DQN on the same environment, data and payoff table.
On real data the plain DQN is the one that works, and the reason is
structural rather than a matter of tuning -- see the next two sections.
`eval_checkpoint.py` scores either kind against any labeled dataset, and
`live/run_agent.py` detects which kind a checkpoint holds and loads it
accordingly.

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

### What the Q-matrix actually learned

Probing the trained network settles the question. Over 2000 evaluation
states, the best defender action *per attacker-class column* is identical
at every single state: BENIGN->ALLOW (1999/2000), RECON->DECEIVE
(2000/2000), DOS->BLOCK_SRC, C2->ISOLATE_HOST. The network reproduced the
payoff table and learned nothing state-dependent about which class it is
looking at -- because `Q(s, a_d, a_a)` is only ever updated at the column
that actually occurred, so its target is `payoff(a_d, c) + gamma*V(s')`
and the state-dependent part (`V`) shifts a whole column uniformly without
changing any ranking. The network never *has* to tell the classes apart.

That is why swapping the action-extraction rule doesn't help: the LP gives
"always LOG_ALERT", `argmax_a E_class[Q]` gives "always BLOCK_SRC", and
both are exactly the corresponding constant baseline. A plain DQN has no
class axis, so inferring the class from the features is the only way it
can score -- which is what `train_dqn.py` is for.

## Training data beyond IoT-23

IoT-23 alone does not generalize: a DQN trained on it scored **-0.548** on
labeled malware HTTP/TLS traffic, allowing essentially everything, worse
than the always-LOG_ALERT baseline (+0.090). IoT-23's malicious traffic is
IoT botnet scanning and flooding (`conn_state=S0`/`REJ`, no service); C2
over TLS is a well-formed session (`SF`, `ShADadFf`) that looks exactly
like IoT-23's *benign* traffic.

`pcap_dataset.py` closes that gap by building labeled Zeek data from
scenario pcaps plus their IOC reports:

```bash
# pcaps in /root/pcaps, matching IOC write-ups in /root/indicators
python -m zeek_acd.pcap_dataset --out-dir data/mta

# or label an already-running sensor's logs (TSV or JSON) against every
# indicator at once -- --ioc-glob splits a live log by scenario, so some
# scenarios can be held out of training entirely
python -m zeek_acd.pcap_dataset --zeek-log-dir /opt/zeek/logs/current \
    --ioc-glob '2025-*.txt' --out-file data/live2025/conn.log.labeled \
    --unmatched drop
```

Indicators are extracted by `iocs.py` from defanged report text. The
defanging is what makes it reliable: a real indicator always carries
`[.]`, `[:]` or `hxxp`, while filenames and hashes in the same document
never do (`PO-12345.zip` is otherwise indistinguishable from a hostname
under the `.zip` TLD). A connection is matched to an indicator by
responder IP, by the TLS SNI / HTTP Host recorded for its `uid`, or by
the address DNS handed out for an indicator domain *in that same
capture*. On 23 scenario pcaps that left 1 unmatched external connection
out of 403.

`train_dqn.py --data` can be repeated to mix datasets, with `--repeat` to
stop a 45k-row dataset from drowning a 400-row one. Only training records
are repeated; eval records never are.

## Context features must be relative

The first version of `context.py` used raw counts, and it broke in a way
worth recording. Held-out *pcap* scenarios looked excellent (100%
detection, 8% false positives) while the same traffic replayed onto a live
interface gave 77% false positives -- and re-scoring a later, busier
snapshot of the same live log turned the earlier 8% into 78%.

It was not the pipeline (comparing every field and every feature for 400
shared `uid`s between the offline and live paths showed them agreeing) and
not packet loss (0.56%). It was that one connection's context depends on
how much traffic happens to be on the wire: the same host replaying the
same scenarios has a median `src_conns_long` of 10 inside a per-scenario
pcap and 316 on a busy live interface, still climbing as the capture runs.

So every context feature is now a share or a ratio -- a source's share of
the network's traffic in the window, destinations per connection, the
destination's share of all traffic -- and raw counts are gone. Scanning
still stands out, through fan-out and failed-handshake ratio, which don't
move with capture length.

## Tuning the payoff table

The payoff table is the lever on the false-positive / missed-detection
trade-off, and the default one barely charges for a false positive:
`LOG_ALERT` on benign traffic costs -0.05 while a correct one on an attack
pays +0.20, so "alert on everything" scores almost as well as being right.
In practice the false-positive rate wandered between 1% and 96% across
checkpoints of one run with almost no change in reward.

`payoffs/low_fp.json` charges about four times as much for every action
taken against benign traffic and leaves the detection payoffs alone. Pass
it to `train_dqn.py --payoff` (also accepted by `train.py` and
`eval_checkpoint.py`); with it the false-positive rate stayed within
1-6% across checkpoints while detection stayed at 100%.

## Caveats / what a v2 should improve

- The benign class in the labeled malware data is almost entirely DNS and
  NTP to the local resolver -- there is no ordinary outbound HTTPS in it.
  The measured false-positive rate is only as trustworthy as that, and a
  capture of normal browsing is the single most useful thing to add.
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
