"""Cross-flow context for a stream of Zeek connection records.

A single ``conn.log`` line carries almost no evidence about the traffic
patterns that actually give malware away. A TLS session to a C2 server and
a TLS session to a news site have the same shape: ``conn_state=SF``,
``history=ShADadFf``, a kilobyte each way. What separates them is context
-- the C2 flow repeats on a timer, goes to a host nobody else in the
network talks to, and belongs to a source that has just opened a burst of
short-lived connections.

This module computes exactly that, in one streaming pass over records in
log order, so the same class can run offline over a training file and
online over a live tail:

    ctx = ConnContext()
    for record in log:
        record.update(ctx.update(record))     # adds ctx_* keys

``features.raw_feature_vector`` picks the ``ctx_*`` keys up if they are
there and uses zeros if they are not, so records that never went through a
context pass still produce a valid (if less informative) vector.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

# Keys added to each record, in the order they enter the feature vector.
#
# These are deliberately *relative* -- shares and ratios rather than raw
# counts. An absolute count depends on how much traffic happens to be on
# the wire: the same host replaying the same scenarios produced a median
# ctx_src_conns_long of 10 inside a per-scenario pcap and 316 on a busy
# live interface, and a model that keyed on the raw number flagged 78% of
# benign traffic once the live capture got busier. A share of the
# network's traffic in the same window does not move like that. Raw
# connection volume is left out for the same reason: high-volume scanning
# still shows up, through fan-out and failed-handshake ratio.
CONTEXT_FEATURES = [
    "ctx_src_share_short",    # this source's share of all traffic (short window)
    "ctx_src_share_long",     # ... and over the long window
    "ctx_src_fanout",         # distinct destinations per connection: ~1 = scanning
    "ctx_src_port_fanout",    # distinct destination ports per connection
    "ctx_src_fail_ratio",     # fraction of its connections that never established
    "ctx_src_gap",            # log2 seconds since this source's previous connection
    "ctx_pair_share",         # share of this source's traffic going to this dst:port
    "ctx_pair_iat_mean",      # log2 mean inter-arrival time to that dst:port
    "ctx_pair_iat_cv",        # its coefficient of variation -- low = beaconing
    "ctx_dst_share",          # this destination's share of all traffic seen so far
    "ctx_dst_source_share",   # share of known sources that ever talked to it
]

# Counts and time gaps are log-compressed before they leave this module.
# A host that opened 10 connections and one that opened 300 are different,
# but not 30x different -- and the raw counts depend heavily on how the
# capture was taken: one scenario per pcap gives a source a handful of
# connections, the same traffic replayed onto a live wire alongside
# everything else gives it hundreds. Log scaling keeps a model trained on
# one composition usable on the other.
def _log(x: float) -> float:
    return math.log2(1.0 + max(0.0, x))


SHORT_WINDOW = 60.0
LONG_WINDOW = 600.0
MAX_PAIR_HISTORY = 32

# Zeek conn_state values that mean the handshake never completed.
FAILED_STATES = {"S0", "REJ", "RSTO", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"}


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ConnContext:
    """Rolling per-source and per-destination statistics.

    Memory is bounded: per-source event deques are trimmed to the long
    window on every update, and per-pair inter-arrival history keeps only
    the last ``MAX_PAIR_HISTORY`` timestamps. Sources idle for longer than
    ``forget_after`` are dropped so a long capture doesn't accumulate
    state for hosts that went away.
    """

    def __init__(self, short_window: float = SHORT_WINDOW,
                 long_window: float = LONG_WINDOW,
                 forget_after: float = 3600.0):
        self.short_window = short_window
        self.long_window = long_window
        self.forget_after = forget_after
        # src -> deque of (ts, dst, dport, failed)
        self._src_events: dict[str, deque] = defaultdict(deque)
        self._src_last_ts: dict[str, float] = {}
        # (src, dst, dport) -> deque of ts
        self._pair_ts: dict[tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=MAX_PAIR_HISTORY))
        self._dst_counts: dict[str, int] = defaultdict(int)
        self._dst_sources: dict[str, set[str]] = defaultdict(set)
        # Network-wide denominators, so every per-source number can be
        # expressed as a share of what the whole network was doing.
        self._all_events: deque = deque()
        self._total_records = 0
        self._all_sources: set[str] = set()
        self._last_prune = 0.0

    def _prune_sources(self, now: float) -> None:
        """Drop hosts that have been silent for longer than forget_after.
        Runs at most once per long window -- it walks every known source."""
        if now - self._last_prune < self.long_window:
            return
        self._last_prune = now
        stale = [src for src, ts in self._src_last_ts.items()
                 if now - ts > self.forget_after]
        for src in stale:
            self._src_events.pop(src, None)
            self._src_last_ts.pop(src, None)
        if stale:
            stale_set = set(stale)
            for key in [k for k in self._pair_ts if k[0] in stale_set]:
                del self._pair_ts[key]

    def update(self, record: dict) -> dict[str, float]:
        """Folds one record in and returns its context features.

        Called in log order. The returned features describe the state
        *before* this connection, plus this connection itself -- the same
        thing a live sensor would know at decision time.
        """
        ts = _float(record.get("ts"))
        src = record.get("id.orig_h") or ""
        dst = record.get("id.resp_h") or ""
        dport = str(record.get("id.resp_p") or "")
        conn_state = record.get("conn_state") or ""
        failed = conn_state in FAILED_STATES

        events = self._src_events[src]
        cutoff = ts - self.long_window
        while events and events[0][0] < cutoff:
            events.popleft()
        while self._all_events and self._all_events[0] < cutoff:
            self._all_events.popleft()

        short_cutoff = ts - self.short_window
        conns_short = sum(1 for e in events if e[0] >= short_cutoff)
        conns_long = len(events)
        all_short = sum(1 for t in self._all_events if t >= short_cutoff)
        all_long = len(self._all_events)
        distinct_dst = len({e[1] for e in events})
        distinct_dport = len({e[2] for e in events})
        fail_ratio = (sum(1 for e in events if e[3]) / conns_long) if conns_long else 0.0
        gap = ts - self._src_last_ts[src] if src in self._src_last_ts else self.long_window

        pair_key = (src, dst, dport)
        pair_ts = self._pair_ts[pair_key]
        recent_pair = [t for t in pair_ts if t >= cutoff]
        if len(recent_pair) >= 2:
            gaps = [b - a for a, b in zip(recent_pair, recent_pair[1:])]
            iat_mean = sum(gaps) / len(gaps)
            if iat_mean > 0 and len(gaps) >= 2:
                var = sum((g - iat_mean) ** 2 for g in gaps) / len(gaps)
                iat_cv = math.sqrt(var) / iat_mean
            else:
                # A single observed gap has no spread; treat it as perfectly
                # regular rather than as unknown.
                iat_cv = 0.0
        else:
            iat_mean = 0.0
            iat_cv = 1.0  # unknown: looks as irregular as anything else

        features = {
            "ctx_src_share_short": conns_short / all_short if all_short else 0.0,
            "ctx_src_share_long": conns_long / all_long if all_long else 0.0,
            "ctx_src_fanout": distinct_dst / conns_long if conns_long else 0.0,
            "ctx_src_port_fanout": distinct_dport / conns_long if conns_long else 0.0,
            "ctx_src_fail_ratio": float(fail_ratio),
            "ctx_src_gap": _log(min(gap, self.long_window)),
            "ctx_pair_share": len(recent_pair) / conns_long if conns_long else 0.0,
            "ctx_pair_iat_mean": _log(min(iat_mean, self.long_window)),
            "ctx_pair_iat_cv": float(iat_cv),
            "ctx_dst_share": (self._dst_counts[dst] / self._total_records
                              if self._total_records else 0.0),
            "ctx_dst_source_share": (len(self._dst_sources[dst]) / len(self._all_sources)
                                     if self._all_sources else 0.0),
        }

        events.append((ts, dst, dport, failed))
        self._all_events.append(ts)
        self._src_last_ts[src] = ts
        pair_ts.append(ts)
        self._dst_counts[dst] += 1
        self._dst_sources[dst].add(src)
        self._total_records += 1
        self._all_sources.add(src)
        self._prune_sources(ts)
        return features


def annotate(records: list[dict], context: ConnContext | None = None) -> list[dict]:
    """Runs a context pass over a list of records, in timestamp order, and
    writes the ``ctx_*`` keys into each record in place.

    Records are sorted by ``ts`` first: Zeek writes a connection when it
    *ends*, so a log is only roughly ordered by start time, and the rolling
    windows want start order.
    """
    context = context or ConnContext()
    for record in sorted(records, key=lambda r: _float(r.get("ts"))):
        record.update(context.update(record))
    return records
