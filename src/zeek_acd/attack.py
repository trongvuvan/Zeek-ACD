"""A MITRE ATT&CK-grounded action space for the self-play attacker.

The original self-play attacker (``selfplay.py``) picks a coarse traffic
*class* (RECON / DOS / C2 / OTHER_MALICIOUS / BENIGN) and one of five
*evasion* knobs. That is enough to find "an" evasion, but it treats, say,
"C2" as a single behaviour -- whereas a real command-and-control channel
can look like a steady HTTPS beacon (T1071.001), a jittered malleable
profile, a fast-flux / DGA channel that rotates destinations (T1568.002),
or a fallback channel that goes low-and-slow (T1008). Those are different
things on the wire and a defender may catch some and miss others.

This module replaces the 5x5 grid with a *catalog of ATT&CK techniques*.
Each technique pins two things:

  * ``base_class`` -- which real Zeek connection records to draw from, and
    therefore how the reward is scored (the payoff table is still keyed by
    :class:`~zeek_acd.game.AttackerClass`). Features stay realistic because
    they come from a real record of that class.
  * a *shaping profile* -- timing, destination, port and byte-volume knobs
    that turn that record into the technique's on-the-wire signature.

So the attacker's action becomes "pick a technique", and every technique
carries its MITRE id and tactic. Training against this catalog lets us
watch how the defender reacts to each distinct technique instead of to a
lumped class, which is what makes the self-play diagnostic useful.

The shaping vocabulary is a superset of the old evasion modes, so the
classic 5x5 action space is expressible as a (degenerate) catalog and the
two code paths share one emitter in ``selfplay.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import AttackerClass

# --- shaping vocabulary -----------------------------------------------------
# timing: gap between successive connections from the attacker.
#   steady  -- constant base gap (a textbook beacon)
#   jitter  -- randomized gap (breaks beacon-regularity features)
#   slow    -- long gap (low-and-slow; shrinks the source's traffic share
#              but costs episode opportunities)
#   burst   -- near-zero gap (flooding / fast scanning)
TIMINGS = ("steady", "jitter", "slow", "burst")
# dest: which destination host the connection goes to.
#   fixed   -- one host, so a (src,dst,port) beacon accumulates share
#   spread  -- a new host every connection (fast-flux / host sweep), so no
#              single destination builds up a telltale share
DESTS = ("fixed", "spread")
# port: destination port behaviour.
#   fixed   -- pinned to the standard C2 port (443)
#   keep    -- keep the sampled record's own port (non-standard-port C2)
#   scan    -- a new port every connection (service scanning)
PORTS = ("fixed", "keep", "scan")


@dataclass(frozen=True)
class Technique:
    """One ATT&CK technique the attacker can choose.

    ``pad`` is ``None`` or an ``(lo, hi)`` multiplier range applied to the
    byte counts (data transfer / exfiltration inflate them; a plain beacon
    leaves them alone)."""

    id: str            # MITRE ATT&CK id, e.g. "T1071.001"
    name: str
    tactic: str        # ATT&CK tactic, e.g. "command-and-control"
    base_class: AttackerClass
    timing: str = "steady"
    dest: str = "fixed"
    port: str = "fixed"
    pad: tuple[float, float] | None = None

    def __post_init__(self):
        assert self.timing in TIMINGS, self.timing
        assert self.dest in DESTS, self.dest
        assert self.port in PORTS, self.port


# --- the catalog ------------------------------------------------------------
# Techniques span six tactics. Several share a base_class but differ in
# shaping -- the whole point is that "C2" or "exfil" is not one behaviour.
_ATTACK = [
    # Reconnaissance / Discovery -------------------------------------------
    Technique("T1595.001", "Active Scanning: IP block sweep", "reconnaissance",
              AttackerClass.RECON, timing="burst", dest="spread", port="fixed"),
    Technique("T1046", "Network Service Scanning", "discovery",
              AttackerClass.RECON, timing="burst", dest="spread", port="scan"),
    Technique("T1046.slow", "Network Service Scanning (low-and-slow)", "discovery",
              AttackerClass.RECON, timing="slow", dest="spread", port="scan"),
    # Command and Control ---------------------------------------------------
    Technique("T1071.001", "Web Protocols beacon", "command-and-control",
              AttackerClass.C2, timing="steady", dest="fixed", port="fixed"),
    Technique("T1071.001.jitter", "Web Protocols beacon (jittered sleep)",
              "command-and-control", AttackerClass.C2, timing="jitter",
              dest="fixed", port="fixed"),
    Technique("T1571", "Non-Standard Port C2", "command-and-control",
              AttackerClass.C2, timing="steady", dest="fixed", port="keep"),
    Technique("T1568.002", "Dynamic Resolution: DGA / fast-flux",
              "command-and-control", AttackerClass.C2, timing="steady",
              dest="spread", port="fixed"),
    Technique("T1008", "Fallback Channels (low-and-slow)", "command-and-control",
              AttackerClass.C2, timing="slow", dest="spread", port="fixed"),
    # Ingress / payload -----------------------------------------------------
    Technique("T1105", "Ingress Tool Transfer", "command-and-control",
              AttackerClass.OTHER_MALICIOUS, timing="steady", dest="fixed",
              port="fixed", pad=(3.0, 8.0)),
    # Exfiltration ----------------------------------------------------------
    Technique("T1041", "Exfiltration Over C2 Channel", "exfiltration",
              AttackerClass.OTHER_MALICIOUS, timing="steady", dest="fixed",
              port="fixed", pad=(8.0, 20.0)),
    Technique("T1048", "Exfil Over Alternative Protocol", "exfiltration",
              AttackerClass.OTHER_MALICIOUS, timing="jitter", dest="fixed",
              port="keep", pad=(5.0, 15.0)),
    Technique("T1030", "Data Transfer Size Limits (chunked exfil)", "exfiltration",
              AttackerClass.OTHER_MALICIOUS, timing="jitter", dest="fixed",
              port="fixed"),
    # Impact ----------------------------------------------------------------
    Technique("T1498", "Network Denial of Service", "impact",
              AttackerClass.DOS, timing="burst", dest="fixed", port="fixed"),
    Technique("T1499", "Endpoint Denial of Service", "impact",
              AttackerClass.DOS, timing="burst", dest="fixed", port="keep"),
    Technique("T1498.slow", "Slow-rate Denial of Service", "impact",
              AttackerClass.DOS, timing="slow", dest="fixed", port="fixed"),
    # Defense Evasion -------------------------------------------------------
    Technique("T1071.blend", "Blend in with benign web traffic", "defense-evasion",
              AttackerClass.BENIGN, timing="steady", dest="fixed", port="fixed"),
]

_CATALOGS: dict[str, list[Technique]] = {"attack": _ATTACK}


def build_catalog(name: str = "attack") -> list[Technique]:
    """Return a named technique catalog (currently only ``attack``)."""
    if name not in _CATALOGS:
        raise SystemExit(f"unknown attacker catalog {name!r}; "
                         f"choose from {sorted(_CATALOGS)}")
    return _CATALOGS[name]


def tactics(catalog: list[Technique]) -> list[str]:
    """Distinct tactics in ``catalog``, in first-seen order."""
    seen: list[str] = []
    for t in catalog:
        if t.tactic not in seen:
            seen.append(t.tactic)
    return seen
