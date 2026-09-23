"""Parsing defanged indicator-of-compromise reports into matchable sets.

The reports in ``/root/indicators`` are prose write-ups (malware-traffic
-analysis style) where every network indicator is *defanged* so it can't be
clicked by accident:

    hxxps[:]//rooinson[.]icu/api/v1/e08a3c4
    193.187.91[.]281:55016  <-- C2 traffic, HTTPS TLSv1.0

That defanging is what makes the extraction reliable here: a real
indicator always carries ``[.]``/``[:]``/``hxxp``, while filenames and
hashes in the same document never do. So rather than guessing which
dotted token is a hostname (``PO-12345.zip`` looks exactly like a domain
under the ``.zip`` TLD), we only take tokens that were defanged.

Each indicator is tagged with the attacker class it belongs to, read from
the words around it: the line itself first, then the section heading it
sits under. ``report.txt`` -> ``IocSet`` -> per-connection labels is what
``pcap_dataset.py`` uses to turn a pcap into training data.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path

from .game import AttackerClass

# Defanged forms, most specific first. Applied left to right.
_REFANG = [
    ("[.]", "."),
    ("[:]", ":"),
    ("(.)", "."),
    ("[dot]", "."),
    ("hxxps", "https"),
    ("hxxp", "http"),
]

_URL_RE = re.compile(r"https?://([A-Za-z0-9._-]+)(?::(\d+))?(/[^\s,;)\]]*)?")
_HOST_RE = re.compile(r"\b((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})\b")
_IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d+))?\b")

# Words that place an indicator in a class. Checked on the indicator's own
# line first, then on the section heading above it.
_C2_WORDS = ("c2", "c&c", "command and control", "beacon", "handshake",
             "rat ", " rat", "post-infection", "callback")
_EXFIL_WORDS = ("exfil", "smtp", "data theft", "stolen")


def refang(text: str) -> str:
    out = text
    for a, b in _REFANG:
        out = out.replace(a, b)
        out = out.replace(a.upper(), b)
    return out


def _class_for(line: str, heading: str) -> AttackerClass:
    """C2 vs. the rest. These reports are infection write-ups: there is no
    scanning or flooding in them, so every indicator is either C2 or some
    other malicious step (payload download, redirect chain, exfil)."""
    for text in (line.lower(), heading.lower()):
        if any(w in text for w in _C2_WORDS):
            return AttackerClass.C2
        if any(w in text for w in _EXFIL_WORDS):
            return AttackerClass.OTHER_MALICIOUS
    return AttackerClass.OTHER_MALICIOUS


@dataclass
class IocSet:
    """Network indicators from one report, each with its attacker class."""

    name: str
    domains: dict[str, AttackerClass] = field(default_factory=dict)
    ips: dict[str, AttackerClass] = field(default_factory=dict)
    # /24s of indicator IPs that didn't parse as valid addresses (the
    # source reports occasionally carry a mangled octet, e.g.
    # "193.187.91[.]281"; the traffic still lands in that network).
    networks: dict[str, AttackerClass] = field(default_factory=dict)

    def add_domain(self, domain: str, cls: AttackerClass) -> None:
        domain = domain.lower().rstrip(".")
        if not domain or "." not in domain:
            return
        # A domain seen as C2 anywhere in the report stays C2.
        if self.domains.get(domain) != AttackerClass.C2:
            self.domains[domain] = cls

    def add_ip(self, ip: str, cls: AttackerClass) -> None:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            octets = ip.split(".")
            if len(octets) == 4:
                self.networks[".".join(octets[:3])] = cls
            return
        if self.ips.get(ip) != AttackerClass.C2:
            self.ips[ip] = cls

    def classify_ip(self, ip: str | None) -> AttackerClass | None:
        if not ip:
            return None
        if ip in self.ips:
            return self.ips[ip]
        prefix = ip.rsplit(".", 1)[0]
        return self.networks.get(prefix)

    def classify_domain(self, domain: str | None) -> AttackerClass | None:
        if not domain:
            return None
        domain = domain.lower().rstrip(".")
        if domain in self.domains:
            return self.domains[domain]
        # A subdomain of an indicator domain counts as the same indicator.
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self.domains:
                return self.domains[parent]
        return None

    def __len__(self) -> int:
        return len(self.domains) + len(self.ips) + len(self.networks)


def parse_report(path: str | Path) -> IocSet:
    path = Path(path)
    iocs = IocSet(name=path.stem)
    heading = ""

    for raw_line in path.read_text(errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Section headings are the all-caps lines ("INFECTION TRAFFIC:").
        letters = [c for c in stripped if c.isalpha()]
        if letters and all(c.isupper() for c in letters) and not stripped.startswith("-"):
            heading = stripped
            continue

        # Only defanged text carries indicators; everything else in these
        # reports is prose, hashes and filenames.
        if not any(marker in stripped for marker in ("[.]", "[:]", "(.)", "[dot]", "hxxp")):
            continue

        line = refang(stripped)
        cls = _class_for(stripped, heading)

        hosts_from_urls = set()
        for host, _port, _path in _URL_RE.findall(line):
            hosts_from_urls.add(host)
            if _IP_RE.fullmatch(host):
                iocs.add_ip(host, cls)
            else:
                iocs.add_domain(host, cls)

        for ip, _port in _IP_RE.findall(line):
            iocs.add_ip(ip, cls)

        # Bare hostnames (mail servers, C2 domains quoted without a scheme).
        line_without_ips = _IP_RE.sub(" ", line)
        for host in _HOST_RE.findall(line_without_ips):
            if host in hosts_from_urls:
                continue
            iocs.add_domain(host, cls)

    return iocs


def load_reports(directory: str | Path) -> dict[str, IocSet]:
    """Every ``*.txt`` report in ``directory``, keyed by its stem so it can
    be matched with the pcap of the same name."""
    directory = Path(directory)
    return {p.stem: parse_report(p) for p in sorted(directory.glob("*.txt"))}
