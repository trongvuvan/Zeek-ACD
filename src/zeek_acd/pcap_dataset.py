"""Turn a directory of scenario pcaps into labeled Zeek training data.

For every ``<name>.pcap`` there is a ``<name>.txt`` IOC report; this runs
Zeek over the pcap, then labels each connection by matching it against the
report's indicators, and writes a ``conn.log.labeled`` in exactly the
IoT-23 layout (Zeek TSV plus ``label`` and ``detailed-label`` columns), so
``data.py`` / ``train_dqn.py`` read it with no special casing.

A connection is matched to an indicator in three ways, in order:

1. its responder IP is an indicator IP (or falls in an indicator /24);
2. the TLS SNI (``ssl.log``) or HTTP ``Host`` (``http.log``) recorded for
   that ``uid`` is an indicator domain;
3. its responder IP was handed out by DNS in *this* capture as an answer
   for an indicator domain (``dns.log``) -- which is how a plain TCP or
   TLS-without-SNI flow to a malicious host still gets labeled.

Everything to a local/private address (the DNS resolver, NTP, ARP-ish
chatter) is BENIGN. What is left -- external traffic that matched no
indicator -- is controlled by ``--unmatched``. These captures are
purpose-built infection traffic, so ``malicious`` (the default) is the
honest reading for them; pass ``benign`` or ``drop`` for captures that
also contain real background browsing.
"""

from __future__ import annotations

import argparse
import gzip
import ipaddress
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .game import AttackerClass
from .iocs import IocSet, parse_report

ZEEK_BIN_CANDIDATES = ["zeek", "/usr/local/zeek/bin/zeek", "/opt/zeek/bin/zeek"]

# What we write into the two extra columns, per class. ``game.classify_label``
# maps these back: "C&C" -> C2, anything else malicious -> OTHER_MALICIOUS.
DETAILED_LABEL = {
    AttackerClass.C2: "C&C",
    AttackerClass.OTHER_MALICIOUS: "Malware-Traffic",
    AttackerClass.RECON: "PortScan",
    AttackerClass.DOS: "DDoS",
}


def find_zeek() -> str:
    for candidate in ZEEK_BIN_CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    raise SystemExit("zeek binary not found; looked for: " + ", ".join(ZEEK_BIN_CANDIDATES))


def run_zeek(zeek: str, pcap: Path, out_dir: Path, ignore_checksums: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # A capture taken on the sending host sees its own packets before the NIC
    # fills in checksums; without -C Zeek discards all of them.
    cmd = [zeek, "-C", "-r", str(pcap)] if ignore_checksums else [zeek, "-r", str(pcap)]
    proc = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"zeek failed on {pcap.name}: {proc.stderr.strip()[:400]}")


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Zeek TSV -> (field names, rows). Returns empty when the log is absent
    (a capture with no DNS simply has no dns.log)."""
    if not path.exists():
        return [], []
    return parse_tsv_text(path.read_text(errors="replace"))


def parse_tsv_text(text: str) -> tuple[list[str], list[list[str]]]:
    fields: list[str] = []
    rows: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("#fields"):
            fields = line.split("\t")[1:]
        elif line.startswith("#"):
            continue
        elif line.strip():
            rows.append(line.split("\t"))
    return fields, rows


def read_log(directory: Path, name: str) -> list[dict[str, str]]:
    """One Zeek log as a list of records, in whichever format it is on disk.

    A ``zeek -r`` run writes TSV; a zeekctl deployment with
    ``LogAscii::use_json = T`` writes one JSON object per line. Live logs
    here are the JSON kind, pcap replays the TSV kind, and both have to be
    labelable.

    A rotated log directory has no ``conn.log`` at all -- zeekctl archives
    it as ``conn.<from>-<to>.log.gz``, often several per day and not
    necessarily all in the same format. Those are read and concatenated.
    """
    stem = name[:-4] if name.endswith(".log") else name
    path = directory / name
    if not path.exists():
        rotated = sorted(directory.glob(f"{stem}.*.log.gz")) + \
                  sorted(directory.glob(f"{stem}.*.log"))
        records: list[dict[str, str]] = []
        for part in rotated:
            records.extend(_read_log_file(part))
        return records
    return _read_log_file(path)


def _read_log_file(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as fh:
            text = fh.read()
    else:
        text = path.read_text(errors="replace")
    if text.lstrip().startswith("{"):
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append({k: ("" if v is None else str(v)) for k, v in record.items()})
        return records
    fields, rows = parse_tsv_text(text)
    return [dict(zip(fields, row)) for row in rows]


def column(fields: list[str], row: list[str], name: str) -> str | None:
    try:
        value = row[fields.index(name)]
    except (ValueError, IndexError):
        return None
    return None if value in ("-", "(empty)") else value


def is_local(ip: str | None) -> bool:
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_private or addr.is_multicast or addr.is_loopback or addr.is_link_local


def build_uid_domains(log_dir: Path) -> dict[str, set[str]]:
    """uid -> hostnames that connection asked for (TLS SNI, HTTP Host)."""
    uid_domains: dict[str, set[str]] = {}
    for filename, field_name in (("ssl.log", "server_name"), ("http.log", "host")):
        for record in read_log(log_dir, filename):
            uid = record.get("uid")
            host = record.get(field_name)
            if uid and host and host not in ("-", "(empty)"):
                uid_domains.setdefault(uid, set()).add(host.lower().rstrip("."))
    return uid_domains


def build_dns_ip_classes(log_dir: Path, iocs: IocSet) -> dict[str, AttackerClass]:
    """IP -> class, for every address this capture's DNS handed out as an
    answer to an indicator domain."""
    ip_classes: dict[str, AttackerClass] = {}
    for record in read_log(log_dir, "dns.log"):
        query = record.get("query")
        answers = record.get("answers")
        if not query or not answers or answers in ("-", "(empty)"):
            continue
        # JSON logs carry answers as a list, TSV as a comma-joined string.
        answers = answers.strip("[]").replace("'", "").replace('"', "")
        cls = iocs.classify_domain(query)
        if cls is None:
            continue
        for answer in answers.split(","):
            answer = answer.strip()
            try:
                ipaddress.ip_address(answer)
            except ValueError:
                continue  # CNAME, not an address
            if ip_classes.get(answer) != AttackerClass.C2:
                ip_classes[answer] = cls
    return ip_classes


def label_conn_log(log_dir: Path, iocs: IocSet, unmatched: str) -> tuple[list[str], Counter]:
    """Returns the ``conn.log.labeled`` lines (header included) and a count
    per label, so the caller can report how much of the capture matched."""
    conn_path = log_dir / "conn.log"
    if not conn_path.exists():
        return [], Counter()

    uid_domains = build_uid_domains(log_dir)
    dns_ip_classes = build_dns_ip_classes(log_dir, iocs)

    fields, _ = read_tsv(conn_path)
    counts: Counter = Counter()
    out: list[str] = []

    for line in conn_path.read_text(errors="replace").splitlines():
        if line.startswith("#fields"):
            out.append(line + "\tlabel\tdetailed-label")
            continue
        if line.startswith("#types"):
            out.append(line + "\tstring\tstring")
            continue
        if line.startswith("#"):
            out.append(line)
            continue
        if not line.strip():
            continue

        row = line.split("\t")
        uid = column(fields, row, "uid")
        resp_h = column(fields, row, "id.resp_h")

        cls: AttackerClass | None = None
        if is_local(resp_h):
            cls = AttackerClass.BENIGN
        else:
            cls = iocs.classify_ip(resp_h)
            if cls is None:
                for domain in uid_domains.get(uid or "", ()):
                    cls = iocs.classify_domain(domain)
                    if cls is not None:
                        break
            if cls is None and resp_h in dns_ip_classes:
                cls = dns_ip_classes[resp_h]
            if cls is None:
                counts["unmatched-external"] += 1
                if unmatched == "drop":
                    continue
                cls = (AttackerClass.OTHER_MALICIOUS if unmatched == "malicious"
                       else AttackerClass.BENIGN)

        if cls == AttackerClass.BENIGN:
            out.append(line + "\tBenign\t-")
        else:
            out.append(line + f"\tMalicious\t{DETAILED_LABEL[cls]}")
        counts[cls.name] += 1

    return out, counts


# The conn.log columns we write out, in Zeek's own order. Anything the
# source log doesn't carry is written as the unset marker.
CONN_FIELDS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig",
    "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes",
]


def merge_reports(ioc_dir: Path, pattern: str = "*.txt") -> IocSet:
    """One IocSet covering every report in the directory.

    A live sensor has no idea which scenario a connection belongs to, so
    labeling a live log means matching against all known indicators at
    once. C2 wins over other classes when reports disagree."""
    merged = IocSet(name="merged")
    for report in sorted(ioc_dir.glob(pattern)):
        iocs = parse_report(report)
        for domain, cls in iocs.domains.items():
            merged.add_domain(domain, cls)
        for ip, cls in iocs.ips.items():
            merged.add_ip(ip, cls)
        for net, cls in iocs.networks.items():
            if merged.networks.get(net) != AttackerClass.C2:
                merged.networks[net] = cls
    return merged


def classify_record(record: dict, iocs: IocSet, uid_domains: dict[str, set[str]],
                    dns_ip_classes: dict[str, AttackerClass],
                    unmatched: str,
                    benign_sources: frozenset[str] = frozenset()) -> AttackerClass | None:
    """The class for one connection, or None when ``--unmatched drop`` says
    to throw it away."""
    resp_h = record.get("id.resp_h")
    if is_local(resp_h):
        return AttackerClass.BENIGN

    cls = iocs.classify_ip(resp_h)
    if cls is None:
        for domain in uid_domains.get(record.get("uid") or "", ()):
            cls = iocs.classify_domain(domain)
            if cls is not None:
                break
    if cls is None and resp_h in dns_ip_classes:
        cls = dns_ip_classes[resp_h]
    if cls is not None:
        return cls
    # Hosts named with --benign-src are known not to be replaying attack
    # traffic, so what they do that matches no indicator is ordinary
    # traffic -- which is where outbound HTTPS to real sites comes from.
    if record.get("id.orig_h") in benign_sources:
        return AttackerClass.BENIGN
    if unmatched == "drop":
        return None
    return AttackerClass.OTHER_MALICIOUS if unmatched == "malicious" else AttackerClass.BENIGN


def write_labeled_tsv(rows: list[tuple[dict, AttackerClass]], path: Path) -> None:
    """Writes records plus their labels as a Zeek TSV in the IoT-23 layout."""
    fields = CONN_FIELDS + ["label", "detailed-label"]
    lines = ["#separator \\x09", "#set_separator\t,", "#empty_field\t(empty)",
             "#unset_field\t-", "#path\tconn", "#fields\t" + "\t".join(fields)]
    for record, cls in rows:
        values = [str(record.get(name, "-") or "-") for name in CONN_FIELDS]
        if cls == AttackerClass.BENIGN:
            values += ["Benign", "-"]
        else:
            values += ["Malicious", DETAILED_LABEL[cls]]
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n")


def label_log_dir(log_dir: Path, iocs: IocSet, unmatched: str,
                  benign_sources: frozenset[str] = frozenset()) -> tuple[list, Counter]:
    """Labels an existing Zeek log directory (TSV or JSON), e.g. a live
    spool directory, rather than a pcap we just ran Zeek over."""
    conn_records = read_log(log_dir, "conn.log")
    uid_domains = build_uid_domains(log_dir)
    dns_ip_classes = build_dns_ip_classes(log_dir, iocs)
    counts: Counter = Counter()
    rows = []
    for record in conn_records:
        if not is_local(record.get("id.resp_h")):
            matched = (iocs.classify_ip(record.get("id.resp_h")) is not None
                       or record.get("id.resp_h") in dns_ip_classes
                       or any(iocs.classify_domain(d) is not None
                              for d in uid_domains.get(record.get("uid") or "", ())))
            if not matched:
                counts["unmatched-external"] += 1
        cls = classify_record(record, iocs, uid_domains, dns_ip_classes, unmatched,
                              benign_sources)
        if cls is None:
            continue
        counts[cls.name] += 1
        rows.append((record, cls))
    return rows, counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcap-dir", default="/root/pcaps")
    p.add_argument("--zeek-log-dir", default=None,
                    help="label an existing Zeek log directory (e.g. a live spool dir, "
                         "TSV or JSON) against every IOC report, instead of running "
                         "Zeek over pcaps")
    p.add_argument("--ioc-glob", default="*.txt",
                    help="restrict --zeek-log-dir labeling to these reports, e.g. '2025-*.txt'. "
                         "Traffic belonging to the other scenarios then matches nothing and is "
                         "handled by --unmatched, which is how a live log gets split into a "
                         "training half and a held-out half by scenario")
    p.add_argument("--pcap", default=None,
                    help="run Zeek over this one capture and label it like --zeek-log-dir "
                         "(against every IOC report, --unmatched for the rest). For a "
                         "capture of known-clean traffic, e.g. from tools/gen_benign.sh")
    p.add_argument("--ignore-checksums", action="store_true",
                    help="pass -C to Zeek; needed for captures taken on the sending host "
                         "(NIC checksum offloading)")
    p.add_argument("--benign-src", action="append", default=None,
                    help="a source IP whose unmatched traffic is ordinary traffic rather "
                         "than an unknown; repeat per host. Use it for hosts you know are "
                         "not replaying attack captures -- their normal browsing is the "
                         "benign class the malware captures don't contain")
    p.add_argument("--out-file", default=None,
                    help="where --zeek-log-dir writes its labeled log "
                         "(default: <out-dir>/live/conn.log.labeled)")
    p.add_argument("--ioc-dir", default="/root/indicators")
    p.add_argument("--out-dir", default="data/mta")
    p.add_argument("--unmatched", choices=["malicious", "benign", "drop"], default="malicious",
                    help="what to do with external traffic that matched no indicator")
    p.add_argument("--keep-zeek-logs", action="store_true",
                    help="keep the raw per-pcap Zeek output next to the labeled log")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.pcap:
        work_dir = Path(tempfile.mkdtemp(prefix="zeek-acd-"))
        try:
            run_zeek(find_zeek(), Path(args.pcap).resolve(), work_dir, args.ignore_checksums)
            iocs = merge_reports(Path(args.ioc_dir), args.ioc_glob)
            rows, counts = label_log_dir(work_dir, iocs, args.unmatched,
                                         frozenset(args.benign_src or ()))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    elif args.zeek_log_dir:
        iocs = merge_reports(Path(args.ioc_dir), args.ioc_glob)
        rows, counts = label_log_dir(Path(args.zeek_log_dir), iocs, args.unmatched,
                                     frozenset(args.benign_src or ()))
    if args.pcap or args.zeek_log_dir:
        if not rows:
            raise SystemExit(f"no conn.log records found in {args.pcap or args.zeek_log_dir}")
        out_file = Path(args.out_file or Path(args.out_dir) / "live" / "conn.log.labeled")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        write_labeled_tsv(rows, out_file)
        print(f"[pcap-dataset] {len(iocs)} indicators from {args.ioc_dir}")
        print(f"[pcap-dataset] {out_file}: " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return

    zeek = find_zeek()
    pcap_dir = Path(args.pcap_dir)
    ioc_dir = Path(args.ioc_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pcaps = sorted(pcap_dir.glob("*.pcap")) + sorted(pcap_dir.glob("*.pcapng"))
    if not pcaps:
        raise SystemExit(f"no pcaps in {pcap_dir}")

    grand_total: Counter = Counter()
    written = 0

    for pcap in pcaps:
        report = ioc_dir / f"{pcap.stem}.txt"
        if not report.exists():
            print(f"[pcap-dataset] {pcap.name}: no matching IOC report, skipped")
            continue
        iocs = parse_report(report)

        scenario_dir = out_root / pcap.stem
        scenario_dir.mkdir(parents=True, exist_ok=True)
        work_dir = (scenario_dir / "zeek" if args.keep_zeek_logs
                    else Path(tempfile.mkdtemp(prefix="zeek-acd-")))
        try:
            run_zeek(zeek, pcap, work_dir)
            lines, counts = label_conn_log(work_dir, iocs, args.unmatched)
        finally:
            if not args.keep_zeek_logs:
                shutil.rmtree(work_dir, ignore_errors=True)

        if not lines:
            print(f"[pcap-dataset] {pcap.name}: zeek produced no conn.log, skipped")
            continue

        (scenario_dir / "conn.log.labeled").write_text("\n".join(lines) + "\n")
        written += 1
        grand_total.update(counts)
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[pcap-dataset] {pcap.stem}: {len(iocs)} indicators -> {summary}")

    print(f"\n[pcap-dataset] wrote {written} labeled logs under {out_root}")
    print("[pcap-dataset] totals: " + "  ".join(f"{k}={v}" for k, v in sorted(grand_total.items())))
    if grand_total["unmatched-external"]:
        print(f"[pcap-dataset] note: {grand_total['unmatched-external']} external connections "
              f"matched no indicator and were treated as '{args.unmatched}'")


if __name__ == "__main__":
    sys.exit(main())
