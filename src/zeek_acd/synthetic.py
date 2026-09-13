"""Synthetic Zeek conn-log-shaped records for smoke-testing the pipeline
without needing to first download a real dataset (e.g. IoT-23)."""

from __future__ import annotations

import numpy as np

PROFILES = {
    "BENIGN": dict(detailed="-", label="Benign", dur=(0.01, 5.0), bytes=(40, 4000),
                   proto=("tcp", "udp"), conn_state=("SF", "S1", "RSTO"), service=("http", "dns", "-")),
    "RECON": dict(detailed="PartOfAHorizontalPortScan", label="Malicious", dur=(0.0, 0.05),
                  bytes=(0, 60), proto=("tcp",), conn_state=("S0", "REJ"), service=("-",)),
    "DOS": dict(detailed="DDoS", label="Malicious", dur=(0.0, 0.2), bytes=(0, 200),
                proto=("udp", "tcp"), conn_state=("S0", "SF"), service=("-",)),
    "C2": dict(detailed="C&C", label="Malicious", dur=(5.0, 120.0), bytes=(20, 500),
               proto=("tcp",), conn_state=("SF", "S1"), service=("-", "ssl")),
    "OTHER_MALICIOUS": dict(detailed="FileDownload", label="Malicious", dur=(1.0, 30.0),
                             bytes=(1000, 500000), proto=("tcp",), conn_state=("SF",), service=("-",)),
}


def _random_ip(rng: np.random.Generator, prefix: str) -> str:
    return f"{prefix}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"


def generate_synthetic_records(
    n: int, seed: int = 0, class_weights: dict[str, float] | None = None
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    weights = class_weights or {
        "BENIGN": 0.7, "RECON": 0.12, "DOS": 0.08, "C2": 0.05, "OTHER_MALICIOUS": 0.05
    }
    names = list(weights.keys())
    probs = np.array([weights[k] for k in names], dtype=np.float64)
    probs /= probs.sum()

    records = []
    n_sources = max(20, n // 30)
    source_ips = [_random_ip(rng, "10.0") for _ in range(n_sources)]
    # A handful of sources are "compromised" and repeatedly emit one
    # non-benign class, so suppression actions have something to suppress.
    hostile_sources = {
        src: names[int(rng.integers(1, len(names)))]
        for src in rng.choice(source_ips, size=max(1, n_sources // 5), replace=False)
    }

    for i in range(n):
        src = source_ips[int(rng.integers(0, n_sources))]
        cls = hostile_sources.get(src) if src in hostile_sources and rng.random() < 0.8 else None
        if cls is None:
            cls = names[int(rng.choice(len(names), p=probs))]
        p = PROFILES[cls]

        duration = float(rng.uniform(*p["dur"]))
        orig_bytes = int(rng.uniform(*p["bytes"]))
        resp_bytes = int(orig_bytes * rng.uniform(0.2, 1.5))
        proto = str(rng.choice(p["proto"]))
        conn_state = str(rng.choice(p["conn_state"]))
        service = str(rng.choice(p["service"]))
        history = "".join(rng.choice(list("ShAaDdFf"), size=int(rng.integers(1, 5))))

        records.append({
            "ts": f"{1600000000 + i}.000000",
            "uid": f"C{i:08d}",
            "id.orig_h": src,
            "id.orig_p": str(int(rng.integers(1024, 65535))),
            "id.resp_h": _random_ip(rng, "172.16"),
            "id.resp_p": str(int(rng.choice([80, 443, 53, 23, 8080]))),
            "proto": proto,
            "service": service,
            "duration": f"{duration:.6f}",
            "orig_bytes": str(orig_bytes),
            "resp_bytes": str(resp_bytes),
            "conn_state": conn_state,
            "local_orig": "-",
            "local_resp": "-",
            "missed_bytes": "0",
            "history": history,
            "orig_pkts": str(max(1, orig_bytes // 60)),
            "orig_ip_bytes": str(orig_bytes + 40),
            "resp_pkts": str(max(0, resp_bytes // 60)),
            "resp_ip_bytes": str(resp_bytes + 40 if resp_bytes else 0),
            "tunnel_parents": "-",
            "label": p["label"],
            "detailed-label": p["detailed"],
        })
    return records
