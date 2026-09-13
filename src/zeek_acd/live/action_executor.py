"""Turns a defender decision into an effect on the world.

Safety model: the default executor only writes an audit record and never
touches the network. A real enforcement backend (currently: nftables) must
be explicitly selected *and* explicitly taken out of dry-run mode by the
operator -- this module will not do that on its own, and every action is
always logged first regardless of which backend is active.
"""

from __future__ import annotations

import ipaddress
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..game import DefenderAction

ENFORCEABLE_ACTIONS = {DefenderAction.BLOCK_SRC, DefenderAction.ISOLATE_HOST, DefenderAction.RATE_LIMIT}


@dataclass
class Decision:
    ts: float
    uid: Optional[str]
    orig_h: Optional[str]
    resp_h: Optional[str]
    action: DefenderAction
    strategy: list[float]


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: Decision, extra: Optional[dict] = None) -> None:
        entry = {
            "ts": decision.ts,
            "uid": decision.uid,
            "orig_h": decision.orig_h,
            "resp_h": decision.resp_h,
            "action": decision.action.name,
            "strategy": decision.strategy,
        }
        if extra:
            entry.update(extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class ActionExecutor:
    """Base class. ``execute`` is always called; subclasses decide what,
    if anything, happens beyond the audit log entry every executor writes.
    """

    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log

    def execute(self, decision: Decision) -> None:
        raise NotImplementedError


class LogOnlyExecutor(ActionExecutor):
    """The default and recommended executor: observe-and-recommend only.
    Nothing on the network is touched."""

    def execute(self, decision: Decision) -> None:
        self.audit_log.write(decision, extra={"enforced": False, "backend": "log_only"})


class NftablesExecutor(ActionExecutor):
    """Adds source IPs to an nftables set for BLOCK_SRC / ISOLATE_HOST, and
    to a rate-limiting set for RATE_LIMIT. Requires ``dry_run=False`` to
    actually invoke ``nft``; otherwise it behaves like ``LogOnlyExecutor``
    but labels the audit entry so operators can see what *would* have run.

    You are responsible for pre-creating the referenced table/sets, e.g.:
        nft add table inet acd
        nft add set inet acd blocked '{ type ipv4_addr; }'
        nft add set inet acd ratelimited '{ type ipv4_addr; }'
        nft add rule inet acd input ip saddr @blocked drop
    """

    def __init__(
        self,
        audit_log: AuditLog,
        table: str = "inet acd",
        blocked_set: str = "blocked",
        ratelimited_set: str = "ratelimited",
        dry_run: bool = True,
    ):
        super().__init__(audit_log)
        self.table = table
        self.blocked_set = blocked_set
        self.ratelimited_set = ratelimited_set
        self.dry_run = dry_run

    def _valid_ip(self, ip: Optional[str]) -> Optional[str]:
        if not ip:
            return None
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            return None

    def _nft_add(self, set_name: str, ip: str) -> tuple[bool, str]:
        cmd = ["nft", "add", "element"] + self.table.split() + [set_name, "{", ip, "}"]
        if self.dry_run:
            return True, "dry_run: " + " ".join(cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
            return True, " ".join(cmd)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return False, f"{' '.join(cmd)} FAILED: {exc}"

    def execute(self, decision: Decision) -> None:
        ip = self._valid_ip(decision.orig_h)
        extra = {"enforced": False, "backend": "nftables", "dry_run": self.dry_run}

        if decision.action not in ENFORCEABLE_ACTIONS or ip is None:
            self.audit_log.write(decision, extra=extra)
            return

        set_name = self.ratelimited_set if decision.action == DefenderAction.RATE_LIMIT else self.blocked_set
        ok, detail = self._nft_add(set_name, ip)
        extra.update({"enforced": ok and not self.dry_run, "command": detail})
        self.audit_log.write(decision, extra=extra)


def build_executor(
    kind: str,
    audit_path: str,
    dry_run: bool = True,
    table: str = "inet acd",
    blocked_set: str = "blocked",
    ratelimited_set: str = "ratelimited",
) -> ActionExecutor:
    audit_log = AuditLog(audit_path)
    if kind == "log_only":
        return LogOnlyExecutor(audit_log)
    if kind == "nftables":
        return NftablesExecutor(
            audit_log, table=table, blocked_set=blocked_set,
            ratelimited_set=ratelimited_set, dry_run=dry_run,
        )
    raise ValueError(f"unknown executor kind: {kind}")
