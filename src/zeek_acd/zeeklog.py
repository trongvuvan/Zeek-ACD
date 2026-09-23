"""Reader for Zeek's default ASCII TSV log format.

Handles both static files (e.g. an IoT-23 ``conn.log.labeled``) and the
header conventions Zeek writes for any ``*.log`` file (``#separator``,
``#fields``, ``#types``, ...). The same parser is reused for offline
training data and for tailing a live ``conn.log``.

IoT-23's ``conn.log.labeled`` files carry two extra columns (``label`` and
``detailed-label``) after ``tunnel_parents``, in one of several shapes
depending on the mirror/scenario:

- joined to ``tunnel_parents`` by runs of spaces instead of tabs, in both
  the ``#fields`` header and the data rows;
- proper tab-separated columns, but named ``det_label`` in the header;
- absent from ``#fields`` entirely (callers pass ``extra_fields``).

All three are normalized below so records always expose ``label`` and
``detailed-label``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterator, Optional, TextIO

UNSET = "-"
EMPTY = "(empty)"

# Header names some IoT-23 files use for the canonical column names.
FIELD_ALIASES = {"det_label": "detailed-label"}


@dataclass
class ZeekLogHeader:
    separator: str = "\t"
    set_separator: str = ","
    empty_field: str = "(empty)"
    unset_field: str = "-"
    path: str = ""
    fields: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    """Trailing columns present in data rows but absent from ``#fields``
    (the IoT-23 ``label`` / ``detailed-label`` convention)."""


def _unescape_separator(token: str) -> str:
    # Zeek encodes the separator itself as \x09 etc. in the header line.
    if token.startswith("\\x"):
        return chr(int(token[2:], 16))
    return token


def parse_header(lines: list[str], extra_fields: Optional[list[str]] = None) -> ZeekLogHeader:
    header = ZeekLogHeader(extra_fields=extra_fields or [])
    for line in lines:
        if not line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        key = parts[0].lstrip("#")
        if key == "separator":
            header.separator = _unescape_separator(parts[1])
        elif key == "set_separator":
            header.set_separator = parts[1]
        elif key == "empty_field":
            header.empty_field = parts[1]
        elif key == "unset_field":
            header.unset_field = parts[1]
        elif key == "path":
            header.path = parts[1]
        elif key == "fields":
            # Space-joined names (IoT-23 quirk) are split into separate columns.
            names = [n for part in parts[1:] for n in part.split()]
            header.fields = [FIELD_ALIASES.get(n, n) for n in names]
        elif key == "types":
            header.types = parts[1:]
    # Extra names already declared in the header must not be appended again.
    header.extra_fields = [f for f in header.extra_fields if f not in header.fields]
    return header


def parse_row(header: ZeekLogHeader, line: str) -> Optional[dict[str, str]]:
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    values = line.split(header.separator)
    names = header.fields + header.extra_fields
    if len(values) < len(names) and values and " " in values[-1]:
        # IoT-23 rows whose trailing columns are joined by spaces, not tabs.
        values = values[:-1] + values[-1].split()
    if len(values) < len(header.fields):
        return None
    record = dict(zip(names, values))
    # Any columns beyond declared+extra names are ignored; any declared
    # extra names beyond the row's length are left absent.
    return record


class ZeekLogFile:
    """Parses a complete, static Zeek TSV log file (e.g. training data)."""

    def __init__(self, fh: TextIO, extra_fields: Optional[list[str]] = None):
        self._fh = fh
        header_lines: list[str] = []
        # readline() rather than iteration: text-file iteration disables tell().
        while True:
            pos = fh.tell()
            raw = fh.readline()
            if not raw.startswith("#"):
                break
            header_lines.append(raw)
        fh.seek(pos)
        self.header = parse_header(header_lines, extra_fields=extra_fields)

    def __iter__(self) -> Iterator[dict[str, str]]:
        for raw in self._fh:
            rec = parse_row(self.header, raw)
            if rec is not None:
                yield rec


def open_log(path: str, extra_fields: Optional[list[str]] = None) -> ZeekLogFile:
    fh = open(path, "r", encoding="utf-8", errors="replace")
    return ZeekLogFile(fh, extra_fields=extra_fields)


def records_from_string(text: str, extra_fields: Optional[list[str]] = None) -> list[dict[str, str]]:
    return list(ZeekLogFile(io.StringIO(text), extra_fields=extra_fields))
