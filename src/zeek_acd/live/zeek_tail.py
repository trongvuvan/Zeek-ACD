"""Follows a live Zeek log file (default TSV ``conn.log``, or JSON when
Zeek is configured with ``LogAscii::use_json = T``) the way ``tail -F``
does: it keeps reading appended lines and reopens the file if Zeek rotates
it (new inode at the same path).
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional

from ..zeeklog import ZeekLogHeader, parse_header, parse_row

POLL_INTERVAL_SECONDS = 0.5


class LiveZeekTail:
    def __init__(self, path: str, format: str = "auto", extra_fields: Optional[list[str]] = None):
        self.path = path
        self.format = format
        self.extra_fields = extra_fields or []
        self._fh = None
        self._ino = None
        self._header: Optional[ZeekLogHeader] = None
        self._resolved_format: Optional[str] = None

    def _open(self, seek_to_end: bool) -> None:
        self._fh = open(self.path, "r", encoding="utf-8", errors="replace")
        self._ino = os.fstat(self._fh.fileno()).st_ino
        self._header = None
        self._resolved_format = None
        if seek_to_end:
            self._fh.seek(0, os.SEEK_END)
        else:
            self._consume_header_if_tsv()

    def _consume_header_if_tsv(self) -> None:
        pos = self._fh.tell()
        header_lines = []
        line = self._fh.readline()
        while line.startswith("#"):
            header_lines.append(line)
            pos = self._fh.tell()
            line = self._fh.readline()
        self._fh.seek(pos)
        if header_lines:
            self._header = parse_header(header_lines, extra_fields=self.extra_fields)
            self._resolved_format = "tsv"

    def _rotated(self) -> bool:
        try:
            return os.stat(self.path).st_ino != self._ino
        except FileNotFoundError:
            return False

    def _parse_line(self, line: str) -> Optional[dict]:
        line = line.rstrip("\n")
        if not line:
            return None
        if line.startswith("#"):
            if self.format in ("auto", "tsv"):
                self._header = parse_header([line], extra_fields=self.extra_fields)
            return None
        if self._resolved_format is None:
            self._resolved_format = "json" if line.lstrip().startswith("{") else "tsv"
        if self._resolved_format == "json" and self.format != "tsv":
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
        if self._header is None:
            return None
        return parse_row(self._header, line)

    def __iter__(self) -> Iterator[dict]:
        """Yields records forever, starting from the current end of file.
        Use ``from_start()`` instead to also read what's already there."""
        self._open(seek_to_end=True)
        yield from self._follow()

    def from_start(self) -> Iterator[dict]:
        self._open(seek_to_end=False)
        yield from self._follow()

    def _follow(self) -> Iterator[dict]:
        while True:
            line = self._fh.readline()
            if line:
                rec = self._parse_line(line)
                if rec is not None:
                    yield rec
                continue
            if self._rotated():
                self._fh.close()
                self._open(seek_to_end=False)
                continue
            time.sleep(POLL_INTERVAL_SECONDS)
