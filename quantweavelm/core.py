"""Canonical I/O and strict input validation primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class QuantWeaveError(ValueError):
    """Safe user-facing failure for malformed or inconsistent evidence."""


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically with a trailing newline."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: str | Path, max_bytes: int = 2_000_000) -> Any:
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            raise QuantWeaveError(f"{source} exceeds the {max_bytes}-byte limit")
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuantWeaveError(f"cannot read JSON from {source}: {exc}") from exc


def load_jsonl(path: str | Path, max_bytes: int = 20_000_000,
               max_rows: int = 1_000_000) -> list[Any]:
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            raise QuantWeaveError(f"{source} exceeds the {max_bytes}-byte limit")
        rows = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise QuantWeaveError(f"blank line at {source}:{number}")
            if len(rows) >= max_rows:
                raise QuantWeaveError(f"{source} exceeds the {max_rows}-row limit")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise QuantWeaveError(f"invalid JSON at {source}:{number}: {exc.msg}") from exc
        return rows
    except (OSError, UnicodeError) as exc:
        raise QuantWeaveError(f"cannot read JSONL from {source}: {exc}") from exc


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QuantWeaveError(f"{field} must be a finite JSON number")
    return float(value)


def timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise QuantWeaveError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise QuantWeaveError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise QuantWeaveError(f"{field} must use canonical ISO-8601 seconds and Z")
    return parsed
