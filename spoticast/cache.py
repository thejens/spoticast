"""Shared disk cache — all cached data lives in .research_cache/ for visibility."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".research_cache")


def cache_key(*parts: str) -> str:
    """Generate a short hex key from arbitrary string parts."""
    raw = "|".join(p.strip().lower() for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _path(prefix: str, key: str) -> Path:
    subdir = CACHE_DIR / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key}.json"


def get(prefix: str, key: str) -> Any | None:
    """Read a cached value. Returns None on miss or corrupt data."""
    p = _path(prefix, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def put(prefix: str, key: str, data: Any) -> None:
    """Write a value to the disk cache."""
    try:
        _path(prefix, key).write_text(json.dumps(data, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Cache write failed (%s/%s): %s", prefix, key, exc)
