"""SQLite 情报缓存：search_cache 表（address / intelligence / timestamp）。"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_CACHE_TTL_SEC = 24 * 3600


def _db_path() -> Path:
    custom = (os.environ.get("SEARCH_CACHE_DATABASE") or "").strip()
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent / "search_cache.sqlite"


def init_search_cache_table() -> None:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_cache (
                address TEXT PRIMARY KEY NOT NULL,
                intelligence TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )


def get_cached_intelligence(address: str) -> str | None:
    """未过期则返回已存摘要（不含「来自本地缓存」后缀），过期或无记录返回 None。"""
    init_search_cache_table()
    addr = address.strip()
    path = _db_path()
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT intelligence, timestamp FROM search_cache WHERE address = ?",
            (addr,),
        ).fetchone()
        if not row:
            return None
        intel, ts = row
        if time.time() - float(ts) > _CACHE_TTL_SEC:
            conn.execute("DELETE FROM search_cache WHERE address = ?", (addr,))
            return None
    return intel


def set_cached_intelligence(address: str, intelligence: str) -> None:
    init_search_cache_table()
    addr = address.strip()
    path = _db_path()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO search_cache (address, intelligence, timestamp)
            VALUES (?, ?, ?)
            """,
            (addr, intelligence, time.time()),
        )
