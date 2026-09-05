from __future__ import annotations

import re
import sqlite3
from collections import OrderedDict

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def build_fts_query(question: str) -> str:
    tokens = [t for t in TOKEN_RE.findall(question) if len(t.strip()) > 1]
    tokens = list(OrderedDict.fromkeys(tokens))
    if not tokens:
        return '""'
    escaped = [f'"{token.replace(chr(34), "")}"' for token in tokens]
    return " OR ".join(escaped)


def sparse_search(
    conn: sqlite3.Connection,
    question: str,
    top_k: int = 100,
    context_weight: float = 1.8,
    raw_weight: float = 1.0,
) -> list[int]:
    fts_query = build_fts_query(question)
    if fts_query == '""':
        return []

    rows = conn.execute(
        """
        SELECT rowid, bm25(node_fts, ?, ?) AS rank_score
        FROM node_fts
        WHERE node_fts MATCH ?
        ORDER BY rank_score
        LIMIT ?
        """,
        (context_weight, raw_weight, fts_query, int(top_k)),
    ).fetchall()
    return [int(row[0]) for row in rows]
