from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from .io import iter_context_documents
from .parser import parse_document
from .schema import LegalNode


def create_database(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute(
        """
        CREATE TABLE nodes (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT UNIQUE NOT NULL,
            document_id TEXT NOT NULL,
            source_name TEXT,
            node_type TEXT NOT NULL,
            label TEXT,
            depth INTEGER NOT NULL,
            parent_id TEXT,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            order_index INTEGER NOT NULL,
            header_text TEXT,
            legal_path TEXT,
            context_text TEXT,
            raw_text TEXT,
            retrieval_text TEXT,
            is_indexable INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX idx_nodes_parent ON nodes(parent_id)")
    conn.execute("CREATE INDEX idx_nodes_document_offsets ON nodes(document_id, start_offset)")
    conn.execute("CREATE INDEX idx_nodes_indexable ON nodes(is_indexable)")

    # rowid is intentionally aligned with nodes.row_id for indexable nodes.
    conn.execute(
        """
        CREATE VIRTUAL TABLE node_fts USING fts5(
            context_text,
            raw_text,
            tokenize='unicode61 remove_diacritics 0'
        )
        """
    )
    conn.commit()
    return conn


def _insert_node(conn: sqlite3.Connection, node: LegalNode) -> int:
    cursor = conn.execute(
        """
        INSERT INTO nodes (
            node_id, document_id, source_name, node_type, label, depth,
            parent_id, start_offset, end_offset, order_index,
            header_text, legal_path, context_text, raw_text,
            retrieval_text, is_indexable
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node.node_id,
            node.document_id,
            node.source_name,
            node.node_type,
            node.label,
            node.depth,
            node.parent_id,
            node.start_offset,
            node.end_offset,
            node.order_index,
            node.header_text,
            node.legal_path,
            node.context_text,
            node.raw_text,
            node.retrieval_text,
            int(node.is_indexable),
        ),
    )
    row_id = int(cursor.lastrowid)

    if node.is_indexable:
        conn.execute(
            "INSERT INTO node_fts(rowid, context_text, raw_text) VALUES (?, ?, ?)",
            (row_id, node.context_text, node.raw_text),
        )
    return row_id


def build_corpus_database(source_path: str | Path, db_path: str | Path) -> dict:
    conn = create_database(db_path)
    total_documents = 0
    total_nodes = 0
    total_indexable = 0

    try:
        for doc in tqdm(iter_context_documents(source_path), desc="Parsing legal documents"):
            document_id = str(doc.get("id", total_documents))
            source_name = str(doc.get("name", ""))
            passage = str(doc.get("passage", ""))

            nodes = parse_document(document_id, source_name, passage)
            for node in nodes:
                _insert_node(conn, node)
                total_nodes += 1
                total_indexable += int(node.is_indexable)

            total_documents += 1
            if total_documents % 100 == 0:
                conn.commit()

        conn.commit()
    finally:
        conn.close()

    return {
        "documents": total_documents,
        "nodes": total_nodes,
        "indexable_nodes": total_indexable,
        "db_path": str(db_path),
    }


def connect_database(db_path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(db_path)
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_node_by_row_id(conn: sqlite3.Connection, row_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM nodes WHERE row_id = ?", (row_id,)).fetchone()
    return dict(row) if row else None


def row_id_from_node_id(conn: sqlite3.Connection, node_id: str | None) -> int | None:
    if not node_id:
        return None
    row = conn.execute("SELECT row_id FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
    return int(row[0]) if row else None


def get_children_row_ids(
    conn: sqlite3.Connection,
    node_id: str,
    indexable_only: bool = True,
) -> list[int]:
    if indexable_only:
        rows = conn.execute(
            """
            SELECT row_id FROM nodes
            WHERE parent_id = ? AND is_indexable = 1
            ORDER BY start_offset
            """,
            (node_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT row_id FROM nodes WHERE parent_id = ? ORDER BY start_offset",
            (node_id,),
        ).fetchall()
    return [int(r[0]) for r in rows]


def fetch_nodes_map(conn: sqlite3.Connection, row_ids: Iterable[int]) -> dict[int, dict]:
    row_ids = list(dict.fromkeys(int(x) for x in row_ids))
    if not row_ids:
        return {}
    placeholders = ",".join("?" for _ in row_ids)
    rows = conn.execute(
        f"SELECT * FROM nodes WHERE row_id IN ({placeholders})",
        row_ids,
    ).fetchall()
    return {int(row["row_id"]): dict(row) for row in rows}
