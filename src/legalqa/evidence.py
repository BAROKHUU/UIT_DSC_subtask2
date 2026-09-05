from __future__ import annotations

import sqlite3

from .database import fetch_nodes_map


def contains_node(parent: dict, child: dict) -> bool:
    return (
        parent["document_id"] == child["document_id"]
        and int(parent["start_offset"]) <= int(child["start_offset"])
        and int(parent["end_offset"]) >= int(child["end_offset"])
        and int(parent["depth"]) < int(child["depth"])
    )


def select_evidence(
    conn: sqlite3.Connection,
    reranked: list[tuple[int, float]],
    threshold: float,
    max_nodes: int = 10,
) -> list[dict]:
    """Keep relevant fine-grained nodes, then restore original legal order."""
    if not reranked:
        return []

    score_map = {int(row_id): float(score) for row_id, score in reranked}
    selected_ids = [int(row_id) for row_id, score in reranked if score >= threshold]

    # Safety fallback: do not return an empty answer only because tau is too high.
    if not selected_ids:
        selected_ids = [int(reranked[0][0])]

    selected_ids = list(dict.fromkeys(selected_ids))
    node_map = fetch_nodes_map(conn, selected_ids)

    candidates: list[tuple[int, float, dict]] = []
    for row_id in selected_ids:
        node = node_map.get(row_id)
        if node:
            candidates.append((row_id, score_map.get(row_id, threshold), node))

    # Deeper (smaller) legal units first. If both a parent and one or more
    # descendants pass the relevance threshold, retain only those descendants.
    # Never promote relevant children back to their broader parent: a parent's
    # raw_text may span every child and would leak unrelated provisions.
    candidates.sort(key=lambda x: (int(x[2]["depth"]), x[1]), reverse=True)
    compacted: list[tuple[int, float, dict]] = []
    for candidate in candidates:
        cand_node = candidate[2]
        if any(contains_node(cand_node, kept[2]) for kept in compacted):
            continue
        compacted.append(candidate)

    # Enforce output budget by relevance, then restore source order.
    compacted.sort(key=lambda x: x[1], reverse=True)
    compacted = compacted[:max_nodes]
    compacted.sort(key=lambda x: (str(x[2]["document_id"]), int(x[2]["start_offset"])))

    evidence: list[dict] = []
    for row_id, score, node in compacted:
        item = dict(node)
        item["rerank_score"] = float(score)
        evidence.append(item)
    return evidence
