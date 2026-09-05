from __future__ import annotations

import sqlite3
from collections import defaultdict

from .database import fetch_nodes_map, row_id_from_node_id


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
    max_nodes: int = 6,
    parent_margin: float = 0.05,
) -> list[dict]:
    """Threshold, compact hierarchy, then restore original legal order."""
    if not reranked:
        return []

    score_map = {int(row_id): float(score) for row_id, score in reranked}
    selected_ids = [int(row_id) for row_id, score in reranked if score >= threshold]

    # Safety fallback: do not return an empty answer only because tau is too high.
    if not selected_ids:
        selected_ids = [int(reranked[0][0])]

    node_map = fetch_nodes_map(conn, selected_ids)

    # If multiple point children from the same clause are selected and the
    # parent itself is nearly above threshold, promote to the parent clause.
    point_groups: dict[str, list[int]] = defaultdict(list)
    for row_id in list(selected_ids):
        node = node_map.get(row_id)
        if node and node["node_type"] == "point" and node.get("parent_id"):
            point_groups[node["parent_id"]].append(row_id)

    for parent_node_id, point_ids in point_groups.items():
        if len(point_ids) < 2:
            continue
        parent_row_id = row_id_from_node_id(conn, parent_node_id)
        if parent_row_id is None:
            continue
        parent_score = score_map.get(parent_row_id)
        if parent_score is not None and parent_score >= threshold - parent_margin:
            selected_ids = [x for x in selected_ids if x not in point_ids]
            selected_ids.append(parent_row_id)

    selected_ids = list(dict.fromkeys(selected_ids))
    node_map = fetch_nodes_map(conn, selected_ids)

    candidates: list[tuple[int, float, dict]] = []
    for row_id in selected_ids:
        node = node_map.get(row_id)
        if node:
            candidates.append((row_id, score_map.get(row_id, threshold), node))

    # Deeper (smaller) legal units first. A parent is removed if a smaller
    # sufficient descendant has already been retained.
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
