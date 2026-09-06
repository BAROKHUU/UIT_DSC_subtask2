from __future__ import annotations

import re
import sqlite3

from .database import fetch_nodes_map


ARTICLE_HEADING_RE = re.compile(r'^Điều\s+[^.\n]+\.\s*(.+)$', re.MULTILINE)
RANK_QUALIFIERS = ('chính', 'trung cấp', 'cao cấp')
AMENDMENT_MARKERS = ('sua-doi', 'bo-sung', 'bai-bo', 'thay-the')
SOURCE_KEY_PATTERNS = (
    re.compile(r'(?:^|-)Thong-tu-(?:so-)?(\d+)-(\d{4})-TT-([A-Z0-9]+(?:-[A-Z0-9]+)*)'),
    re.compile(r'(?:^|-)Nghi-dinh-(\d+)-(\d{4})-ND-CP(?:-|$)'),
    re.compile(r'(?:^|-)Quyet-dinh-(\d+)-(?:\d{4}-)?QD-([A-Za-z0-9]+)'),
    re.compile(r'(?:^|-)Luat-(?:so-)?(\d+)-(\d{4})-QH(\d+)'),
)
AMENDED_PATH_RE = re.compile(
    r'\b(?:sửa đổi|bổ sung|bãi bỏ|thay thế)'
    r'(?:\s*,\s*bổ sung)?\s+'
    r'(?:điểm\s+(?P<point>[a-zđ])\s+)?'
    r'(?:khoản\s+(?P<clause>\d+)\s+)?'
    r'Điều\s+(?P<article>\d+[a-zđ]?)',
    re.IGNORECASE,
)


def normalize_spaces(text: str) -> str:
    return ' '.join(text.casefold().split())


def source_document_keys(source_name: str) -> list[str]:
    '''Return structured document identifiers in their order in a source slug.'''
    found: list[tuple[int, str]] = []
    for pattern in SOURCE_KEY_PATTERNS:
        for match in pattern.finditer(source_name):
            key = '-'.join(part.casefold() for part in match.groups())
            found.append((match.start(), key))
    found.sort(key=lambda item: item[0])
    return list(dict.fromkeys(key for _, key in found))


def extract_amended_paths(text: str) -> list[tuple[str, str | None, str | None]]:
    '''Extract target article, clause and point paths named by an amendment.'''
    normalized = ' '.join(text.split())
    paths: list[tuple[str, str | None, str | None]] = []
    for match in AMENDED_PATH_RE.finditer(normalized):
        path = (
            match.group('article').casefold(),
            match.group('clause'),
            match.group('point').casefold() if match.group('point') else None,
        )
        if path not in paths:
            paths.append(path)
    return paths


def parse_legal_path(path: str) -> tuple[str, str | None, str | None] | None:
    article = re.search(r'\bĐiều\s+(\d+[a-zđ]?)', path, re.IGNORECASE)
    if not article:
        return None
    clause = re.search(r'\bkhoản\s+(\d+)', path, re.IGNORECASE)
    point = re.search(r'\bđiểm\s+([a-zđ])', path, re.IGNORECASE)
    return (
        article.group(1).casefold(),
        clause.group(1) if clause else None,
        point.group(1).casefold() if point else None,
    )


def paths_overlap(
    node_path: tuple[str, str | None, str | None],
    amended_path: tuple[str, str | None, str | None],
) -> bool:
    '''Return true when a node is the changed provision or one of its ancestors/children.'''
    node_article, node_clause, node_point = node_path
    amended_article, amended_clause, amended_point = amended_path
    if node_article != amended_article:
        return False
    if node_clause is None or amended_clause is None:
        return True
    if node_clause != amended_clause:
        return False
    if node_point is None or amended_point is None:
        return True
    return node_point == amended_point


def prefer_amended_candidates(
    reranked: list[tuple[int, float]],
    node_map: dict[int, dict],
) -> list[tuple[int, float]]:
    '''Remove an original provision only when a selected amendment explicitly replaces it.'''
    amendments: list[
        tuple[set[str], list[tuple[str, str | None, str | None]]]
    ] = []
    for row_id, _ in reranked:
        node = node_map.get(int(row_id))
        if not node:
            continue
        source_name = str(node.get('source_name', ''))
        if not any(marker in source_name.casefold() for marker in AMENDMENT_MARKERS):
            continue
        document_keys = source_document_keys(source_name)
        if len(document_keys) < 2:
            continue
        target_paths = extract_amended_paths(
            '\n'.join(
                (
                    str(node.get('header_text', '')),
                    str(node.get('raw_text', '')),
                )
            )
        )
        if target_paths:
            amendments.append((set(document_keys[1:]), target_paths))

    if not amendments:
        return reranked

    preferred: list[tuple[int, float]] = []
    for row_id, score in reranked:
        node = node_map.get(int(row_id))
        if not node:
            preferred.append((row_id, score))
            continue
        document_keys = source_document_keys(str(node.get('source_name', '')))
        node_path = parse_legal_path(str(node.get('legal_path', '')))
        superseded = bool(document_keys and node_path) and any(
            document_keys[0] in target_keys
            and any(paths_overlap(node_path, target_path) for target_path in target_paths)
            for target_keys, target_paths in amendments
        )
        if not superseded:
            preferred.append((row_id, score))
    return preferred or reranked


def has_unrequested_rank_qualifier(question: str, node: dict) -> bool:
    '''Detect a sibling rank such as trung cấp when the question asks for the base rank.'''
    context = str(node.get('context_text', ''))
    article_match = ARTICLE_HEADING_RE.search(context)
    if not article_match:
        return False

    article_title = re.sub(r'\s*\(mã số.*$', '', article_match.group(1), flags=re.I)
    article_title = re.sub(r'^ngạch\s+', '', article_title, flags=re.I)
    normalized_title = normalize_spaces(article_title)
    normalized_question = normalize_spaces(question)

    for qualifier in RANK_QUALIFIERS:
        if not re.search(rf'\b{re.escape(qualifier)}\b', normalized_title):
            continue
        base_title = re.sub(
            rf'\b{re.escape(qualifier)}\b',
            '',
            normalized_title,
        )
        base_title = normalize_spaces(base_title)
        if base_title in normalized_question and normalized_title not in normalized_question:
            return True
    return False


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
    question: str = '',
) -> list[dict]:
    """Keep relevant fine-grained nodes, then restore original legal order."""
    if not reranked:
        return []

    reranked_node_map = fetch_nodes_map(conn, [row_id for row_id, _ in reranked])
    reranked = prefer_amended_candidates(reranked, reranked_node_map)

    if question:
        rank_matched = [
            (row_id, score)
            for row_id, score in reranked
            if row_id not in reranked_node_map
            or not has_unrequested_rank_qualifier(
                question,
                reranked_node_map[row_id],
            )
        ]
        if rank_matched:
            reranked = rank_matched

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
