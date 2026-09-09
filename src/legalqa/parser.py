from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .schema import LegalNode


PART_RE = re.compile(r"^PHẦN\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHƯƠNG\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^MỤC\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^TIỂU\s+MỤC\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^ĐIỀU\s+(\d+[A-ZĐ]?)\b", re.IGNORECASE)
EXPLICIT_CLAUSE_RE = re.compile(r"^KHOẢN\s+(\d+)\b", re.IGNORECASE)
EXPLICIT_POINT_RE = re.compile(r"^ĐIỂM\s+([a-zđ])\b", re.IGNORECASE)
NUMERIC_SECTION_RE = re.compile(r"^(\d+(?:\.\d+){1,5})\s+\S")
ROMAN_SECTION_RE = re.compile(r"^([IVXLCDM]{1,8})[.)]\s+\S")
NUMBERED_ITEM_RE = re.compile(r"^(\d+)[.)]\s+\S")
LETTER_ITEM_RE = re.compile(r"^([a-zđ])[.)]\s+\S", re.IGNORECASE)

SPARSE_INDEXABLE_TYPES = {
    "part",
    "chapter",
    "section",
    "article",
    "clause",
    "point",
    "numeric_section",
    "roman_section",
    "numbered_item",
    "letter_item",
    "document_body",
}


def prepare_passage(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    return text


def normalize_line_for_detection(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def classify_structural_line(
    line: str,
    allow_numbered_child: bool,
) -> Optional[Tuple[str, str, int]]:
    s = normalize_line_for_detection(line)
    if not s or len(s) > 600:
        return None

    m = PART_RE.match(s)
    if m:
        return "part", m.group(1), 10

    m = CHAPTER_RE.match(s)
    if m:
        return "chapter", m.group(1), 20

    m = SECTION_RE.match(s)
    if m:
        return "section", m.group(1), 30

    m = SUBSECTION_RE.match(s)
    if m:
        label = m.group(1)
        return "numeric_section", label, 35 + 5 * label.count(".")

    m = ARTICLE_RE.match(s)
    if m:
        return "article", m.group(1), 40

    m = EXPLICIT_CLAUSE_RE.match(s)
    if m:
        return "numbered_item", m.group(1), 50

    m = EXPLICIT_POINT_RE.match(s)
    if m:
        return "letter_item", m.group(1).lower(), 60

    # Technical standards: 3.2, 3.2.1, 3.2.1.1, ...
    m = NUMERIC_SECTION_RE.match(s)
    if m:
        label = m.group(1)
        return "numeric_section", label, 35 + 5 * label.count(".")

    # I. INTRODUCTION / II. REQUIREMENTS ...
    m = ROMAN_SECTION_RE.match(s)
    if m:
        return "roman_section", m.group(1), 35

    if allow_numbered_child:
        m = NUMBERED_ITEM_RE.match(s)
        if m:
            return "numbered_item", m.group(1), 50

        m = LETTER_ITEM_RE.match(s)
        if m:
            return "letter_item", m.group(1).lower(), 60

    return None


def build_legal_path(node: LegalNode, node_by_id: Dict[str, LegalNode]) -> str:
    chain = [node]
    cur = node
    while cur.parent_id and cur.parent_id in node_by_id:
        cur = node_by_id[cur.parent_id]
        chain.append(cur)
    chain.reverse()

    article = clause = point = numeric = None
    for n in chain:
        if n.node_type == "article":
            article = n.label
        elif n.node_type == "clause":
            clause = n.label
        elif n.node_type == "point":
            point = n.label
        elif n.node_type == "numeric_section":
            numeric = n.label

    if point and clause and article:
        return f"điểm {point} khoản {clause} Điều {article}"
    if clause and article:
        return f"khoản {clause} Điều {article}"
    if article:
        return f"Điều {article}"
    if numeric:
        return f"mục {numeric}"
    return node.header_text[:150]


def _make_document_fallback(document_id: str, source_name: str, text: str) -> List[LegalNode]:
    raw = text.strip()
    if not raw:
        return []
    node = LegalNode(
        node_id=f"{document_id}:0:document",
        document_id=str(document_id),
        source_name=source_name,
        node_type="document_body",
        label="document",
        depth=0,
        start_offset=0,
        end_offset=len(text),
        parent_id=None,
        order_index=0,
        header_text=source_name,
        raw_text=raw,
        context_text=f"Văn bản: {re.sub(r'[-_]+', ' ', source_name)}",
        retrieval_text=f"Văn bản: {re.sub(r'[-_]+', ' ', source_name)}\n\nNội dung:\n{raw}",
        legal_path=source_name,
        is_sparse_indexable=True,
    )
    return [node]


def parse_document(document_id: str, source_name: str, passage: str) -> List[LegalNode]:
    """Parse one legal passage into a flexible hierarchy.

    The parser is intentionally heuristic because the corpus mixes laws,
    decrees, decisions, standards, appendices and technical procedures.
    Offsets are preserved so evidence can later be reconstructed in original order.
    """
    text = prepare_passage(passage)

    line_infos: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        clean_line = line.rstrip("\n")
        line_infos.append((offset, clean_line))
        offset += len(line)

    nodes: List[LegalNode] = []
    stack: List[LegalNode] = []
    structural_parent_active = False

    for start_offset, line in line_infos:
        detected = classify_structural_line(line, allow_numbered_child=structural_parent_active)
        if detected is None:
            continue

        node_type, label, depth = detected
        if node_type in {
            "part",
            "chapter",
            "section",
            "article",
            "numeric_section",
            "roman_section",
        }:
            structural_parent_active = True

        while stack and stack[-1].depth >= depth:
            stack.pop()

        parent = stack[-1] if stack else None
        node = LegalNode(
            node_id=f"{document_id}:{start_offset}",
            document_id=str(document_id),
            source_name=source_name,
            node_type=node_type,
            label=str(label),
            depth=depth,
            start_offset=start_offset,
            end_offset=-1,
            parent_id=parent.node_id if parent else None,
            order_index=len(nodes),
        )
        nodes.append(node)
        stack.append(node)

    if not nodes:
        return _make_document_fallback(str(document_id), source_name, text)

    # A node ends when the next node of equal or shallower depth starts.
    for i, node in enumerate(nodes):
        node_end = len(text)
        for next_node in nodes[i + 1 :]:
            if next_node.depth <= node.depth:
                node_end = next_node.start_offset
                break
        node.end_offset = node_end
        node.raw_text = text[node.start_offset : node.end_offset].strip()

    node_by_id = {n.node_id: n for n in nodes}
    children_map: dict[str, list[LegalNode]] = defaultdict(list)
    for node in nodes:
        if node.parent_id:
            children_map[node.parent_id].append(node)

    def ancestor_types(node: LegalNode) -> list[str]:
        types: list[str] = []
        cur = node
        while cur.parent_id and cur.parent_id in node_by_id:
            cur = node_by_id[cur.parent_id]
            types.append(cur.node_type)
        return types

    # Infer legal semantics from generic numbered / letter children.
    for node in nodes:
        ancestors = ancestor_types(node)
        if node.node_type == "numbered_item" and "article" in ancestors:
            node.node_type = "clause"
        elif node.node_type == "letter_item" and "article" in ancestors:
            node.node_type = "point"

    # Header = the current node's prefix before its first structural child.
    for node in nodes:
        children = children_map.get(node.node_id, [])
        first_child_start = min((c.start_offset for c in children), default=node.end_offset)
        own_prefix = text[node.start_offset:first_child_start]
        own_prefix = " ".join(own_prefix.split())
        node.header_text = own_prefix[:400]

    clean_source_name = re.sub(r"[-_]+", " ", source_name)
    for node in nodes:
        ancestors: list[LegalNode] = []
        cur = node
        while cur.parent_id and cur.parent_id in node_by_id:
            parent = node_by_id[cur.parent_id]
            ancestors.append(parent)
            cur = parent
        ancestors.reverse()

        context_parts = [f"Văn bản: {clean_source_name}"]
        for ancestor in ancestors:
            if ancestor.header_text:
                context_parts.append(ancestor.header_text)
        if node.header_text:
            context_parts.append(node.header_text)

        node.context_text = "\n".join(context_parts)
        node.retrieval_text = node.context_text + "\n\nNội dung:\n" + node.raw_text
        node.legal_path = build_legal_path(node, node_by_id)
        node.is_sparse_indexable = node.node_type in SPARSE_INDEXABLE_TYPES

    return nodes
