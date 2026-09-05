from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LegalNode:
    node_id: str
    document_id: str
    source_name: str
    node_type: str
    label: str
    depth: int
    start_offset: int
    end_offset: int
    parent_id: Optional[str]
    order_index: int
    header_text: str = ""
    raw_text: str = ""
    context_text: str = ""
    retrieval_text: str = ""
    legal_path: str = ""
    is_indexable: bool = False
