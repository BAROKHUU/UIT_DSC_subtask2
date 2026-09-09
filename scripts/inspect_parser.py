#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from legalqa.io import iter_context_documents
from legalqa.parser import parse_document
from legalqa.utils import configure_utf8_stdio


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Inspect parsed hierarchy for a few corpus documents")
    parser.add_argument("--source", required=True, help="selected-contexts.zip or extracted directory")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=80)
    args = parser.parse_args()

    for doc_idx, doc in enumerate(iter_context_documents(args.source)):
        if doc_idx >= args.limit:
            break
        nodes = parse_document(str(doc.get("id", doc_idx)), str(doc.get("name", "")), str(doc.get("passage", "")))
        print("=" * 100)
        print(f"DOCUMENT {doc.get('id')} | {doc.get('name')}")
        for node in nodes[: args.max_nodes]:
            indent = "  " * max(0, node.depth // 10 - 1)
            preview = " ".join(node.header_text.split())[:120]
            print(
                f"{indent}- {node.node_type:<16} label={node.label:<8} "
                f"sparse={int(node.is_sparse_indexable)} | {preview}"
            )


if __name__ == "__main__":
    main()
