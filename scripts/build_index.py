#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalqa.config import load_config, resolve_path
from legalqa.database import build_corpus_database
from legalqa.utils import configure_utf8_stdio, save_json, set_seed


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Build the hierarchical SQLite FTS5 legal index")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    set_seed(int(cfg["runtime"]["seed"]))

    context_source = resolve_path(project_root, cfg["paths"]["context_source"])
    db_path = resolve_path(project_root, cfg["paths"]["db_path"])
    run_dir = resolve_path(project_root, cfg["paths"]["run_dir"])
    assert context_source and db_path and run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[1/1] Building hierarchical SQLite database + FTS5...")
    stats = build_corpus_database(context_source, db_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    save_json(run_dir / "index_stats.json", stats)

    print("Done.")
    print(f"SQLite: {db_path}")
    print(f"Stats : {run_dir / 'index_stats.json'}")


if __name__ == "__main__":
    main()
