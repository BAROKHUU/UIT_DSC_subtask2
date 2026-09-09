#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalqa.config import load_config, resolve_path
from legalqa.database import build_corpus_database, connect_database
from legalqa.dense import build_dense_index
from legalqa.utils import choose_device, configure_utf8_stdio, save_json, set_seed


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description='Build SQLite/FTS5 and the contextual semantic-leaf FAISS index'
    )
    parser.add_argument("--config", default="configs/default.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--dense-only',
        action='store_true',
        help='Reuse the existing SQLite database and rebuild only FAISS + dense_units',
    )
    mode.add_argument(
        '--skip-dense',
        action='store_true',
        help='Build only SQLite hierarchy + FTS5',
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    set_seed(int(cfg["runtime"]["seed"]))

    context_source = resolve_path(project_root, cfg["paths"]["context_source"])
    db_path = resolve_path(project_root, cfg["paths"]["db_path"])
    dense_index_path = resolve_path(project_root, cfg['paths'].get('dense_index_path'))
    run_dir = resolve_path(project_root, cfg["paths"]["run_dir"])
    assert context_source and db_path and run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    dense_enabled = bool(cfg.get('dense', {}).get('enabled', True)) and not args.skip_dense
    if dense_enabled and dense_index_path is None:
        parser.error('paths.dense_index_path is required when dense indexing is enabled')
    stats: dict = {}

    if not args.dense_only:
        print('[1/2] Building full SQLite hierarchy + near-full-detail FTS5...')
        stats['sqlite'] = build_corpus_database(context_source, db_path)
        print(json.dumps(stats['sqlite'], ensure_ascii=False, indent=2))
    elif not db_path.exists():
        parser.error(f'Cannot use --dense-only; SQLite database does not exist: {db_path}')

    if dense_enabled:
        print('[2/2] Building contextual semantic-leaf FAISS index...')
        conn = connect_database(db_path)
        try:
            stats['dense'] = build_dense_index(
                conn,
                dense_index_path,
                model_name=str(cfg['models']['embedding_model']),
                device=choose_device(str(cfg.get('dense', {}).get('device', 'auto'))),
                dense_cfg=cfg.get('dense', {}),
                trust_remote_code=bool(cfg.get('models', {}).get('trust_remote_code', True)),
            )
        finally:
            conn.close()
        print(json.dumps(stats['dense'], ensure_ascii=False, indent=2))
    else:
        print('[2/2] Dense build skipped.')

    save_json(run_dir / "index_stats.json", stats)

    print("Done.")
    print(f"SQLite: {db_path}")
    if dense_enabled:
        print(f'FAISS : {dense_index_path}')
    print(f"Stats : {run_dir / 'index_stats.json'}")


if __name__ == "__main__":
    main()
