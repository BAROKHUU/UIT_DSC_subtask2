#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from legalqa.config import load_config, resolve_path
from legalqa.evaluation import approximate_meteor, rouge_l
from legalqa.pipeline import LegalQAPipeline
from legalqa.utils import configure_utf8_stdio, load_json, save_json, set_seed


def normalize_train(data) -> list[dict]:
    rows: list[dict] = []
    if isinstance(data, dict):
        for qid, value in data.items():
            if isinstance(value, dict) and value.get("question") and value.get("answer"):
                rows.append({"id": str(qid), "question": value["question"], "answer": value["answer"]})
    elif isinstance(data, list):
        for i, value in enumerate(data):
            if isinstance(value, dict) and value.get("question") and value.get("answer"):
                rows.append({"id": str(value.get("id", i)), "question": value["question"], "answer": value["answer"]})
    return rows


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Tune reranker threshold on a train validation split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="Optional small validation limit for quick experiments")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--cache", default="runs/default/rerank_cache.json")
    parser.add_argument("--output", default="runs/default/threshold_results.json")
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help='Temporarily override the BM25 result count before RRF',
    )
    parser.add_argument('--dense-top-k', type=int, default=None)
    parser.add_argument('--fusion-top-k', type=int, default=None)
    parser.add_argument('--reranker-candidates', type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    if args.sparse_top_k is not None:
        if args.sparse_top_k < 1:
            parser.error("--sparse-top-k must be at least 1")
        cfg["retrieval"]["sparse_top_k"] = args.sparse_top_k
    for arg_name, config_name in (
        ('dense_top_k', 'dense_top_k'),
        ('fusion_top_k', 'fusion_top_k'),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            if value < 1:
                parser.error(f'--{arg_name.replace("_", "-")} must be at least 1')
            cfg['retrieval'][config_name] = value
    if args.reranker_candidates is not None:
        if args.reranker_candidates < 1:
            parser.error('--reranker-candidates must be at least 1')
        cfg.setdefault('hierarchy', {})['max_candidates'] = args.reranker_candidates
    seed = int(cfg["runtime"]["seed"])
    set_seed(seed)

    train_path = resolve_path(project_root, cfg["paths"]["train_path"])
    db_path = resolve_path(project_root, cfg["paths"]["db_path"])
    dense_index_path = resolve_path(project_root, cfg['paths'].get('dense_index_path'))
    assert train_path and db_path

    rows = normalize_train(load_json(train_path))
    if not rows:
        raise RuntimeError("No train examples with non-null answers were found.")

    rng = random.Random(seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:val_size]
    if args.limit is not None:
        val_rows = val_rows[: args.limit]

    cache_path = project_root / args.cache
    pipeline = LegalQAPipeline(cfg, db_path, dense_index_path=dense_index_path)
    try:
        cache_metadata = {
            'retrieval': cfg['retrieval'],
            'dense': cfg.get('dense', {}),
            'hierarchy': cfg.get('hierarchy', {}),
            'embedding_model': str(cfg['models']['embedding_model']),
            "reranker_model": str(cfg["models"]["reranker_model"]),
            "reranker_max_length": int(cfg["reranker"]["max_length"]),
        }
        cache = None
        if cache_path.exists():
            cache_payload = load_json(cache_path)
            if (
                isinstance(cache_payload, dict)
                and cache_payload.get("metadata") == cache_metadata
                and isinstance(cache_payload.get("items"), dict)
            ):
                cache = cache_payload["items"]
                print(f"Loaded compatible rerank cache: {cache_path}")
            else:
                print(f"Ignoring incompatible rerank cache: {cache_path}")

        if cache is None:
            cache = {}
            for item in tqdm(val_rows, desc="Retrieval + reranking cache"):
                stages = pipeline.retrieve_and_rerank(str(item["question"]))
                cache[item["id"]] = stages["reranked"]
            save_json(cache_path, {"metadata": cache_metadata, "items": cache})
            print(f"Saved rerank cache: {cache_path}")

        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
        results = []

        for tau in thresholds:
            meteor_scores = []
            rouge_scores = []
            used = 0
            for item in val_rows:
                reranked_raw = cache.get(item["id"])
                if reranked_raw is None:
                    continue
                reranked = [(int(x[0]), float(x[1])) for x in reranked_raw]
                prediction, _ = pipeline.build_answer_from_reranked(
                    str(item["question"]),
                    reranked,
                    threshold=tau,
                )
                gold = str(item["answer"])
                meteor_scores.append(approximate_meteor(gold, prediction))
                rouge_scores.append(rouge_l(gold, prediction))
                used += 1

            result = {
                "threshold": tau,
                "n": used,
                "approx_meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
                "rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

        results.sort(key=lambda x: (x["approx_meteor"], x["rouge_l"]), reverse=True)
        output_path = project_root / args.output
        save_json(output_path, results)
        print("\nBest configuration:")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
        print(f"Saved results to {output_path}")
        print("NOTE: approximate_meteor is only a local proxy. Use the organizer's official scorer if released.")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
