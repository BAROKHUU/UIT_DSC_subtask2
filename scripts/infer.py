#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from legalqa.config import load_config, resolve_path
from legalqa.pipeline import LegalQAPipeline
from legalqa.utils import append_jsonl, configure_utf8_stdio, load_json, save_json, set_seed


def normalize_questions(data) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for qid, value in data.items():
            if isinstance(value, dict) and "question" in value:
                items.append((str(qid), str(value["question"])))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            if isinstance(value, dict) and "question" in value:
                qid = str(value.get("id", i))
                items.append((qid, str(value["question"])))
    return items


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run LegalQA inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", default=None, help="Single question")
    parser.add_argument("--input", default=None, help="JSON file such as public-official.json")
    parser.add_argument("--output", default=None, help="Output predictions JSON")
    parser.add_argument("--debug-jsonl", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume batch inference from an existing output JSON file",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save batch predictions after every N new answers (0 disables checkpoints)",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help='Temporarily override the BM25 result count before RRF',
    )
    parser.add_argument('--dense-top-k', type=int, default=None, help='Override FAISS Top-K before bundle expansion and RRF')
    parser.add_argument('--fusion-top-k', type=int, default=None, help='Override the number of RRF seeds')
    parser.add_argument(
        '--reranker-candidates',
        type=int,
        default=None,
        help='Override hierarchy.max_candidates (hard upper bound sent to reranker)',
    )
    parser.add_argument('--disable-dense', action='store_true', help='Run BM25-only without loading FAISS/embedding model')
    args = parser.parse_args()

    if not args.question and not args.input:
        parser.error("Provide --question or --input")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be at least 0")

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    if args.sparse_top_k is not None:
        if args.sparse_top_k < 1:
            parser.error("--sparse-top-k must be at least 1")
        cfg["retrieval"]["sparse_top_k"] = args.sparse_top_k
    if args.dense_top_k is not None:
        if args.dense_top_k < 1:
            parser.error('--dense-top-k must be at least 1')
        cfg['retrieval']['dense_top_k'] = args.dense_top_k
    if args.fusion_top_k is not None:
        if args.fusion_top_k < 1:
            parser.error('--fusion-top-k must be at least 1')
        cfg['retrieval']['fusion_top_k'] = args.fusion_top_k
    if args.reranker_candidates is not None:
        if args.reranker_candidates < 1:
            parser.error('--reranker-candidates must be at least 1')
        cfg.setdefault('hierarchy', {})['max_candidates'] = args.reranker_candidates
    if args.disable_dense:
        cfg.setdefault('dense', {})['enabled'] = False
    set_seed(int(cfg["runtime"]["seed"]))

    db_path = resolve_path(project_root, cfg["paths"]["db_path"])
    dense_index_path = resolve_path(project_root, cfg['paths'].get('dense_index_path'))
    assert db_path

    pipeline = LegalQAPipeline(cfg, db_path, dense_index_path=dense_index_path)
    try:
        if args.question:
            answer, debug = pipeline.answer(args.question, threshold=args.threshold)
            print(answer)
            if args.debug_jsonl:
                append_jsonl(project_root / args.debug_jsonl, debug)
            return

        input_path = project_root / args.input
        data = load_json(input_path)
        questions = normalize_questions(data)
        output_path = project_root / (args.output or "runs/default/predictions.json")
        predictions: dict = {}
        if args.resume and output_path.exists():
            existing = load_json(output_path)
            if not isinstance(existing, dict):
                raise ValueError(f"Cannot resume: {output_path} is not a JSON object")
            predictions.update(existing)

        pending_questions = [item for item in questions if item[0] not in predictions]
        if predictions:
            print(
                f"Resuming with {len(predictions)} completed; "
                f"{len(pending_questions)} questions remaining."
            )

        debug_path = project_root / args.debug_jsonl if args.debug_jsonl else None
        completed_since_start = 0
        try:
            for qid, question in tqdm(pending_questions, desc="Inference"):
                answer, debug = pipeline.answer(question, threshold=args.threshold)
                predictions[qid] = {"question": question, "answer": answer}
                completed_since_start += 1
                if debug_path:
                    debug["id"] = qid
                    append_jsonl(debug_path, debug)
                if (
                    args.checkpoint_every > 0
                    and completed_since_start % args.checkpoint_every == 0
                ):
                    save_json(output_path, predictions)
        except KeyboardInterrupt:
            save_json(output_path, predictions)
            print(f"Interrupted; saved {len(predictions)} predictions to {output_path}")
            raise

        save_json(output_path, predictions)
        print(f"Saved {len(predictions)} predictions to {output_path}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
