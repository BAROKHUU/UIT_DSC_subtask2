from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .evidence import select_evidence
from .generation import HFIntroGenerator, build_final_answer
from .reranker import VietnameseReranker, rerank_candidates
from .sparse import sparse_search
from .utils import choose_device


class LegalQAPipeline:
    def __init__(self, cfg: dict, db_path: str | Path) -> None:
        self.cfg = cfg
        self.device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
        trust_remote_code = bool(cfg.get("models", {}).get("trust_remote_code", True))

        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

        reranker_name = cfg["models"]["reranker_model"]
        self.reranker = VietnameseReranker(
            reranker_name,
            device=self.device,
            max_length=int(cfg["reranker"]["max_length"]),
            trust_remote_code=trust_remote_code,
        )

        self.intro_generator: Optional[HFIntroGenerator] = None
        use_llm_intro = bool(cfg.get("answer", {}).get("use_llm_intro", False))
        intro_model = str(cfg.get("models", {}).get("intro_llm_model", "")).strip()
        if use_llm_intro:
            if not intro_model:
                raise ValueError("answer.use_llm_intro=true but models.intro_llm_model is empty")
            self.intro_generator = HFIntroGenerator(
                intro_model,
                max_new_tokens=int(cfg["answer"].get("max_intro_new_tokens", 64)),
                trust_remote_code=trust_remote_code,
            )

    def close(self) -> None:
        self.conn.close()

    def retrieve_and_rerank(self, question: str) -> dict:
        r_cfg = self.cfg["retrieval"]
        rr_cfg = self.cfg["reranker"]
        candidate_top_k = int(r_cfg["sparse_top_k"])
        if candidate_top_k < 1:
            raise ValueError("retrieval.sparse_top_k must be at least 1")

        # BM25 is the only first-stage retriever. Its Top-K results are passed
        # directly to the reranker, so retrieval.sparse_top_k is exactly the
        # maximum reranker candidate-set size (or fewer if BM25 returns fewer).
        sparse_ids = sparse_search(
            self.conn,
            question,
            top_k=candidate_top_k,
        )
        reranked = rerank_candidates(
            self.conn,
            self.reranker,
            question,
            sparse_ids,
            batch_size=int(rr_cfg["batch_size"]),
        )

        return {
            "sparse_ids": sparse_ids,
            "reranked": reranked,
        }

    def build_answer_from_reranked(
        self,
        question: str,
        reranked: list[tuple[int, float]],
        threshold: float | None = None,
        use_configured_intro: bool = True,
    ) -> tuple[str, list[dict]]:
        threshold = (
            float(self.cfg["reranker"]["threshold"])
            if threshold is None
            else float(threshold)
        )
        evidence = select_evidence(
            self.conn,
            reranked,
            threshold=threshold,
            max_nodes=int(self.cfg["answer"]["max_evidence_nodes"]),
            parent_margin=float(self.cfg["answer"]["parent_rescue_margin"]),
        )
        intro_generator = self.intro_generator if use_configured_intro else None
        answer = build_final_answer(question, evidence, intro_generator=intro_generator)
        return answer, evidence

    def answer(self, question: str, threshold: float | None = None) -> tuple[str, dict]:
        stages = self.retrieve_and_rerank(question)
        answer, evidence = self.build_answer_from_reranked(
            question,
            stages["reranked"],
            threshold=threshold,
            use_configured_intro=True,
        )
        debug = {
            "question": question,
            "sparse_top": stages["sparse_ids"][:20],
            "reranker_candidate_count": len(stages["sparse_ids"]),
            "reranked_top": stages["reranked"][:30],
            "threshold": float(self.cfg["reranker"]["threshold"] if threshold is None else threshold),
            "evidence": [
                {
                    "row_id": int(node["row_id"]),
                    "document_id": str(node["document_id"]),
                    "source_name": node.get("source_name", ""),
                    "legal_path": node.get("legal_path", ""),
                    "score": float(node["rerank_score"]),
                    "text": node.get("raw_text", ""),
                }
                for node in evidence
            ],
            "answer": answer,
        }
        return answer, debug
