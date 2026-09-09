from __future__ import annotations

import sqlite3
from pathlib import Path

from .dense import (
    DenseRetriever,
    dense_mapping_signature,
    dense_ranked_node_ids,
    resolve_dense_hits,
)
from .evidence import select_evidence
from .fusion import reciprocal_rank_fusion
from .generation import build_final_answer
from .hierarchy import expand_hierarchy
from .reranker import VietnameseReranker, rerank_candidates
from .sparse import sparse_search
from .utils import choose_device


class LegalQAPipeline:
    def __init__(
        self,
        cfg: dict,
        db_path: str | Path,
        dense_index_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
        trust_remote_code = bool(cfg.get("models", {}).get("trust_remote_code", True))

        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

        dense_cfg = cfg.get('dense', {})
        self.dense_retriever = None
        if bool(dense_cfg.get('enabled', True)):
            if dense_index_path is None:
                configured_path = cfg.get('paths', {}).get('dense_index_path')
                if not configured_path:
                    raise ValueError('paths.dense_index_path is required when dense.enabled is true')
                dense_index_path = Path(configured_path)
            dense_device = choose_device(str(dense_cfg.get('device', self.device)))
            self.dense_retriever = DenseRetriever(
                dense_index_path,
                model_name=str(cfg['models']['embedding_model']),
                device=dense_device,
                dense_cfg=dense_cfg,
                trust_remote_code=trust_remote_code,
            )
            try:
                dense_unit_count = int(
                    self.conn.execute('SELECT count(*) FROM dense_units').fetchone()[0]
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    'SQLite database has no dense_units mapping. Rebuild it with '
                    'python scripts/build_index.py.'
                ) from exc
            if dense_unit_count != int(self.dense_retriever.index.ntotal):
                raise RuntimeError(
                    'SQLite dense_units and FAISS index have different sizes. '
                    'Rebuild both with python scripts/build_index.py.'
                )
            expected_signature = self.dense_retriever.manifest.get('mapping_signature')
            if expected_signature != dense_mapping_signature(self.conn):
                raise RuntimeError(
                    'SQLite dense_units mapping does not belong to this FAISS index. '
                    'Rebuild both with python scripts/build_index.py.'
                )

        reranker_name = cfg["models"]["reranker_model"]
        self.reranker = VietnameseReranker(
            reranker_name,
            device=self.device,
            max_length=int(cfg["reranker"]["max_length"]),
            trust_remote_code=trust_remote_code,
        )

    def close(self) -> None:
        self.conn.close()

    def retrieve_and_rerank(self, question: str) -> dict:
        r_cfg = self.cfg["retrieval"]
        rr_cfg = self.cfg["reranker"]
        sparse_top_k = int(r_cfg["sparse_top_k"])
        dense_top_k = int(r_cfg.get('dense_top_k', 100))
        fusion_top_k = int(r_cfg.get('fusion_top_k', 120))
        if sparse_top_k < 1:
            raise ValueError("retrieval.sparse_top_k must be at least 1")
        if self.dense_retriever is not None and dense_top_k < 1:
            raise ValueError('retrieval.dense_top_k must be at least 1')
        if fusion_top_k < 1:
            raise ValueError('retrieval.fusion_top_k must be at least 1')

        sparse_ids = sparse_search(
            self.conn,
            question,
            top_k=sparse_top_k,
            context_weight=float(r_cfg.get('bm25_context_weight', 1.8)),
            raw_weight=float(r_cfg.get('bm25_raw_weight', 1.0)),
        )
        dense_hits = (
            self.dense_retriever.search(question, top_k=dense_top_k)
            if self.dense_retriever is not None
            else []
        )
        resolved_dense_hits = resolve_dense_hits(self.conn, dense_hits)
        dense_ranked_ids = dense_ranked_node_ids(resolved_dense_hits)
        fused = reciprocal_rank_fusion(
            sparse_ids,
            dense_ranked_ids,
            rrf_k=int(r_cfg.get('rrf_k', 60)),
            sparse_weight=float(r_cfg.get('sparse_weight', 1.0)),
            dense_weight=float(r_cfg.get('dense_weight', 1.0)),
            top_k=fusion_top_k,
        )
        fused_ids = [row_id for row_id, _ in fused]
        expanded_ids = expand_hierarchy(
            self.conn,
            fused_ids,
            self.cfg.get('hierarchy', {}),
        )
        reranked = rerank_candidates(
            self.conn,
            self.reranker,
            question,
            expanded_ids,
            batch_size=int(rr_cfg["batch_size"]),
        )

        return {
            "sparse_ids": sparse_ids,
            'dense_hits': resolved_dense_hits,
            'dense_ranked_ids': dense_ranked_ids,
            'fused': fused,
            'expanded_ids': expanded_ids,
            "reranked": reranked,
        }

    def build_answer_from_reranked(
        self,
        question: str,
        reranked: list[tuple[int, float]],
        threshold: float | None = None,
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
            question=question,
            max_nodes=int(self.cfg["answer"]["max_evidence_nodes"]),
        )
        answer = build_final_answer(question, evidence)
        return answer, evidence

    def answer(self, question: str, threshold: float | None = None) -> tuple[str, dict]:
        stages = self.retrieve_and_rerank(question)
        answer, evidence = self.build_answer_from_reranked(
            question,
            stages["reranked"],
            threshold=threshold,
        )
        debug = {
            "question": question,
            "sparse_top": stages["sparse_ids"][:20],
            'dense_top': stages['dense_hits'][:20],
            'fused_top': stages['fused'][:30],
            'expanded_top': stages['expanded_ids'][:50],
            "reranker_candidate_count": len(stages['expanded_ids']),
            "reranked_top": stages["reranked"][:30],
            "threshold": float(self.cfg["reranker"]["threshold"] if threshold is None else threshold),
            "evidence": [
                {
                    "row_id": int(node["row_id"]),
                    "document_id": str(node["document_id"]),
                    "source_name": node.get("source_name", ""),
                    "node_type": node.get("node_type", ""),
                    "parent_id": node.get("parent_id"),
                    "legal_path": node.get("legal_path", ""),
                    "score": float(node["rerank_score"]),
                    "text": node.get("raw_text", ""),
                }
                for node in evidence
            ],
            "answer": answer,
        }
        return answer, debug
