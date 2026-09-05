from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_model_source(model_name: str) -> str:
    """Prefer a cached snapshot to avoid unnecessary Hub metadata requests."""
    if Path(model_name).exists():
        return model_name

    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except LocalEntryNotFoundError:
        return model_name


class VietnameseReranker:
    """Generic Hugging Face sequence-classification reranker wrapper.

    Some reranker repositories use custom code or different score semantics.
    If your chosen model documents a custom scoring API, adapt only this class;
    the rest of the project can remain unchanged.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        trust_remote_code: bool = True,
    ) -> None:
        self.device = device
        self.max_length = max_length
        model_source = resolve_model_source(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Could not load the reranker tokenizer. Install the project "
                "dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        all_scores: list[float] = []

        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits

            if logits.shape[-1] == 1:
                scores = torch.sigmoid(logits[:, 0])
            else:
                scores = torch.softmax(logits, dim=-1)[:, -1]

            all_scores.extend(scores.detach().cpu().float().tolist())

        return all_scores


def rerank_candidates(
    conn: sqlite3.Connection,
    reranker: VietnameseReranker,
    question: str,
    candidate_ids: list[int],
    batch_size: int = 32,
) -> list[tuple[int, float]]:
    if not candidate_ids:
        return []

    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT row_id, retrieval_text FROM nodes WHERE row_id IN ({placeholders})",
        candidate_ids,
    ).fetchall()
    text_map = {int(row[0]): str(row[1]) for row in rows}
    valid_ids = [row_id for row_id in candidate_ids if row_id in text_map]
    passages = [text_map[row_id] for row_id in valid_ids]

    scores = reranker.score(question, passages, batch_size=batch_size)
    ranked = list(zip(valid_ids, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
