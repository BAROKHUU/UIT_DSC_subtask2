from __future__ import annotations


def reciprocal_rank_fusion(
    sparse_ids: list[int],
    dense_ranked_ids: list[tuple[int, int]],
    rrf_k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse BM25 ranks and dense ranks without mixing incomparable raw scores."""
    if rrf_k < 1:
        raise ValueError('retrieval.rrf_k must be at least 1')
    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    order = 0

    for rank, raw_row_id in enumerate(sparse_ids, start=1):
        row_id = int(raw_row_id)
        if row_id in first_seen:
            continue
        first_seen[row_id] = order
        order += 1
        scores[row_id] = scores.get(row_id, 0.0) + float(sparse_weight) / (rrf_k + rank)

    seen_dense: set[int] = set()
    for raw_row_id, raw_rank in dense_ranked_ids:
        row_id = int(raw_row_id)
        rank = int(raw_rank)
        if row_id in seen_dense:
            continue
        seen_dense.add(row_id)
        if row_id not in first_seen:
            first_seen[row_id] = order
            order += 1
        scores[row_id] = scores.get(row_id, 0.0) + float(dense_weight) / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    if top_k is not None:
        if top_k < 1:
            raise ValueError('retrieval.fusion_top_k must be at least 1')
        ranked = ranked[:top_k]
    return ranked
