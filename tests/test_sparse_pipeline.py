from legalqa import pipeline as pipeline_module


def make_pipeline(sparse_top_k: int = 100, dense_enabled: bool = True):
    pipeline = pipeline_module.LegalQAPipeline.__new__(pipeline_module.LegalQAPipeline)
    pipeline.cfg = {
        'retrieval': {
            'sparse_top_k': sparse_top_k,
            'dense_top_k': 2,
            'fusion_top_k': 4,
            'rrf_k': 60,
            'sparse_weight': 1.0,
            'dense_weight': 1.0,
            'bm25_context_weight': 1.8,
            'bm25_raw_weight': 1.0,
        },
        'hierarchy': {'enabled': True},
        'reranker': {'batch_size': 16},
    }
    pipeline.conn = object()
    pipeline.reranker = object()
    pipeline.dense_retriever = object() if dense_enabled else None
    return pipeline


def test_hybrid_stages_feed_expanded_real_nodes_to_reranker(monkeypatch):
    pipeline = make_pipeline(sparse_top_k=3)
    sparse_ids = [11, 22, 33]
    dense_hits = [(5, 0.91), (7, 0.82)]
    resolved = [
        {'dense_id': 5, 'score': 0.91, 'unit_type': 'point_bundle', 'node_row_ids': [44, 55]},
        {'dense_id': 7, 'score': 0.82, 'unit_type': 'clause', 'node_row_ids': [66]},
    ]
    fused = [(11, 0.03), (44, 0.02), (55, 0.02), (66, 0.01)]
    expanded = [11, 44, 55, 66, 77]
    calls = {}

    def fake_sparse_search(conn, question, top_k, context_weight, raw_weight):
        calls['sparse'] = (conn, question, top_k, context_weight, raw_weight)
        return sparse_ids

    class FakeDense:
        def search(self, question, top_k):
            calls['dense'] = (question, top_k)
            return dense_hits

    def fake_rerank(conn, reranker, question, candidate_ids, batch_size):
        calls['reranker'] = (candidate_ids, batch_size)
        return [(77, 0.9), (55, 0.8)]

    pipeline.dense_retriever = FakeDense()
    monkeypatch.setattr(pipeline_module, 'sparse_search', fake_sparse_search)
    monkeypatch.setattr(pipeline_module, 'resolve_dense_hits', lambda conn, hits: resolved)
    monkeypatch.setattr(
        pipeline_module,
        'dense_ranked_node_ids',
        lambda hits: [(44, 1), (55, 1), (66, 2)],
    )
    monkeypatch.setattr(pipeline_module, 'reciprocal_rank_fusion', lambda *args, **kwargs: fused)
    monkeypatch.setattr(pipeline_module, 'expand_hierarchy', lambda conn, ids, cfg: expanded)
    monkeypatch.setattr(pipeline_module, 'rerank_candidates', fake_rerank)

    stages = pipeline.retrieve_and_rerank('cau hoi')

    assert calls['sparse'] == (pipeline.conn, 'cau hoi', 3, 1.8, 1.0)
    assert calls['dense'] == ('cau hoi', 2)
    assert calls['reranker'] == (expanded, 16)
    assert stages['dense_hits'] == resolved
    assert stages['fused'] == fused
    assert stages['expanded_ids'] == expanded
    assert stages['reranked'] == [(77, 0.9), (55, 0.8)]


def test_sparse_top_k_must_be_positive():
    pipeline = make_pipeline(sparse_top_k=0, dense_enabled=False)

    try:
        pipeline.retrieve_and_rerank('cau hoi')
    except ValueError as exc:
        assert str(exc) == 'retrieval.sparse_top_k must be at least 1'
    else:
        raise AssertionError('Expected invalid sparse_top_k to raise ValueError')
