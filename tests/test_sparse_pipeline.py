from legalqa import pipeline as pipeline_module


def make_pipeline(sparse_top_k: int = 100):
    pipeline = pipeline_module.LegalQAPipeline.__new__(pipeline_module.LegalQAPipeline)
    pipeline.cfg = {
        "retrieval": {"sparse_top_k": sparse_top_k},
        "reranker": {"batch_size": 16},
    }
    pipeline.conn = object()
    pipeline.reranker = object()
    return pipeline


def test_sparse_results_are_fed_directly_to_reranker(monkeypatch):
    pipeline = make_pipeline(sparse_top_k=3)
    sparse_ids = [11, 22, 33]
    calls = {}

    def fake_sparse_search(conn, question, top_k):
        calls["sparse"] = (conn, question, top_k)
        return sparse_ids

    def fake_rerank_candidates(conn, reranker, question, candidate_ids, batch_size):
        calls["reranker"] = (conn, reranker, question, candidate_ids, batch_size)
        return [(33, 0.9), (11, 0.8), (22, 0.7)]

    monkeypatch.setattr(pipeline_module, "sparse_search", fake_sparse_search)
    monkeypatch.setattr(pipeline_module, "rerank_candidates", fake_rerank_candidates)

    stages = pipeline.retrieve_and_rerank("cau hoi")

    assert calls["sparse"] == (pipeline.conn, "cau hoi", 3)
    assert calls["reranker"] == (
        pipeline.conn,
        pipeline.reranker,
        "cau hoi",
        sparse_ids,
        16,
    )
    assert stages == {
        "sparse_ids": sparse_ids,
        "reranked": [(33, 0.9), (11, 0.8), (22, 0.7)],
    }


def test_sparse_top_k_must_be_positive():
    pipeline = make_pipeline(sparse_top_k=0)

    try:
        pipeline.retrieve_and_rerank("cau hoi")
    except ValueError as exc:
        assert str(exc) == "retrieval.sparse_top_k must be at least 1"
    else:
        raise AssertionError("Expected invalid sparse_top_k to raise ValueError")
