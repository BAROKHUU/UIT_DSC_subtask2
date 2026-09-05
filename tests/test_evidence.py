from legalqa import evidence as evidence_module


def test_relevant_children_are_not_promoted_to_parent(monkeypatch):
    nodes = {
        1: {
            "row_id": 1,
            "node_id": "parent",
            "parent_id": None,
            "node_type": "clause",
            "document_id": "doc",
            "start_offset": 0,
            "end_offset": 100,
            "depth": 20,
        },
        2: {
            "row_id": 2,
            "node_id": "child-a",
            "parent_id": "parent",
            "node_type": "point",
            "document_id": "doc",
            "start_offset": 10,
            "end_offset": 40,
            "depth": 30,
        },
        3: {
            "row_id": 3,
            "node_id": "child-b",
            "parent_id": "parent",
            "node_type": "point",
            "document_id": "doc",
            "start_offset": 40,
            "end_offset": 80,
            "depth": 30,
        },
    }

    def fake_fetch_nodes_map(conn, row_ids):
        return {row_id: nodes[row_id] for row_id in row_ids}

    monkeypatch.setattr(evidence_module, "fetch_nodes_map", fake_fetch_nodes_map)

    selected = evidence_module.select_evidence(
        conn=None,
        reranked=[(1, 0.95), (2, 0.90), (3, 0.85)],
        threshold=0.80,
        max_nodes=6,
    )

    assert [node["row_id"] for node in selected] == [2, 3]
    assert [node["rerank_score"] for node in selected] == [0.90, 0.85]
