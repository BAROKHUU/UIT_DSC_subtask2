from legalqa import evidence as evidence_module


def test_amended_provision_replaces_original_even_with_a_lower_score(monkeypatch):
    nodes = {
        1: {
            'row_id': 1,
            'node_id': 'original-clause',
            'parent_id': 'original-article',
            'node_type': 'clause',
            'document_id': 'original-doc',
            'source_name': 'Thong-tu-12-2020-TT-BGTVT-quan-ly-van-tai',
            'legal_path': 'khoản 4 Điều 20',
            'header_text': '4. Quy định cũ.',
            'raw_text': '4. Quy định cũ.',
            'context_text': 'Điều 20. Quy định vận tải',
            'start_offset': 10,
            'end_offset': 20,
            'depth': 50,
        },
        2: {
            'row_id': 2,
            'node_id': 'amended-clause',
            'parent_id': 'amending-article',
            'node_type': 'clause',
            'document_id': 'amending-doc',
            'source_name': (
                'Thong-tu-17-2022-TT-BGTVT-sua-doi-'
                'Thong-tu-12-2020-TT-BGTVT-quan-ly-van-tai'
            ),
            'legal_path': 'khoản 2 Điều 1',
            'header_text': '2. Sửa đổi, bổ sung khoản 4 Điều 20 như sau:',
            'raw_text': '2. Sửa đổi, bổ sung khoản 4 Điều 20 như sau: Quy định mới.',
            'context_text': 'Điều 1. Sửa đổi, bổ sung một số điều',
            'start_offset': 10,
            'end_offset': 20,
            'depth': 50,
        },
    }

    def fake_fetch_nodes_map(conn, row_ids):
        return {row_id: nodes[row_id] for row_id in row_ids}

    monkeypatch.setattr(evidence_module, 'fetch_nodes_map', fake_fetch_nodes_map)

    selected = evidence_module.select_evidence(
        conn=None,
        reranked=[(1, 0.98), (2, 0.75)],
        threshold=0.90,
        max_nodes=6,
    )

    assert [node['row_id'] for node in selected] == [2]
    assert selected[0]['rerank_score'] == 0.75


def test_amendment_does_not_remove_an_unrelated_original_provision(monkeypatch):
    nodes = {
        1: {
            'row_id': 1,
            'node_id': 'unrelated-original-clause',
            'parent_id': 'original-article',
            'node_type': 'clause',
            'document_id': 'original-doc',
            'source_name': 'Thong-tu-12-2020-TT-BGTVT-quan-ly-van-tai',
            'legal_path': 'khoản 1 Điều 25',
            'header_text': '1. Nội dung vẫn còn hiệu lực.',
            'raw_text': '1. Nội dung vẫn còn hiệu lực.',
            'context_text': 'Điều 25. Lệnh vận chuyển',
            'start_offset': 10,
            'end_offset': 20,
            'depth': 50,
        },
        2: {
            'row_id': 2,
            'node_id': 'amended-clause',
            'parent_id': 'amending-article',
            'node_type': 'clause',
            'document_id': 'amending-doc',
            'source_name': (
                'Thong-tu-17-2022-TT-BGTVT-sua-doi-'
                'Thong-tu-12-2020-TT-BGTVT-quan-ly-van-tai'
            ),
            'legal_path': 'khoản 2 Điều 1',
            'header_text': '2. Sửa đổi, bổ sung khoản 4 Điều 20 như sau:',
            'raw_text': '2. Sửa đổi, bổ sung khoản 4 Điều 20 như sau: Quy định mới.',
            'context_text': 'Điều 1. Sửa đổi, bổ sung một số điều',
            'start_offset': 10,
            'end_offset': 20,
            'depth': 50,
        },
    }

    def fake_fetch_nodes_map(conn, row_ids):
        return {row_id: nodes[row_id] for row_id in row_ids}

    monkeypatch.setattr(evidence_module, 'fetch_nodes_map', fake_fetch_nodes_map)

    selected = evidence_module.select_evidence(
        conn=None,
        reranked=[(1, 0.98), (2, 0.95)],
        threshold=0.90,
        max_nodes=6,
    )

    assert sorted(node['row_id'] for node in selected) == [1, 2]


def test_unrequested_lower_rank_is_removed_when_base_rank_is_available(monkeypatch):
    nodes = {
        1: {
            'row_id': 1,
            'node_id': 'regular-rank',
            'parent_id': 'regular-clause',
            'node_type': 'point',
            'document_id': 'doc',
            'start_offset': 10,
            'end_offset': 20,
            'depth': 30,
            'context_text': (
                'Điều 26. Ngạch thuyền viên kiểm ngư (mã số: 25.313)\n'
                '3. Tiêu chuẩn về năng lực chuyên môn, nghiệp vụ'
            ),
        },
        2: {
            'row_id': 2,
            'node_id': 'intermediate-rank',
            'parent_id': 'intermediate-clause',
            'node_type': 'point',
            'document_id': 'doc',
            'start_offset': 30,
            'end_offset': 40,
            'depth': 30,
            'context_text': (
                'Điều 27. Ngạch thuyền viên kiểm ngư trung cấp (mã số: 25.314)\n'
                '3. Tiêu chuẩn về năng lực chuyên môn, nghiệp vụ'
            ),
        },
    }

    def fake_fetch_nodes_map(conn, row_ids):
        return {row_id: nodes[row_id] for row_id in row_ids}

    monkeypatch.setattr(evidence_module, 'fetch_nodes_map', fake_fetch_nodes_map)

    selected = evidence_module.select_evidence(
        conn=None,
        reranked=[(2, 0.95), (1, 0.90)],
        threshold=0.80,
        max_nodes=6,
        question=(
            'Công chức được bổ nhiệm vào ngạch thuyền viên kiểm ngư có tiêu chuẩn '
            'về năng lực chuyên môn, nghiệp vụ như thế nào?'
        ),
    )

    assert [node['row_id'] for node in selected] == [1]


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
