import sqlite3

import json

import numpy as np

from legalqa import dense as dense_module
from legalqa.dense import (
    build_dense_index,
    build_contextual_dense_text,
    dense_ranked_node_ids,
    iter_semantic_units,
    resolve_dense_hits,
)


def make_conn():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute(
        '''
        CREATE TABLE nodes (
            row_id INTEGER PRIMARY KEY,
            node_id TEXT,
            parent_id TEXT,
            node_type TEXT,
            source_name TEXT,
            context_text TEXT,
            raw_text TEXT,
            start_offset INTEGER
        )
        '''
    )
    conn.execute(
        '''
        CREATE TABLE dense_units (
            dense_id INTEGER PRIMARY KEY,
            unit_type TEXT,
            node_id TEXT,
            parent_id TEXT,
            member_node_ids TEXT
        )
        '''
    )
    return conn


def add_node(conn, row_id, node_id, parent_id, node_type, raw_text, context_text, start):
    conn.execute(
        'INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (row_id, node_id, parent_id, node_type, 'thong-tu-01', context_text, raw_text, start),
    )


def test_semantic_leaf_policy_and_short_point_bundle():
    conn = make_conn()
    article_context = 'Văn bản: thong tu 01\nĐiều 1. Tiêu chuẩn'
    clause_context = article_context + '\n1. Năng lực chuyên môn'
    add_node(conn, 1, 'article-1', None, 'article', 'Điều 1 gồm nhiều khoản', article_context, 0)
    add_node(conn, 2, 'clause-1', 'article-1', 'clause', '1. Có các điểm', clause_context, 10)
    add_node(conn, 3, 'point-a', 'clause-1', 'point', 'a) Nội dung ngắn.', clause_context + '\na)', 20)
    add_node(conn, 4, 'point-b', 'clause-1', 'point', 'b) Nội dung ngắn khác.', clause_context + '\nb)', 30)
    long_point = 'c) ' + ('x' * 177)
    add_node(conn, 5, 'point-c', 'clause-1', 'point', long_point, clause_context + '\nc)', 40)
    add_node(conn, 6, 'clause-2', 'article-1', 'clause', '2. Khoản lá độc lập.', article_context + '\n2. Khoản lá độc lập.', 50)
    add_node(conn, 7, 'numbered-1', None, 'numbered_item', '1) Không dense.', '', 60)
    add_node(conn, 8, 'article-leaf', None, 'article', 'Điều 2. Điều độc lập.', 'Điều 2. Điều độc lập.', 70)
    add_node(conn, 9, 'numeric-leaf', None, 'numeric_section', '3.2 Yêu cầu.', '', 80)
    add_node(conn, 10, 'roman-leaf', None, 'roman_section', 'II. Quy định.', '', 90)
    add_node(conn, 11, 'part-leaf', None, 'part', 'PHẦN I', '', 100)
    add_node(conn, 12, 'chapter-leaf', None, 'chapter', 'CHƯƠNG I', '', 110)
    add_node(conn, 13, 'section-leaf', None, 'section', 'MỤC I', '', 120)
    add_node(conn, 14, 'letter-leaf', None, 'letter_item', 'a) Không dense.', '', 130)

    units = list(iter_semantic_units(conn, point_min_chars=180))
    direct_ids = {unit.node_id for unit in units if unit.unit_type != 'point_bundle'}
    bundle = next(unit for unit in units if unit.unit_type == 'point_bundle')

    assert direct_ids == {
        'point-c',
        'clause-2',
        'article-leaf',
        'numeric-leaf',
        'roman-leaf',
    }
    assert bundle.parent_id == 'clause-1'
    assert bundle.member_node_ids == ('point-a', 'point-b')
    assert 'a) Nội dung ngắn.' in bundle.dense_text
    assert 'b) Nội dung ngắn khác.' in bundle.dense_text
    assert 'numbered-1' not in direct_ids


def test_contextual_text_uses_headings_but_not_ancestor_body():
    text = build_contextual_dense_text(
        'thong-tu-01',
        'Văn bản: thong tu 01\nĐiều 8. Tiêu chuẩn\n3. Về nghiệp vụ',
        'a) Có năng lực thực hiện nhiệm vụ.',
    )

    assert 'Văn bản: thong tu 01' in text
    assert 'Điều: Điều 8. Tiêu chuẩn' in text
    assert 'Khoản: 3. Về nghiệp vụ' in text
    assert text.endswith('a) Có năng lực thực hiện nhiệm vụ.')
    assert 'toàn bộ nội dung điều' not in text


def test_bundle_hit_expands_to_real_nodes_before_rrf():
    conn = make_conn()
    context = 'Văn bản: thong tu 01\nĐiều 1\n1. Khoản'
    add_node(conn, 1, 'point-a', 'clause-1', 'point', 'a) A', context, 10)
    add_node(conn, 2, 'point-b', 'clause-1', 'point', 'b) B', context, 20)
    conn.execute(
        'INSERT INTO dense_units VALUES (?, ?, ?, ?, ?)',
        (0, 'point_bundle', None, 'clause-1', json.dumps(['point-a', 'point-b'])),
    )

    resolved = resolve_dense_hits(conn, [(0, 0.9)])
    ranked = dense_ranked_node_ids(resolved)

    assert resolved[0]['unit_type'] == 'point_bundle'
    assert resolved[0]['node_row_ids'] == [1, 2]
    assert ranked == [(1, 1), (2, 1)]


def test_faiss_build_persists_unit_mapping_and_manifest(monkeypatch, tmp_path):
    conn = make_conn()
    add_node(conn, 1, 'article-leaf', None, 'article', 'Điều 1. Nội dung.', 'Điều 1. Nội dung.', 0)

    class FakeEmbedder:
        def __init__(self, **kwargs):
            pass

        def encode(self, texts, is_query=False):
            vectors = np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(dense_module, 'SemanticEmbedder', FakeEmbedder)
    index_path = tmp_path / 'semantic.faiss'
    stats = build_dense_index(
        conn,
        index_path,
        model_name='fake/model',
        device='cpu',
        dense_cfg={'batch_size': 2, 'point_min_chars': 180, 'max_length': 64},
    )

    mapping = conn.execute('SELECT * FROM dense_units').fetchone()
    assert index_path.exists()
    assert (tmp_path / 'semantic.faiss.meta.json').exists()
    assert stats['units'] == 1
    assert mapping['unit_type'] == 'article'
    assert json.loads(mapping['member_node_ids']) == ['article-leaf']
    assert stats['mapping_signature']
