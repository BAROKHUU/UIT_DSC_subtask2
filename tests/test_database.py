from legalqa.database import _insert_node, create_database
from legalqa.schema import LegalNode


def make_node(node_id: str, node_type: str, header: str, raw: str) -> LegalNode:
    return LegalNode(
        node_id=node_id,
        document_id='doc',
        source_name='source',
        node_type=node_type,
        label='1',
        depth=1,
        start_offset=0,
        end_offset=len(raw),
        parent_id=None,
        order_index=0,
        header_text=header,
        raw_text=raw,
        context_text='Văn bản: source\n' + header,
        retrieval_text=raw,
        is_sparse_indexable=True,
    )


def test_structural_container_is_in_fts_without_duplicating_descendant_body(tmp_path):
    conn = create_database(tmp_path / 'legal.sqlite')
    chapter = make_node('chapter', 'chapter', 'CHƯƠNG I', 'CHƯƠNG I\nbody của mọi điều con')
    article = make_node('article', 'article', 'Điều 1', 'Điều 1\nbody chi tiết')

    chapter_row_id = _insert_node(conn, chapter)
    article_row_id = _insert_node(conn, article)

    chapter_fts = conn.execute(
        'SELECT raw_text FROM node_fts WHERE rowid = ?', (chapter_row_id,)
    ).fetchone()[0]
    article_fts = conn.execute(
        'SELECT raw_text FROM node_fts WHERE rowid = ?', (article_row_id,)
    ).fetchone()[0]

    assert chapter_fts == 'CHƯƠNG I'
    assert article_fts == 'Điều 1\nbody chi tiết'
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dense_units'"
    ).fetchone()
    conn.close()
