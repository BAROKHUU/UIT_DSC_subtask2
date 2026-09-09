from legalqa.parser import parse_document


def test_basic_legal_hierarchy():
    passage = """CHƯƠNG I\nQUY ĐỊNH CHUNG\n\nĐiều 1. Điều kiện cấp phép\n1. Tổ chức phải đáp ứng:\na) Có đủ nhân sự;\nb) Có cơ sở vật chất.\n2. Điều kiện khác.\n"""
    nodes = parse_document("1", "sample", passage)
    types = [n.node_type for n in nodes]
    assert "article" in types
    assert "clause" in types
    assert "point" in types

    point = next(n for n in nodes if n.node_type == "point" and n.label == "a")
    assert "Điều 1" in point.context_text
    assert point.raw_text.startswith("a)")
    assert point.legal_path == "điểm a khoản 1 Điều 1"
    chapter = next(n for n in nodes if n.node_type == "chapter")
    assert chapter.is_sparse_indexable


def test_repeated_article_number_has_unique_node_id():
    passage = """Điều 1. Phần thứ nhất\n1. Nội dung A.\n\nĐiều 2. Chuyển tiếp\nNội dung.\n\nĐiều 1. Quy chế ban hành kèm theo\n1. Nội dung B.\n"""
    nodes = parse_document("99", "repeated", passage)
    article_ones = [n for n in nodes if n.node_type == "article" and n.label == "1"]
    assert len(article_ones) == 2
    assert article_ones[0].node_id != article_ones[1].node_id
