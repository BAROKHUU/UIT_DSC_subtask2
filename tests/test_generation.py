from legalqa.generation import build_final_answer


def test_final_answer_has_intro_and_bullets_without_mid_sentence_breaks_or_numbers():
    evidence = [
        {
            "raw_text": (
                "1. Tổ chức, hộ gia đình và cá nhân trên địa bàn là lực\n\n"
                "lượng tại chỗ thực hiện hoạt động phòng, chống thiên tai."
            )
        },
        {
            "raw_text": (
                "2. Nguồn nhân lực bao gồm:\n\n"
                "a) Dân quân tự vệ;\n\n"
                "b) Quân đội nhân dân."
            )
        },
    ]

    answer = build_final_answer("Câu hỏi", evidence)

    assert answer == (
        "Đối với Câu hỏi, quy định liên quan như sau:\n"
        "- Tổ chức, hộ gia đình và cá nhân trên địa bàn là lực lượng tại chỗ "
        "thực hiện hoạt động phòng, chống thiên tai.\n"
        "- Nguồn nhân lực bao gồm:\n"
        "- Dân quân tự vệ;\n"
        "- Quân đội nhân dân."
    )


def test_final_answer_keeps_intro_and_ignores_empty_evidence():
    evidence = [{"raw_text": "   \n"}, {"raw_text": "Nội dung hợp lệ."}]

    assert build_final_answer("Chủ đề là gì?", evidence) == (
        "Đối với Chủ đề, quy định liên quan như sau:\n"
        "- Nội dung hợp lệ."
    )


def test_intro_collapses_sibling_points_to_their_shared_clause():
    evidence = [
        {'legal_path': 'điểm a khoản 3 Điều 26', 'raw_text': 'Nội dung điểm a.'},
        {'legal_path': 'điểm b khoản 3 Điều 26', 'raw_text': 'Nội dung điểm b.'},
        {'legal_path': 'điểm d khoản 3 Điều 26', 'raw_text': 'Nội dung điểm d.'},
        {'legal_path': 'điểm đ khoản 3 Điều 26', 'raw_text': 'Nội dung điểm đ.'},
    ]

    answer = build_final_answer(
        'Công chức được bổ nhiệm vào ngạch thuyền viên kiểm ngư có tiêu chuẩn '
        'về năng lực chuyên môn, nghiệp vụ như thế nào?',
        evidence,
    )

    assert answer.startswith(
        'Đối với Công chức được bổ nhiệm vào ngạch thuyền viên kiểm ngư có tiêu '
        'chuẩn về năng lực chuyên môn, nghiệp vụ, các quy định liên quan tại '
        'khoản 3 Điều 26 như sau:\n'
    )
    assert 'điểm a khoản 3 Điều 26' not in answer


def test_intro_includes_a_human_readable_source_citation():
    evidence = [
        {
            'legal_path': 'điểm a khoản 3 Điều 26',
            'source_name': (
                'Thong-tu-07-2015-TT-BNV-chuc-danh-tieu-chuan-ngach-cong-chuc'
            ),
            'raw_text': 'Nội dung điểm a.',
        },
        {
            'legal_path': 'điểm b khoản 3 Điều 27',
            'source_name': (
                'Thong-tu-07-2015-TT-BNV-chuc-danh-tieu-chuan-ngach-cong-chuc'
            ),
            'raw_text': 'Nội dung điểm b.',
        },
    ]

    answer = build_final_answer('Tiêu chuẩn được quy định như thế nào?', evidence)

    assert answer.startswith(
        'Đối với Tiêu chuẩn, theo khoản 3 Điều 26, khoản 3 Điều 27 '
        'Thông tư 07/2015/TT-BNV quy định như sau:\n'
    )
    assert answer.count('Thông tư 07/2015/TT-BNV') == 1


def test_intro_groups_sibling_clauses_under_their_shared_article():
    evidence = [
        {'legal_path': 'Điều 6', 'raw_text': 'Nội dung Điều 6.'},
        {'legal_path': 'khoản 1 Điều 6', 'raw_text': 'Nội dung khoản 1.'},
        {'legal_path': 'khoản 3 Điều 6', 'raw_text': 'Nội dung khoản 3.'},
        {'legal_path': 'khoản 4 Điều 6', 'raw_text': 'Nội dung khoản 4.'},
        {'legal_path': 'khoản 1 Điều 1', 'raw_text': 'Nội dung Điều 1.'},
    ]

    answer = build_final_answer('Câu hỏi', evidence)

    assert answer.startswith(
        'Đối với Câu hỏi, các quy định liên quan tại '
        'khoản 1, khoản 3, khoản 4 Điều 6, khoản 1 Điều 1 như sau:\n'
    )
    assert 'Điều 6, khoản 1 Điều 6' not in answer


def test_final_answer_is_empty_when_evidence_has_no_text():
    assert build_final_answer("Chủ đề là gì?", [{"raw_text": " \n\n "}]) == ""
