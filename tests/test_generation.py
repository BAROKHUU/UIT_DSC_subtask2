from legalqa.generation import build_final_answer


def test_final_answer_has_intro_and_plain_lines_without_mid_sentence_breaks_or_numbers():
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
        "Tổ chức, hộ gia đình và cá nhân trên địa bàn là lực lượng tại chỗ "
        "thực hiện hoạt động phòng, chống thiên tai.\n"
        "Nguồn nhân lực bao gồm:\n"
        "Dân quân tự vệ;\n"
        "Quân đội nhân dân."
    )


def test_final_answer_keeps_intro_and_ignores_empty_evidence():
    evidence = [{"raw_text": "   \n"}, {"raw_text": "Nội dung hợp lệ."}]

    assert build_final_answer("Chủ đề là gì?", evidence) == (
        "Đối với Chủ đề, quy định liên quan như sau:\n"
        "Nội dung hợp lệ."
    )


def test_final_answer_is_empty_when_evidence_has_no_text():
    assert build_final_answer("Chủ đề là gì?", [{"raw_text": " \n\n "}]) == ""
