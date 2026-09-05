from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION_SUFFIXES = [
    "được quy định như thế nào",
    "được quy định thế nào",
    "được thực hiện như thế nào",
    "được thực hiện thế nào",
    "bao gồm những gì",
    "gồm những gì",
    "như thế nào",
    "thế nào",
    "là gì",
    "ra sao",
]

INTRO_SYSTEM_PROMPT = """
Bạn chỉ có nhiệm vụ viết MỘT câu mở đầu ngắn cho câu trả lời pháp luật tiếng Việt.

QUY TẮC BẮT BUỘC:
1. Chỉ xác định chủ thể/chủ đề của câu hỏi.
2. Có thể nhắc Điều/Khoản/Điểm được cung cấp.
3. Không được tự trả lời nội dung pháp luật.
4. Không được thêm mức phạt, thời hạn, điều kiện, nghĩa vụ, quyền hạn hoặc kết luận pháp lý.
5. Không suy diễn và không giải thích.
6. Chỉ output đúng MỘT câu.
7. Ưu tiên wording đã xuất hiện trong câu hỏi.
8. Phần quy định pháp luật sẽ được chương trình tự động nối nguyên văn sau câu của bạn.
""".strip()


def derive_subject_hint(question: str) -> str:
    text = question.strip().rstrip("? ")
    lower = text.lower()
    for suffix in QUESTION_SUFFIXES:
        if lower.endswith(suffix):
            text = text[: len(text) - len(suffix)].rstrip(" ,:")
            break
    return text


def unique_legal_paths(evidence: list[dict], limit: int = 4) -> list[str]:
    paths: list[str] = []
    for node in evidence:
        path = str(node.get("legal_path", "")).strip()
        if path and path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def deterministic_intro(question: str, evidence: list[dict]) -> str:
    subject = derive_subject_hint(question)
    paths = unique_legal_paths(evidence)
    if paths:
        return f"Đối với {subject}, các quy định liên quan tại {', '.join(paths)} như sau:"
    return f"Đối với {subject}, quy định liên quan như sau:"


def build_intro_prompt(question: str, evidence: list[dict]) -> str:
    subject = derive_subject_hint(question)
    paths = "; ".join(unique_legal_paths(evidence))
    return (
        f"CÂU HỎI:\n{question}\n\n"
        f"CHỦ THỂ:\n{subject}\n\n"
        f"CĂN CỨ ĐÃ TRUY XUẤT:\n{paths}\n\n"
        "Viết đúng một câu mở đầu."
    )


class HFIntroGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 64,
        trust_remote_code: bool = True,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, question: str, evidence: list[dict]) -> str:
        user_prompt = build_intro_prompt(question, evidence)
        messages = [
            {"role": "system", "content": INTRO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = INTRO_SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nTrả lời:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        intro = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Hard boundary: only the first non-empty line is accepted.
        intro = next((line.strip() for line in intro.splitlines() if line.strip()), "")
        return intro or deterministic_intro(question, evidence)


def build_final_answer(
    question: str,
    evidence: list[dict],
    intro_generator: Optional[HFIntroGenerator] = None,
) -> str:
    if not evidence:
        return ""

    intro = (
        intro_generator.generate(question, evidence)
        if intro_generator is not None
        else deterministic_intro(question, evidence)
    )

    # Critical boundary: evidence body is NEVER passed through the LLM.
    evidence_texts = [str(node.get("raw_text", "")).strip() for node in evidence]
    evidence_texts = [text for text in evidence_texts if text]
    return intro + "\n" + "\n".join(evidence_texts)
