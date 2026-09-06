from __future__ import annotations

import re
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

# Legal nodes often start with markers such as ``1.``, ``2)``, ``3.2`` or
# ``a)``. Output paragraphs are separated by line breaks, so these markers are
# removed instead of creating numbered or bulleted lists.
LIST_MARKER_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[.)]"
    r"|\d+(?:\.\d+)+"
    r"|[A-Za-zĐđ][.)]"
    r"|[IVXLCDM]+[.)]"
    r")\s+",
    re.IGNORECASE,
)

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


SOURCE_CITATION_PATTERNS = [
    (
        re.compile(
            r'(?:^|-)Thong-tu-(?:so-)?(\d+)-(\d{4})-TT-([A-Z0-9]+(?:-[A-Z0-9]+)*)'
        ),
        lambda match: f'Thông tư {match.group(1)}/{match.group(2)}/TT-{match.group(3)}',
    ),
    (
        re.compile(r'(?:^|-)Nghi-dinh-(\d+)-(\d{4})-ND-CP(?:-|$)'),
        lambda match: f'Nghị định {match.group(1)}/{match.group(2)}/NĐ-CP',
    ),
    (
        re.compile(r'(?:^|-)Quyet-dinh-(\d+)-(?:\d{4}-)?QD-([A-Za-z0-9]+)'),
        lambda match: f'Quyết định {match.group(1)}/QĐ-{match.group(2)}',
    ),
    (
        re.compile(r'(?:^|-)Nghi-quyet-(\d+)-(\d{4})-NQ-([A-Za-z0-9]+)'),
        lambda match: f'Nghị quyết {match.group(1)}/{match.group(2)}/NQ-{match.group(3)}',
    ),
]


def derive_subject_hint(question: str) -> str:
    text = question.strip().rstrip("? ")
    lower = text.lower()
    for suffix in QUESTION_SUFFIXES:
        if lower.endswith(suffix):
            text = text[: len(text) - len(suffix)].rstrip(" ,:")
            break
    return text


def legal_path_for_intro(path: str) -> str:
    '''Keep intro citations concise by referring to a point's parent clause.'''
    normalized = path.strip()
    point_match = re.match(r'^điểm\s+\S+\s+(khoản\s+.+)$', normalized, re.IGNORECASE)
    if point_match:
        return point_match.group(1)
    return normalized


def source_citation_for_intro(source_name: str) -> str:
    '''Extract a concise, human-readable citation from a source filename.'''
    normalized = source_name.strip()
    for pattern, formatter in SOURCE_CITATION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return formatter(match)
    return ''


def compact_legal_paths(paths: list[str]) -> list[str]:
    '''Group sibling clauses and show their shared article only once.'''
    article_groups: dict[str, list[str]] = {}
    ordered_items: list[tuple[str, str]] = []
    literal_paths: set[str] = set()

    for path in paths:
        clause_match = re.fullmatch(
            r'khoản\s+(\d+)\s+Điều\s+(\d+[a-zđ]?)',
            path,
            re.IGNORECASE,
        )
        article_match = re.fullmatch(
            r'Điều\s+(\d+[a-zđ]?)',
            path,
            re.IGNORECASE,
        )
        if clause_match or article_match:
            article = (clause_match or article_match).group(
                2 if clause_match else 1
            )
            if article not in article_groups:
                article_groups[article] = []
                ordered_items.append(('article', article))
            if clause_match:
                clause = clause_match.group(1)
                if clause not in article_groups[article]:
                    article_groups[article].append(clause)
            continue

        if path not in literal_paths:
            literal_paths.add(path)
            ordered_items.append(('literal', path))

    compacted: list[str] = []
    for item_type, value in ordered_items:
        if item_type == 'literal':
            compacted.append(value)
            continue
        clauses = article_groups[value]
        if clauses:
            clause_text = ', '.join(f'khoản {clause}' for clause in clauses)
            compacted.append(f'{clause_text} Điều {value}')
        else:
            compacted.append(f'Điều {value}')
    return compacted


def unique_legal_paths(evidence: list[dict], limit: int = 4) -> list[str]:
    paths: list[str] = []
    for node in evidence:
        path = str(node.get("legal_path", "")).strip()
        path = legal_path_for_intro(path)
        if path and path not in paths:
            paths.append(path)
    return compact_legal_paths(paths)[:limit]


def unique_legal_references(evidence: list[dict], limit: int = 4) -> list[str]:
    paths_by_source: dict[str, list[str]] = {}
    for node in evidence:
        path = legal_path_for_intro(str(node.get('legal_path', '')))
        source = source_citation_for_intro(str(node.get('source_name', '')))
        paths = paths_by_source.setdefault(source, [])
        if path and path not in paths:
            paths.append(path)

    references: list[str] = []
    for source, paths in paths_by_source.items():
        joined_paths = ', '.join(compact_legal_paths(paths))
        reference = ' '.join(part for part in (joined_paths, source) if part)
        if reference:
            references.append(reference)
        if len(references) >= limit:
            break
    return references


def deterministic_intro(question: str, evidence: list[dict]) -> str:
    subject = derive_subject_hint(question)
    sources = [
        source_citation_for_intro(str(node.get('source_name', '')))
        for node in evidence
    ]
    if any(sources):
        references = unique_legal_references(evidence)
        joined_references = ', '.join(references)
        return f'Đối với {subject}, theo {joined_references} quy định như sau:'
    paths = unique_legal_paths(evidence)
    if paths:
        return f"Đối với {subject}, các quy định liên quan tại {', '.join(paths)} như sau:"
    return f"Đối với {subject}, quy định liên quan như sau:"


def build_intro_prompt(question: str, evidence: list[dict]) -> str:
    subject = derive_subject_hint(question)
    paths = '; '.join(unique_legal_references(evidence))
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
    paragraphs: list[str] = []
    for node in evidence:
        current_parts: list[str] = []

        def flush() -> None:
            if current_parts:
                paragraphs.append(" ".join(current_parts).strip())
                current_parts.clear()

        for raw_line in str(node.get("raw_text", "")).splitlines():
            line = " ".join(raw_line.replace("\u00a0", " ").split())
            if not line:
                continue

            marker = LIST_MARKER_RE.match(line)
            if marker:
                flush()
                line = line[marker.end() :].strip()
            if line:
                current_parts.append(line)

        flush()

    paragraphs = [text for text in paragraphs if text]
    if not paragraphs:
        return ""

    intro = (
        intro_generator.generate(question, evidence)
        if intro_generator is not None
        else deterministic_intro(question, evidence)
    )
    evidence_lines = [f"- {text}" for text in paragraphs]
    return "\n".join([intro, *evidence_lines])
