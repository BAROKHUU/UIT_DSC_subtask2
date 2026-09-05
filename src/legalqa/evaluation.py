from __future__ import annotations

import re

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+|[^\s]", re.UNICODE)


def tokenize_vi(text: str) -> list[str]:
    # Approximation only. Replace with the competition's official tokenizer/scorer
    # if organizers release one.
    return TOKEN_RE.findall(text.lower())


def approximate_meteor(reference: str, prediction: str) -> float:
    return float(meteor_score([tokenize_vi(reference)], tokenize_vi(prediction)))


def rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
