from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz
from sacrebleu import corpus_bleu


@dataclass
class OcrMetrics:
    char_accuracy: float


def ocr_char_accuracy(pred: str, ref: str) -> OcrMetrics:
    if not ref:
        return OcrMetrics(char_accuracy=0.0)
    score = fuzz.ratio(pred, ref) / 100.0
    return OcrMetrics(char_accuracy=score)


def translation_bleu(hypotheses: list[str], references: list[list[str]]) -> float:
    if not hypotheses or not references:
        return 0.0
    return float(corpus_bleu(hypotheses, references).score)


def span_f1(pred_spans: set[str], gold_spans: set[str]) -> tuple[float, float, float]:
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def qa_exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def qa_f1(pred: str, gold: str) -> float:
    pred_t = pred.lower().split()
    gold_t = gold.lower().split()
    if not pred_t and not gold_t:
        return 1.0
    if not pred_t or not gold_t:
        return 0.0
    common = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(gold_t)
    return 2 * p * r / (p + r)
