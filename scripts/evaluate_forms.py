#!/usr/bin/env python3
"""Evaluate extraction and grounded Q&A on reviewed form examples (JSONL)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gov_form_assistant.evaluation import ocr_char_accuracy, qa_exact_match, qa_f1  # noqa: E402
from gov_form_assistant.field_extraction import extract_fields  # noqa: E402
from gov_form_assistant.pipeline import GovFormPipeline  # noqa: E402


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def is_abstention(answer: str) -> bool:
    text = norm(answer)
    return any(phrase in text for phrase in (
        "could not find", "not stated", "not mentioned", "not available", "cannot determine",
    ))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not row.get("id") or not (row.get("context") or row.get("image")):
                raise ValueError(f"Line {line_no} must include id and either context or image")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/indian_forms.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/latest_report.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="CUDA is used by model inference when available; cuda errors if unavailable.")
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    if not rows:
        raise SystemExit(f"No reviewed records in {args.dataset}. See data/evaluation/README.md.")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA device.")
    if args.device == "cpu":
        # Keep the runtime honest: current model loaders select CUDA from availability.
        # Device selection is reported below; the environment variable is set before loading models.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pipeline = GovFormPipeline(ROOT / "config.yaml", use_gpu=args.device != "cpu", lazy=True)
    field_tp = field_fp = field_fn = 0
    ocr_scores: list[float] = []
    qa_total = qa_em = qa_f1_sum = abstain_correct = 0
    errors: list[dict[str, Any]] = []
    by_language: dict[str, dict[str, int]] = {}

    for row in rows:
        context = row.get("context", "")
        if row.get("image"):
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Image for {row['id']} not found: {image_path}")
            inferred = pipeline.process_file(image_path, target_lang="en", simplify=False,
                                             source_lang=str(row.get("language", "auto")))
            context = inferred.ocr_text
            if inferred.ocr_confidence is not None:
                row["ocr_confidence"] = inferred.ocr_confidence
        if not context:
            raise ValueError(f"Record {row['id']} produced no text")
        predicted_fields = extract_fields(context).fields
        gold_fields = row.get("fields", {})
        pred_set = {f"{key}:{norm(value)}" for key, value in predicted_fields.items()}
        gold_set = {f"{key}:{norm(value)}" for key, value in gold_fields.items()}
        field_tp += len(pred_set & gold_set)
        field_fp += len(pred_set - gold_set)
        field_fn += len(gold_set - pred_set)
        if row.get("ocr_reference") is not None:
            ocr_scores.append(ocr_char_accuracy(context, row["ocr_reference"]).char_accuracy)

        lang = str(row.get("language", "unknown"))
        lang_counts = by_language.setdefault(lang, {"documents": 0, "questions": 0})
        lang_counts["documents"] += 1
        for item in row.get("questions", []):
            question, expected = item["question"], item.get("answer")
            predicted = pipeline.ask(question, context)
            qa_total += 1
            lang_counts["questions"] += 1
            if expected is None:
                correct = is_abstention(predicted)
                abstain_correct += int(correct)
                score = 1.0 if correct else 0.0
                exact = correct
            else:
                exact = qa_exact_match(predicted, str(expected))
                score = qa_f1(predicted, str(expected))
            qa_em += int(exact)
            qa_f1_sum += score
            if not exact:
                errors.append({"id": row["id"], "language": lang, "question": question,
                               "expected": expected, "predicted": predicted})

    precision = field_tp / (field_tp + field_fp) if field_tp + field_fp else 0.0
    recall = field_tp / (field_tp + field_fn) if field_tp + field_fn else 0.0
    field_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "dataset": str(args.dataset), "documents": len(rows), "by_language": by_language,
        "ocr_char_similarity_mean": sum(ocr_scores) / len(ocr_scores) if ocr_scores else None,
        "field_extraction": {"true_positive": field_tp, "false_positive": field_fp,
                             "false_negative": field_fn, "precision": precision,
                             "recall": recall, "f1": field_f1},
        "qa": {"questions": qa_total, "exact_match": qa_em / qa_total if qa_total else None,
               "token_f1_or_abstention_accuracy": qa_f1_sum / qa_total if qa_total else None,
               "abstention_correct": abstain_correct},
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False, indent=2))
    print(f"Full per-example report: {args.output}")


if __name__ == "__main__":
    main()
