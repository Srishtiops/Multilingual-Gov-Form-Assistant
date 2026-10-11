#!/usr/bin/env python3
"""Run lightweight metric demos on downloaded data samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gov_form_assistant.evaluation import ocr_char_accuracy, qa_exact_match, qa_f1, translation_bleu  # noqa: E402


def main() -> None:
    print("=== Evaluation utilities (sample) ===")
    ocr = ocr_char_accuracy("Applicant Name: Ravi Kumar", "Applicant Name: Ravi Kumar")
    print(f"OCR char accuracy (demo): {ocr.char_accuracy:.2f}")

    bleu = translation_bleu(["the form must be signed"], [["the form must be signed"]])
    print(f"BLEU (demo): {bleu:.2f}")

    em = qa_exact_match("Attach photograph", "attach photograph")
    f1 = qa_f1("Attach a recent photograph", "Attach photograph")
    print(f"QA EM (case-insensitive demo): {em}, token F1: {f1:.2f}")

    saman = ROOT / "data" / "raw" / "samanantar" / "hi_en_sample.jsonl"
    if saman.exists():
        hyps, refs = [], []
        with saman.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                row = json.loads(line)
                if row.get("tgt") and row.get("src"):
                    hyps.append(row["src"])
                    refs.append([row["tgt"]])
        if hyps:
            print(f"Samanantar sample BLEU (src vs tgt, 100 lines): {translation_bleu(hyps, refs):.2f}")
    else:
        print("Samanantar sample not found; run download_datasets.py first.")


if __name__ == "__main__":
    main()
