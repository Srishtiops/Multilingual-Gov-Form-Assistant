from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

INDIAN_LINE_LABELS = [
    "unlabeled",
    "instruction",
    "required_document",
    "field",
    "contact_detail",
    "disclaimer",
    "other",
]


def extract_annotation_lines(pdf_path: Path, use_gpu: bool = False) -> list[str]:
    from gov_form_assistant.document_io import extract_pdf_text_pages, load_document
    from gov_form_assistant.ocr import FormOcr

    if pdf_path.suffix.lower() == ".pdf":
        pages = extract_pdf_text_pages(pdf_path)
        if pages and all(sum(char.isalnum() for char in page) >= 60 for page in pages):
            text_pages = pages
        else:
            images = load_document(pdf_path)
            reader = FormOcr(gpu=use_gpu, min_confidence=0.2)
            text_pages = [
                pages[index]
                if index < len(pages) and sum(char.isalnum() for char in pages[index]) >= 60
                else reader.full_text(image)
                for index, image in enumerate(images)
            ]
    else:
        from gov_form_assistant.document_io import load_document
        from gov_form_assistant.ocr import FormOcr

        reader = FormOcr(gpu=use_gpu, min_confidence=0.2)
        text_pages = [reader.full_text(image) for image in load_document(pdf_path)]

    lines: list[str] = []
    for page_text in text_pages:
        lines.extend(line.strip() for line in page_text.splitlines() if line.strip())
    return lines


def document_key(source_name: str, contents: bytes) -> str:
    digest = hashlib.sha256(contents).hexdigest()
    return f"{Path(source_name).name}:{digest}"


def save_document_annotations(
    annotation_path: Path,
    document_id: str,
    source_name: str,
    lines: list[str],
    labels: list[str],
) -> int:
    if len(lines) != len(labels):
        raise ValueError("Each source line must have exactly one label.")
    if any(not label or label == "unlabeled" for label in labels):
        raise ValueError("Label every line before saving the document.")
    unknown = sorted(set(labels) - set(INDIAN_LINE_LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    records = [
        {
            "document_id": document_id,
            "source": source_name,
            "line_index": index,
            "text": text,
            "label": label,
            "task": "indian_government_form_line_classification",
        }
        for index, (text, label) in enumerate(zip(lines, labels))
    ]
    existing: list[dict[str, Any]] = []
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    retained = [record for record in existing if record.get("document_id") != document_id]
    updated = retained + records

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{annotation_path.name}.", suffix=".tmp", dir=annotation_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in updated:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_name, annotation_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return len(records)
