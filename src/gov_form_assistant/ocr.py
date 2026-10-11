from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]]


def _reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: sum(point[1] for point in item.box) / len(item.box)):
        top = min(point[1] for point in line.box)
        bottom = max(point[1] for point in line.box)
        line_height = max(1.0, bottom - top)
        matching_row = None
        for row in rows:
            row_top = min(point[1] for item in row for point in item.box)
            row_bottom = max(point[1] for item in row for point in item.box)
            row_height = max(1.0, row_bottom - row_top)
            overlap = max(0.0, min(bottom, row_bottom) - max(top, row_top))
            if overlap >= 0.35 * min(line_height, row_height):
                matching_row = row
                break
        if matching_row is None:
            rows.append([line])
        else:
            matching_row.append(line)

    ordered: list[OcrLine] = []
    for row in sorted(rows, key=lambda items: min(point[1] for item in items for point in item.box)):
        ordered.extend(sorted(row, key=lambda item: min(point[0] for point in item.box)))
    return ordered


class FormOcr:
    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.2,
    ):
        import easyocr

        langs = languages or ["en", "hi"]
        # EasyOCR has no separate Marathi model; Devanagari uses the Hindi recognizer.
        ocr_langs = ["hi" if lang == "mr" else lang for lang in langs]
        ocr_langs = list(dict.fromkeys(ocr_langs))
        self.languages = langs
        self.reader = easyocr.Reader(ocr_langs, gpu=gpu, verbose=False)
        self.min_confidence = min_confidence

    def extract(self, image: np.ndarray) -> list[OcrLine]:
        # Small scans lose thin Devanagari strokes during OCR. Upscale modestly and
        # improve local contrast while retaining the original color image when already large.
        import cv2

        height, width = image.shape[:2]
        scale = min(2.0, 1800.0 / max(height, width))
        prepared = image
        if scale > 1.05:
            prepared = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if prepared.ndim == 3:
            gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
        else:
            gray = prepared
        if gray.shape[0] < 1800 and gray.shape[1] < 1800:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        raw = self.reader.readtext(gray, detail=1, paragraph=False)
        lines: list[OcrLine] = []
        for box, text, conf in raw:
            confidence = float(conf)
            cleaned_text = text.strip()
            if not cleaned_text or confidence < self.min_confidence:
                continue
            normalized_box = [[float(x), float(y)] for x, y in box]
            lines.append(OcrLine(text=cleaned_text, confidence=confidence, box=normalized_box))
        return _reading_order(lines)

    def full_text(self, image: np.ndarray) -> str:
        return "\n".join(line.text for line in self.extract(image) if line.text)

    def read_with_confidence(self, image: np.ndarray) -> tuple[str, float | None, list[dict[str, float | str]]]:
        lines = self.extract(image)
        details = [{"text": line.text, "confidence": line.confidence} for line in lines]
        mean_confidence = sum(line.confidence for line in lines) / len(lines) if lines else None
        return "\n".join(line.text for line in lines), mean_confidence, details
