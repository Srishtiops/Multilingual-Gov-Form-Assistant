from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images: list[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def extract_pdf_text_pages(pdf_path: Path) -> list[str]:
    with fitz.open(pdf_path) as doc:
        return [page.get_text("text").strip() for page in doc]


def load_document(path: Path) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pil_images = pdf_to_images(path)
    else:
        pil_images = [Image.open(path).convert("RGB")]
    return [np.array(img) for img in pil_images]
