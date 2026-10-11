"""Optional LayoutLMv3-based token classification hook for form fields."""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


@lru_cache(maxsize=1)
def _load_layoutlm(model_id: str, cache_dir: str | None):
    processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=False, cache_dir=cache_dir)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_id, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model, device


def layoutlm_ready(model_id: str, cache_dir: str | None = None) -> bool:
    try:
        _load_layoutlm(model_id, cache_dir)
        return True
    except Exception:
        return False
