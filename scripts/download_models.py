#!/usr/bin/env python3
"""Prefetch Hugging Face models and warm EasyOCR weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LayoutLMv3Processor


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_hf_model(model_id: str, cache_dir: Path) -> bool:
    print(f"Downloading {model_id}...")
    try:
        snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
        print(f"  cached under {cache_dir}")
        return True
    except Exception as e:
        print(f"  failed {model_id}: {e}")
        return False


def warm_transformers(model_id: str, cache_dir: Path, kind: str) -> None:
    print(f"Loading {kind} weights for {model_id} (validation)...")
    if kind == "layoutlm":
        LayoutLMv3Processor.from_pretrained(model_id, cache_dir=str(cache_dir))
    else:
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=str(cache_dir))
    print("  OK")


def warm_easyocr() -> None:
    print("Initializing EasyOCR (downloads detection/recognition weights)...")
    import easyocr

    _ = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)
    print("  EasyOCR ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--skip-easyocr", action="store_true")
    args = parser.parse_args()

    root = args.config.parent.resolve()
    cfg = load_config(args.config)
    cache = (root / cfg["paths"]["cache_dir"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    m = cfg["models"]
    tr = m["translation"]
    preferred = [tr["id"], tr.get("id_1b", tr["id"]), tr["alt_indic"], tr.get("alt_indic_1b", tr["alt_indic"])]
    fallback_public = tr.get("fallback_public", "facebook/m2m100_418M")
    translation_model = fallback_public
    if any(download_hf_model(model, cache) for model in preferred):
        translation_model = tr["id"]
    else:
        print("All gated translation repos failed; trying public multilingual fallback...")
        download_hf_model(fallback_public, cache)
        translation_model = fallback_public

    download_hf_model(m["document_understanding"]["id"], cache)
    download_hf_model(m["qa_explanation"]["id"], cache)

    warm_transformers(translation_model, cache, "translation")
    warm_transformers(m["document_understanding"]["id"], cache, "layoutlm")
    warm_transformers(m["qa_explanation"]["id"], cache, "qa")

    if not args.skip_easyocr:
        warm_easyocr()

    print("Model download complete.")


if __name__ == "__main__":
    main()
