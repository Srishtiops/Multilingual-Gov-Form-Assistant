#!/usr/bin/env python3
"""Download parallel and QA training resources for en, hi, and mr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_samanantar(out_dir: Path, hf_id: str, languages: tuple[str, ...], max_rows: int) -> None:
    target = out_dir / "samanantar"
    target.mkdir(parents=True, exist_ok=True)
    parallel_dir = out_dir.parent / "training" / "parallel"
    for lang in languages:
        try:
            split = f"train[:{max_rows}]" if max_rows > 0 else "train"
            dataset = load_dataset(hf_id, lang, split=split)
        except Exception as exc:
            print(f"  Samanantar {lang} skip: {exc}")
            continue
        raw_rows = [{"src": row.get("src"), "tgt": row.get("tgt"), "lang": lang} for row in tqdm(dataset, desc=lang)]
        save_jsonl(raw_rows, target / f"{lang}_en.jsonl")
        parallel_rows: list[dict] = []
        for row in raw_rows:
            src_text = (row.get("src") or "").strip()
            tgt_text = (row.get("tgt") or "").strip()
            if not src_text or not tgt_text:
                continue
            parallel_rows.append({"src_lang": lang, "tgt_lang": "en", "src": src_text, "tgt": tgt_text})
            parallel_rows.append({"src_lang": "en", "tgt_lang": lang, "src": tgt_text, "tgt": src_text})
        save_jsonl(parallel_rows, parallel_dir / f"samanantar_{lang}.jsonl")
        print(f"  Samanantar {lang}: {len(parallel_rows)} directed pairs -> {parallel_dir / f'samanantar_{lang}.jsonl'}")


def download_flores(out_dir: Path, max_dev: int = 500) -> None:
    parallel_dir = out_dir.parent / "training" / "parallel"
    eval_dir = out_dir.parent / "training" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    lang_map = {"eng_Latn": "en", "hin_Deva": "hi", "mar_Deva": "mr"}
    try:
        dataset = load_dataset("facebook/flores", "all", split=f"dev[:{max_dev}]")
    except Exception as exc:
        print(f"  Flores skip: {exc}")
        return
    by_id: dict[int, dict[str, str]] = {}
    for row in dataset:
        sentence_id = int(row["id"])
        flores_lang = row["iso_639_3"]
        if flores_lang not in lang_map:
            continue
        by_id.setdefault(sentence_id, {})[lang_map[flores_lang]] = row["sentence"].strip()
    pairs: list[dict] = []
    eval_rows: list[dict] = []
    for sentence_id, texts in sorted(by_id.items()):
        if not {"en", "hi", "mr"} <= texts.keys():
            continue
        for src_lang, tgt_lang in (("en", "hi"), ("hi", "en"), ("en", "mr"), ("mr", "en"), ("hi", "mr"), ("mr", "hi")):
            pairs.append({
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": texts[src_lang],
                "tgt": texts[tgt_lang],
                "source": "flores_dev",
                "id": sentence_id,
            })
        eval_rows.append({"id": sentence_id, "en": texts["en"], "hi": texts["hi"], "mr": texts["mr"]})
    save_jsonl(pairs, parallel_dir / "flores_dev_multilingual.jsonl")
    save_jsonl(eval_rows, eval_dir / "flores_en_hi_mr.jsonl")
    print(f"  Flores: {len(pairs)} directed pairs, {len(eval_rows)} trilingual sentences")


def download_opus100_hi(out_dir: Path, max_rows: int) -> None:
    parallel_dir = out_dir.parent / "training" / "parallel"
    try:
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split=f"train[:{max_rows}]")
    except Exception as exc:
        print(f"  OPUS-100 en-hi skip: {exc}")
        return
    rows: list[dict] = []
    for item in tqdm(dataset, desc="opus100-en-hi"):
        src = (item.get("translation") or {}).get("en", "").strip()
        tgt = (item.get("translation") or {}).get("hi", "").strip()
        if not src or not tgt:
            continue
        rows.append({"src_lang": "en", "tgt_lang": "hi", "src": src, "tgt": tgt, "source": "opus100"})
        rows.append({"src_lang": "hi", "tgt_lang": "en", "src": tgt, "tgt": src, "source": "opus100"})
    save_jsonl(rows, parallel_dir / "opus100_en_hi.jsonl")
    print(f"  OPUS-100 en-hi: {len(rows)} directed pairs")


def export_tydi_hindi_qa(raw_root: Path, hf_id: str, max_rows: int) -> None:
    qa_dir = raw_root.parent / "training" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    local = raw_root / "tydi_qa"
    try:
        if local.exists():
            from datasets import load_from_disk

            stored = load_from_disk(str(local))
            split = stored["train"] if "train" in stored else stored[list(stored.keys())[0]]
        else:
            split = load_dataset(hf_id, "primary_task", split="train")
    except Exception as exc:
        print(f"  TyDi QA export skip: {exc}")
        return
    rows: list[dict] = []
    for item in tqdm(split, desc="tydi-hi"):
        if str(item.get("language", "")).lower() != "hindi":
            continue
        context = (item.get("context") or "").strip()
        question = (item.get("question") or "").strip()
        answers = item.get("answers") or {}
        answer_list = answers.get("text") or []
        if not context or not question:
            continue
        rows.append({
            "language": "hi",
            "context": context,
            "question": question,
            "answers": [a.strip() for a in answer_list if a and str(a).strip()],
            "source": "tydiqa",
        })
        if len(rows) >= max_rows:
            break
    save_jsonl(rows, qa_dir / "tydi_hindi_train.jsonl")
    print(f"  TyDi QA Hindi: {len(rows)} examples -> {qa_dir / 'tydi_hindi_train.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--max-rows", type=int, default=25000, help="Max parallel rows per Samanantar language.")
    parser.add_argument("--opus-max-rows", type=int, default=15000)
    parser.add_argument("--skip-flores", action="store_true")
    parser.add_argument("--skip-opus", action="store_true")
    parser.add_argument("--skip-tydi", action="store_true")
    args = parser.parse_args()

    root = args.config.parent.resolve()
    cfg = load_config(args.config)
    raw_root = (root / cfg["paths"]["data_root"] / "raw").resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "training" / "parallel").mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg.get("datasets", {})
    samanantar_id = ds_cfg.get("samanantar_sample", "ai4bharat/samanantar")
    languages = tuple(cfg.get("project", {}).get("languages", ["en", "hi", "mr"]))
    indic_langs = tuple(lang for lang in languages if lang != "en")

    print("Downloading Samanantar parallel corpora...")
    download_samanantar(raw_root, samanantar_id, indic_langs, args.max_rows)

    if not args.skip_flores:
        print("Downloading Flores dev (en/hi/mr)...")
        download_flores(raw_root)

    if not args.skip_opus:
        print("Downloading OPUS-100 en-hi sample...")
        download_opus100_hi(raw_root, args.opus_max_rows)

    if not args.skip_tydi:
        print("Exporting TyDi QA Hindi for QA training/eval...")
        export_tydi_hindi_qa(raw_root, ds_cfg.get("tydi_qa", "google-research-datasets/tydiqa"), min(args.max_rows, 5000))

    manifest = {
        "languages": list(languages),
        "parallel_dir": "data/training/parallel",
        "notes": "See data/training/DATA_SOURCES.md for full catalog and licensing.",
    }
    manifest_path = root / "data" / "training" / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {manifest_path}")
    print("Indic training data download complete.")


if __name__ == "__main__":
    main()
