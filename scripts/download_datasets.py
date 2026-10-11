#!/usr/bin/env python3
"""Download FUNSD, XFUND, TyDi QA, and Samanantar sample splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_funsd(out_dir: Path, hf_id: str) -> None:
    print(f"Downloading FUNSD ({hf_id})...")
    ds = load_dataset(hf_id)
    target = out_dir / "funsd"
    target.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target))
    print(f"  saved to {target}")


def download_xfund(out_dir: Path, hf_id: str, github_base: str, langs: list[str] | None = None) -> None:
    target = out_dir / "xfund"
    target.mkdir(parents=True, exist_ok=True)
    langs = langs or ["de", "es", "fr", "it", "ja", "pt", "zh"]

    print(f"Downloading XFUND annotations from GitHub ({github_base})...")
    for lang in langs:
        for split in ("train", "val"):
            name = f"{lang}.{split}.json"
            url = f"{github_base}/{name}"
            dest = target / name
            try:
                r = requests.get(url, timeout=120, headers={"User-Agent": "GovFormAssistant/1.0"})
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"  saved {name}")
            except Exception as e:
                print(f"  skip {name}: {e}")

    print(f"Downloading XFUND HF mirror ({hf_id}) if available...")
    try:
        ds = load_dataset(hf_id, "de")
        ds.save_to_disk(str(target / "hf_de_subset"))
        print(f"  saved HF subset to {target / 'hf_de_subset'}")
    except Exception as e:
        print(f"  HF XFUND optional skip: {e}")


def download_tydi_qa(out_dir: Path, hf_id: str) -> None:
    print(f"Downloading TyDi QA ({hf_id})...")
    # Primary English secondary task is smaller; full goldp is large
    try:
        ds = load_dataset(hf_id, "primary_task")
    except Exception:
        ds = load_dataset(hf_id)
    target = out_dir / "tydi_qa"
    target.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target))
    print(f"  saved to {target}")


def download_samanantar(out_dir: Path, hf_id: str, max_rows: int = 5000) -> None:
    print(f"Downloading Samanantar sample ({hf_id})...")
    target = out_dir / "samanantar"
    target.mkdir(parents=True, exist_ok=True)
    for lang in ("hi", "mr"):
        try:
            ds = load_dataset(hf_id, lang, split=f"train[:{max_rows}]")
        except Exception as e:
            print(f"  Samanantar {lang} skip: {e}")
            continue
        rows = [{"src": r.get("src"), "tgt": r.get("tgt"), "lang": lang} for r in tqdm(ds, desc=lang)]
        save_jsonl(rows, target / f"{lang}_en_sample.jsonl")
        print(f"  saved {len(rows)} pairs -> {target / f'{lang}_en_sample.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--skip-samanantar", action="store_true")
    args = parser.parse_args()

    root = args.config.parent.resolve()
    cfg = load_config(args.config)
    data_root = (root / cfg["paths"]["data_root"] / "raw").resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["datasets"]
    download_funsd(data_root, ds_cfg["funsd"])
    download_xfund(
        data_root,
        ds_cfg.get("xfund_hf", "FrancophonIA/XFUND"),
        ds_cfg.get("xfund_github_release", "https://github.com/doc-analysis/XFUND/releases/download/v1.0"),
    )
    download_tydi_qa(data_root, ds_cfg["tydi_qa"])
    if not args.skip_samanantar:
        download_samanantar(data_root, ds_cfg["samanantar_sample"])

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
