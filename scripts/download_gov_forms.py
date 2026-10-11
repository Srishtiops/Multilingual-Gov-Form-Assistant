#!/usr/bin/env python3
"""Download publicly available Indian government form PDFs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_url(url: str, dest: Path, timeout: int = 120) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            with dest.open("rb") as handle:
                is_pdf = handle.read(5) == b"%PDF-"
            if is_pdf:
                print(f"  already present; keeping {dest.name}")
                return True
            print(f"  existing file is not a PDF; keeping {dest.name}")
            return False
        except OSError as error:
            print(f"  cannot inspect existing {dest.name}: {error}")
            return False
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "GovFormAssistant/1.0"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF-"):
            raise ValueError("response did not contain a PDF file")
        temporary_path = dest.with_name(f"{dest.name}.download")
        temporary_path.write_bytes(r.content)
        os.replace(temporary_path, dest)
        print(f"  saved {dest.name} ({len(r.content)} bytes)")
        return True
    except Exception as e:
        print(f"  failed {url}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    root = args.config.parent.resolve()
    cfg = load_config(args.config)
    out = (root / cfg["paths"]["data_root"] / "raw" / "gov_forms").resolve()
    out.mkdir(parents=True, exist_ok=True)

    readme = out / "PLACEHOLDER_FORMS.txt"
    if not readme.exists():
        readme.write_text(
            "Add local PDFs for Passport, Driving Licence, Voter ID if official URLs change.\n"
            "Recommended: download from passportindia.gov.in, parivahan.gov.in, voters.eci.gov.in\n",
            encoding="utf-8",
        )

    ok = 0
    for item in cfg.get("gov_form_sources", []):
        print(f"Fetching {item['name']}...")
        if download_url(item["url"], out / item["filename"]):
            ok += 1

    print(f"Downloaded {ok}/{len(cfg.get('gov_form_sources', []))} configured forms -> {out}")


if __name__ == "__main__":
    main()
