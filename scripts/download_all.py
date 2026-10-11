#!/usr/bin/env python3
"""Download datasets, models, and government form PDFs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(Path(__file__).parent / script)] + (extra or [])
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[1]))


def main() -> None:
    steps = [
        ("download_datasets.py", []),
        ("download_indic_training_data.py", ["--max-rows", "10000"]),
        ("download_gov_forms.py", []),
        ("build_glossary.py", []),
        ("download_models.py", []),
    ]
    for script, extra in steps:
        code = run(script, extra)
        if code != 0:
            print(f"Warning: {script} exited with {code}")
    print("\nAll download steps finished (check warnings above).")


if __name__ == "__main__":
    main()
