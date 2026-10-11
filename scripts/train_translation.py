#!/usr/bin/env python3
"""Fine-tune the public NLLB fallback on local en/hi/mr parallel rows (+ glossary)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from gov_form_assistant.translation import LANG_CODES

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_parallel_rows(parallel_dir: Path, max_rows: int, seed: int) -> list[dict]:
    gov_rows: list[dict] = []
    general_rows: list[dict] = []
    for path in sorted(parallel_dir.glob("*.jsonl")):
        is_gov = "gov" in path.name.lower() or "glossary" in path.name.lower()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                src = (item.get("src") or "").strip()
                tgt = (item.get("tgt") or "").strip()
                src_lang = item.get("src_lang")
                tgt_lang = item.get("tgt_lang")
                if not src or not tgt or src_lang not in LANG_CODES or tgt_lang not in LANG_CODES:
                    continue
                record = {"src": src, "tgt": tgt, "src_lang": src_lang, "tgt_lang": tgt_lang}
                if is_gov:
                    gov_rows.append(record)
                else:
                    general_rows.append(record)
    rng = random.Random(seed)
    rng.shuffle(gov_rows)
    rng.shuffle(general_rows)
    if max_rows > 0:
        remain = max(0, max_rows - len(gov_rows))
        rows = gov_rows + general_rows[:remain]
    else:
        rows = gov_rows + general_rows
    rng.shuffle(rows)
    if len(rows) < 100:
        raise RuntimeError(
            f"Need at least 100 parallel rows under {parallel_dir}. "
            "Run scripts/download_indic_training_data.py and scripts/build_glossary.py first."
        )
    return rows


def build_dataset(rows: list[dict], tokenizer, max_source: int, max_target: int):
    def preprocess(example):
        src_lang = example["src_lang"]
        tgt_lang = example["tgt_lang"]
        tokenizer.src_lang = LANG_CODES[src_lang]
        tokenizer.tgt_lang = LANG_CODES[tgt_lang]
        model_inputs = tokenizer(
            example["src"],
            text_target=example["tgt"],
            max_length=max_source,
            truncation=True,
        )
        return model_inputs

    dataset = Dataset.from_list(rows)
    return dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = args.config.parent.resolve()
    cache_dir = str(root / cfg["paths"]["cache_dir"])
    base_model = cfg["models"]["translation"].get("fallback_public", "facebook/nllb-200-distilled-600M")
    output_dir = args.output_dir or (root / "models" / "training" / "nllb_gov_domain")
    parallel_dir = root / "data" / "training" / "parallel"

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    rows = load_parallel_rows(parallel_dir, args.max_rows, args.seed)
    split_index = max(1, int(len(rows) * 0.05))
    eval_rows = rows[:split_index]
    train_rows = rows[split_index:]

    print(f"Base model: {base_model}")
    print(f"Train pairs: {len(train_rows)}, eval pairs: {len(eval_rows)}")

    kwargs = {"cache_dir": cache_dir}
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            **kwargs,
            torch_dtype=torch.float32,
        )
    except Exception:
        kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            **kwargs,
            torch_dtype=torch.float32,
        )
    train_dataset = build_dataset(train_rows, tokenizer, max_source=128, max_target=128)
    eval_dataset = build_dataset(eval_rows, tokenizer, max_source=128, max_target=128)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=max(20, args.max_steps // 4),
        save_steps=max(20, args.max_steps // 4),
        save_total_limit=1,
        logging_steps=10,
        predict_with_generate=False,
        generation_max_length=128,
        fp16=False,  # Model loaded as float32; fp16 gradient scaler would fail with float32 weights.
        report_to=[],
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {
        "base_model": base_model,
        "train_pairs": len(train_rows),
        "eval_pairs": len(eval_rows),
        "max_steps": args.max_steps,
        "device": device,
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Fine-tuned model saved to {output_dir}")
    print("Set models.translation.finetuned_local in config.yaml to use it in the app.")


if __name__ == "__main__":
    main()
