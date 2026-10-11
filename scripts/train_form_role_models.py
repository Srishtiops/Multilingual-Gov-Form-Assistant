#!/usr/bin/env python3
"""Train scratch form-role classifiers on FUNSD and XFUND annotations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
FORM_ROLE_LABELS = ["header", "question", "answer", "other"]
INDIAN_FORM_LABELS = ["instruction", "required_document", "field", "contact_detail", "disclaimer", "other"]
LABELS = FORM_ROLE_LABELS.copy()
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TOKEN_PATTERN = re.compile(r"[\w\u0900-\u0963\u0966-\u097f]+", flags=re.UNICODE)


def set_labels(labels: list[str]) -> None:
    global LABELS, LABEL_TO_ID
    LABELS = labels
    LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def _reading_lines(words: list[str], boxes: list[list[int]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for index in sorted(range(len(words)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2):
        top, bottom = boxes[index][1], boxes[index][3]
        height = max(1, bottom - top)
        row_match = None
        for row in rows:
            row_top = min(boxes[i][1] for i in row)
            row_bottom = max(boxes[i][3] for i in row)
            overlap = max(0, min(bottom, row_bottom) - max(top, row_top))
            if overlap >= 0.35 * min(height, max(1, row_bottom - row_top)):
                row_match = row
                break
        if row_match is None:
            rows.append([index])
        else:
            row_match.append(index)
    return [sorted(row, key=lambda i: boxes[i][0]) for row in rows]


def load_training_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    funsd_path = ROOT / "data" / "raw" / "funsd"
    if funsd_path.exists():
        dataset = load_from_disk(str(funsd_path))
        for split_name, destination in (("train", train), ("test", test)):
            if split_name not in dataset:
                continue
            split = dataset[split_name]
            tag_names = split.features["ner_tags"].feature.names
            for row in split:
                words = row["words"]
                boxes = row["bboxes"]
                tags = [tag_names[tag_id].split("-")[-1].lower() for tag_id in row["ner_tags"]]
                for indexes in _reading_lines(words, boxes):
                    text = " ".join(words[index] for index in indexes).strip()
                    line_tags = [tags[index] for index in indexes if tags[index] != "o"]
                    label = Counter(line_tags).most_common(1)[0][0] if line_tags else "other"
                    if text and label in LABEL_TO_ID:
                        destination.append({
                            "text": text,
                            "label": label,
                            "document_id": f"funsd:{row['id']}",
                            "source": "FUNSD",
                            "split": split_name,
                        })

    xfund_path = ROOT / "data" / "raw" / "xfund"
    for language in ("de", "es", "fr", "it", "ja", "pt", "zh"):
        for split_name, destination in (("train", train), ("val", test)):
            path = xfund_path / f"{language}.{split_name}.json"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            for document in payload.get("documents", []):
                document_id = f"xfund:{language}:{document.get('id', document.get('uid', 'unknown'))}"
                for item in document.get("document", []):
                    label = item.get("label", "other").lower()
                    text = " ".join(str(item.get("text", "")).split())
                    if text and label in LABEL_TO_ID:
                        destination.append({
                            "text": text,
                            "label": label,
                            "document_id": document_id,
                            "source": f"XFUND-{language}",
                            "split": split_name,
                        })

    if not train or not test:
        raise RuntimeError("Expected FUNSD train/test and/or XFUND train/val files under data/raw.")
    return train, test


def load_indian_examples(annotation_path: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not annotation_path.exists():
        raise RuntimeError(
            f"No Indian-form annotations found at {annotation_path}. Label official PDFs in the app first."
        )
    examples: list[dict[str, Any]] = []
    with annotation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            label = str(item.get("label", "")).strip()
            text = " ".join(str(item.get("text", "")).split())
            if label == "unlabeled":
                continue
            if label not in INDIAN_FORM_LABELS:
                raise ValueError(f"Unknown Indian-form label on annotation line {line_number}: {label}")
            if not text or not item.get("document_id"):
                raise ValueError(f"Annotation line {line_number} is missing text or document_id.")
            examples.append({
                "text": text,
                "label": label,
                "document_id": str(item["document_id"]),
                "source": str(item.get("source", "Indian public form")),
                "split": "annotated",
            })

    document_ids = sorted({item["document_id"] for item in examples})
    if len(document_ids) < 5:
        raise RuntimeError(
            f"Need labeled lines from at least 5 distinct Indian-form documents for train/validation/test "
            f"splits; found {len(document_ids)}. Add and label more official forms first."
        )
    random.Random(seed).shuffle(document_ids)
    test_count = max(1, round(len(document_ids) * 0.2))
    test_documents = set(document_ids[:test_count])
    train_examples = [item for item in examples if item["document_id"] not in test_documents]
    test_examples = [item for item in examples if item["document_id"] in test_documents]
    if len({item["label"] for item in train_examples}) < 2:
        raise RuntimeError("Indian-form training split needs at least two distinct labels.")
    return train_examples, test_examples


def split_train_documents(
    examples: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    document_ids = sorted({example["document_id"] for example in examples})
    random.Random(seed).shuffle(document_ids)
    validation_count = max(1, int(len(document_ids) * validation_fraction))
    validation_ids = set(document_ids[:validation_count])
    fit_examples = [item for item in examples if item["document_id"] not in validation_ids]
    validation_examples = [item for item in examples if item["document_id"] in validation_ids]
    return fit_examples, validation_examples


def _metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def save_classical_model(name: str, model: Pipeline, output_dir: Path) -> None:
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    artifact_dir = output_dir / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": name,
        "labels": [str(label) for label in classifier.classes_],
        "vectorizer": {
            "class": type(vectorizer).__name__,
            "analyzer": vectorizer.analyzer,
            "ngram_range": list(vectorizer.ngram_range),
            "lowercase": vectorizer.lowercase,
            "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        },
        "classifier": type(classifier).__name__,
    }
    arrays: dict[str, np.ndarray] = {}
    if isinstance(vectorizer, TfidfVectorizer):
        metadata["vectorizer"]["norm"] = vectorizer.norm
        metadata["vectorizer"]["use_idf"] = vectorizer.use_idf
        metadata["vectorizer"]["smooth_idf"] = vectorizer.smooth_idf
        arrays["idf"] = np.asarray(vectorizer.idf_, dtype=np.float64)
    if isinstance(classifier, LogisticRegression):
        arrays["coefficients"] = np.asarray(classifier.coef_, dtype=np.float64)
        arrays["intercept"] = np.asarray(classifier.intercept_, dtype=np.float64)
    elif isinstance(classifier, MultinomialNB):
        arrays["feature_log_prob"] = np.asarray(classifier.feature_log_prob_, dtype=np.float64)
        arrays["class_log_prior"] = np.asarray(classifier.class_log_prior_, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported classifier for safe persistence: {type(classifier).__name__}")
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(artifact_dir / "parameters.npz", **arrays)


def train_classical_models(
    fit_examples: list[dict[str, Any]], test_examples: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    train_text = [item["text"] for item in fit_examples]
    train_labels = [item["label"] for item in fit_examples]
    test_text = [item["text"] for item in test_examples]
    test_labels = [item["label"] for item in test_examples]
    results: dict[str, Any] = {}

    models: dict[str, Pipeline] = {
        "tfidf_logistic_regression": Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=180000)),
            ("classifier", LogisticRegression(max_iter=700, class_weight="balanced", C=2.0)),
        ]),
        "count_multinomial_naive_bayes": Pipeline([
            ("features", CountVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=180000)),
            ("classifier", MultinomialNB(alpha=0.2)),
        ]),
    }
    for name, model in models.items():
        model.fit(train_text, train_labels)
        predictions = model.predict(test_text).tolist()
        results[name] = _metrics(test_labels, predictions)
        save_classical_model(name, model, output_dir)
    return results


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class LineDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], vocabulary: dict[str, int], max_tokens: int):
        self.sequences: list[torch.Tensor] = []
        self.labels: list[int] = []
        for example in examples:
            tokens = tokenize(example["text"])[:max_tokens]
            ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokens] or [vocabulary["<unk>"]]
            self.sequences.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(LABEL_TO_ID[example["label"]])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.sequences[index], self.labels[index]


def collate_lines(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class ScratchBiLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, class_count: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, class_count)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        outputs, _ = pad_packed_sequence(encoded, batch_first=True)
        mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths.to(outputs.device)[:, None]
        pooled = (outputs * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(outputs.device).clamp(min=1).unsqueeze(1)
        return self.classifier(self.dropout(pooled))


class ScratchTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        class_count: int,
        max_tokens: int,
        model_dim: int = 128,
        head_count: int = 4,
        layer_count: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, model_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_tokens, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=head_count,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layer_count)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(model_dim, class_count)
        self.max_tokens = max_tokens

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device).expand(batch_size, -1)
        mask = positions < lengths.to(token_ids.device)[:, None]
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=~mask)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(encoded.device).clamp(min=1).unsqueeze(1)
        return self.classifier(self.dropout(self.norm(pooled)))


def _predict_neural(model: nn.Module, loader: DataLoader, device: torch.device) -> list[str]:
    model.eval()
    predicted: list[str] = []
    with torch.no_grad():
        for token_ids, lengths, _ in loader:
            logits = model(token_ids.to(device), lengths)
            predicted.extend(LABELS[index] for index in logits.argmax(dim=1).cpu().tolist())
    return predicted


def train_scratch_neural(
    architecture: str,
    fit_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    max_tokens: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    vocabulary = {"<pad>": 0, "<unk>": 1}
    token_counts = Counter(token for example in fit_examples for token in tokenize(example["text"]))
    for token, _ in token_counts.most_common():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    train_data = LineDataset(fit_examples, vocabulary, max_tokens)
    validation_data = LineDataset(validation_examples, vocabulary, max_tokens)
    test_data = LineDataset(test_examples, vocabulary, max_tokens)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_lines,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_lines)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_lines)

    if architecture == "bilstm":
        model: nn.Module = ScratchBiLSTM(len(vocabulary), 96, 96, len(LABELS), 0.25)
        model_config = {"embedding_dim": 96, "hidden_dim": 96, "dropout": 0.25}
    elif architecture == "transformer":
        model = ScratchTransformer(len(vocabulary), len(LABELS), max_tokens)
        model_config = {"model_dim": 128, "head_count": 4, "layer_count": 2, "dropout": 0.2}
    else:
        raise ValueError(f"Unsupported scratch architecture: {architecture}")
    model = model.to(device)
    counts = Counter(LABEL_TO_ID[example["label"]] for example in fit_examples)
    weights = torch.tensor(
        [len(fit_examples) / (len(LABELS) * max(1, counts[index])) for index in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_f1 = -1.0
    patience = 3
    stale_epochs = 0
    training_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for token_ids, lengths, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, lengths)
            loss = loss_function(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach())

        validation_predictions = _predict_neural(model, validation_loader, device)
        validation_labels = [item["label"] for item in validation_examples]
        validation_scores = _metrics(validation_labels, validation_predictions)
        mean_loss = running_loss / max(1, len(train_loader))
        training_history.append({
            "epoch": float(epoch),
            "train_loss": mean_loss,
            "validation_macro_f1": validation_scores["f1_macro"],
        })
        print(
            f"{architecture} epoch {epoch}/{epochs}: loss={mean_loss:.4f} "
            f"validation_macro_f1={validation_scores['f1_macro']:.4f} device={device}"
        )
        if validation_scores["f1_macro"] > best_validation_f1:
            best_validation_f1 = validation_scores["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("BiLSTM training produced no checkpoint.")
    model.load_state_dict(best_state)
    test_predictions = _predict_neural(model, test_loader, device)
    test_labels = [item["label"] for item in test_examples]
    checkpoint_dir = output_dir / f"scratch_{architecture}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(checkpoint_dir / "model.safetensors"),
    )
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "architecture": architecture,
            "vocabulary": vocabulary,
            "labels": LABELS,
            "max_tokens": max_tokens,
            "trained_from_random_initialization": True,
            **model_config,
        }, handle, ensure_ascii=False)
    return {
        **_metrics(test_labels, test_predictions),
        "device": str(device),
        "epochs_completed": len(training_history),
        "best_validation_macro_f1": best_validation_f1,
        "history": training_history,
    }


def training_signature(
    task: str,
    train_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
    epochs: int,
    batch_size: int,
    seed: int,
) -> str:
    payload = {
        "task": task,
        "labels": LABELS,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "train": sorted((item["document_id"], item["label"], item["text"]) for item in train_examples),
        "test": sorted((item["document_id"], item["label"], item["text"]) for item in test_examples),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def has_complete_artifacts(output_dir: Path) -> bool:
    required = [
        output_dir / "tfidf_logistic_regression" / "metadata.json",
        output_dir / "tfidf_logistic_regression" / "parameters.npz",
        output_dir / "count_multinomial_naive_bayes" / "metadata.json",
        output_dir / "count_multinomial_naive_bayes" / "parameters.npz",
        output_dir / "scratch_bilstm" / "metadata.json",
        output_dir / "scratch_bilstm" / "model.safetensors",
        output_dir / "scratch_transformer" / "metadata.json",
        output_dir / "scratch_transformer" / "model.safetensors",
        output_dir / "metrics.json",
    ]
    return all(path.is_file() for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("form-role", "indian"), default="form-role")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true", help="Retrain even if a matching complete run exists.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")

    labels = FORM_ROLE_LABELS if args.task == "form-role" else INDIAN_FORM_LABELS
    set_labels(labels)
    output_dir = args.output_dir or ROOT / "models" / "training" / (
        "form_role" if args.task == "form-role" else "indian_forms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "form-role":
        train_examples, test_examples = load_training_examples()
    else:
        train_examples, test_examples = load_indian_examples(
            ROOT / "data" / "annotations" / "indian_form_lines.jsonl", args.seed
        )
    fit_examples, validation_examples = split_train_documents(
        train_examples, args.seed, args.validation_fraction
    )
    signature = training_signature(
        args.task, fit_examples, test_examples, args.epochs, args.batch_size, args.seed
    )
    manifest_path = output_dir / "manifest.json"
    if not args.force and manifest_path.exists() and has_complete_artifacts(output_dir):
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("signature") == signature:
            print(f"Matching completed run already exists at {output_dir}; skipping training.")
            print(f"Saved metrics: {output_dir / 'metrics.json'}")
            return

    print(f"Training examples: {len(fit_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Held-out test examples: {len(test_examples)}")
    print(f"Training documents: {len({item['document_id'] for item in fit_examples})}")
    print(f"Validation documents: {len({item['document_id'] for item in validation_examples})}")
    print(f"Test documents: {len({item['document_id'] for item in test_examples})}")
    print(f"Scratch neural model device: {device}")

    results: dict[str, Any] = {
        "task": (
            "generic multilingual form-role line classification"
            if args.task == "form-role"
            else "Indian government form line classification"
        ),
        "labels": LABELS,
        "warning": (
            "Generic benchmark scores are not Indian-government-form accuracy."
            if args.task == "form-role"
            else "Indian-form scores apply only to the annotated documents and source forms in this split."
        ),
        "training_examples": len(fit_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "training_label_counts": dict(Counter(item["label"] for item in fit_examples)),
        "validation_label_counts": dict(Counter(item["label"] for item in validation_examples)),
        "test_label_counts": dict(Counter(item["label"] for item in test_examples)),
        "test_source_counts": dict(Counter(item["source"] for item in test_examples)),
        "models": {},
    }
    results["models"].update(train_classical_models(fit_examples, test_examples, output_dir))
    for architecture in ("bilstm", "transformer"):
        results["models"][f"scratch_{architecture}"] = train_scratch_neural(
            architecture,
            fit_examples,
            validation_examples,
            test_examples,
            output_dir,
            device,
            args.epochs,
            args.batch_size,
            args.seed,
        )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "signature": signature,
            "task": args.task,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device_used_for_neural_models": str(device),
            "checkpoint_format": "safetensors + JSON + NPZ; no pickle required",
        }, handle, indent=2)
    print(f"Metrics and checkpoints saved to: {output_dir}")
    for model_name, metrics in results["models"].items():
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f} "
            f"macro_precision={metrics['precision_macro']:.4f} "
            f"macro_recall={metrics['recall_macro']:.4f} "
            f"macro_f1={metrics['f1_macro']:.4f} "
            f"weighted_f1={metrics['f1_weighted']:.4f}"
        )


if __name__ == "__main__":
    main()
