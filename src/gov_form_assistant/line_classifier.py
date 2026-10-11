from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> dict[str, int]:
    lowered = text.lower()
    counts: dict[str, int] = {}
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for index in range(max(0, len(lowered) - n + 1)):
            gram = lowered[index : index + n]
            counts[gram] = counts.get(gram, 0) + 1
    return counts


class TfidfLogisticLineClassifier:
    """Load the JSON/NPZ checkpoint written by train_form_role_models.py."""

    def __init__(self, artifact_dir: Path):
        with (artifact_dir / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        arrays = np.load(artifact_dir / "parameters.npz")
        self.labels: list[str] = list(metadata["labels"])
        vectorizer = metadata["vectorizer"]
        self.vocabulary: dict[str, int] = {str(k): int(v) for k, v in vectorizer["vocabulary"].items()}
        self.ngram_range = tuple(vectorizer["ngram_range"])
        self.idf = arrays["idf"]
        self.coefficients = arrays["coefficients"]
        self.intercept = arrays["intercept"]

    def _vectorize(self, text: str) -> np.ndarray:
        counts = _char_ngrams(text, self.ngram_range)
        size = len(self.idf)
        vector = np.zeros(size, dtype=np.float64)
        for gram, count in counts.items():
            index = self.vocabulary.get(gram)
            if index is None:
                continue
            vector[index] = float(count)
        nonzero = vector > 0
        vector[nonzero] *= self.idf[nonzero]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def predict(self, text: str) -> str:
        features = self._vectorize(text)
        scores = features @ self.coefficients.T + self.intercept
        return self.labels[int(np.argmax(scores))]


@lru_cache(maxsize=2)
def load_line_classifier(root: str, task: str) -> TfidfLogisticLineClassifier | None:
    artifact = Path(root) / "models" / "training" / task / "tfidf_logistic_regression"
    if not (artifact / "metadata.json").is_file():
        return None
    return TfidfLogisticLineClassifier(artifact)
