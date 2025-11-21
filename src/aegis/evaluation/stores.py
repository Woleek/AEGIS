import json
import numpy as np


import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .datasets import FaceDataset
from ..models import BaseEmbedder
from ..utils import generate_key


@dataclass
class EmbeddingStore:
    """Utility container for caching embeddings on disk."""

    cache_file: Path

    def load(self) -> Optional[Dict[str, np.ndarray]]:
        if self.cache_file.exists():
            with self.cache_file.open("rb") as handle:
                data = pickle.load(handle)
            if isinstance(data, dict):
                return data
        return None

    def save(self, embeddings: Dict[str, np.ndarray]) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("wb") as handle:
            pickle.dump(embeddings, handle)


@dataclass
class ThresholdCacheStore:
    """Persist fitted verification thresholds along with diagnostics."""

    cache_file: Path

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.cache_file.exists():
            return None
        with self.cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, payload: Dict[str, Any]) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


@dataclass
class VerificationThresholdResult:
    dataset: str
    embedder: str
    distance_threshold: float
    eer: float
    eer_score_threshold: float
    far: float
    frr: float
    roc_auc: float
    cache_file: Path
    plot_path: Path


def load_embeddings(
    embedder: BaseEmbedder,
    dataset: FaceDataset,
    dataset_root: Path,
    cache_path: Path,
    key_prefix: Optional[str] = None,
    load_from_cache: bool = True,
) -> Dict[str, np.ndarray]:
    store = EmbeddingStore(cache_path)
    cached = store.load()
    if cached is not None and load_from_cache:
        return cached
    embeddings = embedder.embed(
        dataset.paths, lambda p: generate_key(dataset_root, p, prefix=key_prefix)
    )
    store.save(embeddings)
    return embeddings
