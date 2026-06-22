"""Disk-based evaluation adapter bridging registry FaceEmbedders to the
`embed(paths, key_fn) -> dict[str, np.ndarray]` interface used by the
evaluation pipeline (src/aegis/evaluation/stores.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .base import BaseEmbedder, FaceEmbedder, chunk_sequence
from .registry import get_verification_model


def _load_hwc_tensor(path: Path) -> torch.Tensor | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float() / 255.0  # HWC in [0,1]


class EvalEmbedderAdapter(BaseEmbedder):
    """Wrap a registry FaceEmbedder for disk-batch evaluation."""

    def __init__(self, model: FaceEmbedder, batch_size: int = 8, desc: str | None = None) -> None:
        self.model = model
        self.batch_size = max(1, batch_size)
        self.desc = desc or f"Embedding images with {getattr(getattr(model, 'spec', None), 'display_name', 'model')}"

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec if n == 0 else vec / n

    def embed(self, paths: Sequence[Path], key_fn: Callable[[Path], str]) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        total = len(paths) // self.batch_size + 1
        for chunk in tqdm(chunk_sequence(paths, self.batch_size), desc=self.desc, total=total):
            images, keys = [], []
            for path in chunk:
                t = _load_hwc_tensor(Path(path))
                if t is None:
                    continue
                images.append(t)
                keys.append(key_fn(path))
            if not images:
                continue
            with torch.no_grad():
                outs = self.model.embed_batch(images)
            for key, emb in zip(keys, outs):
                if emb is None:
                    continue
                vec = emb.squeeze(0).detach().cpu().numpy()
                embeddings[key] = self._norm(vec)
        return embeddings


def get_eval_embedder(
    model_name: str,
    device: torch.device | str,
    batch_size: int = 8,
    variant: str | None = None,
) -> EvalEmbedderAdapter:
    """Construct a registry model and wrap it for disk-batch evaluation."""

    model = get_verification_model(model_name, device, variant)
    return EvalEmbedderAdapter(model, batch_size=batch_size)
