import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence
import torch
from PIL import Image
from torchvision import transforms


def load_image_from_file(file_path: str) -> torch.Tensor:
    """
    Load an image from a file and preprocess it for the verification system.
    """
    image = Image.open(file_path).convert("RGB")  # (H, W, C)
    image = transforms.ToTensor()(image)  # (C, H, W)
    return image


def generate_key(root: Path, path: Path, prefix: Optional[str] = None) -> str:
    relative = path.relative_to(root)
    key = str(relative).replace(os.sep, "/")
    if prefix:
        return f"{prefix}___{key}"
    return key


def load_image_map(
    paths: Sequence[str],
    root: str,
    *,
    key_prefix: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Read image files into memory keyed by dataset-relative identifier."""

    final_paths = []
    images: Dict[str, np.ndarray] = {}
    for path in paths:
        path = Path(root) / path
        final_paths.append(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to read image '{path}', skipping.")
            continue
        key = generate_key(root, path, prefix=key_prefix)
        images[key] = image
    return images, final_paths


def ensure_csv_parent(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
