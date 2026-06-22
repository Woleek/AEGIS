"""CosFace embedder implementations for identity masking and evaluation.

This module provides two embedder classes following the same pattern as SwinFace/TransFace:
- CosFaceEmbedder: For batch evaluation with numpy arrays
- CosFace: For single-image attack scenarios with PyTorch tensors

CosFace paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition (CVPR 2018)
Architecture: IR-SE50 backbone trained with LMCL loss, input 112x112, output 512-dim embeddings.
"""

from pathlib import Path
from typing import Dict, List, Literal, Sequence

import cv2
import insightface
from insightface.utils.face_align import estimate_norm
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from .base import (
    BaseEmbedder,
    chunk_sequence,
    load_insightface_detector,
    resolve_compute_device,
    warp_affine_pytorch,
)
from .cosface import load_cosface_model
from ..config import MODELS_DIR

_FACE_SIZE = 112


class CosFaceTorchModel(nn.Module):
    """PyTorch wrapper for CosFace (IR-SE50 Backbone) model."""

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
        device: str = "cuda",
    ):
        super(CosFaceTorchModel, self).__init__()
        self.device = device
        self._prepare_model(checkpoint_path, freeze, device)

    def _prepare_model(self, checkpoint_path: str, freeze: bool, device: str):
        self.model = load_cosface_model(checkpoint_path, device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.model.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized face embeddings.

        Args:
            images: Tensor of shape [B, 3, 112, 112] with pixel values in [-1, 1].

        Returns:
            Tensor of shape [B, 512] with L2-normalized embeddings.
        """
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")
        return self.model(images)

    def preprocess_face(self, face_crop: torch.Tensor) -> torch.Tensor:
        """Preprocess face crop for CosFace model.

        Args:
            face_crop: Tensor with pixel values in [0, 255], shape [C, H, W].

        Returns:
            Normalized tensor with values in [-1, 1], shape [C, 112, 112].
        """
        face = face_crop.float() / 255.0
        face = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(face)
        return face


class CosFaceEmbedder(BaseEmbedder):
    """CosFace embedder for batch evaluation.

    Handles batch processing of images for evaluation, including face detection
    and alignment using InsightFace. Uses 112x112 face crops.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        batch_size: int,
        model_path: Path | None = None,
    ) -> None:
        """Initialize CosFace embedder.

        Args:
            device: Device to run inference on ('cpu' or 'cuda').
            batch_size: Batch size for processing.
            model_path: Path to checkpoint file. If None, uses default path.
        """
        self.requested_device = device
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.batch_size = batch_size
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        ort.set_default_logger_severity(3)

        self.detect_model = load_insightface_detector(self.ctx_id)

        if model_path is None:
            model_path = MODELS_DIR / "cosface" / "cosface_ir50_ms1mv2.pth"
        self.model = CosFaceTorchModel(
            checkpoint_path=str(model_path),
            device=self.device_str,
        )
        self.model.eval()

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _align_face(self, image: np.ndarray) -> np.ndarray:
        """Detect and align face in image.

        Args:
            image: RGB image as numpy array.

        Returns:
            Aligned face crop of size 112x112.
        """
        try:
            bboxes, kpss = self.detect_model.detect(image, max_num=1)
            if kpss is not None and len(kpss) > 0:
                return insightface.utils.face_align.norm_crop(image, landmark=kpss[0])
        except Exception as e:
            print(f"Error aligning face: {e}")
        return cv2.resize(image, (_FACE_SIZE, _FACE_SIZE), interpolation=cv2.INTER_LINEAR)

    def _prepare_tensor(self, aligned_image: np.ndarray) -> torch.Tensor:
        """Prepare aligned image for model input.

        Args:
            aligned_image: Aligned face crop as numpy array (112x112x3).

        Returns:
            Preprocessed tensor ready for model input.
        """
        rgb = aligned_image.copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = self.model.preprocess_face(tensor)
        return tensor

    def embed(self, paths: Sequence[Path], key_fn) -> Dict[str, np.ndarray]:
        """Embed a batch of images.

        Args:
            paths: Sequence of image file paths.
            key_fn: Function to extract key from path.

        Returns:
            Dictionary mapping keys to normalized embeddings.
        """
        embeddings: Dict[str, np.ndarray] = {}
        for chunk in tqdm(
            chunk_sequence(paths, self.batch_size),
            desc="Embedding images with CosFace",
            total=len(paths) // self.batch_size + 1,
        ):
            batch_tensors: List[torch.Tensor] = []
            valid_keys: List[str] = []
            for path in chunk:
                image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                if image is None:
                    continue
                aligned = self._align_face(image)
                tensor = self._prepare_tensor(aligned)
                batch_tensors.append(tensor)
                valid_keys.append(key_fn(path))
            if not batch_tensors:
                continue
            batch = torch.stack(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                features = self.model(batch).detach().cpu().numpy()
            for key, feat in zip(valid_keys, features):
                embeddings[key] = self._norm(feat)
        return embeddings


class CosFace(CosFaceEmbedder):
    """CosFace embedder for single-image attack scenarios.

    Processes PyTorch tensors for use in iterative adversarial attacks,
    maintaining gradient flow.
    """

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        """Detect and align face in a PyTorch tensor.

        Args:
            image: Image tensor of shape (H, W, C) with values in [0, 1].

        Returns:
            Aligned face crop tensor of shape (C, 112, 112).

        Raises:
            ValueError: If no face is detected.
        """
        img_arr = image.detach().cpu().numpy() * 255
        bboxes, kpss = self.detect_model.detect(img_arr, max_num=1)
        if kpss is not None and len(kpss) > 0:
            M = estimate_norm(kpss[0], image_size=_FACE_SIZE)
            crop = warp_affine_pytorch(
                image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
                m_matrix=torch.from_numpy(M).unsqueeze(0).to(image.device).float(),
                out_size=(_FACE_SIZE, _FACE_SIZE),
            ).squeeze(0)
        else:
            raise ValueError("No face detected in the image for CosFace embedding.")
        return crop

    def _prepare_tensor(self, aligned_image: torch.Tensor) -> torch.Tensor:
        """Prepare aligned tensor for model input.

        Args:
            aligned_image: Aligned face crop tensor of shape (C, H, W) in [0, 1].

        Returns:
            Preprocessed tensor ready for model input with batch dimension.
        """
        tensor = aligned_image.float() * 255.0
        tensor = self.model.preprocess_face(tensor).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _norm(vec: torch.Tensor) -> torch.Tensor:
        """L2 normalize a tensor."""
        norm = torch.norm(vec, p=2, dim=1, keepdim=True)
        if norm == 0:
            return vec
        return vec / norm

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a single image tensor.

        Args:
            image: Image tensor of shape (H, W, C) with values in [0, 1].

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        aligned = self._align_face(image)
        tensor = self._prepare_tensor(aligned)
        emb = self.model(tensor)
        emb = self._norm(emb)
        return emb

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Call embed method."""
        return self.embed(image)
