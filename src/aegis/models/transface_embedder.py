"""TransFace embedder implementations for identity masking and evaluation.

This module provides two embedder classes following the same pattern as AdaFace/ArcFace/SwinFace:
- TransFaceEmbedder: For batch evaluation with numpy arrays
- TransFace: For single-image attack scenarios with PyTorch tensors

TransFace paper: Calibrating Transformer Training for Face Recognition (ICCV 2023)
Repository: https://github.com/DanJun6737/TransFace
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
from .transface import get_model
from ..config import MODELS_DIR


class TransFaceTorchModel(nn.Module):
    """PyTorch wrapper for TransFace model."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_s",
        freeze: bool = True,
        device: str = "cuda",
    ):
        super(TransFaceTorchModel, self).__init__()
        self.device = device
        self.model_type = model_type
        self._prepare_model(checkpoint_path, freeze, device)

    def _prepare_model(self, checkpoint_path: str, freeze: bool, device: str):
        # Create model architecture
        self.model = get_model(self.model_type)

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=True,
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.model.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized face embeddings.

        Args:
            images: Tensor of shape [B, 3, 112, 112] with pixel values in [-1, 1]

        Returns:
            Tensor of shape [B, 512] with L2-normalized embeddings
        """
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")

        embeddings = self.model(images)

        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def preprocess_face(self, face_crop: torch.Tensor) -> torch.Tensor:
        """Preprocess face crop for TransFace model.

        Args:
            face_crop: Tensor with pixel values in [0, 255]

        Returns:
            Normalized tensor with values in [-1, 1]
        """
        face = face_crop.float() / 255.0
        face = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(face)
        return face


class TransFaceEmbedder(BaseEmbedder):
    """TransFace embedder for batch evaluation.

    This class handles batch processing of images for evaluation,
    including face detection and alignment using InsightFace.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        batch_size: int,
        model_path: Path | None = None,
        model_type: str = "vit_s",
    ) -> None:
        """Initialize TransFace embedder.

        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            batch_size: Batch size for processing
            model_path: Path to checkpoint file. If None, uses default path.
            model_type: Model variant (vit_t, vit_s, vit_b, vit_l)
        """
        self.requested_device = device
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.batch_size = batch_size
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        # Silence ONNX Runtime warnings
        ort.set_default_logger_severity(3)

        # Load face detector
        self.detect_model = load_insightface_detector(self.ctx_id)

        # Load TransFace model
        if model_path is None:
            model_path = MODELS_DIR / "transface" / "transface_s_ms1mv2.pt"

        self.model = TransFaceTorchModel(
            checkpoint_path=str(model_path),
            model_type=model_type,
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
            image: RGB image as numpy array

        Returns:
            Aligned face crop of size 112x112
        """
        try:
            bboxes, kpss = self.detect_model.detect(image, max_num=1)
            if kpss is not None and len(kpss) > 0:
                return insightface.utils.face_align.norm_crop(image, landmark=kpss[0])
        except Exception as e:
            print(f"Error aligning face: {e}")
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _prepare_tensor(self, aligned_image: np.ndarray) -> torch.Tensor:
        """Prepare aligned image for model input.

        Args:
            aligned_image: Aligned face crop as numpy array (112x112x3)

        Returns:
            Preprocessed tensor ready for model input
        """
        rgb = aligned_image.copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = self.model.preprocess_face(tensor)
        return tensor

    def embed(self, paths: Sequence[Path], key_fn) -> Dict[str, np.ndarray]:
        """Embed a batch of images.

        Args:
            paths: Sequence of image file paths
            key_fn: Function to extract key from path

        Returns:
            Dictionary mapping keys to normalized embeddings
        """
        embeddings: Dict[str, np.ndarray] = {}
        for chunk in tqdm(
            chunk_sequence(paths, self.batch_size),
            desc="Embedding images with TransFace",
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


class TransFace(TransFaceEmbedder):
    """TransFace embedder for single-image attack scenarios.

    This class processes PyTorch tensors for use in iterative
    adversarial attacks, maintaining gradient flow.
    """

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        """Detect and align face in a PyTorch tensor.

        Args:
            image: Image tensor of shape (H, W, C) with values in [0, 1]

        Returns:
            Aligned face crop tensor of shape (C, H, W)

        Raises:
            ValueError: If no face is detected
        """
        img_arr = image.detach().cpu().numpy() * 255
        bboxes, kpss = self.detect_model.detect(img_arr, max_num=1)
        if kpss is not None and len(kpss) > 0:
            M = estimate_norm(kpss[0], image_size=112)
            crop = warp_affine_pytorch(
                image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
                m_matrix=torch.from_numpy(M).unsqueeze(0).to(image.device).float(),
                out_size=(112, 112),
            ).squeeze(0)
        else:
            raise ValueError("No face detected in the image for TransFace embedding.")
        return crop

    def _prepare_tensor(self, aligned_image: torch.Tensor) -> torch.Tensor:
        """Prepare aligned tensor for model input.

        Args:
            aligned_image: Aligned face crop tensor of shape (C, H, W) in [0, 1]

        Returns:
            Preprocessed tensor ready for model input with batch dimension
        """
        tensor = aligned_image.float() * 255.0  # Scale to [0, 255]
        tensor = self.model.preprocess_face(tensor).unsqueeze(0)  # Add batch dimension
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
            image: Image tensor of shape (H, W, C) with values in [0, 1]

        Returns:
            Normalized embedding tensor of shape (1, 512)
        """
        aligned = self._align_face(image)
        tensor = self._prepare_tensor(aligned)
        emb = self.model(tensor)
        emb = self._norm(emb)
        return emb

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Call embed method."""
        return self.embed(image)
