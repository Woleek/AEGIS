import sys
from ..config import ROOT_DIR, MODELS_DIR

sys.path.append((ROOT_DIR.resolve() / "insightface").as_posix())
import torch
from recognition.arcface_torch.backbones import get_model
from insightface.utils.face_align import estimate_norm
from torchvision import transforms
from ..models.base import (
    chunk_sequence,
    BaseEmbedder,
    load_insightface_detector,
    warp_affine_pytorch,
)
import cv2
import insightface
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import os
from pathlib import Path
from typing import Dict, List, Literal, Sequence


class ArcFaceEmbedder(BaseEmbedder):
    """Wraps InsightFace ArcFace models for feature extraction."""

    def __init__(self, device: Literal["cpu", "cuda"], batch_size: int) -> None:
        self.device = device
        self.batch_size = batch_size
        self.ctx_id = 0 if device == "cuda" else -1
        # Quieten ONNX Runtime warnings to keep logs clean.
        ort.set_default_logger_severity(3)
        self.detect_model, self.recogn_model = self._load_models()

    def _load_models(self):
        detect_model = load_insightface_detector(self.ctx_id)
        recog_model = insightface.model_zoo.get_model(
            os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
        )
        recog_model.prepare(ctx_id=self.ctx_id)
        return detect_model, recog_model

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def embed(self, paths: Sequence[Path], key_fn) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        for chunk in tqdm(
            chunk_sequence(paths, self.batch_size),
            desc="Embedding images with ArcFace",
            total=len(paths) // self.batch_size + 1,
        ):
            images: List[np.ndarray] = []
            valid_keys: List[str] = []
            for path in chunk:
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                if img is None:
                    continue
                try:
                    bboxes, kpss = self.detect_model.detect(img, max_num=1)
                    if kpss is None or len(kpss) == 0:
                        crop = img
                    else:
                        crop = insightface.utils.face_align.norm_crop(
                            img, landmark=kpss[0]
                        )
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    crop = img
                images.append(crop)
                valid_keys.append(key_fn(path))
            if not images:
                continue
            feats = self.recogn_model.get_feat(images)
            for key, feat in zip(valid_keys, feats):
                embeddings[key] = self._norm(feat)
        return embeddings


class ArcFace(ArcFaceEmbedder):
    def _load_models(self):
        detect_model = load_insightface_detector(self.ctx_id)
        recog_model = get_model("r50", dropout=0.0, fp16=False, num_features=512)
        recog_model.load_state_dict(torch.load(MODELS_DIR / "arcface_ir50_ms1mv3.pth"))
        recog_model.eval()
        recog_model.to(self.device)
        return detect_model, recog_model

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
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
            raise ValueError("No face detected in the image for AdaFace embedding.")
        return crop

    def _prepare_tensor(self, aligned_image: torch.Tensor) -> torch.Tensor:
        tensor = aligned_image.float()
        tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(
            tensor
        ).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)

    @staticmethod
    def _norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True)
        if norm == 0:
            return vec
        return vec / norm

    # ArcFace expects (N, C, H, W) in [-1, 1]
    def embed(self, image: torch.Tensor) -> torch.Tensor:
        aligned = self._align_face(image)
        tensor = self._prepare_tensor(aligned)
        emb = self.recogn_model(tensor)
        emb = self._norm(emb)
        return emb

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
