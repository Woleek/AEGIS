import os
import hashlib
import insightface
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import torch
from typing import Any, Dict, Iterator, Literal, Protocol, Sequence, Tuple, cast, runtime_checkable

import torch.nn.functional as F

# Load the CUDA/cuDNN shared libraries bundled with torch's nvidia-* wheels so
# the onnxruntime-gpu CUDAExecutionProvider can initialise without a system CUDA
# install or a manually-set LD_LIBRARY_PATH. Must run before any InsightFace
# InferenceSession is created
try:  # pragma: no cover - environment dependent
    import onnxruntime as _ort

    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
except Exception:
    pass


def resolve_compute_device(requested: Literal["cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    """Return a usable device string, falling back to CPU when CUDA is unavailable."""

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def load_insightface_detector(ctx_id: int):
    """Load the InsightFace face detector used for alignment."""

    insightface.utils.ensure_available("models", "buffalo_l", root="~/.insightface")
    detect_model = insightface.model_zoo.get_model(
        os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
    )
    det_size = (640, 640)
    detect_model.prepare(ctx_id=ctx_id, det_size=det_size, input_size=det_size)
    return detect_model


class BaseEmbedder:
    """Interface for feature embedders used throughout the evaluation pipeline."""

    def embed(
        self, paths: Sequence[Path], key_fn
    ) -> Dict[str, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError


def chunk_sequence(data: Sequence[Path], chunk_size: int) -> Iterator[Sequence[Path]]:
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


def warp_affine_pytorch(
    image_tensor: torch.Tensor,
    m_matrix: torch.Tensor,
    out_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Apply OpenCV-style affine transformation to a PyTorch tensor.
    This function provides a PyTorch-native equivalent of cv2.warpAffine.

    Args:
        image_tensor: Input image batch as a tensor with shape (N, C, H_in, W_in).
        m_matrix: Batch of OpenCV-style 2x3 affine matrices (N, 2, 3).
        out_size: The target output size as a (H_out, W_out) tuple.
                  Note: cv2.warpAffine takes (W, H), but this function
                  follows the PyTorch (H, W) convention.

    Returns:
        The warped image tensor with shape (N, C, H_out, W_out).
    """
    N, C, H_in, W_in = image_tensor.shape
    H_out, W_out = out_size
    device = image_tensor.device

    # Augment and Invert M
    m_aug = torch.cat(
        [m_matrix, torch.zeros(N, 1, 3, device=device, dtype=m_matrix.dtype)],
        dim=1,
    )
    m_aug[:, 2, 2] = 1.0

    # (N, 3, 3) -> (N, 3, 3)
    try:
        m_inv = torch.inverse(m_aug)
    except torch._C.LinAlgError as e:
        print(f"Error inverting matrix: {e}")
        print("M matrix may be singular. Using pseudo-inverse instead.")
        m_inv = torch.linalg.pinv(m_aug)

    m_inv = m_inv[:, :2, :]  # (N, 2, 3)

    # Build the `theta` matrix for F.affine_grid
    # `m_inv` maps pixel coordinates in the output image to pixel
    # coordinates in the input image.
    # `F.affine_grid` requires `theta` to map normalized coordinates
    # in the output to normalized coordinates in the input.

    # Get components [a, b, c] and [d, e, f]
    # (N, 1)
    a = m_inv[:, 0, 0].unsqueeze(1)
    b = m_inv[:, 0, 1].unsqueeze(1)
    c = m_inv[:, 0, 2].unsqueeze(1)
    d = m_inv[:, 1, 0].unsqueeze(1)
    e = m_inv[:, 1, 1].unsqueeze(1)
    f = m_inv[:, 1, 2].unsqueeze(1)

    # Convert scalar sizes to float tensors for broadcasting
    W_s_f = torch.tensor(W_in, dtype=m_matrix.dtype, device=device)
    H_s_f = torch.tensor(H_in, dtype=m_matrix.dtype, device=device)
    W_d_f = torch.tensor(W_out, dtype=m_matrix.dtype, device=device)
    H_d_f = torch.tensor(H_out, dtype=m_matrix.dtype, device=device)

    # Map pixel centers to normalized coordinates [-1, 1]
    theta = torch.zeros(N, 2, 3, device=device, dtype=m_matrix.dtype)

    theta[:, 0, 0] = a * W_d_f / W_s_f
    theta[:, 0, 1] = b * H_d_f / W_s_f
    theta[:, 0, 2] = (
        (2.0 * a * (W_d_f / 2.0 - 0.5) + 2.0 * b * (H_d_f / 2.0 - 0.5) + 2.0 * c)
        / W_s_f
        + (1.0 / W_s_f)
        - 1.0
    )

    theta[:, 1, 0] = d * W_d_f / H_s_f
    theta[:, 1, 1] = e * H_d_f / H_s_f
    theta[:, 1, 2] = (
        (2.0 * d * (W_d_f / 2.0 - 0.5) + 2.0 * e * (H_d_f / 2.0 - 0.5) + 2.0 * f)
        / H_s_f
        + (1.0 / H_s_f)
        - 1.0
    )

    # Create Grid and Sample
    grid = F.affine_grid(theta, size=(N, C, H_out, W_out), align_corners=False)

    # Sample the image using the grid
    aligned_image = F.grid_sample(
        image_tensor,
        grid,
        mode="bilinear",  # Equivalent to cv2.INTER_LINEAR
        padding_mode="zeros",  # Equivalent to borderValue=0.0
        align_corners=False,
    )

    return aligned_image


class FaceVerificationError(RuntimeError):
    """Base class for face verification failures."""


class FaceNotDetectedError(FaceVerificationError):
    """Raised when a face cannot be detected or aligned."""


class ModelAssetMissingError(FaceVerificationError):
    """Raised when required model files are unavailable."""


class UnsupportedModelVariantError(FaceVerificationError):
    """Raised when a requested model variant is not registered."""


@dataclass(frozen=True)
class ModelAssetSpec:
    """A required runtime asset for a face verification model."""

    key: str
    path: Path
    description: str
    source_url: str | None = None
    sha256: str | None = None
    install_hint: str | None = None
    auto_download: bool = False
    alt_paths: tuple[Path, ...] = field(default_factory=tuple)

    def expanded_path(self) -> Path:
        return Path(os.path.expanduser(self.path.as_posix()))

    def expanded_alt_paths(self) -> tuple[Path, ...]:
        return tuple(Path(os.path.expanduser(path.as_posix())) for path in self.alt_paths)

    def candidate_paths(self) -> tuple[Path, ...]:
        return (self.expanded_path(), *self.expanded_alt_paths())


@dataclass(frozen=True)
class VerificationModelSpec:
    """Declarative metadata for a face verification model variant."""

    name: str
    variant: str
    display_name: str
    embedding_dim: int
    input_size: tuple[int, int]
    normalization: str
    detector: str | None = None
    alignment: str | None = None
    threshold: float | None = None
    assets: tuple[ModelAssetSpec, ...] = field(default_factory=tuple)
    requirements: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self.variant}"


@runtime_checkable
class FaceEmbedder(Protocol):
    """Runtime contract expected by the attack and verification pipelines."""

    spec: VerificationModelSpec
    device: torch.device

    def embed(self, image: torch.Tensor) -> torch.Tensor: ...

    def embed_batch(
        self,
        images: list[torch.Tensor],
        detections: "list[FaceDetection | None] | None" = None,
    ) -> list[torch.Tensor | None]: ...

    def validate_assets(self) -> None: ...

    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


class FaceDetector(Protocol):
    """Minimal detector interface used by the FR wrappers."""

    def prepare(self, ctx_id: int, **kwargs: Any) -> None: ...

    def detect(
        self,
        img: Any,
        input_size: Any | None = None,
        max_num: int = 0,
        metric: str = "default",
    ) -> tuple[Any, Any | None]: ...


def get_detector_asset_path(package_name: str = "buffalo_l", filename: str = "det_10g.onnx") -> Path:
    """Return the expected path of an InsightFace detector asset."""

    return Path(f"~/.insightface/models/{package_name}/{filename}").expanduser()


def ensure_asset_present(asset: ModelAssetSpec) -> Path:
    """Validate that a model asset exists and matches the declared hash when given."""

    for path in asset.candidate_paths():
        if not path.exists():
            continue
        if asset.sha256 is not None:
            digest = compute_file_sha256(path)
            if digest != asset.sha256:
                raise ModelAssetMissingError(
                    f"Asset '{asset.key}' failed SHA256 validation: {path} "
                    f"(expected {asset.sha256}, got {digest})."
                )
        return path

    hint = f" {asset.install_hint}" if asset.install_hint else ""
    candidates = ", ".join(str(path) for path in asset.candidate_paths())
    raise ModelAssetMissingError(
        f"Missing asset '{asset.key}' for {asset.description}. "
        f"Checked: {candidates}.{hint}"
    )


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA256 hash of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_insightface_detector_model(
    ctx_id: int,
    package_name: str = "buffalo_l",
) -> "FaceDetector":
    """Load the InsightFace face detector used for alignment (Protocol-typed)."""

    insightface.utils.ensure_available("models", package_name, root="~/.insightface")
    detector_path = get_detector_asset_path(package_name=package_name)
    if not detector_path.exists():
        raise ModelAssetMissingError(
            f"InsightFace detector asset is missing after download attempt: {detector_path}."
        )
    detect_model = insightface.model_zoo.get_model(detector_path.as_posix())
    det_size = (640, 640)
    if detect_model is None:
        raise ModelAssetMissingError(
            f"InsightFace detector failed to load from: {detector_path}."
        )
    detector = cast(FaceDetector, detect_model)
    detector.prepare(ctx_id=ctx_id, det_size=det_size, input_size=det_size)
    return detector


@lru_cache(maxsize=4)
def get_shared_insightface_detector(
    ctx_id: int,
    package_name: str = "buffalo_l",
) -> "FaceDetector":
    """Return a process-wide singleton InsightFace detector for a given ctx_id."""

    return load_shared_insightface_detector_model(ctx_id=ctx_id, package_name=package_name)


@dataclass(frozen=True)
class FaceDetection:
    """A single detected face: bbox + 5-point landmarks."""

    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    image_hw: tuple[int, int]


def detect_face_for_embed(
    image: torch.Tensor,
    detector: "FaceDetector",
) -> "FaceDetection | None":
    """Run InsightFace detection on a single HWC RGB [0,1] image tensor."""

    img_arr = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.float32)
    bboxes, kpss = detector.detect(img_arr, max_num=1)
    if bboxes is None:
        return None
    bbox_arr = np.asarray(bboxes, dtype=np.float32)
    if bbox_arr.size == 0:
        return None
    if kpss is None or len(kpss) == 0:
        return None
    box_row = bbox_arr[0]
    score = float(box_row[4]) if box_row.shape[0] >= 5 else 1.0
    return FaceDetection(
        bbox=np.asarray(box_row[:4], dtype=np.float32),
        landmarks=np.asarray(kpss[0], dtype=np.float32),
        score=score,
        image_hw=(int(image.shape[0]), int(image.shape[1])),
    )


def detect_faces_for_embed(
    images: list[torch.Tensor],
    detector: "FaceDetector",
) -> list["FaceDetection | None"]:
    """Batch version of `detect_face_for_embed` — one detection per image."""

    return [detect_face_for_embed(image, detector) for image in images]
