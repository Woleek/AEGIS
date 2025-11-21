import os
import insightface
import numpy as np
from pathlib import Path
import torch
from typing import Dict, Iterator, Literal, Sequence, Tuple

import torch.nn.functional as F


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
