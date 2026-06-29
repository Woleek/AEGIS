"""
Robustness Testing Script

Creates degraded versions of protected avatar renders to test if privacy
protection perturbations remain effective under various image transformations
that an adversary might use to circumvent the protection.

Includes:
- Image degradations (compression, noise, blur, etc.)
- Adversarial purification methods (face restoration, denoising)
- Combined attack pipelines

Usage:
    uv run scripts/robustness_test.py \
        --input-dir ./datasets/seed42/NeRSembleMasked_adaface_all/eps_0.100/renders \
        --output-base ./datasets/seed42/NeRSembleMasked_adaface_all/eps_0.100/robustness \
        --degradations all

The script creates subdirectories for each degradation type, which can then be
evaluated using the existing evaluate.py script.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Optional imports for neural network-based purification
FACEXLIB_AVAILABLE = True
DIFFUSERS_AVAILABLE = True
REALESRGAN_AVAILABLE = True

import types as _types
import torchvision.transforms.functional as _tv_functional
_ft = _types.ModuleType("torchvision.transforms.functional_tensor")
_ft.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)

from gfpgan import GFPGANer

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# =============================================================================
# Degradation Functions
# =============================================================================


def jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def gaussian_noise(image: np.ndarray, std: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to the image."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filtering (common anti-perturbation technique)."""
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Apply bilateral filtering (edge-preserving smoothing)."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def downscale_upscale(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    """Downscale and upscale back to original size (resolution loss)."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness_change(image: np.ndarray, delta: int = 30) -> np.ndarray:
    """Adjust brightness."""
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def contrast_change(image: np.ndarray, alpha: float = 1.3) -> np.ndarray:
    """Adjust contrast."""
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    adjusted = (image.astype(np.float32) - mean) * alpha + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(image, table)


def saturation_change(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """Adjust color saturation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def salt_and_pepper(image: np.ndarray, prob: float = 0.01) -> np.ndarray:
    """Add salt and pepper noise."""
    output = image.copy()
    # Salt
    salt_mask = np.random.random(image.shape[:2]) < prob / 2
    output[salt_mask] = 255
    # Pepper
    pepper_mask = np.random.random(image.shape[:2]) < prob / 2
    output[pepper_mask] = 0
    return output


def bit_depth_reduction(image: np.ndarray, bits: int = 4) -> np.ndarray:
    """Reduce bit depth (quantization)."""
    factor = 256 // (2**bits)
    return (image // factor * factor).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization per channel."""
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
    )
    channels = cv2.split(image)
    eq_channels = [clahe_obj.apply(ch) for ch in channels]
    return cv2.merge(eq_channels)


def motion_blur(image: np.ndarray, kernel_size: int = 15, angle: int = 0) -> np.ndarray:
    """Apply motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    # Rotate kernel for different angles
    if angle != 0:
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return cv2.filter2D(image, -1, kernel)


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply sharpening filter."""
    kernel = np.array([[-1, -1, -1], [-1, 9 + strength, -1], [-1, -1, -1]]) / (
        1 + strength
    )
    return cv2.filter2D(image, -1, kernel)


# =============================================================================
# Classical Denoising / Purification Methods
# =============================================================================


def non_local_means(
    image: np.ndarray, h: float = 10, template_window: int = 7, search_window: int = 21
) -> np.ndarray:
    """Apply Non-Local Means denoising (effective against high-freq perturbations)."""
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


def total_variation_denoise(
    image: np.ndarray, weight: float = 0.1, n_iter: int = 100
) -> np.ndarray:
    """Apply Total Variation denoising (Rudin-Osher-Fatemi model).

    TV denoising is particularly effective against adversarial perturbations
    as it smooths while preserving edges.
    """
    from scipy.ndimage import laplace

    img = image.astype(np.float64) / 255.0
    denoised = img.copy()

    for _ in range(n_iter):
        # Compute gradient magnitude
        grad_x = np.roll(denoised, -1, axis=1) - denoised
        grad_y = np.roll(denoised, -1, axis=0) - denoised
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Compute divergence of normalized gradient
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        div = (nx - np.roll(nx, 1, axis=1)) + (ny - np.roll(ny, 1, axis=0))

        # Update
        denoised = denoised + weight * div + 0.1 * (img - denoised)
        denoised = np.clip(denoised, 0, 1)

    return (denoised * 255).astype(np.uint8)


def wavelet_denoise(image: np.ndarray, sigma: float = 20) -> np.ndarray:
    """Apply wavelet-based denoising using BayesShrink."""
    from skimage.restoration import denoise_wavelet

    # Convert to float [0, 1]
    img_float = image.astype(np.float64) / 255.0
    denoised = denoise_wavelet(
        img_float,
        channel_axis=2,
        rescale_sigma=True,
        sigma=sigma / 255.0,
        mode="soft",
        wavelet="db1",
    )
    return (np.clip(denoised, 0, 1) * 255).astype(np.uint8)


def iterative_jpeg(
    image: np.ndarray, quality: int = 50, iterations: int = 5
) -> np.ndarray:
    """Apply iterative JPEG compression (perturbation laundering)."""
    result = image
    for _ in range(iterations):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", result, encode_param)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


def blur_sharpen_cycle(image: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Apply blur-sharpen cycles (laundering attack)."""
    result = image
    for _ in range(iterations):
        # Blur
        result = cv2.GaussianBlur(result, (3, 3), 0)
        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
        result = cv2.filter2D(result, -1, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def feature_squeeze(
    image: np.ndarray, bit_depth: int = 5, median_size: int = 2
) -> np.ndarray:
    """Apply feature squeezing defense (Xu et al., 2017).

    Combines bit depth reduction and spatial smoothing.
    """
    # Bit depth reduction
    factor = 256 // (2**bit_depth)
    squeezed = (image // factor * factor).astype(np.uint8)
    # Spatial smoothing (using small median filter if size > 1)
    if median_size > 1:
        squeezed = cv2.medianBlur(squeezed, median_size * 2 - 1)
    return squeezed


def random_transform(image: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply random geometric transformations (input transformation defense).

    Applies random crop, resize, and rotation to break perturbation patterns.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = image.shape[:2]

    # Random crop (90-100% of original)
    crop_ratio = np.random.uniform(0.9, 1.0)
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_y = np.random.randint(0, h - crop_h + 1)
    start_x = np.random.randint(0, w - crop_w + 1)
    cropped = image[start_y : start_y + crop_h, start_x : start_x + crop_w]

    # Resize back to original
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Small random rotation (-5 to 5 degrees)
    angle = np.random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(resized, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return rotated


def autoencoder_denoise(image: np.ndarray) -> np.ndarray:
    """Placeholder for autoencoder-based denoising.

    In practice, this would use a trained denoising autoencoder.
    """
    raise NotImplementedError(
        "Autoencoder-based denoising not implemented. Please use a specific denoising method instead."
    )


# =============================================================================
# Neural Network-based Purification
# =============================================================================

# Global model cache to avoid reloading
_MODEL_CACHE = {}


def _get_gfpgan_model(version: str = "1.4", device: str = "cuda"):
    """Load GFPGAN model (cached)."""
    cache_key = f"gfpgan_{version}"
    if cache_key not in _MODEL_CACHE:
        model_path = ROOT_DIR / "models" / f"GFPGANv{version}.pth"
        if not model_path.exists():
            # Try to download
            import urllib.request

            url = f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{version}.pth"
            print(f"Downloading GFPGAN model to {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(model_path))

        _MODEL_CACHE[cache_key] = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            device=device,
        )
    return _MODEL_CACHE[cache_key]


def _get_realesrgan_model(scale: int = 2, device: str = "cuda"):
    """Load Real-ESRGAN model (cached)."""
    cache_key = f"realesrgan_{scale}"
    if cache_key not in _MODEL_CACHE:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        model_path = ROOT_DIR / "models" / f"RealESRGAN_x{scale}plus.pth"

        if not model_path.exists():
            import urllib.request

            url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x{scale}plus.pth"
            print(f"Downloading Real-ESRGAN model to {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(model_path))

        _MODEL_CACHE[cache_key] = RealESRGANer(
            scale=scale, model_path=str(model_path), model=model, device=device
        )
    return _MODEL_CACHE[cache_key]


def gfpgan_restore(
    image: np.ndarray, version: str = "1.4", device: str = "cuda"
) -> np.ndarray:
    """Apply GFPGAN face restoration.

    GFPGAN is very effective at removing adversarial perturbations because it
    reconstructs the face using learned facial priors.
    """
    if not FACEXLIB_AVAILABLE:
        raise ImportError(
            "GFPGAN requires facexlib. Install with: pip install gfpgan facexlib"
        )

    restorer = _get_gfpgan_model(version, device)
    # GFPGAN expects BGR
    _, _, restored = restorer.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    return restored


def realesrgan_restore(
    image: np.ndarray, scale: int = 2, device: str = "cuda"
) -> np.ndarray:
    """Apply Real-ESRGAN super-resolution (removes perturbations as side effect).

    Upscales then downscales to original size, smoothing perturbations.
    """
    if not REALESRGAN_AVAILABLE:
        raise ImportError(
            "Real-ESRGAN requires realesrgan and basicsr. Install with: pip install realesrgan basicsr"
        )

    h, w = image.shape[:2]
    upsampler = _get_realesrgan_model(scale, device)
    output, _ = upsampler.enhance(image, outscale=scale)
    # Downscale back to original size
    return cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)


# =============================================================================
# Combined Attack Pipelines
# =============================================================================


def social_media_pipeline(image: np.ndarray) -> np.ndarray:
    """Simulate social media upload/download pipeline."""
    result = cv2.resize(
        image,
        (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)),
        interpolation=cv2.INTER_AREA,
    )
    result = jpeg_compression(result, quality=75)
    result = cv2.resize(
        result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    result = cv2.filter2D(result, -1, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def messaging_pipeline(image: np.ndarray) -> np.ndarray:
    """Simulate messaging app compression pipeline."""
    # Aggressive resize
    result = cv2.resize(
        image,
        (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)),
        interpolation=cv2.INTER_AREA,
    )
    # Heavy JPEG
    result = jpeg_compression(result, quality=40)
    # Resize back
    result = cv2.resize(
        result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return result


def adversarial_removal_pipeline(image: np.ndarray) -> np.ndarray:
    """Aggressive adversarial perturbation removal pipeline."""
    # Median filter to remove high-frequency perturbations
    result = cv2.medianBlur(image, 3)
    # Bilateral to preserve edges
    result = cv2.bilateralFilter(result, 9, 75, 75)
    # Light JPEG to further smooth
    result = jpeg_compression(result, quality=80)
    return result


def full_purification_pipeline(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Most aggressive purification attempt using all available methods."""
    # Start with classical methods
    result = non_local_means(image, h=10)
    result = cv2.bilateralFilter(result, 9, 75, 75)

    # Apply neural restoration (requires GFPGAN)
    if FACEXLIB_AVAILABLE:
        result = gfpgan_restore(result, device=device)
    else:
        raise ImportError(
            "Full purification pipeline requires GFPGAN. Install with: pip install gfpgan facexlib"
        )

    # Final JPEG to clean up any remaining artifacts
    result = jpeg_compression(result, quality=90)
    return result


# =============================================================================
# Degradation Registry
# =============================================================================

# Each entry: (function, kwargs, description)
DEGRADATIONS: dict[str, tuple[Callable, dict, str]] = {
    # JPEG compression at various quality levels
    "jpeg_q10": (jpeg_compression, {"quality": 10}, "JPEG compression Q=10 (severe)"),
    "jpeg_q30": (jpeg_compression, {"quality": 30}, "JPEG compression Q=30 (heavy)"),
    "jpeg_q50": (jpeg_compression, {"quality": 50}, "JPEG compression Q=50 (moderate)"),
    "jpeg_q70": (jpeg_compression, {"quality": 70}, "JPEG compression Q=70 (light)"),
    # Noise
    "noise_std5": (gaussian_noise, {"std": 5.0}, "Gaussian noise std=5"),
    "noise_std10": (gaussian_noise, {"std": 10.0}, "Gaussian noise std=10"),
    "noise_std20": (gaussian_noise, {"std": 20.0}, "Gaussian noise std=20"),
    "salt_pepper_1pct": (salt_and_pepper, {"prob": 0.01}, "Salt & pepper noise 1%"),
    "salt_pepper_5pct": (salt_and_pepper, {"prob": 0.05}, "Salt & pepper noise 5%"),
    # Blur / smoothing
    "blur_3x3": (gaussian_blur, {"kernel_size": 3}, "Gaussian blur 3x3"),
    "blur_5x5": (gaussian_blur, {"kernel_size": 5}, "Gaussian blur 5x5"),
    "blur_7x7": (gaussian_blur, {"kernel_size": 7}, "Gaussian blur 7x7"),
    "median_3x3": (median_filter, {"kernel_size": 3}, "Median filter 3x3"),
    "median_5x5": (median_filter, {"kernel_size": 5}, "Median filter 5x5"),
    "bilateral": (bilateral_filter, {}, "Bilateral filter (edge-preserving)"),
    "motion_blur": (motion_blur, {"kernel_size": 15}, "Motion blur"),
    # Resolution
    "downscale_0.25": (
        downscale_upscale,
        {"scale_factor": 0.25},
        "Downscale to 25% and back",
    ),
    "downscale_0.5": (
        downscale_upscale,
        {"scale_factor": 0.5},
        "Downscale to 50% and back",
    ),
    # Brightness / contrast
    "bright_+30": (brightness_change, {"delta": 30}, "Brightness +30"),
    "bright_-30": (brightness_change, {"delta": -30}, "Brightness -30"),
    "contrast_0.7": (contrast_change, {"alpha": 0.7}, "Contrast reduction (0.7x)"),
    "contrast_1.3": (contrast_change, {"alpha": 1.3}, "Contrast increase (1.3x)"),
    "gamma_0.7": (gamma_correction, {"gamma": 0.7}, "Gamma correction 0.7 (brighten)"),
    "gamma_1.5": (gamma_correction, {"gamma": 1.5}, "Gamma correction 1.5 (darken)"),
    # Color
    "saturation_0.5": (saturation_change, {"factor": 0.5}, "Saturation reduced 50%"),
    "saturation_1.5": (saturation_change, {"factor": 1.5}, "Saturation increased 50%"),
    # Quantization
    "bit_depth_4": (bit_depth_reduction, {"bits": 4}, "Reduce to 4-bit color depth"),
    "bit_depth_6": (bit_depth_reduction, {"bits": 6}, "Reduce to 6-bit color depth"),
    # Enhancement
    "hist_eq": (histogram_equalization, {}, "Histogram equalization"),
    "clahe": (clahe, {}, "CLAHE enhancement"),
    "sharpen": (sharpen, {"strength": 1.0}, "Sharpening filter"),
    # ==========================================================================
    # Classical denoising / purification (perturbation removal)
    # ==========================================================================
    "nlm": (non_local_means, {}, "Non-Local Means denoising"),
    "nlm_strong": (non_local_means, {"h": 20}, "Non-Local Means denoising (strong)"),
    "tv_denoise": (total_variation_denoise, {}, "Total Variation denoising"),
    "tv_denoise_strong": (
        total_variation_denoise,
        {"weight": 0.2, "n_iter": 150},
        "Total Variation denoising (strong)",
    ),
    "wavelet": (wavelet_denoise, {}, "Wavelet denoising"),
    "wavelet_strong": (wavelet_denoise, {"sigma": 40}, "Wavelet denoising (strong)"),
    # Iterative / laundering attacks
    "jpeg_iter_5": (
        iterative_jpeg,
        {"quality": 70, "iterations": 5},
        "Iterative JPEG x5 (laundering)",
    ),
    "jpeg_iter_10": (
        iterative_jpeg,
        {"quality": 70, "iterations": 10},
        "Iterative JPEG x10 (laundering)",
    ),
    "blur_sharpen_3": (blur_sharpen_cycle, {"iterations": 3}, "Blur-sharpen cycle x3"),
    "blur_sharpen_5": (blur_sharpen_cycle, {"iterations": 5}, "Blur-sharpen cycle x5"),
    # Input transformation defenses
    "feature_squeeze": (feature_squeeze, {}, "Feature squeezing defense"),
    "random_transform": (random_transform, {}, "Random geometric transform"),
    # ==========================================================================
    # Neural network-based purification 
    # ==========================================================================
    "gfpgan": (gfpgan_restore, {}, "GFPGAN face restoration"),
    "realesrgan": (realesrgan_restore, {}, "Real-ESRGAN super-resolution"),
    # ==========================================================================
    # Combined attack pipelines
    # ==========================================================================
    "pipeline_social": (social_media_pipeline, {}, "Social media upload pipeline"),
    "pipeline_messaging": (messaging_pipeline, {}, "Messaging app pipeline"),
    "pipeline_adversarial": (
        adversarial_removal_pipeline,
        {},
        "Adversarial removal pipeline",
    ),
    "pipeline_full": (full_purification_pipeline, {}, "Full purification pipeline"),
}

# Predefined groups
DEGRADATION_GROUPS = {
    "all": list(DEGRADATIONS.keys()),
    "compression": ["jpeg_q10", "jpeg_q30", "jpeg_q50", "jpeg_q70"],
    "noise": [
        "noise_std5",
        "noise_std10",
        "noise_std20",
        "salt_pepper_1pct",
        "salt_pepper_5pct",
    ],
    "blur": [
        "blur_3x3",
        "blur_5x5",
        "blur_7x7",
        "median_3x3",
        "median_5x5",
        "bilateral",
        "motion_blur",
    ],
    "smoothing": [
        "median_3x3",
        "median_5x5",
        "bilateral",
    ],  # Anti-perturbation techniques
    "resolution": ["downscale_0.25", "downscale_0.5"],
    "lighting": [
        "bright_+30",
        "bright_-30",
        "contrast_0.7",
        "contrast_1.3",
        "gamma_0.7",
        "gamma_1.5",
    ],
    "color": ["saturation_0.5", "saturation_1.5"],
    "quantization": ["bit_depth_4", "bit_depth_6"],
    "enhancement": ["hist_eq", "clahe", "sharpen"],
    # Classical denoising / purification
    "denoise": [
        "nlm",
        "nlm_strong",
        "tv_denoise",
        "tv_denoise_strong",
        "wavelet",
        "wavelet_strong",
    ],
    "laundering": ["jpeg_iter_5", "jpeg_iter_10", "blur_sharpen_3", "blur_sharpen_5"],
    "defenses": ["feature_squeeze", "random_transform"],
    # Neural network purification
    "neural": ["gfpgan", "realesrgan"],
    # Combined pipelines
    "pipelines": [
        "pipeline_social",
        "pipeline_messaging",
        "pipeline_adversarial",
        "pipeline_full",
    ],
    # Common real-world scenarios
    "social_media": ["jpeg_q50", "downscale_0.5", "sharpen"],
    "messaging": ["jpeg_q30", "downscale_0.25"],
    # Deliberate perturbation removal attacks
    "adversarial": ["median_3x3", "median_5x5", "bilateral", "blur_3x3", "jpeg_q50"],
    # Most likely to break protection (comprehensive attack suite)
    "purification": [
        "nlm",
        "tv_denoise",
        "wavelet",
        "jpeg_iter_5",
        "blur_sharpen_3",
        "gfpgan",
        "realesrgan",
        "pipeline_adversarial",
        "pipeline_full",
    ],
}


def parse_degradations(degradation_spec: list[str]) -> list[str]:
    """Parse degradation specification into list of degradation names."""
    result = []
    for spec in degradation_spec:
        if spec in DEGRADATION_GROUPS:
            result.extend(DEGRADATION_GROUPS[spec])
        elif spec in DEGRADATIONS:
            result.append(spec)
        else:
            raise ValueError(f"Unknown degradation or group: {spec}")
    # Remove duplicates while preserving order
    return list(dict.fromkeys(result))


# =============================================================================
# Main Processing
# =============================================================================


NEURAL_METHODS = {"gfpgan", "realesrgan", "pipeline_full"}


def process_directory(
    input_dir: Path,
    output_base: Path,
    degradations: list[str],
    extension: str = ".png",
    seed: int | None = None,
    device: str = "cuda",
) -> dict[str, Path]:
    """
    Process all images in input directory with specified degradations.

    Args:
        input_dir: Directory containing input images
        output_base: Base directory for output (subdirs created per degradation)
        degradations: List of degradation names to apply
        extension: File extension to look for
        seed: Random seed for reproducibility
        device: Device for neural network methods (cuda/cpu)

    Returns:
        Dictionary mapping degradation name to output directory path
    """
    if seed is not None:
        np.random.seed(seed)

    # Find all images
    images = sorted(input_dir.glob(f"*{extension}"))
    if not images:
        # Try common extensions
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            images = sorted(input_dir.glob(f"*{ext}"))
            if images:
                extension = ext
                break

    if not images:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(images)} images in {input_dir}")

    output_dirs = {}

    for deg_name in degradations:
        if deg_name not in DEGRADATIONS:
            print(f"Warning: Unknown degradation '{deg_name}', skipping")
            continue

        func, kwargs, description = DEGRADATIONS[deg_name]
        # Inject device for neural methods
        if deg_name in NEURAL_METHODS:
            kwargs = {**kwargs, "device": device}
        output_dir = output_base / deg_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[deg_name] = output_dir

        # Check if all images already exist in output directory
        all_exist = all((output_dir / img_path.name).exists() for img_path in images)
        if all_exist:
            print(f"\n[SKIP] {description}")
            print(f"Output: {output_dir} (all {len(images)} images already exist)")
            continue

        print(f"\nApplying: {description}")
        print(f"Output: {output_dir}")

        for img_path in tqdm(images, desc=deg_name, leave=False):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Apply degradation
            degraded = func(img, **kwargs)

            # Save with same filename
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), degraded)

    return output_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Apply image degradations to test robustness of privacy protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available degradations:

  IMAGE DEGRADATIONS:
    Compression:  jpeg_q10, jpeg_q30, jpeg_q50, jpeg_q70
    Noise:        noise_std5, noise_std10, noise_std20, salt_pepper_1pct, salt_pepper_5pct
    Blur:         blur_3x3, blur_5x5, blur_7x7, median_3x3, median_5x5, bilateral, motion_blur
    Resolution:   downscale_0.25, downscale_0.5
    Lighting:     bright_+30, bright_-30, contrast_0.7, contrast_1.3, gamma_0.7, gamma_1.5
    Color:        saturation_0.5, saturation_1.5
    Quantization: bit_depth_4, bit_depth_6
    Enhancement:  hist_eq, clahe, sharpen

  PERTURBATION REMOVAL (classical):
    Denoising:    nlm, nlm_strong, tv_denoise, tv_denoise_strong, wavelet, wavelet_strong
    Laundering:   jpeg_iter_5, jpeg_iter_10, blur_sharpen_3, blur_sharpen_5
    Defenses:     feature_squeeze, random_transform

  PERTURBATION REMOVAL (neural - requires optional deps):
    Face restore: gfpgan (GFPGAN face restoration)
    Super-res:    realesrgan (Real-ESRGAN upscaling)

  COMBINED PIPELINES:
    pipeline_social      - Social media upload simulation
    pipeline_messaging   - Messaging app compression
    pipeline_adversarial - Aggressive perturbation removal
    pipeline_full        - Full purification (classical + neural)

Predefined groups:
  all          - All degradations and purification methods
  compression  - JPEG compression variants
  noise        - Gaussian and salt/pepper noise
  blur         - Blurring filters
  smoothing    - Anti-perturbation smoothing (median, bilateral)
  resolution   - Downscaling
  lighting     - Brightness, contrast, gamma
  color        - Saturation changes
  quantization - Bit depth reduction
  enhancement  - Histogram equalization, CLAHE, sharpening
  denoise      - Classical denoising methods (NLM, TV, wavelet)
  laundering   - Iterative attacks to wash out perturbations
  defenses     - Input transformation defenses
  neural       - Neural network purification (GFPGAN, Real-ESRGAN)
  pipelines    - Combined attack pipelines
  purification - Comprehensive purification attack suite (recommended)
  adversarial  - Basic adversarial perturbation removal
  social_media - Typical social media pipeline
  messaging    - Typical messaging app pipeline

Example:
  # Test against purification attacks (most likely to break protection)
  uv run scripts/robustness_test.py \\
      --input-dir ./datasets/seed42/NeRSembleMasked_adaface_all/eps_0.100/renders \\
      --output-base ./datasets/seed42/NeRSembleMasked_adaface_all/eps_0.100/robustness \\
      --degradations purification

  # Test against neural restoration only
  uv run scripts/robustness_test.py \\
      --input-dir ./renders --output-base ./robustness --degradations neural

  # Then evaluate with:
  uv run scripts/evaluate.py \\
      --anonymized-path ./datasets/.../robustness/gfpgan \\
      --anonymized-dataset NeRSembleReconst \\
      --gallery-dataset NeRSembleGT \\
      --embedder adaface \\
      verification

Optional dependencies for neural purification:
  uv add gfpgan facexlib     # For GFPGAN
  uv add realesrgan basicsr  # For Real-ESRGAN
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input images (e.g., renders/)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Base directory for output (subdirectories created per degradation)",
    )
    parser.add_argument(
        "--degradations",
        nargs="+",
        default=["all"],
        help="Degradations or groups to apply (default: all)",
    )
    parser.add_argument(
        "--extension",
        default=".png",
        help="Image file extension to process (default: .png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available degradations and exit",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for neural network methods (default: cuda)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available degradations:\n")
        for name, (_, _, desc) in sorted(DEGRADATIONS.items()):
            print(f"  {name:20s} - {desc}")
        print("\nPredefined groups:\n")
        for group, members in sorted(DEGRADATION_GROUPS.items()):
            print(
                f"  {group:15s} - {', '.join(members[:3])}{'...' if len(members) > 3 else ''}"
            )
        return

    # Validate required arguments
    if args.input_dir is None:
        parser.error("--input-dir is required")
    if args.output_base is None:
        parser.error("--output-base is required")

    # Parse degradation specification
    try:
        degradations = parse_degradations(args.degradations)
    except ValueError as e:
        parser.error(str(e))

    print(
        f"Will apply {len(degradations)} degradations: {', '.join(degradations[:5])}{'...' if len(degradations) > 5 else ''}"
    )

    # Process
    output_dirs = process_directory(
        input_dir=args.input_dir,
        output_base=args.output_base,
        degradations=degradations,
        extension=args.extension,
        seed=args.seed,
        device=args.device,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Robustness test complete!")
    print("=" * 60)
    print(f"\nCreated {len(output_dirs)} degraded datasets:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")

    print("\nTo evaluate robustness, run evaluate.py on each directory:")
    print(f"  uv run scripts/evaluate.py \\")
    print(f"      --anonymized-path <output_dir> \\")
    print(f"      --anonymized-dataset NeRSembleReconst \\")
    print(f"      --gallery-dataset NeRSembleGT \\")
    print(f"      --embedder <embedder> \\")
    print(f"      verification")


if __name__ == "__main__":
    main()
