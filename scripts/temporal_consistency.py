"""
Temporal consistency analysis for AEGIS-masked avatars.

Renders a combined camera+expression sequence (sinusoidal orbit + all timesteps)
for both masked and unmasked avatars, then computes per-frame:
  - Utility SSIM        (masked vs unmasked)
  - Temporal SSIM       (consecutive frames)
  - Temporal Warping Error (Lai et al. ECCV 2018)
  - Identity CosSim     (masked frame vs reference embed) [held-out recognizer, default ir152]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim2d
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "GaussianAvatars"))
sys.path.insert(0, str(ROOT / "src"))

from aegis.models import FaceNotDetectedError, get_verification_model
from aegis.models.base import load_insightface_detector

# Held-out recognizer for the identity-cosine curve (variant + calibrated tau).
VARIANT = {"ir152": "r152", "irse50": "ir_se50", "mobileface": None, "facenet": "vggface2"}
THRESHOLD = {"ir152": 0.2561, "irse50": 0.3293, "mobileface": 0.3520, "facenet": 0.4981}
from aegis.splat import PipelineConfig, load_gaussians, render_single_frame
from utils.viewer_utils import OrbitCamera

MODELS_DIR = ROOT / "models"
W, H, FOVY = 540, 540, 20


# ── Face detection / cropping ────────────────────────────────────────────────


def face_bbox(frame_f32: np.ndarray, detector):
    """Detect face in a float32 [0,1] RGB frame; return clamped (x1,y1,x2,y2) or None."""
    h, w = frame_f32.shape[:2]
    bboxes, _ = detector.detect((frame_f32 * 255).astype(np.uint8), max_num=1)
    if bboxes is not None and len(bboxes) > 0:
        x1, y1, x2, y2, _ = bboxes[0].astype(int)
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return None


def crop(frame: np.ndarray, bbox) -> np.ndarray:
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


# ── Temporal warping error (Lai et al. ECCV 2018) ────────────────────────────


def _to_raft_tensor(rgb_f32: np.ndarray) -> torch.Tensor:
    """(H,W,3) float32 [0,1] RGB → (1,3,H,W) tensor in [-1,1] on cuda (RAFT input range)."""
    t = torch.from_numpy(np.ascontiguousarray(rgb_f32)).permute(2, 0, 1).unsqueeze(0).cuda()
    return 2.0 * t - 1.0


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img using flow. img: (1,C,H,W), flow: (1,2,H,W) in pixel units."""
    _, _, H_, W_ = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(H_, device=img.device, dtype=torch.float32),
        torch.arange(W_, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), 0).unsqueeze(0)  # (1, 2, H, W)
    vgrid = grid + flow
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W_ - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H_ - 1, 1) - 1.0
    return F.grid_sample(img, vgrid.permute(0, 2, 3, 1),
                         mode="bilinear", padding_mode="border", align_corners=True)

def warping_error_pair(prev_m, curr_m, prev_u, curr_u, flow_model):
    """Returns (e_warp_m, e_warp_u) in one RAFT batch."""
    p = torch.cat([_to_raft_tensor(prev_m), _to_raft_tensor(prev_u)], dim=0)
    c = torch.cat([_to_raft_tensor(curr_m), _to_raft_tensor(curr_u)], dim=0)

    H_, W_ = p.shape[2], p.shape[3]
    Hp = ((H_ + 7) // 8) * 8
    Wp = ((W_ + 7) // 8) * 8
    pad = (0, Wp - W_, 0, Hp - H_)
    if pad[1] or pad[3]:
        p = F.pad(p, pad)
        c = F.pad(c, pad)

    with torch.no_grad():
        flow = flow_model(c, p, num_flow_updates=6)[-1]

    p_warped = _warp(p, flow)
    if pad[1] or pad[3]:
        c        = c[..., :H_, :W_]
        p_warped = p_warped[..., :H_, :W_]

    err = (c - p_warped).abs() / 2.0  # (2, 3, H, W)
    return err[0].mean().item(), err[1].mean().item()

def warping_error(prev_f32: np.ndarray, curr_f32: np.ndarray,
                  flow_model: torch.nn.Module) -> float:
    """E_warp = mean L1 between curr and warp(prev → curr) [Lai et al. 2018].

    prev_f32, curr_f32: (H, W, 3) float32 in [0, 1].
    """
    p = _to_raft_tensor(prev_f32)
    c = _to_raft_tensor(curr_f32)

    # RAFT requires H, W divisible by 8
    H_, W_ = p.shape[2], p.shape[3]
    Hp = ((H_ + 7) // 8) * 8
    Wp = ((W_ + 7) // 8) * 8
    pad = (0, Wp - W_, 0, Hp - H_)  # (left, right, top, bottom)
    if pad[1] or pad[3]:
        p = F.pad(p, pad)
        c = F.pad(c, pad)

    with torch.no_grad():
        # flow from current → previous (so we can warp previous to current)
        flow = flow_model(c, p)[-1]

    p_warped = _warp(p, flow)

    if pad[1] or pad[3]:
        c        = c[..., :H_, :W_]
        p_warped = p_warped[..., :H_, :W_]

    # de-normalize from [-1,1] back to [0,1] before taking abs
    return ((c - p_warped).abs() / 2.0).mean().item()


# ── Rendering ────────────────────────────────────────────────────────────────


def render_sequence(gaussians, cam, pipeline, timesteps, cam_angles_x, cam_angles_y):
    """Render N frames; returns list of (H, W, 3) uint8 arrays."""
    frames = []
    for ts, ax, ay in zip(timesteps, cam_angles_x, cam_angles_y):
        gaussians.select_mesh_by_timestep(int(ts))
        cam.orbit_x(ax)
        cam.orbit_y(ay)
        rgb = render_single_frame(gaussians, cam, pipeline)
        cam.orbit_y(-ay)
        cam.orbit_x(-ax)
        frames.append(
            (np.clip(rgb.cpu().detach().numpy(), 0, 1) * 255).astype(np.uint8)
        )
    return frames


def overlay_metrics(frame, ssim_val, cossim_val, threshold):
    """Draw SSIM (bottom-left) and CosSim (bottom-right) onto a uint8 RGB frame."""
    frame = frame.copy()
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.55, 1
    pad   = 8

    def put(text, x, y, color):
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color,     thick,     cv2.LINE_AA)

    put(f"SSIM {ssim_val:.3f}", pad, h - pad, (255, 255, 255))

    cos_text  = f"Sim {cossim_val:.3f}"
    cos_color = (0, 210, 0) if cossim_val > threshold else (210, 0, 0)
    tw        = cv2.getTextSize(cos_text, font, scale, thick)[0][0]
    put(cos_text, w - tw - pad, h - pad, cos_color)
    return frame


def save_video(frames, path, fps, crf=18):
    """Encode frames to H.264 via ffmpeg (crf=18 → near-lossless)."""
    import subprocess
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


# ── Plot ─────────────────────────────────────────────────────────────────────


def make_plot(n_frames, u_ssim, cos_sims, t_ssim_u, t_ssim_m, e_warp_u, e_warp_m,
              masked_frames, thumb_every, out_path, ver_threshold):
    x   = np.arange(n_frames)
    x_t = np.arange(1, n_frames)  # temporal metrics start at frame 1

    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 1.8], hspace=0.30,
                           top=0.97, bottom=0.04, left=0.07, right=0.98)

    ax1   = fig.add_subplot(gs[0])
    ax2   = fig.add_subplot(gs[1], sharex=ax1)
    ax3   = fig.add_subplot(gs[2], sharex=ax1)
    ax4   = fig.add_subplot(gs[3], sharex=ax1)
    ax_tb = fig.add_subplot(gs[4])
    ax_tb.axis("off")

    # ax1 — Utility SSIM (per frame, masked vs unmasked)
    ax1.plot(x, u_ssim, color="steelblue", lw=1.5)
    ax1.set_ylabel("Utility SSIM\n(masked vs unmasked, ↑=fidelity)", fontsize=9)
    ax1.set_ylim(0.8, 1.0)
    ax1.grid(True, alpha=0.3)

    # ax2 — CosSim vs reference identity (masked sequence)
    ax2.plot(x, cos_sims, color="darkorange", lw=1.5)
    ax2.axhline(ver_threshold, color="red", ls="--", lw=1,
                label=f"Threshold {ver_threshold:.3f} (↓=protected)")
    ax2.set_ylabel("CosSim\n(masked vs reference)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ax3 — Temporal SSIM, consecutive frames
    ax3.plot(x_t, t_ssim_u, color="green",   lw=1.4, label="Unmasked", alpha=0.85)
    ax3.plot(x_t, t_ssim_m, color="crimson", lw=1.4, label="Masked",   alpha=0.85)
    ax3.fill_between(x_t, t_ssim_u, t_ssim_m,
                     alpha=0.15, color="purple",
                     label="Δ (gap = flicker added by masking)")
    ax3.set_ylabel("Temporal SSIM\n(consecutive frames, ↑=stable)", fontsize=9)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(True, alpha=0.3)

    # ax4 — Temporal Warping Error
    ax4.plot(x_t, e_warp_u, color="green",   lw=1.4, label="Unmasked", alpha=0.85)
    ax4.plot(x_t, e_warp_m, color="crimson", lw=1.4, label="Masked",   alpha=0.85)
    ax4.fill_between(x_t, e_warp_u, e_warp_m,
                     alpha=0.15, color="purple",
                     label="Δ (gap = motion-corrected flicker from masking)")
    ax4.set_ylabel("Warping Error\n(flow-aligned L1, ↓=stable)", fontsize=9)
    ax4.set_xlabel("Frame", fontsize=9)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Thumbnail row — masked frames sampled across the sequence
    thumb_indices = list(range(0, n_frames, thumb_every))
    n_thumbs = len(thumb_indices)
    ax_tb.set_title("Masked frames (sampled)", fontsize=8, pad=2)
    for j, idx in enumerate(thumb_indices):
        x_frac = (j + 0.5) / n_thumbs
        w_frac = 0.85 / n_thumbs
        newax  = ax_tb.inset_axes(
            (x_frac - w_frac / 2, 0.05, w_frac, 0.85),
            transform=ax_tb.transAxes,
        )
        newax.imshow(masked_frames[idx], interpolation="lanczos")
        newax.set_title(f"{idx}", fontsize=5, pad=1)
        newax.axis("off")

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.savefig(str(out_path.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="AEGIS temporal consistency analysis")
    p.add_argument("--unmasked-ply",    required=True)
    p.add_argument("--masked-ply",      required=True)
    p.add_argument("--reference-image", required=True)
    p.add_argument("--radius",          type=float, default=1.0,
                   help="Camera orbit radius (1=NeRSemble, 20=FaceScape)")
    p.add_argument("--output-dir",      default="output/temporal_consistency")
    p.add_argument("--n-frames",        type=int, default=None,
                   help="Max frames to render (default: all timesteps)")
    p.add_argument("--camera-boundary-angles", type=float, nargs=6,
                   default=[-0.5, 0.5, -0.5, 0.5, 0.0, 0.0],
                   metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
                   help="Camera orbit bounds (same format as mask_avatar.py)")
    p.add_argument("--embedder", default="ir152", choices=list(VARIANT),
                   help="Held-out recognizer for the identity-cosine curve (default: ir152)")
    p.add_argument("--ver-threshold",   type=float, default=None,
                   help="CosSim verification threshold (default: the embedder's calibrated tau)")
    p.add_argument("--fps",             type=int, default=25)
    p.add_argument("--thumb-every",     type=int, default=None,
                   help="Thumbnail every N frames (default: ~8 thumbnails)")
    p.add_argument("--no-video",        action="store_true",
                   help="Skip video rendering and saving")
    p.add_argument("--motion-npz",      default=None,
                   help="Path to motion_*.npz (from _extract_motion_pattern.py). "
                        "Overwrites per-timestep FLAME params on both avatars "
                        "(expr/rotation/neck_pose/jaw_pose/eyes_pose/translation) "
                        "so all subjects animate with identical motion.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ver_threshold is None:
        args.ver_threshold = THRESHOLD[args.embedder]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PipelineConfig(background_color=[1.0, 1.0, 1.0])

    # ── Load avatars ──────────────────────────────────────────────────────────
    print("Loading avatars...")

    def _unwrap(g):
        return g[0] if isinstance(g, tuple) else g

    unmasked = _unwrap(load_gaussians(Path(args.unmasked_ply)))
    masked   = _unwrap(load_gaussians(Path(args.masked_ply)))

    if args.motion_npz:
        motion = np.load(args.motion_npz)
        motion_keys = ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "dynamic_offset"]
        n_frames = int(motion["expr"].shape[0])
        for g in [unmasked, masked]:
            for k in motion_keys:
                g.flame_param[k] = torch.from_numpy(motion[k]).float().cuda()
            g.num_timesteps = n_frames
        if "cam_angles_x" in motion:
            cam_angles_x = motion["cam_angles_x"]
            cam_angles_y = motion["cam_angles_y"]
        else:
            x_min, x_max, y_min, y_max = args.camera_boundary_angles[:4]
            t = np.linspace(0, 2 * np.pi, n_frames)
            cam_angles_x = (x_min + x_max) / 2 + (x_max - x_min) / 2 * np.sin(t)
            cam_angles_y = (y_min + y_max) / 2 + (y_max - y_min) / 2 * np.sin(2 * t)
        print(f"Motion loaded from {args.motion_npz}: {n_frames} frames")
    else:
        n_ts     = min(unmasked.num_timesteps, masked.num_timesteps)
        n_frames = min(args.n_frames, n_ts) if args.n_frames else n_ts
        print(f"Timesteps available: {n_ts} → rendering {n_frames} frames")
        x_min, x_max, y_min, y_max = args.camera_boundary_angles[:4]
        t = np.linspace(0, 2 * np.pi, n_frames)
        cam_angles_x = (x_min + x_max) / 2 + (x_max - x_min) / 2 * np.sin(t)
        cam_angles_y = (y_min + y_max) / 2 + (y_max - y_min) / 2 * np.sin(2 * t)

    timesteps = np.arange(n_frames)

    cam = OrbitCamera(W, H, r=args.radius, fovy=FOVY, convention="opencv")

    # ── Load detector, embedder, optical flow ─────────────────────────────────
    print("Loading face detector...")
    detector = load_insightface_detector(ctx_id=0)

    print(f"Loading face embedder ({args.embedder}, held-out)...")
    embedder = get_verification_model(args.embedder, "cuda", VARIANT[args.embedder])

    def embed_frame(arr_f32: np.ndarray):
        """Embed an (H,W,3) float32 [0,1] RGB frame; returns (1,D) cpu tensor or None."""
        t = torch.from_numpy(np.ascontiguousarray(arr_f32)).cuda()
        try:
            e = embedder.embed_batch([t])[0]
        except FaceNotDetectedError:
            return None
        return None if e is None else e.reshape(1, -1).cpu()

    ref_img = cv2.cvtColor(cv2.imread(str(args.reference_image)), cv2.COLOR_BGR2RGB)
    ref_t = embed_frame(ref_img.astype(np.float32) / 255)
    if ref_t is None:
        raise RuntimeError(f"No face detected in reference image {args.reference_image}")

    print("Loading RAFT optical flow model...")
    flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                            progress=False).cuda().eval()

    # ── Render ────────────────────────────────────────────────────────────────
    print("Rendering unmasked sequence...")
    unmasked_frames = render_sequence(unmasked, cam, pipeline, timesteps,
                                      cam_angles_x, cam_angles_y)
    print("Rendering masked sequence...")
    masked_frames   = render_sequence(masked,   cam, pipeline, timesteps,
                                      cam_angles_x, cam_angles_y)

    # ── Per-frame metrics ─────────────────────────────────────────────────────
    print("Computing per-frame metrics...")
    unmasked_f = [f.astype(np.float32) / 255 for f in unmasked_frames]
    masked_f   = [f.astype(np.float32) / 255 for f in masked_frames]

    print("Detecting face regions...")
    unmasked_bboxes = [face_bbox(uf, detector) for uf in unmasked_f]
    n_detected = sum(b is not None for b in unmasked_bboxes)
    print(f"Face detected in {n_detected}/{n_frames} frames")

    valid_bboxes = [b for b in unmasked_bboxes if b is not None]
    volume_bbox  = (tuple(int(v) for v in np.median(valid_bboxes, axis=0))
                    if valid_bboxes else None)

    u_ssim, t_ssim_m, t_ssim_u = [], [], []
    e_warp_m, e_warp_u         = [], []
    cos_sims_m, cos_sims_u     = [], []

    for i, (mf, uf) in enumerate(tqdm(zip(masked_f, unmasked_f), total=len(masked_f))):

        # Utility SSIM uses per-frame face bbox
        bbox_i = unmasked_bboxes[i]
        u_ssim.append(ssim2d(crop(mf, bbox_i), crop(uf, bbox_i),
                              channel_axis=2, data_range=1.0))

        # Temporal SSIM and warping error use the previous frame's bbox
        # so both crops in the comparison are pixel-aligned
        if i > 0:
            bbox_prev = unmasked_bboxes[i - 1]
            t_ssim_m.append(ssim2d(crop(mf, bbox_prev),
                                    crop(masked_f[i - 1], bbox_prev),
                                    channel_axis=2, data_range=1.0))
            t_ssim_u.append(ssim2d(crop(uf, bbox_prev),
                                    crop(unmasked_f[i - 1], bbox_prev),
                                    channel_axis=2, data_range=1.0))

            # Warping error on a fixed face crop (volume_bbox) for stability
            prev_m = crop(masked_f[i - 1],   volume_bbox)
            curr_m = crop(mf,                volume_bbox)
            prev_u = crop(unmasked_f[i - 1], volume_bbox)
            curr_u = crop(uf,                volume_bbox)
            em, eu = warping_error_pair(prev_m, curr_m, prev_u, curr_u, flow_model)
            e_warp_m.append(em)
            e_warp_u.append(eu)

        # Identity CosSim (held-out recognizer, masked frame vs reference embedding)
        with torch.no_grad():
            for arr, store in [(mf, cos_sims_m), (uf, cos_sims_u)]:
                emb = embed_frame(arr)
                cs = (float("nan") if emb is None
                      else torch.cosine_similarity(emb, ref_t, dim=1).item())
                store.append(cs)

    cos_sims = cos_sims_m  # plot uses masked CosSim

    # ── Overlay metrics onto frames and save videos ───────────────────────────
    if not args.no_video:
        masked_out   = [overlay_metrics(f, u_ssim[i], cos_sims_m[i], args.ver_threshold)
                        for i, f in enumerate(masked_frames)]
        unmasked_out = [overlay_metrics(f, u_ssim[i], cos_sims_u[i], args.ver_threshold)
                        for i, f in enumerate(unmasked_frames)]
        save_video(unmasked_out, out_dir / "unmasked.mp4", args.fps)
        save_video(masked_out,   out_dir / "masked.mp4",   args.fps)
        print(f"Videos saved → {out_dir}/{{unmasked,masked}}.mp4")
        del unmasked_out, masked_out
    del unmasked_f, masked_f

    # ── Save per-frame CSV ────────────────────────────────────────────────────
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "u_ssim",
            "tssim_unmasked", "tssim_masked", "tssim_delta",
            "ewarp_unmasked", "ewarp_masked", "ewarp_delta",
            "cossim_masked", "cossim_unmasked",
        ])
        for i in range(n_frames - 1):
            writer.writerow([
                i + 1,
                round(u_ssim[i + 1], 6),
                round(t_ssim_u[i],   6),
                round(t_ssim_m[i],   6),
                round(t_ssim_u[i] - t_ssim_m[i], 6),
                round(e_warp_u[i],   6),
                round(e_warp_m[i],   6),
                round(e_warp_m[i] - e_warp_u[i], 6),
                round(cos_sims_m[i + 1], 6),
                round(cos_sims_u[i + 1], 6),
            ])
    print(f"Per-frame metrics → {csv_path}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    summary = {
        # Utility
        "u_ssim_mean":          round(float(np.mean(u_ssim)), 6),
        "u_ssim_std":           round(float(np.std(u_ssim)),  6),
        # Temporal SSIM
        "tssim_unmasked_mean":  round(float(np.mean(t_ssim_u)), 6),
        "tssim_unmasked_std":   round(float(np.std(t_ssim_u)),  6),
        "tssim_masked_mean":    round(float(np.mean(t_ssim_m)), 6),
        "tssim_masked_std":     round(float(np.std(t_ssim_m)),  6),
        "tssim_delta_mean":     round(float(np.mean(t_ssim_u) - np.mean(t_ssim_m)), 6),
        # Temporal warping error
        "ewarp_unmasked_mean":  round(float(np.mean(e_warp_u)), 6),
        "ewarp_unmasked_std":   round(float(np.std(e_warp_u)),  6),
        "ewarp_masked_mean":    round(float(np.mean(e_warp_m)), 6),
        "ewarp_masked_std":     round(float(np.std(e_warp_m)),  6),
        "ewarp_delta_mean":     round(float(np.mean(e_warp_m) - np.mean(e_warp_u)), 6),
        # Identity
        "cossim_masked_mean":   round(float(np.nanmean(cos_sims_m)), 6),
        "cossim_masked_std":    round(float(np.nanstd(cos_sims_m)),  6),
        "cossim_masked_max":    round(float(np.nanmax(cos_sims_m)),  6),
        "cossim_unmasked_mean": round(float(np.nanmean(cos_sims_u)), 6),
        "cossim_unmasked_min":  round(float(np.nanmin(cos_sims_u)),  6),
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics → {json_path}")

    # ── Console report ────────────────────────────────────────────────────────
    print()
    print(f"U-SSIM (utility, masked vs unmasked):     {np.mean(u_ssim):.4f} ± {np.std(u_ssim):.4f}")
    print(f"T-SSIM unmasked (baseline):                {np.mean(t_ssim_u):.4f} ± {np.std(t_ssim_u):.4f}")
    print(f"T-SSIM masked:                             {np.mean(t_ssim_m):.4f} ± {np.std(t_ssim_m):.4f}")
    print(f"T-SSIM Δ (baseline − masked, flicker):     {np.mean(t_ssim_u) - np.mean(t_ssim_m):+.4f}")
    print(f"E-warp unmasked (baseline):                {np.mean(e_warp_u):.6f}")
    print(f"E-warp masked:                             {np.mean(e_warp_m):.6f}")
    print(f"E-warp Δ (masked − baseline, flicker):     {np.mean(e_warp_m) - np.mean(e_warp_u):+.6f}")
    print(f"CosSim masked (mean / max worst-case):     {np.nanmean(cos_sims_m):.4f} / {np.nanmax(cos_sims_m):.4f}")
    print(f"CosSim unmasked (mean / min):              {np.nanmean(cos_sims_u):.4f} / {np.nanmin(cos_sims_u):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    thumb_every = args.thumb_every or max(1, n_frames // 8)
    make_plot(n_frames, u_ssim, cos_sims, t_ssim_u, t_ssim_m, e_warp_u, e_warp_m,
              masked_frames, thumb_every, out_dir / "temporal_consistency.png",
              args.ver_threshold)
    print(f"Plot saved → {out_dir}/temporal_consistency.png")


if __name__ == "__main__":
    main()
