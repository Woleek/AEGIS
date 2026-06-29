"""
Viewpoint-Robust Leakage (VRL) for AEGIS-masked 3D avatars.

For each subject: renders a K x K uniform grid of viewpoints over
--camera-boundary-angles (orbit_x, orbit_y; roll fixed at its midpoint), embeds
every rendered view with each held-out FR model, and compares it against the
subject's GT-gallery identity prototype (mean L2-normalised embedding). With the
per-view genuine cosine similarities it produces, per FR model:

  Leakage curve   L(tau)   = mean_{s,v} 1[ cos(view, gt) >= tau ]   (survival CDF)
  Scalar          VRL-AUC  = (1/(1-tau*)) * INT_{tau*}^{1} L(tau) dtau   (lower = better)
  Operating leak  L(tau*)  = re-identification rate at the FR threshold tau*
  Worst-case      sigma_wc(s) = max_v cos(view, gt);  WC-leak = mean_s 1[sigma_wc >= tau*]

tau* is each model's calibrated verification threshold (TAR@FAR), read straight
from VerificationModelSpec.threshold.

Inputs (two modes):
  (A) Gallery mode:  --avatar-dir  <dir of <subject>/point_cloud.ply>
                     --gt-dir      <dir of <subject>/*.png|jpg>   (reference gallery)
  (B) Single mode:   --avatar-ply  <point_cloud.ply>
                     --reference-image <img> [<img> ...]   (one subject)

Per-subject radius: NeRSemble avatars use --radius 1, FaceScape --radius 20.
For a combined run pass --radius-map '{"306": 1, "122": 20, ...}'.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "GaussianAvatars"))
sys.path.insert(0, str(ROOT / "src"))

from aegis.models import get_verification_model, get_model_spec  # noqa: E402
from aegis.splat import PipelineConfig, load_gaussians, render_single_frame  # noqa: E402
from utils.viewer_utils import OrbitCamera  # noqa: E402

W, H, FOVY = 540, 540, 20
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_MODELS = ["ir152", "irse50", "mobileface", "facenet"]


# ── Inputs ───────────────────────────────────────────────────────────────────


def _unwrap(g):
    return g[0] if isinstance(g, tuple) else g


def discover_subjects(avatar_dir: Path, gt_dir: Path) -> list[tuple[str, Path, list[Path]]]:
    """Return [(subject_id, point_cloud.ply, [gt images])] for subjects in both dirs."""
    subjects = []
    for sub in sorted(p for p in avatar_dir.iterdir() if p.is_dir()):
        ply = sub / "point_cloud.ply"
        if not ply.exists():
            cand = sorted(sub.rglob("point_cloud.ply"))
            if not cand:
                continue
            ply = cand[-1]
        gt_sub = gt_dir / sub.name
        if not gt_sub.is_dir():
            continue
        gts = sorted(p for p in gt_sub.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not gts:
            continue
        subjects.append((sub.name, ply, gts))
    return subjects


def build_grid(boundary: list[float], k: int) -> list[tuple[float, float, float]]:
    """K x K uniform (orbit_x, orbit_y) grid, roll fixed at midpoint."""
    x_min, x_max, y_min, y_max, z_min, z_max = boundary[:6]
    xs = np.linspace(x_min, x_max, k)
    ys = np.linspace(y_min, y_max, k)
    z_mid = (z_min + z_max) / 2.0
    return [(float(ox), float(oy), z_mid) for ox in xs for oy in ys]


# ── Rendering ────────────────────────────────────────────────────────────────


def render_views(gaussians, cam, pipeline, viewpoints) -> list[torch.Tensor]:
    """Render one (H, W, C) float[0,1] tensor per viewpoint, at timestep 0."""
    if hasattr(gaussians, "select_mesh_by_timestep"):
        gaussians.select_mesh_by_timestep(0)
    frames = []
    for ox, oy, oz in viewpoints:
        cam.orbit_x(ox)
        cam.orbit_y(oy)
        if oz:
            cam.orbit_z(oz)
        rgb = render_single_frame(gaussians, cam, pipeline)  # (H, W, C) cuda [0,1]
        frames.append(rgb.detach().clamp(0.0, 1.0))
        if oz:
            cam.orbit_z(-oz)
        cam.orbit_y(-oy)
        cam.orbit_x(-ox)
    return frames


# ── Embedding ────────────────────────────────────────────────────────────────


def _load_hwc(path: Path, device: str) -> torch.Tensor | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float().div(255.0).to(device)


def _l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def reference_embedding(model, gt_paths: list[Path], device: str) -> torch.Tensor | None:
    """Mean L2-normalised GT-gallery embedding (the identity prototype)."""
    imgs = [t for t in (_load_hwc(p, device) for p in gt_paths) if t is not None]
    if not imgs:
        return None
    embs = [e for e in model.embed_batch(imgs) if e is not None]
    if not embs:
        return None
    return _l2(torch.cat(embs, dim=0).mean(dim=0, keepdim=True))  # (1, D)


def view_similarities(model, frames: list[torch.Tensor], ref: torch.Tensor,
                      chunk: int = 16) -> list[float | None]:
    """Cosine similarity of each rendered view to ref; None where no face aligns."""
    sims: list[float | None] = []
    for i in range(0, len(frames), chunk):
        batch = frames[i : i + chunk]
        embs = model.embed_batch(batch)
        for emb in embs:
            if emb is None:
                sims.append(None)
            else:
                sims.append(torch.cosine_similarity(emb, ref.expand_as(emb), dim=1).item())
    return sims


# ── Metrics ──────────────────────────────────────────────────────────────────


def leakage_curve(sims: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """L(tau) = fraction of similarities >= tau."""
    return (sims[None, :] >= taus[:, None]).mean(axis=1)


def vrl_auc(sims: np.ndarray, tau_star: float, n: int = 1024) -> float:
    """Normalised area under L(tau) over the re-identifiable region [tau*, 1]."""
    taus = np.linspace(tau_star, 1.0, n)
    L = leakage_curve(sims, taus)
    _trapz = getattr(np, "trapezoid", np.trapz)
    area = float(_trapz(L, taus))
    span = max(1.0 - tau_star, 1e-9)
    return area / span


def compute_model_metrics(per_subject_sims: dict[str, list[float]],
                          tau_star: float) -> dict:
    all_sims = np.array([s for v in per_subject_sims.values() for s in v], dtype=np.float64)
    wc = np.array([max(v) for v in per_subject_sims.values() if v], dtype=np.float64)
    if all_sims.size == 0:
        return {"n_views": 0, "n_subjects": 0}
    return {
        "tau_star": round(tau_star, 6),
        "n_views": int(all_sims.size),
        "n_subjects": int(wc.size),
        "mean_sim": round(float(all_sims.mean()), 6),
        "max_sim": round(float(all_sims.max()), 6),
        # headline scalar
        "vrl_auc": round(vrl_auc(all_sims, tau_star), 6),
        # operating-point leakage (re-identification rate at tau*)
        "op_leak_rate": round(float((all_sims >= tau_star).mean()), 6),
        # worst-case-over-views
        "wc_sim_mean": round(float(wc.mean()), 6),
        "wc_sim_median": round(float(np.median(wc)), 6),
        "wc_sim_max": round(float(wc.max()), 6),
        "wc_leak_rate": round(float((wc >= tau_star).mean()), 6),
    }


# ── Plot ─────────────────────────────────────────────────────────────────────


def make_plot(model_data: dict, out_path: Path):
    """model_data[name] = {sims, wc, tau_star, metrics}."""
    taus = np.linspace(-0.25, 1.0, 1000)
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )

    for i, (name, d) in enumerate(model_data.items()):
        c = cmap(i % 10)
        sims = d["sims"]
        wc = d["wc"]
        ts = d["tau_star"]
        m = d["metrics"]
        L = leakage_curve(sims, taus)
        Lwc = leakage_curve(wc, taus)

        # (a) leakage curve + worst-case curve + tau* line + AUC_op shading
        ax1.plot(taus, L, color=c, lw=1.8,
                 label=f"{name}  VRL-AUC={m['vrl_auc']:.3f}  L(τ*)={m['op_leak_rate']:.2f}")
        ax1.plot(taus, Lwc, color=c, lw=1.2, ls=":", alpha=0.8)
        ax1.axvline(ts, color=c, ls="--", lw=1.0, alpha=0.7)
        mask = taus >= ts
        ax1.fill_between(taus[mask], L[mask], 0, color=c, alpha=0.08)

        # (b) worst-case similarity distribution
        parts = ax2.violinplot(wc, positions=[i], widths=0.7, showmeans=True,
                               showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(c)
            pc.set_alpha(0.35)
        ax2.hlines(ts, i - 0.4, i + 0.4, color=c, ls="--", lw=1.2)

    ax1.set_xlabel("Cosine-similarity threshold τ")
    ax1.set_ylabel("Leakage rate  L(τ) = P(σ ≥ τ)")
    ax1.set_title("Viewpoint leakage curve\n(solid=all views, dotted=worst-case, "
                  "dashed=τ*, shaded=VRL-AUC region)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(-0.25, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_xticks(range(len(model_data)))
    ax2.set_xticklabels(list(model_data.keys()), rotation=20, fontsize=8)
    ax2.set_ylabel("Worst-case per-subject σ_wc = max_v cos(view, gt)")
    ax2.set_title("Worst-viewpoint similarity per subject\n(dashed = τ*; above τ* ⇒ re-identifiable)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.savefig(str(out_path.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Viewpoint-Robust Leakage (VRL) eval")
    p.add_argument("--avatar-dir", type=Path,
                   help="Dir of <subject>/point_cloud.ply (gallery mode).")
    p.add_argument("--gt-dir", type=Path,
                   help="Dir of <subject>/*.png reference gallery (gallery mode).")
    p.add_argument("--avatar-ply", type=Path,
                   help="Single avatar point_cloud.ply (single mode).")
    p.add_argument("--reference-image", type=Path, nargs="+",
                   help="GT image(s) for the single avatar (single mode).")
    p.add_argument("--subject", type=str, default="subject",
                   help="Subject id label in single mode.")
    p.add_argument("--embedders", nargs="+", default=DEFAULT_MODELS,
                   help=f"Held-out eval FR models. Default: {' '.join(DEFAULT_MODELS)}")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Optional per-embedder variant (aligned with --embedders).")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help="Optional τ* overrides; else use "
                        "each model's calibrated VerificationModelSpec.threshold.")
    p.add_argument("--camera-boundary-angles", type=float, nargs=6,
                   default=[-0.5, 0.5, -0.5, 0.5, 0.0, 0.0],
                   metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"))
    p.add_argument("--eval-view-grid", type=int, default=9,
                   help="K for the K x K viewpoint grid (default 9 -> 81 views).")
    p.add_argument("--radius", type=float, default=1.0,
                   help="Camera orbit radius (1=NeRSemble, 20=FaceScape).")
    p.add_argument("--radius-map", type=str, default=None,
                   help='JSON {subject: radius} overriding --radius per subject.')
    p.add_argument("--output-dir", type=Path, default=Path("output/viewpoint_leakage"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve_threshold(name: str, variant: str | None, override: float | None) -> float:
    if override is not None:
        return override
    spec = get_model_spec(name, variant)
    if spec.threshold is None:
        raise SystemExit(
            f"Model '{name}' (variant {variant or 'default'}) has no calibrated "
            f"threshold; pass --thresholds to override."
        )
    return float(spec.threshold)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subject list.
    if args.avatar_dir and args.gt_dir:
        subjects = discover_subjects(args.avatar_dir, args.gt_dir)
        if not subjects:
            raise SystemExit(f"No matching subjects in {args.avatar_dir} / {args.gt_dir}.")
    elif args.avatar_ply and args.reference_image:
        subjects = [(args.subject, args.avatar_ply, list(args.reference_image))]
    else:
        raise SystemExit("Provide --avatar-dir + --gt-dir, or --avatar-ply + --reference-image.")

    radius_map = json.loads(args.radius_map) if args.radius_map else {}
    variants = args.variants or [None] * len(args.embedders)
    if len(variants) != len(args.embedders):
        raise SystemExit("--variants must align with --embedders.")
    overrides = args.thresholds or [None] * len(args.embedders)
    if len(overrides) != len(args.embedders):
        raise SystemExit("--thresholds must align with --embedders.")

    print(f"Subjects: {len(subjects)} | Models: {', '.join(args.embedders)} | "
          f"Grid: {args.eval_view_grid}x{args.eval_view_grid}")

    pipeline = PipelineConfig(background_color=[1.0, 1.0, 1.0])
    viewpoints = build_grid(args.camera_boundary_angles, args.eval_view_grid)

    # Load each eval model once.
    models = {}
    tau_stars = {}
    for name, variant, override in zip(args.embedders, variants, overrides):
        models[name] = get_verification_model(name, args.device, variant)
        tau_stars[name] = resolve_threshold(name, variant, override)
        print(f"  {name}: τ* = {tau_stars[name]:.4f}")

    # per_subject_sims[model][subject] = [cossim per detected view]
    per_subject_sims = {name: {} for name in args.embedders}
    rows = []
    n_views_total = len(viewpoints)

    for sid, ply, gts in tqdm(subjects, desc="Subjects"):
        radius = float(radius_map.get(sid, args.radius))
        gaussians = _unwrap(load_gaussians(Path(ply)))
        cam = OrbitCamera(W, H, r=radius, fovy=FOVY, convention="opencv")
        frames = render_views(gaussians, cam, pipeline, viewpoints)

        for name in args.embedders:
            model = models[name]
            ref = reference_embedding(model, gts, args.device)
            if ref is None:
                print(f"  [warn] {name}: no GT face for subject {sid}; skipping.")
                continue
            sims = view_similarities(model, frames, ref)
            kept = [s for s in sims if s is not None]
            per_subject_sims[name][sid] = kept
            for (ox, oy, _), s in zip(viewpoints, sims):
                rows.append([name, sid, round(ox, 4), round(oy, 4),
                             "" if s is None else round(s, 6),
                             int(s is not None)])

        del frames, gaussians
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Write per-view CSV ────────────────────────────────────────────────────
    csv_path = args.output_dir / "per_view.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subject", "orbit_x", "orbit_y", "cossim", "detected"])
        w.writerows(rows)
    print(f"Per-view similarities → {csv_path}")

    # ── Metrics + plot ────────────────────────────────────────────────────────
    summary = {
        "config": {
            "embedders": args.embedders,
            "camera_boundary_angles": args.camera_boundary_angles,
            "eval_view_grid": args.eval_view_grid,
            "views_per_subject": n_views_total,
            "n_subjects": len(subjects),
        },
        "models": {},
    }
    model_data = {}
    for name in args.embedders:
        psm = per_subject_sims[name]
        if not any(psm.values()):
            print(f"  [warn] {name}: no detected views; skipping metrics.")
            continue
        m = compute_model_metrics(psm, tau_stars[name])
        det = m["n_views"] / max(len(subjects) * n_views_total, 1)
        m["detection_rate"] = round(det, 4)
        summary["models"][name] = m
        all_sims = np.array([s for v in psm.values() for s in v], dtype=np.float64)
        wc = np.array([max(v) for v in psm.values() if v], dtype=np.float64)
        model_data[name] = {"sims": all_sims, "wc": wc,
                            "tau_star": tau_stars[name], "metrics": m}

    json_path = args.output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {json_path}")

    if model_data:
        plot_path = args.output_dir / "viewpoint_leakage.png"
        make_plot(model_data, plot_path)
        print(f"Plot → {plot_path}")

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'model':<12} {'τ*':>7} {'VRL-AUC':>9} {'L(τ*)':>8} {'WC-leak':>9} "
          f"{'WCsim_mean':>11} {'WCsim_max':>10} {'det%':>6}")
    print("-" * 78)
    for name, m in summary["models"].items():
        print(f"{name:<12} {m['tau_star']:>7.3f} {m['vrl_auc']:>9.4f} "
              f"{m['op_leak_rate']:>8.3f} {m['wc_leak_rate']:>9.3f} "
              f"{m['wc_sim_mean']:>11.4f} {m['wc_sim_max']:>10.4f} "
              f"{m['detection_rate']*100:>5.1f}%")
    print("=" * 78)
    print("VRL-AUC ↓ better (leakage mass above τ*); L(τ*)=re-id rate at threshold; "
          "WC-leak=frac subjects re-identifiable from ≥1 view.")


if __name__ == "__main__":
    main()
