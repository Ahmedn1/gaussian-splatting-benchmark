"""
07_visualize_gaussians.py — inspect a trained 3DGS point cloud
==============================================================

Opens `experiments/static/{scene}/{exp}/point_cloud/iteration_*/point_cloud.ply`
and produces diagnostic plots that tell you a lot about how training went:

  • Gaussian count vs. iteration (if multiple iterations are saved)
  • Scale histogram (per-Gaussian "radius" — telltale of densification health)
  • Opacity histogram (most Gaussians should be high-opacity by the end)
  • Spatial XYZ scatter, colored by SH DC term (the base color)
  • Camera frustums overlaid in the same coord frame for context

WHY THIS MATTERS
----------------
3DGS optimization is mostly opaque from loss curves alone. The Gaussian
distribution itself is the most diagnostic signal:

  - Too few Gaussians (< ~10k for an indoor scene): under-densification,
    PSNR will be capped no matter how long you train.
  - Many tiny scales near zero: split is firing too aggressively.
  - Many low-opacity Gaussians: pruning isn't working; you're carrying dead
    weight that slows rendering.
  - Gaussians far from the visual hull: floaters — the classic 3DGS failure
    mode in undertrained / few-view regimes.

Usage:
    python scripts/07_visualize_gaussians.py --scene scene_S004 --experiment baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
EXP_DIR      = PROJECT_ROOT / "experiments" / "static"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh_dc_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    """
    SH DC term in 3DGS: stored as f_dc_0/1/2 (3 channels, one per RGB).
    Convert to viewable RGB: rgb ≈ 0.5 + C0 * sh_dc, where C0 = 0.282...
    Then clip to [0, 1].
    """
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * sh_dc
    return np.clip(rgb, 0.0, 1.0)


def load_ply_gaussians(ply_path: Path) -> dict:
    """Load a 3DGS .ply and return the raw fields (still in their stored space)."""
    from plyfile import PlyData
    data = PlyData.read(str(ply_path))
    v = data["vertex"].data

    out = {
        "xyz":       np.stack([v["x"], v["y"], v["z"]], axis=-1),
        "opacity":   sigmoid(v["opacity"]),  # stored as logit
        "scales":    np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)),
        # SH DC term (3 channels)
        "sh_dc":     np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1),
    }
    return out


def plot_scale_hist(scales: np.ndarray, ax):
    """Histogram of the geometric mean scale per Gaussian."""
    geo_mean = np.exp(np.mean(np.log(scales + 1e-12), axis=-1))
    ax.hist(np.log10(geo_mean + 1e-12), bins=80, color="steelblue", alpha=0.85)
    ax.set_xlabel("log₁₀(scale)  [world units]")
    ax.set_ylabel("Gaussians")
    ax.set_title(f"Scale distribution (n={len(scales):,})")
    median = float(np.median(geo_mean))
    ax.axvline(np.log10(median), color="crimson", linestyle="--",
               label=f"median = {median:.4f}")
    ax.legend()


def plot_opacity_hist(opacity: np.ndarray, ax):
    ax.hist(opacity, bins=50, color="darkgreen", alpha=0.85)
    ax.set_xlabel("opacity (sigmoid)")
    ax.set_ylabel("Gaussians")
    n_low = int(np.sum(opacity < 0.05))
    ax.set_title(f"Opacity distribution  ({n_low:,} below 0.05)")
    ax.axvline(0.05, color="crimson", linestyle="--", label="prune threshold ≈ 0.05")
    ax.legend()


def plot_3d_scatter(xyz: np.ndarray, colors: np.ndarray, ax, max_points: int = 50_000):
    """3D scatter of Gaussian centers, colored by SH DC."""
    if len(xyz) > max_points:
        idx = np.random.RandomState(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        colors = colors[idx]

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=0.5, marker=".", alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Gaussian centers (showing {len(xyz):,})")

    # Equal aspect
    extents = np.stack([xyz.min(axis=0), xyz.max(axis=0)])
    span = (extents[1] - extents[0]).max() / 2
    mid = extents.mean(axis=0)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def plot_camera_frustums(scene_proc: Path, ax):
    """Overlay camera positions on the 3D scatter (read from full sparse)."""
    sys_path_added = False
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        eval_mod = import_module("06_evaluate")
        sys_path_added = True
    except Exception:
        # Fall back: import the helpers we need by re-implementing minimally
        eval_mod = None

    # Use 06's COLMAP readers if importable, else a tiny inline copy
    if eval_mod is None:
        # Minimal inline reader for cams + images
        import struct
        def _read_cams(p):
            cams = {}
            with open(p, "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
                for _ in range(n):
                    cam_id, model_id = struct.unpack("Ii", f.read(8))
                    w, h = struct.unpack("QQ", f.read(16))
                    n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
                    params = struct.unpack(f"{n_params}d", f.read(8 * n_params))
                    cams[cam_id] = {"width": w, "height": h, "params": list(params)}
            return cams

        def _read_imgs(p):
            imgs = {}
            with open(p, "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
                for _ in range(n):
                    im_id = struct.unpack("I", f.read(4))[0]
                    qw, qx, qy, qz, tx, ty, tz = struct.unpack("7d", f.read(56))
                    cam_id = struct.unpack("I", f.read(4))[0]
                    name = b""
                    while True:
                        b = f.read(1)
                        if b == b"\x00": break
                        name += b
                    n_pts = struct.unpack("Q", f.read(8))[0]
                    f.read(n_pts * 20)
                    imgs[im_id] = {
                        "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                        "tx": tx, "ty": ty, "tz": tz,
                        "camera_id": cam_id, "name": name.decode(),
                    }
            return imgs

        def _quat(qw, qx, qy, qz):
            return np.array([
                [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
                [2*qx*qy+2*qz*qw,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy],
            ])
        read_cameras_bin, read_images_bin, quat_to_rotmat = _read_cams, _read_imgs, _quat
    else:
        read_cameras_bin = eval_mod.read_cameras_bin
        read_images_bin  = eval_mod.read_images_bin
        quat_to_rotmat   = eval_mod.quat_to_rotmat

    sparse = scene_proc / "static" / "sparse" / "0"
    if not (sparse / "images.bin").exists():
        return

    images = read_images_bin(sparse / "images.bin")
    cams   = read_cameras_bin(sparse / "cameras.bin")

    for im in images.values():
        R_cw = quat_to_rotmat(im["qw"], im["qx"], im["qy"], im["qz"])
        t_cw = np.array([im["tx"], im["ty"], im["tz"]])
        # Camera center in world: C = -R_cwᵀ · t_cw
        C = -R_cw.T @ t_cw
        ax.scatter(*C, c="red", marker="^", s=80,
                   edgecolors="black", linewidths=0.8, zorder=10)
        ax.text(C[0], C[1], C[2], "  " + Path(im["name"]).stem,
                fontsize=8, color="darkred", zorder=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene",      type=str, default="scene_S004")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--iteration",  type=int, default=None,
                        help="Iteration to inspect (defaults to latest)")
    parser.add_argument("--exp_root",   type=str, default=None,
                        help="Override experiments root (e.g. for dynamic experiments)")
    args = parser.parse_args()

    exp_root = Path(args.exp_root) if args.exp_root else EXP_DIR
    model_dir = exp_root / args.scene / args.experiment
    pc_dir = model_dir / "point_cloud"
    iter_dirs = sorted(pc_dir.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if not iter_dirs:
        raise FileNotFoundError(f"No saved point clouds in {pc_dir}")

    if args.iteration is not None:
        ply_dir = pc_dir / f"iteration_{args.iteration}"
    else:
        ply_dir = iter_dirs[-1]

    iteration = int(ply_dir.name.split("_")[1])
    ply_path = ply_dir / "point_cloud.ply"
    print(f"Loading {ply_path}")

    g = load_ply_gaussians(ply_path)
    n = len(g["xyz"])
    print(f"  {n:,} Gaussians loaded")
    print(f"  scale geom-mean median: {np.median(np.exp(g['scales'].mean(axis=-1))):.5f}")
    print(f"  median opacity: {np.median(g['opacity']):.3f}")

    rgb = sh_dc_to_rgb(g["sh_dc"])

    # ── Plot ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"{args.scene} / {args.experiment} @ iter {iteration:,} — "
                 f"{n:,} Gaussians", fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    plot_scale_hist(g["scales"], ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_opacity_hist(g["opacity"], ax2)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_3d_scatter(g["xyz"], rgb, ax3)
    plot_camera_frustums(PROC_DIR / args.scene, ax3)

    # Population summary text panel
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    n_low_op = int(np.sum(g["opacity"] < 0.05))
    n_tiny   = int(np.sum(np.exp(g["scales"].mean(axis=-1)) < 1e-3))
    summary = (
        f"Gaussians:                {n:,}\n"
        f"  Low-opacity (<0.05):    {n_low_op:,}  ({n_low_op/n*100:.1f}%)\n"
        f"  Tiny scale (<1e-3):     {n_tiny:,}  ({n_tiny/n*100:.1f}%)\n"
        f"\n"
        f"Opacity:\n"
        f"  median = {np.median(g['opacity']):.3f}\n"
        f"  mean   = {np.mean(g['opacity']):.3f}\n"
        f"\n"
        f"Scale (geom-mean per Gaussian):\n"
        f"  median = {np.median(np.exp(g['scales'].mean(axis=-1))):.5f}\n"
        f"  p99    = {np.percentile(np.exp(g['scales'].mean(axis=-1)), 99):.5f}\n"
        f"\n"
        f"XYZ extent:\n"
        f"  X: {g['xyz'][:,0].min():+7.2f} → {g['xyz'][:,0].max():+7.2f}\n"
        f"  Y: {g['xyz'][:,1].min():+7.2f} → {g['xyz'][:,1].max():+7.2f}\n"
        f"  Z: {g['xyz'][:,2].min():+7.2f} → {g['xyz'][:,2].max():+7.2f}\n"
    )
    ax4.text(0.0, 1.0, summary, family="monospace", fontsize=10,
             verticalalignment="top", transform=ax4.transAxes)

    out = model_dir / "eval" / f"iteration_{iteration}" / "gaussians_diagnostic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
