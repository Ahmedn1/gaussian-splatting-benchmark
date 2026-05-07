"""
06_evaluate.py — Phase 1 evaluation: render from held-out cam04 and score it
============================================================================

Loads a trained 3DGS point cloud (from `experiments/static/{scene}/{exp}/`) and
renders it from cam04's pose. cam04 was *withheld* during training, so this is a
true novel-view evaluation.

WHAT THIS GIVES YOU
-------------------
- Render of cam04's view from the trained Gaussians
- The ground-truth cam04 image side-by-side
- PSNR, SSIM, LPIPS — the standard NVS metrics
- A per-camera report (we also render the 3 training views as a sanity check;
  those should be near-perfect since the model literally trained on them)

WHY NOT USE gaussian-splatting/render.py?
-----------------------------------------
Their render.py rebuilds the full Scene object from the train sparse, which
only contains cam01/02/03 — cam04 isn't there. So we'd have to pollute the
training sparse to render the held-out view, which would defeat the point of
holding it out. Instead we load the trained .ply directly, build a single
Camera object from `static/sparse/0/` (which has all 4 cameras), and call the
differentiable rasterizer ourselves.

METRICS — quick refresher
-------------------------
- **PSNR** (peak signal-to-noise ratio): pixel-wise MSE on a log scale.
  Higher = better. >30 dB is "good", >35 dB is "very good". Easy to game by
  blurring (low MSE doesn't require sharp detail).
- **SSIM** (structural similarity): correlates local luminance/contrast/
  structure. Range [0, 1], higher = better. Better at penalizing blur than PSNR.
- **LPIPS** (learned perceptual): distance in VGG feature space. Lower = better.
  Best correlate of human perception — punishes blur and floaters. Use this as
  the "real" number when reporting.

Usage:
    python scripts/06_evaluate.py --scene scene_S004 --experiment baseline
    python scripts/06_evaluate.py --scene scene_S004 --experiment baseline --iteration 7000
"""

import argparse
import json
import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
EXP_DIR      = PROJECT_ROOT / "experiments" / "static"
GS_DIR       = PROJECT_ROOT / "repos" / "gaussian-splatting"

# Make gaussian-splatting importable
sys.path.insert(0, str(GS_DIR))


# ── COLMAP read helpers ────────────────────────────────────────────────────────

def _read_next_bytes(f, n, fmt):
    return struct.unpack(fmt, f.read(n))


def read_cameras_bin(path: Path) -> dict:
    """Return {cam_id: {model_id, width, height, params}}."""
    cams = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id = _read_next_bytes(f, 8, "Ii")
            w, h = _read_next_bytes(f, 16, "QQ")
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = _read_next_bytes(f, 8 * n_params, f"{n_params}d")
            cams[cam_id] = {
                "model_id": model_id, "width": w, "height": h,
                "params": list(params),
            }
    return cams


def read_images_bin(path: Path) -> dict:
    """Return {im_id: {qw,qx,qy,qz,tx,ty,tz,camera_id,name}}."""
    images = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            im_id = _read_next_bytes(f, 4, "I")[0]
            qw, qx, qy, qz, tx, ty, tz = _read_next_bytes(f, 56, "7d")
            cam_id = _read_next_bytes(f, 4, "I")[0]
            # Null-terminated name
            name = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            n_pts = _read_next_bytes(f, 8, "Q")[0]
            f.read(n_pts * (8 + 8 + 4))  # skip 2D points (we don't need them)
            images[im_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": cam_id, "name": name.decode("utf-8"),
            }
    return images


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


# ── Build a Camera object that gaussian-splatting's renderer accepts ───────────

def build_gs_camera(image_entry: dict, cam_entry: dict, image_path: Path,
                    uid: int = 0):
    """
    Construct a scene.cameras.Camera object from a COLMAP image+camera pair.

    The renderer expects a Camera with FoVx, FoVy, R, T, image, image_width,
    image_height, and various world_view/proj transforms.
    """
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    from PIL import Image

    qw, qx, qy, qz = image_entry["qw"], image_entry["qx"], image_entry["qy"], image_entry["qz"]
    tx, ty, tz     = image_entry["tx"], image_entry["ty"], image_entry["tz"]

    # COLMAP gives us cam_from_world: R_cw, t_cw such that x_cam = R_cw·x_world + t_cw
    # gaussian-splatting expects R as world-from-cam (transpose) and T as the cam-frame translation.
    # See gaussian-splatting/scene/dataset_readers.py readColmapCameras() for the convention.
    R_cw = quat_to_rotmat(qw, qx, qy, qz)
    R = R_cw.T            # world-from-cam (their convention)
    T = np.array([tx, ty, tz], dtype=np.float64)

    width  = cam_entry["width"]
    height = cam_entry["height"]
    fx, fy = cam_entry["params"][0], cam_entry["params"][1]
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    img_pil = Image.open(image_path).convert("RGB")
    if img_pil.size != (width, height):
        img_pil = img_pil.resize((width, height), Image.LANCZOS)

    # Camera signature differs slightly between gaussian-splatting versions.
    # Try the new (post-2024) signature first, fall back to the older one.
    try:
        cam = Camera(
            resolution=(width, height),
            colmap_id=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            depth_params=None, image=img_pil, invdepthmap=None,
            image_name=image_path.stem, uid=uid,
        )
    except TypeError:
        cam = Camera(
            colmap_id=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            image=img_pil, gt_alpha_mask=None,
            image_name=image_path.stem, uid=uid,
        )
    return cam


# ── Render & metrics ───────────────────────────────────────────────────────────

def render_view(camera, gaussians, pipeline, background) -> torch.Tensor:
    """Render one camera view through gaussian-splatting's renderer. Returns (3, H, W) float in [0, 1]."""
    from gaussian_renderer import render

    result = render(camera, gaussians, pipeline, background)
    img = result["render"]  # (3, H, W), values can drift slightly outside [0, 1]
    return img.clamp(0.0, 1.0)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, lpips_model) -> dict:
    """
    pred, gt: (3, H, W) in [0, 1] on GPU.
    Returns {psnr, ssim, lpips}.
    """
    from utils.image_utils import psnr as gs_psnr
    from utils.loss_utils import ssim as gs_ssim

    pred_b = pred.unsqueeze(0)
    gt_b   = gt.unsqueeze(0)

    psnr_val = gs_psnr(pred_b, gt_b).mean().item()
    ssim_val = gs_ssim(pred_b, gt_b).mean().item()

    # LPIPS expects [-1, 1]
    lpips_val = lpips_model(pred_b * 2 - 1, gt_b * 2 - 1).item()
    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene",      type=str, default="scene_S004")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--iteration",  type=int, default=None,
                        help="Iteration to evaluate (defaults to the latest saved)")
    parser.add_argument("--sh_degree",  type=int, default=3,
                        help="Must match what was used during training")
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    # Locate the trained model
    model_dir = EXP_DIR / args.scene / args.experiment
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}\n"
                                f"Train first with 05_train_static_3dgs.py")

    pc_dir = model_dir / "point_cloud"
    iter_dirs = sorted(pc_dir.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if not iter_dirs:
        raise FileNotFoundError(f"No saved point clouds in {pc_dir}")

    if args.iteration is not None:
        ply_dir = pc_dir / f"iteration_{args.iteration}"
        if not ply_dir.exists():
            raise FileNotFoundError(f"Iteration {args.iteration} not found. "
                                    f"Available: {[p.name for p in iter_dirs]}")
    else:
        ply_dir = iter_dirs[-1]

    iteration = int(ply_dir.name.split("_")[1])
    ply_path  = ply_dir / "point_cloud.ply"
    print(f"Evaluating: {model_dir.name} @ iter {iteration}")
    print(f"  PLY: {ply_path}")

    # Load Gaussians
    from scene import GaussianModel
    from arguments import PipelineParams
    from argparse import Namespace

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = args.sh_degree
    print(f"  Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")

    # Build pipeline params (defaults)
    pipeline = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Read full 4-camera sparse and the source images
    scene_proc = PROC_DIR / args.scene
    full_sparse = scene_proc / "static" / "sparse" / "0"
    images_dir  = scene_proc / "static" / "images"

    cams_bin   = read_cameras_bin(full_sparse / "cameras.bin")
    images_bin = read_images_bin(full_sparse / "images.bin")

    # Read which cams were train vs test
    meta_path = scene_proc / "dynamic_dataset_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_cams = set(meta["train_cameras"])
        test_cams  = set(meta["test_cameras"])
    else:
        train_cams = {"cam01", "cam02", "cam03"}
        test_cams  = {"cam04"}

    # Initialize LPIPS
    print("  Loading LPIPS (alex)...")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").cuda()
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    # Output dir
    eval_dir = model_dir / "eval" / f"iteration_{iteration}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "renders").mkdir(exist_ok=True)
    (eval_dir / "gt").mkdir(exist_ok=True)

    # Render every camera (train + test) and compute metrics
    per_image: dict = {}
    train_metrics, test_metrics = [], []

    print("\nRendering each camera...")
    for im_id, im in sorted(images_bin.items(), key=lambda kv: kv[1]["name"]):
        cam_entry = cams_bin[im["camera_id"]]
        img_path  = images_dir / im["name"]
        if not img_path.exists():
            print(f"  ⚠ skipping {im['name']} (file not found)")
            continue

        camera = build_gs_camera(im, cam_entry, img_path, uid=im_id)

        with torch.no_grad():
            pred = render_view(camera, gaussians, pipeline, background)

        gt = camera.original_image[:3, :, :].to(pred.device)
        if gt.max() > 1.5:
            gt = gt / 255.0

        with torch.no_grad():
            metrics = compute_metrics(pred, gt, lpips_model)

        cam_name = Path(im["name"]).stem  # "cam04"
        is_test = cam_name in test_cams
        split = "test" if is_test else "train"

        per_image[cam_name] = {**metrics, "split": split}
        (test_metrics if is_test else train_metrics).append(metrics)

        # Save the rendered + GT
        torchvision.utils.save_image(pred, str(eval_dir / "renders" / f"{cam_name}.png"))
        torchvision.utils.save_image(gt,   str(eval_dir / "gt"      / f"{cam_name}.png"))

        marker = "🟦 TEST " if is_test else "🟢 train"
        print(f"  {marker} {cam_name}: PSNR={metrics['psnr']:6.2f}  "
              f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")

    # Aggregate
    def mean(metrics_list, key):
        return float(np.mean([m[key] for m in metrics_list])) if metrics_list else float("nan")

    summary = {
        "scene": args.scene,
        "experiment": args.experiment,
        "iteration": iteration,
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "train": {
            "n": len(train_metrics),
            "psnr":  mean(train_metrics, "psnr"),
            "ssim":  mean(train_metrics, "ssim"),
            "lpips": mean(train_metrics, "lpips"),
        },
        "test": {
            "n": len(test_metrics),
            "psnr":  mean(test_metrics, "psnr"),
            "ssim":  mean(test_metrics, "ssim"),
            "lpips": mean(test_metrics, "lpips"),
        },
        "per_image": per_image,
    }

    summary_path = eval_dir / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Train ({summary['train']['n']} cams):  PSNR={summary['train']['psnr']:.2f}  "
          f"SSIM={summary['train']['ssim']:.4f}  LPIPS={summary['train']['lpips']:.4f}")
    print(f"Test  ({summary['test']['n']} cams):  PSNR={summary['test']['psnr']:.2f}  "
          f"SSIM={summary['test']['ssim']:.4f}  LPIPS={summary['test']['lpips']:.4f}")
    print(f"\nSaved: {summary_path}")
    print(f"       {eval_dir / 'renders'}/  {eval_dir / 'gt'}/")


if __name__ == "__main__":
    main()
