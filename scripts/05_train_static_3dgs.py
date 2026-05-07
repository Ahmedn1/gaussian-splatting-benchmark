"""
05_train_static_3dgs.py — Phase 1: Static 3DGS Training
=========================================================

Trains a static 3D Gaussian Splatting model on one reference frame per camera.

This is Phase 1 of our learning plan. Before touching dynamic scenes,
we master the static case — it's simpler, faster to iterate on, and
builds the intuition needed for 4DGS.

WHAT THIS SCRIPT DOES
----------------------
1. Rebuilds static/sparse/0 with correct 4-entry images.bin
   (The sparse from MASt3R had 12 entries for 12 ref frames;
    static training only needs 4 — one per camera)

2. Creates a training-only scene (cam01, cam02, cam03)
   cam04 is withheld for evaluation — the model never sees it

3. Reinstalls the correct CUDA rasterizer for gaussian-splatting
   (Deformable-3DGS uses a different version with the same package name)

4. Runs gaussian-splatting/train.py via subprocess

5. Saves all outputs under experiments/static/

WHY ONLY 3 TRAINING IMAGES?
-----------------------------
With 4 synchronized cameras, holding out 1 full camera (25%) is the
correct evaluation methodology for novel view synthesis. Using random
frame splits would let the model interpolate between nearly-identical
views — too easy to be a real test.

The honest expectation: 3 images is very few for 3DGS. The official
paper uses 24-300 images. With 3 views, the model will:
  - Perfectly reconstruct the 3 training viewpoints (near zero loss)
  - Produce mediocre results from cam04's novel angle
  - Show visible Gaussian "floaters" in undertrained regions

This is the learning point: few views → poor generalization.
The dynamic dataset fixes this (453 training images across 3 cameras).

Usage:
    python scripts/05_train_static_3dgs.py --scene scene_S004
    python scripts/05_train_static_3dgs.py --scene scene_S004 --iterations 7000
    python scripts/05_train_static_3dgs.py --scene scene_S004 --sh_degree 1
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
REPOS_DIR    = PROJECT_ROOT / "repos"
EXP_DIR      = PROJECT_ROOT / "experiments" / "static"
GS_DIR       = REPOS_DIR / "gaussian-splatting"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"


# ── CUDA rasterizer management ─────────────────────────────────────────────────

def ensure_static_rasterizer():
    """
    Ensure gaussian-splatting's rasterizer is installed (not the depth version).
    Both ship as `diff_gaussian_rasterization` v0.0.0 — pip won't reinstall
    unless we force it. We detect the depth-diff version by the
    `means2D_densify` forward param and only then force-reinstall.
    """
    # Check what's currently active
    needs_reinstall = True
    try:
        import inspect
        from diff_gaussian_rasterization import GaussianRasterizer
        params = inspect.signature(GaussianRasterizer.forward).parameters
        if "means2D_densify" not in params:
            needs_reinstall = False
            print("  ✅ gaussian-splatting rasterizer already active")
    except Exception:
        pass

    if not needs_reinstall:
        return

    rast_dir = GS_DIR / "submodules" / "diff-gaussian-rasterization"
    print("  Reinstalling gaussian-splatting rasterizer "
          "(depth-diff version detected; force --force-reinstall)...")
    cuda_home = str(Path.home() / "cuda-home")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         "--force-reinstall", "--no-deps", "-q", "."],
        cwd=rast_dir,
        capture_output=True, text=True,
        env={**os.environ,
             "CUDA_HOME": cuda_home,
             "CC":  "/usr/bin/gcc-12",
             "CXX": "/usr/bin/g++-12",
             # nvcc needs cicc on PATH
             "PATH": f"{cuda_home}/bin:" + os.environ.get("PATH", "")},
    )
    if result.returncode != 0:
        print(f"  ⚠ rasterizer reinstall failed: {result.stderr[-300:]}")
        raise RuntimeError("rasterizer reinstall failed")
    print("  ✅ gaussian-splatting rasterizer active")


# ── Sparse rebuild ─────────────────────────────────────────────────────────────

def rebuild_static_sparse(scene_name: str, train_cams: list[str], test_cams: list[str]):
    """
    Rebuild static/sparse/0 with correct entries.

    MASt3R produced 12 entries (3 ref frames × 4 cameras).
    For static training we need exactly 4 entries, one per camera,
    with image names matching the actual files in static/images/.

    We pick the MIDDLE reference frame pose for each camera (ref01)
    as the representative pose, and average focal lengths.

    Args:
        scene_name:  e.g. 'scene_S004'
        train_cams:  e.g. ['cam01', 'cam02', 'cam03']
        test_cams:   e.g. ['cam04']
    """
    import pycolmap

    scene_proc    = PROC_DIR / scene_name
    workspace_sparse = scene_proc / "colmap_workspace" / "sparse" / "0"

    # Build separate train and test sparse directories
    train_sparse = scene_proc / "static_train" / "sparse" / "0"
    train_sparse.mkdir(parents=True, exist_ok=True)

    # Load the full MASt3R reconstruction
    recon = pycolmap.Reconstruction(str(workspace_sparse))

    # Group reference frames by camera name
    cam_groups = {}  # "cam01" → list of image entries
    for im_id, image in recon.images.items():
        stem = Path(image.name).stem          # "cam01_ref01"
        cam_name = stem.split("_ref")[0]      # "cam01"
        if cam_name not in cam_groups:
            cam_groups[cam_name] = []
        cam_groups[cam_name].append((im_id, image))

    # Build cameras.bin and images.bin for training cameras
    sys.path.insert(0, str(SCRIPTS_DIR))
    from _colmap_io import write_cameras_bin, write_images_bin, write_points3d_bin

    new_cameras = {}
    new_images  = {}

    for new_cam_id, cam_name in enumerate(sorted(cam_groups.keys()), start=1):
        if cam_name not in train_cams + test_cams:
            continue

        entries = cam_groups[cam_name]

        # Pick middle reference frame for the pose
        mid_entry = entries[len(entries) // 2]
        _, mid_image = mid_entry

        # Average focal lengths across all ref frames for this camera
        focals = []
        widths, heights = [], []
        for _, img in entries:
            cam = recon.cameras[img.camera_id]
            focals.append(cam.params[0])   # fx (= fy for SIMPLE_PINHOLE or equal for PINHOLE)
            widths.append(cam.width)
            heights.append(cam.height)

        avg_focal = float(np.mean(focals))
        width     = int(np.mean(widths))
        height    = int(np.mean(heights))

        # Camera intrinsics entry (PINHOLE model, model_id=1)
        new_cameras[new_cam_id] = {
            "model_id": 1,   # PINHOLE
            "width":  width,
            "height": height,
            "params": [avg_focal, avg_focal,       # fx, fy
                       width / 2.0, height / 2.0], # cx, cy
        }

        # Camera pose entry
        cfw = mid_image.cam_from_world()
        q   = cfw.rotation.quat
        t   = cfw.translation

        # Image name must EXACTLY match file in images/ folder
        image_file = f"{cam_name}.jpg"

        new_images[new_cam_id] = {
            "qw": float(q[3]), "qx": float(q[0]),
            "qy": float(q[1]), "qz": float(q[2]),
            "tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
            "camera_id": new_cam_id,
            "name": image_file,
        }

    # Filter to only training cameras for the training sparse
    train_camera_ids = {cid: cam for cid, cam in new_cameras.items()
                        if new_images[cid]["name"].split(".")[0] in train_cams}
    train_image_ids  = {iid: img for iid, img in new_images.items()
                        if img["name"].split(".")[0] in train_cams}

    write_cameras_bin(train_camera_ids, train_sparse / "cameras.bin")
    write_images_bin(train_image_ids, train_sparse / "images.bin")

    # Copy point cloud (same for all — full MASt3R reconstruction)
    shutil.copy2(workspace_sparse / "points3D.bin", train_sparse / "points3D.bin")

    # Also save full 4-camera sparse (for evaluation rendering later)
    full_sparse = scene_proc / "static" / "sparse" / "0"
    full_sparse.mkdir(parents=True, exist_ok=True)
    write_cameras_bin(new_cameras, full_sparse / "cameras.bin")
    write_images_bin(new_images, full_sparse / "images.bin")
    shutil.copy2(workspace_sparse / "points3D.bin", full_sparse / "points3D.bin")

    n_train = len(train_camera_ids)
    print(f"  ✅ Rebuilt sparse: {n_train} training cameras")
    print(f"     Training: {train_cams}")
    print(f"     Test:     {test_cams}")
    return new_images  # return full dict (including test camera) for evaluation use


def setup_train_images(scene_name: str, train_cams: list[str]):
    """
    Create static_train/images/ with ONLY training camera images.
    The model must never see cam04 during training.
    """
    scene_proc  = PROC_DIR / scene_name
    src_images  = scene_proc / "static" / "images"
    train_images = scene_proc / "static_train" / "images"
    train_images.mkdir(parents=True, exist_ok=True)

    for cam in train_cams:
        src = src_images / f"{cam}.jpg"
        dst = train_images / f"{cam}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    n = len(list(train_images.glob("*.jpg")))
    print(f"  ✅ Training images: {n} ({train_cams})")


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(scene_name: str, experiment_name: str,
                 iterations: int, sh_degree: int,
                 extra_args: list[str] = None) -> Path:
    """
    Run gaussian-splatting/train.py on the static training dataset.

    Args:
        scene_name:       e.g. 'scene_S004'
        experiment_name:  e.g. 'baseline' → saved under experiments/static/scene_S004/baseline/
        iterations:       number of training iterations (default: 30000)
        sh_degree:        spherical harmonic degree (0-3)
        extra_args:       additional CLI arguments

    Returns:
        Path to the trained model directory
    """
    scene_proc  = PROC_DIR / scene_name
    source_path = scene_proc / "static_train"
    model_path  = EXP_DIR / scene_name / experiment_name

    model_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(GS_DIR / "train.py"),
        "--source_path",    str(source_path),
        "--model_path",     str(model_path),
        "--iterations",     str(iterations),
        "--sh_degree",      str(sh_degree),
        "--test_iterations", str(iterations),       # evaluate at end
        "--save_iterations", str(iterations),       # save at end
        "--checkpoint_iterations",                  # also save mid-training
            str(iterations // 2), str(iterations),
        "--quiet",                                   # suppress verbose logging
        "--disable_viewer",                          # no GUI
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Running: {' '.join(cmd[-8:])}")
    print(f"  Model will be saved to: {model_path}")

    # Set environment
    cuda_home = str(Path.home() / "cuda-home")
    env = {
        **os.environ,
        "CUDA_HOME":  cuda_home,
        "CC":          "/usr/bin/gcc-12",
        "CXX":         "/usr/bin/g++-12",
        "PYTHONPATH":  str(GS_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":        f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
    }

    t_start = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(GS_DIR))
    elapsed = time.time() - t_start

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")

    print(f"\n  ✅ Training complete in {elapsed/60:.1f} minutes")

    # Save training config
    config = {
        "scene": scene_name,
        "experiment": experiment_name,
        "source_path": str(source_path),
        "iterations": iterations,
        "sh_degree": sh_degree,
        "train_time_minutes": elapsed / 60,
    }
    with open(model_path / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model_path


def count_gaussians(model_path: Path, iteration: int) -> int:
    """Count the number of Gaussians in a trained model."""
    try:
        from plyfile import PlyData
        ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            return -1
        data = PlyData.read(str(ply_path))
        return len(data["vertex"])
    except Exception:
        return -1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train static 3DGS model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--scene",       type=str, default="scene_S004")
    parser.add_argument("--experiment",  type=str, default="baseline",
                        help="Experiment name (subfolder under experiments/static/)")
    parser.add_argument("--iterations",  type=int, default=30_000)
    parser.add_argument("--sh_degree",   type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help="Spherical harmonic degree for view-dependent color")
    parser.add_argument("--train_cams",  nargs="+", default=["cam01", "cam02", "cam03"],
                        help="Cameras to train on")
    parser.add_argument("--test_cams",   nargs="+", default=["cam04"],
                        help="Cameras held out for evaluation")
    parser.add_argument("--skip_rebuild", action="store_true",
                        help="Skip sparse rebuild (reuse existing)")
    # Unknown args (e.g. --densify_until_iter) are forwarded to gaussian-splatting/train.py
    args, extra_train_args = parser.parse_known_args()

    print(f"\n{'='*60}")
    print(f"Phase 1: Static 3DGS Training")
    print(f"  Scene:      {args.scene}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Iterations: {args.iterations:,}")
    print(f"  SH degree:  {args.sh_degree}")
    print(f"  Train cams: {args.train_cams}")
    print(f"  Test cams:  {args.test_cams}")
    print(f"{'='*60}")

    # Step 1: Fix the rasterizer
    print("\n[1/4] Ensuring correct CUDA rasterizer...")
    ensure_static_rasterizer()

    # Step 2: Rebuild sparse with correct entries
    if not args.skip_rebuild:
        print("\n[2/4] Rebuilding static sparse reconstruction...")
        sys.path.insert(0, str(SCRIPTS_DIR))
        rebuild_static_sparse(args.scene, args.train_cams, args.test_cams)
        setup_train_images(args.scene, args.train_cams)
    else:
        print("\n[2/4] Skipping sparse rebuild (--skip_rebuild set)")

    # Step 3: Train
    print(f"\n[3/4] Training static 3DGS ({args.iterations:,} iterations)...")
    print(f"      Expected time: ~{args.iterations/30000*35:.0f} minutes on RTX 3080")
    if extra_train_args:
        print(f"      Forwarding extra args to train.py: {extra_train_args}")
    model_path = run_training(
        scene_name=args.scene,
        experiment_name=args.experiment,
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        extra_args=extra_train_args,
    )

    # Step 4: Report
    print(f"\n[4/4] Training summary:")
    n_gaussians = count_gaussians(model_path, args.iterations)
    print(f"  Gaussians:    {n_gaussians:,}")
    print(f"  Model saved:  {model_path}")
    print(f"\n  Next: python scripts/06_evaluate.py --scene {args.scene} --experiment {args.experiment}")


if __name__ == "__main__":
    main()
