"""
02b_run_mast3r.py — Camera Pose Estimation via MASt3R
======================================================

WHY MASt3R INSTEAD OF COLMAP
------------------------------
Standard COLMAP (SfM) requires:
  1. Many distinctive feature keypoints per image
  2. Feature matches between image pairs (>15 inlier matches per pair)
  3. A non-degenerate baseline between matched pairs

Our dataset fails condition 2: surveillance cameras placed to maximize room
coverage have minimal visual overlap. COLMAP finds 0-20 cross-camera inlier
matches — not enough for reliable essential matrix estimation.

MASt3R (Matching And Stereo 3D Reconstruction, Leroy et al. 2024 - CVPR 2024)
takes a fundamentally different approach:

  Instead of detecting handcrafted SIFT keypoints and matching descriptors,
  MASt3R uses a large Vision Transformer (ViT-Large) trained on millions of
  image pairs to:
    1. Simultaneously detect and match features across a pair of images
    2. Estimate the 3D structure visible in both images
    3. Predict per-pixel 3D point maps in each camera's coordinate frame

  From the predicted 3D point maps + correspondences, camera poses are
  recovered via point cloud alignment (Procrustes / bundle adjustment).
  This works even with 10-20% visual overlap, where SIFT-COLMAP fails.

THE MAST3R PIPELINE
--------------------
For N images, MASt3R runs pairwise inference on all N*(N-1)/2 pairs:

  For each pair (img_i, img_j):
    1. Encode both images with ViT backbone
    2. Cross-attend features between the two images (like a stereo network)
    3. Predict: 
       - pointmap_i: [H, W, 3] — 3D coords of each pixel in img_i's frame
       - pointmap_j: [H, W, 3] — 3D coords of each pixel in img_j's frame
       - confidence_i, confidence_j: [H, W] — per-pixel reliability
    4. From the overlapping points, estimate relative pose (R, t) between cameras

  Global alignment then solves a joint optimization over all pairwise
  predictions to find globally consistent poses.

For our 12 images (3 frames × 4 cameras): 12*11/2 = 66 pairs. Reasonable.

OUTPUT FORMAT
-------------
Same as COLMAP: sparse/0/ with cameras.bin, images.bin, points3D.bin.
So the downstream pipeline (03_prepare_dataset.py) doesn't need to change.

MEMORY REQUIREMENTS
-------------------
MASt3R processes image pairs at 512px resolution.
Each pair requires ~2GB VRAM. With 12 images total, processed in batches.
RTX 3080 (16GB) can handle this comfortably.

Usage:
    python scripts/02b_run_mast3r.py --scene scene_S004
    python scripts/02b_run_mast3r.py --scene all
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2

# Add MASt3R to path
MAST3R_DIR = Path(__file__).parent.parent / "repos" / "mast3r"
DUST3R_DIR = MAST3R_DIR / "dust3r"
sys.path.insert(0, str(MAST3R_DIR))
sys.path.insert(0, str(DUST3R_DIR))

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
CHECKPOINT   = MAST3R_DIR / "checkpoints" / "model.safetensors"


def load_mast3r_model(device: str = "cuda") -> object:
    """Load the MASt3R model from checkpoint."""
    from mast3r.model import AsymmetricMASt3R

    print(f"Loading MASt3R model from {CHECKPOINT}...")
    model = AsymmetricMASt3R.from_pretrained(
        str(MAST3R_DIR / "checkpoints")
    )
    model = model.to(device).eval()
    print(f"  ✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model


def run_mast3r_pipeline(scene_name: str,
                         image_size: int = 512,
                         device: str = "cuda") -> None:
    """
    Run MASt3R to estimate camera poses from the COLMAP input images.

    This replaces 02_run_colmap.py for scenes where COLMAP fails due to
    insufficient cross-camera feature overlap.

    Args:
        scene_name:  e.g. 'scene_S004'
        image_size:  Resolution for MASt3R inference (512 recommended)
        device:      'cuda' or 'cpu'
    """
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    scene_proc    = PROC_DIR / scene_name
    colmap_input  = scene_proc / "colmap_input"
    workspace_dir = scene_proc / "colmap_workspace"
    sparse_dir    = workspace_dir / "sparse" / "0"

    if not colmap_input.exists():
        raise FileNotFoundError(
            f"COLMAP input not found: {colmap_input}\n"
            f"Run 01_extract_frames.py first."
        )

    image_paths = sorted(colmap_input.glob("*.jpg"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images in {colmap_input}")

    print(f"\n{'='*60}")
    print(f"MASt3R Pose Estimation: {scene_name}")
    print(f"  Images: {len(image_paths)}")
    print(f"  Resolution: {image_size}px")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── Load model ──────────────────────────────────────────────────────────────
    model = AsymmetricMASt3R.from_pretrained(str(MAST3R_DIR / "checkpoints"))
    model = model.to(device).eval()
    print(f"  ✅ Model loaded")

    # ── Load images ─────────────────────────────────────────────────────────────
    # MASt3R's load_images resizes to image_size while preserving aspect ratio
    print(f"\n[1/3] Loading and preprocessing images...")
    images = load_images([str(p) for p in image_paths],
                          size=image_size, verbose=True)
    print(f"  Loaded {len(images)} images at {image_size}px")

    # ── Generate pairs and run pairwise inference ───────────────────────────────
    # make_pairs: creates all N*(N-1)/2 image pairs
    # prefilter_pairs: optionally skip obviously non-overlapping pairs
    print(f"\n[2/3] Running pairwise MASt3R inference...")
    n_pairs = len(images) * (len(images) - 1) // 2
    print(f"  Processing {n_pairs} image pairs...")

    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    # Run inference: the core MASt3R forward pass
    # For each pair: (view1, view2) → (pred1, pred2)
    # Each pred contains: 'pts3d' (pointmap), 'conf' (confidence)
    with torch.no_grad():
        output = inference(pairs, model, device,
                          batch_size=1, verbose=True)

    print(f"  ✅ Pairwise inference complete")

    # ── Global alignment ────────────────────────────────────────────────────────
    # GlobalAligner solves for globally consistent camera poses
    # by minimizing the inconsistencies between all pairwise predictions.
    #
    # This is analogous to bundle adjustment in COLMAP:
    # - Find camera poses R, t for all images
    # - Find 3D point positions X
    # - Minimize: Σ ||proj(X, R_i, t_i) - x_i||²
    # But instead of reprojection error, MASt3R minimizes 3D point map errors.
    #
    # GlobalAlignerMode options:
    #   POINTCLOUD_WITH_KNOWN_CAMERA_FOCAL: if focal length known
    #   POINTCLOUD: fully unconstrained (focal + poses all estimated)
    print(f"\n[3/3] Running global alignment (bundle adjustment)...")

    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=True,
    )

    # Optimize: joint optimization of all poses and 3D structure
    loss = scene.compute_global_alignment(
        init="mst",            # minimum spanning tree initialization
        niter=300,             # optimization iterations
        schedule="cosine",     # learning rate schedule
        lr=0.01,               # initial learning rate
    )
    print(f"  ✅ Global alignment converged (final loss: {loss:.4f})")

    # ── Extract results ──────────────────────────────────────────────────────────
    poses     = scene.get_im_poses()         # [N, 4, 4] camera-to-world transforms
    focals    = scene.get_focals()           # [N] focal lengths (pixels)
    pts3d     = scene.get_pts3d()            # [N, H, W, 3] point maps
    conf      = scene.get_conf()             # [N, H, W] confidence maps
    imgs_orig = scene.imgs                   # original images for color

    n_images = len(image_paths)
    print(f"\n  Results:")
    print(f"  Camera poses:    {poses.shape}")
    print(f"  Focal lengths:   {focals.detach().cpu().numpy().round(1).tolist()}")

    # ── Convert to COLMAP format ─────────────────────────────────────────────────
    # We need to write cameras.bin, images.bin, points3D.bin
    # in COLMAP's binary format so the downstream pipeline works unchanged.
    print(f"\n  Converting to COLMAP format...")
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Camera intrinsics
    # MASt3R estimates one focal length per image (SIMPLE_PINHOLE model)
    # For 3DGS, we need (fx, fy, cx, cy) — use focal for both fx, fy
    # and image center for cx, cy
    true_shape = images[0]["true_shape"]  # tensor of shape [1, 2] or [2]
    if true_shape.ndim == 2:
        img_h, img_w = int(true_shape[0, 0]), int(true_shape[0, 1])
    else:
        img_h, img_w = int(true_shape[0]), int(true_shape[1])

    cameras_dict = {}
    for i in range(n_images):
        f = float(focals[i].detach().cpu().item())
        # Scale focal length back to original image resolution
        # (MASt3R resizes to image_size, we want pose in original coords)
        orig_img = cv2.imread(str(image_paths[i]))
        orig_h, orig_w = orig_img.shape[:2]
        scale_x = orig_w / img_w
        scale_y = orig_h / img_h
        f_orig = f * ((scale_x + scale_y) / 2)  # average scale

        cameras_dict[i + 1] = {
            "model_id": 1,   # PINHOLE model
            "width":  orig_w,
            "height": orig_h,
            "params": [f_orig, f_orig,      # fx, fy
                       orig_w / 2.0,         # cx
                       orig_h / 2.0],        # cy
        }

    # Camera extrinsics (world-to-camera transforms)
    # MASt3R gives camera-to-world (4x4), we need world-to-camera (R, t)
    images_dict = {}
    for i in range(n_images):
        c2w = poses[i].detach().cpu().numpy()  # [4,4] camera-to-world
        # Invert to get world-to-camera
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Convert rotation matrix to quaternion (qw, qx, qy, qz)
        # Using scipy for stability
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]

        images_dict[i + 1] = {
            "qw": float(q[3]),
            "qx": float(q[0]),
            "qy": float(q[1]),
            "qz": float(q[2]),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "tz": float(t[2]),
            "camera_id": i + 1,  # one camera model per image
            "name": image_paths[i].name,
        }

    # Sparse 3D point cloud: sample high-confidence points from the point maps
    # This is our replacement for COLMAP's triangulated point cloud.
    # 3DGS uses this as initialization for the Gaussian positions.
    print(f"  Building sparse point cloud from MASt3R predictions...")
    pts_xyz = []
    pts_rgb = []

    for i in range(n_images):
        pts = pts3d[i].detach().cpu().numpy().reshape(-1, 3)   # [H*W, 3]
        cf  = conf[i].detach().cpu().numpy().reshape(-1)       # [H*W]
        img_rgb = (imgs_orig[i] * 255).astype(np.uint8).reshape(-1, 3)

        # Keep only high-confidence points (top 20%)
        threshold = np.percentile(cf, 80)
        mask = cf > threshold

        pts_xyz.append(pts[mask])
        pts_rgb.append(img_rgb[mask])

    all_pts = np.concatenate(pts_xyz, axis=0)
    all_rgb = np.concatenate(pts_rgb, axis=0)

    # Subsample to keep point cloud manageable (COLMAP typically has 10k-100k points)
    if len(all_pts) > 50000:
        idx = np.random.choice(len(all_pts), 50000, replace=False)
        all_pts = all_pts[idx]
        all_rgb = all_rgb[idx]

    print(f"  Sparse point cloud: {len(all_pts)} points")

    # Write to COLMAP binary format
    # Import writers from 03_prepare_dataset.py
    sys.path.insert(0, str(Path(__file__).parent))
    from _colmap_io import write_cameras_bin, write_images_bin, write_points3d_bin

    points3D_dict = {
        i + 1: {
            "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
            "r": int(rgb[0]), "g": int(rgb[1]), "b": int(rgb[2]),
            "error": 1.0,
            "track": [],  # no track info from MASt3R
        }
        for i, (pt, rgb) in enumerate(zip(all_pts, all_rgb))
    }

    write_cameras_bin(cameras_dict, sparse_dir / "cameras.bin")
    write_images_bin(images_dict, sparse_dir / "images.bin")
    write_points3d_bin(points3D_dict, sparse_dir / "points3D.bin")

    print(f"  ✅ COLMAP format files written to {sparse_dir}")

    # ── Print pose summary ───────────────────────────────────────────────────────
    print(f"\n  Camera positions (world coordinates):")
    for i, im_path in enumerate(image_paths):
        c2w = poses[i].detach().cpu().numpy()
        pos = c2w[:3, 3]  # camera position in world
        print(f"    {im_path.name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # ── Save metadata ────────────────────────────────────────────────────────────
    meta = {
        "scene": scene_name,
        "method": "MASt3R",
        "n_images": n_images,
        "image_size_for_inference": image_size,
        "final_loss": float(loss),
        "n_points3D": len(all_pts),
        "focal_lengths": focals.detach().cpu().numpy().tolist(),
    }
    with open(workspace_dir / "mast3r_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ MASt3R pose estimation complete!")
    print(f"   Results: {sparse_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate camera poses with MASt3R (for limited-overlap scenes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--scene", type=str, default="scene_S004",
                        help="Scene name or 'all'")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for MASt3R inference (default: 512)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted([d.name for d in PROC_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("scene_")])
    else:
        scenes = [args.scene]

    for scene in scenes:
        try:
            run_mast3r_pipeline(scene, args.image_size, args.device)
        except Exception as e:
            print(f"❌ {scene}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
