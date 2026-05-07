"""
02_run_colmap.py — Camera Pose Estimation via COLMAP
=====================================================

Runs the full COLMAP Structure-from-Motion (SfM) pipeline on the reference
frames extracted in step 01, producing:
  - Camera intrinsics  (focal length, principal point, distortion coefficients)
  - Camera extrinsics  (rotation R and translation t for each camera)
  - Sparse 3D point cloud (triangulated feature matches)

These outputs are the PREREQUISITE for 3DGS training. Without accurate camera
poses, the Gaussians have no coordinate frame to be placed in.

THE COLMAP PIPELINE (3 stages)
--------------------------------

Stage 1: Feature Extraction
  For each image, detect keypoints (distinctive local regions) and compute
  descriptors (fingerprints that identify those regions).
  We use SIFT (Scale-Invariant Feature Transform):
    - Detects corners/blobs at multiple scales
    - Produces 128-dimensional descriptor per keypoint
    - "Scale invariant": a keypoint at 1x zoom matches the same point at 2x zoom
    - "Rotation invariant": works regardless of image orientation
  COLMAP typically finds 5,000-50,000 keypoints per image.

Stage 2: Feature Matching
  For each pair of images, find which keypoints in image A correspond to the
  same 3D point as keypoints in image B. We use "exhaustive matching" which
  compares every pair — appropriate here because we only have 4 images.
  (For datasets with 100s of images, use sequential or vocabulary tree matching.)
  
  After matching, COLMAP runs geometric verification:
    - Estimates fundamental matrix / homography between each pair
    - Rejects "outlier" matches that aren't geometrically consistent (RANSAC)
    - Keeps "inlier" matches that agree with a consistent geometric relationship

Stage 3: Sparse Reconstruction (Incremental Mapping)
  Given the verified matches, triangulate 3D point positions and estimate
  camera poses simultaneously. This is a bundle adjustment problem:
    - Start with 2 cameras, initialize their poses from the essential matrix
    - Add cameras one by one, estimating each new pose from matches to existing 3D points
    - After each addition, run bundle adjustment to minimize reprojection error
    - Reprojection error = distance between (a) where a 3D point projects into an image
      and (b) where it was actually observed as a keypoint
  
  The output is the sparse reconstruction: camera poses + sparse 3D point cloud.

WHY ACCURACY MATTERS
---------------------
3DGS is trained by rendering from known camera poses and comparing to real images.
If poses are wrong by even 1-2 degrees, the model will try to explain the
inconsistency by adding geometry that doesn't exist (floaters) or smearing
geometry across space (blurring). Pose accuracy directly determines output quality.

For our dataset: 4 synchronized fixed cameras → relatively easy for COLMAP.
The challenge: each camera only overlaps partially with neighbors. COLMAP needs
feature matches between cameras to triangulate the baseline. If adjacent cameras
share too few visible features, the reconstruction will fail.

CAMERA MODEL: OPENCV
---------------------
We use the OPENCV camera model which parameterizes:
  - fx, fy: focal lengths (pixels)
  - cx, cy: principal point (image center, typically W/2, H/2)
  - k1, k2: radial distortion coefficients
  - p1, p2: tangential distortion coefficients
  
  Distortion model: x_distorted = x(1 + k1*r² + k2*r⁴) + 2p1*xy + p2*(r²+2x²)
                   y_distorted = y(1 + k1*r² + k2*r⁴) + p1*(r²+2y²) + 2p2*xy
  
  WHY OPENCV not PINHOLE: Our cameras show barrel distortion (straight lines
  appear curved). PINHOLE ignores this → bad reprojection errors at image edges
  → Gaussians placed in wrong locations near frame borders.
  
  If k1 ≈ 0 after optimization: distortion was negligible anyway. No harm done.
  If k1 ≠ 0: we correctly accounted for it. 3DGS will get undistorted images.

OUTPUT FORMAT
-------------
data/processed/scene_S003/
└── colmap_workspace/
    ├── database.db          ← COLMAP's SQLite database (features + matches)
    ├── images/              ← Symlink to colmap_input/ (reference frames)
    └── sparse/
        └── 0/
            ├── cameras.bin  ← Camera intrinsics (focal length, distortion etc.)
            ├── images.bin   ← Camera extrinsics (R, t per image) + image filenames
            └── points3D.bin ← Sparse 3D points with colors and track info

Usage:
    python scripts/02_run_colmap.py --scene scene_S003
    python scripts/02_run_colmap.py --scene scene_S003 --camera_model PINHOLE
    python scripts/02_run_colmap.py --scene all
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def run_colmap_pipeline(scene_name: str,
                         camera_model: str = "OPENCV",
                         max_num_features: int = 8192,
                         min_num_inliers: int = 15) -> pycolmap.Reconstruction:
    """
    Run the complete COLMAP SfM pipeline on a scene's reference frames.

    Args:
        scene_name:         e.g. 'scene_S003'
        camera_model:       'OPENCV' (with distortion) or 'PINHOLE' (without)
        max_num_features:   Max SIFT keypoints per image (more = slower matching)
        min_num_inliers:    Min verified matches needed between an image pair
                            to accept it as a valid overlap. Raise if you get
                            spurious connections; lower if cameras barely overlap.

    Returns:
        The pycolmap Reconstruction object (contains all cameras + points)
    """

    scene_proc = PROC_DIR / scene_name
    colmap_input_dir = scene_proc / "colmap_input"
    workspace_dir    = scene_proc / "colmap_workspace"
    sparse_dir       = workspace_dir / "sparse" / "0"

    # Validate inputs
    if not colmap_input_dir.exists():
        raise FileNotFoundError(
            f"COLMAP input not found: {colmap_input_dir}\n"
            f"Run 01_extract_frames.py first."
        )

    images = list(colmap_input_dir.glob("*.jpg"))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {colmap_input_dir}")

    # Parse which cameras are present (from filenames like cam01_ref00.jpg)
    cam_names = sorted(set(
        im.stem.split("_ref")[0] if "_ref" in im.stem else im.stem
        for im in images
    ))

    print(f"\n{'='*60}")
    print(f"COLMAP SfM: {scene_name}")
    print(f"  Input images:      {len(images)}  ({[im.name for im in sorted(images)][:6]}...)")
    print(f"  Cameras detected:  {cam_names}")
    print(f"  Frames per camera: {len(images) // max(len(cam_names), 1)}")
    print(f"  Camera model:      {camera_model}")
    print(f"  Max features:      {max_num_features}")
    print(f"  Workspace:         {workspace_dir}")
    print(f"{'='*60}")

    # Create workspace
    workspace_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    db_path = workspace_dir / "database.db"

    # ── Stage 1: Feature Extraction ────────────────────────────────────────────
    print("\n[1/3] Extracting SIFT features...")

    # pycolmap 4.0 API: FeatureExtractionOptions wraps sift + gpu settings
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.sift.max_num_features = max_num_features
    extract_opts.use_gpu = True
    extract_opts.gpu_index = "0"

    # ImageReaderOptions: set camera model here
    # Camera mode: SINGLE means all images share ONE camera model.
    # We have 4 different cameras but with the same lens type → SINGLE is a
    # reasonable approximation. In production you'd use PER_IMAGE with priors.
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model  # "OPENCV" or "PINHOLE"

    pycolmap.extract_features(
        database_path=db_path,
        image_path=colmap_input_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extract_opts,
    )
    print("  ✅ Feature extraction complete")

    # ── Stage 2: Feature Matching ───────────────────────────────────────────────
    print("\n[2/3] Matching features between image pairs...")

    # Exhaustive matching: compare ALL pairs.
    # For N=4 images: 4C2 = 6 pairs. Trivial.
    # For N=100 images: 100C2 = 4950 pairs. Still OK.
    # For N=1000 images: 499,500 pairs. Too slow → use sequential or vocab tree.
    # pycolmap 4.0 API: FeatureMatchingOptions
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = True
    match_opts.gpu_index = "0"

    # TwoViewGeometryOptions controls geometric verification
    verify_opts = pycolmap.TwoViewGeometryOptions()
    verify_opts.min_num_inliers = min_num_inliers

    pycolmap.match_exhaustive(
        database_path=db_path,
        matching_options=match_opts,
        verification_options=verify_opts,
    )
    print("  ✅ Feature matching complete")

    # Report match statistics from database
    try:
        db = pycolmap.Database(str(db_path))
        two_view_geoms = db.read_all_two_view_geometries()
        valid_pairs = [(k, v) for k, v in two_view_geoms.items()
                       if v.inlier_matches is not None and
                       len(v.inlier_matches) >= min_num_inliers]
        print(f"  Valid image pairs with ≥{min_num_inliers} inliers: {len(valid_pairs)}/{len(two_view_geoms)}")
        for (im1_id, im2_id), geom in sorted(valid_pairs):
            n_inliers = len(geom.inlier_matches)
            print(f"    pair ({im1_id},{im2_id}): {n_inliers} inliers")
    except Exception as e:
        print(f"  (Could not read match stats: {e})")

    # ── Stage 3: Sparse Reconstruction ─────────────────────────────────────────
    print("\n[3/3] Running sparse reconstruction (incremental mapping)...")

    mapper_opts = pycolmap.IncrementalPipelineOptions()
    # Minimum number of matches per image to register it into the reconstruction.
    # With only 4 cameras, we want to be permissive.
    mapper_opts.min_num_matches = 10

    # Run incremental mapping
    # Returns a dict of {reconstruction_id: Reconstruction}
    # Usually just one reconstruction (id=0) unless the scene is disconnected
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=colmap_input_dir,
        output_path=workspace_dir / "sparse",
        options=mapper_opts,
    )

    if len(maps) == 0:
        raise RuntimeError(
            "COLMAP reconstruction FAILED — no reconstruction produced.\n"
            "Possible causes:\n"
            "  1. Too few feature matches between cameras (cameras don't overlap enough)\n"
            "  2. Too few images (need at least 2 with good overlap)\n"
            "  3. Feature extraction failed (check GPU availability)\n"
            "Try: --camera_model PINHOLE, or check the reference frames visually."
        )

    # Take the largest reconstruction if multiple exist
    best_recon = max(maps.values(), key=lambda r: len(r.images))

    print(f"\n  ✅ Reconstruction complete!")
    print(f"     Registered images: {len(best_recon.images)} / {len(images)}")
    print(f"     3D points:         {len(best_recon.points3D)}")
    print(f"     Mean track length: {best_recon.compute_mean_track_length():.2f}")
    print(f"     Mean reprojection error: {best_recon.compute_mean_reprojection_error():.3f} px")

    # ── Validate: check all cameras were registered ─────────────────────────────
    n_registered = len(best_recon.images)
    if n_registered < len(images):
        unregistered = len(images) - n_registered
        print(f"\n  ⚠️  WARNING: {unregistered} image(s) could not be registered!")
        print(f"     This means some cameras have no estimated pose.")
        print(f"     Those cameras CANNOT be used for training.")
        print(f"     Check if those cameras have visual overlap with the others.")

    # ── Write output ────────────────────────────────────────────────────────────
    # Save the best reconstruction to sparse/0/ in COLMAP binary format
    # This is the format that 3DGS (gaussian-splatting) reads directly.
    best_recon.write(workspace_dir / "sparse" / "0")
    print(f"\n  Saved to: {workspace_dir / 'sparse' / '0'}")

    # ── Print camera details ────────────────────────────────────────────────────
    print("\n  Camera parameters:")
    for cam_id, cam in best_recon.cameras.items():
        print(f"    Camera {cam_id}: {cam.model.name}")
        print(f"      Image size: {cam.width} x {cam.height}")
        params = cam.params
        if camera_model == "OPENCV":
            print(f"      fx={params[0]:.1f}, fy={params[1]:.1f}")
            print(f"      cx={params[2]:.1f}, cy={params[3]:.1f}")
            if len(params) > 4:
                print(f"      k1={params[4]:.6f}, k2={params[5]:.6f}")
                if len(params) > 6:
                    print(f"      p1={params[6]:.6f}, p2={params[7]:.6f}")
        else:
            print(f"      fx={params[0]:.1f}, fy={params[1]:.1f}")
            print(f"      cx={params[2]:.1f}, cy={params[3]:.1f}")

    # ── Print pose details ──────────────────────────────────────────────────────
    print("\n  Image poses (camera-to-world translation):")
    for im_id, image in sorted(best_recon.images.items()):
        # image.cam_from_world is a Rigid3d: rotation + translation
        # translation = position of world origin in camera space
        # camera position in world = -R^T * t
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        cam_pos_world = -R.T @ t  # Camera position in world coordinates
        print(f"    {image.name}: world_pos = [{cam_pos_world[0]:.3f}, "
              f"{cam_pos_world[1]:.3f}, {cam_pos_world[2]:.3f}]")

    # ── Save metadata ────────────────────────────────────────────────────────────
    meta = {
        "scene": scene_name,
        "camera_model": camera_model,
        "n_images_input":     len(images),
        "n_images_registered": n_registered,
        "n_points3D":         len(best_recon.points3D),
        "mean_track_length":  best_recon.compute_mean_track_length(),
        "mean_reprojection_error_px": best_recon.compute_mean_reprojection_error(),
    }
    with open(workspace_dir / "colmap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return best_recon


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM to estimate camera poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--scene", type=str, default="scene_S003",
                        help="Scene name or 'all'")
    parser.add_argument("--camera_model", type=str, default="OPENCV",
                        choices=["OPENCV", "PINHOLE"],
                        help="Camera model (OPENCV handles distortion, PINHOLE doesn't)")
    parser.add_argument("--max_num_features", type=int, default=8192,
                        help="Max SIFT features per image (default: 8192)")
    parser.add_argument("--min_num_inliers", type=int, default=15,
                        help="Min inlier matches to accept a camera pair (default: 15)")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted([d.name for d in PROC_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("scene_")])
    else:
        scenes = [args.scene]

    for scene in scenes:
        try:
            recon = run_colmap_pipeline(
                scene_name=scene,
                camera_model=args.camera_model,
                max_num_features=args.max_num_features,
                min_num_inliers=args.min_num_inliers,
            )
            print(f"\n✅ {scene}: COLMAP done — {len(recon.images)} cameras, "
                  f"{len(recon.points3D)} 3D points")
        except Exception as e:
            print(f"\n❌ {scene}: COLMAP failed — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
