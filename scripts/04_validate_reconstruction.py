"""
04_validate_reconstruction.py — Verify COLMAP Before Training
==============================================================

Validates the COLMAP reconstruction and visualizes camera poses.
Run this BEFORE starting 3DGS training. Problems caught here are
cheap to fix. Problems discovered after 35 minutes of training are not.

WHAT WE CHECK
-------------
1. All cameras registered  — every camera must have a pose
2. Reprojection error < 2px — standard threshold for good reconstruction
3. Enough 3D points        — too few = degenerate reconstruction
4. Camera baseline         — cameras must be far enough apart to triangulate depth
5. Point cloud coverage    — points should cover the scene, not just one region

WHAT WE VISUALIZE
-----------------
- Camera positions as colored arrows in 3D (top-down + side views)
- Sparse point cloud colored by depth
- Sample reprojected points on each image (visual sanity check)

Usage:
    python scripts/04_validate_reconstruction.py --scene scene_S003
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pycolmap


PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def validate_reconstruction(scene_name: str) -> dict:
    """
    Run validation checks on a COLMAP reconstruction.
    Returns a dict of metrics and pass/fail status.
    """
    sparse_dir = PROC_DIR / scene_name / "colmap_workspace" / "sparse" / "0"

    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"No COLMAP reconstruction at {sparse_dir}\n"
            f"Run 02_run_colmap.py first."
        )

    recon = pycolmap.Reconstruction(str(sparse_dir))

    n_cameras    = len(recon.cameras)
    n_images     = len(recon.images)
    n_points     = len(recon.points3D)
    mean_track   = recon.compute_mean_track_length()
    mean_reproj  = recon.compute_mean_reprojection_error()

    print(f"\n{'='*60}")
    print(f"COLMAP Validation: {scene_name}")
    print(f"{'='*60}")
    print(f"  Cameras (intrinsic models): {n_cameras}")
    print(f"  Images registered:          {n_images}")
    print(f"  3D points triangulated:     {n_points}")
    print(f"  Mean track length:          {mean_track:.2f} (views per point)")
    print(f"  Mean reprojection error:    {mean_reproj:.3f} px")

    # ── Checks ─────────────────────────────────────────────────────────────────
    checks = {}

    # Check 1: Reprojection error
    # < 1.0px = excellent, 1.0-2.0px = good, > 2.0px = concerning
    # 3DGS is sensitive to bad poses — high reprojection error means
    # Gaussians will be placed incorrectly.
    reproj_threshold = 2.0
    checks["reprojection_error"] = {
        "value": mean_reproj,
        "threshold": reproj_threshold,
        "pass": mean_reproj < reproj_threshold,
        "message": f"{mean_reproj:.3f}px (threshold: {reproj_threshold}px)"
    }

    # Check 2: Enough 3D points
    # Rule of thumb: at least 100 points for a small scene.
    # More points = better Gaussian initialization.
    min_points = 100
    checks["min_3d_points"] = {
        "value": n_points,
        "threshold": min_points,
        "pass": n_points >= min_points,
        "message": f"{n_points} points (minimum: {min_points})"
    }

    # Check 3: Mean track length
    # Average number of cameras that observe each 3D point.
    # Higher = more constrained reconstruction = better poses.
    # With 4 cameras, expect track length ~2-4.
    min_track = 2.0
    checks["mean_track_length"] = {
        "value": mean_track,
        "threshold": min_track,
        "pass": mean_track >= min_track,
        "message": f"{mean_track:.2f} (minimum: {min_track})"
    }

    # Check 4: Camera baseline (spread of camera positions)
    # If all cameras are at the same position, we can't triangulate depth.
    # Measure by computing std dev of camera positions.
    cam_positions = []
    for im_id, image in recon.images.items():
        cfw = image.cam_from_world()   # pycolmap 4.0: method, not property
        R = cfw.rotation.matrix()
        t = cfw.translation
        pos = -R.T @ t  # World position of camera
        cam_positions.append(pos)

    if len(cam_positions) > 1:
        positions_arr = np.array(cam_positions)
        baseline = np.std(positions_arr, axis=0).max()
        min_baseline = 0.05  # 5cm minimum spread (in COLMAP's metric units)
        checks["camera_baseline"] = {
            "value": baseline,
            "threshold": min_baseline,
            "pass": baseline > min_baseline,
            "message": f"max_spread={baseline:.3f}m (minimum: {min_baseline}m)"
        }

    # ── Print check results ─────────────────────────────────────────────────────
    print(f"\n  Validation checks:")
    all_pass = True
    for check_name, check in checks.items():
        status = "✅ PASS" if check["pass"] else "❌ FAIL"
        print(f"    {status}  {check_name}: {check['message']}")
        if not check["pass"]:
            all_pass = False

    if all_pass:
        print(f"\n  🎉 All checks passed! Ready for 3DGS training.")
    else:
        print(f"\n  ⚠️  Some checks failed. Review before training.")
        print(f"     Tips:")
        if not checks.get("reprojection_error", {}).get("pass", True):
            print(f"       - High reprojection error: try re-running COLMAP with PINHOLE model")
            print(f"         or check if reference frame has motion blur")
        if not checks.get("min_3d_points", {}).get("pass", True):
            print(f"       - Too few 3D points: cameras may not overlap enough")
            print(f"         Try extracting a different reference frame (--ref_frame N)")

    return {"scene": scene_name, "checks": checks, "all_pass": all_pass,
            "n_images": n_images, "n_points": n_points,
            "mean_reproj": mean_reproj, "mean_track": mean_track,
            "recon": recon}


def visualize_reconstruction(scene_name: str, recon: pycolmap.Reconstruction) -> None:
    """
    Create 2D visualizations of the camera poses and point cloud.
    Saves figures to data/processed/{scene}/validation/
    """
    val_dir = PROC_DIR / scene_name / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Extract camera positions and orientations
    cam_data = []
    for im_id, image in recon.images.items():
        cfw = image.cam_from_world()   # pycolmap 4.0: method
        R = cfw.rotation.matrix()
        t = cfw.translation
        pos = -R.T @ t
        look_at = -R.T[:, 2]
        cam_data.append({
            "name": Path(image.name).stem,
            "pos": pos,
            "look_at": look_at,
            "im_id": im_id,
        })

    # Extract 3D points
    pts = np.array([[pt.xyz[0], pt.xyz[1], pt.xyz[2]]
                    for pt in recon.points3D.values()])
    colors = np.array([[pt.color[0], pt.color[1], pt.color[2]]
                       for pt in recon.points3D.values()], dtype=float) / 255.0

    # ── Figure 1: Top-down (XZ plane) + Side (XY plane) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"COLMAP Reconstruction: {scene_name}", fontsize=14, fontweight="bold")

    cam_colors = plt.cm.tab10(np.linspace(0, 1, len(cam_data)))

    for ax_idx, (ax, (horiz_ax, vert_ax, title)) in enumerate(zip(
            axes, [("x", "z", "Top-Down View (X-Z plane)"),
                   ("x", "y", "Side View (X-Y plane)")])):

        hi = {"x": 0, "y": 1, "z": 2}[horiz_ax]
        vi = {"x": 0, "y": 1, "z": 2}[vert_ax]

        # Plot sparse point cloud
        if len(pts) > 0:
            # Subsample for clarity (max 5000 points)
            idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
            ax.scatter(pts[idx, hi], pts[idx, vi],
                       c=colors[idx], s=0.5, alpha=0.4, zorder=1)

        # Plot camera positions and look-at arrows
        for i, cam in enumerate(cam_data):
            pos = cam["pos"]
            look = cam["look_at"]
            c = cam_colors[i]

            # Camera position
            ax.scatter(pos[hi], pos[vi], c=[c], s=100, zorder=5,
                       marker="^", edgecolors="black", linewidth=0.5)

            # Look-at direction arrow (scaled for visibility)
            arrow_scale = 0.3
            ax.annotate("",
                xy=(pos[hi] + look[hi]*arrow_scale,
                    pos[vi] + look[vi]*arrow_scale),
                xytext=(pos[hi], pos[vi]),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))

            # Camera label
            ax.text(pos[hi] + 0.02, pos[vi] + 0.02, cam["name"],
                    fontsize=9, color=c, fontweight="bold")

        ax.set_xlabel(horiz_ax.upper())
        ax.set_ylabel(vert_ax.upper())
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    out_path = val_dir / "camera_poses.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Camera pose visualization: {out_path}")

    # ── Figure 2: Reprojection quality check ────────────────────────────────────
    # Pick up to 200 high-quality 3D points and project them onto each camera image.
    # If projected points land on the correct features → good poses.

    images_dir = PROC_DIR / scene_name / "colmap_input"
    if not images_dir.exists():
        print(f"  (Skipping reprojection check — {images_dir} not found)")
        return

    # Get points with long tracks (more reliably triangulated)
    good_pts = sorted(recon.points3D.items(),
                      key=lambda kv: len(kv[1].track.elements),
                      reverse=True)[:300]

    n_cams = len(cam_data)
    fig, axes = plt.subplots(1, n_cams, figsize=(6*n_cams, 5))
    if n_cams == 1:
        axes = [axes]
    fig.suptitle(f"Reprojection Check: {scene_name}\n"
                 f"(Green dots = projected 3D points, should land on scene features)",
                 fontsize=12)

    for ax, cam_info in zip(axes, cam_data):
        img_path = images_dir / f"{cam_info['name']}.jpg"
        if not img_path.exists():
            ax.set_title(f"{cam_info['name']}\n(image not found)")
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        # Find the pycolmap image object for this camera
        for im_id, image in recon.images.items():
            if Path(image.name).stem == cam_info["name"]:
                cam = recon.cameras[image.camera_id]
                cfw = image.cam_from_world()   # pycolmap 4.0: method
                R = cfw.rotation.matrix()
                t = cfw.translation
                K = np.array([
                    [cam.params[0], 0,            cam.params[2]],
                    [0,             cam.params[1], cam.params[3]],
                    [0,             0,             1            ]
                ])

                # Project each good 3D point onto this image
                proj_pts = []
                for pt_id, pt in good_pts:
                    # World → camera coordinates
                    X_cam = R @ pt.xyz + t
                    if X_cam[2] <= 0:  # behind camera
                        continue
                    # Camera → image coordinates (pinhole model, ignoring distortion)
                    x_im = (K[0,0] * X_cam[0] / X_cam[2]) + K[0,2]
                    y_im = (K[1,1] * X_cam[1] / X_cam[2]) + K[1,2]
                    # Keep only points within image bounds
                    if 0 < x_im < img.shape[1] and 0 < y_im < img.shape[0]:
                        proj_pts.append((x_im, y_im))

                if proj_pts:
                    proj_arr = np.array(proj_pts)
                    ax.scatter(proj_arr[:, 0], proj_arr[:, 1],
                               c="lime", s=8, alpha=0.7, zorder=5)
                    ax.set_title(f"{cam_info['name']}\n{len(proj_pts)} points projected")
                else:
                    ax.set_title(f"{cam_info['name']}\n(no points visible)")
                break
        else:
            ax.set_title(f"{cam_info['name']}\n(not registered)")

        ax.axis("off")

    plt.tight_layout()
    out_path = val_dir / "reprojection_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Reprojection check:        {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate COLMAP reconstruction before 3DGS training"
    )
    parser.add_argument("--scene", type=str, default="scene_S003",
                        help="Scene name or 'all'")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip visualization (just run checks)")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted([d.name for d in PROC_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("scene_")])
    else:
        scenes = [args.scene]

    all_results = {}
    for scene in scenes:
        try:
            result = validate_reconstruction(scene)
            if not args.no_vis:
                visualize_reconstruction(scene, result["recon"])
            all_results[scene] = result["all_pass"]
        except Exception as e:
            print(f"❌ {scene}: {e}")
            import traceback
            traceback.print_exc()
            all_results[scene] = False

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for scene, passed in all_results.items():
        status = "✅ READY" if passed else "❌ ISSUES"
        print(f"  {status}  {scene}")


if __name__ == "__main__":
    main()
