"""
03_prepare_dataset.py — Build 3DGS-Ready Dataset Structure
===========================================================

Takes the COLMAP reconstruction + extracted frames and builds the exact
directory structure that gaussian-splatting and Deformable-3D-Gaussians expect.

Two outputs are produced:

  1. STATIC dataset  — for Phase 1: static 3DGS
     One image per camera, poses from COLMAP.
     Trains in ~35 minutes on RTX 3080.

  2. DYNAMIC dataset — for Phase 2: Deformable 3DGS
     All frames from all cameras, same COLMAP poses replicated per timestep.
     The deformation network learns how the scene moves over time.

THE KEY CHALLENGE: REPLICATING POSES FOR DYNAMIC FRAMES
--------------------------------------------------------
COLMAP ran on 1 reference frame per camera and produced poses:
  cam01.jpg → R₁, t₁
  cam02.jpg → R₂, t₂
  cam03.jpg → R₃, t₃
  cam04.jpg → R₄, t₄

For dynamic training, we have 119 timesteps × 4 cameras = 476 images.
Each image from cam01 (regardless of timestep) has the SAME pose as cam01's
reference frame — because the camera doesn't move!

So we need to create an images.bin where:
  cam01_frame_0000.jpg → R₁, t₁
  cam01_frame_0003.jpg → R₁, t₁   ← same pose!
  cam01_frame_0006.jpg → R₁, t₁
  ...
  cam02_frame_0000.jpg → R₂, t₂
  cam02_frame_0003.jpg → R₂, t₂
  ...

This is perfectly valid — we're just telling the renderer "this image was taken
from camera 1's position" for every frame from camera 1.

TRAIN / TEST SPLIT DESIGN
--------------------------
We hold out ONE ENTIRE CAMERA for novel-view evaluation.

Why this split (not random):
  - Random split mixes timesteps from the same camera → the model has seen
    nearly identical views during training. It's too easy. Not a fair test.
  - Holding out a whole camera tests TRUE novel view synthesis:
    "Can you render what camera 4 would see, having never seen its view?"
  - This is how published papers evaluate multi-camera dynamic NVS.

We choose cam04 as the test camera (last camera = holds-out 25% of viewpoints).

FILE NAMING CONVENTION
----------------------
Images are named:   {cam_id}_frame_{src_idx:04d}.jpg
  e.g.  cam01_frame_0000.jpg
        cam01_frame_0003.jpg
        cam04_frame_0000.jpg  (test camera)

This naming means the sorted order interleaves cameras:
  cam01_frame_0000.jpg
  cam01_frame_0003.jpg
  ...
  cam02_frame_0000.jpg
  ...

For Deformable-3DGS's llffhold=8 split, this would mix timesteps from the
same camera in train/test, which we don't want. So we set --eval to False
and manage train/test ourselves via the --images flag or custom split.

OUTPUT STRUCTURE
----------------
data/processed/scene_S003/
├── static/                     ← Phase 1: static 3DGS training
│   ├── images/
│   │   ├── cam01.jpg           (training)
│   │   ├── cam02.jpg           (training)
│   │   ├── cam03.jpg           (training)
│   │   └── cam04.jpg           (test — held out)
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
│
└── dynamic/                    ← Phase 2: 4DGS training
    ├── train/
    │   ├── images/
    │   │   ├── cam01_frame_0000.jpg
    │   │   ├── cam01_frame_0003.jpg
    │   │   ...
    │   │   ├── cam03_frame_0000.jpg
    │   │   └── ...
    │   └── sparse/0/
    │       ├── cameras.bin     (cam01, cam02, cam03 intrinsics)
    │       ├── images.bin      (all train frames with replicated poses)
    │       └── points3D.bin    (from COLMAP)
    └── test/
        ├── images/
        │   ├── cam04_frame_0000.jpg
        │   └── ...
        └── sparse/0/
            └── ...             (cam04 intrinsics + poses)

Usage:
    python scripts/03_prepare_dataset.py --scene scene_S003
    python scripts/03_prepare_dataset.py --scene scene_S003 --test_cam cam04
    python scripts/03_prepare_dataset.py --scene all
"""

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Optional

import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


# ── COLMAP Binary I/O ─────────────────────────────────────────────────────────
# We need to write cameras.bin and images.bin from scratch for the dynamic dataset.
# These are COLMAP's binary format files. The format is documented at:
# https://colmap.github.io/format.html
#
# WHY write from scratch: We need to create entries for all dynamic frames,
# but COLMAP only produced entries for the reference frames. We copy the pose
# from the reference frame of each camera and assign it to all frames from
# that camera.

def write_cameras_bin(cameras: dict, path: Path):
    """
    Write a COLMAP cameras.bin file.

    cameras: dict mapping camera_id (int) to dict with keys:
      model_id, width, height, params (list of floats)

    Binary format:
      uint64: number of cameras
      for each camera:
        uint32: camera_id
        int32:  model_id
        uint64: width
        uint64: height
        float64[]: params (variable length depending on model)
    """
    # Model IDs in COLMAP:
    CAMERA_MODEL_IDS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
    }
    # Number of parameters per model:
    CAMERA_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))  # uint64: count
        for cam_id, cam in cameras.items():
            model_id = cam["model_id"]
            params = cam["params"]
            f.write(struct.pack("I", cam_id))            # uint32: camera_id
            f.write(struct.pack("i", model_id))          # int32: model_id
            f.write(struct.pack("Q", cam["width"]))      # uint64: width
            f.write(struct.pack("Q", cam["height"]))     # uint64: height
            f.write(struct.pack(f"{len(params)}d",
                                *params))                # float64[]: params


def write_images_bin(images: dict, path: Path):
    """
    Write a COLMAP images.bin file.

    images: dict mapping image_id (int) to dict with keys:
      qw, qx, qy, qz  (quaternion: camera rotation, world-to-camera)
      tx, ty, tz       (translation: world-to-camera)
      camera_id        (which camera model to use)
      name             (image filename, relative to images/ folder)

    Binary format:
      uint64: number of images
      for each image:
        uint32: image_id
        float64[4]: qvec (qw, qx, qy, qz)
        float64[3]: tvec (tx, ty, tz)
        uint32: camera_id
        char[]: name (null-terminated string)
        uint64: num_points2D
        for each 2D point:
          float64: x
          float64: y
          int64: point3D_id (-1 if unmatched)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))  # uint64: count
        for im_id, im in images.items():
            f.write(struct.pack("I", im_id))     # uint32: image_id
            f.write(struct.pack("4d", im["qw"], im["qx"],
                                im["qy"], im["qz"]))    # float64[4]: qvec
            f.write(struct.pack("3d", im["tx"],
                                im["ty"], im["tz"]))    # float64[3]: tvec
            f.write(struct.pack("I", im["camera_id"])) # uint32: camera_id
            # Name: null-terminated string
            name_bytes = im["name"].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            # No 2D point observations (we don't need them for 3DGS training)
            f.write(struct.pack("Q", 0))         # uint64: num_points2D = 0


def write_points3d_bin(points: dict, path: Path):
    """
    Write a COLMAP points3D.bin file.

    points: dict mapping point3D_id (int) to dict with keys:
      x, y, z    (3D position)
      r, g, b    (color, uint8)
      error      (reprojection error)
      track      (list of (image_id, point2D_idx) tuples)

    We copy this directly from the COLMAP reconstruction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(points)))  # uint64: count
        for pt_id, pt in points.items():
            f.write(struct.pack("Q", pt_id))
            f.write(struct.pack("3d", pt["x"], pt["y"], pt["z"]))
            f.write(struct.pack("3B", pt["r"], pt["g"], pt["b"]))
            f.write(struct.pack("d", pt["error"]))
            f.write(struct.pack("Q", len(pt["track"])))
            for image_id, point2d_idx in pt["track"]:
                f.write(struct.pack("I", image_id))
                f.write(struct.pack("I", point2d_idx))


def load_colmap_reconstruction(sparse_dir: Path) -> tuple[dict, dict, dict]:
    """
    Load a COLMAP reconstruction from binary files.
    Returns (cameras, images, points3D) as dicts.
    """
    import pycolmap
    recon = pycolmap.Reconstruction(str(sparse_dir))

    # Extract cameras
    cameras = {}
    for cam_id, cam in recon.cameras.items():
        cameras[cam_id] = {
            "model_id": cam.model.value,
            "model_name": cam.model.name,
            "width":    cam.width,
            "height":   cam.height,
            "params":   list(cam.params),
        }

    # Extract images (poses)
    # Note: in pycolmap 4.0, cam_from_world is a METHOD, not a property
    images = {}
    for im_id, image in recon.images.items():
        cfw = image.cam_from_world()   # call as function in pycolmap 4.0
        q = cfw.rotation.quat          # [qx, qy, qz, qw]
        t = cfw.translation
        images[im_id] = {
            "qw": float(q[3]),
            "qx": float(q[0]),
            "qy": float(q[1]),
            "qz": float(q[2]),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "tz": float(t[2]),
            "camera_id": image.camera_id,
            "name":      image.name,
        }

    # Extract 3D points
    points3D = {}
    for pt_id, pt in recon.points3D.items():
        track = [(elem.image_id, elem.point2D_idx) for elem in pt.track.elements]
        points3D[pt_id] = {
            "x": float(pt.xyz[0]),
            "y": float(pt.xyz[1]),
            "z": float(pt.xyz[2]),
            "r": int(pt.color[0]),
            "g": int(pt.color[1]),
            "b": int(pt.color[2]),
            "error": float(pt.error),
            "track": track,
        }

    return cameras, images, points3D


def prepare_static_dataset(scene_name: str, test_cam: str = "cam04") -> None:
    """
    Build the static 3DGS dataset structure.

    For static training, we just need to copy:
    - Reference images (already in colmap_input/)  →  static/images/
    - COLMAP sparse reconstruction                 →  static/sparse/0/
    """
    scene_proc = PROC_DIR / scene_name
    colmap_workspace = scene_proc / "colmap_workspace"
    static_dir = scene_proc / "static"

    print(f"\n{'─'*50}")
    print(f"[STATIC] Preparing static dataset for {scene_name}")
    print(f"  Test camera: {test_cam} (held out for evaluation)")

    # Images are already in static/images/ from step 01
    # We just need to verify and link the sparse reconstruction
    images_dir = static_dir / "images"
    sparse_src  = colmap_workspace / "sparse" / "0"
    sparse_dst  = static_dir / "sparse" / "0"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Static images not found: {images_dir}\n"
            f"Run 01_extract_frames.py first."
        )
    if not sparse_src.exists():
        raise FileNotFoundError(
            f"COLMAP sparse not found: {sparse_src}\n"
            f"Run 02_run_colmap.py first."
        )

    # Copy sparse files
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sparse_src / f
        dst = sparse_dst / f
        if src.exists():
            shutil.copy2(src, dst)
    print(f"  ✅ Sparse reconstruction copied to {sparse_dst}")

    # List training vs test images
    all_images = sorted(images_dir.glob("*.jpg"))
    train_imgs = [im for im in all_images if test_cam not in im.name]
    test_imgs  = [im for im in all_images if test_cam in im.name]
    print(f"  Training images: {[im.name for im in train_imgs]}")
    print(f"  Test images:     {[im.name for im in test_imgs]}")

    # Save split info
    split_info = {
        "train": [im.name for im in train_imgs],
        "test":  [im.name for im in test_imgs],
    }
    with open(static_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"  ✅ Static dataset ready at: {static_dir}")


def prepare_dynamic_dataset(scene_name: str, test_cam: str = "cam04") -> None:
    """
    Build the dynamic 4DGS dataset structure.

    Steps:
    1. Load COLMAP reconstruction (camera poses for reference frames)
    2. Build a lookup: cam_name → (camera_id, pose)
    3. For each dynamic frame, create an entry in images.bin with the
       corresponding camera's pose
    4. Write cameras.bin and images.bin for train and test splits
    5. Copy images into images/ directories
    """
    scene_proc = PROC_DIR / scene_name
    colmap_workspace = scene_proc / "colmap_workspace"
    dynamic_src = scene_proc / "dynamic"
    train_dir = scene_proc / "dynamic" / "3dgs_train"
    test_dir  = scene_proc / "dynamic" / "3dgs_test"

    print(f"\n{'─'*50}")
    print(f"[DYNAMIC] Preparing dynamic dataset for {scene_name}")
    print(f"  Test camera: {test_cam} (held out for evaluation)")

    # Load COLMAP reconstruction
    sparse_src = colmap_workspace / "sparse" / "0"
    if not sparse_src.exists():
        raise FileNotFoundError(f"COLMAP sparse not found: {sparse_src}")

    cameras, ref_images, points3D = load_colmap_reconstruction(sparse_src)

    # Build cam_name → pose lookup from reference images.
    # Reference images may be named:
    #   "cam01.jpg"        (single frame per camera) → cam_name = "cam01"
    #   "cam01_ref00.jpg"  (multi-frame per camera)  → cam_name = "cam01"
    # We average all reference poses for each camera to get one representative pose.

    from collections import defaultdict

    cam_pose_groups = defaultdict(list)  # "cam01" → list of pose dicts
    for im_id, im_info in ref_images.items():
        stem = Path(im_info["name"]).stem       # "cam01" or "cam01_ref00"
        # Strip _refXX suffix if present
        cam_name = stem.split("_ref")[0]        # "cam01"
        cam_pose_groups[cam_name].append({
            "qw": im_info["qw"], "qx": im_info["qx"],
            "qy": im_info["qy"], "qz": im_info["qz"],
            "tx": im_info["tx"], "ty": im_info["ty"], "tz": im_info["tz"],
            "camera_id": im_info["camera_id"],
        })

    # For each camera, use the first reference pose (they should be nearly identical
    # for a fixed camera; averaging quaternions requires special handling)
    cam_pose_lookup = {}  # "cam01" → single pose dict
    for cam_name, poses_list in cam_pose_groups.items():
        # Use the first pose (most stable reference)
        # In practice, for a fixed camera all reference poses should be nearly identical
        cam_pose_lookup[cam_name] = poses_list[0]

    print(f"  Cameras with poses: {list(cam_pose_lookup.keys())}")

    # Find all dynamic frames per camera
    cam_dirs = sorted([d for d in dynamic_src.iterdir()
                       if d.is_dir() and d.name.startswith("cam")])

    if len(cam_dirs) == 0:
        raise FileNotFoundError(
            f"No camera frame directories found in {dynamic_src}\n"
            f"Run 01_extract_frames.py first."
        )

    # Split cameras into train/test
    train_cams = [d for d in cam_dirs if d.name != test_cam]
    test_cams  = [d for d in cam_dirs if d.name == test_cam]

    print(f"  Train cameras: {[d.name for d in train_cams]}")
    print(f"  Test cameras:  {[d.name for d in test_cams]}")

    def build_split(split_cams, split_dir, split_name):
        """Build images/ and sparse/0/ for one split (train or test)."""
        images_dir = split_dir / "images"
        sparse_dir = split_dir / "sparse" / "0"
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        new_images = {}   # image_id → pose entry
        image_id = 1      # COLMAP image IDs start at 1

        n_copied = 0
        for cam_dir in split_cams:
            cam_name = cam_dir.name  # e.g. "cam01"

            if cam_name not in cam_pose_lookup:
                print(f"  ⚠️  No COLMAP pose for {cam_name} — skipping")
                continue

            pose = cam_pose_lookup[cam_name]
            frames = sorted(cam_dir.glob("frame_*.jpg"))

            for frame_path in frames:
                # Output name: cam01_frame_0000.jpg
                out_name = f"{cam_name}_{frame_path.name}"
                out_path = images_dir / out_name

                # Copy (or hardlink) image
                if not out_path.exists():
                    shutil.copy2(frame_path, out_path)
                n_copied += 1

                # Create images.bin entry with the camera's fixed pose
                new_images[image_id] = {
                    "qw": pose["qw"], "qx": pose["qx"],
                    "qy": pose["qy"], "qz": pose["qz"],
                    "tx": pose["tx"], "ty": pose["ty"], "tz": pose["tz"],
                    "camera_id": pose["camera_id"],
                    "name": out_name,
                }
                image_id += 1

        # Write cameras.bin (same cameras as COLMAP, just filtered to split cams)
        split_cam_ids = set(cam_pose_lookup[d.name]["camera_id"]
                            for d in split_cams if d.name in cam_pose_lookup)
        split_cameras = {cid: cameras[cid] for cid in split_cam_ids
                         if cid in cameras}
        write_cameras_bin(split_cameras, sparse_dir / "cameras.bin")

        # Write images.bin
        write_images_bin(new_images, sparse_dir / "images.bin")

        # Copy points3D.bin (same for both splits — it's the scene structure)
        shutil.copy2(sparse_src / "points3D.bin", sparse_dir / "points3D.bin")

        print(f"  ✅ {split_name}: {n_copied} images, {len(new_images)} pose entries")
        print(f"     → {split_dir}")
        return n_copied

    print(f"\n  Building train split...")
    n_train = build_split(train_cams, train_dir, "Train")
    print(f"\n  Building test split...")
    n_test  = build_split(test_cams, test_dir, "Test")

    # Save metadata
    meta = {
        "scene": scene_name,
        "test_camera": test_cam,
        "train_cameras": [d.name for d in train_cams],
        "test_cameras": [d.name for d in test_cams],
        "n_train_images": n_train,
        "n_test_images": n_test,
    }
    with open(scene_proc / "dynamic_dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Dynamic dataset ready")
    print(f"     Train: {train_dir}  ({n_train} images)")
    print(f"     Test:  {test_dir}   ({n_test} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 3DGS-ready dataset from COLMAP reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--scene", type=str, default="scene_S003",
                        help="Scene name or 'all'")
    parser.add_argument("--test_cam", type=str, default="cam04",
                        help="Camera to hold out for testing (default: cam04)")
    parser.add_argument("--skip_static", action="store_true",
                        help="Skip static dataset preparation")
    parser.add_argument("--skip_dynamic", action="store_true",
                        help="Skip dynamic dataset preparation")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted([d.name for d in PROC_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("scene_")])
    else:
        scenes = [args.scene]

    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Preparing dataset: {scene}")
        print(f"{'='*60}")
        try:
            if not args.skip_static:
                prepare_static_dataset(scene, args.test_cam)
            if not args.skip_dynamic:
                prepare_dynamic_dataset(scene, args.test_cam)
            print(f"\n✅ {scene}: Dataset preparation complete")
        except Exception as e:
            print(f"\n❌ {scene}: Failed — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
