"""
09_train_dynamic_4dgs.py — Phase 4: Dynamic Scene Reconstruction
================================================================

Trains a Deformable-3DGS model on the multi-camera video. This is the payoff
of the dataset's temporal dimension: where static 3DGS only sees one frame per
camera (3 views, 0 timesteps), the deformable model gets 453 views across 151
synchronized timesteps from cams 1–3 — and we hold out cam04 as a novel view.

WHAT THIS SCRIPT DOES
---------------------
1. Builds a *combined* source directory `dynamic/4dgs_combined/` containing all
   604 images (453 train + 151 test) and a unified sparse with 4 cameras.
   This is what train.py's `readColmapSceneInfo()` expects — it splits internally.

2. Patches Deformable-3DGS's `scene/dataset_readers.py` for our naming scheme:
     a) fid extraction: original code does `int(image_name) / (num_frames-1)`,
        which crashes on `cam01_frame_0000`. We extract the frame index via
        regex and normalize by the maximum frame index in the dataset.
     b) train/test split: original code uses `llffhold` (every Nth image goes
        to test). We override that to make any image starting with the held-out
        camera prefix the test set — a much cleaner novel-view eval.

   Patches are idempotent (guarded by a marker comment) and leave a `.bak`.

3. Reinstalls `depth-diff-gaussian-rasterization` (the rasterizer Deformable-3DGS
   needs — different package from gaussian-splatting's, despite the same name).

4. Runs Deformable-3DGS's `train.py` via subprocess with `--eval`. Test PSNR on
   the held-out camera is reported every test_iteration.

WHY DEFORMABLE-3DGS (vs 4DGS)
-----------------------------
Deformable-3DGS factors the scene into:
  - A *canonical* set of 3D Gaussians (the "rest pose")
  - A small MLP that maps (Gaussian xyz, time) → (Δxyz, Δrotation, Δscale)

This is more interpretable than 4DGS's all-Gaussian-attributes-as-temporal-
polynomials approach. For a learning project, the canonical+deformation split
is easier to reason about — and easier to ablate (freeze canonical, train only
deformation, etc.).

Usage:
    python scripts/09_train_dynamic_4dgs.py --scene scene_S004
    python scripts/09_train_dynamic_4dgs.py --scene scene_S004 --iterations 20000
    python scripts/09_train_dynamic_4dgs.py --scene scene_S004 --skip_rebuild --skip_install
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

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
REPOS_DIR    = PROJECT_ROOT / "repos"
EXP_DIR      = PROJECT_ROOT / "experiments" / "dynamic"
DEFORM_DIR   = REPOS_DIR / "Deformable-3D-Gaussians"


# ── Combined source build ─────────────────────────────────────────────────────

def _read_imgs_bin(path: Path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            im_id = struct.unpack("I", f.read(4))[0]
            quat = struct.unpack("4d", f.read(32))
            tvec = struct.unpack("3d", f.read(24))
            cam_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            n_pts = struct.unpack("Q", f.read(8))[0]
            f.read(n_pts * 20)
            imgs[im_id] = {
                "qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3],
                "tx": tvec[0], "ty": tvec[1], "tz": tvec[2],
                "camera_id": cam_id, "name": name.decode("utf-8"),
            }
    return imgs


def _read_cams_bin(path: Path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id = struct.unpack("Ii", f.read(8))
            w, h = struct.unpack("QQ", f.read(16))
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = struct.unpack(f"{n_params}d", f.read(8 * n_params))
            cams[cam_id] = {"model_id": model_id, "width": w,
                            "height": h, "params": list(params)}
    return cams


def build_combined_source(scene_name: str) -> Path:
    """
    Create dynamic/4dgs_combined/ with all 4 cameras × 151 frames.
    Sparse cams.bin has 4 entries; images.bin has 604.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from _colmap_io import write_cameras_bin, write_images_bin

    scene_proc = PROC_DIR / scene_name
    src_train  = scene_proc / "dynamic" / "3dgs_train"
    src_test   = scene_proc / "dynamic" / "3dgs_test"
    dst        = scene_proc / "dynamic" / "4dgs_combined"

    if (dst / "sparse" / "0" / "images.bin").exists():
        n_imgs = struct.unpack("Q", open(dst / "sparse" / "0" / "images.bin", "rb").read(8))[0]
        if n_imgs > 0 and (dst / "images").exists():
            n_files = len(list((dst / "images").glob("*.jpg")))
            if n_files == n_imgs:
                print(f"  ✅ Combined source already exists ({n_imgs} images)")
                return dst

    print(f"  Building combined source at {dst}")
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(exist_ok=True)

    # Read source sparse files
    train_cams = _read_cams_bin(src_train / "sparse" / "0" / "cameras.bin")
    train_imgs = _read_imgs_bin(src_train / "sparse" / "0" / "images.bin")
    test_cams  = _read_cams_bin(src_test  / "sparse" / "0" / "cameras.bin")
    test_imgs  = _read_imgs_bin(src_test  / "sparse" / "0" / "images.bin")

    # Merge cameras (re-id to 1..N)
    merged_cams = {}
    cam_id_map  = {}  # old → new (per source)

    next_cid = 1
    for old_id, cam in train_cams.items():
        merged_cams[next_cid] = cam
        cam_id_map[("train", old_id)] = next_cid
        next_cid += 1
    for old_id, cam in test_cams.items():
        merged_cams[next_cid] = cam
        cam_id_map[("test", old_id)] = next_cid
        next_cid += 1

    # Merge images — re-id to 1..N
    merged_imgs = {}
    next_iid = 1
    for old_id, im in sorted(train_imgs.items(), key=lambda kv: kv[1]["name"]):
        new_im = {**im, "camera_id": cam_id_map[("train", im["camera_id"])]}
        merged_imgs[next_iid] = new_im
        next_iid += 1
    for old_id, im in sorted(test_imgs.items(), key=lambda kv: kv[1]["name"]):
        new_im = {**im, "camera_id": cam_id_map[("test", im["camera_id"])]}
        merged_imgs[next_iid] = new_im
        next_iid += 1

    write_cameras_bin(merged_cams, dst / "sparse" / "0" / "cameras.bin")
    write_images_bin(merged_imgs,  dst / "sparse" / "0" / "images.bin")
    # The point cloud is identical in train & test (same MASt3R reconstruction)
    shutil.copy2(src_train / "sparse" / "0" / "points3D.bin",
                 dst / "sparse" / "0" / "points3D.bin")

    # Copy images
    for src_dir in (src_train / "images", src_test / "images"):
        for img in src_dir.glob("*.jpg"):
            dest = dst / "images" / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    n_files = len(list((dst / "images").glob("*.jpg")))
    print(f"  ✅ Combined source: {len(merged_cams)} cameras, "
          f"{len(merged_imgs)} sparse entries, {n_files} image files")
    return dst


# ── Dataset reader patches ────────────────────────────────────────────────────

PATCH_MARKER = "# PATCH-NR-PROJECT-v1"


def patch_dataset_readers(test_cam_prefix: str = "cam04"):
    """
    Apply two idempotent patches to Deformable-3D-Gaussians/scene/dataset_readers.py:

    Patch 1: replace the fid line in readColmapCameras() to handle
             cam{N}_frame_{F} naming.
    Patch 2: replace the llffhold split in readColmapSceneInfo() to split by
             image name prefix (the held-out camera).
    """
    target = DEFORM_DIR / "scene" / "dataset_readers.py"
    src = target.read_text()

    if PATCH_MARKER in src:
        print(f"  ⏩ dataset_readers.py already patched ({PATCH_MARKER})")
        return

    # Backup once
    bak = target.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_text(src)

    new = src

    # ── Patch 1: fid extraction ──
    old_fid = "        fid = int(image_name) / (num_frames - 1)"
    new_fid = (
        "        # === " + PATCH_MARKER + " — fid from cam{N}_frame_{F} ===\n"
        "        try:\n"
        "            fid = int(image_name) / (num_frames - 1)\n"
        "        except ValueError:\n"
        "            import re as _re\n"
        "            m = _re.search(r'frame_(\\d+)', image_name)\n"
        "            if m is None:\n"
        "                raise ValueError(f'cannot extract frame index from {image_name!r}')\n"
        "            # Normalize across the dataset's actual frame range\n"
        "            if not hasattr(readColmapCameras, '_max_frame'):\n"
        "                _frames = []\n"
        "                for _k in cam_extrinsics:\n"
        "                    _nm = os.path.basename(cam_extrinsics[_k].name).split('.')[0]\n"
        "                    _mm = _re.search(r'frame_(\\d+)', _nm)\n"
        "                    if _mm: _frames.append(int(_mm.group(1)))\n"
        "                readColmapCameras._max_frame = max(_frames) if _frames else 1\n"
        "            fid = int(m.group(1)) / max(readColmapCameras._max_frame, 1)"
    )
    if old_fid not in new:
        raise RuntimeError(
            f"Could not find expected fid line in {target}. "
            f"Has the upstream changed? Look for 'fid = int(image_name)'.")
    new = new.replace(old_fid, new_fid)

    # ── Patch 2: train/test split ──
    old_split = (
        "    if eval:\n"
        "        train_cam_infos = [c for idx, c in enumerate(\n"
        "            cam_infos) if idx % llffhold != 0]\n"
        "        test_cam_infos = [c for idx, c in enumerate(\n"
        "            cam_infos) if idx % llffhold == 0]"
    )
    new_split = (
        "    if eval:\n"
        "        # === " + PATCH_MARKER + " — split by camera-name prefix ===\n"
        f"        _test_prefix = os.environ.get('NR_TEST_CAM_PREFIX', '{test_cam_prefix}')\n"
        "        train_cam_infos = [c for c in cam_infos if not c.image_name.startswith(_test_prefix)]\n"
        "        test_cam_infos  = [c for c in cam_infos if c.image_name.startswith(_test_prefix)]\n"
        "        if not test_cam_infos:\n"
        "            # Fall back to llffhold if the prefix matches nothing\n"
        "            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]\n"
        "            test_cam_infos  = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]"
    )
    if old_split not in new:
        raise RuntimeError(
            f"Could not find expected llffhold split block in {target}.")
    new = new.replace(old_split, new_split)

    target.write_text(new)
    print(f"  ✅ Patched {target.name} (backup: {bak.name})")
    print(f"     - fid extraction now handles cam{{N}}_frame_{{F}}")
    print(f"     - train/test split uses image-name prefix '{test_cam_prefix}*'")


# ── Rasterizer ────────────────────────────────────────────────────────────────

def ensure_dynamic_rasterizer():
    """
    Install Deformable-3DGS's depth-diff-gaussian-rasterization.
    Note: this OVERWRITES gaussian-splatting's diff-gaussian-rasterization (same
    package name). After running 09, you'd need to re-run 05's rasterizer setup
    to retrain a static model.
    """
    rast_dir = DEFORM_DIR / "submodules" / "depth-diff-gaussian-rasterization"
    if not rast_dir.exists():
        # Try to fetch submodules
        print("  Fetching submodules...")
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"],
                       cwd=DEFORM_DIR, check=True)

    # Apply -fpermissive fix if not already applied (CUDA 12 / gcc-12 quirk)
    setup_py = rast_dir / "setup.py"
    setup_src = setup_py.read_text()
    if "-fpermissive" not in setup_src:
        setup_src = setup_src.replace(
            'extra_compile_args={"nvcc": ["-I"',
            'extra_compile_args={"nvcc": ["-Xcompiler=-fpermissive", "-I"',
        )
        setup_py.write_text(setup_src)
        print("  Applied -fpermissive fix to setup.py")

    print("  Installing depth-diff-gaussian-rasterization (~45s)...")
    cuda_home = str(Path.home() / "cuda-home")
    env = {
        **os.environ,
        "CUDA_HOME": cuda_home,
        "CC":  "/usr/bin/gcc-12",
        "CXX": "/usr/bin/g++-12",
        # nvcc needs cicc on PATH
        "PATH": f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
    }
    # Force reinstall — both rasterizers share package name diff_gaussian_rasterization
    # at version 0.0.0, so pip would otherwise treat the existing one as up-to-date.
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         "--force-reinstall", "--no-deps", "-q", "."],
        cwd=rast_dir, env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ install failed:\n{result.stderr[-500:]}")
        raise RuntimeError("rasterizer install failed")

    # Verify
    verify = subprocess.run(
        [sys.executable, "-c",
         "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer; "
         "import inspect; "
         "sig = inspect.signature(GaussianRasterizer.forward); "
         "params = list(sig.parameters.keys()); "
         "assert 'means2D_densify' in params, f'wrong rasterizer (missing means2D_densify): {params}'; "
         "print('OK')"],
        capture_output=True, text=True,
    )
    if verify.returncode != 0:
        print(f"  ✗ verification failed: {verify.stderr}")
        raise RuntimeError("wrong rasterizer is active")
    print("  ✅ depth-diff-gaussian-rasterization is active")


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(scene_name: str, experiment_name: str,
                 source_path: Path, iterations: int) -> Path:
    """Run Deformable-3DGS train.py via subprocess."""
    model_path = EXP_DIR / scene_name / experiment_name
    model_path.mkdir(parents=True, exist_ok=True)

    save_iters_full = [7_000, 15_000, 25_000, 40_000]
    save_iters = [s for s in save_iters_full if s <= iterations] + [iterations]
    save_iters = sorted(set(save_iters))

    test_iters_full = [1_000, 5_000, 10_000, 20_000, 30_000, 40_000]
    test_iters = [t for t in test_iters_full if t <= iterations] + [iterations]
    test_iters = sorted(set(test_iters))

    cmd = [
        sys.executable, str(DEFORM_DIR / "train.py"),
        "--source_path", str(source_path),
        "--model_path",  str(model_path),
        "--iterations",  str(iterations),
        "--eval",
        "--test_iterations", *[str(t) for t in test_iters],
        "--save_iterations", *[str(s) for s in save_iters],
        "--quiet",
    ]
    print(f"\n  Running: {' '.join(cmd[:6])} … (test @ {test_iters}, save @ {save_iters})")
    print(f"  Model will be saved to: {model_path}")

    cuda_home = str(Path.home() / "cuda-home")
    env = {
        **os.environ,
        "CUDA_HOME": cuda_home,
        "CC":  "/usr/bin/gcc-12",
        "CXX": "/usr/bin/g++-12",
        "PYTHONPATH": str(DEFORM_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":       f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
        "NR_TEST_CAM_PREFIX": "cam04",
    }

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(DEFORM_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit {result.returncode})")

    print(f"\n  ✅ Training complete in {elapsed/60:.1f} min")

    config = {
        "scene": scene_name, "experiment": experiment_name,
        "source_path": str(source_path), "iterations": iterations,
        "test_iterations": test_iters, "save_iterations": save_iters,
        "train_time_minutes": elapsed / 60,
    }
    with open(model_path / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene",      type=str, default="scene_S004")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--iterations", type=int, default=40_000,
                        help="Default 40k = Deformable-3DGS paper setting")
    parser.add_argument("--test_cam",   type=str, default="cam04",
                        help="Camera prefix to hold out for test")
    parser.add_argument("--skip_rebuild", action="store_true",
                        help="Skip combined-source rebuild")
    parser.add_argument("--skip_install", action="store_true",
                        help="Skip rasterizer (re)install")
    parser.add_argument("--skip_patch",   action="store_true",
                        help="Skip dataset_readers.py patching")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Phase 4: Dynamic 4DGS (Deformable-3DGS)")
    print(f"  Scene:      {args.scene}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Iterations: {args.iterations:,}")
    print(f"  Test cam:   {args.test_cam}")
    print(f"{'=' * 60}")

    # Step 1: combined source
    if not args.skip_rebuild:
        print("\n[1/4] Building combined train+test source...")
        source_path = build_combined_source(args.scene)
    else:
        source_path = PROC_DIR / args.scene / "dynamic" / "4dgs_combined"
        print(f"\n[1/4] Skipping rebuild. Using {source_path}")
        if not (source_path / "sparse" / "0" / "images.bin").exists():
            raise FileNotFoundError("Combined source missing — drop --skip_rebuild")

    # Step 2: patches
    if not args.skip_patch:
        print("\n[2/4] Patching Deformable-3DGS dataset_readers.py...")
        patch_dataset_readers(test_cam_prefix=args.test_cam)
    else:
        print("\n[2/4] Skipping patches")

    # Step 3: rasterizer
    if not args.skip_install:
        print("\n[3/4] Installing dynamic CUDA rasterizer...")
        ensure_dynamic_rasterizer()
    else:
        print("\n[3/4] Skipping rasterizer install")

    # Step 4: train
    print(f"\n[4/4] Training (~{args.iterations/40000*90:.0f} min on RTX 3080)...")
    model_path = run_training(args.scene, args.experiment,
                              source_path, args.iterations)

    print(f"\nDone. Model: {model_path}")
    print(f"  Test PSNR was reported during training (cam04 held out)")
    print(f"  Render with: cd repos/Deformable-3D-Gaussians && "
          f"python render.py -m {model_path}")


if __name__ == "__main__":
    main()
