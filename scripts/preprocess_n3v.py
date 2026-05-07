"""
preprocess_n3v.py — Neural 3D Video dataset preprocessing for all three methods.

Usage:
    python scripts/preprocess_n3v.py --scene flame_salmon_1

Input:  data/raw/n3v/{scene}/   with cam*.mp4 + poses_bounds.npy
Output: data/processed/n3v/{scene}/  with:
  - cam*/images/NNNN.png            (1352x1014, 300 frames; for hustvl)
  - frames/cam*/NNNN.png            (symlinks; for Deformable)
  - images/cam*_NNNN.png            (symlinks; for fudan flat format)
  - transforms_train.json / transforms_test.json  (Blender; for fudan)
  - poses_bounds.npy                (for Deformable)
  - points3d.ply                    (from COLMAP MVS; for fudan)
  - points3D_downsample2.ply        (downsampled <40k; for hustvl)
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import glob
import numpy as np
from pathlib import Path

FFMPEG = str(Path.home() / "miniconda3/envs/nr-4dgs-hustvl/lib/python3.10/site-packages"
             "/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
COLMAP = str(Path.home() / "miniconda3/envs/neural-rendering/bin/colmap")
PROJECT_ROOT = Path(__file__).parent.parent
N_FRAMES = 300
OUT_W, OUT_H = 1352, 1014  # half native resolution (2704x2028)
TEST_CAM_IDX = 0  # cam00 is test, rest are train


# ── helper ────────────────────────────────────────────────────────────────────

def run(cmd, **kw):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-10))


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ── COLMAP database helpers (ported from n3v2blender.py) ──────────────────────

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    params BLOB, prior_focal_length INTEGER NOT NULL)"""
CREATE_IMAGES_TABLE = f"""CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
    prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
    prior_tx REAL, prior_ty REAL, prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""
CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
    data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
    data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
    data BLOB)"""
CREATE_TWO_VIEW_GEOMETRIES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
    data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)"""
CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()

    def create_tables(self):
        self.execute(CREATE_CAMERAS_TABLE)
        self.execute(CREATE_IMAGES_TABLE)
        self.execute(CREATE_KEYPOINTS_TABLE)
        self.execute(CREATE_DESCRIPTORS_TABLE)
        self.execute(CREATE_MATCHES_TABLE)
        self.execute(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.execute(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params, prior_focal_length=False, camera_id=None):
        params = np.asarray(params, dtype=np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, params.tobytes(), prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2], prior_q[3],
             prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid


def camTodatabase(cameras_txt, db_path):
    db = COLMAPDatabase.connect(db_path)
    with open(cameras_txt) as f:
        line = f.readline().strip()
    parts = line.split()
    camera_id = int(parts[0])
    model = parts[1]
    w, h = int(parts[2]), int(parts[3])
    params = [float(x) for x in parts[4:]]
    model_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                "RADIAL": 3, "OPENCV": 4}[model]
    db.add_camera(model_id, w, h, params, prior_focal_length=True, camera_id=camera_id)
    db.commit()
    db.close()


# ── main stages ───────────────────────────────────────────────────────────────

def stage_extract_frames(raw_dir: Path, out_dir: Path):
    """Extract first N_FRAMES from each cam*.mp4 at OUT_W x OUT_H."""
    videos = sorted(raw_dir.glob("cam*.mp4"))
    print(f"[1/6] Extracting {N_FRAMES} frames from {len(videos)} videos → {OUT_W}x{OUT_H}")
    for vid in videos:
        cam_name = vid.stem  # e.g. cam00
        images_dir = out_dir / cam_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # check if already done
        existing = list(images_dir.glob("*.png"))
        if len(existing) >= N_FRAMES:
            print(f"  {cam_name}: {len(existing)} frames already exist, skipping")
            continue
        run(f'{FFMPEG} -y -i "{vid}" -frames:v {N_FRAMES} '
            f'-vf "scale={OUT_W}:{OUT_H}" -start_number 0 '
            f'"{images_dir}/%04d.png"')
    print("  Frame extraction done.")


def stage_generate_transforms(raw_dir: Path, out_dir: Path):
    """Generate transforms_train.json and transforms_test.json from poses_bounds.npy."""
    print("[2/6] Generating transforms JSON from poses_bounds.npy")
    poses_bounds = np.load(raw_dir / "poses_bounds.npy")
    N = poses_bounds.shape[0]  # number of cameras

    poses_raw = poses_bounds[:, :15].reshape(-1, 3, 5)
    bounds = poses_bounds[:, -2:]
    H_orig, W_orig, fl_orig = poses_raw[0, :, -1]
    # scale focal to half resolution
    fl = fl_orig * (OUT_W / W_orig)
    cx, cy = OUT_W / 2.0, OUT_H / 2.0

    # Convert LLFF poses to OpenCV/NeRF convention (from n3v2blender.py)
    poses = np.concatenate([poses_raw[..., 1:2], poses_raw[..., 0:1],
                            -poses_raw[..., 2:3], poses_raw[..., 3:4]], -1)
    last_row = np.tile(np.array([0, 0, 0, 1]), (N, 1, 1))
    poses = np.concatenate([poses, last_row], axis=1)

    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1

    up = poses[:, 0:3, 1].sum(0)
    up /= np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    poses = R @ poses

    # center scene
    totp = np.zeros(3); totw = 0.0
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(N):
            if j == i:
                continue
            md = poses[j, :3, :]
            dot = mf[:3, 2] @ md[:3, 2]
            if dot > 0.9999:
                continue
            res = np.linalg.solve(
                np.array([[mf[:3, 2] @ mf[:3, 2], -mf[:3, 2] @ md[:3, 2]],
                          [mf[:3, 2] @ md[:3, 2], -md[:3, 2] @ md[:3, 2]]]),
                np.array([mf[:3, 3] @ mf[:3, 2] - mf[:3, 3] @ mf[:3, 2],
                          mf[:3, 3] @ md[:3, 2] - md[:3, 3] @ md[:3, 2]]))
            p = (mf[:3, 3] + res[0] * mf[:3, 2] + md[:3, 3] + res[1] * md[:3, 2]) / 2
            w = 1.0 / (np.linalg.norm(mf[:3, 3] + res[0] * mf[:3, 2] - p) + 1e-6)
            if w > 0.01:
                totp += p * w
                totw += w
    if totw > 0:
        totp /= totw
    poses[:, :3, 3] -= totp
    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 3] *= 4.0 / avglen

    # Use actual sorted mp4 filenames (some cams may be missing, e.g. cam03)
    videos = sorted(raw_dir.glob("cam*.mp4"))
    cams = [v.stem for v in videos]  # ['cam00', 'cam01', 'cam02', 'cam04', ...]
    assert len(cams) == N, f"poses_bounds has {N} rows but found {len(cams)} cam*.mp4 files"
    train_frames, test_frames = [], []
    for i, cam in enumerate(cams):
        cam_dir = out_dir / cam / "images"
        frame_files = sorted(cam_dir.glob("*.png"))[:N_FRAMES]
        frames = [
            {
                "file_path": f"{cam}/images/{f.stem}",
                "transform_matrix": poses[i].tolist(),
                "time": int(f.stem) / 30.0,
            }
            for f in frame_files
        ]
        if i == TEST_CAM_IDX:
            test_frames.extend(frames)
        else:
            train_frames.extend(frames)

    base = {"w": OUT_W, "h": OUT_H, "fl_x": fl, "fl_y": fl, "cx": cx, "cy": cy}
    (out_dir / "transforms_train.json").write_text(
        json.dumps({**base, "frames": train_frames}, indent=2))
    (out_dir / "transforms_test.json").write_text(
        json.dumps({**base, "frames": test_frames}, indent=2))

    # Also write poses_bounds.npy for Deformable
    shutil.copy(raw_dir / "poses_bounds.npy", out_dir / "poses_bounds.npy")
    print(f"  {len(train_frames)} train frames, {len(test_frames)} test frames written.")
    return poses, fl, cx, cy


def stage_colmap(out_dir: Path, poses, fl, cx, cy):
    """Run COLMAP on first-frame images (known poses) to build point cloud."""
    print("[3/6] Running COLMAP point triangulation + MVS")
    tmp = out_dir / "colmap_workspace"
    sparse = tmp / "created" / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    (tmp / "dense").mkdir(exist_ok=True)
    colmap_images = tmp / "images"
    colmap_images.mkdir(exist_ok=True)

    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cams = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    N = len(cams)

    # cameras.txt
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {OUT_W} {OUT_H} {fl} {fl} {cx} {cy}")

    # images.txt + symlinks
    fname2pose = {}
    for i, cam_dir in enumerate(cams):
        first_frame = sorted((cam_dir / "images").glob("*.png"))[0]
        fname = f"{cam_dir.name}_{first_frame.name}"  # e.g. cam00_0000.png
        pose = poses[i] @ blender2opencv
        fname2pose[fname] = pose
        symlink = colmap_images / fname
        if not symlink.exists():
            symlink.symlink_to(first_frame.resolve())

    images_txt_lines = []
    for idx, (fname, pose) in enumerate(fname2pose.items(), 1):
        R = np.linalg.inv(pose[:3, :3])
        T = -R @ pose[:3, 3]
        q = rotmat2qvec(R)
        images_txt_lines.append(f"{idx} {q[0]} {q[1]} {q[2]} {q[3]} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n")
    (sparse / "images.txt").write_text("".join(images_txt_lines))
    (sparse / "points3D.txt").write_text("")

    db_path = tmp / "database.db"
    if db_path.exists():
        db_path.unlink()

    # Use known camera intrinsics directly — avoids the camTodatabase conflict
    run(f'{COLMAP} feature_extractor --database_path "{db_path}" '
        f'--image_path "{colmap_images}" '
        f'--ImageReader.camera_model PINHOLE '
        f'--ImageReader.camera_params "{fl},{fl},{cx},{cy}" '
        f'--ImageReader.single_camera 1 '
        f'--SiftExtraction.max_image_size 4096')

    run(f'{COLMAP} exhaustive_matcher --database_path "{db_path}"')

    tri_sparse = tmp / "triangulated" / "sparse"
    tri_sparse.mkdir(parents=True, exist_ok=True)
    run(f'{COLMAP} point_triangulator '
        f'--database_path "{db_path}" '
        f'--image_path "{colmap_images}" '
        f'--input_path "{sparse}" '
        f'--output_path "{tri_sparse}"')

    # MVS dense reconstruction
    dense = tmp / "dense"
    run(f'{COLMAP} image_undistorter '
        f'--image_path "{colmap_images}" '
        f'--input_path "{tri_sparse}" '
        f'--output_path "{dense}"')
    run(f'{COLMAP} patch_match_stereo --workspace_path "{dense}"')

    ply_out = out_dir / "points3d.ply"
    run(f'{COLMAP} stereo_fusion '
        f'--workspace_path "{dense}" '
        f'--output_path "{ply_out}"')

    # clean up workspace (keep ply)
    shutil.rmtree(tmp)
    vis = Path(str(ply_out) + ".vis")
    if vis.exists():
        vis.unlink()
    print(f"  Point cloud saved to {ply_out}")


def stage_downsample(out_dir: Path):
    """Downsample points3d.ply → points3D_downsample2.ply (<40k pts) for hustvl."""
    print("[4/6] Downsampling point cloud for hustvl")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(out_dir / "points3d.ply"))
    print(f"  Input: {len(pcd.points)} points")
    voxel_size = 0.02
    while len(pcd.points) > 40000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  → {len(pcd.points)} points (voxel={voxel_size:.3f})")
        voxel_size += 0.01
    out_path = out_dir / "points3D_downsample2.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  Saved {len(pcd.points)} points to {out_path}")


def stage_frames_symlinks(out_dir: Path):
    """Create frames/ symlink tree for Deformable-3DGS."""
    print("[5/6] Creating frames/ symlink tree for Deformable-3DGS")
    frames_root = out_dir / "frames"
    for cam_dir in sorted(out_dir.glob("cam*")):
        if not cam_dir.is_dir():
            continue
        dst_dir = frames_root / cam_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted((cam_dir / "images").glob("*.png")):
            link = dst_dir / png.name
            if not link.exists():
                link.symlink_to(png.resolve())
    print(f"  Done: {frames_root}")


def stage_flat_images_symlinks(out_dir: Path):
    """Create images/ flat directory (cam00_NNNN.png) for fudan training."""
    print("[6/6] Creating images/ flat symlinks for fudan")
    flat = out_dir / "images"
    flat.mkdir(exist_ok=True)
    for cam_dir in sorted(out_dir.glob("cam*")):
        if not cam_dir.is_dir():
            continue
        for png in sorted((cam_dir / "images").glob("*.png")):
            link = flat / f"{cam_dir.name}_{png.name}"
            if not link.exists():
                link.symlink_to(png.resolve())
    print(f"  Done: {flat}")


def stage_hustvl_dir(raw_dir: Path, out_dir: Path):
    """Create a separate hustvl source dir with mp4 symlinks + points3D_downsample2.ply.

    hustvl scene/__init__.py checks for transforms_train.json BEFORE poses_bounds.npy,
    so if both exist (as in our main processed dir) it picks Blender mode. We need a
    directory that has ONLY poses_bounds.npy (+ mp4 + point cloud) for dynerf mode.
    """
    print("[+] Creating hustvl-specific source dir (dynerf mode)")
    hustvl_dir = out_dir.parent / (out_dir.name + "_hustvl_src")
    hustvl_dir.mkdir(exist_ok=True)

    # cam*.mp4 symlinks
    for mp4 in sorted(raw_dir.glob("cam*.mp4")):
        link = hustvl_dir / mp4.name
        if not link.exists():
            link.symlink_to(mp4.resolve())

    # poses_bounds.npy
    dst = hustvl_dir / "poses_bounds.npy"
    if not dst.exists():
        shutil.copy(raw_dir / "poses_bounds.npy", dst)

    # cam*/images symlinks (so Neural3D_NDC_Dataset reads pre-extracted frames)
    for cam_dir in sorted(out_dir.glob("cam*")):
        if not cam_dir.is_dir():
            continue
        h_cam = hustvl_dir / cam_dir.name
        h_cam.mkdir(exist_ok=True)
        h_imgs = h_cam / "images"
        if not h_imgs.exists():
            h_imgs.symlink_to((cam_dir / "images").resolve())

    # points3D_downsample2.ply
    src_ply = out_dir / "points3D_downsample2.ply"
    dst_ply = hustvl_dir / "points3D_downsample2.ply"
    if src_ply.exists() and not dst_ply.exists():
        shutil.copy(src_ply, dst_ply)

    print(f"  hustvl source dir: {hustvl_dir}")
    return hustvl_dir


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="flame_salmon_1")
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_colmap", action="store_true")
    args = ap.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw" / "n3v" / args.scene
    out_dir = PROJECT_ROOT / "data" / "processed" / "n3v" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        stage_extract_frames(raw_dir, out_dir)

    poses, fl, cx, cy = stage_generate_transforms(raw_dir, out_dir)

    if not args.skip_colmap and not (out_dir / "points3d.ply").exists():
        stage_colmap(out_dir, poses, fl, cx, cy)
    else:
        print("[3/6] COLMAP: points3d.ply already exists, skipping")

    if not (out_dir / "points3D_downsample2.ply").exists():
        stage_downsample(out_dir)
    else:
        print("[4/6] Downsample: points3D_downsample2.ply already exists, skipping")

    stage_frames_symlinks(out_dir)
    stage_flat_images_symlinks(out_dir)
    hustvl_src = stage_hustvl_dir(raw_dir, out_dir)

    print("\n=== Preprocessing complete ===")
    print(f"Main processed dir:  {out_dir}")
    print(f"hustvl source dir:   {hustvl_src}")
    print("  frames/cam*/NNNN.png     → for Deformable (plenopticVideo, poses_bounds.npy)")
    print("  images/cam*_NNNN.png     → for fudan (Blender, transforms_*.json)")
    print("  ..._hustvl_src/cam*.mp4  → for hustvl (dynerf, poses_bounds.npy)")
    print("  points3d.ply             → fudan init")
    print("  points3D_downsample2.ply → hustvl init")


if __name__ == "__main__":
    main()
