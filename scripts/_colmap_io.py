"""
_colmap_io.py — Shared COLMAP binary I/O utilities
Used by both 03_prepare_dataset.py and 02b_run_mast3r.py
"""
import struct
from pathlib import Path


def write_cameras_bin(cameras: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam_id, cam in cameras.items():
            params = cam["params"]
            f.write(struct.pack("I", cam_id))
            f.write(struct.pack("i", cam["model_id"]))
            f.write(struct.pack("Q", cam["width"]))
            f.write(struct.pack("Q", cam["height"]))
            f.write(struct.pack(f"{len(params)}d", *params))


def write_images_bin(images: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for im_id, im in images.items():
            f.write(struct.pack("I", im_id))
            f.write(struct.pack("4d", im["qw"], im["qx"], im["qy"], im["qz"]))
            f.write(struct.pack("3d", im["tx"], im["ty"], im["tz"]))
            f.write(struct.pack("I", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))


def write_points3d_bin(points: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(points)))
        for pt_id, pt in points.items():
            f.write(struct.pack("Q", pt_id))
            f.write(struct.pack("3d", pt["x"], pt["y"], pt["z"]))
            f.write(struct.pack("3B", pt["r"], pt["g"], pt["b"]))
            f.write(struct.pack("d", pt["error"]))
            f.write(struct.pack("Q", len(pt["track"])))
            for image_id, point2d_idx in pt["track"]:
                f.write(struct.pack("I", image_id))
                f.write(struct.pack("I", point2d_idx))
