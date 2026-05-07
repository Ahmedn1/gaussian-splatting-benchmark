"""
01_extract_frames.py — Frame Extraction Pipeline
=================================================

Extracts frames from raw MPEG videos and organizes them for:
  1. COLMAP input      — one reference frame per camera (for pose estimation)
  2. Static 3DGS       — one frame per camera at a chosen reference timestep
  3. Dynamic 3DGS/4DGS — all frames at reduced FPS, per camera

WHY THIS STEP EXISTS
--------------------
Raw video is not directly usable by 3DGS or COLMAP. We need:
  - Individual JPEG/PNG frames (COLMAP and 3DGS both work on images)
  - Consistent naming conventions (critical for train/test splitting)
  - Controlled frame rate (30fps = 10,000+ images per scene = too slow to train on)

KEY DECISIONS MADE HERE
-----------------------
1. Reference frame = frame 0 per camera
   Why: The scene starts with people walking IN, so early frames may be cleaner.
   If frame 0 has motion blur or a person mid-entry, we can change this.

2. Dynamic FPS = 10 (from 30fps original)
   Why: Every 3rd frame gives good temporal coverage while cutting data 3x.
   At 10fps, S003 (357 frames) becomes 119 timesteps × 4 cameras = 476 images.
   Manageable for training (~1hr on RTX 3080).

3. Resolution scale = 0.5 (960x540)
   Why: Full 1920x1080 is expensive (each image = ~6MB uncompressed).
   0.5x gives good quality/speed tradeoff. 3DGS renders at whatever res we give it.
   Note: COLMAP feature extraction scales well but large images = more SIFT keypoints
   = slower matching. 960x540 is a good sweet spot.

4. Camera model in COLMAP = OPENCV
   Why: Our cameras show barrel distortion (wide-angle lenses).
   OPENCV model handles: fx, fy, cx, cy, k1, k2, p1, p2 (radial + tangential dist.)
   vs. PINHOLE which ignores distortion entirely.
   Getting distortion right is important for accurate 3DGS reconstruction.

OUTPUT STRUCTURE
----------------
data/processed/
└── scene_S003/
    ├── colmap_input/          ← COLMAP reference images (1 per camera)
    │   ├── cam01.jpg
    │   ├── cam02.jpg
    │   ├── cam03.jpg
    │   └── cam04.jpg
    ├── static/
    │   └── images/            ← Same as colmap_input (1 frame per cam)
    │       ├── cam01.jpg
    │       └── ...
    └── dynamic/
        ├── cam01/             ← All frames for cam01 at target FPS
        │   ├── frame_0000.jpg
        │   ├── frame_0003.jpg   (every 3rd frame of original 30fps)
        │   └── ...
        ├── cam02/
        ├── cam03/
        └── cam04/

Usage:
    python scripts/01_extract_frames.py --scene scene_S003
    python scripts/01_extract_frames.py --scene all
    python scripts/01_extract_frames.py --scene scene_S003 --scale 1.0 --dynamic_fps 5
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def get_video_info(video_path: Path) -> dict:
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def extract_frame(video_path: Path, frame_idx: int, out_path: Path,
                  scale: float = 1.0, jpeg_quality: int = 95) -> bool:
    """
    Extract a single frame from a video and save it as JPEG.

    Args:
        video_path:    Path to the source video
        frame_idx:     Which frame to extract (0-indexed)
        out_path:      Where to save the extracted frame
        scale:         Resize factor (0.5 = half resolution)
        jpeg_quality:  JPEG compression quality (95 = near-lossless)

    Returns:
        True if successful, False otherwise
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    if scale != 1.0:
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        # INTER_AREA is best for downscaling — averages pixels, avoids aliasing
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return True


def extract_frames_at_fps(video_path: Path, out_dir: Path,
                           target_fps: float, scale: float = 1.0,
                           jpeg_quality: int = 95) -> list[int]:
    """
    Extract frames from a video at a target frame rate.

    Example: 30fps video → target_fps=10 → extract every 3rd frame.

    The frame indices extracted are returned so they can be recorded
    and matched across cameras (important: all cameras must have the
    same set of timesteps, otherwise the dynamic dataset is misaligned).

    Args:
        video_path:  Source video
        out_dir:     Where to save frames (named frame_XXXX.jpg)
        target_fps:  Desired output frame rate
        scale:       Resize factor
        jpeg_quality: JPEG quality

    Returns:
        List of original frame indices that were extracted
    """
    info = get_video_info(video_path)
    src_fps    = info["fps"]
    n_frames   = info["n_frames"]

    # Compute stride: how many source frames to skip between extractions
    # stride=1 → keep every frame (target_fps = src_fps)
    # stride=3 → keep every 3rd frame (target_fps = src_fps / 3)
    stride = max(1, round(src_fps / target_fps))
    frame_indices = list(range(0, n_frames, stride))

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    extracted = []
    for idx in tqdm(frame_indices, desc=f"  {video_path.name}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if scale != 1.0:
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Naming convention: frame_XXXX.jpg where XXXX is the SOURCE frame index
        # WHY source frame index: keeps alignment across cameras obvious.
        # frame_0000.jpg from cam01 and frame_0000.jpg from cam02 are the same timestep.
        out_path = out_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        extracted.append(idx)

    cap.release()
    return extracted


def process_scene(scene_name: str, scale: float = 0.5,
                  dynamic_fps: float = 10.0, ref_frame: int = 0,
                  n_colmap_frames: int = 3,
                  jpeg_quality: int = 95) -> dict:
    """
    Process one scene: extract COLMAP reference frames + static frames + dynamic frames.

    Args:
        scene_name:   e.g. 'scene_S003'
        scale:        Resolution scaling factor (0.5 = half)
        dynamic_fps:  Target FPS for dynamic frame extraction
        ref_frame:    Which source frame to use as the reference (for COLMAP + static)
        jpeg_quality: JPEG compression quality

    Returns:
        Dictionary with stats about what was extracted
    """
    scene_raw_dir  = RAW_DIR / scene_name
    scene_proc_dir = PROC_DIR / scene_name

    if not scene_raw_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_raw_dir}")

    # Find all camera videos for this scene
    videos = sorted(scene_raw_dir.glob("*.mpeg"))
    videos = [v for v in videos if v.stat().st_size > 0]  # skip empty files

    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"  Cameras found:    {len(videos)}")
    print(f"  Scale:            {scale}x")
    print(f"  Dynamic FPS:      {dynamic_fps}")
    print(f"  COLMAP frames/cam:{n_colmap_frames}  (more = more robust pose estimation)")
    print(f"{'='*60}")

    # ── Reference frame extraction ──────────────────────────────────────────────
    # For COLMAP: n_colmap_frames per camera, evenly spaced through the video.
    # WHY multiple frames: With only 1 frame per camera, cross-camera feature
    # matching can be sparse (cameras see different parts of the scene).
    # Multiple frames give COLMAP more opportunities to:
    #   1. Build within-camera tracks (same camera, different timesteps)
    #   2. Find cross-camera matches via shared background features
    # The cameras are FIXED, so all frames from cam01 have the SAME pose.
    # COLMAP will estimate one pose per camera (not per frame) via bundle adjustment.
    #
    # For static 3DGS: use just ONE reference frame per camera (the middle one).
    # This keeps the static dataset clean and consistent.

    colmap_dir = scene_proc_dir / "colmap_input"
    static_dir = scene_proc_dir / "static" / "images"

    print(f"\n[1/2] Extracting reference frames ({n_colmap_frames} per camera)...")
    cam_infos = []
    for video_path in videos:
        cam_id = video_path.stem.split("_cam")[1]
        cam_name = f"cam{cam_id}"
        info = get_video_info(video_path)
        n_frames = info["n_frames"]

        # Choose evenly-spaced frame indices for COLMAP
        # Skip first and last 5% of video (often has entry/exit motion)
        start = max(0, int(n_frames * 0.05))
        end   = min(n_frames - 1, int(n_frames * 0.95))
        colmap_indices = np.linspace(start, end, n_colmap_frames, dtype=int).tolist()

        # Extract multiple frames for COLMAP
        for i, fidx in enumerate(colmap_indices):
            out_name = f"{cam_name}_ref{i:02d}.jpg"  # cam01_ref00.jpg, cam01_ref01.jpg
            extract_frame(video_path, fidx, colmap_dir / out_name, scale, jpeg_quality)

        # Extract SINGLE frame for static training (use middle frame)
        mid_idx = colmap_indices[len(colmap_indices) // 2]
        static_name = f"{cam_name}.jpg"
        success_static = extract_frame(video_path, mid_idx,
                                        static_dir / static_name, scale, jpeg_quality)

        status = "✅" if success_static else "❌"
        print(f"  {status} {cam_name}: {info['width']}x{info['height']} → "
              f"{int(info['width']*scale)}x{int(info['height']*scale)} | "
              f"COLMAP frames: {colmap_indices} | static frame: {mid_idx}")
        cam_infos.append({"cam_id": cam_id, "cam_name": cam_name,
                          "n_frames": n_frames, "fps": info["fps"],
                          "colmap_frame_indices": colmap_indices,
                          "static_frame_idx": mid_idx})

    # ── Dynamic frame extraction ────────────────────────────────────────────────
    # Extract all frames at target_fps from each camera.
    # Frames are stored per-camera in separate subdirectories.
    # WHY per-camera subdirs: easier to manage and inspect. Later we'll
    # merge them into the 3DGS images/ folder with camera-prefixed names.

    print(f"\n[2/2] Extracting dynamic frames at {dynamic_fps}fps...")
    all_extracted_indices = None  # Will be set from first camera
    dynamic_stats = {}

    for video_path in videos:
        cam_id = video_path.stem.split("_cam")[1]
        cam_name = f"cam{cam_id}"
        out_dir = scene_proc_dir / "dynamic" / cam_name

        print(f"\n  {cam_name}:")
        extracted = extract_frames_at_fps(video_path, out_dir,
                                           dynamic_fps, scale, jpeg_quality)

        # Verify frame alignment across cameras
        # All cameras MUST have the same set of frame indices extracted
        if all_extracted_indices is None:
            all_extracted_indices = extracted
        else:
            if extracted != all_extracted_indices:
                print(f"  ⚠️  Frame count mismatch vs cam01!")
                print(f"     Expected {len(all_extracted_indices)} frames, got {len(extracted)}")
                # Use intersection to keep only aligned frames
                common = sorted(set(all_extracted_indices) & set(extracted))
                print(f"     Using {len(common)} common frames")

        duration = len(extracted) / dynamic_fps
        dynamic_stats[cam_name] = {"n_frames": len(extracted), "duration": duration}
        print(f"  → {len(extracted)} frames | {duration:.1f}s at {dynamic_fps}fps")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"✅ {scene_name} extraction complete")
    print(f"   COLMAP input:  {colmap_dir}  ({len(videos)} images)")
    print(f"   Static images: {static_dir}  ({len(videos)} images)")
    print(f"   Dynamic:       {scene_proc_dir / 'dynamic'}")
    for cam, stats in dynamic_stats.items():
        print(f"     {cam}: {stats['n_frames']} frames")

    # Save metadata
    import json
    meta = {
        "scene":       scene_name,
        "scale":       scale,
        "dynamic_fps": dynamic_fps,
        "ref_frame":   ref_frame,
        "cameras":     cam_infos,
        "dynamic_frame_indices": all_extracted_indices,
        "n_dynamic_frames": len(all_extracted_indices) if all_extracted_indices else 0,
    }
    meta_path = scene_proc_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Metadata:      {meta_path}")

    return meta


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from raw MPEG videos for 3DGS/4DGS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--scene", type=str, default="scene_S003",
                        help="Scene name (e.g. scene_S003) or 'all' for all scenes")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Resolution scale factor (default: 0.5 = half resolution)")
    parser.add_argument("--dynamic_fps", type=float, default=10.0,
                        help="Target FPS for dynamic frame extraction (default: 10)")
    parser.add_argument("--ref_frame", type=int, default=0,
                        help="Source frame index to use as reference (default: 0)")
    parser.add_argument("--n_colmap_frames", type=int, default=3,
                        help="Number of frames per camera for COLMAP input (default: 3)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality for saved frames (default: 95)")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted([d.name for d in RAW_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("scene_")])
        print(f"Processing all {len(scenes)} scenes: {scenes}")
    else:
        scenes = [args.scene]

    results = {}
    for scene in scenes:
        try:
            meta = process_scene(
                scene_name=scene,
                scale=args.scale,
                dynamic_fps=args.dynamic_fps,
                ref_frame=args.ref_frame,
                n_colmap_frames=args.n_colmap_frames,
                jpeg_quality=args.jpeg_quality,
            )
            results[scene] = {"status": "ok", "meta": meta}
        except Exception as e:
            print(f"❌ Failed to process {scene}: {e}")
            results[scene] = {"status": "error", "error": str(e)}

    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    for scene, result in results.items():
        status = "✅" if result["status"] == "ok" else "❌"
        if result["status"] == "ok":
            n_cams   = len(result["meta"]["cameras"])
            n_dynfr  = result["meta"]["n_dynamic_frames"]
            print(f"  {status} {scene}: {n_cams} cameras, {n_dynfr} dynamic frames/camera")
        else:
            print(f"  {status} {scene}: {result['error']}")


if __name__ == "__main__":
    main()
