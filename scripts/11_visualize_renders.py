"""
11_visualize_renders.py — turn on-disk renders into composite images & video
=============================================================================

Reads the PNGs that 06_evaluate.py and Deformable-3DGS's render.py wrote, and
produces:

  1. A method-comparison composite PNG: one row per static experiment, columns
     for each camera × (render | GT | amplified |error|).  Test cam (cam04)
     gets a colored border so you can find it instantly.

  2. A per-experiment detail PNG, same layout as (1) but for one method, at
     higher per-cell resolution.

  3. A dynamic side-by-side MP4: cam04 over 151 frames, three panels per frame
     (render | GT | depth), encoded at the dataset's effective dynamic FPS.

No GPU, no model loading — pure image composition over existing artifacts.

Usage:
    python scripts/11_visualize_renders.py --scene scene_S004
    python scripts/11_visualize_renders.py --scene scene_S004 --skip_video
    python scripts/11_visualize_renders.py --scene scene_S004 --only_dynamic
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
EXP_STATIC   = PROJECT_ROOT / "experiments" / "static"
EXP_DYNAMIC  = PROJECT_ROOT / "experiments" / "dynamic"

# Display constants
DIFF_AMP   = 4.0           # multiply |render − gt| by this before clipping
CAM_BORDER_PX = 6          # border width around test-cam tiles
COLOR_TEST  = (255, 96,  96)
COLOR_TRAIN = (96, 192,  96)
LABEL_BG    = (28, 28, 28)
LABEL_FG    = (240, 240, 240)


# ── small helpers ──────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.ImageFont:
    """Load a TTF if available, else fall back to PIL's bitmap default."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _read_rgb(path: Path) -> np.ndarray:
    """Load PNG as float32 RGB in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _abs_diff(render: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-pixel |render − gt|, gain-amplified, clipped to [0, 1]."""
    return np.clip(np.abs(render - gt) * DIFF_AMP, 0.0, 1.0)


def _psnr(render: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((render - gt) ** 2))
    if mse < 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _bordered(img_arr: np.ndarray, color, w: int = CAM_BORDER_PX) -> np.ndarray:
    """Paint a border of `color` on top of img_arr (uint8, HxWx3). Inplace returns."""
    out = img_arr.copy()
    out[:w, :, :]  = color
    out[-w:, :, :] = color
    out[:, :w, :]  = color
    out[:, -w:, :] = color
    return out


def _label_strip(text: str, w: int, h: int = 28,
                 fg=LABEL_FG, bg=LABEL_BG) -> np.ndarray:
    """Render a horizontal text strip of size (h, w, 3)."""
    im = Image.new("RGB", (w, h), bg)
    drw = ImageDraw.Draw(im)
    drw.text((6, 4), text, fill=fg, font=_font(14))
    return np.asarray(im)


def _depth_to_rgb(depth_path: Path) -> np.ndarray:
    """
    Deformable-3DGS saves depth as a 16-bit-ish PNG. Open as numeric array,
    normalize to [0, 1], then apply a perceptually-decent colormap.
    """
    pil = Image.open(depth_path)
    arr = np.asarray(pil)
    if arr.ndim == 3:
        # if it's already RGB, just return as-is
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr)
    # turbo-ish: interpolate through a small palette
    palette = np.array([
        [48,  18,  59],   # deep blue
        [70,  43, 145],
        [40, 119, 174],
        [69, 195, 165],
        [196, 230,  43],  # green-yellow
        [253, 165,  44],
        [180,  29,  10],  # red
    ], dtype=np.float32) / 255.0
    idx = arr * (len(palette) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(palette) - 1)
    frac = (idx - lo)[..., None]
    out = (1 - frac) * palette[lo] + frac * palette[hi]
    return out  # H, W, 3 in [0, 1]


# ── static composite ──────────────────────────────────────────────────────────

CAM_NAMES = ["cam01", "cam02", "cam03", "cam04"]
TRAIN_CAMS_DEFAULT = {"cam01", "cam02", "cam03"}
TEST_CAMS_DEFAULT  = {"cam04"}


def _read_split(scene_proc: Path):
    meta = scene_proc / "dynamic_dataset_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        return set(m["train_cameras"]), set(m["test_cameras"])
    return TRAIN_CAMS_DEFAULT, TEST_CAMS_DEFAULT


def _latest_eval_dir(exp_dir: Path) -> Path | None:
    eval_root = exp_dir / "eval"
    if not eval_root.exists():
        return None
    iters = sorted(eval_root.glob("iteration_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    return iters[-1] if iters else None


def _load_metrics(eval_dir: Path) -> dict:
    p = eval_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _per_camera_triplet(eval_dir: Path, cam: str, train_cams) -> np.ndarray:
    """
    Return a (h, 3w, 3) uint8 strip: render | GT | amp-diff for one camera.
    Borders are painted around each tile (red for test, green for train).
    """
    render_path = eval_dir / "renders" / f"{cam}.png"
    gt_path     = eval_dir / "gt"      / f"{cam}.png"
    if not (render_path.exists() and gt_path.exists()):
        return None

    render = _read_rgb(render_path)
    gt     = _read_rgb(gt_path)
    h, w, _ = render.shape
    diff = _abs_diff(render, gt)

    color = COLOR_TRAIN if cam in train_cams else COLOR_TEST
    panels = [_bordered(_to_uint8(arr), color) for arr in (render, gt, diff)]
    return np.concatenate(panels, axis=1)


def build_method_comparison(scene: str, out_path: Path):
    scene_static = EXP_STATIC / scene
    if not scene_static.exists():
        print(f"  ⚠ no static experiments at {scene_static}")
        return

    exps = sorted([p for p in scene_static.iterdir() if p.is_dir()])
    train_cams, _ = _read_split(PROC_DIR / scene)

    rows = []
    for exp_dir in exps:
        eval_dir = _latest_eval_dir(exp_dir)
        if eval_dir is None:
            print(f"  ⏩ no eval for {exp_dir.name}, skipping row")
            continue

        # Compose a row: 4 camera triplets stacked horizontally
        cam_triplets = []
        for cam in CAM_NAMES:
            t = _per_camera_triplet(eval_dir, cam, train_cams)
            if t is not None:
                cam_triplets.append(t)
        if not cam_triplets:
            continue
        row_imgs = np.concatenate(cam_triplets, axis=1)

        # Add a left-side label with method + summary metrics
        m = _load_metrics(eval_dir)
        test = m.get("test", {})
        train = m.get("train", {})
        n_g = m.get("n_gaussians", 0)
        label_txt = (f"{exp_dir.name}  |  iter {m.get('iteration', '?'):>5}  |  "
                     f"#g={n_g:,}\n"
                     f"train  PSNR={train.get('psnr', 0):5.2f}  "
                     f"SSIM={train.get('ssim', 0):.3f}  "
                     f"LPIPS={train.get('lpips', 0):.3f}\n"
                     f"test   PSNR={test.get('psnr', 0):5.2f}  "
                     f"SSIM={test.get('ssim', 0):.3f}  "
                     f"LPIPS={test.get('lpips', 0):.3f}")
        label_w = 360
        label = Image.new("RGB", (label_w, row_imgs.shape[0]), LABEL_BG)
        drw = ImageDraw.Draw(label)
        drw.multiline_text((10, 10), label_txt, fill=LABEL_FG,
                           font=_font(16), spacing=6)
        row = np.concatenate([np.asarray(label), row_imgs], axis=1)
        rows.append(row)

    if not rows:
        print("  ⚠ nothing to compose")
        return

    # Header strip with column captions
    label_w = 360
    triplet_w = rows[0].shape[1] - label_w
    cell_w = triplet_w // (4 * 3)
    header_text = []
    for cam in CAM_NAMES:
        header_text.extend([f"{cam} render", f"{cam} GT", f"{cam} |Δ|×{int(DIFF_AMP)}"])
    header_pieces = [np.asarray(Image.new("RGB", (label_w, 32), LABEL_BG))]
    for txt in header_text:
        header_pieces.append(_label_strip(txt, cell_w, 32))
    # Pad to match exact width
    header = np.concatenate(header_pieces, axis=1)
    if header.shape[1] != rows[0].shape[1]:
        pad = rows[0].shape[1] - header.shape[1]
        if pad > 0:
            header = np.concatenate(
                [header, np.full((header.shape[0], pad, 3), LABEL_BG, dtype=np.uint8)],
                axis=1,
            )
        else:
            header = header[:, :rows[0].shape[1]]

    final = np.concatenate([header] + rows, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(out_path, optimize=True)
    print(f"  ✅ method comparison: {out_path}  ({final.shape[1]}×{final.shape[0]})")


# ── per-experiment detail panels ──────────────────────────────────────────────

def build_experiment_detail(scene: str, exp_dir: Path):
    eval_dir = _latest_eval_dir(exp_dir)
    if eval_dir is None:
        return
    train_cams, _ = _read_split(PROC_DIR / scene)

    # 4 rows × (render | GT | diff). Each row a camera.
    cam_rows = []
    for cam in CAM_NAMES:
        t = _per_camera_triplet(eval_dir, cam, train_cams)
        if t is None:
            continue
        # Left-side label per camera
        label_w = 100
        is_test = cam not in train_cams
        label = Image.new("RGB", (label_w, t.shape[0]), LABEL_BG)
        drw = ImageDraw.Draw(label)
        tag = "TEST" if is_test else "train"
        drw.text((12, t.shape[0]//2 - 24), cam, fill=LABEL_FG, font=_font(20))
        drw.text((12, t.shape[0]//2 + 4),  tag,
                 fill=COLOR_TEST if is_test else COLOR_TRAIN, font=_font(16))
        cam_rows.append(np.concatenate([np.asarray(label), t], axis=1))

    if not cam_rows:
        return

    # Title bar with metrics
    m = _load_metrics(eval_dir)
    test = m.get("test", {})
    train = m.get("train", {})
    title_h = 60
    title = Image.new("RGB", (cam_rows[0].shape[1], title_h), LABEL_BG)
    drw = ImageDraw.Draw(title)
    drw.text((20, 8),
             f"{exp_dir.name}   iter={m.get('iteration', '?')}   "
             f"#g={m.get('n_gaussians', 0):,}",
             fill=LABEL_FG, font=_font(20))
    drw.text((20, 34),
             f"train PSNR/SSIM/LPIPS = "
             f"{train.get('psnr', 0):.2f}/{train.get('ssim', 0):.3f}/{train.get('lpips', 0):.3f}      "
             f"test PSNR/SSIM/LPIPS = "
             f"{test.get('psnr', 0):.2f}/{test.get('ssim', 0):.3f}/{test.get('lpips', 0):.3f}",
             fill=LABEL_FG, font=_font(15))

    # Column header
    col_h = 28
    full_w = cam_rows[0].shape[1]
    cell_w = (full_w - 100) // 3  # the row label is 100 wide; rest is render|gt|diff
    cols = [np.asarray(Image.new("RGB", (100, col_h), LABEL_BG)),
            _label_strip("render", cell_w, col_h),
            _label_strip("ground truth", cell_w, col_h),
            _label_strip(f"|Δ|×{int(DIFF_AMP)}", cell_w, col_h)]
    col_header = np.concatenate(cols, axis=1)
    if col_header.shape[1] != full_w:
        diff = full_w - col_header.shape[1]
        if diff > 0:
            col_header = np.concatenate(
                [col_header,
                 np.full((col_h, diff, 3), LABEL_BG, dtype=np.uint8)], axis=1)
        else:
            col_header = col_header[:, :full_w]

    final = np.concatenate([np.asarray(title), col_header] + cam_rows, axis=0)
    out = eval_dir / "render_panel.png"
    Image.fromarray(final).save(out, optimize=True)
    print(f"    ✅ {exp_dir.name}: {out}")


# ── dynamic video ──────────────────────────────────────────────────────────────

def build_dynamic_video(scene: str, out_path: Path,
                         exp_name: str = "baseline", iteration: int = 40000):
    test_root = (EXP_DYNAMIC / scene / exp_name / "test" / f"ours_{iteration}")
    renders_dir = test_root / "renders"
    gt_dir      = test_root / "gt"
    depth_dir   = test_root / "depth"
    if not renders_dir.exists():
        print(f"  ⚠ no dynamic renders at {renders_dir}")
        return

    frame_files = sorted(renders_dir.glob("*.png"))
    if not frame_files:
        print(f"  ⚠ no PNGs in {renders_dir}")
        return

    # Get FPS from metadata (effective dynamic FPS)
    meta_path = PROC_DIR / scene / "metadata.json"
    fps = 10.0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        fps = float(m.get("dynamic_fps", 10.0))

    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Encoding {len(frame_files)} frames @ {fps} fps → {out_path}")

    # Use ffmpeg writer; libx264 + yuv420p plays everywhere
    with imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for i, render_p in enumerate(frame_files):
            gt_p = gt_dir / render_p.name
            depth_p = depth_dir / render_p.name
            if not gt_p.exists():
                continue

            render = _read_rgb(render_p)
            gt     = _read_rgb(gt_p)
            psnr_v = _psnr(render, gt)

            depth_rgb = _depth_to_rgb(depth_p) if depth_p.exists() else np.zeros_like(render)

            # Build a single 3-panel frame with thin separators
            sep = np.full((render.shape[0], 4, 3), 1.0, dtype=np.float32)
            row = np.concatenate(
                [render, sep, gt, sep, depth_rgb], axis=1
            )
            row_u = _to_uint8(row)

            # Bottom strip with frame index + per-frame PSNR
            strip_h = 36
            strip = Image.new("RGB", (row_u.shape[1], strip_h), LABEL_BG)
            drw = ImageDraw.Draw(strip)
            drw.text((12, 8),
                     f"frame {i:03d}/{len(frame_files)-1}    "
                     f"render | GT | depth     "
                     f"PSNR={psnr_v:.2f} dB",
                     fill=LABEL_FG, font=_font(16))
            frame = np.concatenate([row_u, np.asarray(strip)], axis=0)
            writer.append_data(frame)

    print(f"  ✅ video: {out_path}  ({len(frame_files)} frames @ {fps} fps)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str, default="scene_S004")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip dynamic video encoding")
    parser.add_argument("--skip_static", action="store_true",
                        help="Skip static composites")
    parser.add_argument("--only_dynamic", action="store_true",
                        help="Only build the dynamic video")
    parser.add_argument("--dynamic_exp",  type=str, default="baseline")
    parser.add_argument("--dynamic_iter", type=int, default=40000)
    args = parser.parse_args()

    do_static  = not (args.skip_static or args.only_dynamic)
    do_dynamic = not args.skip_video

    print(f"Visualizing renders for {args.scene}...")

    if do_static:
        # 1) Method-comparison composite
        print("\n[1/2] Method-comparison composite...")
        build_method_comparison(
            args.scene,
            EXP_STATIC / args.scene / "comparison_panel.png",
        )

        # 2) Per-experiment details
        print("\n[2/2] Per-experiment detail panels...")
        scene_static = EXP_STATIC / args.scene
        for exp_dir in sorted(p for p in scene_static.iterdir() if p.is_dir()):
            build_experiment_detail(args.scene, exp_dir)

    if do_dynamic:
        print("\n[+]  Dynamic video...")
        out = (EXP_DYNAMIC / args.scene / args.dynamic_exp / "cam04_compare.mp4")
        build_dynamic_video(args.scene, out,
                            exp_name=args.dynamic_exp,
                            iteration=args.dynamic_iter)

    print("\nDone.")


if __name__ == "__main__":
    main()
