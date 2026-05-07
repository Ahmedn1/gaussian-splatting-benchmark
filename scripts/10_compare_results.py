"""
10_compare_results.py — gather every experiment into one comparison
====================================================================

Walks `experiments/static/*/` and `experiments/dynamic/*/`, parses each run's
`metrics.json` (or training log for dynamic), and prints + saves a single
comparison table.

Also computes a **nearest-neighbor IBR baseline**: for the static case,
"render" cam04 by simply copying its nearest training camera's image. This
gives us a non-ML floor — any 3DGS run worth its compute should beat it.

A real research write-up would also include:
  - Mip-NeRF 360 reproduction on the same scene (NeRF baseline)
  - 4D Gaussian Splatting reproduction (4DGS baseline)
But those are entire repos to set up — for this learning project we keep it
to: IBR < 3DGS-static < Deformable-3DGS-dynamic.

OUTPUT
------
- Pretty table to stdout
- `experiments/comparison_table.json` with raw metrics
- `experiments/comparison_table.md` (Markdown for the README)

Usage:
    python scripts/10_compare_results.py --scene scene_S004
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
EXP_DIR      = PROJECT_ROOT / "experiments"


# ── Static baseline: nearest-neighbor IBR ─────────────────────────────────────

def compute_ibr_baseline(scene_name: str) -> dict | None:
    """
    For each test camera, copy the nearest training camera's image and score
    that. The "nearest" is by camera-center distance in the world frame.

    Returns a metrics dict like the eval scripts produce, or None on failure.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        # Reuse 06's COLMAP readers
        eval_mod = __import__("06_evaluate")
        read_cams_bin   = eval_mod.read_cameras_bin
        read_images_bin = eval_mod.read_images_bin
        quat_to_rotmat  = eval_mod.quat_to_rotmat
    except Exception as e:
        print(f"  ⚠ couldn't import 06_evaluate ({e}); skipping IBR baseline")
        return None

    sparse = PROC_DIR / scene_name / "static" / "sparse" / "0"
    images_dir = PROC_DIR / scene_name / "static" / "images"
    if not (sparse / "images.bin").exists():
        return None

    images_bin = read_images_bin(sparse / "images.bin")
    cams_bin   = read_cams_bin(sparse / "cameras.bin")

    # Read split
    meta_path = PROC_DIR / scene_name / "dynamic_dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        train_cams_set = set(meta["train_cameras"])
        test_cams_set  = set(meta["test_cameras"])
    else:
        train_cams_set = {"cam01", "cam02", "cam03"}
        test_cams_set  = {"cam04"}

    # Compute camera centers
    centers = {}
    for im in images_bin.values():
        R_cw = quat_to_rotmat(im["qw"], im["qx"], im["qy"], im["qz"])
        t = np.array([im["tx"], im["ty"], im["tz"]])
        C = -R_cw.T @ t
        centers[Path(im["name"]).stem] = (C, im["name"])

    # For each test cam, find nearest train cam
    try:
        import torch
        import lpips
        import torchvision.transforms.functional as TF
        from PIL import Image
    except ImportError as e:
        print(f"  ⚠ missing dep for IBR scoring ({e}); skipping")
        return None

    print("  Loading LPIPS for IBR baseline...")
    lpips_model = lpips.LPIPS(net="alex").cuda()
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    test_metrics = []
    per_image = {}

    for test_cam in test_cams_set:
        if test_cam not in centers:
            continue
        C_test, fname_test = centers[test_cam]
        nearest = min(
            (c for c in train_cams_set if c in centers),
            key=lambda c: np.linalg.norm(centers[c][0] - C_test),
        )
        fname_train = centers[nearest][1]

        gt_pil   = Image.open(images_dir / fname_test).convert("RGB")
        pred_pil = Image.open(images_dir / fname_train).convert("RGB").resize(gt_pil.size)

        gt   = TF.to_tensor(gt_pil).unsqueeze(0).cuda()
        pred = TF.to_tensor(pred_pil).unsqueeze(0).cuda()

        # Inline metrics
        mse  = ((pred - gt) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        # Quick SSIM via a simple fallback
        try:
            from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
            ssim_v = ssim_fn(pred, gt, data_range=1.0).item()
        except Exception:
            # Plain Pearson on 11×11 windows is overkill here; fall back to luminance corr
            ssim_v = float(np.clip(np.corrcoef(pred.flatten().cpu().numpy(),
                                                gt.flatten().cpu().numpy())[0, 1], 0, 1))

        with __import__("torch").no_grad():
            lpips_v = lpips_model(pred * 2 - 1, gt * 2 - 1).item()

        per_image[test_cam] = {
            "psnr": psnr, "ssim": ssim_v, "lpips": lpips_v,
            "split": "test", "ibr_nearest": nearest,
        }
        test_metrics.append(per_image[test_cam])

    if not test_metrics:
        return None

    return {
        "method": "ibr_nearest_neighbor",
        "test": {
            "n": len(test_metrics),
            "psnr":  float(np.mean([m["psnr"] for m in test_metrics])),
            "ssim":  float(np.mean([m["ssim"] for m in test_metrics])),
            "lpips": float(np.mean([m["lpips"] for m in test_metrics])),
        },
        "per_image": per_image,
    }


# ── Static gathering ──────────────────────────────────────────────────────────

def gather_static_runs(scene_name: str) -> list[dict]:
    runs = []
    scene_static = EXP_DIR / "static" / scene_name
    if not scene_static.exists():
        return runs

    for exp_dir in sorted(scene_static.iterdir()):
        if not exp_dir.is_dir():
            continue
        eval_dirs = sorted((exp_dir / "eval").glob("iteration_*"),
                           key=lambda p: int(p.name.split("_")[1])) if (exp_dir / "eval").exists() else []
        if not eval_dirs:
            continue
        m_path = eval_dirs[-1] / "metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        config = {}
        cfg_path = exp_dir / "train_config.json"
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text())
        runs.append({
            "method": f"3dgs_static [{exp_dir.name}]",
            "iteration": m.get("iteration"),
            "n_gaussians": m.get("n_gaussians"),
            "train_time_min": config.get("train_time_minutes"),
            "train": m.get("train"),
            "test":  m.get("test"),
        })
    return runs


# ── Dynamic gathering ─────────────────────────────────────────────────────────

def parse_deformable_log(model_dir: Path) -> dict | None:
    """
    Deformable-3DGS doesn't write a metrics.json automatically. Best signals:
      - Final test PSNR from train_config.json (we wrote it)
      - or stdout log of the training process if you piped it

    For now, we just record what we know — train time and "best PSNR" if the
    user has run the official metrics.py separately.
    """
    cfg_path = model_dir / "train_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())

    # Try to find a results.json from running their metrics.py
    results = None
    res_path = model_dir / "results.json"
    if res_path.exists():
        try:
            results = json.loads(res_path.read_text())
        except Exception:
            results = None

    return {"config": cfg, "results": results}


def gather_dynamic_runs(scene_name: str) -> list[dict]:
    runs = []
    scene_dyn = EXP_DIR / "dynamic" / scene_name
    if not scene_dyn.exists():
        return runs

    for exp_dir in sorted(scene_dyn.iterdir()):
        if not exp_dir.is_dir():
            continue
        info = parse_deformable_log(exp_dir)
        if not info:
            continue
        cfg = info["config"]
        res = info["results"]

        test_metrics = None
        if res:
            # Deformable-3DGS metrics.py writes structure: {scene_path: {"PSNR": ..., "SSIM":..., "LPIPS":...}}
            try:
                inner = next(iter(res.values()))
                test_metrics = {
                    "psnr":  float(inner.get("PSNR",  inner.get("psnr",  float("nan")))),
                    "ssim":  float(inner.get("SSIM",  inner.get("ssim",  float("nan")))),
                    "lpips": float(inner.get("LPIPS", inner.get("lpips", float("nan")))),
                    "n":     None,
                }
            except Exception:
                test_metrics = None

        runs.append({
            "method": f"deformable_3dgs [{exp_dir.name}]",
            "iteration": cfg.get("iterations"),
            "n_gaussians": None,
            "train_time_min": cfg.get("train_time_minutes"),
            "train": None,
            "test": test_metrics,
        })
    return runs


# ── Pretty printing ───────────────────────────────────────────────────────────

def fmt_metrics(m):
    if not m:
        return "       —             —            —     "
    psnr = f"{m['psnr']:5.2f}" if m.get("psnr") is not None and not np.isnan(m["psnr"]) else "  —  "
    ssim = f"{m['ssim']:.4f}" if m.get("ssim") is not None and not np.isnan(m["ssim"]) else "  —   "
    lpips = f"{m['lpips']:.4f}" if m.get("lpips") is not None and not np.isnan(m["lpips"]) else "  —   "
    return f"PSNR={psnr}  SSIM={ssim}  LPIPS={lpips}"


def print_table(scene_name, ibr, static_runs, dynamic_runs):
    print(f"\n{'═' * 100}")
    print(f"  {scene_name}  —  Comparison")
    print("═" * 100)

    header = f"{'Method':<40}  {'Train Time':>10}  {'#Gauss':>10}  {'Test (held-out cam) metrics':<45}"
    print(header)
    print("─" * 100)

    if ibr:
        print(f"{'ibr_nearest_neighbor (baseline)':<40}  {'—':>10}  {'—':>10}  {fmt_metrics(ibr['test'])}")
    for r in static_runs:
        time_s = f"{r['train_time_min']:.1f}m" if r.get("train_time_min") else "—"
        ng = f"{r['n_gaussians']:,}" if r.get("n_gaussians") else "—"
        print(f"{r['method']:<40}  {time_s:>10}  {ng:>10}  {fmt_metrics(r['test'])}")
    for r in dynamic_runs:
        time_s = f"{r['train_time_min']:.1f}m" if r.get("train_time_min") else "—"
        ng = f"{r['n_gaussians']:,}" if r.get("n_gaussians") else "—"
        print(f"{r['method']:<40}  {time_s:>10}  {ng:>10}  {fmt_metrics(r['test'])}")
    print("─" * 100)


def to_markdown(scene_name, ibr, static_runs, dynamic_runs) -> str:
    lines = [
        f"## Comparison — {scene_name}",
        "",
        "| Method | Train Time | #Gaussians | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def cell(m, k, fmt):
        if not m or m.get(k) is None or (isinstance(m[k], float) and np.isnan(m[k])):
            return "—"
        return fmt.format(m[k])

    rows = []
    if ibr:
        rows.append(("ibr_nearest_neighbor (baseline)", None, None, ibr["test"]))
    for r in static_runs:
        rows.append((r["method"], r.get("train_time_min"), r.get("n_gaussians"), r.get("test")))
    for r in dynamic_runs:
        rows.append((r["method"], r.get("train_time_min"), r.get("n_gaussians"), r.get("test")))

    for method, t, ng, m in rows:
        t_s = f"{t:.1f} min" if t else "—"
        ng_s = f"{ng:,}" if ng else "—"
        lines.append(
            f"| {method} | {t_s} | {ng_s} | "
            f"{cell(m, 'psnr', '{:.2f}')} | "
            f"{cell(m, 'ssim', '{:.4f}')} | "
            f"{cell(m, 'lpips', '{:.4f}')} |"
        )

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str, default="scene_S004")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip computing the IBR baseline")
    args = parser.parse_args()

    print("Gathering experiments...")
    static_runs  = gather_static_runs(args.scene)
    dynamic_runs = gather_dynamic_runs(args.scene)
    print(f"  Static: {len(static_runs)} run(s)")
    print(f"  Dynamic: {len(dynamic_runs)} run(s)")

    ibr = None
    if not args.no_baseline:
        print("Computing IBR baseline...")
        try:
            ibr = compute_ibr_baseline(args.scene)
            if ibr:
                print(f"  ✓ IBR test PSNR: {ibr['test']['psnr']:.2f}")
        except Exception as e:
            print(f"  ⚠ IBR baseline failed: {e}")

    print_table(args.scene, ibr, static_runs, dynamic_runs)

    out_json = EXP_DIR / f"comparison_{args.scene}.json"
    out_md   = EXP_DIR / f"comparison_{args.scene}.md"

    payload = {
        "scene": args.scene,
        "ibr_baseline": ibr,
        "static_runs": static_runs,
        "dynamic_runs": dynamic_runs,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text(to_markdown(args.scene, ibr, static_runs, dynamic_runs))

    print(f"\nSaved: {out_json}")
    print(f"       {out_md}")


if __name__ == "__main__":
    main()
