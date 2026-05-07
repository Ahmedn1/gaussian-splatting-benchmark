"""
15_benchmark_4dgs_hustvl.py — Benchmark hustvl/4DGaussians on D-NeRF Synthetic
==============================================================================

Runs hustvl/4DGaussians (Wu et al., CVPR 2024) train → render → metrics on
each of the 8 D-NeRF scenes. Apples-to-apples comparison with our Stage 2
Deformable-3DGS results.

Uses a separate conda env (`nr-4dgs-hustvl`) with hustvl's rasterizer to
avoid the diff_gaussian_rasterization namespace clash.

Outputs:
    experiments/benchmarks/dnerf_synthetic_4dgs_hustvl/{scene}/
        point_cloud/iteration_*/...
        test/ours_20000/{renders,gt}/
        results.json

Usage:
    python scripts/15_benchmark_4dgs_hustvl.py
    python scripts/15_benchmark_4dgs_hustvl.py --scenes bouncingballs
    python scripts/15_benchmark_4dgs_hustvl.py --scenes bouncingballs --iterations 7000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "benchmarks" / "dnerf_synthetic"
EXP_DIR      = PROJECT_ROOT / "experiments" / "benchmarks" / "dnerf_synthetic_4dgs_hustvl"
HUSTVL_DIR   = PROJECT_ROOT / "repos" / "4DGaussians"
HUSTVL_PY    = Path.home() / "miniconda3" / "envs" / "nr-4dgs-hustvl" / "bin" / "python"

ALL_SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
              "lego", "mutant", "standup", "trex"]


def build_env() -> dict:
    cuda_home = str(Path.home() / "cuda-home")
    return {
        **os.environ,
        "CUDA_HOME":  cuda_home,
        "CC":         "/usr/bin/gcc-12",
        "CXX":        "/usr/bin/g++-12",
        "PYTHONPATH": str(HUSTVL_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":       f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
        "TORCH_CUDA_ARCH_LIST": "8.6",
    }


def run(cmd: list[str], cwd: Path, env: dict, label: str):
    print(f"\n  [{label}] {' '.join(str(c) for c in cmd[-6:])}")
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {r.returncode})")
    print(f"  [{label}] done in {dt/60:.1f} min")
    return dt


def benchmark_scene(scene: str, iterations: int, force: bool):
    src = DATA_DIR / scene
    if not src.exists():
        raise FileNotFoundError(f"Scene data not found: {src}")
    out = EXP_DIR / scene
    out.mkdir(parents=True, exist_ok=True)

    cfg = HUSTVL_DIR / "arguments" / "dnerf" / f"{scene}.py"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing hustvl config: {cfg}")

    # hustvl saves to model_path/point_cloud/iteration_<save_iter>/...
    ply = out / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    results = out / "results.json"
    env = build_env()

    # ── Train ──
    if ply.exists() and not force:
        print(f"  ✓ {scene}: trained checkpoint exists, skipping train")
        train_dt = 0.0
    else:
        train_dt = run(
            [HUSTVL_PY, HUSTVL_DIR / "train.py",
             "--source_path",     src,
             "--model_path",      out,
             "--iterations",      iterations,
             "--save_iterations", iterations,
             "--test_iterations", iterations,
             "--expname",         f"dnerf/{scene}",
             "--configs",         cfg,
             "--quiet"],
            cwd=HUSTVL_DIR, env=env, label=f"train/{scene}",
        )

    # ── Render ──
    rendered = out / "test" / f"ours_{iterations}" / "renders"
    if rendered.exists() and any(rendered.iterdir()) and not force:
        print(f"  ✓ {scene}: renders exist, skipping render")
    else:
        run(
            [HUSTVL_PY, HUSTVL_DIR / "render.py",
             "--model_path", out,
             "--iteration",  iterations,
             "--skip_train",
             "--skip_video",
             "--configs",    cfg,
             "--quiet"],
            cwd=HUSTVL_DIR, env=env, label=f"render/{scene}",
        )

    # ── Metrics ──
    if results.exists() and not force:
        print(f"  ✓ {scene}: results.json exists, skipping metrics")
    else:
        run(
            [HUSTVL_PY, HUSTVL_DIR / "metrics.py",
             "--model_paths", out],
            cwd=HUSTVL_DIR, env=env, label=f"metrics/{scene}",
        )

    if results.exists():
        with open(results) as f:
            data = json.load(f)
        # hustvl writes {method: {metrics}} (already keyed by method, scene_dir
        # stripped on dump per metrics.py line 107)
        method_key = f"ours_{iterations}"
        m = data.get(method_key) or next(iter(data.values()))
        print(f"  📊 {scene}: PSNR={m.get('PSNR', 0):.2f}  "
              f"SSIM={m.get('SSIM', 0):.3f}  LPIPS={m.get('LPIPS-vgg', m.get('LPIPS', 0)):.3f}")
        return {"scene": scene, "iterations": iterations,
                "train_minutes": train_dt / 60, **m}
    return {"scene": scene, "iterations": iterations,
            "train_minutes": train_dt / 60}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES)
    p.add_argument("--iterations", type=int, default=20_000,
                   help="Default 20000 (matches hustvl's D-NeRF config)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not HUSTVL_PY.exists():
        sys.exit(f"hustvl env python not found: {HUSTVL_PY}")

    print(f"\nBenchmark: hustvl-4DGaussians on D-NeRF Synthetic "
          f"({len(args.scenes)} scenes, {args.iterations} iters)")
    print("=" * 70)

    summary = []
    t_start = time.time()
    for scene in args.scenes:
        print(f"\n── {scene} " + "─" * (66 - len(scene)))
        try:
            row = benchmark_scene(scene, args.iterations, args.force)
            summary.append(row)
        except Exception as e:
            print(f"  ❌ {scene} failed: {e}")
            summary.append({"scene": scene, "error": str(e)})

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EXP_DIR / f"summary_{args.iterations}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Total wallclock: {(time.time() - t_start) / 60:.1f} min")
    print(f"Summary: {summary_path}")
    print("\n  scene           PSNR    SSIM    LPIPS")
    print("  ───────────────────────────────────────")
    psnrs = []
    for r in summary:
        if "PSNR" in r:
            psnrs.append(r["PSNR"])
            lp = r.get("LPIPS-vgg", r.get("LPIPS", 0))
            print(f"  {r['scene']:<14} {r['PSNR']:6.2f}  "
                  f"{r['SSIM']:.3f}  {lp:.3f}")
        else:
            print(f"  {r['scene']:<14} {r.get('error', 'no metrics')}")
    if psnrs:
        print(f"  {'mean':<14} {sum(psnrs)/len(psnrs):6.2f}")


if __name__ == "__main__":
    main()
