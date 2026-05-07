"""
13_benchmark_dnerf.py — Benchmark Deformable-3DGS on D-NeRF Synthetic
=====================================================================

Runs Deformable-3DGS train → render → metrics on each of the 8 D-NeRF scenes
and writes results.json per scene. Sanity check for the dynamic pipeline:
published Deformable-3DGS PSNR ≈ 38–40 dB. If we land within ~1 dB, our
dynamic code path is sound.

Data layout expected:
    data/benchmarks/dnerf_synthetic/{bouncingballs,hellwarrior,hook,
                                     jumpingjacks,lego,mutant,standup,trex}/
        transforms_train.json, transforms_test.json, transforms_val.json
        train/, test/, val/  (RGBA PNGs with `time` field per frame)

Outputs:
    experiments/benchmarks/dnerf_synthetic/{scene}/
        point_cloud/iteration_40000/point_cloud.ply
        deform/iteration_40000/...
        test/ours_40000/{renders,gt,depth}/
        results.json

Usage:
    python scripts/13_benchmark_dnerf.py
    python scripts/13_benchmark_dnerf.py --scenes bouncingballs
    python scripts/13_benchmark_dnerf.py --scenes bouncingballs --iterations 7000
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "benchmarks" / "dnerf_synthetic"
EXP_DIR      = PROJECT_ROOT / "experiments" / "benchmarks" / "dnerf_synthetic"
DEFORM_DIR   = PROJECT_ROOT / "repos" / "Deformable-3D-Gaussians"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

ALL_SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
              "lego", "mutant", "standup", "trex"]


def build_env() -> dict:
    cuda_home = str(Path.home() / "cuda-home")
    return {
        **os.environ,
        "CUDA_HOME":  cuda_home,
        "CC":         "/usr/bin/gcc-12",
        "CXX":        "/usr/bin/g++-12",
        "PYTHONPATH": str(DEFORM_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":       f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
        "TORCH_CUDA_ARCH_LIST": "8.6",
    }


def ensure_dynamic_rasterizer():
    """Reuse the depth-diff rasterizer install from script 09."""
    spec = importlib.util.spec_from_file_location(
        "_s09", SCRIPTS_DIR / "09_train_dynamic_4dgs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_dynamic_rasterizer()


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

    ply = out / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    results = out / "results.json"
    env = build_env()

    # ── Train ──
    if ply.exists() and not force:
        print(f"  ✓ {scene}: trained checkpoint exists, skipping train")
        train_dt = 0.0
    else:
        train_dt = run(
            [sys.executable, DEFORM_DIR / "train.py",
             "--source_path",     src,
             "--model_path",      out,
             "--iterations",      iterations,
             "--test_iterations", iterations,
             "--save_iterations", iterations,
             "--white_background",
             "--eval",
             "--is_blender",
             "--quiet"],
            cwd=DEFORM_DIR, env=env, label=f"train/{scene}",
        )

    # ── Render ──
    rendered = out / "test" / f"ours_{iterations}" / "renders"
    if rendered.exists() and any(rendered.iterdir()) and not force:
        print(f"  ✓ {scene}: renders exist, skipping render")
    else:
        run(
            [sys.executable, DEFORM_DIR / "render.py",
             "--model_path", out,
             "--iteration",  iterations,
             "--skip_train",
             "--mode",       "render",
             "--quiet"],
            cwd=DEFORM_DIR, env=env, label=f"render/{scene}",
        )

    # ── Metrics ──
    if results.exists() and not force:
        print(f"  ✓ {scene}: results.json exists, skipping metrics")
    else:
        run(
            [sys.executable, DEFORM_DIR / "metrics.py",
             "--model_paths", out],
            cwd=DEFORM_DIR, env=env, label=f"metrics/{scene}",
        )

    # ── Read & report ──
    if results.exists():
        with open(results) as f:
            data = json.load(f)
        # Deformable's metrics writes {method_label: {SSIM, PSNR, LPIPS}}
        # method label is "ours_{iter}"
        iter_key = f"ours_{iterations}"
        m = data.get(iter_key) or next(iter(data.values()))
        print(f"  📊 {scene}: PSNR={m['PSNR']:.2f}  "
              f"SSIM={m['SSIM']:.3f}  LPIPS={m['LPIPS']:.3f}")
        return {"scene": scene, "iterations": iterations,
                "train_minutes": train_dt / 60, **m}
    return {"scene": scene, "iterations": iterations,
            "train_minutes": train_dt / 60}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES)
    p.add_argument("--iterations", type=int, default=40_000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"\nBenchmark: D-NeRF Synthetic ({len(args.scenes)} scenes, "
          f"{args.iterations} iters)")
    print("=" * 70)

    ensure_dynamic_rasterizer()

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
            print(f"  {r['scene']:<14} {r['PSNR']:6.2f}  "
                  f"{r['SSIM']:.3f}  {r['LPIPS']:.3f}")
        else:
            print(f"  {r['scene']:<14} {r.get('error', 'no metrics')}")
    if psnrs:
        print(f"  {'mean':<14} {sum(psnrs)/len(psnrs):6.2f}")


if __name__ == "__main__":
    main()
