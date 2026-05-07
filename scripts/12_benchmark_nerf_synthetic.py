"""
12_benchmark_nerf_synthetic.py — Benchmark vanilla 3DGS on NeRF Synthetic
=========================================================================

Runs the reference 3DGS pipeline (train → render → metrics) on each of the
8 NeRF Synthetic scenes and writes results.json per scene. This is a sanity
check: published 3DGS PSNR ≈ 33 dB. If we land within ~1 dB, our static
pipeline (scripts/05) is sound and the scene_S004 disaster was purely a
data problem.

Data layout expected:
    data/benchmarks/nerf_synthetic/{lego,chair,drums,ficus,hotdog,
                                    materials,mic,ship}/
        transforms_train.json, transforms_test.json, transforms_val.json
        train/, test/, val/  (PNGs)

Outputs:
    experiments/benchmarks/nerf_synthetic/{scene}/
        point_cloud/iteration_30000/point_cloud.ply
        test/ours_30000/{renders,gt}/
        results.json   ← PSNR / SSIM / LPIPS over test split

Usage:
    python scripts/12_benchmark_nerf_synthetic.py
    python scripts/12_benchmark_nerf_synthetic.py --scenes lego chair
    python scripts/12_benchmark_nerf_synthetic.py --scenes lego --iterations 7000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "benchmarks" / "nerf_synthetic"
EXP_DIR      = PROJECT_ROOT / "experiments" / "benchmarks" / "nerf_synthetic"
GS_DIR       = PROJECT_ROOT / "repos" / "gaussian-splatting"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

ALL_SCENES = ["lego", "chair", "drums", "ficus", "hotdog",
              "materials", "mic", "ship"]


def build_env() -> dict:
    """Subprocess env with CUDA paths + non-depth rasterizer assumed active."""
    cuda_home = str(Path.home() / "cuda-home")
    return {
        **os.environ,
        "CUDA_HOME":  cuda_home,
        "CC":         "/usr/bin/gcc-12",
        "CXX":        "/usr/bin/g++-12",
        "PYTHONPATH": str(GS_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":       f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
        "TORCH_CUDA_ARCH_LIST": "8.6",
    }


def ensure_static_rasterizer():
    """Reuse the rasterizer-management routine from script 05."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    # Import lazily to avoid dragging in script 05's argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_s05", SCRIPTS_DIR / "05_train_static_3dgs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_static_rasterizer()


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
            [sys.executable, GS_DIR / "train.py",
             "--source_path",     src,
             "--model_path",      out,
             "--iterations",      iterations,
             "--test_iterations", iterations,
             "--save_iterations", iterations,
             "--white_background",
             "--eval",
             "--quiet",
             "--disable_viewer"],
            cwd=GS_DIR, env=env, label=f"train/{scene}",
        )

    # ── Render ──
    rendered = out / "test" / f"ours_{iterations}" / "renders"
    if rendered.exists() and any(rendered.iterdir()) and not force:
        print(f"  ✓ {scene}: renders exist, skipping render")
    else:
        run(
            [sys.executable, GS_DIR / "render.py",
             "--model_path", out,
             "--iteration",  iterations,
             "--skip_train",
             "--quiet"],
            cwd=GS_DIR, env=env, label=f"render/{scene}",
        )

    # ── Metrics ──
    if results.exists() and not force:
        print(f"  ✓ {scene}: results.json exists, skipping metrics")
    else:
        run(
            [sys.executable, GS_DIR / "metrics.py",
             "--model_paths", out],
            cwd=GS_DIR, env=env, label=f"metrics/{scene}",
        )

    # ── Read & report ──
    if results.exists():
        with open(results) as f:
            data = json.load(f)
        # results.json is keyed by iteration label, e.g. "ours_30000"
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
    p.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                   choices=ALL_SCENES,
                   help="Subset of scenes to run (default: all 8)")
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--force", action="store_true",
                   help="Re-run train/render/metrics even if outputs exist")
    args = p.parse_args()

    print(f"\nBenchmark: NeRF Synthetic ({len(args.scenes)} scenes, "
          f"{args.iterations} iters)")
    print("=" * 70)

    ensure_static_rasterizer()

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

    # ── Aggregate summary ──
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EXP_DIR / f"summary_{args.iterations}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Total wallclock: {(time.time() - t_start) / 60:.1f} min")
    print(f"Summary: {summary_path}")
    print("\n  scene         PSNR    SSIM    LPIPS")
    print("  ─────────────────────────────────────")
    psnrs = []
    for r in summary:
        if "PSNR" in r:
            psnrs.append(r["PSNR"])
            print(f"  {r['scene']:<12} {r['PSNR']:6.2f}  "
                  f"{r['SSIM']:.3f}  {r['LPIPS']:.3f}")
        else:
            print(f"  {r['scene']:<12} {r.get('error', 'no metrics')}")
    if psnrs:
        print(f"  {'mean':<12} {sum(psnrs)/len(psnrs):6.2f}")


if __name__ == "__main__":
    main()
