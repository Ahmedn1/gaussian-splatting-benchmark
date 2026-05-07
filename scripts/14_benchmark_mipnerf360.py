"""
14_benchmark_mipnerf360.py — Benchmark vanilla 3DGS on Mip-NeRF 360
====================================================================

Runs the reference 3DGS pipeline (train → render → metrics) on each of the
7 Mip-NeRF 360 v2 scenes and writes results.json per scene. This stresses
the static pipeline on REAL captures (Stage 1 was clean Blender synthetic).

Per the 3DGS paper recipe:
    outdoor scenes (bicycle, garden, stump): train at images_4 (4x down)
    indoor scenes (bonsai, counter, kitchen, room): train at images_2 (2x down)

Data layout expected (after unpacking 360_v2.zip):
    data/benchmarks/mipnerf360/
        bicycle/{images,images_2,images_4,images_8,sparse/0,poses_bounds.npy}
        bonsai/...
        ...

Outputs:
    experiments/benchmarks/mipnerf360/{scene}/
        point_cloud/iteration_30000/point_cloud.ply
        test/ours_30000/{renders,gt}/
        results.json

Usage:
    python scripts/14_benchmark_mipnerf360.py
    python scripts/14_benchmark_mipnerf360.py --scenes garden
    python scripts/14_benchmark_mipnerf360.py --scenes garden --iterations 7000
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
DATA_DIR     = PROJECT_ROOT / "data" / "benchmarks" / "mipnerf360"
EXP_DIR      = PROJECT_ROOT / "experiments" / "benchmarks" / "mipnerf360"
GS_DIR       = PROJECT_ROOT / "repos" / "gaussian-splatting"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

# Per-scene image folder per the 3DGS paper recipe.
# Outdoor → images_4, indoor → images_2.
SCENE_RESOLUTION = {
    "bicycle":  "images_4",
    "garden":   "images_4",
    "stump":    "images_4",
    "bonsai":   "images_2",
    "counter":  "images_2",
    "kitchen":  "images_2",
    "room":     "images_2",
}
ALL_SCENES = list(SCENE_RESOLUTION.keys())


def build_env() -> dict:
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
    spec = importlib.util.spec_from_file_location(
        "_s05", SCRIPTS_DIR / "05_train_static_3dgs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_static_rasterizer()


def run(cmd: list[str], cwd: Path, env: dict, label: str):
    print(f"\n  [{label}] {' '.join(str(c) for c in cmd[-8:])}")
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

    images_folder = SCENE_RESOLUTION[scene]
    if not (src / images_folder).exists():
        raise FileNotFoundError(f"{src / images_folder} missing — extract zip?")

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
             "--images",          images_folder,
             "--iterations",      iterations,
             "--test_iterations", iterations,
             "--save_iterations", iterations,
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

    if results.exists():
        with open(results) as f:
            data = json.load(f)
        iter_key = f"ours_{iterations}"
        m = data.get(iter_key) or next(iter(data.values()))
        print(f"  📊 {scene}: PSNR={m['PSNR']:.2f}  "
              f"SSIM={m['SSIM']:.3f}  LPIPS={m['LPIPS']:.3f}")
        return {"scene": scene, "iterations": iterations,
                "resolution": images_folder, "train_minutes": train_dt / 60, **m}
    return {"scene": scene, "iterations": iterations,
            "resolution": images_folder, "train_minutes": train_dt / 60}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES)
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"\nBenchmark: Mip-NeRF 360 ({len(args.scenes)} scenes, "
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

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EXP_DIR / f"summary_{args.iterations}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Total wallclock: {(time.time() - t_start) / 60:.1f} min")
    print(f"Summary: {summary_path}")
    print("\n  scene       PSNR    SSIM    LPIPS")
    print("  ─────────────────────────────────")
    psnrs = []
    for r in summary:
        if "PSNR" in r:
            psnrs.append(r["PSNR"])
            print(f"  {r['scene']:<10} {r['PSNR']:6.2f}  "
                  f"{r['SSIM']:.3f}  {r['LPIPS']:.3f}")
        else:
            print(f"  {r['scene']:<10} {r.get('error', 'no metrics')}")
    if psnrs:
        print(f"  {'mean':<10} {sum(psnrs)/len(psnrs):6.2f}")


if __name__ == "__main__":
    main()
