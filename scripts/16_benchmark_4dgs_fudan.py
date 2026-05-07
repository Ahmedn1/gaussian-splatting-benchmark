"""
16_benchmark_4dgs_fudan.py — Benchmark fudan-zvg/4d-gaussian-splatting
======================================================================

Runs fudan-zvg/4d-gaussian-splatting (Yang et al., ICLR 2024) on the 8
D-NeRF scenes for apples-to-apples comparison with our Stage 2 Deformable
and Stage-4-hustvl results.

The fudan repo doesn't ship a separate render.py / metrics.py — its train.py
runs the test-set eval at every `--test_iterations` step and prints PSNR
to stdout. We use a small post-train eval helper that imports the repo's
modules to compute SSIM/LPIPS/PSNR and write `results.json` per scene.

Usage:
    python scripts/16_benchmark_4dgs_fudan.py
    python scripts/16_benchmark_4dgs_fudan.py --scenes bouncingballs
    python scripts/16_benchmark_4dgs_fudan.py --scenes bouncingballs --iterations 7000
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "benchmarks" / "dnerf_synthetic"
EXP_DIR      = PROJECT_ROOT / "experiments" / "benchmarks" / "dnerf_synthetic_4dgs_fudan"
FUDAN_DIR    = PROJECT_ROOT / "repos" / "4d-gaussian-splatting"
FUDAN_PY     = Path.home() / "miniconda3" / "envs" / "nr-4dgs-fudan" / "bin" / "python"
EVAL_HELPER  = PROJECT_ROOT / "scripts" / "_fudan_eval_helper.py"

ALL_SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
              "lego", "mutant", "standup", "trex"]


def build_env() -> dict:
    cuda_home = str(Path.home() / "cuda-home")
    return {
        **os.environ,
        "CUDA_HOME":  cuda_home,
        "CC":         "/usr/bin/gcc-12",
        "CXX":        "/usr/bin/g++-12",
        "PYTHONPATH": str(FUDAN_DIR) + ":" + os.environ.get("PYTHONPATH", ""),
        "PATH":       f"{cuda_home}/bin:" + os.environ.get("PATH", ""),
        "TORCH_CUDA_ARCH_LIST": "8.6",
    }


def run(cmd, cwd, env, label):
    print(f"\n  [{label}] {' '.join(str(c) for c in cmd[-6:])}")
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {r.returncode})")
    print(f"  [{label}] done in {dt/60:.1f} min")
    return dt


def make_scene_config(scene: str, src: Path, out: Path, iterations: int) -> Path:
    """Read upstream scene yaml; override paths + iterations; write to out/cfg.yaml."""
    base_cfg = FUDAN_DIR / "configs" / "dnerf" / f"{scene}.yaml"
    cfg = yaml.safe_load(base_cfg.read_text())
    cfg["ModelParams"]["source_path"] = str(src)
    cfg["ModelParams"]["model_path"] = str(out)
    cfg["OptimizationParams"]["iterations"] = iterations
    target = out / "cfg.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg))
    return target


def benchmark_scene(scene: str, iterations: int, force: bool):
    src = DATA_DIR / scene
    if not src.exists():
        raise FileNotFoundError(f"Scene data not found: {src}")
    out = EXP_DIR / scene
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # fudan saves checkpoints as chkpnt{iter}.pth
    chkpnt = out / f"chkpnt{iterations}.pth"
    results = out / "results.json"
    env = build_env()

    cfg_path = make_scene_config(scene, src, out, iterations)

    # ── Train ──
    if chkpnt.exists() and not force:
        print(f"  ✓ {scene}: trained checkpoint exists, skipping train")
        train_dt = 0.0
    else:
        train_dt = run(
            [FUDAN_PY, FUDAN_DIR / "train.py",
             "--config",         cfg_path,
             "--save_iterations", iterations,
             "--test_iterations", iterations,
             "--quiet"],
            cwd=FUDAN_DIR, env=env, label=f"train/{scene}",
        )

    # ── Metrics (custom helper, since fudan has no separate render.py / metrics.py) ──
    if results.exists() and not force:
        print(f"  ✓ {scene}: results.json exists, skipping metrics")
    else:
        run(
            [FUDAN_PY, EVAL_HELPER,
             "--config",     cfg_path,
             "--checkpoint", chkpnt,
             "--out",        results],
            cwd=FUDAN_DIR, env=env, label=f"eval/{scene}",
        )

    if results.exists():
        m = json.loads(results.read_text())
        print(f"  📊 {scene}: PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.3f}  "
              f"LPIPS={m['LPIPS']:.3f}")
        return {"scene": scene, "iterations": iterations,
                "train_minutes": train_dt / 60, **m}
    return {"scene": scene, "iterations": iterations,
            "train_minutes": train_dt / 60}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES)
    p.add_argument("--iterations", type=int, default=20_000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not FUDAN_PY.exists():
        sys.exit(f"fudan env python not found: {FUDAN_PY}")
    if not EVAL_HELPER.exists():
        sys.exit(f"eval helper not found: {EVAL_HELPER}")

    print(f"\nBenchmark: fudan-4DGaussians on D-NeRF Synthetic "
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
            print(f"  {r['scene']:<14} {r['PSNR']:6.2f}  "
                  f"{r['SSIM']:.3f}  {r['LPIPS']:.3f}")
        else:
            print(f"  {r['scene']:<14} {r.get('error', 'no metrics')}")
    if psnrs:
        print(f"  {'mean':<14} {sum(psnrs)/len(psnrs):6.2f}")


if __name__ == "__main__":
    main()
