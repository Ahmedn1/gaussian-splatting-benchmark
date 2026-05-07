"""
08_run_ablations.py — orchestrate static-3DGS ablation runs
============================================================

Calls 05_train_static_3dgs.py + 06_evaluate.py for every config in a sweep,
collects the resulting metrics into one table, and prints/saves a comparison.

DEFAULT SWEEP
-------------
| name       | iterations | sh_degree | densify | What we learn               |
|------------|-----------:|----------:|:-------:|------------------------------|
| short      |       5000 |         3 |   on    | Convergence speed early      |
| baseline   |      30000 |         3 |   on    | Reference run                |
| sh0        |      30000 |         0 |   on    | View-dependent color matters |
| sh1        |      30000 |         1 |   on    | …how much does SH order help |
| no_densify |      30000 |         3 |  off    | Cost of fixed-point cloud    |

Each run lives in `experiments/static/{scene}/{name}/`.

WHY THIS LIST
-------------
With only 3 training images, this is more of a "demonstration of which
hyperparameters do anything visible" than a proper ablation. With 100+ views
you'd vary view-count, densification thresholds, learning rate schedules. Here
we focus on knobs whose effect should be visible at this data scale.

Usage:
    python scripts/08_run_ablations.py --scene scene_S004
    python scripts/08_run_ablations.py --scene scene_S004 --only baseline,sh0
    python scripts/08_run_ablations.py --scene scene_S004 --resume   # skip already-trained runs
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXP_DIR      = PROJECT_ROOT / "experiments" / "static"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"


# ── Sweep definition ──────────────────────────────────────────────────────────

SWEEP = [
    # (name,         iterations, sh_degree, extra train args)
    ("short",        5_000,  3, []),
    ("baseline",     30_000, 3, []),
    ("sh0",          30_000, 0, []),
    ("sh1",          30_000, 1, []),
    # Densification is on by default; turn it off via gaussian-splatting's flag.
    # `--densify_until_iter 0` disables clone/split for the entire run.
    ("no_densify",   30_000, 3, ["--densify_until_iter", "0"]),
]


def run_step(cmd, label):
    print(f"\n{'─' * 60}\n  ▶ {label}\n  $ {' '.join(map(str, cmd))}\n{'─' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}) after {dt:.0f}s")
    print(f"  ✓ {label} done in {dt/60:.1f} min")
    return dt


def metrics_path(scene, name, iteration):
    return EXP_DIR / scene / name / "eval" / f"iteration_{iteration}" / "metrics.json"


def load_metrics(scene, name, iteration):
    p = metrics_path(scene, name, iteration)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def format_row(name, m):
    if m is None:
        return f"{name:<14}  —"
    n_g = m.get("n_gaussians", 0)
    tr = m["train"]; te = m["test"]
    return (f"{name:<14}  "
            f"n={n_g:>7,}  "
            f"train: PSNR={tr['psnr']:5.2f}/SSIM={tr['ssim']:.3f}/LPIPS={tr['lpips']:.3f}    "
            f"test: PSNR={te['psnr']:5.2f}/SSIM={te['ssim']:.3f}/LPIPS={te['lpips']:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str, default="scene_S004")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of run names")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose metrics.json already exists")
    parser.add_argument("--skip_train", action="store_true",
                        help="Don't train; only re-evaluate + aggregate")
    args = parser.parse_args()

    selected = SWEEP
    if args.only:
        wanted = set(args.only.split(","))
        selected = [s for s in SWEEP if s[0] in wanted]
        if not selected:
            raise ValueError(f"--only matched nothing. Available: {[s[0] for s in SWEEP]}")

    print(f"Running ablations on {args.scene}: {[s[0] for s in selected]}")

    timings = {}

    for name, iters, sh, extra in selected:
        print(f"\n{'═' * 60}\n  Configuration: {name}\n{'═' * 60}")

        if args.resume and load_metrics(args.scene, name, iters) is not None:
            print(f"  ⏩ resume: {name} already has metrics, skipping.")
            continue

        # ── 1. Train (subprocess so a crash in one config doesn't kill the rest) ──
        if not args.skip_train:
            train_cmd = [
                sys.executable, str(SCRIPTS_DIR / "05_train_static_3dgs.py"),
                "--scene",      args.scene,
                "--experiment", name,
                "--iterations", str(iters),
                "--sh_degree",  str(sh),
            ]
            # First run does the rasterizer reinstall + sparse rebuild;
            # later runs skip those (--skip_rebuild) for speed.
            if name != selected[0][0]:
                train_cmd.append("--skip_rebuild")
            if extra:
                # The train script forwards `extra_args` only via its own API,
                # so we splice them after `--`.
                train_cmd.extend(extra)
            try:
                t = run_step(train_cmd, f"train [{name}]")
                timings[name] = t
            except RuntimeError as e:
                print(f"  ✗ {name} training failed: {e}")
                continue
        else:
            print(f"  ⏩ skip_train: not training {name}")

        # ── 2. Evaluate ──
        eval_cmd = [
            sys.executable, str(SCRIPTS_DIR / "06_evaluate.py"),
            "--scene",      args.scene,
            "--experiment", name,
            "--sh_degree",  str(sh),
        ]
        try:
            run_step(eval_cmd, f"eval  [{name}]")
        except RuntimeError as e:
            print(f"  ✗ {name} eval failed: {e}")
            continue

    # ── 3. Aggregate ──
    print("\n" + "═" * 80)
    print("ABLATION SUMMARY")
    print("═" * 80)

    rows = []
    for name, iters, sh, extra in selected:
        m = load_metrics(args.scene, name, iters)
        rows.append((name, sh, iters, extra, m))
        print(format_row(name, m))

    # Save a tidy comparison file
    summary = {
        "scene": args.scene,
        "runs": [
            {
                "name": r[0], "sh_degree": r[1], "iterations": r[2],
                "extra_args": r[3],
                "metrics": r[4],
                "train_time_min": timings.get(r[0]),
            }
            for r in rows
        ],
    }
    out_path = EXP_DIR / args.scene / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
