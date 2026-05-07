"""
17_compare_dynamic.py — 3-way comparison: Deformable / hustvl-4DGS / fudan-4DGS
==============================================================================

Sources:
    Deformable      experiments/benchmarks/dnerf_synthetic/{scene}/results.json
    hustvl-4DGS     experiments/benchmarks/dnerf_synthetic_4dgs_hustvl/{scene}/results.json
    fudan-4DGS      experiments/benchmarks/dnerf_synthetic_4dgs_fudan/{scene}/results.json
    + scene_S004    experiments/dynamic/scene_S004/{deformable,4dgs_hustvl,4dgs_fudan}/...
    + flame_salmon  experiments/dynamic/n3v/flame_salmon_1_{deformable,hustvl,fudan}/...
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BENCH = PROJECT_ROOT / "experiments" / "benchmarks"
S004  = PROJECT_ROOT / "experiments" / "dynamic" / "scene_S004"
N3V   = PROJECT_ROOT / "experiments" / "dynamic" / "n3v"
OUT   = BENCH / "dynamic_comparison.md"

DNERF_SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
                "lego", "mutant", "standup", "trex"]

METHODS = [
    ("Deformable",  BENCH / "dnerf_synthetic",                  "deformable"),
    ("4DGS-hustvl", BENCH / "dnerf_synthetic_4dgs_hustvl",       "4dgs_hustvl"),
    ("4DGS-fudan",  BENCH / "dnerf_synthetic_4dgs_fudan",        "4dgs_fudan"),
]

# Published numbers — paper Table 1 (Yang/Deformable) and 4DGS papers.
# Used as reference column in the table.
PAPER = {
    "Deformable": {  # Yang et al. supplemental Table 3
        "bouncingballs": 41.01, "hellwarrior": 41.54, "hook": 37.42,
        "jumpingjacks": 37.72, "lego": 33.07, "mutant": 42.63,
        "standup": 44.62, "trex": 38.10,
    },
    "4DGS-hustvl": {  # Wu et al. CVPR 2024 Table 5 (approximate)
        "bouncingballs": 42.16, "hellwarrior": 41.54, "hook": 35.96,
        "jumpingjacks": 35.99, "lego": 25.18, "mutant": 43.81,
        "standup": 47.48, "trex": 34.65,
    },
    "4DGS-fudan": {  # Yang et al. ICLR 2024 Table 1 (approximate)
        "bouncingballs": 42.03, "hellwarrior": 42.66, "hook": 35.75,
        "jumpingjacks": 39.15, "lego": 25.51, "mutant": 43.40,
        "standup": 46.99, "trex": 33.25,
    },
}


def load_metrics(results_path: Path) -> dict | None:
    if not results_path.exists():
        return None
    data = json.loads(results_path.read_text())
    # Different repos key results.json differently — pick the first leaf
    # dict that has PSNR.
    def find_metrics(obj):
        if isinstance(obj, dict):
            if "PSNR" in obj:
                return obj
            for v in obj.values():
                m = find_metrics(v)
                if m is not None:
                    return m
        return None
    return find_metrics(data)


def fmt(x, prec=2):
    return "—" if x is None else f"{x:.{prec}f}"


def collect_dnerf():
    rows = {}  # scene -> {method_label: {PSNR, SSIM, LPIPS}}
    for scene in DNERF_SCENES:
        rows[scene] = {}
        for label, base, _ in METHODS:
            m = load_metrics(base / scene / "results.json")
            if m is not None:
                rows[scene][label] = {
                    "PSNR":  m.get("PSNR"),
                    "SSIM":  m.get("SSIM"),
                    # Both LPIPS-vgg and LPIPS keys exist depending on repo
                    "LPIPS": m.get("LPIPS-vgg", m.get("LPIPS")),
                }
    return rows


def collect_s004():
    """Each method's scene_S004 result lives under a different layout —
    walk gracefully and fall back to None."""
    out = {}
    candidates = [
        ("Deformable",  S004 / "baseline" / "results.json"),
        ("Deformable",  S004 / "deformable" / "results.json"),
        ("4DGS-hustvl", S004 / "4dgs_hustvl" / "results.json"),
        ("4DGS-fudan",  S004 / "4dgs_fudan" / "results.json"),
    ]
    for label, p in candidates:
        if p.exists() and label not in out:
            m = load_metrics(p)
            if m is not None:
                out[label] = {
                    "PSNR":  m.get("PSNR"),
                    "SSIM":  m.get("SSIM"),
                    "LPIPS": m.get("LPIPS-vgg", m.get("LPIPS")),
                }
    return out


def collect_n3v(scene="flame_salmon_1"):
    """Collect N3V/flame_salmon results for each method."""
    out = {}
    candidates = [
        ("Deformable",  N3V / f"{scene}_deformable" / "results.json"),
        ("4DGS-hustvl", N3V / f"{scene}_hustvl" / "results.json"),
        ("4DGS-fudan",  N3V / f"{scene}_fudan" / "results.json"),
    ]
    for label, p in candidates:
        if p.exists():
            m = load_metrics(p)
            if m is not None:
                out[label] = {
                    "PSNR":  m.get("PSNR"),
                    "SSIM":  m.get("SSIM"),
                    "LPIPS": m.get("LPIPS-vgg", m.get("LPIPS")),
                }
    return out


def render_md(dnerf, s004, n3v) -> str:
    lines = []
    lines.append("# Dynamic 3DGS comparison — D-NeRF Synthetic + scene_S004 + N3V/flame_salmon_1\n")
    lines.append("All numbers are PSNR (dB) on the test split.\n")
    lines.append("## D-NeRF Synthetic — per-scene PSNR\n")

    cols = [m[0] for m in METHODS]
    head = ["scene"] + cols + [f"{c} (paper)" for c in cols]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")

    means_ours = {c: [] for c in cols}
    means_paper = {c: [] for c in cols}
    for scene in DNERF_SCENES:
        row = [scene]
        for c in cols:
            v = dnerf[scene].get(c, {}).get("PSNR")
            row.append(fmt(v))
            if v is not None:
                means_ours[c].append(v)
        for c in cols:
            v = PAPER.get(c, {}).get(scene)
            row.append(fmt(v))
            if v is not None:
                means_paper[c].append(v)
        lines.append("| " + " | ".join(row) + " |")

    # Mean row
    mrow = ["**mean**"]
    for c in cols:
        mrow.append(fmt(sum(means_ours[c]) / len(means_ours[c]) if means_ours[c] else None))
    for c in cols:
        mrow.append(fmt(sum(means_paper[c]) / len(means_paper[c]) if means_paper[c] else None))
    lines.append("| " + " | ".join(mrow) + " |")

    # Mean excluding lego + hellwarrior (known dataset issues)
    excl = {c: [v for s, v in zip(DNERF_SCENES, [dnerf[s].get(c, {}).get("PSNR") for s in DNERF_SCENES])
                if s not in {"lego", "hellwarrior"} and v is not None] for c in cols}
    excl_p = {c: [v for s in DNERF_SCENES if s not in {"lego", "hellwarrior"}
                  for v in [PAPER.get(c, {}).get(s)] if v is not None] for c in cols}
    erow = ["**mean (excl. lego + hellwarrior)**"]
    for c in cols:
        erow.append(fmt(sum(excl[c]) / len(excl[c]) if excl[c] else None))
    for c in cols:
        erow.append(fmt(sum(excl_p[c]) / len(excl_p[c]) if excl_p[c] else None))
    lines.append("| " + " | ".join(erow) + " |")

    # SSIM/LPIPS table
    lines.append("\n## D-NeRF Synthetic — SSIM / LPIPS (mean across 8 scenes)\n")
    lines.append("| metric | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for metric in ["SSIM", "LPIPS"]:
        vals = []
        for c in cols:
            xs = [dnerf[s].get(c, {}).get(metric) for s in DNERF_SCENES]
            xs = [x for x in xs if x is not None]
            vals.append(fmt(sum(xs) / len(xs) if xs else None, 3))
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    # scene_S004
    lines.append("\n## scene_S004 (cam04 hold-out)\n")
    lines.append("| method | PSNR | SSIM | LPIPS |")
    lines.append("|---|---|---|---|")
    for c in cols:
        m = s004.get(c, {})
        lines.append(f"| {c} | {fmt(m.get('PSNR'))} | {fmt(m.get('SSIM'),3)} | {fmt(m.get('LPIPS'),3)} |")

    # N3V flame_salmon_1
    lines.append("\n## N3V / flame_salmon_1 (cam00 hold-out)\n")
    lines.append("Published paper numbers (approximate): "
                 "hustvl ≈ 27.96 dB, fudan ≈ 28.72 dB (from respective papers)\n")
    lines.append("| method | PSNR | SSIM | LPIPS |")
    lines.append("|---|---|---|---|")
    for c in cols:
        m = n3v.get(c, {})
        lines.append(f"| {c} | {fmt(m.get('PSNR'))} | {fmt(m.get('SSIM'),3)} | {fmt(m.get('LPIPS'),3)} |")

    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    BENCH.mkdir(parents=True, exist_ok=True)
    dnerf = collect_dnerf()
    s004  = collect_s004()
    n3v   = collect_n3v()
    md = render_md(dnerf, s004, n3v)
    Path(args.out).write_text(md)
    print(f"Wrote {args.out} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
