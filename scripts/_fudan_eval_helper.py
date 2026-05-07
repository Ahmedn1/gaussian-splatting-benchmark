"""
_fudan_eval_helper.py — runs INSIDE the nr-4dgs-fudan env.
Loads a fudan-zvg trained checkpoint, renders test cameras, computes
PSNR / SSIM / LPIPS, writes results.json.

Invoked by scripts/16_benchmark_4dgs_fudan.py:
    python _fudan_eval_helper.py --config <path> --checkpoint <pth> --out <json>

Imports fudan modules — must run from inside repos/4d-gaussian-splatting
with that repo on PYTHONPATH (the parent driver sets PYTHONPATH).
"""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import lpips as lpips_pkg
from omegaconf import OmegaConf, DictConfig

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_args_from_config(config_path):
    """Mimic train.py's arg construction so we can build dataset/pipe/opt."""
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args([])

    cfg = OmegaConf.load(config_path)
    def merge(key, host):
        if isinstance(host[key], DictConfig):
            for k1 in host[key].keys():
                merge(k1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        merge(k, cfg)

    return args, lp, op, pp


@torch.no_grad()
def main():
    a = parse_args()
    args, lp, op, pp = build_args_from_config(a.config)

    dataset = lp.extract(args)
    pipe    = pp.extract(args)
    opt     = op.extract(args)

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
        prefilter_var=dataset.prefilter_var,
    )
    scene = Scene(dataset, gaussians,
                  num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration)
    gaussians.training_setup(opt)

    model_params, _ = torch.load(a.checkpoint)
    gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cams = scene.getTestCameras()
    if not test_cams:
        raise RuntimeError("No test cameras — was --eval set during train?")

    lpips_model = lpips_pkg.LPIPS(net="vgg").cuda().eval()

    psnrs, ssims, lpipss = [], [], []
    for batch in test_cams:
        # In fudan, getTestCameras returns either Camera objects or
        # (gt_image, viewpoint) tuples depending on dataloader setting.
        if isinstance(batch, (list, tuple)):
            gt, viewpoint = batch
            gt = gt.cuda()
            viewpoint = viewpoint.cuda()
        else:
            viewpoint = batch
            gt = viewpoint.original_image.cuda() if hasattr(viewpoint, "original_image") \
                  else viewpoint.image.cuda()
        out = render(viewpoint, gaussians, pipe, background)
        img = torch.clamp(out["render"], 0.0, 1.0)

        psnrs.append(psnr(img, gt).mean().item())
        ssims.append(ssim(img, gt).mean().item())
        # lpips wants [-1,1]
        lpipss.append(lpips_model(img.unsqueeze(0) * 2 - 1,
                                  gt.unsqueeze(0)  * 2 - 1).mean().item())

    out = {"PSNR":  sum(psnrs)  / len(psnrs),
           "SSIM":  sum(ssims)  / len(ssims),
           "LPIPS": sum(lpipss) / len(lpipss),
           "n_test_cams": len(test_cams)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {a.out}: PSNR={out['PSNR']:.2f}  "
          f"SSIM={out['SSIM']:.3f}  LPIPS={out['LPIPS']:.3f}")


if __name__ == "__main__":
    main()
