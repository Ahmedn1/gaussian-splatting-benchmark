# Dynamic 3DGS comparison — D-NeRF Synthetic + scene_S004 + N3V/flame_salmon_1

All numbers are PSNR (dB) on the test split.

## D-NeRF Synthetic — per-scene PSNR

| scene | Deformable | 4DGS-hustvl | 4DGS-fudan | Deformable (paper) | 4DGS-hustvl (paper) | 4DGS-fudan (paper) |
|---|---|---|---|---|---|---|
| bouncingballs | 43.30 | 40.87 | 33.39 | 41.01 | 42.16 | 42.03 |
| hellwarrior | 32.17 | 28.83 | 34.75 | 41.54 | 41.54 | 42.66 |
| hook | 35.41 | 32.86 | 32.38 | 37.42 | 35.96 | 35.75 |
| jumpingjacks | 37.21 | 35.23 | 30.75 | 37.72 | 35.99 | 39.15 |
| lego | 25.08 | 25.02 | 25.56 | 33.07 | 25.18 | 25.51 |
| mutant | 40.53 | 37.87 | 38.92 | 42.63 | 43.81 | 43.40 |
| standup | 41.00 | 38.08 | 39.40 | 44.62 | 47.48 | 46.99 |
| trex | 38.38 | 34.29 | 29.90 | 38.10 | 34.65 | 33.25 |
| **mean** | 36.64 | 34.13 | 33.13 | 39.51 | 38.35 | 38.59 |
| **mean (excl. lego + hellwarrior)** | 39.31 | 36.53 | 34.12 | 40.25 | 40.01 | 40.10 |

## D-NeRF Synthetic — SSIM / LPIPS (mean across 8 scenes)

| metric | Deformable | 4DGS-hustvl | 4DGS-fudan |
|---|---|---|---|
| SSIM | 0.985 | 0.979 | 0.968 |
| LPIPS | 0.017 | 0.026 | 0.042 |

## scene_S004 (cam04 hold-out)

| method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Deformable | 11.65 | 0.366 | 0.593 |
| 4DGS-hustvl | 11.58 | 0.464 | 0.616 |
| 4DGS-fudan | 11.97 | 0.499 | 0.616 |

## N3V / flame_salmon_1 (cam00 hold-out)

Published paper numbers (approximate): hustvl ≈ 27.96 dB, fudan ≈ 28.72 dB (from respective papers)

| method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Deformable | 24.78 | 0.881 | 0.202 |
| 4DGS-hustvl | 26.79 | 0.871 | 0.215 |
| 4DGS-fudan | 28.97 | 0.930 | 0.133 |
