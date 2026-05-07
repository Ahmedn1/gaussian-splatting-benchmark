## Comparison — scene_S004

| Method | Train Time | #Gaussians | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| ibr_nearest_neighbor (baseline) | — | — | 11.36 | 0.2096 | 0.6785 |
| 3dgs_static [baseline] | 11.8 min | 301,513 | 12.17 | 0.4559 | 0.6453 |
| 3dgs_static [no_densify] | 5.5 min | 50,000 | 10.85 | 0.5450 | 0.7846 |
| 3dgs_static [sh0] | 10.0 min | 303,159 | 11.46 | 0.3921 | 0.6789 |
| 3dgs_static [sh1] | 10.5 min | 301,668 | 12.55 | 0.4417 | 0.6519 |
| 3dgs_static [short] | 1.4 min | 269,897 | 11.73 | 0.4593 | 0.6516 |
| deformable_3dgs [baseline] | 166.1 min | — | 11.65 | 0.3663 | 0.5934 |
