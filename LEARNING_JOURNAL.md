# Neural Rendering — Learning Project

> **Goal:** Learn Neural Rendering end-to-end through a self-directed Kaggle-style project.
> **Rule #1:** No copying existing Kaggle submissions.
> **Rule #2:** Literature review & theory first, implementation second.

---

## Table of Contents

1. [What Is Neural Rendering?](#1-what-is-neural-rendering)
2. [A Brief History of 3D Scene Representation](#2-a-brief-history-of-3d-scene-representation)
3. [The Neural Rendering Family Tree](#3-the-neural-rendering-family-tree)
4. [Deep Dive — Key Methods](#4-deep-dive--key-methods)
   - 4.1 [NeRF (2020) — The Watershed Moment](#41-nerf-2020--the-watershed-moment)
   - 4.2 [Mip-NeRF & Mip-NeRF 360 (2021-22)](#42-mip-nerf--mip-nerf-360-2021-22)
   - 4.3 [Instant-NGP (2022)](#43-instant-ngp-2022)
   - 4.4 [TensoRF (2022)](#44-tensorf-2022)
   - 4.5 [3D Gaussian Splatting (2023) — Current SOTA](#45-3d-gaussian-splatting-2023--current-sota)
   - 4.6 [4D / Dynamic Scene Methods](#46-4d--dynamic-scene-methods)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Kaggle Competition Landscape](#6-kaggle-competition-landscape)
7. [Our Dataset](#7-our-dataset)
8. [Proposed Project Plan](#8-proposed-project-plan)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [References](#10-references)

---

## 1. What Is Neural Rendering?

Neural Rendering sits at the intersection of **computer graphics** and **deep learning**. The core idea is to use neural networks (or neural representations) to **model a 3D scene** and then **render novel views** of that scene from arbitrary camera positions.

Traditional graphics pipelines require explicit geometry (meshes, point clouds, voxels) + manually authored materials + a rasterizer or ray tracer. Neural rendering replaces some or all of those components with learned representations.

**The key tasks in Neural Rendering:**

| Task | Description |
|------|-------------|
| **Novel View Synthesis (NVS)** | Given N images of a scene from known poses, synthesize the scene from a new unseen viewpoint |
| **3D Reconstruction** | Recover the underlying geometry |
| **Scene Editing / Relighting** | Manipulate the scene and re-render |
| **Dynamic / Video NVS** | Handle scenes that change over time |

For this project, our primary task is **Novel View Synthesis** — the most tractable and well-studied task, and the one all major competitions and benchmarks target.

---

## 2. A Brief History of 3D Scene Representation

Understanding *why* neural rendering emerged requires knowing what came before it.

### Era 1: Classical Explicit Representations (pre-2000)

- **Meshes:** Triangles with textures. Great for real-time, terrible for capturing fine detail or translucency.
- **Point Clouds:** Raw sensor output (LiDAR, stereo). Hard to render smoothly.
- **Voxel Grids:** Discretize space into a 3D grid. Memory grows as O(N³). 512³ = 134M voxels.

### Era 2: Image-Based Rendering (IBR, 1990s–2010s)

The insight: why reconstruct 3D explicitly when you can interpolate between real photos?

- **Light Fields (Levoy & Hanrahan, 1996):** Parameterize all rays in a scene as a 4D function. Store densely sampled images. NVS = interpolation. Works beautifully but requires hundreds of images and huge storage.
- **Lumigraph (Gortler et al., 1996):** Similar to light fields but with depth-based warping.
- **DeepStereo (Flynn et al., 2016):** First deep-learning-based IBR. CNN learns to interpolate between views. Still purely image-space, no explicit 3D.

### Era 3: Deep Learning Meets 3D (2016–2019)

- **ONet / DeepSDF (2019):** Represent surfaces as implicit neural functions — a network f(x,y,z) → occupancy or signed distance. First time a neural network *is* the 3D shape.
- **Scene Representation Networks (SRN, 2019):** A neural network maps (x,y,z) → feature, then a differentiable renderer produces images. Can be trained from images alone. Limited quality.
- **Neural Volumes (Lombardi et al., 2019, Facebook Reality Labs):** MLP-decoded voxel grid with opacity and color. Produced surprisingly good head avatars. First industrial-quality neural renderer.

### Era 4: The NeRF Explosion (2020–present)

NeRF (Mildenhall et al., 2020) changed everything. See Section 4.

---

## 3. The Neural Rendering Family Tree

```
Classical IBR (1996)
    └── DeepStereo, Deep View Synthesis (2016-2019)
            └── NeRF (2020) ←─── THE WATERSHED
                    ├── Speed improvements
                    │       ├── NSVF (2020) — octree-based
                    │       ├── KiloNeRF (2021) — many tiny MLPs
                    │       ├── PlenOctrees (2021) — bake to octree
                    │       ├── TensoRF (2022) — tensor decomposition
                    │       └── Instant-NGP (2022) — hash encoding ← NVIDIA, 5s training
                    │
                    ├── Quality improvements
                    │       ├── Mip-NeRF (2021) — anti-aliasing
                    │       ├── Mip-NeRF 360 (2022) — unbounded scenes
                    │       └── Zip-NeRF (2023) — hash + mip
                    │
                    ├── Generalizable (no per-scene fitting)
                    │       ├── pixelNeRF (2021) — conditioned on images
                    │       ├── MVSNeRF (2021) — cost volume
                    │       └── ZeroNeRF, NeuRay (2022)
                    │
                    ├── Dynamic / 4D scenes
                    │       ├── D-NeRF (2021) — deformation field
                    │       ├── NeRFlow (2021)
                    │       ├── HyperNeRF (2021) — Google
                    │       ├── DynNeRF / Neural 3D Video (2022)
                    │       └── 4D Gaussian Splatting (2023-24)
                    │
                    └── Gaussian Splatting branch (2023) ← CURRENT SOTA
                            ├── 3DGS (Kerbl et al., 2023)
                            ├── Deformable 3DGS (2023)
                            ├── 4D Gaussian Splatting (2023)
                            ├── GaussianAvatar (2024)
                            └── Scaffold-GS, Mip-Splatting (2024)
```

---

## 4. Deep Dive — Key Methods

### 4.1 NeRF (2020) — The Watershed Moment

**Paper:** "Representing Scenes as Neural Radiance Fields for View Synthesis"
Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng — ECCV 2020 Best Paper

#### The Core Idea

A scene is represented as a **Neural Radiance Field** — a continuous volumetric function:

```
F_θ : (x, y, z, θ, φ) → (RGB, σ)
```

- **(x, y, z):** 3D position in the scene
- **(θ, φ):** viewing direction (elevation, azimuth)
- **RGB:** emitted color at that point, from that direction (view-dependent appearance)
- **σ:** volume density (how opaque that point is)

F_θ is implemented as a simple **8-layer MLP** (~1M parameters).

#### Positional Encoding

MLPs are biased toward low-frequency functions (spectral bias). To represent fine detail, coordinates are lifted to high-frequency features:

```
γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

L=10 for position (60-dim), L=4 for direction (24-dim). This is analogous to the Fourier features in NLP's positional encodings — you already know this concept!

#### Volume Rendering (The Differentiable Renderer)

To render a pixel, cast a ray r(t) = o + td from camera origin o in direction d. Sample N points along the ray. The expected color is:

```
C(r) = Σᵢ Tᵢ · αᵢ · cᵢ

where:
  Tᵢ = exp(-Σⱼ<ᵢ σⱼ δⱼ)   # accumulated transmittance (how much light reaches point i)
  αᵢ = 1 - exp(-σᵢ δᵢ)     # opacity of segment i
  δᵢ = tᵢ₊₁ - tᵢ           # segment length
  cᵢ = predicted RGB at point i
```

This is the **classical volume rendering equation** made differentiable. The loss is simply:

```
L = Σ_rays ||C(r) - C_gt(r)||²
```

#### Training Details

- **Hierarchical sampling:** First a "coarse" network samples uniformly. Then a "fine" network samples more densely where the coarse network found high density (near surfaces). Two networks, one loss.
- **Training time:** 1-2 days on a V100 for one scene (100 images → 400k iterations)
- **Inference:** ~30 seconds per image

#### Strengths & Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|-------------|--------------|
| Photorealistic NVS | Extremely slow training (days) |
| Handles view-dependent effects (reflections) | Slow inference (~30s/image) |
| No explicit 3D needed | One model per scene |
| Works with ~100 images | Requires known camera poses (COLMAP) |
| Compact representation | Struggles with large/unbounded scenes |

---

### 4.2 Mip-NeRF & Mip-NeRF 360 (2021-22)

**Problem with NeRF:** Rays are infinitely thin. Real cameras have pixels that subtend a *cone*, not a ray. This causes aliasing — blurry at distance, jagged up close.

**Mip-NeRF** (Barron et al., 2021): Cast **cones** instead of rays. Represent each sample not as a point but as a **Gaussian** approximating the conical frustum. The positional encoding is replaced with **Integrated Positional Encoding (IPE)** — analytically integrate γ(p) over the Gaussian. Result: 60% PSNR improvement over NeRF, equivalent to 2 bits of extra image quality.

**Mip-NeRF 360** (Barron et al., 2022): Extends to **unbounded outdoor scenes** (the hard problem — background extends to infinity). Key innovations:
1. **Scene contraction:** Nonlinearly "squish" space beyond a sphere into a bounded volume
2. **Proposal network:** Lightweight density predictor to guide sampling (replaces coarse/fine)
3. **Appearance regularization:** Prevent floaters

This is the quality gold standard for NeRF-based methods.

---

### 4.3 Instant-NGP (2022)

**Paper:** "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
Müller, Evans, Schied, Keller — NVIDIA — SIGGRAPH 2022

#### The Core Insight

The MLP bottleneck in NeRF is the positional encoding + large network. What if we cached learned features in a spatial data structure?

**Multiresolution Hash Encoding:**
- Maintain L = 16 levels of spatial hash tables, each with different resolution
- At query point x, look up the 8 surrounding voxel corners at each level, interpolate
- Hash collisions handled gracefully by the network learning to resolve them
- Total: 16 levels × T=2^19 entries × F=2 features = ~16M floats
- Tiny MLP (2 hidden layers, 64 neurons) operates on these looked-up features

#### Results

- **Training time: 5 seconds** (vs. 1-2 days for NeRF, 30x faster than any prior work)
- **Inference: real-time** (>60 FPS at 1080p with a custom CUDA kernel)
- Quality roughly matches NeRF, slightly below Mip-NeRF

This democratized NeRF. You can now train a NeRF at your desk during a coffee break.

---

### 4.4 TensoRF (2022)

**Paper:** "TensoRF: Tensorial Radiance Fields" — Chen et al., ECCV 2022

Represents the radiance field as a 4D tensor (X×Y×Z×C) decomposed using **CP decomposition** or **VM (Vector-Matrix) decomposition**:

```
F(x,y,z) = Σₙ vˣₙ(x) ⊗ Mʸᶻₙ(y,z) + Σₙ vʸₙ(y) ⊗ Mˣᶻₙ(x,z) + Σₙ vᶻₙ(z) ⊗ Mˣʸₙ(x,y)
```

Think of it like a learned 3D lookup table that's compressed via matrix factorization. Much smaller than a full voxel grid, much faster than a pure MLP.

- **Training: ~10 minutes**
- **Compact:** 4MB model
- **Quality:** Better than original NeRF

Great for understanding the design space between pure implicit (MLP) and pure explicit (voxel grid) representations.

---

### 4.5 3D Gaussian Splatting (2023) — Current SOTA

**Paper:** "3D Gaussian Splatting for Real-Time Novel View Synthesis"
Kerbl, Kopanas, Leimkühler, Drettakis — SIGGRAPH 2023

This is the most important method to understand right now. It's the dominant approach in research and industry as of 2024-2025.

#### The Paradigm Shift

NeRF is **implicit** and **continuous** — you query a function. 3DGS is **explicit** and **discrete** — you store a set of primitives (Gaussians) that you render with **splatting** (projecting 3D primitives onto the 2D image plane).

#### What Is a 3D Gaussian?

Each Gaussian primitive has:
```
- μ ∈ R³           : center position
- Σ ∈ R^(3×3)      : 3D covariance (shape/orientation of the ellipsoid)
                     decomposed as Σ = RSS^T R^T (rotation R, scale S) for optimization
- α ∈ [0,1]        : opacity
- sh_coeffs ∈ R^k  : Spherical Harmonic coefficients for view-dependent color
                     (degree 3 = 48 coefficients per channel × 3 = 144 floats for color)
```

A scene is typically represented by **1-6 million Gaussians**.

#### Rendering Pipeline (Differentiable Splatting)

1. **Project** each 3D Gaussian to a 2D Gaussian on the image plane (closed-form projection)
2. **Sort** Gaussians by depth (back-to-front, GPU radix sort)
3. **Rasterize:** For each pixel, alpha-composite the overlapping Gaussians:

```
C(pixel) = Σᵢ cᵢ αᵢ Πⱼ<ᵢ (1 - αⱼ)
```

Where cᵢ is the color from SH evaluation at the viewing direction. This is exactly the same alpha-compositing equation as NeRF's volume rendering — but done in 2D.

#### Training — Adaptive Density Control

Start from a **SfM point cloud** (COLMAP sparse reconstruction) as initial Gaussian positions. Then iterate:

1. Forward render → compare to GT image → backprop through rasterizer
2. Every 100 iterations: **Adaptive Density Control (ADC):**
   - **Clone:** Gaussians with high positional gradient but small scale → clone and perturb (underfitting a region)
   - **Split:** Gaussians with high positional gradient and large scale → split into two smaller ones (one big Gaussian covering too much)
   - **Prune:** Gaussians with α < threshold → remove (transparent, useless)

This is like neural architecture search for a geometric representation — the model decides how many primitives it needs and where.

#### Results

| Method | Train Time | FPS | PSNR (Tanks & Temples) |
|--------|-----------|-----|------------------------|
| NeRF | ~24h | 0.03 | 26.5 |
| Instant-NGP | 5min | 60+ | 27.0 |
| Mip-NeRF 360 | ~12h | 0.06 | 27.7 |
| **3DGS** | **35min** | **>100** | **27.2** |

3DGS matches Mip-NeRF 360 quality, trains in 35 minutes, and renders at **>100 FPS** (real-time!). This is why it dominates.

#### Weaknesses

- Storage: 1-2 GB per scene (millions of Gaussians)
- Requires COLMAP poses (same as NeRF)
- Artifacts in thin structures and highly specular surfaces
- Popping/sorting artifacts during camera motion
- Struggles with large dynamic scenes

---

### 4.6 4D / Dynamic Scene Methods

When the scene moves (our dataset has video!), you need to handle time as a 4th dimension.

#### Neural 3D Video Synthesis (Li et al., CVPR 2022)

- Extends NeRF to video by decomposing appearance into a static base + dynamic residuals
- Per-frame latent codes + a deformation network

#### HyperNeRF (Park et al., 2021 — Google)

- Lifts NeRF to a higher-dimensional "hyperspace"
- A slice of that hyperspace at each timestamp gives the scene's appearance
- Very elegant but slow

#### 4D Gaussian Splatting (Wu et al., 2023 / Yang et al., 2023)

Multiple independent works converged simultaneously:

**Approach 1 — Deformable 3DGS:**
- Canonical 3DGS + a deformation MLP: Δμ, ΔR, ΔS = f_θ(μ, t)
- Train the deformation field + Gaussians jointly

**Approach 2 — 4D Primitives:**
- Extend Gaussians to 4D (x,y,z,t) with a 4D covariance
- Marginalizing over time gives you the 3D state at any instant

**RealTime4DGS / Spacetime Gaussians (2024):**
- Current practical SOTA for monocular video
- Represents motion with polynomial time coefficients per Gaussian

---

## 5. Evaluation Metrics

These are the standard metrics for NVS. Know them well — they're what every paper reports.

### PSNR — Peak Signal-to-Noise Ratio
```
PSNR = 10 · log₁₀(MAX² / MSE)
```
- Measured in dB. Higher is better.
- Rule of thumb: <25 poor, 25-30 good, >30 excellent
- Simple but correlates reasonably with perceptual quality
- **Limitation:** Two images can have same PSNR but look very different perceptually

### SSIM — Structural Similarity Index
```
SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
```
Measures luminance, contrast, and structure separately. Range [0, 1], higher is better.

### LPIPS — Learned Perceptual Image Patch Similarity
- Uses deep features (VGG/AlexNet) to measure perceptual similarity
- Range [0, 1], **lower is better**
- Best correlation with human judgment
- The metric you should care most about

### For Dynamic Scenes — t-PSNR, D-SSIM
Same metrics computed per-frame and averaged over time.

---

## 6. Kaggle Competition Landscape

After researching all active and past Kaggle competitions:

### Directly Relevant Competitions

| Competition | Status | Task | Notes |
|-------------|--------|------|-------|
| **NeRF Quality Assessment** | No dedicated competition found | — | — |
| **3D Reconstruction** | No active competition | — | — |
| **Waymo Open Dataset** | Challenge (non-Kaggle) | NVS for autonomous driving | Very complex |
| **MultiNeRF / Arenas** | Academic benchmarks | NVS | Not Kaggle |

### The Reality

There is **no active Kaggle competition specifically for Neural Rendering / Novel View Synthesis** as of April 2026. This is actually common for cutting-edge CV topics — the research community runs its own benchmarks (NeRF Synthetic, Tanks and Temples, Mip-NeRF 360 dataset, DL3DV).

### Related Kaggle Competitions Worth Noting

| Competition | Relevance |
|-------------|-----------|
| Google Universal Image Embedding (2022) | 3D understanding adjacent |
| Image Matching Challenge (2022, 2023, 2024) | COLMAP / SfM — the *prerequisite* to NeRF |
| Kaggle — Google Research Football | Volumetric video adjacent |

### The Image Matching Challenge — Highly Relevant!

The **Image Matching Challenge** (IMC) on Kaggle is the *closest* active competition. It tests:
- Feature matching between images
- Relative pose estimation
- Reconstruction quality (mAA — mean Average Accuracy of camera poses)

**This is literally step 0 of every NeRF/3DGS pipeline.** Every method needs accurate camera poses (from COLMAP or similar). IMC 2024 used the exact COLMAP pipeline we'd use.

---

## 7. Our Dataset

**Dataset:** [Multi-Camera Real Scene Video Dataset](https://www.kaggle.com/datasets/maadaaai/mulit-camera-real-scene-video-dataset)

### What We Know

- Multi-camera setup (synchronized cameras from multiple viewpoints)
- Real scenes (not synthetic)
- Video data (temporal dimension)
- This is actually a **perfect setup for 3DGS/4DGS** because:
  - Multiple synchronized cameras = known relative geometry (easier pose estimation)
  - Video = we can study both static (single frame) and dynamic (full video) NVS

### What We Need to Investigate (after download)
- Number of cameras
- Frame rate and resolution
- Scene types (indoor/outdoor, static/dynamic)
- Whether camera calibration is provided or must be computed

### Why This Dataset Is Actually Better Than a Competition Dataset

A competition dataset would give us fixed train/test splits and a leaderboard. Our dataset forces us to:
1. Design our own experimental protocol
2. Justify our train/test split
3. Choose our own metrics
4. Compare against baseline reproductions

This is **more realistic** to actual ML engineering work.

---

## 8. Proposed Project Plan

### The Learning Arc: Walk Before You Run

```
Phase 1: Foundations        → Understand the pipeline, run existing code
Phase 2: Static Scenes      → 3DGS on single frames (one scene at a time)
Phase 3: Quality Analysis   → Systematic evaluation, ablations
Phase 4: Dynamic Scenes     → 4DGS / Deformable 3DGS on video
Phase 5: (Optional) Custom  → Modify the pipeline, add your own contribution
```

### Why This Order?

- **Phase 1** avoids the biggest beginner mistake: jumping to dynamic scenes without understanding the static foundation. The additional complexity of dynamics is multiplicative, not additive.
- **Phase 2** on static builds intuition for what Gaussians look like, how ADC works, what hyperparameters matter.
- **Phase 3** teaches you to be a rigorous experimentalist — critical for publishing or competing.
- **Phase 4** is where the dataset's video nature pays off.

### Our Specific Task Formulation

> **Given:** N synchronized video streams from N cameras (known relative poses or to-be-estimated)
>
> **Task A (Static):** Use frame 0 from all cameras to train a 3DGS model. Hold out 20% of cameras for evaluation. Report PSNR, SSIM, LPIPS.
>
> **Task B (Dynamic):** Use all frames from all cameras to train a 4DGS model. Hold out 20% of cameras AND 20% of timesteps. Report per-frame metrics.
>
> **Baseline:** Nearest-neighbor view interpolation (classical IBR baseline — always have a non-ML baseline)

---

## 9. Implementation Roadmap

### Step 0: Environment Setup

```bash
# Core dependencies
conda create -n neural-rendering python=3.10
conda activate neural-rendering

# For 3D Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# For camera pose estimation (prerequisite)
pip install pycolmap
# or use COLMAP binary

# For evaluation
pip install lpips torchmetrics

# For data exploration
pip install kaggle opencv-python imageio matplotlib
```

### Step 1: Data Acquisition & Exploration (Week 1)

```bash
kaggle datasets download -d maadaaai/mulit-camera-real-scene-video-dataset
```

- Inspect frame count, resolution, camera count
- Check if calibration files exist (intrinsics K, extrinsics R,t)
- Visualize camera poses in 3D (Open3D or matplotlib)
- Extract a representative static frame set

### Step 2: Camera Pose Estimation with COLMAP (Week 1)

If poses are not provided:
1. Extract frames at 1 FPS
2. Run COLMAP SfM (Structure from Motion):
   - Feature extraction (SIFT or SuperPoint)
   - Feature matching (exhaustive or sequential)
   - Sparse reconstruction → camera poses + sparse point cloud
3. Convert to 3DGS format (provided scripts in the repo)

**Why COLMAP matters:** If COLMAP fails or gives bad poses, *nothing downstream works*. This is often the hardest practical step.

### Step 3: Static 3DGS Baseline (Week 2)

```python
# Training a 3DGS model (using official repo)
python train.py \
    -s data/scene_001 \        # path to COLMAP output
    -m output/scene_001 \      # where to save
    --iterations 30000 \
    --eval                     # hold out test cameras

# Render test views
python render.py -m output/scene_001

# Evaluate
python metrics.py -m output/scene_001
```

Key things to observe:
- Point cloud initialization quality
- Loss curve (should drop steeply then plateau)
- Where ADC clones/splits/prunes happen
- Which views fail (look for high-error views)

### Step 4: Ablation Studies (Week 2-3)

| Experiment | What You Learn |
|------------|---------------|
| Vary # training images (20/50/100%) | How many views does 3DGS need? |
| Vary # iterations (5k/15k/30k) | Convergence behavior |
| No ADC vs. ADC | How much does adaptive density help? |
| SH degree 0 vs. 3 | How much does view-dependent color matter? |
| Different scenes | How scene type affects quality |

### Step 5: Dynamic 3DGS (Week 3-4)

Recommended repo: **4D-Gaussian** or **Deformable-3D-Gaussians**

```bash
git clone https://github.com/hustvl/4d-gaussian-splatting
git clone https://github.com/ingra14m/Deformable-3D-Gaussians
```

The deformable approach is more interpretable for learning:
- The canonical Gaussians are easy to visualize
- The deformation field can be probed

### Step 6: Comparison & Write-Up (Week 4)

Produce a clean comparison table:

| Method | Train Time | PSNR | SSIM | LPIPS | Notes |
|--------|-----------|------|------|-------|-------|
| Nearest-neighbor IBR baseline | — | ? | ? | ? | |
| 3DGS (static, frame 0) | ? | ? | ? | ? | |
| Deformable 3DGS (video) | ? | ? | ? | ? | |
| 4DGS (video) | ? | ? | ? | ? | |

---

## 10. References

### Foundational Papers (read in this order)

1. **NeRF:** Mildenhall et al. (2020). "Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020. https://arxiv.org/abs/2003.08934

2. **Mip-NeRF:** Barron et al. (2021). "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." ICCV 2021. https://arxiv.org/abs/2103.13415

3. **Mip-NeRF 360:** Barron et al. (2022). CVPR 2022. https://arxiv.org/abs/2111.12077

4. **Instant-NGP:** Müller et al. (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." SIGGRAPH 2022. https://arxiv.org/abs/2201.05989

5. **TensoRF:** Chen et al. (2022). ECCV 2022. https://arxiv.org/abs/2203.09517

6. **3D Gaussian Splatting:** Kerbl et al. (2023). "3D Gaussian Splatting for Real-Time Novel View Synthesis." SIGGRAPH 2023. https://arxiv.org/abs/2308.04079

### Dynamic / 4D

7. **D-NeRF:** Pumarola et al. (2021). https://arxiv.org/abs/2011.13961
8. **HyperNeRF:** Park et al. (2021). https://arxiv.org/abs/2106.13228
9. **Neural 3D Video:** Li et al. (2022). CVPR 2022. https://arxiv.org/abs/2103.02597
10. **4D Gaussian Splatting:** Wu et al. (2023). https://arxiv.org/abs/2310.08528
11. **Deformable 3DGS:** Yang et al. (2023). https://arxiv.org/abs/2309.13101

### Survey Papers (great for getting the big picture fast)

12. **State of the Art in Neural Rendering:** Tewari et al. (2020). https://arxiv.org/abs/2004.03805
13. **NeRF in the Wild:** Martin-Brualla et al. (2021). CVPR 2021.

### Official Code Repositories

- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Instant-NGP: https://github.com/NVlabs/instant-ngp
- Nerfstudio (unified framework — START HERE): https://github.com/nerfstudio-project/nerfstudio
- 4D Gaussian Splatting: https://github.com/hustvl/4d-gaussian-splatting
- Deformable 3DGS: https://github.com/ingra14m/Deformable-3D-Gaussians

### Benchmarks & Datasets

- **Blender Synthetic (NeRF Synthetic):** 8 object scenes, perfect GT poses
- **Tanks and Temples:** Real outdoor scenes
- **Mip-NeRF 360 dataset:** Unbounded indoor/outdoor scenes
- **DL3DV-10K:** 10,000 real scenes, 2024 benchmark
- **Our dataset:** Multi-camera real scene video

---

*Document maintained as part of the Neural Rendering Learning Project.*
*Last updated: 2026-04-23*
