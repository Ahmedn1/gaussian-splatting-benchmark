#!/usr/bin/env bash
# =============================================================================
# Neural Rendering Environment Activation Script
#
# Usage:  source scripts/activate_env.sh
#
# What this does:
#   1. Activates the 'neural-rendering' conda environment
#   2. Sets CUDA_HOME to our custom symlink tree (nvcc 12.0 split across /usr)
#   3. Sets CC/CXX to gcc-12 (nvcc 12.0 max supported host compiler)
#   4. Adds CUDA bins and project scripts to PATH
#   5. Sets TORCH_CUDA_ARCH_LIST for RTX 3080 Laptop (sm_86)
#
# Why these settings are needed:
#   - Ubuntu 24.04 ships GCC 13 by default; nvcc 12.0 only supports up to GCC 12
#   - CUDA toolkit is split across /usr/lib/nvidia-cuda-toolkit (bins),
#     /usr/include (headers), /usr/lib/x86_64-linux-gnu (libs)
#   - ~/cuda-home provides a unified view with symlinks
#   - TORCH_CUDA_ARCH_LIST avoids nvcc probing all architectures at build time
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural-rendering

# CUDA setup
export CUDA_HOME=~/cuda-home
export CUDA_PATH=~/cuda-home
export PATH="$CUDA_HOME/bin:$PATH"
# NOTE: Do NOT add system CUDA libs to LD_LIBRARY_PATH at runtime.
# PyTorch 2.3+ ships bundled CUDA runtime via nvidia-* pip packages.
# System libs conflict with the bundled ones. Only needed at compile time:
export CUDA_BUILD_LD_FLAGS="-L$CUDA_HOME/lib64 -L/usr/lib/x86_64-linux-gnu"

# Host compiler — gcc-12 is the highest version nvcc 12.0 supports
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# GPU architecture — RTX 3080 Laptop = Ampere = sm_86
# Setting this prevents nvcc from querying all architectures during build
export TORCH_CUDA_ARCH_LIST="8.6"

# Project paths
export NEURAL_RENDERING_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/repos/gaussian-splatting:$PYTHONPATH"

echo "✅ Neural Rendering environment activated"
echo "   Conda env:    neural-rendering"
echo "   PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:         $(python -c 'import torch; print(torch.version.cuda)')"
echo "   GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "   CUDA_HOME:    $CUDA_HOME"
echo "   CC:           $CC ($(gcc-12 --version | head -1))"
echo "   Project root: $NEURAL_RENDERING_ROOT"
