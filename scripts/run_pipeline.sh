#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Master Preprocessing Pipeline
#
# Runs all 4 preprocessing steps in order for a given scene.
# Source the activation script first, then run this.
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/run_pipeline.sh scene_S003
#   bash scripts/run_pipeline.sh scene_S003 --scale 1.0 --dynamic_fps 5
#
# Steps:
#   01. Extract frames from raw MPEG videos
#   02. Run COLMAP to estimate camera poses
#   03. Prepare 3DGS-ready dataset structure
#   04. Validate reconstruction and visualize
# =============================================================================

set -e  # Exit on any error

SCENE=${1:-scene_S003}
SCALE=${SCALE:-0.5}
DYNAMIC_FPS=${DYNAMIC_FPS:-10}
CAMERA_MODEL=${CAMERA_MODEL:-OPENCV}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Neural Rendering Preprocessing Pipeline"
echo "============================================================"
echo "  Scene:        $SCENE"
echo "  Scale:        ${SCALE}x"
echo "  Dynamic FPS:  ${DYNAMIC_FPS}"
echo "  Camera model: ${CAMERA_MODEL}"
echo "============================================================"

# Step 1: Extract frames
echo ""
echo "[Step 1/4] Extracting frames..."
python "$SCRIPT_DIR/01_extract_frames.py" \
    --scene "$SCENE" \
    --scale "$SCALE" \
    --dynamic_fps "$DYNAMIC_FPS"

# Step 2: Run COLMAP
echo ""
echo "[Step 2/4] Running COLMAP..."
python "$SCRIPT_DIR/02_run_colmap.py" \
    --scene "$SCENE" \
    --camera_model "$CAMERA_MODEL"

# Step 3: Prepare 3DGS dataset
echo ""
echo "[Step 3/4] Preparing 3DGS dataset..."
python "$SCRIPT_DIR/03_prepare_dataset.py" \
    --scene "$SCENE"

# Step 4: Validate
echo ""
echo "[Step 4/4] Validating reconstruction..."
python "$SCRIPT_DIR/04_validate_reconstruction.py" \
    --scene "$SCENE"

echo ""
echo "============================================================"
echo "✅ Pipeline complete for $SCENE"
echo "   Results: data/processed/$SCENE/"
echo "   Validation: data/processed/$SCENE/validation/"
echo "============================================================"
