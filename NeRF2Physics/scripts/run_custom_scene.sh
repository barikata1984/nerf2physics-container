#!/usr/bin/env bash
# End-to-end pipeline for ONE custom scene: NeRF training -> feature fusion ->
# captioning -> material proposal -> voxel density volume -> visualization.
#
# This is a thin wrapper around the original NeRF2Physics scripts plus the
# two local extension scripts in scripts/. It does not add any new logic of
# its own beyond argument plumbing and status checks.
#
# Usage:
#   scripts/run_custom_scene.sh <data_dir> <scene_name> [resolution]
#
# Example:
#   scripts/run_custom_scene.sh data/my_dataset my_object 64
#
# Expects data_dir/scenes/<scene_name>/{images/, transforms.json} to already
# exist (see README_LOCAL.md "3. Custom dataset layout"). Requires
# OPENAI_API_KEY (or my_api_key.py) and a BLIP-2 checkpoint to be available
# before the captioning/material-proposal stages -- see README_LOCAL.md.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <data_dir> <scene_name> [resolution]" >&2
  exit 1
fi

DATA_DIR="$1"
SCENE_NAME="$2"
RESOLUTION="${3:-64}"
SCENE_DIR="${DATA_DIR}/scenes/${SCENE_NAME}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.pixi/bin:$PATH"

if [ ! -d "${SCENE_DIR}/images" ]; then
  echo "error: ${SCENE_DIR}/images not found. Set up the scene first (see README_LOCAL.md)." >&2
  exit 1
fi

# The original scripts (ns_reconstruction.py, feature_fusion.py, ...) select
# scenes by sorted-list index (--start_idx/--end_idx), not by name -- there is
# no "run just this scene name" flag. Resolve SCENE_NAME to its index so this
# wrapper stays correct even if data_dir/scenes has more than one scene.
SCENE_IDX="$(pixi run python -c "
import os, sys
scenes = sorted(os.listdir(os.path.join('${DATA_DIR}', 'scenes')))
print(scenes.index('${SCENE_NAME}'))
" 2>/dev/null | tail -1)"
SCENE_END=$((SCENE_IDX + 1))

run() {
  echo "+++ $*"
  pixi run "$@"
}

echo "=== [1/6] Nerfstudio reconstruction (ns-train + ns-export + ns-render depth) ==="
run python ns_reconstruction.py --data_dir "$DATA_DIR" --start_idx "$SCENE_IDX" --end_idx "$SCENE_END"

echo "=== [2/6] CLIP feature fusion ==="
run python feature_fusion.py --data_dir "$DATA_DIR" --start_idx "$SCENE_IDX" --end_idx "$SCENE_END"

echo "=== [3/6] BLIP-2 captioning ==="
if [ -z "${BLIP2_MODEL_DIR:-}" ]; then
  echo "warning: BLIP2_MODEL_DIR not set, using script default (./blip2-flan-t5-xl)" >&2
fi
run python captioning.py --data_dir "$DATA_DIR" --start_idx "$SCENE_IDX" --end_idx "$SCENE_END" \
  ${BLIP2_MODEL_DIR:+--blip2_model_dir "$BLIP2_MODEL_DIR"}

echo "=== [4/6] LLM material proposal (density) ==="
if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "my_api_key.py" ]; then
  echo "error: set OPENAI_API_KEY or create my_api_key.py (see README_LOCAL.md '2. API key')." >&2
  exit 1
fi
run python material_proposal.py --data_dir "$DATA_DIR" --start_idx "$SCENE_IDX" --end_idx "$SCENE_END" \
  --property_name density

echo "=== [5/6] AABB voxel-grid density volume (local extension) ==="
run python scripts/query_density_volume.py \
  --scene_dir "$SCENE_DIR" \
  --mats_load_name info_new \
  --resolution "$RESOLUTION" "$RESOLUTION" "$RESOLUTION" \
  --output "outputs/${SCENE_NAME}/density/density_volume.npz"

echo "=== [6/6] Visualization (local extension) ==="
run python scripts/visualize_density_volume.py \
  --npz "outputs/${SCENE_NAME}/density/density_volume.npz"

echo "done. outputs under outputs/${SCENE_NAME}/density/"
