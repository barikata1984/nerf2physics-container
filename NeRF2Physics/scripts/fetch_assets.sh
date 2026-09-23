#!/usr/bin/env bash
# Fetch the large assets that are deliberately NOT in git, and lay out a dataset
# as a NeRF2Physics scene. Idempotent.
#
#   scripts/fetch_assets.sh [--dataset /path/to/sledgehammer_merged_seed42] [--scene sledgehammer]
#
# BLIP-2 goes to $BLIP2_DIR (default: ../blip2-flan-t5-xl next to this repo), safetensors only
# (~16 GB; the redundant pytorch_model-*.bin copies are excluded via lfs.fetchexclude).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLIP2_DIR="${BLIP2_DIR:-$REPO/../blip2-flan-t5-xl}"
DATASET=""; SCENE="sledgehammer"; DATA_DIR="$REPO/data/merged_seed42"
while [ $# -gt 0 ]; do case "$1" in
  --dataset) DATASET="$2"; shift 2;; --scene) SCENE="$2"; shift 2;; --data_dir) DATA_DIR="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 1;; esac; done

echo "== BLIP-2 -> $BLIP2_DIR"
if [ ! -f "$BLIP2_DIR/model-00002-of-00002.safetensors" ]; then
  git clone --no-checkout https://huggingface.co/Salesforce/blip2-flan-t5-xl "$BLIP2_DIR"
  git -C "$BLIP2_DIR" config lfs.fetchexclude "pytorch_model-00001-of-00002.bin,pytorch_model-00002-of-00002.bin"
  git -C "$BLIP2_DIR" checkout main
  git -C "$BLIP2_DIR" lfs pull
fi
echo "   $(du -sh "$BLIP2_DIR" | cut -f1) present"

if [ -n "$DATASET" ]; then
  S="$DATA_DIR/scenes/$SCENE"; echo "== scene $S <- $DATASET"
  [ -d "$DATASET/complete" ] && [ -f "$DATASET/transforms.json" ] || { echo "dataset must contain complete/ and transforms.json" >&2; exit 1; }
  mkdir -p "$S"
  ln -sfn "$DATASET/complete" "$S/images"      # NeRF2Physics reads images/
  ln -sfn "$DATASET/complete" "$S/complete"    # nerfstudio follows transforms.json file_path "complete/..."
  [ -d "$DATASET/masks" ] && ln -sfn "$DATASET/masks" "$S/masks"
  [ -f "$DATASET/ground_truth.csv" ] && ln -sfn "$DATASET/ground_truth.csv" "$S/ground_truth.csv"
  cp "$DATASET/transforms.json" "$S/transforms.json"
  # material dictionary produced in the original session (LLM step needs an API key otherwise)
  REF="$REPO/results/$SCENE/info_new.json"
  [ -f "$REF" ] && [ ! -f "$S/info_new.json" ] && cp "$REF" "$S/info_new.json" && echo "   info_new.json restored from results/"
  echo "   images: $(ls "$S/images" | wc -l)"
fi
echo "done"
