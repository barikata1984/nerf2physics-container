#!/usr/bin/env bash
# Reproduce the sledgehammer results end-to-end (see README_LOCAL.md).
#   scripts/reproduce_sledgehammer.sh [--iters 20000] [--skip_train]
# Prerequisites: pixi env installed (pixi install && pixi run tcnn-install),
# scripts/fetch_assets.sh --dataset <path> run once, BLIP2_DIR set if not at ../blip2-flan-t5-xl.
# NeRF training is not bit-reproducible on CUDA; expect small numeric drift vs results/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"
ITERS=20000; SKIP_TRAIN=0
while [ $# -gt 0 ]; do case "$1" in --iters) ITERS="$2"; shift 2;; --skip_train) SKIP_TRAIN=1; shift;; *) echo "unknown $1" >&2; exit 1;; esac; done
DATA=data/merged_seed42; SCENE=sledgehammer; S=$DATA/scenes/$SCENE
GT=$S/ground_truth.csv; OUT=outputs/$SCENE/reproduce; mkdir -p "$OUT"
BLIP2_DIR="${BLIP2_DIR:-$REPO/../blip2-flan-t5-xl}"
A=0.1950048121760585   # GT voxel cube half-extent (aabb_scale) -> 128^3 grid identical to ground_truth.csv
step() { echo "[$(date +%H:%M:%S)] === $* ==="; }

if [ "$SKIP_TRAIN" = 0 ]; then
  step "1 NeRF reconstruction ($ITERS iters). NOTE: upstream's uniform/0.4/6.0 sampling breaks this thin object"
  pixi run python ns_reconstruction.py --data_dir $DATA --start_idx 0 --end_idx 1 --vis_mode tensorboard \
    --training_iters "$ITERS" --near_plane 0.05 --far_plane 2.0 --proposal_initial_sampler piecewise
fi
step "2 CLIP feature fusion"
pixi run python feature_fusion.py --data_dir $DATA --start_idx 0 --end_idx 1
if [ ! -f "$S/info_new.json" ] || ! grep -q candidate_materials_density "$S/info_new.json"; then
  step "3 captioning + LLM material proposal (needs BLIP-2 and an API key; skipped if results/ copy exists)"
  pixi run python captioning.py --data_dir $DATA --start_idx 0 --end_idx 1 --blip2_model_dir "$BLIP2_DIR"
  pixi run python material_proposal.py --data_dir $DATA --start_idx 0 --end_idx 1 --property_name density "${LLM_ARGS[@]:-}"
else
  step "3 material dictionary: using existing $S/info_new.json"
fi
step "4 official mass prediction (integral)"
pixi run python predict_property.py --data_dir $DATA --start_idx 0 --end_idx 1 --property_name density \
  --mats_load_name info_new --prediction_mode integral | grep -E "candidate|kg" | tee "$OUT/predict_property.txt"
step "5 official visualization + comparison figures"
pixi run python visualization.py --data_dir $DATA --scene_name $SCENE --mats_load_name info_new --show 0 \
  --viz_save_name ${SCENE}_paper --pt_size 3
pixi run python scripts/make_comparison_figure.py --viz_dir viz/${SCENE}_paper --viz_save_name ${SCENE}_paper --output "$OUT/comparison.png"
pixi run python scripts/make_comparison_figure.py --viz_dir viz/${SCENE}_paper --viz_save_name ${SCENE}_paper \
  --density_cool --scene_dir $S --output "$OUT/comparison_cool.png"
step "6 voxel density volume on the GT 128^3 grid (this repo's extension)"
pixi run python scripts/query_density_volume.py --scene_dir $S --mats_load_name info_new \
  --aabb_min -$A -$A -$A --aabb_max $A $A $A --resolution 128 128 128 --query_batch_size 16384 \
  --output "$OUT/density_volume.npz" --report_mass --assume_metric | tail -3
step "7 inertial parameters + sliced Wasserstein vs ground truth"
pixi run python scripts/evaluate_vs_gt.py --npz "$OUT/density_volume.npz" --gt_csv "$GT" --out_json "$OUT/metrics.json"
step "8 volume renders"
pixi run python scripts/visualize_density_volume.py --npz "$OUT/density_volume.npz"
pixi run python scripts/visualize_density_volume_mayavi.py --npz "$OUT/density_volume.npz" \
  --normalize max --colormap cool --opacity 0.99 --interpolation nearest --output "$OUT/density_mayavi_cool_massnorm.png"
step "DONE -> $OUT (compare with results/$SCENE/)"
