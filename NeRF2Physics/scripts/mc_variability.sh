#!/usr/bin/env bash
# Monte-Carlo estimate of the whole pipeline's output variability: N full
# re-runs (NeRF seed, point-cloud sampling, and LLM dictionary all vary),
# each evaluated on the GT 128^3 grid. Summarize with scripts/mc_summary.py.
#   scripts/mc_variability.sh --dataset /path/to/sledgehammer_merged_seed42 --samples samples.json [--n 5] [--iters 20000]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; export PATH="$HOME/.pixi/bin:$PATH"
N=5; ITERS=20000; DATASET=""; SAMPLES=""; SCENE=sledgehammer
while [ $# -gt 0 ]; do case "$1" in --n) N=$2; shift 2;; --iters) ITERS=$2; shift 2;; --dataset) DATASET=$2; shift 2;; --samples) SAMPLES=$2; shift 2;; *) echo "unknown $1" >&2; exit 1;; esac; done
[ -n "$DATASET" ] && [ -n "$SAMPLES" ] || { echo "need --dataset and --samples" >&2; exit 1; }
A=0.1950048121760585; OUT=outputs/$SCENE/mc; mkdir -p "$OUT"
for k in $(seq 1 "$N"); do
  D=data/mc_run_$k; S=$D/scenes/$SCENE; R=$OUT/run_$k; mkdir -p "$S" "$R"
  ln -sfn "$DATASET/complete" "$S/images"; ln -sfn "$DATASET/complete" "$S/complete"
  ln -sfn "$DATASET/masks" "$S/masks"; ln -sfn "$DATASET/ground_truth.csv" "$S/ground_truth.csv"; cp "$DATASET/transforms.json" "$S/"
  pixi run python - "$SAMPLES" "$k" "$S" <<'EOF'
import json, sys
samples = json.load(open(sys.argv[1]))['samples']; k = int(sys.argv[2]); s = sys.argv[3]
base = json.load(open('results/sledgehammer/info_new.json'))
base['candidate_materials_density'] = samples[k % len(samples)]
json.dump(base, open(s + '/info_new.json', 'w'), indent=2)
print('[mc] run', k, 'dictionary:', base['candidate_materials_density'])
EOF
  SEED=$((1000 + k)); echo "[$(date +%H:%M:%S)] [mc] run $k: seed=$SEED"
  if [ ! -f "$S/ns/point_cloud.ply" ]; then
    pixi run python ns_reconstruction.py --data_dir "$D" --start_idx 0 --end_idx 1 --vis_mode tensorboard --nerf_seed "$SEED" \
      --training_iters "$ITERS" --near_plane 0.05 --far_plane 2.0 --proposal_initial_sampler piecewise > "$R/recon.log" 2>&1
  fi
  pixi run python feature_fusion.py --data_dir "$D" --start_idx 0 --end_idx 1 > "$R/fusion.log" 2>&1
  pixi run python scripts/query_density_volume.py --scene_dir "$S" --mats_load_name info_new \
    --aabb_min -$A -$A -$A --aabb_max $A $A $A --resolution 128 128 128 --query_batch_size 16384 \
    --output "$R/density_volume.npz" > "$R/voxel.log" 2>&1
  pixi run python scripts/evaluate_vs_gt.py --npz "$R/density_volume.npz" --gt_csv "$S/ground_truth.csv" --out_json "$R/metrics.json" 2>&1 | grep "^\[metrics\]"
  echo "[$(date +%H:%M:%S)] [mc] run $k done"
done
echo "[mc] all $N runs done -> $OUT"
