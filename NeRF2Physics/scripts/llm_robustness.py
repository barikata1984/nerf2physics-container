"""
Robustness of the downstream density/mass estimates to the LLM material
dictionary (the material-proposal step is sampled; everything after it is
deterministic). Takes N candidate-material strings, and for each one runs the
same voxel query (scripts/query_density_volume.py on the GT 128^3 grid) and
evaluation (scripts/evaluate_vs_gt.py), then summarizes the spread.

    pixi run python scripts/llm_robustness.py --scene_dir <scene> --gt_csv <csv> \
        --samples samples.json --out_dir outputs/<scene>/llm_robustness

samples.json: {"samples": ["(steel: 7700-8050 kg/m^3);...", ...]}
"""
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gpt_inference import parse_material_list  # noqa: E402

A = 0.1950048121760585  # merged_seed42 GT cube half-extent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True); ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--samples', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--resolution', type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    samples = json.load(open(a.samples))['samples']
    base = json.load(open(os.path.join(a.scene_dir, 'info_new.json')))
    rows = []
    for i, mats in enumerate(samples):
        parsed = parse_material_list(mats)
        if parsed is None:
            print(f'[sample {i}] UNPARSEABLE, skipped: {mats}'); continue
        info = dict(base); info['candidate_materials_density'] = mats
        name = f'info_llm{i:02d}'
        with open(os.path.join(a.scene_dir, name + '.json'), 'w') as f:
            json.dump(info, f)
        npz = os.path.join(a.out_dir, f'density_volume_{i:02d}.npz'); mj = os.path.join(a.out_dir, f'metrics_{i:02d}.json')
        subprocess.run([sys.executable, 'scripts/query_density_volume.py', '--scene_dir', a.scene_dir, '--mats_load_name', name,
                        '--aabb_min', str(-A), str(-A), str(-A), '--aabb_max', str(A), str(A), str(A),
                        '--resolution', str(a.resolution), str(a.resolution), str(a.resolution), '--query_batch_size', '16384',
                        '--output', npz], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, 'scripts/evaluate_vs_gt.py', '--npz', npz, '--gt_csv', a.gt_csv, '--out_json', mj],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(os.path.join(a.scene_dir, name + '.json'))
        M = json.load(open(mj)); names, vals = parsed
        mids = [np.mean(v) for v in vals]
        row = {'i': i, 'materials': names, 'midpoints': mids, 'dict_mean_density': float(np.mean(mids)),
               'mass_kg': M['predicted']['mass_kg'], 'mass_ratio': M['compare']['mass_ratio_pred_over_gt'],
               'com_err_cm': M['compare']['com_error_m'] * 100, 'inertia_relerr': M['compare']['inertia_origin_rel_frobenius_error'],
               'swd_cm': M['sliced_wasserstein_m']['pred_vs_gt'] * 100,
               'wood_over_steel': M.get('voxelwise', {}).get('wood_over_steel_ratio', {}).get('pred', float('nan'))}
        rows.append(row)
        print(f"[sample {i:2d}] mass={row['mass_kg']:.3f}kg ({row['mass_ratio']:.2f}xGT) COM err={row['com_err_cm']:.1f}cm "
              f"SWD={row['swd_cm']:.2f}cm wood/steel={row['wood_over_steel']:.2f} dict_mean={row['dict_mean_density']:.0f} :: {', '.join(names)}")
    keys = ['mass_kg', 'mass_ratio', 'com_err_cm', 'inertia_relerr', 'swd_cm', 'wood_over_steel', 'dict_mean_density']
    summary = {k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows])),
                   'min': float(np.min([r[k] for r in rows])), 'max': float(np.max([r[k] for r in rows]))} for k in keys}
    json.dump({'rows': rows, 'summary': summary, 'gt_mass_kg': 1.1206, 'gt_wood_over_steel': 550 / 7850},
              open(os.path.join(a.out_dir, 'llm_robustness.json'), 'w'), indent=2)
    print('[summary] n=%d' % len(rows))
    for k, s in summary.items():
        print(f"[summary] {k:>18}: mean={s['mean']:.3f} std={s['std']:.3f} min={s['min']:.3f} max={s['max']:.3f}")


if __name__ == '__main__':
    main()
