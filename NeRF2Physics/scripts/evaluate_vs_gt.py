"""
Compare a density_volume.npz against a merged_seed42-style ground_truth.csv
and write metrics.json: mass / COM / inertia (scripts/inertial_params.py),
sliced Wasserstein distance (scripts/sliced_wasserstein.py), and -- when the
two grids coincide -- voxel-wise occupancy agreement and the predicted density
at GT-steel vs GT-wood voxels.

    pixi run python scripts/evaluate_vs_gt.py --npz <density_volume.npz> --gt_csv <ground_truth.csv> --out_json metrics.json
"""
import argparse, json
import numpy as np
import pandas as pd

from inertial_params import load_npz, load_gt, inertia_tensor
from sliced_wasserstein import swd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--out_json', required=True); ap.add_argument('--n_proj', type=int, default=256)
    a = ap.parse_args()
    d = np.load(a.npz)
    ppts, pm = load_npz(a.npz); hdr, gpts, gm = load_gt(a.gt_csv)
    M = {}
    for label, pts, m in [('predicted', ppts, pm), ('gt_voxels', gpts, gm)]:
        mass = float(m.sum()); com = (pts * m[:, None]).sum(0) / mass
        M[label] = {'mass_kg': mass, 'com': com.tolist(),
                    'inertia_about_origin': inertia_tensor(pts, m, np.zeros(3)).tolist(),
                    'inertia_about_com': inertia_tensor(pts, m, com).tolist(), 'n_voxels': int(len(pts))}
    M['gt_header'] = {k: (float(v) if not isinstance(v, str) else v) for k, v in hdr.items()}
    Ip, Ig = np.array(M['predicted']['inertia_about_origin']), np.array(M['gt_voxels']['inertia_about_origin'])
    M['compare'] = {
        'mass_ratio_pred_over_gt': M['predicted']['mass_kg'] / M['gt_voxels']['mass_kg'],
        'com_error_m': float(np.linalg.norm(np.array(M['predicted']['com']) - np.array(M['gt_voxels']['com']))),
        'inertia_origin_rel_frobenius_error': float(np.linalg.norm(Ip - Ig) / np.linalg.norm(Ig)),
        'inertia_origin_diag_ratio': (np.diag(Ip) / np.diag(Ig)).tolist(),
    }
    s_mean, s_std = swd(ppts, pm, gpts, gm, a.n_proj)
    u_mean, _ = swd(gpts, np.ones(len(gpts)), gpts, gm, a.n_proj)
    g_mean, _ = swd(ppts, np.ones(len(ppts)), gpts, np.ones(len(gpts)), a.n_proj)
    M['sliced_wasserstein_m'] = {'pred_vs_gt': s_mean, 'pred_vs_gt_std': s_std,
                                 'baseline_uniform_density_on_gt_shape_vs_gt': u_mean,
                                 'geometry_only_uniform_pred_vs_uniform_gt': g_mean, 'n_proj': a.n_proj}
    # voxel-wise comparison if grids coincide
    raw = pd.read_csv(a.gt_csv, skiprows=2, names=['x', 'y', 'z', 'mass', 'mass_density'], dtype=str)
    gt = raw.apply(pd.to_numeric, errors='coerce').dropna()
    xs = np.sort(np.unique(gt.x.round(6)))
    if len(xs) == len(d['x_centers']) and np.allclose(xs, d['x_centers'], atol=1e-6):
        step = xs[1] - xs[0]
        idx = [np.round((gt[c].values - xs[0]) / step).astype(int) for c in ('x', 'y', 'z')]
        G = np.zeros(d['density'].shape); G[idx[0], idx[1], idx[2]] = gt.mass_density.values
        occ = d['occupancy'].astype(bool); dens = d['density']
        steel, wood = G > 5000, (G > 0) & (G < 5000)
        M['voxelwise'] = {
            'grid': 'identical to ground_truth.csv',
            'pred_occupied': int(occ.sum()), 'gt_occupied': int((G > 0).sum()),
            'false_positive_voxels': int((occ & (G == 0)).sum()), 'missed_gt_voxels': int(((G > 0) & ~occ).sum()),
            'iou': float((occ & (G > 0)).sum() / (occ | (G > 0)).sum()),
            'pred_density_at_gt_steel': float(dens[steel & occ].mean()), 'gt_steel_density': float(G[steel].mean()),
            'pred_density_at_gt_wood': float(dens[wood & occ].mean()), 'gt_wood_density': float(G[wood].mean()),
        }
        M['voxelwise']['wood_over_steel_ratio'] = {'pred': M['voxelwise']['pred_density_at_gt_wood'] / M['voxelwise']['pred_density_at_gt_steel'],
                                                  'gt': M['voxelwise']['gt_wood_density'] / M['voxelwise']['gt_steel_density']}
    with open(a.out_json, 'w') as f:
        json.dump(M, f, indent=2)
    c, s = M['compare'], M['sliced_wasserstein_m']
    print(f"[metrics] mass pred/GT={c['mass_ratio_pred_over_gt']:.3f}  COM err={c['com_error_m']*100:.1f} cm  "
          f"inertia relerr={c['inertia_origin_rel_frobenius_error']:.3f}  SWD={s['pred_vs_gt']*100:.2f} cm "
          f"(uniform-density baseline {s['baseline_uniform_density_on_gt_shape_vs_gt']*100:.2f}, geometry-only {s['geometry_only_uniform_pred_vs_uniform_gt']*100:.2f})")
    if 'voxelwise' in M:
        v = M['voxelwise']; print(f"[metrics] voxel IoU={v['iou']:.3f}  wood/steel density ratio pred={v['wood_over_steel_ratio']['pred']:.3f} gt={v['wood_over_steel_ratio']['gt']:.3f}")
    print('saved', a.out_json)


if __name__ == '__main__':
    main()
