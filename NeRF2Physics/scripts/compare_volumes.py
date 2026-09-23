"""
Run-to-run variability of predicted density volumes on a shared grid.
Given >=2 density_volume.npz files (same AABB/resolution), report per-run
mass/COM and pairwise: occupancy IoU, voxel-wise density differences on the
jointly occupied voxels, COM distance, and sliced Wasserstein distance.

    pixi run python scripts/compare_volumes.py A.npz B.npz [C.npz ...] [--out_json x.json]
"""
import argparse, itertools, json
import numpy as np
from inertial_params import load_npz
from sliced_wasserstein import swd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('npz', nargs='+'); ap.add_argument('--out_json'); ap.add_argument('--n_proj', type=int, default=256)
    a = ap.parse_args()
    vols = []
    for p in a.npz:
        d = np.load(p); occ = d['occupancy'].astype(bool); dens = d['density']; dV = float(np.prod(d['voxel_size']))
        pts, m = load_npz(p); com = (pts * m[:, None]).sum(0) / m.sum()
        vols.append({'path': p, 'occ': occ, 'dens': dens, 'pts': pts, 'm': m, 'mass': float(m.sum()), 'com': com, 'dV': dV})
        print(f"[run] {p}: mass={m.sum():.4f} kg  COM={np.round(com, 4)}  occupied={int(occ.sum())}  "
              f"density mean/std over occupied={dens[occ].mean():.0f}/{dens[occ].std():.0f}")
    pairs = []
    for (i, A), (j, B) in itertools.combinations(enumerate(vols), 2):
        both = A['occ'] & B['occ']; either = A['occ'] | B['occ']
        diff = A['dens'][both] - B['dens'][both]
        s, _ = swd(A['pts'], A['m'], B['pts'], B['m'], a.n_proj)
        r = {'i': i, 'j': j, 'occupancy_iou': float(both.sum() / either.sum()),
             'mass_diff_kg': A['mass'] - B['mass'], 'mass_rel_diff': (A['mass'] - B['mass']) / max(A['mass'], B['mass']),
             'com_dist_cm': float(np.linalg.norm(A['com'] - B['com']) * 100),
             'voxel_density_diff_mean': float(diff.mean()), 'voxel_density_diff_std': float(diff.std()),
             'voxel_density_absdiff_p50_p95': [float(np.percentile(np.abs(diff), 50)), float(np.percentile(np.abs(diff), 95))],
             'swd_cm': s * 100}
        pairs.append(r)
        print(f"[pair {i}-{j}] occ IoU={r['occupancy_iou']:.3f}  mass diff={r['mass_diff_kg']:+.4f} kg ({100*r['mass_rel_diff']:+.1f}%)  "
              f"COM dist={r['com_dist_cm']:.2f} cm  voxel |dens diff| p50/p95={r['voxel_density_absdiff_p50_p95'][0]:.0f}/{r['voxel_density_absdiff_p50_p95'][1]:.0f} kg/m3 "
              f"(mean {r['voxel_density_diff_mean']:+.0f}, std {r['voxel_density_diff_std']:.0f})  SWD={r['swd_cm']:.2f} cm")
    if a.out_json:
        json.dump({'runs': [{k: v for k, v in r.items() if k in ('path', 'mass')} | {'com': r['com'].tolist()} for r in vols], 'pairs': pairs},
                  open(a.out_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
