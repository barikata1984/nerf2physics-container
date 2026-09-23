"""
Sliced Wasserstein distance (SWD) between the predicted mass distribution
(density_volume.npz) and the ground-truth voxel mass distribution
(ground_truth.csv), both treated as probability measures over 3D points
(voxel centers weighted by voxel mass, normalized). Total mass is compared
separately in scripts/inertial_params.py -- SWD here measures *where* the
mass sits, not how much.

SWD_1(mu, nu) = E_theta[ W_1(proj_theta mu, proj_theta nu) ], estimated with
--n_proj random unit directions and scipy's exact weighted 1-D Wasserstein.
Units: same as the coordinate frame (meters for merged_seed42).

Baselines printed for interpretation: GT vs GT-geometry-with-uniform-density
(what a perfect-shape / no-material-knowledge predictor would score) and
GT vs the same GT shifted by 1 cm along z (a scale reference).
"""
import argparse
import numpy as np
from scipy.stats import wasserstein_distance

from inertial_params import load_npz, load_gt


def swd(p1, w1, p2, w2, n_proj=256, seed=0):
    rng = np.random.default_rng(seed)
    w1 = w1 / w1.sum(); w2 = w2 / w2.sum()
    dirs = rng.normal(size=(n_proj, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ds = [wasserstein_distance(p1 @ d, p2 @ d, w1, w2) for d in dirs]
    return float(np.mean(ds)), float(np.std(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--n_proj', type=int, default=256)
    a = ap.parse_args()
    ppts, pm = load_npz(a.npz)
    _, gpts, gm = load_gt(a.gt_csv)
    m, s = swd(ppts, pm, gpts, gm, a.n_proj)
    print(f"[SWD] predicted vs GT: {m*100:.3f} cm  (+/- {s*100:.3f} over {a.n_proj} projections)")
    m_u, _ = swd(gpts, np.ones(len(gpts)), gpts, gm, a.n_proj)
    print(f"[SWD baseline] GT geometry with uniform density vs GT: {m_u*100:.3f} cm")
    m_sh, _ = swd(gpts + np.array([0, 0, 0.01]), gm, gpts, gm, a.n_proj)
    print(f"[SWD baseline] GT shifted by 1 cm along z vs GT: {m_sh*100:.3f} cm")
    m_geo, _ = swd(ppts, np.ones(len(ppts)), gpts, np.ones(len(gpts)), a.n_proj)
    print(f"[SWD geometry-only] predicted occupancy (uniform) vs GT occupancy (uniform): {m_geo*100:.3f} cm")


if __name__ == '__main__':
    main()
