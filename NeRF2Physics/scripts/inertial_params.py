"""
Inertial parameters (mass, center of mass, inertia tensor) from a density
volume, and comparison against a ground-truth voxel file / header.

Two inputs are supported, so the same code path is used for both sides:
  --npz  density_volume.npz from scripts/query_density_volume.py
         (mass per voxel = density * occupancy * voxel volume)
  --gt_csv  merged_seed42-style ground_truth.csv (header row with total_mass,
         mx,my,mz, ixx..izx, then x,y,z,mass,mass_density rows)

Inertia is reported about the world origin AND about the COM (both conventions
are printed, the header's convention is inferred by which one it matches).
Absolute mass is only meaningful if the volume's coordinate frame is metric;
for merged_seed42 that was verified by matching the reconstruction's extent to
the GT voxel extent (see README_LOCAL.md).
"""
import argparse
import numpy as np
import pandas as pd


def inertia_tensor(pts, m, about):
    r = pts - about
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    Ixx = np.sum(m * (y**2 + z**2)); Iyy = np.sum(m * (x**2 + z**2)); Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y); Iyz = -np.sum(m * y * z); Izx = -np.sum(m * z * x)
    return np.array([[Ixx, Ixy, Izx], [Ixy, Iyy, Iyz], [Izx, Iyz, Izz]])


def params_from_points(pts, m, label):
    M = m.sum(); com = (pts * m[:, None]).sum(0) / M
    I0 = inertia_tensor(pts, m, np.zeros(3)); Ic = inertia_tensor(pts, m, com)
    print(f"[{label}] mass={M:.4f} kg  COM={np.round(com, 4)}")
    for name, I in [('about origin', I0), ('about COM', Ic)]:
        print(f"[{label}] I {name}: ixx={I[0,0]:.5f} iyy={I[1,1]:.5f} izz={I[2,2]:.5f} ixy={I[0,1]:.2e} iyz={I[1,2]:.2e} izx={I[0,2]:.2e}")
    return M, com, I0, Ic


def load_npz(path):
    d = np.load(path)
    dens, occ = d['density'], d['occupancy'].astype(bool)
    xx, yy, zz = np.meshgrid(d['x_centers'], d['y_centers'], d['z_centers'], indexing='ij')
    dV = float(np.prod(d['voxel_size']))
    sel = occ & (dens > 0)
    pts = np.stack([xx[sel], yy[sel], zz[sel]], -1); m = dens[sel] * dV
    return pts, m


def load_gt(path):
    hdr = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    raw = pd.read_csv(path, skiprows=2, names=['x', 'y', 'z', 'mass', 'mass_density'], dtype=str)
    gt = raw.apply(pd.to_numeric, errors='coerce').dropna()
    gt = gt[gt.mass > 0]
    return hdr, gt[['x', 'y', 'z']].values, gt['mass'].values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz'); p.add_argument('--gt_csv')
    a = p.parse_args()
    if a.gt_csv:
        hdr, gpts, gm = load_gt(a.gt_csv)
        print(f"[GT header] mass={hdr['total_mass']:.4f} COM=({hdr['mx']:.4f},{hdr['my']:.4f},{hdr['mz']:.4f}) "
              f"ixx={hdr['ixx']:.5f} iyy={hdr['iyy']:.5f} izz={hdr['izz']:.5f} ixy={hdr['ixy']:.2e} iyz={hdr['iyz']:.2e} izx={hdr['izx']:.2e}")
        Mg, comg, I0g, Icg = params_from_points(gpts, gm, 'GT voxels')
    if a.npz:
        ppts, pm = load_npz(a.npz)
        Mp, comp, I0p, Icp = params_from_points(ppts, pm, 'predicted')
        if a.gt_csv:
            print(f"[compare] mass pred/GT = {Mp/Mg:.3f}   |COM error| = {np.linalg.norm(comp-comg)*100:.2f} cm (object length ~39 cm)")
            for name, Ip, Ig in [('origin', I0p, I0g), ('COM', Icp, Icg)]:
                rel = np.linalg.norm(Ip - Ig) / np.linalg.norm(Ig)
                print(f"[compare] inertia about {name}: relative Frobenius error = {rel:.3f}  (diag pred/GT = {np.round(np.diag(Ip)/np.diag(Ig), 3)})")


if __name__ == '__main__':
    main()
