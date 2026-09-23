"""
Visualize a density_volume.npz produced by query_density_volume.py.

(A) 3D: occupied voxel centers as an Open3D point cloud, colored by density.
(B) 2D: XY / XZ / YZ orthogonal slices through the volume center, via matplotlib.

Both are saved under outputs/<scene>/density/ (created if missing). No new
dependencies beyond what's already in requirements.txt / pixi.toml
(open3d, matplotlib, numpy).

Example:
    pixi run python scripts/visualize_density_volume.py \\
        --npz outputs/B075YQXRBS_ATVPDKIKX0DER/density/density_volume.npz
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def parse_args():
    p = argparse.ArgumentParser(description='Visualize a density_volume.npz.')
    p.add_argument('--npz', type=str, required=True)
    p.add_argument('--out_dir', type=str, default=None,
                    help='default: same directory as --npz')
    p.add_argument('--cmap', type=str, default='inferno')
    p.add_argument('--point_size', type=float, default=4.0)
    p.add_argument('--show', action='store_true', help='also open an interactive Open3D window')
    return p.parse_args()


def load_volume(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        'density': d['density'],
        'occupancy': d['occupancy'],
        'x_centers': d['x_centers'],
        'y_centers': d['y_centers'],
        'z_centers': d['z_centers'],
        'aabb_min': d['aabb_min'],
        'aabb_max': d['aabb_max'],
        'voxel_size': d['voxel_size'],
    }


def visualize_3d(vol, out_path, cmap_name='inferno', point_size=4.0, show=False):
    import open3d as o3d

    occ = vol['occupancy'].astype(bool)
    density = vol['density']
    xx, yy, zz = np.meshgrid(vol['x_centers'], vol['y_centers'], vol['z_centers'], indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1)[occ]
    vals = density[occ]

    if len(pts) == 0:
        print('no occupied voxels -- skipping 3D visualization')
        return

    vmin, vmax = np.percentile(vals, [1, 99])
    vmax = max(vmax, vmin + 1e-6)
    cmap = mpl.colormaps[cmap_name]
    colors = cmap(np.clip((vals - vmin) / (vmax - vmin), 0, 1))[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if show:
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=1024)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.light_on = False
        vis.update_renderer()
        vis.poll_events()
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()
        print('saved:', out_path, '(%d occupied voxels, density range shown: %.1f-%.1f kg/m^3)' %
              (len(pts), vmin, vmax))
    except Exception as e:
        # Off-screen rendering needs a working display/EGL backend, which may
        # not be available in a headless container. Fall back to a .ply the
        # user can open locally (e.g. `pixi run python -c "import open3d as o3d;
        # o3d.visualization.draw_geometries([o3d.io.read_point_cloud('...')])"`
        # or any other point cloud viewer).
        ply_path = os.path.splitext(out_path)[0] + '.ply'
        o3d.io.write_point_cloud(ply_path, pcd)
        print('off-screen render failed (%s); saved colored point cloud instead: %s' % (e, ply_path))


def visualize_slices(vol, out_path, cmap_name='inferno'):
    density = vol['density']
    occ = vol['occupancy'].astype(bool)
    masked = np.where(occ, density, np.nan)
    nx, ny, nz = density.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    finite_vals = masked[np.isfinite(masked)]
    if finite_vals.size == 0:
        print('no occupied voxels -- skipping slice visualization')
        return
    vmin, vmax = np.nanpercentile(finite_vals, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    slices = [
        ('XY (z=center)', masked[:, :, cz].T, vol['x_centers'], vol['y_centers']),
        ('XZ (y=center)', masked[:, cy, :].T, vol['x_centers'], vol['z_centers']),
        ('YZ (x=center)', masked[cx, :, :].T, vol['y_centers'], vol['z_centers']),
    ]
    im = None
    for ax, (title, sl, ax1, ax2) in zip(axes, slices):
        extent = [ax1.min(), ax1.max(), ax2.min(), ax2.max()]
        im = ax.imshow(sl, origin='lower', extent=extent, cmap=cmap_name, vmin=vmin, vmax=vmax,
                        aspect='equal')
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label='density (kg/m^3)')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out_path)


def main():
    args = parse_args()
    vol = load_volume(args.npz)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    visualize_3d(vol, os.path.join(out_dir, 'density_3d.png'), cmap_name=args.cmap,
                 point_size=args.point_size, show=args.show)
    visualize_slices(vol, os.path.join(out_dir, 'density_slices.png'), cmap_name=args.cmap)


if __name__ == '__main__':
    main()
