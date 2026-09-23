"""
Volumetric rendering of density_volume.npz using mayavi, the same technique
used by wisp/trainers/tracker/nemd_tracker.py in the same author's pixi-wisp
repo (https://github.com/barikata1984/pixi-wisp): a true ray-cast volume
(mlab.pipeline.volume over a scalar_field), not a discrete point cloud like
scripts/visualize_density_volume.py's Open3D rendering. Unoccupied voxels are
made fully transparent via the opacity transfer function instead of being
zeroed out and rendered as (visible) black/lowest-colormap voxels.

Example:
    pixi run python scripts/visualize_density_volume_mayavi.py \\
        --npz outputs/sledgehammer/density/density_volume.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mayavi import mlab  # noqa: E402  (import after sys.path insert, mirrors pixi-wisp's own ordering)
from mayavi_vol_renderer import get_custom_colormap, init_mayavi_vol_renderer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Mayavi volumetric rendering of a density_volume.npz.')
    p.add_argument('--npz', type=str, required=True)
    p.add_argument('--output', type=str, default=None,
                    help='default: <npz_dir>/density_volume_mayavi.png')
    p.add_argument('--colormap', type=str, default='viridis',
                    help="mayavi/VTK LUT name (default: 'viridis', same as nemd_tracker's default)")
    p.add_argument('--opacity', type=float, default=1.0,
                    help='opacity of occupied voxels (default: 1.0, opaque -- same as nemd default)')
    p.add_argument('--transparent_input', type=float, default=-1.0,
                    help='scalar value used to mark unoccupied voxels as fully transparent')
    p.add_argument('--figure_size', type=int, default=720)
    p.add_argument('--shade', action='store_true',
                    help='enable volume shading (nemd_tracker runs with shade=False by default)')
    p.add_argument('--azimuth', type=float, default=None)
    p.add_argument('--elevation', type=float, default=None)
    p.add_argument('--distance', type=float, default=None)
    p.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'max'],
                    help="scalar normalization: 'minmax' = 1-99 percentile stretch of occupied densities; "
                         "'max' = per-voxel mass (density*dV) divided by its maximum, so max=1, 0 stays 0")
    p.add_argument('--interpolation', type=str, default='linear', choices=['linear', 'nearest'],
                    help="VTK volume interpolation. 'linear' (pixi-wisp default) blends each occupied voxel "
                         "with its transparent (-1) neighbours, creating a shell of artificially LOW values "
                         "at the surface that dominates opaque renders; 'nearest' shows each voxel's true value")
    p.add_argument('--show', action='store_true',
                    help='open an interactive mayavi/Qt window on $DISPLAY instead of rendering '
                         'offscreen (needs a reachable X server -- see README_LOCAL.md); blocks '
                         'until the window is closed')
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.npz)
    density = d['density']
    occupancy = d['occupancy'].astype(bool)
    x_centers, y_centers, z_centers = d['x_centers'], d['y_centers'], d['z_centers']

    if not occupancy.any():
        print('no occupied voxels -- nothing to render')
        return

    if args.normalize == 'max':
        mass = density * float(np.prod(d['voxel_size']))
        vmin, vmax = 0.0, float(mass[occupancy].max())
        normalized = np.clip(mass / vmax, 0.0, 1.0)
    else:
        vmin, vmax = np.percentile(density[occupancy], [1, 99])
        vmax = max(vmax, vmin + 1e-9)
        normalized = np.clip((density - vmin) / (vmax - vmin), 0.0, 1.0)
    scalars = np.where(occupancy, normalized, args.transparent_input).astype(np.float32)

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    mlab.options.offscreen = not args.show

    _ctf, _otf = get_custom_colormap(
        args.colormap, num_colors=256, opacity=args.opacity, transparent_input=args.transparent_input)

    figure, volume = init_mayavi_vol_renderer(
        scalars, x=xx, y=yy, z=zz, size=args.figure_size, _ctf=_ctf, _otf=_otf, shade=args.shade)
    # VTK applies the opacity transfer function per *unit distance* (default 1.0
    # world unit). Our grid is metric (voxel ~3 mm, object ~2-5 cm thick), so
    # with the default an opacity of 0.99 becomes ~0.01 per voxel and the whole
    # object renders nearly transparent (opacity exactly 1.0 is a special case
    # that stays opaque, which is why the default viridis render looked solid).
    # pixi-wisp renders in NDC (+-1) coordinates where the default is fine.
    volume._volume_property.scalar_opacity_unit_distance = float(np.mean(d['voxel_size']))
    volume._volume_property.interpolation_type = args.interpolation

    if args.azimuth is not None or args.elevation is not None or args.distance is not None:
        mlab.view(figure=figure, azimuth=args.azimuth, elevation=args.elevation, distance=args.distance)

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.npz)),
                                            'density_volume_mayavi.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlab.savefig(out_path, figure=figure)
    if args.normalize == 'max':
        print('saved:', out_path, '(per-voxel mass / max mass, max=%.3e kg, colormap=%s, opacity=%.2f)' %
              (vmax, args.colormap, args.opacity))
    else:
        print('saved:', out_path, '(density range shown: %.1f-%.1f kg/m^3, colormap=%s, opacity=%.2f)' %
              (vmin, vmax, args.colormap, args.opacity))

    if args.show:
        print('opening interactive window on DISPLAY=%s ... close it to exit' % os.environ.get('DISPLAY'))
        mlab.show()


if __name__ == '__main__':
    main()
