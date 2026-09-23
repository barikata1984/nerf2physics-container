"""
Query a NeRF2Physics scene for a dense mass-density volume over an axis-aligned
bounding box (AABB), and write it out as a canonical density_volume.npz.

This is the "step 11-13" extension described in README_LOCAL.md. It does NOT
modify predict_property.py / carving.py -- it only calls their existing public
functions (predict_physical_property_query, carve_torch) with our own voxel
centers instead of the surface / grid points the original scripts use.

Coordinate system: query points and the AABB you pass in are in the same
frame as utils.load_ns_point_cloud()'s output -- i.e. the *original* per-scene
frame used by transforms.json (dataparser normalization already undone). This
is NOT nerfstudio's internal [-1, 1]^3 training coordinate system.

Density units: kg/m^3 (matches the LLM material-proposal prompts in
gpt_inference.py). Values are the mean of the [low, high] range predicted by
predict_physical_property_query(), the same reduction visualization.py uses
for its colormap.

Occupancy: reuses carving.carve_torch() (visual-hull + depth consistency),
applied to our own voxel centers instead of carving.get_carved_pts()'s own
grid. This is an *image/depth-consistency* occupancy proxy, not a physical
measurement of hidden interior structure.

Example:
    pixi run python scripts/query_density_volume.py \\
        --scene_dir data/abo_500/scenes/B075YQXRBS_ATVPDKIKX0DER \\
        --resolution 64 64 64 \\
        --mats_load_name info_new \\
        --output outputs/B075YQXRBS_ATVPDKIKX0DER/density/density_volume.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT  # noqa: E402
from predict_property import predict_physical_property_query  # noqa: E402
from carving import get_bounding_box, project_3d_to_2d_torch  # noqa: E402
from utils import (  # noqa: E402
    load_ns_point_cloud,
    parse_dataparser_transforms_json,
    parse_transforms_json,
    load_images,
    load_depths,
)

import open_clip  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Query a NeRF2Physics scene on an AABB voxel grid.')
    p.add_argument('--scene_dir', type=str, required=True,
                    help='path to a single scene directory, e.g. data/abo_500/scenes/<scene_name>')
    p.add_argument('--property_name', type=str, default='density')
    p.add_argument('--mats_load_name', type=str, default='info_new',
                    help="name of the info json holding candidate materials "
                         "(NOTE: predict_property.py's own CLI default is 'info', "
                         "but captioning.py/material_proposal.py write 'info_new' by default -- "
                         "see README_LOCAL.md 'known quirks')")
    p.add_argument('--feature_load_name', type=str, default='ps56')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--show_mat_seg', type=int, default=0,
                    help='forwarded to predict_physical_property_query(), which reads this attribute '
                         '(default: 0, matches arguments.py); keep 0 for headless batch runs')

    # AABB: if omitted, auto-computed from the scene's own point cloud bounding box.
    p.add_argument('--aabb_min', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    p.add_argument('--aabb_max', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    p.add_argument('--aabb_buffer', type=float, default=0.1,
                    help='fractional buffer applied to the auto-computed AABB (default: 0.1)')
    p.add_argument('--resolution', type=int, nargs=3, default=[64, 64, 64], metavar=('NX', 'NY', 'NZ'))

    p.add_argument('--query_batch_size', type=int, default=8192,
                    help='voxel centers per predict_physical_property_query() call (default: 8192)')

    # Occupancy (reuses carving.carve_torch as-is).
    p.add_argument('--occ_dist_thr_ns', type=float, default=0.01,
                    help='depth-consistency threshold in nerfstudio-normalized units '
                         '(same default as carving.get_carved_pts)')
    p.add_argument('--skip_occupancy', action='store_true',
                    help='skip occupancy carving; density field will not be masked')

    # Optional mass sanity check (step 15).
    p.add_argument('--report_mass', action='store_true')
    p.add_argument('--assume_metric', action='store_true',
                    help='only set this if you have verified transforms.json poses are in real-world '
                         'meters; otherwise total mass is not meaningful and will not be printed as kg')

    p.add_argument('--output', type=str, default=None,
                    help='output .npz path (default: outputs/<scene_name>/density/density_volume.npz)')
    return p.parse_args()


def make_voxel_grid(aabb_min, aabb_max, resolution):
    """Voxel centers for an Nx x Ny x Nz grid over [aabb_min, aabb_max]."""
    axes = []
    voxel_size = np.zeros(3)
    for d in range(3):
        n = resolution[d]
        edges = np.linspace(aabb_min[d], aabb_max[d], n + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        axes.append(centers)
        voxel_size[d] = edges[1] - edges[0]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    grid_pts = np.stack([xx, yy, zz], axis=-1)  # [Nx, Ny, Nz, 3]
    return axes[0], axes[1], axes[2], grid_pts, voxel_size


def compute_occupancy(scene_dir, query_pts, dist_thr_ns, device, batch_size):
    """Occupancy for arbitrary query points, reusing carving.carve_torch().

    Mirrors carving.get_carved_pts()'s data loading, but scores OUR points
    instead of get_carved_pts()'s own internally-generated grid.
    """
    t_file = os.path.join(scene_dir, 'transforms.json')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    img_dir = os.path.join(scene_dir, 'images')
    depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')

    w2cs, K = parse_transforms_json(t_file, return_w2c=True)
    _, scale = parse_dataparser_transforms_json(dt_file)
    imgs, masks = load_images(img_dir, return_masks=True)
    depths = load_depths(depth_dir, Ks=None)
    dist_thr = dist_thr_ns / scale

    masks_t = [torch.from_numpy(m).to(device) for m in masks]
    depths_t = [torch.from_numpy(d).to(device) for d in depths]
    w2cs_t = [torch.from_numpy(w).float().to(device) for w in w2cs]
    K_t = torch.from_numpy(K).float().to(device)

    n = len(query_pts)
    occupied = np.zeros(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pts_batch = torch.from_numpy(query_pts[start:end]).float().to(device)
        occupied[start:end] = _carve_mask(pts_batch, masks_t, depths_t, w2cs_t, K_t, dist_thr)
    return occupied


def _carve_mask(pts, masks, depths, w2cs, K, dist_thr):
    """Same voting logic as carving.carve_torch(), but returns the boolean
    mask instead of the boolean-indexed points -- carve_torch only returns
    `pts[mask]`, which discards which points survived and in what order, so
    it can't be used directly to fill a dense occupancy volume."""
    n_imgs = len(masks)
    with torch.no_grad():
        mask_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        depth_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        for i in range(n_imgs):
            h, w = masks[i].shape
            pts_2d, dists = project_3d_to_2d_torch(pts, w2cs[i], K, return_dists=True)
            pts_2d = torch.round(pts_2d).long()
            pts_2d[:, 0] = torch.clamp(pts_2d[:, 0], 0, w - 1)
            pts_2d[:, 1] = torch.clamp(pts_2d[:, 1], 0, h - 1)
            observed_dists = depths[i]
            is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
            is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
            mask_votes[is_in_mask] += 1
            depth_votes[is_behind_depth] += 1
        keep = (mask_votes == n_imgs) & (depth_votes == n_imgs)
    return keep.cpu().numpy()


def main():
    args = parse_args()
    scene_name = os.path.basename(os.path.normpath(args.scene_dir))

    pcd_file = os.path.join(args.scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(args.scene_dir, 'ns', 'dataparser_transforms.json')

    if args.aabb_min is None or args.aabb_max is None:
        surf_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
        aabb_min, aabb_max = get_bounding_box(surf_pts, percentile=1.0, buffer=args.aabb_buffer)
        print('auto-computed AABB from point cloud: min=%s max=%s' % (aabb_min, aabb_max))
    else:
        aabb_min, aabb_max = np.array(args.aabb_min), np.array(args.aabb_max)

    x_centers, y_centers, z_centers, grid_pts, voxel_size = make_voxel_grid(
        aabb_min, aabb_max, args.resolution)
    flat_pts = grid_pts.reshape(-1, 3).astype(np.float32)
    n_total = len(flat_pts)
    print('voxel grid: %s, %d total voxels, voxel_size=%s' % (args.resolution, n_total, voxel_size))

    # --- density (predict_physical_property_query, used unmodified) ---
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    density_low = np.zeros(n_total, dtype=np.float32)
    density_high = np.zeros(n_total, dtype=np.float32)
    for start in range(0, n_total, args.query_batch_size):
        end = min(start + args.query_batch_size, n_total)
        batch_pts = torch.from_numpy(flat_pts[start:end]).to(args.device)
        vals = predict_physical_property_query(args, batch_pts, args.scene_dir, clip_model, clip_tokenizer,
                                                 return_all=False)
        density_low[start:end] = vals[:, 0]
        density_high[start:end] = vals[:, 1]
        print('  queried %d / %d voxels' % (end, n_total))
    density = (density_low + density_high) / 2.0

    # --- occupancy (carving.carve_torch, used unmodified) ---
    if args.skip_occupancy:
        occupancy = np.ones(n_total, dtype=bool)
        print('--skip_occupancy set: density is NOT masked by occupancy (whole AABB treated as solid)')
    else:
        occupancy = compute_occupancy(args.scene_dir, flat_pts, args.occ_dist_thr_ns, args.device,
                                       args.query_batch_size)
        print('occupied voxels: %d / %d (%.1f%%)' % (occupancy.sum(), n_total, 100 * occupancy.mean()))

    density_final = np.where(occupancy, density, 0.0)

    shape = tuple(args.resolution)
    density_vol = density_final.reshape(shape)
    occupancy_vol = occupancy.reshape(shape)

    out_path = args.output or os.path.join('outputs', scene_name, 'density', 'density_volume.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        density=density_vol.astype(np.float32),
        occupancy=occupancy_vol,
        x_centers=x_centers.astype(np.float32),
        y_centers=y_centers.astype(np.float32),
        z_centers=z_centers.astype(np.float32),
        aabb_min=np.array(aabb_min, dtype=np.float32),
        aabb_max=np.array(aabb_max, dtype=np.float32),
        voxel_size=voxel_size.astype(np.float32),
        coordinate_system=np.array(
            'nerfstudio-dataparser-original (same frame as utils.load_ns_point_cloud() output / '
            'transforms.json camera poses; NOT the internal [-1,1]^3 nerfstudio training box)'),
    )
    print('saved:', out_path)

    if args.report_mass:
        dV = float(np.prod(voxel_size))
        mass = float(density_final.sum() * dV)
        if args.assume_metric:
            print('predicted total mass (assuming transforms.json poses are metric): %.4f kg' % mass)
        else:
            print('metric scale unavailable; absolute mass is not valid '
                  '(raw sum*dV = %.4f in scene units^3 * kg/m^3, meaningless unless poses are metric; '
                  'pass --assume_metric after verifying scale)' % mass)


if __name__ == '__main__':
    main()
