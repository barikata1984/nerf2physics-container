"""
Combine visualization.py's separate RGB / material-segmentation / density
panels (and its legend) into one side-by-side comparison figure, in the style
of the paper's Fig. 1 (Input RGB -> Materials -> Mass Density).

visualization.py itself only writes the individual panel PNGs; this script
does not call it, it just composes files it already produced.

Example:
    pixi run python scripts/make_comparison_figure.py \\
        --viz_dir viz/sledgehammer --viz_save_name sledgehammer
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root, for --density_cool

import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description='Compose an RGB/Materials/Density comparison figure.')
    p.add_argument('--viz_dir', type=str, required=True, help="visualization.py's --viz_save_name output dir")
    p.add_argument('--viz_save_name', type=str, required=True, help='must match visualization.py\'s own arg')
    p.add_argument('--property_name', type=str, default='density')
    p.add_argument('--output', type=str, default=None, help='default: <viz_dir>/<viz_save_name>_comparison.png')
    p.add_argument('--density_cool', action='store_true',
                    help="add a 4th panel: density re-colored with matplotlib 'cool' after min-max normalization "
                         "over the scene's own predicted range (re-runs the CLIP query; needs --scene_dir)")
    p.add_argument('--scene_dir', type=str, default=None)
    p.add_argument('--mats_load_name', type=str, default='info_new')
    p.add_argument('--pt_size', type=float, default=3)
    return p.parse_args()


def render_density_cool(args, view_hw):
    """Same query + camera as visualization.py, but colors = cool(min-max normalized density)."""
    import json, types, numpy as np, torch, open_clip, open3d as o3d, matplotlib as mpl
    from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
    from predict_property import predict_physical_property_query
    from utils import load_ns_point_cloud, parse_transforms_json, load_images
    from visualization import render_pcd, composite_and_save
    sd = args.scene_dir
    a = types.SimpleNamespace(device='cuda', mats_load_name=args.mats_load_name, feature_load_name='ps56',
                              property_name=args.property_name, temperature=0.1, show_mat_seg=0)
    model, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    model.to('cuda'); tok = open_clip.get_tokenizer(CLIP_BACKBONE)
    q = torch.Tensor(load_ns_point_cloud(sd + '/ns/point_cloud.ply', sd + '/ns/dataparser_transforms.json', ds_size=None)).cuda()
    r = predict_physical_property_query(a, q, sd, model, tok, return_all=True)
    vals = r['query_pred_vals'].mean(1); lo, hi = float(vals.min()), float(vals.max())
    colors = mpl.colormaps['cool']((vals - lo) / (hi - lo))[:, :3]
    with open(sd + '/' + args.mats_load_name + '.json') as f:
        view = int(json.load(f).get('idx_to_caption', 0))
    w2cs, K = parse_transforms_json(sd + '/transforms.json', return_w2c=True); w2c = w2cs[view]; w2c[[1, 2]] *= -1
    img = load_images(sd + '/images')[view] / 255.
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(q.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    render = render_pcd(pcd, w2c, K, hw=img.shape[:2], pt_size=args.pt_size)
    out = os.path.join(args.viz_dir, f'{args.viz_save_name}_{args.property_name}_cool.png')
    composite_and_save(img, render, 0.2, out)
    return Image.open(out), lo, hi


def shared_crop_box(imgs, bg=255, pad=20):
    """One crop box, shared across all images, covering the union of their
    non-background content. All of visualization.py's renders share the same
    camera/resolution, so they're pixel-aligned before cropping -- cropping
    each image to its OWN content independently (as an earlier version of
    this script did) breaks that alignment: a panel with slightly less
    reconstructed content (e.g. the steel head, poorly reconstructed by the
    500-iteration debug NeRF) gets a different crop box than the others, so
    the same object part lands at different pixel positions per panel."""
    import numpy as np
    x0 = y0 = float('inf')
    x1 = y1 = 0
    for img in imgs:
        arr = np.array(img.convert('RGB'))
        mask = np.any(arr != bg, axis=-1)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x0, y0 = min(x0, xs.min()), min(y0, ys.min())
        x1, y1 = max(x1, xs.max()), max(y1, ys.max())
    h, w = np.array(imgs[0]).shape[:2]
    return (max(int(x0) - pad, 0), max(int(y0) - pad, 0),
            min(int(x1) + pad, w), min(int(y1) + pad, h))


def main():
    args = parse_args()
    n = args.viz_save_name
    rgb = Image.open(os.path.join(args.viz_dir, f'{n}_rgb.png'))
    seg = Image.open(os.path.join(args.viz_dir, f'{n}_seg.png'))
    density = Image.open(os.path.join(args.viz_dir, f'{n}_{args.property_name}.png'))
    legend = Image.open(os.path.join(args.viz_dir, f'{n}_legend.png'))

    panels = [rgb, seg, density]
    titles = ['Input RGB', 'Materials', f'{args.property_name.capitalize()} (inferno, 500-3500 kg/m^3)']
    cool = None
    if args.density_cool:
        assert args.scene_dir, '--density_cool needs --scene_dir'
        cool, lo, hi = render_density_cool(args, rgb.size)
        panels.append(cool); titles.append(f'{args.property_name.capitalize()} (cool, min-max {lo:.0f}-{hi:.0f})')

    box = shared_crop_box(panels)
    panels = [im.crop(box) for im in panels]

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.5 * len(panels) + 2, 5),
                             gridspec_kw={'width_ratios': [1] * len(panels) + [0.5]})
    for ax, img, title in zip(axes[:-1], panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis('off')
    axes[-1].imshow(legend)
    axes[-1].axis('off')

    out_path = args.output or os.path.join(args.viz_dir, f'{n}_comparison.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out_path)


if __name__ == '__main__':
    main()
