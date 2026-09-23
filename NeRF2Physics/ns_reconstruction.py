import os
import subprocess
import shutil
from utils import get_last_file_in_folder, get_scenes_list
from arguments import get_args


def move_files_to_folder(source_dir, target_dir):
    for file in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))


if __name__ == '__main__':
    
    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    for scene in scenes: 
        base_dir = os.path.join(scenes_dir, scene, 'ns')

        # Calling ns-train
        # NOTE (local-density-volume branch): --pipeline.datamanager.camera-optimizer.mode
        # was renamed to --pipeline.model.camera-optimizer.mode between the nerfstudio
        # version this repo originally targeted and nerfstudio 1.1.5 (pinned in pixi.toml).
        # Same flag, same semantics (camera pose optimization disabled) -- just relocated.
        result = subprocess.run([
            'ns-train', 'nerfacto',
            '--data', os.path.join(scenes_dir, scene),
            '--output_dir', base_dir,
            '--vis', args.vis_mode,
            '--project_name', args.project_name,
            '--experiment_name', scene,
            '--max_num_iterations', str(args.training_iters),
            '--pipeline.model.background-color', 'random',
            '--pipeline.model.camera-optimizer.mode', 'off',
            '--pipeline.model.proposal-initial-sampler', args.proposal_initial_sampler,
            '--pipeline.model.near-plane', str(args.near_plane),
            '--pipeline.model.far-plane', str(args.far_plane),
            '--steps-per-eval-image', '10000',
        ] + (['--machine.seed', str(args.nerf_seed)] if args.nerf_seed >= 0 else []))
        result.check_returncode()

        ns_dir = get_last_file_in_folder(os.path.join(base_dir, '%s/nerfacto' % scene))

        # Copying dataparser_transforms (contains scale)
        result = subprocess.run([
            'scp', '-r', 
            os.path.join(ns_dir, 'dataparser_transforms.json'), 
            os.path.join(base_dir, 'dataparser_transforms.json')
        ])
        result.check_returncode()

        # Calling ns-export pcd
        # NOTE (local-density-volume branch): --use-bounding-box/--bounding-box-min/-max
        # were replaced by an oriented-bounding-box (obb) parameterization in nerfstudio
        # 1.1.5. obb-scale is the FULL box extent (nerfstudio's OrientedBox.within() tests
        # against +/-S/2), so obb-scale = bbox_size reproduces the exact same axis-aligned
        # cube centered at the origin that --bounding-box-min/-max used to describe.
        result = subprocess.run([
            'ns-export', 'pointcloud',
            '--load-config', os.path.join(ns_dir, 'config.yml'),
            '--output-dir', base_dir,
            '--num-points', str(args.num_points),
            '--remove-outliers', 'True',
            '--normal-method', 'open3d',
            '--obb-center', '0', '0', '0',
            '--obb-rotation', '0', '0', '0',
            '--obb-scale', str(args.bbox_size), str(args.bbox_size), str(args.bbox_size),
        ])
        result.check_returncode()

        # Calling ns-render
        result = subprocess.run([
            'ns-render', 'dataset',
            '--load-config', os.path.join(ns_dir, 'config.yml'),
            '--output-path', os.path.join(base_dir, 'renders'),
            '--rendered-output-names', 'raw-depth',
            '--split', 'train+test',
        ])
        result.check_returncode()

        # Collect all depths in one folder
        os.makedirs(os.path.join(base_dir, 'renders', 'depth'), exist_ok=True)
        move_files_to_folder(os.path.join(base_dir, 'renders', 'test', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))
        move_files_to_folder(os.path.join(base_dir, 'renders', 'train', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))

