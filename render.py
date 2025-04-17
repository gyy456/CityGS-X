#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.distributed as dist
from scene import Scene, SceneDataset
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
    # preprocess3dgs_and_all2all,
    # render
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
import torchvision
from utils.general_utils import (
    safe_state,
    set_args,
    init_distributed,
    set_log_file,
    set_cur_iter,
)
from argparse import ArgumentParser
import debugpy
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from gaussian_renderer.workload_division import (
    start_strategy_final,
    DivisionStrategyHistoryFinal,
)
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import utils.general_utils as utils
import cv2
import numpy as np
import matplotlib.pyplot as plt





def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
    to_use = scalar_tensor.view(-1)
    while to_use.shape[0] > 2 ** 24:
        to_use = to_use[::2]

    mi = torch.quantile(to_use, 0.05)
    ma = torch.quantile(to_use, 0.95)

    scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    scalar_tensor = scalar_tensor.clamp_(0, 1)

    scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
    return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depths_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
           
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depths_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    dataset = SceneDataset(views)

    set_cur_iter(iteration)
    generated_cnt = 0

    num_cameras = len(views)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )
    progress_bar = tqdm(
        range(1, num_cameras + 1),
        desc="Rendering progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    for idx in range(1, num_cameras + 1, args.bsz):
        progress_bar.update(args.bsz)

        num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
        batched_cameras, _ = dataset.get_batched_cameras(num_camera_to_load, shuffle =False)
        batched_strategies, gpuid2tasks = start_strategy_final(
            batched_cameras, strategy_history
        )
        load_camera_from_cpu_to_all_gpu_for_eval(
            batched_cameras, batched_strategies, gpuid2tasks
        )
        batched_voxel_mask = [] 
        batched_nearest_voxel_mask= []
        batched_nearest_cameras= []
        for camera in batched_cameras:
            gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
            voxel_visible_mask = prefilter_voxel(camera, gaussians, pipeline, background)
            batched_voxel_mask.append(voxel_visible_mask)
            batched_nearest_voxel_mask.append(None)
            batched_nearest_cameras.append(None)
        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipeline,
            background,
            batched_voxel_mask=batched_voxel_mask,
            batched_strategies=batched_strategies,
            batched_nearest_cameras = batched_nearest_cameras,
            batched_nearest_voxel_mask = batched_nearest_voxel_mask,
            mode="test",
            return_plane = True
        )

        batched_image, batched_compute_locally,  batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, _ = render_final(batched_cameras, batched_screenspace_pkg, batched_strategies)

        for camera_id, (image, gt_camera, render_pkg) in enumerate(
            zip(batched_image, batched_cameras, batched_return_dict)
        ):
            depth = render_pkg["plane_depth"]
            normal = render_pkg["rendered_normal"]         
            actual_idx = idx + camera_id
            if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
                continue
            if generated_cnt == args.generate_num:
                break
            if args.l != -1 and args.r != -1:
                if actual_idx < args.l or actual_idx >= args.r:
                    continue

            generated_cnt += 1

            if (
                image is None or len(image.shape) == 0
            ):  # The image is not rendered locally.
                image = torch.zeros(
                    gt_camera.original_image.shape, device="cuda", dtype=torch.float32
                )
                depth = torch.zeros(
                    (1, gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]),
                    device="cuda",
                    dtype=torch.float32,
                )
                normal = torch.zeros(
                    (3, gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]),
                    device="cuda",
                    dtype=torch.float32,
                )

            if utils.DEFAULT_GROUP.size() > 1:
                torch.distributed.all_reduce(
                    image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                )
                torch.distributed.all_reduce(
                                depth, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )
                torch.distributed.all_reduce(
                    normal, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                )

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

            if utils.GLOBAL_RANK == 0:
                torchvision.utils.save_image(
                    image,
                    os.path.join(render_path, gt_camera.image_name + ".png"),
                )
                torchvision.utils.save_image(
                    gt_image,
                    os.path.join(gts_path, gt_camera.image_name + ".png"),
                )
                depth_RED = visualize_scalars(torch.log(depth.squeeze(0) + 1e-8).detach().cpu())

                depth = depth.detach().cpu().numpy().squeeze(0)
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(depths_path,  gt_camera.image_name + ".png"), depth_color)

                # depth_RED = visualize_scalars(torch.log(depth + 1e-8).detach().cpu())

                plt.imsave(os.path.join(depths_path, 'depth-' +(gt_camera.image_name + '.png') ), depth_RED)
                normal = normal.permute(1,2,0)
                normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                normal = normal.detach().cpu().numpy()
                normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                # torchvision.utils.save_image(
                #     torch.tensor(normal).permute(2,0,1)/255.0,
                #     os.path.join(render_normal_path, gt_camera.image_name + ".png"),
                # )
                cv2.imwrite(os.path.join(render_normal_path,  gt_camera.image_name + ".png"), normal)
    

            gt_camera.original_image = None

        if generated_cnt == args.generate_num:
            break


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        args = utils.get_args()
        # gaussians = GaussianModel(dataset.sh_degree)
        gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.eval()
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--distributed_load", action="store_true")  # TODO: delete this.
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    parser.add_argument('--not_use_dpt_loss', action='store_false', help='Do not load dpt')
    parser.add_argument('--not_use_single_view_loss', action='store_false', help='Do not use single view loss')
    parser.add_argument('--not_use_multi_view_loss', action='store_false', help='Do not load gray image')    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # rank = int(os.environ.get("LOCAL_RANK", 0))
    # # 在调用分布式初始化之前初始化调试器
    # port = 5678 + rank  # 每个进程使用不同的端口
    # # if rank==0:
    # debugpy.listen(('0.0.0.0', port))  # 启动调试器并监听不同的端口
    # print(f"Process {rank} waiting for debugger to attach on port {port}...")
    # debugpy.wait_for_client()  # 程序在这里暂停，直到调试器连接


    init_distributed(args)
    # This script only supports single-gpu rendering.
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.


    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)

    print_all_args(args, log_file)

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        args.skip_train,
        args.skip_test,
    )
