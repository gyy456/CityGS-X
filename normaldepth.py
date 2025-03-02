"""Check the normal consistency between the normals from the pre-trained model and the normals from the depth map."""


import numpy as np
from pathlib import Path
import os
import cv2
# from rich.console import Console
# from rich.progress import track
from utils.point_utils import backproject
from utils.point_utils import compute_angle_between_normals
import torch
import sys
import json
from utils.general_utils import safe_state, init_distributed
import utils.general_utils as utils
from argparse import ArgumentParser
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
import train_internal
import debugpy
from tqdm import tqdm
from scene import Scene_precess
from gaussian_renderer import render_normal

# CONSOLE = Console(width=120)
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
SCALE_FACTOR = 0.001


def depth_path_to_array(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> np.ndarray:
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    return depth


# @dataclass
def single_view_process(dataset_args, opt_args, pipe_args, args):
    """
    Check the normal consistency between the normals from the pre-trained model and the normals from the depth map,
    generate depth confidence mask based on the normal consistency.
    """

    # model_path: Path = Path("dataset/room_datasets/vr_room/iphone/long_capture")
    """transforms file name"""
    angle_treshold = 20.0
    """Angle treshold for normal consistency check. Differences bigger than this threshold will be considered as inconsistent and masked."""


    output_normal_path = os.path.join(args.model_path, "depth_normals")
    output_mask_path = os.path.join(args.model_path, "depth_normals_mask")

    os.makedirs(output_normal_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)


    with torch.no_grad():
        scene = Scene_precess(args)
    # Init dataset
    # set_rays_od(scene.getTrainCameras())
    # if utils.GLOBAL_RANK == 0:
    train_dataset = scene.getTrainCameras()




    # for i in tqdm(range(num_frames), description="Processing frames..."):
    for i, view in enumerate(tqdm(train_dataset, desc = "Processing frames...")):
        # c2w_ref = np.array(sorted_frames[i]["transform_matrix"])



        # if c2w_ref.shape[0] != 4:
        #     c2w_ref = np.concatenate([c2w_ref, np.array([[0, 0, 0, 1]])], axis=0)
        
        w2c = view.world_view_transform
        c2w_ref = w2c.t().inverse()
        # c2w_ref = c2w_ref @ OPENGL_TO_OPENCV
        fx = view.Fx
        fy = view.Fy
        cx = view.Cx
        cy = view.Cy
        h = view.image_width
        w = view.image_height
        depth_i = 1/view.invdepthmap.squeeze().cpu().numpy()
        # depth_i = cv2.resize(depth_i, (w, h), interpolation=cv2.INTER_NEAREST)
        means3d, image_coords = backproject(
            depths=depth_i,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_size=(w, h),
            c2w=c2w_ref,
        )
        cam_center = c2w_ref[:3, 3]

        # normals_from_depth = np.array(pcd.normals)
        normals_from_depth = render_normal(view, depth_i) 


        # check normal direction: if ray dir and normal angle is smaller than 90, reverse normal
        ray_dir = means3d - cam_center.reshape(1, 3)
        normal_dir_not_correct = (ray_dir * normals_from_depth).sum(axis=-1) > 0
        normals_from_depth[normal_dir_not_correct] = -normals_from_depth[
            normal_dir_not_correct
        ]

        normals_from_depth = normals_from_depth.reshape(h, w, 3)

        cv2.imwrite(
            os.path.join(output_normal_path, view.image_name +'.png'),
            ((normals_from_depth + 1) / 2 * 255).astype(np.uint8),
        )


        # mono_normals are saved in [0,1] range, but need to be converted to [-1,1]
        mono_normal = view.normal_gt

        # convert mono normals from camera frame to world coordinate frame, same as normals_from_depth
        w2c = np.linalg.inv(c2w_ref)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        mono_normal = mono_normal.reshape(-1, 3).transpose(1, 0)
        mono_normal = (R @ mono_normal).T
        mono_normal = mono_normal / np.linalg.norm(
            mono_normal, axis=1, keepdims=True
        )
        mono_normal = mono_normal.reshape(h, w, 3)

        # compute angle between normals_from_depth and mono_normal
        degree_map = compute_angle_between_normals(normals_from_depth, mono_normal)
        mask = (degree_map > angle_treshold).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_mask_path, view.image_name +'.png'),
            mask * 255.0,
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
    args = parser.parse_args(sys.argv[1:])

    # Set up distributed training

    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 在调用分布式初始化之前初始化调试器
    # port = 5690 + rank  # 每个进程使用不同的端口
    # # if rank==0:
    # debugpy.listen(('0.0.0.0', port))  # 启动调试器并监听不同的端口
    # print(f"Process {rank} waiting for debugger to attach on port {port}...")
    # debugpy.wait_for_client()  # 程序在这里暂停，直到调试器连接
    init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    args = utils.get_args()

    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(
            group=utils.DEFAULT_GROUP
        )  # log_folder is created before other ranks start writing log.
    if utils.GLOBAL_RANK == 0:
        with open(args.log_folder + "/args.json", "w") as f:
            json.dump(vars(args), f)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)



    single_view_process(
        lp.extract(args), op.extract(args), pp.extract(args), args
    )

