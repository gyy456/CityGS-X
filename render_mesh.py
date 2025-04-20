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
import open3d as o3d
import numpy as np
import  copy
import matplotlib.pyplot as plt


def get_points_from_depth(fov_camera, depth, scale=1):
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    rays_d = fov_camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    R = torch.tensor(fov_camera.R).float().cuda()
    T = torch.tensor(fov_camera.T).float().cuda()
    pts = (pts-T)@R.transpose(-1,-2)
    return pts

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


def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0
def render_set(model_path, name, scene, iteration, views, gaussians, pipeline, background, max_depth=5.0, volume=None, use_depth_filter=False):
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
    depths_tsdf_fusion = []
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



        batched_image, batched_compute_locally,  batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict,  _ = render_final(batched_cameras, batched_screenspace_pkg, batched_strategies
            )

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
                # torchvision.utils.save_image(
                #     gt_image,
                #     os.path.join(gts_path, gt_camera.image_name + ".png"),
                # )

                depth_tsdf = depth.clone().squeeze(0)
                # depth_RED = visualize_scalars(torch.log(depth.squeeze(0) + 1e-8).detach().cpu())

                depth = depth.detach().cpu().numpy().squeeze(0)
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(depths_path,  gt_camera.image_name + ".png"), depth_color)

                # depth_RED = visualize_scalars(torch.log(depth + 1e-8).detach().cpu())

                # plt.imsave(os.path.join(depths_path, 'depth-' +(gt_camera.image_name + '.png') ), depth_RED)

                # normal = normal.permute(1,2,0)
                # normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                # normal = normal.detach().cpu().numpy()
                # normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                # torchvision.utils.save_image(
                #     torch.tensor(normal).permute(2,0,1)/255.0,
                #     os.path.join(render_normal_path, gt_camera.image_name + ".png"),
                # )
                # cv2.imwrite(os.path.join(render_normal_path,  gt_camera.image_name + ".png"), normal)
                depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
    

            gt_camera.original_image = None

        if generated_cnt == args.generate_num:
            break
    if utils.GLOBAL_RANK == 0:
        if volume is not None:
            depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
            for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
                ref_depth = depths_tsdf_fusion[idx].cuda()
                H, W = ref_depth.squeeze().shape
                if use_depth_filter and len(view.nearest_id) > 2:
                    nearest_world_view_transforms = scene.world_view_transforms[view.nearest_id]
                    num_n = nearest_world_view_transforms.shape[0]
                    ## compute geometry consistency mask
                    H, W = ref_depth.squeeze().shape

                    ix, iy = torch.meshgrid(
                        torch.arange(W), torch.arange(H), indexing='xy')
                    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                    pts = get_points_from_depth(view, ref_depth)
                    pts_in_nearest_cam = torch.matmul(nearest_world_view_transforms[:,None,:3,:3].expand(num_n,H*W,3,3).transpose(-1,-2), 
                                                    pts[None,:,:,None].expand(num_n,H*W,3,1))[...,0] + nearest_world_view_transforms[:,None,3,:3] # b, pts, 3

                    depths_nearest = depths_tsdf_fusion[view.nearest_id][:,None].cuda()
                    pts_projections = torch.stack(
                                    [pts_in_nearest_cam[...,0] * view.Fx / pts_in_nearest_cam[...,2] + view.Cx,
                                    pts_in_nearest_cam[...,1] * view.Fy / pts_in_nearest_cam[...,2] + view.Cy], -1).float()
                    d_mask = (pts_projections[..., 0] > 0) & (pts_projections[..., 0] < view.image_width) &\
                            (pts_projections[..., 1] > 0) & (pts_projections[..., 1] < view.image_height)

                    pts_projections[..., 0] /= ((view.image_width - 1) / 2)
                    pts_projections[..., 1] /= ((view.image_height - 1) / 2)
                    pts_projections -= 1
                    pts_projections = pts_projections.view(num_n, -1, 1, 2)
                    map_z = torch.nn.functional.grid_sample(input=depths_nearest,
                                                            grid=pts_projections,
                                                            mode='bilinear',
                                                            padding_mode='border',
                                                            align_corners=True
                                                            )[:,0,:,0]
                    
                    pts_in_nearest_cam[...,0] = pts_in_nearest_cam[...,0]/pts_in_nearest_cam[...,2]*map_z.squeeze()
                    pts_in_nearest_cam[...,1] = pts_in_nearest_cam[...,1]/pts_in_nearest_cam[...,2]*map_z.squeeze()
                    pts_in_nearest_cam[...,2] = map_z.squeeze()
                    pts_ = (pts_in_nearest_cam-nearest_world_view_transforms[:,None,3,:3])
                    pts_ = torch.matmul(nearest_world_view_transforms[:,None,:3,:3].expand(num_n,H*W,3,3), 
                                        pts_[:,:,:,None].expand(num_n,H*W,3,1))[...,0]

                    pts_in_view_cam = pts_ @ view.world_view_transform[:3,:3] + view.world_view_transform[None,None,3,:3]
                    pts_projections = torch.stack(
                                [pts_in_view_cam[...,0] * view.Fx / pts_in_view_cam[...,2] + view.Cx,
                                pts_in_view_cam[...,1] * view.Fy / pts_in_view_cam[...,2] + view.Cy], -1).float()
                    pixel_noise = torch.norm(pts_projections.reshape(num_n, H, W, 2) - pixels[None], dim=-1)
                    d_mask_all = d_mask.reshape(num_n,H,W) & (pixel_noise < 1.0) & (pts_in_view_cam[...,2].reshape(num_n,H,W) > 0.1)
                    d_mask_all = (d_mask_all.sum(0) > 1)
                    ref_depth[~d_mask_all] = 0
                # print('berfore mask max_depth:',ref_depth.max(),'min_depth',ref_depth.min())
                ref_depth[ref_depth>max_depth] = 0
                # print('after:',ref_depth.max(),'min_depth',ref_depth.min())
                # ref_depth = ref_depth * view.sky_mask[0]
                ref_depth = ref_depth.detach().cpu().numpy()
                
                pose = np.identity(4)
                pose[:3,:3] = view.R.transpose(-1,-2)
                pose[:3, 3] = view.T
                # manhattan = True
                # if manhattan:
                #     man_trans = create_man_rans((0,0,0), (0,0,-25))
                #     W2C = np.zeros((4, 4))
                #     W2C[:3, :3] = view.R.transpose(-1,-2)
                #     W2C[:3, -1] = view.T
                #     W2C[3, 3] = 1.0
                #     W2nC = W2C @ np.linalg.inv(man_trans)   # 相机跟着点云旋转平移后得到新的相机坐标系nC

                #     pose = W2nC

                color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".png"))
                depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
                volume.integrate(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                    pose)
def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    max_depth : float, 
    voxel_size : float, 
    use_depth_filter : bool
):
    with torch.no_grad():
        args = utils.get_args()
        # gaussians = GaussianModel(dataset.sh_degree)
        gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
        gaussians.eval()
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0*voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene,
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                max_depth=max_depth, 
                volume=volume, 
                use_depth_filter=use_depth_filter
            )

            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "possion_mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            mesh = clean_mesh(mesh)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)



        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene,
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
    parser.add_argument("--distributed_load", action="store_true")  
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    parser.add_argument("--max_depth", default=5, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument('--not_use_dpt_loss', action='store_false', help='Do not load dpt')
    parser.add_argument('--not_use_single_view_loss', action='store_false', help='Do not use single view loss')
    parser.add_argument('--not_use_multi_view_loss', action='store_false', help='Do not load gray image')      
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)




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
        args.max_depth,
        args.voxel_size,
        args.use_depth_filter
    )
