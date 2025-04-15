import os
import torch
import json
import utils.general_utils as utils
import torch.distributed as dist
import cv2
import numpy as np
import sys
import utils.general_utils as utils

from scene import Scene_precess
from tqdm import tqdm
from utils.general_utils import safe_state, init_distributed
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


def sync_model_with_rank0(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, 0)

def get_points_depth_in_depth_map(fov_camera, depth, points_in_camera_space, scale=1):
    st = max(int(scale/2)-1,0)
    depth_view = depth[None,:,st::scale,st::scale]
    W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
    depth_view = depth_view[:H, :W]
    pts_projections = torch.stack(
                    [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                        points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
    mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
            (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    return map_z, mask
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

def process(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools
    # Init parameterized scene
    
    with torch.no_grad():
        scene = Scene_precess(args)
    # Init dataset
    # set_rays_od(scene.getTrainCameras())
    # if utils.GLOBAL_RANK == 0:
    train_dataset = scene.getTrainCameras()

    depths_tsdf_fusion = []
    for camera in train_dataset:
        if camera.invdepthmap_backup != None:
            depths_tsdf_fusion.append(1/camera.invdepthmap_backup.squeeze())
        else:
            depths_tsdf_fusion.append(1/camera.invdepthmap.squeeze())
    depths_tsdf_fusion_1 = torch.stack(depths_tsdf_fusion)
    for idx, view in enumerate(tqdm(train_dataset, desc = "preprocess")):
        ref_depth = depths_tsdf_fusion[idx].cuda()
        if len(view.nearest_id) > 0:
                
            nearest_world_view_transforms = scene.world_view_transforms[view.nearest_id]
            num_n = nearest_world_view_transforms.shape[0]
            H, W = ref_depth.squeeze().shape

            ix, iy = torch.meshgrid(
                torch.arange(W), torch.arange(H), indexing='xy')
            pixels = torch.stack([ix, iy], dim=-1).float().to("cuda")

            pts = get_points_from_depth(view, ref_depth)
            pts_in_nearest_cam = torch.matmul(nearest_world_view_transforms[:,None,:3,:3].expand(num_n,H*W,3,3).transpose(-1,-2), 
                                                pts[None,:,:,None].expand(num_n,H*W,3,1))[...,0] + nearest_world_view_transforms[:,None,3,:3] # b, pts, 3

            # depths_tsdf_fusion = torch.stack(depths_tsdf_fusion)
            depths_nearest = depths_tsdf_fusion_1[view.nearest_id][:,None].cuda()
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
            d_mask_all = d_mask.reshape(num_n,H,W) & (pixel_noise < args.pixel_thred) & (pts_in_view_cam[...,2].reshape(num_n,H,W) > 0.0)
            d_mask_all = (d_mask_all.sum(0) > 1)   
            ref_depth = ref_depth.detach().cpu().numpy()
            depth_i = (ref_depth - ref_depth.min()) / (ref_depth.max() - ref_depth.min() + 1e-20)
            # depth_i[~d_mask_all] = 0

            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            d_mask_all = d_mask_all.cpu().numpy()
            depth_color[~d_mask_all] = 0 
            # torchvision.utils.save_image(
            #     torch.tensor(depth_color).permute(2,0,1)/255.0,
            #     os.path.join(depths_path, gt_camera.image_name + ".png"),
            # )
            cv2.imwrite(os.path.join(args.model_path, view.image_name + ".png"), depth_color)
            # mask = (d_mask_all).astype(np.uint8)
            # cv2.imwrite(
            #     os.path.join(args.model_path, view.image_name +'.png'),
            #     mask * 255.0,
            # )





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
    parser.add_argument('--not_use_dpt_loss', action='store_true', help='Do not use DPT loss')
    parser.add_argument('--not_use_single_view_loss', action='store_true', help='Do not use single view loss')
    parser.add_argument('--not_use_multi_view_loss', action='store_true', help='Do not use multi view loss')      
    parser.add_argument('--pixel_thred', type=float, default=1, help='pixel thred')
    args = parser.parse_args(sys.argv[1:])

    # Set up distributed training

    rank = int(os.environ.get("LOCAL_RANK", 0))
    
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

    # Initialize log file and print all args
    log_file = open(
        args.log_folder
        + "/python_ws="
        + str(utils.WORLD_SIZE)
        + "_rk="
        + str(utils.GLOBAL_RANK)
        + ".log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    process(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )

    # All done
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    utils.print_rank_0("\nTraining complete.")