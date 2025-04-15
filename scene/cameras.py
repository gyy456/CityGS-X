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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import get_args, get_log_file
import utils.general_utils as utils
import time
import os, cv2

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        image_height,
        image_width,
        uid,
        depth_params=None,
        image_path = None,
        depth_reliables = False,
        invdepthmaps = None,
        normal_mask = None, 
        noraml_gt = None,
        depth_mask = None,
        resized_image_gray = None ,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        # self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.image_path = image_path

        self.image_name = image_name

        args = get_args()
        log_file = get_log_file()

        if args.time_image_loading:
            start_time = time.time()

        if (
            (
                args.local_sampling
                and args.distributed_dataset_storage
                and utils.GLOBAL_RANK == uid % utils.WORLD_SIZE
            )
            or (
                not args.local_sampling
                and args.distributed_dataset_storage
                and utils.LOCAL_RANK == 0
            )
            or (not args.distributed_dataset_storage)
        ):
            # load to cpu
            if image is not None:
                self.original_image_backup = image.contiguous()
            if args.preload_dataset_to_gpu:
                self.original_image_backup = self.original_image_backup.to("cuda")
            self.image_width = image_width
            self.image_height = image_height
            if resized_image_gray is not None:
                if not args.distributed_dataset_storage:
                    self.image_gray = resized_image_gray.contiguous()
                else:
                    self.image_gray_backup = resized_image_gray.clamp(0.0, 1.0).contiguous()
            if invdepthmaps is not None:
                if not args.distributed_dataset_storage:
                    self.invdepthmap = invdepthmaps.contiguous()
                    self.invdepthmap_backup = None
                else:
                    self.invdepthmap_backup = invdepthmaps.contiguous()
                    self.invdepthmap = None
            # if depth_mask is not None:
            #     self.depth_mask = depth_mask.contiguous()
            #     if invdepthmaps is not None and depth_mask is not None:
            #         # Use depth_mask to set invdepthmaps to 0 where mask is 0
                    # self.invdepthmap_backup[self.depth_mask.unsqueeze(0) == 0] = 0
        else:
            self.original_image_backup = None
            self.image_height, self.image_width = utils.get_img_size()

        if args.time_image_loading:
            log_file.write(f"Image processing in {time.time() - start_time} seconds\n")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.world_view_transform_backup = self.world_view_transform.clone().detach()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # if depth is None:
        # self.invdepthmap  = None
        depth_path = self.image_path.replace('images','depths')
        depth_path = depth_path.replace('jpg','png')
        # if os.path.exists(depth_path):
        #     invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
        #     self.invdepthmap = cv2.resize(invdepthmap, self.resolution)
        #     self.invdepthmap[self.invdepthmap < 0] = 0
        self.depth_reliable = True
            # if depth_params is not None:
            #     if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
            #         self.depth_reliable = False
            #         # self.depth_mask *= 0
                
            #     if depth_params["scale"] > 0:
            #         self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]  #统一尺度
            # if self.invdepthmap.ndim != 2:
            #     self.invdepthmap = self.invdepthmap[..., 0]
        # self.invdepthmap = invdepthmaps
        # if args.preload_dataset_to_gpu and depth_reliables is not None:
        #     self.invdepthmap = self.invdepthmap.to("cuda")
        self.normal_mask = normal_mask
        self.noraml_gt = noraml_gt
        # self.depth_mask = depth_mask
        self.ncc_scale = 1
        # self.image_gray = None

    def get_camera2world(self):
        return self.world_view_transform_backup.t().inverse()
    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    def update(self, dx, dy, dz):
        # Update the position of this camera pose. TODO: support updating rotation of camera pose.
        with torch.no_grad():
            c2w = self.get_camera2world()
            c2w[0, 3] += dx
            c2w[1, 3] += dy
            c2w[2, 3] += dz

            t_prime = c2w[:3, 3]
            self.T = (-c2w[:3, :3].t() @ t_prime).cpu().numpy()
            # import pdb; pdb.set_trace()

            self.world_view_transform = (
                torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
                .transpose(0, 1)
                .cuda()
            )
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda()
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
