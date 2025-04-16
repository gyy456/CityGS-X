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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import utils.general_utils as utils
import torch.distributed as dist
import time
import math
from scene.embedding import Embedding
from einops import repeat
from torch_scatter import scatter_max
from pytorch3d.transforms import quaternion_to_matrix
lr_scale_fns = {
    "linear": lambda x: x,
    "sqrt": lambda x: np.sqrt(x),
}
from functools import reduce

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, feat_dim: int=32, 
                 n_offsets: int=5, 
                 fork: int=2,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_level: bool = False,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1):
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive

        # Octree
        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        
        self.start_step = 0
        self.end_step = 0

        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0
    
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(self.view_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()
            
        self.mlp_opacity = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Tanh()
            ).cuda()
        
        self.mlp_cov = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 7*self.n_offsets),
            ).cuda()
        
        self.mlp_color = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()



    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,  # TODO: deal with self.send_to_gpui_cnt
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = (
            xyz_gradient_accum  # TODO: deal with self.send_to_gpui_cnt
        )
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)
    

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask


    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        is_training = self.get_color_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)
        

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)


    def octree_sample(self, data, init_pos):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        # print(f'before unique: {self._level.shape[0]}')
        # self.positions, index = torch.unique(self.positions, dim=0, return_inverse = True)
        # self._level = self._level[index]
        # print(f'after unique: {self._level.shape[0]}')
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")


    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)


    def create_from_pcd(self, points, spatial_lr_scale):
        log_file = utils.get_log_file()
        # loading could replicated on all ranks.

        #octree
        fused_point_cloud = (
            torch.tensor(points).float().cuda()
        )  # It is not contiguous
        points = fused_point_cloud.contiguous()  # Now it's contiguous
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min

        args = utils.get_args()

        if self.base_layer < 0:
            default_voxel_size = args.default_voxel_size
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}') 
        print(f'LOD Levels: {self.levels}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        self.positions = self.positions.contiguous()
        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))

        # random_idx = torch.zeros(
        #     self.positions.shape[0],
        #     device="cuda",
        #     dtype=int
        # )


        # # 将 random_idx 广播到其他进程
        #     # dist.broadcast(random_idx, src=0)
        # if utils.DEFAULT_GROUP.size() > 1:
        #     dist.broadcast(random_idx, src=0, group=utils.DEFAULT_GROUP)

        # anchor = self.positions[random_idx]
        # self._level = self._level[random_idx]
        # scales = scales[random_idx]
        anchor = self.positions



        args = utils.get_args()
        if (
            args.gaussians_distribution
        ):  # shard 3dgs storage across all GPU including dp and mp groups.
            shard_world_size = utils.DEFAULT_GROUP.size()
            shard_rank = utils.DEFAULT_GROUP.rank()

            the_idx = shard_rank
            random_idx = torch.arange(self.positions.shape[0], device="cuda")
            mask = random_idx % shard_world_size == the_idx
            index = random_idx[mask]

            # anchor_ind_l, anchor_ind_r = utils.get_local_chunk_l_r(
            #     anchor.shape[0], shard_world_size, shard_rank
            # )
            # anchor = anchor[anchor_ind_l:anchor_ind_r].contiguous()
            # offsets = offsets[anchor_ind_l:anchor_ind_r].contiguous()
            # anchors_feat = anchors_feat[anchor_ind_l:anchor_ind_r].contiguous()
            # scales = scales[anchor_ind_l:anchor_ind_r].contiguous()
            # rots = rots[anchor_ind_l:anchor_ind_r].contiguous()
            # opacities = opacities[anchor_ind_l:anchor_ind_r].contiguous()
            # self._level = self._level[anchor_ind_l:anchor_ind_r].contiguous()


            anchor = anchor[index].contiguous()
            offsets = offsets[index].contiguous()
            anchors_feat = anchors_feat[index].contiguous()
            scales = scales[index].contiguous()
            rots = rots[index].contiguous()
            opacities = opacities[index].contiguous()
            self._level = self._level[index].contiguous()



            log_file.write(
                "rank: {}, Number of anchor points: {}\n".format(
                    utils.GLOBAL_RANK, anchor.shape[0]
                )
            )

        self._anchor = nn.Parameter(anchor.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.zeros(self._anchor.shape[0], dtype=torch.bool, device="cuda")


    def all_parameters(self):
        return [
            self._anchor,
            self._offset,
            self._anchor_feat,
            self._opacity,
            self._scaling,
            self._rotation,
            # self.mlp_opacity.parameters(),
            # self.mlp_cov.parameters(),
            # self.mlp_color.parameters(),
        ]
    # def get_normal(self, view_cam):
    #     normal_global = self.get_smallest_axis()
    #     gaussian_to_cam_global = view_cam.camera_center - self._xyz
    #     neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
    #     normal_global[neg_mask] = -normal_global[neg_mask]
    #     return normal_global
    # def get_smallest_axis(self, return_idx=False):
    #     rotation_matrices = self.get_rotation_matrix()
    #     smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    #     smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    #     if return_idx:
    #         return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    #     return smallest_axis.squeeze(dim=2)
    

    # def get_rotation_matrix(self):
    #     return quaternion_to_matrix(self.get_rotation)
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

        utils.check_initial_gpu_memory_usage("after training_setup")

    def log_gaussian_stats(self):
        # log the statistics of the gaussian model
        # number of total 3dgs on this rank
        num_3dgs = self._xyz.shape[0]
        # average size of 3dgs
        avg_size = torch.mean(torch.max(self.get_scaling, dim=1).values).item()
        # average opacity
        avg_opacity = torch.mean(self.get_opacity).item()
        stats = {
            "num_3dgs": num_3dgs,
            "avg_size": avg_size,
            "avg_opacity": avg_opacity,
        }

        # get the exp_avg, exp_avg_sq state for all parameters
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    exp_avg_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg"], dim=-1)
                    ).item()
                    exp_avg_sq_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg_sq"], dim=-1)
                    ).item()
        return stats, exp_avg_dict, exp_avg_sq_dict

    def sync_gradients_for_replicated_3dgs_storage(self, batched_screenspace_pkg):
        args = utils.get_args()

        if "visible_count" in args.grad_normalization_mode:
            # allgather visibility filder from all dp workers, so that each worker contains the visibility filter of all data points.
            batched_locally_preprocessed_visibility_filter_int = [
                x.int()
                for x in batched_screenspace_pkg[
                    "batched_locally_preprocessed_visibility_filter"
                ]
            ]
            sum_batched_locally_preprocessed_visibility_filter_int = torch.sum(
                torch.stack(batched_locally_preprocessed_visibility_filter_int), dim=0
            )
            batched_screenspace_pkg[
                "sum_batched_locally_preprocessed_visibility_filter_int"
            ] = sum_batched_locally_preprocessed_visibility_filter_int

        if args.sync_grad_mode == "dense":
            sync_func = sync_gradients_densely
        elif args.sync_grad_mode == "sparse":
            sync_func = sync_gradients_sparsely
        elif args.sync_grad_mode == "fused_dense":
            sync_func = sync_gradients_fused_densely
        elif args.sync_grad_mode == "fused_sparse":
            sync_func = sync_gradients_fused_sparsely
        else:
            assert False, f"sync_grad_mode {args.sync_grad_mode} not supported."

        if not args.gaussians_distribution and utils.DEFAULT_GROUP.size() > 1:
            sync_func(self, utils.DEFAULT_GROUP)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    def save_ply(self, path):

        args = utils.get_args()
        utils.log_cpu_memory_usage("start save_ply")
        group = utils.DEFAULT_GROUP
        if args.gaussians_distribution and not args.distributed_save:
            # gather all gaussians at rank 0
            def gather_uneven_tensors(tensor):
                # gather size of tensors on different ranks
                tensor_sizes = torch.zeros(
                    (group.size()), dtype=torch.int, device="cuda"
                )
                tensor_sizes[group.rank()] = tensor.shape[0]
                dist.all_reduce(tensor_sizes, op=dist.ReduceOp.SUM)
                # move tensor_sizes to CPU and convert to int list
                tensor_sizes = tensor_sizes.cpu().numpy().tolist()

                # NOTE: Internal implementation of gather could not gather tensors of different sizes.
                # So, I do not use dist.gather(tensor, dst=0) but use dist.send(tensor, dst=0) and dist.recv(tensor, src=i) instead.

                # gather tensors on different ranks using grouped send/recv
                gathered_tensors = []
                if group.rank() == 0:
                    for i in range(group.size()):
                        if i == group.rank():
                            gathered_tensors.append(tensor)
                        else:
                            tensor_from_rk_i = torch.zeros(
                                (tensor_sizes[i],) + tensor.shape[1:],
                                dtype=tensor.dtype,
                                device="cuda",
                            )
                            dist.recv(tensor_from_rk_i, src=i)
                            gathered_tensors.append(tensor_from_rk_i)
                    gathered_tensors = torch.cat(gathered_tensors, dim=0)
                else:
                    dist.send(tensor, dst=0)
                # concatenate gathered tensors

                return (
                    gathered_tensors if group.rank() == 0 else tensor
                )  # only return gather tensors at rank 0
            
            anchor = gather_uneven_tensors(self._anchor).detach().cpu().numpy()
            anchor_feats = gather_uneven_tensors(self._anchor_feat).detach().cpu().numpy()
            offsets = gather_uneven_tensors(self._offset)
            offsets = offsets.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = gather_uneven_tensors(self._opacity).detach().cpu().numpy()
            scales = gather_uneven_tensors(self._scaling).detach().cpu().numpy()
            rots = gather_uneven_tensors(self._rotation).detach().cpu().numpy()
            levels = gather_uneven_tensors(self._level).detach().cpu().numpy()
            extra_levels = gather_uneven_tensors(self._extra_level).unsqueeze(dim=1).detach().cpu().numpy()
            infos = np.zeros_like(levels, dtype=np.float32)
            infos[0, 0] = self.voxel_size
            infos[1, 0] = self.standard_dist

            if group.rank() != 0:
                return
        elif args.gaussians_distribution and args.distributed_save:
            assert (
                utils.DEFAULT_GROUP.size() > 1
            ), "distributed_save should be used with more than 1 rank."
            anchor = self._anchor.detach().cpu().numpy()
            levels = self._level.detach().cpu().numpy()
            extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
            infos = np.zeros_like(levels, dtype=np.float32)
            infos[0, 0] = self.voxel_size
            infos[1, 0] = self.standard_dist

            anchor_feats = self._anchor_feat.detach().cpu().numpy()
            offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scales = self._scaling.detach().cpu().numpy()
            rots = self._rotation.detach().cpu().numpy()
            if path.endswith(".ply"):
                path = (
                    path[:-4]
                    + "_rk"
                    + str(utils.GLOBAL_RANK)
                    + "_ws"
                    + str(utils.WORLD_SIZE)
                    + ".ply"
                )
        elif not args.gaussians_distribution:
            if group.rank() != 0:
                return
            anchor = self._anchor.detach().cpu().numpy()
            levels = self._level.detach().cpu().numpy()
            extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
            infos = np.zeros_like(levels, dtype=np.float32)
            infos[0, 0] = self.voxel_size
            infos[1, 0] = self.standard_dist

            anchor_feats = self._anchor_feat.detach().cpu().numpy()
            offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scales = self._scaling.detach().cpu().numpy()
            rots = self._rotation.detach().cpu().numpy()
            if path.endswith(".ply"):
                path = (
                    path[:-4]
                    + "_rk"
                    + str(utils.GLOBAL_RANK)
                    + "_ws"
                    + str(utils.WORLD_SIZE)
                    + ".ply"
                )

        mkdir_p(os.path.dirname(path))

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        # remark: max_radii2D, xyz_gradient_accum and denom are not saved here; they are save elsewhere.

        np.savez(os.path.join(os.path.dirname(path), 'additional_attributes.npz'),init_pos=self.init_pos.detach().cpu())

        # remark: max_radii2D, xyz_gradient_accum and denom are not saved here; they are save elsewhere.
    def save_mlp_checkpoints(self, path):#split or unite
        # mkdir_p(os.path.dirname(path))
        # if path.endswith(".ply"):
        #         path = (
        #             path[:-4]
        #             + "_rk"
        #             + str(utils.GLOBAL_RANK)
        #             + "_ws"
        #             + str(utils.WORLD_SIZE)
        #             + ".ply"
        #         )
        args = utils.get_args()
        utils.log_cpu_memory_usage("start save_ply")
        group = utils.DEFAULT_GROUP
        if  args.distributed_save:
            self.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim).cuda()))
            opacity_mlp.save(os.path.join(path, "_rk"
                    + str(utils.GLOBAL_RANK)+'opacity_mlp.pt'))
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim).cuda()))
            cov_mlp.save(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'cov_mlp.pt'))
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
            color_mlp.save(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'color_mlp.pt'))
            if self.use_feat_bank:
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+self.level_dim).cuda()))
                feature_bank_mlp.save(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'feature_bank_mlp.pt'))
            if self.appearance_dim > 0:
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'embedding_appearance.pt'))
            self.train()
        else:
            if group.rank() != 0:
                return
            param_dict = {}
            param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
            param_dict['cov_mlp'] = self.mlp_cov.state_dict()
            param_dict['color_mlp'] = self.mlp_color.state_dict()
            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()
            torch.save(param_dict, os.path.join(path, 'checkpoints.pth'))


    def load_mlp_checkpoints(self, path, mode = 'split'):
        args = utils.get_args()
        utils.log_cpu_memory_usage("start save_ply")
        group = utils.DEFAULT_GROUP
        if args.distributed_save:
            self.mlp_opacity = torch.jit.load(os.path.join(path, "_rk"
                    + str(utils.GLOBAL_RANK)+'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path,  "_rk"
                    + str(utils.GLOBAL_RANK)+'embedding_appearance.pt')).cuda()
        else :
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])


    def prune_based_on_opacity(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        utils.LOG_FILE.write(
            "Pruning based on opacity. Percent: {:.2f}\n".format(
                100 * prune_mask.sum().item() / prune_mask.shape[0]
            )
        )
        self.prune_points(prune_mask)

    def get_camer_info(self,cameras,scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)

    def distributed_load_ply(self, folder):
        # count the number of files like "point_cloud_rk0_ws4.ply"
        world_size = -1
        for f in os.listdir(folder):
            if "_ws" in f:
                world_size = int(f.split("_ws")[1].split(".")[0])
                break
        assert world_size > 0, "world_size should be greater than 1."

        catted_anchor_feat = []
        catted_offset = []
        catted_levels = []
        catted_extra_level = []
        catted_anchor = []
        catted_opacity = []
        catted_scaling = []
        catted_rotation = []
        
        for rk in range(world_size):
            if utils.GLOBAL_RANK == rk:
                one_checkpoint_path = (
                    folder + "/point_cloud_rk" + str(rk) + "_ws" + str(world_size) + ".ply"
                )
                anchor_feat, level, extra_level, offset, anchor, scaling, opacity, rotation, anchor_mask, levels = (
                    self.load_raw_ply(one_checkpoint_path)
                )
                catted_anchor_feat.append(anchor_feat)
                catted_offset.append(offset)
                catted_levels.append(level)
                catted_extra_level.append(extra_level)
                catted_anchor.append(anchor)
                catted_opacity.append(opacity)
                catted_scaling.append(scaling)
                catted_rotation.append(rotation)
                catted_anchor_feat = np.concatenate(catted_anchor_feat, axis=0)
                catted_offset = np.concatenate(catted_offset, axis=0)
                catted_levels = np.concatenate(catted_levels, axis=0)
                catted_opacity = np.concatenate(catted_opacity, axis=0)
                catted_scaling = np.concatenate(catted_scaling, axis=0)
                catted_rotation = np.concatenate(catted_rotation, axis=0)
                catted_extra_level = np.concatenate(catted_extra_level, axis=0)
                catted_anchor = np.concatenate(catted_anchor, axis=0)
                self._anchor_feat = nn.Parameter(torch.tensor(catted_anchor_feat, dtype=torch.float, device="cuda").requires_grad_(True))
                self._level = torch.tensor(catted_levels, dtype=torch.int, device="cuda")
                self._extra_level = torch.tensor(catted_extra_level, dtype=torch.float, device="cuda").squeeze(dim=1)
                self._offset = nn.Parameter(torch.tensor(catted_offset, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
                self._anchor = nn.Parameter(torch.tensor(catted_anchor, dtype=torch.float, device="cuda").requires_grad_(True))
                self._scaling = nn.Parameter(torch.tensor(catted_scaling, dtype=torch.float, device="cuda").requires_grad_(True))
                self._opacity = nn.Parameter(torch.tensor(catted_opacity, dtype=torch.float, device="cuda").requires_grad_(False))
                self._rotation = nn.Parameter(torch.tensor(catted_rotation, dtype=torch.float, device="cuda").requires_grad_(False))
                self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
                self.levels = torch.max(self._level) - torch.min(self._level) + 1
                self.levels = self.levels.int().item()
                self.active_sh_degree = self.max_sh_degree
                self.init_level = int(self.levels/2)

        load_npz = np.load(folder + '/additional_attributes.npz')
        # self.init_pos = torch.tensor(load_npz['init_pos'],device="cuda")#Add by cjq 1.28

    def load_raw_ply(self, path):
        print("Loading ", path)
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis]
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        chunk = anchor.shape[0] // utils.WORLD_SIZE + 1
        # chunk = anchor.shape[0]
        point_ind_l = chunk * utils.LOCAL_RANK
        point_ind_r = min(chunk * (utils.LOCAL_RANK + 1), anchor.shape[0])
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        anchor_feat = np.ascontiguousarray(anchor_feats)
        level = np.ascontiguousarray(levels)
        extra_level = np.ascontiguousarray(extra_levels)
        offset = np.ascontiguousarray(offsets)
        anchor = np.ascontiguousarray(anchor)
        scaling = np.ascontiguousarray(scales)
        opacity = np.ascontiguousarray(opacities)
        rotation = np.ascontiguousarray(rots)
        anchor_mask = np.ones(anchor.shape[0])
        levels =  1

        return anchor_feat, level, extra_level, offset, anchor, scaling, opacity, rotation, anchor_mask, levels

    def one_file_load_ply(self, folder):
        path = os.path.join(folder, "point_cloud.ply")
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int32)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1

    def load_ply(self, path):
        if os.path.exists(os.path.join(path, "point_cloud.ply")):
            self.one_file_load_ply(path)
        else:
            self.distributed_load_ply(path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.zeros_like(tensor)
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][
                        mask
                    ]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors



    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors



    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        # if iterations>10000:
        #     grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * pixels[update_filter].unsqueeze(-1)
        #     self.offset_gradient_accum[combined_mask] += grad_norm
        #     self.offset_denom[combined_mask] +=  pixels[update_filter].unsqueeze(-1)
        # else:
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * 1
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] +=  1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]


    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)  #下一层的level
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork    #下一层的size
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level) #这一层的offset阈值
            ds_threshold = cur_threshold * update_value  #下一层的offseet阈值
            extra_threshold = cur_threshold * extra_ratio  #anchor的阈值
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)  #loss过大的话用下一层
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            # mask = torch.zeros_like(candidate_mask)
            # offset_reshape = self._offset.reshape(self.get_anchor.shape[0]*self.n_offsets, -1)
            # for box3d in box3ds:
            #     x_min_3d, y_min_3d, z_min_3d, x_max_3d, y_max_3d, z_max_3d = box3d
            #     mask_x = torch.logical_and(offset_reshape[:, 0] > x_min_3d, offset_reshape[:, 0] < x_max_3d)
            #     mask_y = torch.logical_and(offset_reshape[:, 1] > y_min_3d, offset_reshape[:, 1] < y_max_3d)
            #     mask_z = torch.logical_and(offset_reshape[:, 2] > z_min_3d, offset_reshape[:, 2] < z_max_3d)
            #     mask_xyz = torch.logical_and(mask_x, mask_y)
            #     mask_xyz = torch.logical_and(mask_xyz, mask_z)
                
            #     mask = torch.logical_or(mask, mask_xyz)
            # candidate_mask = candidate_mask | mask


            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if not self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (not self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:#progressive走完了之后 对于不是最后一层level的情况
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)  #这一层增加的和下一层增加的
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        # grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        # grads_abs[grads_abs.isnan()] = 0.0


        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)



    def group_for_redistribution(self):
        args = utils.get_args()
        if args.gaussians_distribution:
            return utils.DEFAULT_GROUP
        else:
            return utils.SingleGPUGroup()

    def all2all_gaussian_state(self, state, destination, i2j_send_size):
        comm_group = self.group_for_redistribution()

        # state: (N, ...) tensor
        state_to_gpuj = []
        state_from_gpuj = []
        for j in range(comm_group.size()):  # ugly implementation.
            state_to_gpuj.append(state[destination == j, ...].contiguous())
            state_from_gpuj.append(
                torch.zeros(
                    (i2j_send_size[j][comm_group.rank()], *state.shape[1:]),
                    device="cuda", dtype = state.dtype
                )
            )

        # print(f"before all_to_all, ws={comm_group.size()}, rank={comm_group.rank()}")

        # 假设 state_from_gpuj 和 state_to_gpuj 是需要检查的张量
            # if state_from_gpuj.dtype != state_to_gpuj.dtype:
            #     # 如果不一致，将它们转换为 torch.int32
            #     state_from_gpuj = state_from_gpuj.to(torch.int32)
            #     state_to_gpuj = state_to_gpuj.to(torch.int32)

        torch.distributed.all_to_all(state_from_gpuj, state_to_gpuj, group=comm_group)

        # print(f"after all_to_all, ws={comm_group.size()}, rank={comm_group.rank()}")

        state_from_remote = torch.cat(
            state_from_gpuj, dim=0
        ).contiguous()  # it stucks at here.
        # print(f"state_from_remote, ws={comm_group.size()}, rank={comm_group.rank()}")
        return state_from_remote

    def all2all_tensors_in_optimizer_implementation_1(self, destination, i2j_send_size):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp_opacity" or group["name"] == "mlp_cov" or group["name"] == "mlp_color":
                continue

            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = self.all2all_gaussian_state(
                        stored_state["momentum_buffer"], destination, i2j_send_size
                    )
                else:
                    stored_state["exp_avg"] = self.all2all_gaussian_state(
                        stored_state["exp_avg"], destination, i2j_send_size
                    )
                    stored_state["exp_avg_sq"] = self.all2all_gaussian_state(
                        stored_state["exp_avg_sq"], destination, i2j_send_size
                    )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    self.all2all_gaussian_state(
                        group["params"][0], destination, i2j_send_size
                    ),
                    requires_grad=True,
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    self.all2all_gaussian_state(
                        group["params"][0], destination, i2j_send_size
                    ),
                    requires_grad=True,
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_all_optimizer_states(self):
        all_tensors = []
        all_shapes = []
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    all_tensors.append(stored_state["momentum_buffer"])
                    all_shapes.append(stored_state["momentum_buffer"].shape)
                else:
                    all_tensors.append(stored_state["exp_avg"])
                    all_shapes.append(stored_state["exp_avg"].shape)

                    all_tensors.append(stored_state["exp_avg_sq"])
                    all_shapes.append(stored_state["exp_avg_sq"].shape)

                all_tensors.append(group["params"][0])
                all_shapes.append(group["params"][0].shape)

                # release the memory BUG: release the memory will cause error. Maybe it will release memory which may use later.
                # stored_state["exp_avg"] = None
                # stored_state["exp_avg_sq"] = None
                # group["params"][0] = None

            else:
                all_tensors.append(group["params"][0])
                all_shapes.append(group["params"][0].shape)

                # release the memory BUG: release the memory will cause error. Maybe it will release memory which may use later.
                # group["params"][0] = None
        return all_tensors, all_shapes

    def update_all_optimizer_states(self, updated_tensors):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = updated_tensors.pop(
                        0
                    ).contiguous()
                else:
                    stored_state["exp_avg"] = updated_tensors.pop(0).contiguous()
                    stored_state["exp_avg_sq"] = updated_tensors.pop(0).contiguous()

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    updated_tensors.pop(0).contiguous(), requires_grad=True
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    updated_tensors.pop(0).contiguous(), requires_grad=True
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def all2all_tensors_in_optimizer_implementation_2(self, destination, i2j_send_size):
        # merge into one single all2all kernal launch.

        # get all optimizer states for all2all
        all_tensors, all_shapes = self.get_all_optimizer_states()
        # flatten all tensors with start_dim=1, then concate them at dim=1
        all_tensors_flatten = [tensor.flatten(start_dim=1) for tensor in all_tensors]
        all_tensors_catted = torch.cat(all_tensors_flatten, dim=1).contiguous()
        all_tensors_flatten = None  # release memory

        # all2all
        all_remote_tensors_catted = self.all2all_gaussian_state(
            all_tensors_catted, destination, i2j_send_size
        )
        all_tensors_catted = None  # release memory

        # split all_tensors_catted to original shapes
        all_remote_tensors_flatten = torch.split(
            all_remote_tensors_catted,
            [shape[1:].numel() for shape in all_shapes],
            dim=1,
        )
        all_remote_tensors_catted = None  # release memory
        all_remote_tensors = [
            tensor.view(tensor.shape[:1] + shape[1:])
            for tensor, shape in zip(all_remote_tensors_flatten, all_shapes)
        ]
        all_remote_tensors_flatten = None  # release memory

        # update optimizer states
        optimizable_tensors = self.update_all_optimizer_states(all_remote_tensors)
        all_remote_tensors = None

        return optimizable_tensors

    def all2all_tensors_in_optimizer(self, destination, i2j_send_size):
        return self.all2all_tensors_in_optimizer_implementation_1(
            destination, i2j_send_size
        )
        # return self.all2all_tensors_in_optimizer_implementation_2(destination, i2j_send_size)
        # when cross node all2all on perl, implementation_2 will get stuck at 1600 iterations, I do not know the reason.

    def get_destination_1(self, world_size):
        # norm p=0
        return torch.arange(self.get_anchor.shape[0], device="cuda") % world_size

    def need_redistribute_anchors(self, group):
        args = utils.get_args()
        if group.size() == 1:
            return False
        if utils.get_denfify_iter() == args.redistribute_anchors_frequency:
            # do redistribution after the first densification.
            return True
        local_n_anchors = self.get_anchor.shape[0]
        all_local_n_anchors = [None for _ in range(group.size())]
        torch.distributed.all_gather_object(all_local_n_anchors, local_n_anchors, group=group)
        if min(all_local_n_anchors) * args.redistribute_gaussians_threshold < max(
            all_local_n_anchors
        ):
            return True
        return False

    def redistribute_gaussians(self):
        args = utils.get_args()
        if args.redistribute_anchor_mode == "no_redistribute":
            return

        comm_group_for_redistribution = self.group_for_redistribution()
        if not self.need_redistribute_anchors(comm_group_for_redistribution):
            return

        # Get each anchors' destination GPU.
        if args.redistribute_anchor_mode == "random_redistribute":
            # random redistribution to balance the number of gaussians on each GPU.
            destination = self.get_destination_1(comm_group_for_redistribution.size())
        else:
            raise ValueError(
                "Invalid redistribute_anchor_mode: "
                + args.redistribute_anchor_mode
            )

        # Count the number of anchors to be sent to each GPU.
        local2j_send_size = torch.bincount(
            destination, minlength=comm_group_for_redistribution.size()
        ).int()
        assert (
            len(local2j_send_size) == comm_group_for_redistribution.size()
        ), "local2j_send_size: " + str(local2j_send_size)

        i2j_send_size = torch.zeros(
            (
                comm_group_for_redistribution.size(),
                comm_group_for_redistribution.size(),
            ),
            dtype=torch.int,
            device="cuda",
        )
        torch.distributed.all_gather_into_tensor(
            i2j_send_size, local2j_send_size, group=comm_group_for_redistribution
        )
        i2j_send_size = i2j_send_size.cpu().numpy().tolist()
        # print("rank", utils.LOCAL_RANK, "local2j_send_size: ", local2j_send_size, "i2j_send_size: ", i2j_send_size)

        optimizable_tensors = self.all2all_tensors_in_optimizer(
            destination, i2j_send_size
        )


        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]





        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        # self._level = self._level.unsqueeze(dim=1)
        self._level = self.all2all_gaussian_state(self._level, destination, i2j_send_size)
        self._extra_level = self.all2all_gaussian_state(self._extra_level, destination, i2j_send_size)
        # self._anchor_mask = self.all2all_gaussian_state(self._anchor_mask, destination, i2j_send_size)
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")



        # NOTE: This function is called right after desify_and_prune. Therefore self.xyz_gradient_accum, self.denom and self.max_radii2D are all zero.
        # We do not need to all2all them here.

        # should I all2all send_to_gpui_cnt? I think I should not. Because 1. for simplicity now, 2. we should refresh it and do not use too old statistics.
        self.send_to_gpui_cnt = torch.zeros(
            (self.get_anchor.shape[0], comm_group_for_redistribution.size()),
            dtype=torch.int,
            device="cuda",
        )

        torch.cuda.empty_cache()


def get_sparse_ids(tensors):
    sparse_ids = None
    with torch.no_grad():
        for tensor in tensors:
            # Apply torch.nonzero()
            nonzero_indices = torch.nonzero(tensor)
            # Extract the row indices
            row_indices = nonzero_indices[:, 0]
            # Count unique rows
            if sparse_ids is None:
                sparse_ids = row_indices
            else:
                sparse_ids = torch.cat((sparse_ids, row_indices))

        sparse_ids = torch.unique(sparse_ids, sorted=True)
        return sparse_ids


def sync_gradients_sparsely(gaussians, group):
    with torch.no_grad():
        sparse_ids = get_sparse_ids(
            [gaussians._xyz.grad.data]
        )  # sparse ids are non-zero ids
        # get boolean mask of sparse ids
        sparse_ids_mask = torch.zeros(
            (gaussians._xyz.shape[0]), dtype=torch.bool, device="cuda"
        )
        sparse_ids_mask[sparse_ids] = True

        torch.distributed.all_reduce(sparse_ids_mask, op=dist.ReduceOp.SUM, group=group)

        def sync_grads(data):
            sparse_grads = data.grad.data[
                sparse_ids_mask
            ].contiguous()  # contiguous() memory is needed for collective communication.
            torch.distributed.all_reduce(
                sparse_grads, op=dist.ReduceOp.SUM, group=group
            )
            data.grad.data[sparse_ids_mask] = sparse_grads

        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)
        # We must optimize this, because there should be large kernel launch overhead.

    log_file = utils.get_log_file()
    non_zero_indices_cnt = sparse_ids_mask.sum().item()
    total_indices_cnt = sparse_ids_mask.shape[0]
    log_file.write(
        "iterations: [{}, {}) non_zero_indices_cnt: {} total_indices_cnt: {} ratio: {}\n".format(
            utils.get_cur_iter(),
            utils.get_cur_iter() + utils.get_args().bsz,
            non_zero_indices_cnt,
            total_indices_cnt,
            non_zero_indices_cnt / total_indices_cnt,
        )
    )


def sync_gradients_densely(gaussians, group):
    with torch.no_grad():

        def sync_grads(data):
            torch.distributed.all_reduce(
                data.grad.data, op=dist.ReduceOp.SUM, group=group
            )

        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)


def sync_gradients_fused_densely(gaussians, group):
    with torch.no_grad():
        # 1. cat all parameters' grad to a single tensor
        # 2. allreduce
        # 3. split the allreduced tensor to each parameter's grad
        all_params_grads = [
            param.grad.data
            for param in [
                gaussians._xyz,
                gaussians._features_dc,
                gaussians._features_rest,
                gaussians._opacity,
                gaussians._scaling,
                gaussians._rotation,
            ]
        ]
        all_params_grads_dim1 = [param_grad.shape[1] for param_grad in all_params_grads]
        catted_params_grads = torch.cat(all_params_grads, dim=1).contiguous()
        torch.distributed.all_reduce(
            catted_params_grads, op=dist.ReduceOp.SUM, group=group
        )
        split_params_grads = torch.split(
            catted_params_grads, all_params_grads_dim1, dim=1
        )
        for param_grad, split_param_grad in zip(all_params_grads, split_params_grads):
            param_grad.copy_(split_param_grad)


def sync_gradients_fused_sparsely(gaussians, group):
    raise NotImplementedError("Fused sparse sync gradients is not implemented yet.")
