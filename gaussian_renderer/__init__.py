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
import math
import cv2
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel
import utils.general_utils as utils
import torch.distributed.nn.functional as dist_func
from einops import repeat
from utils.graphics_utils import normal_from_depth_image
from pytorch3d.transforms import quaternion_to_matrix
def get_cuda_args(strategy, mode="train"):  # "test"
    args = utils.get_args()
    iteration = utils.get_cur_iter()

    if mode == "train":
        for x in range(args.bsz):  # This is to make sure we will get the
            if (iteration + x) % args.log_interval == 1:
                iteration += x
                break
        avoid_pixel_all2all = args.image_distribution_config.avoid_pixels_all2all
    elif mode == "test":
        iteration = -1
        avoid_pixel_all2all = False
    else:
        raise ValueError("mode should be train or test.")

    cuda_args = {
        "mode": mode,
        "world_size": str(utils.WORLD_SIZE),
        "global_rank": str(utils.GLOBAL_RANK),
        "local_rank": str(utils.LOCAL_RANK),
        "mp_world_size": str(utils.MP_GROUP.size()),
        "mp_rank": str(utils.MP_GROUP.rank()),
        "log_folder": args.log_folder,
        "log_interval": str(args.log_interval),
        "iteration": str(iteration),
        "zhx_debug": str(args.zhx_debug),
        "zhx_time": str(args.zhx_time),
        "dist_global_strategy": strategy.get_global_strategy_str(),
        "avoid_pixel_all2all": avoid_pixel_all2all,
        "stats_collector": {},
    }
    return cuda_args


def replicated_preprocess3dgs(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    strategy=None,
    mode="train",
    return_plane = True,
):
    """
    preprocess 3dgs.

    all 3DGS are stored replicatedly on all GPUs.
    """
    ########## [START] Prepare CUDA Rasterization Settings ##########
    timers = utils.get_timers()
    if timers is not None:
        timers.start("forward_prepare_args_and_settings")
    # only locally render one camera in a batched cameras
    cuda_args = get_cuda_args(strategy, mode)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=return_plane,
        debug=pipe.debug,
    )

    # # print raster_settings in a json format
    # import json
    # raster_settings_dict = {}
    # raster_settings_dict["viewmatrix"] = raster_settings.viewmatrix.tolist()
    # raster_settings_dict["projmatrix"] = raster_settings.projmatrix.tolist()
    # raster_settings_dict["bg"] = raster_settings.bg.tolist()
    # raster_settings_dict["image_height"] = raster_settings.image_height
    # raster_settings_dict["image_width"] = raster_settings.image_width
    # raster_settings_dict["tanfovx"] = raster_settings.tanfovx
    # raster_settings_dict["tanfovy"] = raster_settings.tanfovy
    # raster_settings_dict["scale_modifier"] = raster_settings.scale_modifier
    # raster_settings_dict["sh_degree"] = raster_settings.sh_degree
    # raster_settings_dict["campos"] = raster_settings.campos.tolist()
    # raster_settings_dict["prefiltered"] = raster_settings.prefiltered
    # raster_settings_dict["debug"] = raster_settings.debug
    # json.dump(raster_settings_dict, open("one_raster_settings_example.json", "w"))
    # exit()

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if timers is not None:
        timers.stop("forward_prepare_args_and_settings")
    ########## [END] Prepare CUDA Rasterization Settings ##########

    ########## [START] Prepare Gaussians for rendering ##########
    if timers is not None:
        timers.start("forward_prepare_gaussians")
    means3D = pc.get_anchor
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    if timers is not None:
        timers.stop("forward_prepare_gaussians")

    utils.check_initial_gpu_memory_usage("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########

    ########## [START] CUDA Rasterization Call ##########
    # Rasterize visible Gaussians to image, obtain their screen-space intermedia parameters.
    if timers is not None:
        timers.start("forward_preprocess_gaussians")
    # [3DGS-wise preprocess]
    means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        shs=shs,
        opacities=opacity,
        cuda_args=cuda_args,
    )
    if mode == "train":
        means2D.retain_grad()
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")

    screenspace_pkg = {
        "rasterizer": rasterizer,
        "cuda_args": cuda_args,
        "locally_preprocessed_mean2D": means2D,
        "locally_preprocessed_radii": radii,
        "means2D_for_render": means2D,
        "rgb_for_render": rgb,
        "conic_opacity_for_render": conic_opacity,
        "radii_for_render": radii,
        "depths_for_render": depths,
    }
    return screenspace_pkg


def all_to_all_communication(
    batched_rasterizers,
    batched_screenspace_params,
    batched_cuda_args,
    batched_strategies,
):
    batched_local2j_ids = []
    batched_local2j_ids_bool = []
    for i in range(utils.DP_GROUP.size()):
        means2D, rgb, conic_opacity, radii, depths = batched_screenspace_params[i]
        local2j_ids, local2j_ids_bool = batched_strategies[i].get_local2j_ids(
            means2D, radii, batched_rasterizers[i].raster_settings, batched_cuda_args[i]
        )
        batched_local2j_ids.append(local2j_ids)
        batched_local2j_ids_bool.append(local2j_ids_bool)

    catted_batched_local2j_ids_bool = torch.cat(batched_local2j_ids_bool, dim=1)

    i2j_send_size = torch.zeros(
        (utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.size()),
        dtype=torch.int,
        device="cuda",
    )
    local2j_send_size = []
    for i in range(utils.DP_GROUP.size()):
        for j in range(utils.MP_GROUP.size()):
            local2j_send_size.append(len(batched_local2j_ids[i][j]))
    local2j_send_size = torch.tensor(local2j_send_size, dtype=torch.int, device="cuda")
    torch.distributed.all_gather_into_tensor(
        i2j_send_size, local2j_send_size, group=utils.DEFAULT_GROUP
    )
    i2j_send_size = i2j_send_size.cpu().numpy().tolist()

    def one_all_to_all(batched_tensors, use_function_version=False):
        tensor_to_rki = []
        tensor_from_rki = []
        for d_i in range(utils.DP_GROUP.size()):
            for d_j in range(utils.MP_GROUP.size()):
                i = d_i * utils.MP_GROUP.size() + d_j
                tensor_to_rki.append(
                    batched_tensors[d_i][batched_local2j_ids[d_i][d_j]].contiguous()
                )  # NCCL communication requires contiguous memory.
                tensor_from_rki.append(
                    torch.zeros(
                        (i2j_send_size[i][utils.DEFAULT_GROUP.rank()],)
                        + batched_tensors[0].shape[1:],
                        dtype=batched_tensors[0].dtype,
                        device="cuda",
                    )
                )

        if use_function_version:
            dist_func.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )  # The function version could naturally enable communication during backward.
        else:
            torch.distributed.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )
        return torch.cat(tensor_from_rki, dim=0).contiguous()

    # Merge means2D, rgb, conic_opacity into one functional all-to-all communication call.
    batched_catted_screenspace_states = []
    batched_catted_screenspace_auxiliary_states = []
    for i in range(utils.DP_GROUP.size()):
        means2D, rgb, conic_opacity, radii, depths = batched_screenspace_params[i]
        if i == 0:
            mean2d_dim1 = means2D.shape[1]
            rgb_dim1 = rgb.shape[1]
            conic_opacity_dim1 = conic_opacity.shape[1]
        batched_catted_screenspace_states.append(
            torch.cat([means2D, rgb, conic_opacity], dim=1).contiguous()
        )
        batched_catted_screenspace_auxiliary_states.append(
            torch.cat(
                [radii.float().unsqueeze(1), depths.unsqueeze(1)], dim=1
            ).contiguous()
        )

    params_redistributed = one_all_to_all(
        batched_catted_screenspace_states, use_function_version=True
    )
    means2D_redistributed, rgb_redistributed, conic_opacity_redistributed = torch.split(
        params_redistributed, [mean2d_dim1, rgb_dim1, conic_opacity_dim1], dim=1
    )
    radii_depth_redistributed = one_all_to_all(
        batched_catted_screenspace_auxiliary_states, use_function_version=False
    )
    radii_redistributed, depths_redistributed = torch.split(
        radii_depth_redistributed, [1, 1], dim=1
    )
    radii_redistributed = radii_redistributed.squeeze(1).int()
    depths_redistributed = depths_redistributed.squeeze(1)

    return (
        means2D_redistributed,
        rgb_redistributed,
        conic_opacity_redistributed,
        radii_redistributed,
        depths_redistributed,
        i2j_send_size,
        catted_batched_local2j_ids_bool,
    )


def distributed_preprocess3dgs_and_all2all(
    batched_viewpoint_cameras,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    batched_strategies=None,
    mode="train",
    return_plane : bool = True
):
    """
    Render the scene.

    distribute gaussians parameters across all GPUs.
    """
    assert (
        utils.DEFAULT_GROUP.size() > 1
    ), "This function is only for distributed training. "

    timers = utils.get_timers()

    ########## [START] Prepare Gaussians for rendering ##########
    if timers is not None:
        timers.start("forward_prepare_gaussians")
    means3D = pc.get_anchor
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    if timers is not None:
        timers.stop("forward_prepare_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########

    if timers is not None:
        timers.start("forward_preprocess_gaussians")
    batched_rasterizers = []
    batched_cuda_args = []
    batched_screenspace_params = []
    batched_means2D = []
    batched_radii = []
    for i, (viewpoint_camera, strategy) in enumerate(
        zip(batched_viewpoint_cameras, batched_strategies)
    ):
        ########## [START] Prepare CUDA Rasterization Settings ##########
        cuda_args = get_cuda_args(strategy, mode)
        batched_cuda_args.append(cuda_args)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        ########## [END] Prepare CUDA Rasterization Settings ##########

        # [3DGS-wise preprocess]
        means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            shs=shs,
            opacities=opacity,
            cuda_args=cuda_args,
        )
        if mode == "train":
            means2D.retain_grad()
        batched_means2D.append(means2D)
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)
    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")

    if timers is not None:
        timers.start("forward_all_to_all_communication")
    (
        means2D_redistributed,
        rgb_redistributed,
        conic_opacity_redistributed,
        radii_redistributed,
        depths_redistributed,
        i2j_send_size,
        local2j_ids_bool,
    ) = all_to_all_communication(
        batched_rasterizers,
        batched_screenspace_params,
        batched_cuda_args,
        batched_strategies,
    )
    utils.check_initial_gpu_memory_usage("after forward_all_to_all_communication")
    if timers is not None:
        timers.stop("forward_all_to_all_communication")

    screenspace_pkg = {
        "batched_locally_preprocessed_mean2D": batched_means2D,
        "batched_locally_preprocessed_radii": batched_radii,
        "rasterizer": batched_rasterizers[utils.DP_GROUP.rank()],
        "cuda_args": batched_cuda_args[utils.DP_GROUP.rank()],
        "means2D_for_render": means2D_redistributed,
        "rgb_for_render": rgb_redistributed,
        "conic_opacity_for_render": conic_opacity_redistributed,
        "radii_for_render": radii_redistributed,
        "depths_for_render": depths_redistributed,
        "i2j_send_size": i2j_send_size,
        "local2j_ids_bool": local2j_ids_bool,
    }
    return screenspace_pkg


def preprocess3dgs_and_all2all(
    batched_cameras, gaussians, pipe_args, background, batched_strategies, mode
):
    args = utils.get_args()

    local_render_viewpoint_cam = batched_cameras[utils.DP_GROUP.rank()]
    local_render_strategy = batched_strategies[utils.DP_GROUP.rank()]

    if args.gaussians_distribution:
        screenspace_pkg = distributed_preprocess3dgs_and_all2all(
            batched_cameras,
            gaussians,
            pipe_args,
            background,
            batched_strategies=batched_strategies,
            mode=mode,
        )
        if mode == "test":
            return screenspace_pkg

        screenspace_pkg["batched_locally_preprocessed_visibility_filter"] = [
            radii > 0 for radii in screenspace_pkg["batched_locally_preprocessed_radii"]
        ]
    else:
        screenspace_pkg = replicated_preprocess3dgs(
            local_render_viewpoint_cam,
            gaussians,
            pipe_args,
            background,
            strategy=local_render_strategy,
            mode=mode,
        )
        if mode == "test":
            return screenspace_pkg

        screenspace_pkg["batched_locally_preprocessed_radii"] = [
            screenspace_pkg["locally_preprocessed_radii"]
        ]
        screenspace_pkg["batched_locally_preprocessed_visibility_filter"] = [
            screenspace_pkg["locally_preprocessed_radii"] > 0
        ]
        screenspace_pkg["batched_locally_preprocessed_mean2D"] = [
            screenspace_pkg["locally_preprocessed_mean2D"]
        ]

    return screenspace_pkg


def render(screenspace_pkg, strategy=None):
    """
    Render the scene.
    """
    timers = utils.get_timers()

    # get compute_locally to know local workload in the end2end distributed training.
    if timers is not None:
        timers.start("forward_compute_locally")
    compute_locally = strategy.get_compute_locally()
    extended_compute_locally = strategy.get_extended_compute_locally()
    if timers is not None:
        timers.stop("forward_compute_locally")
    utils.check_initial_gpu_memory_usage("after forward_compute_locally")

    # render
    if timers is not None:
        timers.start("forward_render_gaussians")
    if screenspace_pkg["means2D_for_render"].shape[0] < 1000:
        # assert utils.get_args().image_distribution_mode == "3", "The image_distribution_mode should be 3."
        # rendered_image = torch.zeros((3, screenspace_pkg["rasterizer"].raster_settings.image_height, screenspace_pkg["rasterizer"].raster_settings.image_width), dtype=torch.float32, device="cuda", requires_grad=True)
        rendered_image = (
            screenspace_pkg["means2D_for_render"].sum()
            + screenspace_pkg["conic_opacity_for_render"].sum()
            + screenspace_pkg["rgb_for_render"].sum()
        )
        screenspace_pkg["cuda_args"]["stats_collector"]["forward_render_time"] = 0.0
        screenspace_pkg["cuda_args"]["stats_collector"]["backward_render_time"] = 0.0
        screenspace_pkg["cuda_args"]["stats_collector"]["forward_loss_time"] = 0.0
        screenspace_pkg["cuda_args"]["stats_collector"]["backward_loss_time"] = 0.0
        return rendered_image, compute_locally
    else:
        rendered_image, n_render, n_consider, n_contrib = screenspace_pkg[
            "rasterizer"
        ].render_gaussians(
            means2D=screenspace_pkg["means2D_for_render"],
            conic_opacity=screenspace_pkg["conic_opacity_for_render"],
            rgb=screenspace_pkg["rgb_for_render"],
            depths=screenspace_pkg["depths_for_render"],
            radii=screenspace_pkg["radii_for_render"],
            compute_locally=compute_locally,
            extended_compute_locally=extended_compute_locally,
            cuda_args=screenspace_pkg["cuda_args"],
        )
    if timers is not None:
        timers.stop("forward_render_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return rendered_image, compute_locally


def get_cuda_args_final(strategy, mode="train"):
    args = utils.get_args()
    iteration = utils.get_cur_iter()

    if mode == "train":
        for x in range(args.bsz):
            if (iteration + x) % args.log_interval == 1:
                iteration += x
                break
    elif mode == "test":
        iteration = -1
    else:
        raise ValueError("mode should be train or test.")

    cuda_args = {
        "mode": mode,
        "world_size": str(utils.WORLD_SIZE),
        "global_rank": str(utils.GLOBAL_RANK),
        "local_rank": str(utils.LOCAL_RANK),
        "mp_world_size": str(strategy.world_size),
        "mp_rank": str(strategy.rank),
        "log_folder": args.log_folder,
        "log_interval": str(args.log_interval),
        "iteration": str(iteration),
        "zhx_debug": str(args.zhx_debug),
        "zhx_time": str(args.zhx_time),
        "avoid_pixel_all2all": False,
        "stats_collector": {},
    }
    return cuda_args





def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo = False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_pure > 0
    return visible_mask








def all_to_all_communication_final(
    batched_rasterizers,
    batched_screenspace_params,
    batched_cuda_args,
    batched_strategies,
):
    num_cameras = len(batched_rasterizers)
    # gpui_to_gpuj_camk_size
    # gpui_to_gpuj_camk_send_ids

    local_to_gpuj_camk_size = [[] for j in range(utils.DEFAULT_GROUP.size())]  
    local_to_gpuj_camk_send_ids = [[] for j in range(utils.DEFAULT_GROUP.size())] 
    for k in range(num_cameras):
        strategy = batched_strategies[k]
        means2D, rgb, conic_opacity, radii, depths, means3D, scales, rotation = batched_screenspace_params[k]
        local2j_ids, local2j_ids_bool = batched_strategies[k].get_local2j_ids(
            means2D, radii, batched_rasterizers[k].raster_settings, batched_cuda_args[k]
        ) 

        for local_id, global_id in enumerate(strategy.gpu_ids):
            local_to_gpuj_camk_size[global_id].append(len(local2j_ids[local_id]))
            local_to_gpuj_camk_send_ids[global_id].append(local2j_ids[local_id]) #

        for j in range(utils.DEFAULT_GROUP.size()):
            if len(local_to_gpuj_camk_size[j]) == k:
                local_to_gpuj_camk_size[j].append(0)
                local_to_gpuj_camk_send_ids[j].append(
                    torch.empty((0, 1), dtype=torch.int64)
                )

    gpui_to_gpuj_imgk_size = torch.zeros(
        (utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.size(), num_cameras),
        dtype=torch.int,
        device="cuda",
    )
    local_to_gpuj_camk_size_tensor = torch.tensor(
        local_to_gpuj_camk_size, dtype=torch.int, device="cuda"
    )
    torch.distributed.all_gather_into_tensor(
        gpui_to_gpuj_imgk_size,
        local_to_gpuj_camk_size_tensor,  #
        group=utils.DEFAULT_GROUP,
    )
    gpui_to_gpuj_imgk_size = gpui_to_gpuj_imgk_size.cpu().numpy().tolist()

    def one_all_to_all(batched_tensors, use_function_version=False):
        tensor_to_rki = []
        tensor_from_rki = []
        for i in range(utils.DEFAULT_GROUP.size()):
            tensor_to_rki_list = []
            tensor_from_rki_size = 0
            for k in range(num_cameras):
                tensor_to_rki_list.append(
                    batched_tensors[k][local_to_gpuj_camk_send_ids[i][k]]  #
                )
                #
                tensor_from_rki_size += gpui_to_gpuj_imgk_size[i][
                    utils.DEFAULT_GROUP.rank()
                ][k]
            tensor_to_rki.append(torch.cat(tensor_to_rki_list, dim=0).contiguous())
            tensor_from_rki.append(
                torch.empty(
                    (tensor_from_rki_size,) + batched_tensors[0].shape[1:],
                    dtype=batched_tensors[0].dtype,
                    device="cuda",
                )
            )#
        if (
            use_function_version
        ):  # FIXME: there is error if I use torch.distributed.nn.functional to replace dist_func here. So weird.
            dist_func.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )  # The function version could naturally enable communication during backward.
        else:
            torch.distributed.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )

        # tensor_from_rki: (world_size, (all data received from all other GPUs))
        for i in range(utils.DEFAULT_GROUP.size()):
            # -> (world_size, num_cameras, *)
            tensor_from_rki[i] = tensor_from_rki[i].split(
                gpui_to_gpuj_imgk_size[i][utils.DEFAULT_GROUP.rank()], dim=0
            )

        tensors_per_camera = []
        for k in range(num_cameras):
            tensors_per_camera.append(
                torch.cat(
                    [tensor_from_rki[i][k] for i in range(utils.DEFAULT_GROUP.size())],
                    dim=0,
                ).contiguous()
            )

        return tensors_per_camera

    # Merge means2D, rgb, conic_opacity into one functional all-to-all communication call.
    batched_catted_screenspace_states = []
    batched_catted_screenspace_auxiliary_states = []
    for k in range(num_cameras):
        means2D, rgb, conic_opacity, radii, depths, means3D, scales, rotation = batched_screenspace_params[k]
        if k == 0:
            mean2d_dim1 = means2D.shape[1]
            rgb_dim1 = rgb.shape[1]
            conic_opacity_dim1 = conic_opacity.shape[1]
        batched_catted_screenspace_states.append(
            torch.cat([means2D, rgb, conic_opacity, means3D, scales, rotation], dim=1).contiguous()
        )
        batched_catted_screenspace_auxiliary_states.append(
            torch.cat(
                [radii.float().unsqueeze(1), depths.unsqueeze(1)], dim=1
            ).contiguous()
        )

    batched_params_redistributed = one_all_to_all(
        batched_catted_screenspace_states, use_function_version=True
    )
    batched_means2D_redistributed = []
    batched_rgb_redistributed = []
    batched_conic_opacity_redistributed = []
    batched_means3D_redistributed = []
    batched_scales_redistributed = []
    batched_rotations_redistributed = []
    # batched_view_points_redistributed = []
    # batched_ts_redistributed = []
    # batched_camera_planes_redistributed = []
    # batched_ray_plane_redistributed = []
    # batched_normals_redistributed = []
    for k in range(num_cameras):
        means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, means3D_redistributed, scales_redistributed, rotations_redistributed= (
            torch.split(
                batched_params_redistributed[k],
                [mean2d_dim1, rgb_dim1, conic_opacity_dim1, 3, 3, 4],
                dim=1,
            )
        )

        batched_means2D_redistributed.append(means2D_redistributed)
        batched_rgb_redistributed.append(rgb_redistributed)
        batched_conic_opacity_redistributed.append(conic_opacity_redistributed)    #gyy
        batched_means3D_redistributed.append(means3D_redistributed)
        batched_scales_redistributed.append(scales_redistributed)
        batched_rotations_redistributed.append(rotations_redistributed)
        # batched_ts_redistributed.append(ts_redistributed)
        # batched_camera_planes_redistributed.append(camera_planes_redistributed)
        # batched_ray_plane_redistributed.append(ray_plane_redistributed)
        # batched_normals_redistributed.append(normals_redistributed)
        # batched_view_points_redistributed.append(view_points_redistributed)

    batched_radii_depth_redistributed = one_all_to_all(   #add normal
        batched_catted_screenspace_auxiliary_states, use_function_version=False
    )
    batched_radii_redistributed = []
    batched_depths_redistributed = []
    for k in range(num_cameras):
        radii_redistributed, depths_redistributed  = torch.split(
            batched_radii_depth_redistributed[k], [1, 1], dim=1
        )

        batched_radii_redistributed.append(radii_redistributed.squeeze(1).int())
        batched_depths_redistributed.append(depths_redistributed.squeeze(1))
        # batched_ts_redistributed.append(ts_redistributed)
        # batched_camera_planes_redistributed.append(camera_planes_redistributed)
        # batched_ray_plane_redistributed.append(ray_plane_redistributed)
        # batched_normals_redistributed.append(normals_redistributed)
        # batched_view_points_redistributed.append(view_points_redistributed)
    return (
        batched_means2D_redistributed,
        batched_rgb_redistributed,
        batched_conic_opacity_redistributed,
        batched_radii_redistributed,
        batched_depths_redistributed,
        batched_means3D_redistributed,
        batched_scales_redistributed,
        batched_rotations_redistributed,
        # batched_view_points_redistributed,
        # batched_ts_redistributed,
        # batched_camera_planes_redistributed,
        # batched_ray_plane_redistributed,
        # batched_normals_redistributed,
        gpui_to_gpuj_imgk_size,
        local_to_gpuj_camk_send_ids,
    )


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
            
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets 

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot



def distributed_preprocess3dgs_and_all2all_final(
    batched_viewpoint_cameras,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    batched_voxel_mask=None, 
    batched_nearest_cameras = None,
    batched_nearest_voxel_mask = None,
    retain_grad=False,
    batched_strategies=None,
    mode="train",
    return_plane : bool = True,
    iterations = 0
):
    
    """
    Render the scene.

    distribute gaussians parameters across all GPUs.
    """
    timers = utils.get_timers()
    args = utils.get_args()

    assert utils.DEFAULT_GROUP.size() == 1 or (
        args.gaussians_distribution and args.image_distribution
    ), "Ensure distributed training given multiple GPU. "

    ########## [START] Prepare Gaussians for rendering ##########
    if timers is not None:
        timers.start("forward_prepare_gaussians")
    batched_means3D = []
    batched_opacity = []
    batched_scales = []
    batched_rotations = []
    batched_colors = []
    batched_neural_opacity = []
    batched_mask = []
    batched_rasterizers = []  # One rasterizer for each picture in a batch
    batched_cuda_args = []  # Per picture in a batch
    batched_screenspace_params = []  # Per picture in a batch
    batched_means2D = []
    batched_radii = []
    batched_out_observe = []
    batched_cuda_args_test = []


    batched_means3D_nearest = []
    batched_opacity_nearest = []
    batched_scales_nearest = []
    batched_rotations_nearest = []
    batched_colors_nearest = []


    batched_means2D_nearest = []

    batched_rasterizers_nearest = []
    batched_screenspace_params_nearest = []
    batched_radii_nearest = []
    if timers is not None:
        timers.start("forward_preprocess_gaussians")


    for viewpoint_camera, viewpoint_camera_nearest, visible_mask, visible_mask_nearest, strategy in zip(batched_viewpoint_cameras, batched_nearest_cameras, batched_voxel_mask, batched_nearest_voxel_mask, batched_strategies):
        if mode == "train":
            xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, mode == "train")
            batched_neural_opacity.append(neural_opacity)
            batched_mask.append(mask)
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, mode == "train")
        batched_means3D.append(xyz)
        batched_colors.append(color)
        batched_opacity.append(opacity)
        batched_scales.append(scaling)
        batched_rotations.append(rot)
        
        ########## [START] Prepare CUDA Rasterization Settings ##########
        cuda_args = get_cuda_args_final(strategy, mode)
        batched_cuda_args.append(cuda_args)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo = return_plane,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        ########## [END] Prepare CUDA Rasterization Settings ##########

        # [3DGS-wise preprocess]

        means2D, _, conic_opacity,  radii, depths = rasterizer.preprocess_gaussians(
            means3D=xyz,
            scales=scaling,
            rotations=rot,
            shs=color,
            opacities=opacity,
            cuda_args=cuda_args,
        )

        if mode == "train":
            means2D.retain_grad()

        batched_means2D.append(means2D)
        screenspace_params = [means2D, color, conic_opacity, radii, depths, xyz, scaling, rot]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)

        if visible_mask_nearest is not None:
            with torch.no_grad():
                cuda_args_test = get_cuda_args_final(strategy, "test")
                batched_cuda_args_test.append(cuda_args_test)
                xyz_nearest, color_nearest, opacity_nearest, scaling_nearest, rot_nearest = generate_neural_gaussians(viewpoint_camera_nearest, pc, visible_mask_nearest, mode == "test")
                tanfovx_nearest = math.tan(viewpoint_camera_nearest.FoVx * 0.5)
                tanfovy_nearest = math.tan(viewpoint_camera_nearest.FoVy * 0.5)

                raster_settings_nearest = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera_nearest.image_height),
                image_width=int(viewpoint_camera_nearest.image_width),
                tanfovx=tanfovx_nearest,
                tanfovy=tanfovy_nearest,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera_nearest.world_view_transform,
                projmatrix=viewpoint_camera_nearest.full_proj_transform,
                sh_degree=pc.active_sh_degree,
                campos=viewpoint_camera_nearest.camera_center,
                prefiltered=False,
                render_geo = return_plane,
                debug=pipe.debug,
                )
                batched_means3D_nearest.append(xyz_nearest)
                batched_opacity_nearest.append(color_nearest)
                batched_scales_nearest.append(opacity_nearest)
                batched_rotations_nearest.append(scaling_nearest)
                batched_colors_nearest.append(rot_nearest)
                tanfovx = math.tan(viewpoint_camera_nearest.FoVx * 0.5)
                tanfovy = math.tan(viewpoint_camera_nearest.FoVy * 0.5)
                rasterizer_nearest  = GaussianRasterizer(raster_settings=raster_settings_nearest)
                
                means2D_nearest, _, conic_opacity_nearest,  radii_nearest, depths_nearest = rasterizer_nearest.preprocess_gaussians(
                    means3D=xyz_nearest,
                    scales=scaling_nearest,
                    rotations=rot_nearest,
                    shs=color_nearest,
                    opacities=opacity_nearest,
                    cuda_args=cuda_args_test,
                )

                batched_means2D_nearest.append(means2D_nearest)
                screenspace_params_nearest = [means2D_nearest, color_nearest, conic_opacity_nearest, radii_nearest, depths_nearest, xyz_nearest, scaling_nearest, rot_nearest]
                batched_rasterizers_nearest.append(rasterizer_nearest)
                batched_screenspace_params_nearest.append(screenspace_params_nearest)
            # batched_screenspace_params_nearest.append(screenspace_params_nearest)
            # batched_radii_nearest.append(radii)

        # batched_out_observe.append(out_observe)
    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")

    if utils.DEFAULT_GROUP.size() == 1:
        if len(batched_rasterizers_nearest) != 0:  #near cam exits
            batched_screenspace_pkg = {
                "batched_locally_preprocessed_mean2D": batched_means2D,
                "batched_locally_preprocessed_visibility_filter": [
                    radii > 0 for radii in batched_radii
                ],
                "batched_locally_preprocessed_radii": batched_radii,
                "batched_locally_opacity": batched_neural_opacity,#gyy
                "batched_locally_offset_mask":batched_mask, #gyy
                "batched_locally_voxel_mask":batched_voxel_mask, #gyy
                "batched_rasterizers": batched_rasterizers,
                "batched_rasterizers_nearest": batched_rasterizers_nearest,
                "batched_cuda_args": batched_cuda_args,
                "batched_cuda_args_test": batched_cuda_args_test,
                "batched_means2D_redistributed": [
                    screenspace_params[0]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_rgb_redistributed": [
                    screenspace_params[1]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_conic_opacity_redistributed": [
                    screenspace_params[2]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_radii_redistributed": [
                    screenspace_params[3]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_depths_redistributed": [
                    screenspace_params[4]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_means3D_redistributed": [
                    screenspace_params[5]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_scales_redistributed": [
                    screenspace_params[6]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_rotations_redistributed": [
                    screenspace_params[7]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_means2D_redistributed_nearest": [
                    screenspace_params[0]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_rgb_redistributed_nearest": [
                    screenspace_params[1]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_conic_opacity_redistributed_nearest": [
                    screenspace_params[2]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_radii_redistributed_nearest": [
                    screenspace_params[3]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_depths_redistributed_nearest": [
                    screenspace_params[4]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_means3D_redistributed_nearest": [
                    screenspace_params[5]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_scales_redistributed_nearest": [
                    screenspace_params[6]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "batched_rotations_redistributed_nearest": [
                    screenspace_params[7]
                    for screenspace_params in batched_screenspace_params_nearest
                ],
                "gpui_to_gpuj_imgk_size": [
                    [[batched_means2D[i].shape[0] for i in range(len(batched_means2D))]]
                ],
            }
        else:
            batched_screenspace_pkg = {
                "batched_locally_preprocessed_mean2D": batched_means2D,
                "batched_locally_preprocessed_visibility_filter": [
                    radii > 0 for radii in batched_radii
                ],
                "batched_locally_preprocessed_radii": batched_radii,
                "batched_locally_opacity": batched_neural_opacity,            #gyy
                "batched_locally_offset_mask":batched_mask, 
                "batched_locally_voxel_mask":batched_voxel_mask, 
                "batched_rasterizers": batched_rasterizers,
                "batched_cuda_args": batched_cuda_args,
                "batched_means2D_redistributed": [
                    screenspace_params[0]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_rgb_redistributed": [
                    screenspace_params[1]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_conic_opacity_redistributed": [
                    screenspace_params[2]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_radii_redistributed": [
                    screenspace_params[3]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_depths_redistributed": [
                    screenspace_params[4]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_means3D_redistributed": [
                    screenspace_params[5]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_scales_redistributed": [
                    screenspace_params[6]
                    for screenspace_params in batched_screenspace_params
                ],
                "batched_rotations_redistributed": [
                    screenspace_params[7]
                    for screenspace_params in batched_screenspace_params
                ],
                "gpui_to_gpuj_imgk_size": [
                    [[batched_means2D[i].shape[0] for i in range(len(batched_means2D))]]
                ],
                "batched_means2D_redistributed_nearest": None,
            }
        return batched_screenspace_pkg

    if timers is not None:
        timers.start("forward_all_to_all_communication")
    (
        batched_means2D_redistributed,
        batched_rgb_redistributed,
        batched_conic_opacity_redistributed,
        batched_radii_redistributed,
        batched_depths_redistributed,
        batched_means3D_redistributed,
        batched_scales_redistributed,
        batched_rotations_redistributed,
        # batched_view_points_redistributed,
        # batched_ts_redistributed,
        # batched_camera_planes_redistributed,
        # batched_ray_plane_redistributed,
        # batched_normals_redistributed,
        gpui_to_gpuj_imgk_size,
        local_to_gpuj_camk_send_ids,
    ) = all_to_all_communication_final(
        batched_rasterizers,
        batched_screenspace_params,
        batched_cuda_args,
        batched_strategies,
    )

    if len(batched_rasterizers_nearest) != 0:
        # with torch.no_grad():
        (
            batched_means2D_redistributed_nearest,
            batched_rgb_redistributed_nearest,
            batched_conic_opacity_redistributed_nearest,
            batched_radii_redistributed_nearest,
            batched_depths_redistributed_nearest,
            batched_means3D_redistributed_nearest,
            batched_scales_redistributed_nearest,
            batched_rotations_redistributed_nearest,
            # batched_view_points_redistributed,
            # batched_ts_redistributed,
            # batched_camera_planes_redistributed,
            # batched_ray_plane_redistributed,
            # batched_normals_redistributed,
            _,
            _,
        ) = all_to_all_communication_final(
            batched_rasterizers_nearest,
            batched_screenspace_params_nearest,
            batched_cuda_args_test,
            batched_strategies,
        )
    else: 
        batched_means2D_redistributed_nearest = None
        batched_rgb_redistributed_nearest = None
        batched_conic_opacity_redistributed_nearest = None
        batched_radii_redistributed_nearest = None
        batched_depths_redistributed_nearest = None
        batched_means3D_redistributed_nearest = None
        batched_scales_redistributed_nearest = None
        batched_rotations_redistributed_nearest = None
        batched_rasterizers_nearest = None
        batched_cuda_args_test = None
    utils.check_initial_gpu_memory_usage("after forward_all_to_all_communication")
    if timers is not None:
        timers.stop("forward_all_to_all_communication")

    batched_screenspace_pkg = {
        "batched_locally_preprocessed_mean2D": batched_means2D,
        "batched_locally_preprocessed_visibility_filter": [
            radii > 0 for radii in batched_radii
        ],
        "batched_locally_preprocessed_radii": batched_radii,
        "batched_locally_opacity": batched_neural_opacity,#gyy
        "batched_locally_offset_mask":batched_mask, #gyy
        "batched_locally_voxel_mask":batched_voxel_mask, #gyy
        "batched_rasterizers": batched_rasterizers,
        "batched_cuda_args": batched_cuda_args,
        "batched_cuda_args_test": batched_cuda_args_test,
        "batched_means2D_redistributed": batched_means2D_redistributed,
        "batched_rgb_redistributed": batched_rgb_redistributed,
        "batched_conic_opacity_redistributed": batched_conic_opacity_redistributed,
        "batched_radii_redistributed": batched_radii_redistributed,
        "batched_depths_redistributed": batched_depths_redistributed,
        "batched_means3D_redistributed": batched_means3D_redistributed,
        "batched_scales_redistributed": batched_scales_redistributed,
        "batched_rotations_redistributed": batched_rotations_redistributed,
        "batched_means2D_redistributed_nearest":  batched_means2D_redistributed_nearest,
        "batched_rgb_redistributed_nearest" : batched_rgb_redistributed_nearest, 
        "batched_conic_opacity_redistributed_nearest": batched_conic_opacity_redistributed_nearest,
        "batched_radii_redistributed_nearest": batched_radii_redistributed_nearest,
        "batched_depths_redistributed_nearest":batched_depths_redistributed_nearest,
        "batched_means3D_redistributed_nearest":batched_means3D_redistributed_nearest,
        "batched_scales_redistributed_nearest":batched_scales_redistributed_nearest,
        "batched_rotations_redistributed_nearest":batched_rotations_redistributed_nearest,
        "batched_rasterizers_nearest": batched_rasterizers_nearest,
        "gpui_to_gpuj_imgk_size": gpui_to_gpuj_imgk_size,
        "local_to_gpuj_camk_send_ids":local_to_gpuj_camk_send_ids,
    }
    return batched_screenspace_pkg


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render_final(batched_cameras, batched_screenspace_pkg, batched_strategies,  tile_size=16 , batched_cameras_nearest =None, rasterizer_nearest = None):
    """
    Render the scene.
    """
    timers = utils.get_timers()

    batched_rendered_image = []
    batched_compute_locally = []
    batched_rendered_normal = []
    batched_out_observe = []
    batched_out_all_map = []
    batched_out_plane_depth = []
    batched_return_dict = []
    batched_return_dict_nearest = []

    for cam_id in range(len(batched_screenspace_pkg["batched_rasterizers"])):
        strategy = batched_strategies[cam_id]
        if utils.GLOBAL_RANK not in strategy.gpu_ids:
            batched_rendered_image.append(None)
            batched_compute_locally.append(None)
            batched_out_observe.append(None)
            return_dict =  {
                # "render": rendered_image,
                "viewspace_points": None,
                "viewspace_points_abs": None,
                "visibility_filter" : None,
                "radii": None,
                "out_observe": None,
                "rendered_normal": None,
                "plane_depth": None,
                "rendered_distance": None,
                "depth_normal": None
                }
            batched_return_dict.append(return_dict)
            batched_return_dict_nearest.append(None)
            continue

        # get compute_locally to know local workload in the end2end distributed training.
        if timers is not None:
            timers.start("forward_compute_locally")
        compute_locally = strategy.get_compute_locally()
        extended_compute_locally = strategy.get_extended_compute_locally()
        if timers is not None:
            timers.stop("forward_compute_locally")

        rasterizer = batched_screenspace_pkg["batched_rasterizers"][cam_id]
        cuda_args = batched_screenspace_pkg["batched_cuda_args"][cam_id]
        means2D_redistributed = batched_screenspace_pkg[
            "batched_means2D_redistributed"
        ][cam_id]
        rgb_redistributed = batched_screenspace_pkg["batched_rgb_redistributed"][cam_id]
        conic_opacity_redistributed = batched_screenspace_pkg[
            "batched_conic_opacity_redistributed"
        ][cam_id]
        radii_redistributed = batched_screenspace_pkg["batched_radii_redistributed"][
            cam_id
        ]
        depths_redistributed = batched_screenspace_pkg["batched_depths_redistributed"][
            cam_id
        ]
        xyz_redistributed = batched_screenspace_pkg["batched_means3D_redistributed"][cam_id]
        scales_redistributed = batched_screenspace_pkg["batched_scales_redistributed"][cam_id]
        rotations_redistributed = batched_screenspace_pkg["batched_rotations_redistributed"][cam_id]




        if  batched_screenspace_pkg["batched_means2D_redistributed_nearest"] != None:

            means2D_redistributed_nearest  = batched_screenspace_pkg["batched_means2D_redistributed_nearest"][cam_id]
            rgb_redistributed_nearest = batched_screenspace_pkg["batched_rgb_redistributed_nearest"][cam_id]
            # rgb_redistributed_nearest = batched_screenspace_pkg["batched_rgb_redistributed"][cam_id]
            conic_opacity_redistributed_nearest = batched_screenspace_pkg["batched_conic_opacity_redistributed_nearest"][cam_id]
            radii_redistributed_nearest = batched_screenspace_pkg["batched_radii_redistributed_nearest"][cam_id]
            depths_redistributed_nearest = batched_screenspace_pkg["batched_depths_redistributed_nearest"][cam_id]
            xyz_redistributed_nearest = batched_screenspace_pkg["batched_means3D_redistributed_nearest"][cam_id]
            scales_redistributed_nearset = batched_screenspace_pkg["batched_scales_redistributed_nearest"][cam_id]
            rotations_redistributed_nearest =batched_screenspace_pkg["batched_rotations_redistributed_nearest"][cam_id]
            rasterizer_nearest = batched_screenspace_pkg["batched_rasterizers_nearest"][cam_id]
            cuda_args_test = batched_screenspace_pkg["batched_cuda_args_test"][cam_id]
        if means2D_redistributed.shape[0] < 10:
            # That means we do not have enough gaussians locally for rendering, that mainly happens because of insufficient initial points.
            rendered_image = (
                means2D_redistributed.sum()
                + conic_opacity_redistributed.sum()
                + rgb_redistributed.sum()
            )
            cuda_args["stats_collector"]["forward_render_time"] = 0.0
            cuda_args["stats_collector"]["backward_render_time"] = 0.0
            cuda_args["stats_collector"]["forward_loss_time"] = 0.0

            return_dict =  {
                # "render": rendered_image,
                "viewspace_points": None,
                "viewspace_points_abs": None,
                "visibility_filter" : None,
                "radii": None,
                "out_observe": None,
                "rendered_normal": None,
                "plane_depth": None,
                "rendered_distance": None,
                "depth_normal": None
                }
            out_all_map = None
            out_observe = None
            out_plane_depth = None
            nearest_render_pkg = None
        else:
            nearest_render_pkg = None
            if rasterizer.raster_settings.render_geo == True:
                # nearest_render_pkg = None
                if eval == True:
                    rasterizer_nearest = None
                if rasterizer_nearest is not None:
                    with torch.no_grad():
                        rotation_matrices = quaternion_to_matrix(torch.nn.functional.normalize(rotations_redistributed_nearest, p=2, dim=-1))
                        smallest_axis_idx =torch.exp(scales_redistributed_nearset).min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
                        
                        normal_global = rotation_matrices.gather(2, smallest_axis_idx).squeeze(dim=2)
                        
                        gaussian_to_cam_global = batched_cameras_nearest[cam_id].camera_center - xyz_redistributed_nearest
                        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
                        normal_global[neg_mask] = -normal_global[neg_mask]
                        global_normal = normal_global
                        
                        local_normal = global_normal @ batched_cameras_nearest[cam_id].world_view_transform[:3,:3]
                        pts_in_cam = xyz_redistributed_nearest @ batched_cameras_nearest[cam_id].world_view_transform[:3,:3] + batched_cameras_nearest[cam_id].world_view_transform[3,:3]
                        depth_z = pts_in_cam[:, 2]
                        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                        input_all_map_1 = torch.zeros((means2D_redistributed_nearest.shape[0], 5)).cuda().float()
                        input_all_map_1[:, :3] = local_normal
                        input_all_map_1[:, 3] = 1.0
                        input_all_map_1[:, 4] = local_distance
                        screenspace_points_abs_1 = torch.zeros_like(xyz_redistributed_nearest, dtype=means2D_redistributed_nearest.dtype, device="cuda") + 0
                        # try:
                        #     screenspace_points_abs_1.retain_grad()
                        #     # screenspace_points_abs.retain_grad()
                        # except:
                        #     pass
                        means2D_abs_1 = screenspace_points_abs_1
                        rendered_image_nearest, n_render, n_consider, n_contrib, out_observe, out_all_map_nearest, out_plane_depth_nearest = (
                            rasterizer_nearest.render_gaussians(
                                means2D=means2D_redistributed_nearest,
                                means2D_abs= means2D_abs_1,
                                conic_opacity=conic_opacity_redistributed_nearest, 
                                rgb=rgb_redistributed_nearest,
                                all_map = input_all_map_1,
                                depths=depths_redistributed_nearest,
                                radii=radii_redistributed_nearest,
                                compute_locally=compute_locally,
                                extended_compute_locally=extended_compute_locally,
                                cuda_args=cuda_args_test,
                            )
                        )
                        rendered_normal = out_all_map_nearest[0:3]
                        rendered_alpha = out_all_map_nearest[3:4, ]
                        rendered_distance = out_all_map_nearest[4:5, ]
                        nearest_render_pkg =  {
                        # "render": rendered_image,
                        # "viewspace_points": means2D_redistributed,
                        # "viewspace_points_abs": means2D_abs,
                        # "visibility_filter" : radii_redistributed > 0,
                        # "radii": radii_redistributed,
                        # "out_observe": out_observe,
                        "rendered_normal": rendered_normal,
                        "plane_depth": out_plane_depth_nearest,
                        "rendered_distance": rendered_distance,
                        # "depth_normal": depth_normal,
                        # "scales_redistributed": scales_redistributed,
                        "rendered_alpha": rendered_alpha
                            }
            else: 
                input_all_map = torch.zeros((means2D_redistributed.shape[0], 5)).cuda().float()
            # render
            if timers is not None:
                timers.start("forward_render_gaussians")
            rotation_matrices = quaternion_to_matrix(torch.nn.functional.normalize(rotations_redistributed, p=2, dim=-1))
            smallest_axis_idx =torch.exp(scales_redistributed).min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
            
            normal_global = rotation_matrices.gather(2, smallest_axis_idx).squeeze(dim=2)
            
            gaussian_to_cam_global = batched_cameras[cam_id].camera_center - xyz_redistributed
            neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
            normal_global[neg_mask] = -normal_global[neg_mask]
            global_normal = normal_global
            
            local_normal = global_normal @ batched_cameras[cam_id].world_view_transform[:3,:3]
            pts_in_cam = xyz_redistributed @ batched_cameras[cam_id].world_view_transform[:3,:3] + batched_cameras[cam_id].world_view_transform[3,:3]
            depth_z = pts_in_cam[:, 2]
            local_distance = (local_normal * pts_in_cam).sum(-1).abs()
            input_all_map = torch.zeros((means2D_redistributed.shape[0], 5)).cuda().float()
            input_all_map[:, :3] = local_normal
            input_all_map[:, 3] = 1.0
            input_all_map[:, 4] = local_distance

            screenspace_points_abs = torch.zeros_like(xyz_redistributed, dtype=means2D_redistributed.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points_abs.retain_grad()
                # screenspace_points_abs.retain_grad()
            except:
                pass
            means2D_abs = screenspace_points_abs
            rendered_image, n_render, n_consider, n_contrib, out_observe, out_all_map, out_plane_depth = (
                rasterizer.render_gaussians(
                    means2D=means2D_redistributed,
                    means2D_abs= means2D_abs,
                    conic_opacity=conic_opacity_redistributed, 
                    rgb=rgb_redistributed,
                    all_map = input_all_map,
                    depths=depths_redistributed,
                    radii=radii_redistributed,
                    compute_locally=compute_locally,
                    extended_compute_locally=extended_compute_locally,
                    cuda_args=cuda_args,
                )
            )
            rendered_normal = out_all_map[0:3]
            rendered_alpha = out_all_map[3:4, ]
            rendered_distance = out_all_map[4:5, ]

            depth_normal = render_normal(batched_cameras[cam_id], out_plane_depth.squeeze()) * (rendered_alpha).detach()
            return_dict =  {
                        # "render": rendered_image,
                        "viewspace_points": means2D_redistributed,
                        "viewspace_points_abs": means2D_abs,
                        "visibility_filter" : radii_redistributed > 0,
                        "radii": radii_redistributed,
                        "out_observe": out_observe,
                        "rendered_normal": rendered_normal,
                        "plane_depth": out_plane_depth,
                        "rendered_distance": rendered_distance,
                        "depth_normal": depth_normal,
                        "scales_redistributed": scales_redistributed,
                        "rendered_alpha": rendered_alpha
                        }
        batched_rendered_image.append(rendered_image)
        batched_out_all_map.append(out_all_map)
        batched_out_observe.append(out_observe)
        batched_out_plane_depth.append(out_plane_depth)
        batched_return_dict.append(return_dict)
        batched_return_dict_nearest.append(nearest_render_pkg)

        if timers is not None:
            timers.stop("forward_render_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return batched_rendered_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, batched_return_dict_nearest
