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
from gsplat import (
    rasterization,
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
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
    means3D = pc.get_xyz
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
    means3D = pc.get_xyz
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

    local_to_gpuj_camk_size = [[] for j in range(utils.DEFAULT_GROUP.size())]  #当前到各个gpu的点的数量
    local_to_gpuj_camk_send_ids = [[] for j in range(utils.DEFAULT_GROUP.size())] #对应的means2d在改线程的索引
    for k in range(num_cameras):
        strategy = batched_strategies[k]
        means2D, rgb, conic_opacity, radii, depths, means3D, scales, rotation = batched_screenspace_params[k]
        local2j_ids, local2j_ids_bool = batched_strategies[k].get_local2j_ids(
            means2D, radii, batched_rasterizers[k].raster_settings, batched_cuda_args[k]
        ) #这里少点很奇怪
        #这里为什么means2d shape减小了 按道理全是相机1平面上且在cuda0上的高斯点呀        #当前线程的高斯点的idx local2j_ids, 需要到gpu0
        for local_id, global_id in enumerate(strategy.gpu_ids):
            local_to_gpuj_camk_size[global_id].append(len(local2j_ids[local_id]))
            local_to_gpuj_camk_send_ids[global_id].append(local2j_ids[local_id]) #线程0到

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
        local_to_gpuj_camk_size_tensor,  #当前线程 到 0 1 2 3的线程的点个数
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
                    batched_tensors[k][local_to_gpuj_camk_send_ids[i][k]]  #i 分配后的gpu位置，k表示该gpu上点属于那些camera
                )
                #tensor_to_rki_list cuda n上所有的高斯点在local的索引
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
            )#这里保存相机对应的信息
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
    )


def gsplat_all_to_all_communication_final(
    batched_screenspace_params, batched_cuda_args, batched_strategies
):
    num_cameras = len(batched_strategies)
    (
        image_height,
        image_width,
        batched_means2D,
        batched_colors,
        batched_conics,
        batched_opacities,
        batched_radiis,
        batched_depths,
    ) = batched_screenspace_params
    # gpui_to_gpuj_camk_size
    # gpui_to_gpuj_camk_send_ids

    local_to_gpuj_camk_size = [[] for j in range(utils.DEFAULT_GROUP.size())]
    local_to_gpuj_camk_send_ids = [[] for j in range(utils.DEFAULT_GROUP.size())]
    for k in range(num_cameras):
        strategy = batched_strategies[k]
        local2j_ids, local2j_ids_bool = batched_strategies[k].gsplat_get_local2j_ids(
            batched_means2D[k],
            batched_radiis[k],
            image_height,
            image_width,
            batched_cuda_args[k],
        )

        for local_id, global_id in enumerate(strategy.gpu_ids):
            local_to_gpuj_camk_size[global_id].append(len(local2j_ids[local_id]))
            local_to_gpuj_camk_send_ids[global_id].append(local2j_ids[local_id])

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
        local_to_gpuj_camk_size_tensor,
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
                    batched_tensors[k][local_to_gpuj_camk_send_ids[i][k]]
                )
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
            )

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

    # Merge means2D, colors, conics, opacities into one functional all-to-all communication call.
    mean2d_dim2 = batched_means2D.shape[2]
    color_dim2 = batched_colors.shape[2]
    conic_dim2 = batched_conics.shape[2]

    batched_catted_screenspace_states = torch.cat(
        [
            batched_means2D,
            batched_colors,
            batched_conics,
            batched_opacities.unsqueeze(2),
        ],
        dim=2,
    ).contiguous()
    batched_catted_screenspace_auxiliary_states = torch.cat(
        [batched_radiis.unsqueeze(2), batched_depths.unsqueeze(2)], dim=2
    ).contiguous()

    batched_params_redistributed = one_all_to_all(
        batched_catted_screenspace_states, use_function_version=True
    )

    batched_means2D_redistributed = []
    batched_colors_redistributed = []
    batched_conics_redistributed = []
    batched_opacities_redistributed = []

    for k in range(num_cameras):
        (
            mean2D_redistributed,
            color_redistributed,
            conic_redistributed,
            opacity_redistributed,
        ) = torch.split(
            batched_params_redistributed[k],
            [mean2d_dim2, color_dim2, conic_dim2, 1],
            dim=1,
        )
        batched_means2D_redistributed.append(mean2D_redistributed)
        batched_colors_redistributed.append(color_redistributed)
        batched_conics_redistributed.append(conic_redistributed)
        batched_opacities_redistributed.append(opacity_redistributed)

    batched_radii_depth_redistributed = one_all_to_all(
        batched_catted_screenspace_auxiliary_states, use_function_version=False
    )

    batched_radiis_redistributed = []
    batched_depths_redistributed = []
    for k in range(num_cameras):
        radii_redistributed, depth_redistributed = torch.split(
            batched_radii_depth_redistributed[k], [1, 1], dim=1
        )
        batched_radiis_redistributed.append(radii_redistributed.squeeze(1).int())
        batched_depths_redistributed.append(depth_redistributed.squeeze(1))

    return (
        batched_means2D_redistributed,
        batched_colors_redistributed,
        batched_conics_redistributed,
        batched_opacities_redistributed,
        batched_radiis_redistributed,
        batched_depths_redistributed,
        gpui_to_gpuj_imgk_size,
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
    retain_grad=False,
    batched_strategies=None,
    mode="train",
    return_plane : bool = True
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
    for viewpoint_camera,visible_mask,strategy in zip(batched_viewpoint_cameras, batched_voxel_mask, batched_strategies):
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
        
    # if timers is not None:
    #     timers.stop("forward_prepare_gaussians")
    # utils.check_initial_gpu_memory_usage("after forward_prepare_gaussians")
    # ########## [END] Prepare Gaussians for rendering ##########

    # if timers is not None:
    #     timers.start("forward_preprocess_gaussians")
    
    # test
    # means3D = xyz
    # scales = scaling
    # rotations = rot
    # color = color
    # opacity = opacity
    # test
    
    # for i, (viewpoint_camera, strategy, means3D, color, opacity, scales, rotations) in enumerate(
    #     zip(batched_viewpoint_cameras, batched_strategies, batched_means3D, batched_colors, batched_opacity, batched_scales, batched_rotations)
    # ):
    # for i, (viewpoint_camera, strategy) in enumerate(
    #     zip(batched_viewpoint_cameras, batched_strategies)
    # ):
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
    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")

    if utils.DEFAULT_GROUP.size() == 1:
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
            # "batched_view_points_redistributed": [
            #     screenspace_params[5]
            #     for screenspace_params in batched_screenspace_params
            # ],
            # "batched_ts_redistributed": [
            #     screenspace_params[6]
            #     for screenspace_params in batched_screenspace_params
            # ],
            # "batched_camera_planes_redistributed": [
            #     screenspace_params[7]
            #     for screenspace_params in batched_screenspace_params
            # ],
            # "batched_ray_plane_redistributed": [
            #     screenspace_params[8]
            #     for screenspace_params in batched_screenspace_params
            # ],
            # "batched_normals_redistributed": [
            #     screenspace_params[9]
            #     for screenspace_params in batched_screenspace_params
            # ],
            "gpui_to_gpuj_imgk_size": [
                [[batched_means2D[i].shape[0] for i in range(len(batched_means2D))]]
            ],
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
    ) = all_to_all_communication_final(
        batched_rasterizers,
        batched_screenspace_params,
        batched_cuda_args,
        batched_strategies,
    )
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
        "batched_means2D_redistributed": batched_means2D_redistributed,
        "batched_rgb_redistributed": batched_rgb_redistributed,
        "batched_conic_opacity_redistributed": batched_conic_opacity_redistributed,
        "batched_radii_redistributed": batched_radii_redistributed,
        "batched_depths_redistributed": batched_depths_redistributed,
        "batched_means3D_redistributed": batched_means3D_redistributed,
        "batched_scales_redistributed": batched_scales_redistributed,
        "batched_rotations_redistributed": batched_rotations_redistributed,
        # "batched_view_points_redistributed":batched_view_points_redistributed,
        # "batched_ts_redistributed":   batched_ts_redistributed,
        # "batched_camera_planes_redistributed": batched_camera_planes_redistributed,
        # "batched_ray_plane_redistributed": batched_ray_plane_redistributed,
        # "batched_normals_redistributed":   batched_normals_redistributed,
        "gpui_to_gpuj_imgk_size": gpui_to_gpuj_imgk_size,
    }
    return batched_screenspace_pkg


def gsplat_distributed_preprocess3dgs_and_all2all_final(
    batched_viewpoint_cameras,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    batched_strategies=None,
    mode="train",
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
    means3D = pc.get_xyz
    opacities = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    shs = pc.get_features  # (N, K, 3)
    sh_degree = pc.active_sh_degree
    if timers is not None:
        timers.stop("forward_prepare_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########

    if timers is not None:
        timers.start("forward_preprocess_gaussians")
    batched_cuda_args = []
    batched_screenspace_params = []

    N = means3D.shape[0]  # number of gaussians
    B = len(batched_viewpoint_cameras)  # number of cameras (aka batch size)

    Ks = []
    viewmats = []

    for viewpoint_camera, strategy in zip(
        batched_viewpoint_cameras, batched_strategies
    ):
        cuda_args = get_cuda_args_final(strategy, mode)
        batched_cuda_args.append(cuda_args)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )
        viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # why transpose
        Ks.append(K)
        viewmats.append(viewmat)

    batched_Ks = torch.stack(Ks)  # (B, 3, 3)
    batched_viewmats = torch.stack(viewmats)  # (B, 4, 4)
    image_width = int(batched_viewpoint_cameras[0].image_width)
    image_height = int(batched_viewpoint_cameras[0].image_height)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=means3D,  # (N, 3)
            covars=None,
            quats=rotations,
            scales=scales,
            viewmats=batched_viewmats,
            Ks=batched_Ks,
            width=image_width,
            height=image_height,
            packed=False,
        )
    )
    batched_opacities = opacities.squeeze(1).repeat(B, 1)  # (N, 1) -> (B, N)

    if mode == "train":
        batched_means2D.retain_grad()

    # Compute colors(shs) for each camera view.
    shs = shs.expand(B, *([-1] * shs.dim()))

    camtoworlds = torch.inverse(batched_viewmats)
    dirs = means3D[None, :, :] - camtoworlds[:, None, :3, 3]
    batched_colors = spherical_harmonics(
        degrees_to_use=sh_degree, dirs=dirs, coeffs=shs, masks=(batched_radiis > 0)
    )
    batched_colors = torch.clamp_min(batched_colors + 0.5, 0.0)  # (B, N, 3)

    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")

    if utils.DEFAULT_GROUP.size() == 1:
        batched_screenspace_pkg = {
            "image_height": image_height,
            "image_width": image_width,
            "backgrounds": bg_color,  # default: None
            "batched_locally_preprocessed_mean2D": batched_means2D,
            "batched_locally_preprocessed_visibility_filter": (batched_radiis > 0),
            "batched_locally_preprocessed_radii": batched_radiis,
            "batched_cuda_args": batched_cuda_args,
            "batched_means2D_redistributed": batched_means2D,
            "batched_colors_redistributed": batched_colors,
            "batched_conics_redistributed": batched_conics,  # Inverse of the projected covariances. Flattened upper triangle with (B, N, 3).
            "batched_opacities_redistributed": batched_opacities,
            "batched_radiis_redistributed": batched_radiis,
            "batched_depths_redistributed": batched_depths,
            "gpui_to_gpuj_imgk_size": [
                [[batched_means2D[i].shape[0] for i in range(B)]]
            ],
        }

        return batched_screenspace_pkg

    batched_screenspace_params = [
        image_height,
        image_width,
        batched_means2D,  # (B, N, 2)
        batched_colors,  # (B, N, 3)
        batched_conics,  # (B, N, 3)
        batched_opacities,  # (B, N)
        batched_radiis,  # (B, N)
        batched_depths,  # (B, N)
    ]

    if timers is not None:
        timers.start("forward_all_to_all_communication")
    (
        batched_means2D_redistributed,
        batched_colors_redistributed,
        batched_conics_redistributed,
        batched_opacities_redistributed,
        batched_radiis_redistributed,
        batched_depths_redistributed,
        gpui_to_gpuj_imgk_size,
    ) = gsplat_all_to_all_communication_final(
        batched_screenspace_params, batched_cuda_args, batched_strategies
    )
    utils.check_initial_gpu_memory_usage("after forward_all_to_all_communication")
    if timers is not None:
        timers.stop("forward_all_to_all_communication")

    batched_screenspace_pkg = {
        "image_height": image_height,
        "image_width": image_width,
        "backgrounds": bg_color,  # default: None
        "batched_locally_preprocessed_mean2D": batched_means2D,
        "batched_locally_preprocessed_visibility_filter": (batched_radiis > 0),
        "batched_locally_preprocessed_radii": batched_radiis,
        "batched_cuda_args": batched_cuda_args,
        "batched_means2D_redistributed": batched_means2D_redistributed,
        "batched_colors_redistributed": batched_colors_redistributed,
        "batched_conics_redistributed": batched_conics_redistributed,  # Inverse of the projected covariances. Flattened upper triangle with (B, N, 3).
        "batched_opacities_redistributed": batched_opacities_redistributed,
        "batched_radiis_redistributed": batched_radiis_redistributed,
        "batched_depths_redistributed": batched_depths_redistributed,
        "gpui_to_gpuj_imgk_size": gpui_to_gpuj_imgk_size,
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

def render_final(batched_cameras, batched_screenspace_pkg, batched_strategies,  tile_size=16):
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


    for cam_id in range(len(batched_screenspace_pkg["batched_rasterizers"])):
        strategy = batched_strategies[cam_id]
        if utils.GLOBAL_RANK not in strategy.gpu_ids:
            batched_rendered_image.append(None)
            batched_compute_locally.append(None)
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
        # view_points_redistibuted = batched_screenspace_pkg["batched_view_points_redistributed"][cam_id]
        # ts_redistributed = batched_screenspace_pkg["batched_ts_redistributed"][cam_id]
        # camera_planes_redistributed = batched_screenspace_pkg["batched_camera_planes_redistributed"][cam_id]
        # ray_plane_redistributed = batched_screenspace_pkg["batched_ray_plane_redistributed"][cam_id]
        # normals_redistributed = batched_screenspace_pkg["batched_normals_redistributed"][cam_id]



        # render
        if timers is not None:
            timers.start("forward_render_gaussians")
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
        else:
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
                        "scales_redistributed": scales_redistributed
                        }
        batched_rendered_image.append(rendered_image)
        batched_out_all_map.append(out_all_map)
        batched_out_observe.append(out_observe)
        batched_out_plane_depth.append(out_plane_depth)
        batched_return_dict.append(return_dict)


        if timers is not None:
            timers.stop("forward_render_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return batched_rendered_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict
def gsplat_render_final(batched_screenspace_pkg, batched_strategies, tile_size=16):
    """
    Render the scene.
    """
    timers = utils.get_timers()
    B = len(batched_strategies)

    # If there's only one gpu, use batched kernel.
    if utils.DEFAULT_GROUP.size() == 1:
        means2D = batched_screenspace_pkg["batched_means2D_redistributed"]
        radiis = batched_screenspace_pkg["batched_radiis_redistributed"]
        depths = batched_screenspace_pkg["batched_depths_redistributed"]
        conics = batched_screenspace_pkg["batched_conics_redistributed"]
        colors = batched_screenspace_pkg["batched_colors_redistributed"]
        opacities = batched_screenspace_pkg["batched_opacities_redistributed"]
        image_width = batched_screenspace_pkg["image_width"]
        image_height = batched_screenspace_pkg["image_height"]
        backgrounds = (
            batched_screenspace_pkg["backgrounds"].repeat(B, 1)
            if batched_screenspace_pkg["backgrounds"] is not None
            else None
        )
        batched_cuda_args = batched_screenspace_pkg["batched_cuda_args"]

        # Identify intersecting tiles.
        tile_width = math.ceil(image_width / float(tile_size))
        tile_height = math.ceil(image_height / float(tile_size))

        # flatten_ids: (C*N)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d=means2D,
            radii=radiis,
            depths=depths,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            packed=False,
        )
        isect_offsets = isect_offset_encode(
            isect_ids, B, tile_width, tile_height
        )  # (B, tile_height, tile_width)

        # TODO: One way to do load balancing: Add two timer operators before and after `rasterize_to_pixels`
        # record_time_start : torch operator(torch.autograd.func)

        # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
        rendered_images, _ = rasterize_to_pixels(
            means2d=means2D,
            conics=conics,
            colors=colors,
            opacities=opacities,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=backgrounds,
        )

        # record_time_finish
        # order: in/out dependency
        # potential memory overhead: grad

        batched_rendered_image = list(
            torch.unbind(rendered_images.permute(0, 3, 1, 2), dim=0)
        )
        batched_compute_locally = []

        for strategy, cuda_args in zip(batched_strategies, batched_cuda_args):
            compute_locally = strategy.get_compute_locally()
            batched_compute_locally.append(compute_locally)

            cuda_args["stats_collector"]["forward_render_time"] = 0.0
            cuda_args["stats_collector"]["backward_render_time"] = 0.0
            cuda_args["stats_collector"]["forward_loss_time"] = 0.0

    else:
        batched_rendered_image = []
        batched_compute_locally = []

        image_width = batched_screenspace_pkg["image_width"]
        image_height = batched_screenspace_pkg["image_height"]

        for cam_id in range(B):
            strategy = batched_strategies[cam_id]
            if utils.GLOBAL_RANK not in strategy.gpu_ids:
                batched_rendered_image.append(None)
                batched_compute_locally.append(None)
                continue

            # get compute_locally to know local workload in the end2end distributed training.
            if timers is not None:
                timers.start("forward_compute_locally")
            compute_locally = strategy.get_compute_locally()
            extended_compute_locally = strategy.get_extended_compute_locally()
            if timers is not None:
                timers.stop("forward_compute_locally")

            means2D_redistributed = batched_screenspace_pkg[
                "batched_means2D_redistributed"
            ][cam_id].unsqueeze(0)
            radiis_redistributed = batched_screenspace_pkg[
                "batched_radiis_redistributed"
            ][cam_id].unsqueeze(0)
            depths_redistributed = batched_screenspace_pkg[
                "batched_depths_redistributed"
            ][cam_id].unsqueeze(0)
            conics_redistributed = batched_screenspace_pkg[
                "batched_conics_redistributed"
            ][cam_id].unsqueeze(0)
            colors_redistributed = batched_screenspace_pkg[
                "batched_colors_redistributed"
            ][cam_id].unsqueeze(0)
            opacities_redistributed = (
                batched_screenspace_pkg["batched_opacities_redistributed"][cam_id]
                .squeeze(1)
                .unsqueeze(0)
            )
            backgrounds = (
                batched_screenspace_pkg["backgrounds"].unsqueeze(0)
                if batched_screenspace_pkg["backgrounds"] is not None
                else None
            )
            cuda_args = batched_screenspace_pkg["batched_cuda_args"][cam_id]

            # render
            if timers is not None:
                timers.start("forward_render_gaussians")

            if means2D_redistributed.shape[1] < 10:
                # That means we do not have enough gaussians locally for rendering, that mainly happens because of insufficient initial points.
                rendered_image = (
                    means2D_redistributed.sum()
                    + opacities_redistributed.sum()
                    + colors_redistributed.sum()
                )
                cuda_args["stats_collector"]["forward_render_time"] = 0.0
                cuda_args["stats_collector"]["backward_render_time"] = 0.0
                cuda_args["stats_collector"]["forward_loss_time"] = 0.0
            else:
                # TODO: fix stats_collector
                cuda_args["stats_collector"]["forward_render_time"] = 0.1
                cuda_args["stats_collector"]["backward_render_time"] = 0.1
                cuda_args["stats_collector"]["forward_loss_time"] = 0.1

                # Identify intersecting tiles.
                tile_width = math.ceil(image_width / float(tile_size))
                tile_height = math.ceil(image_height / float(tile_size))

                # flatten_ids: (C*N)
                _, isect_ids, flatten_ids = isect_tiles(
                    means2d=means2D_redistributed,
                    radii=radiis_redistributed,
                    depths=depths_redistributed,
                    tile_size=tile_size,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    packed=False,
                )
                isect_offsets = isect_offset_encode(
                    isect_ids, 1, tile_width, tile_height
                )  # (B, tile_height, tile_width)

                # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
                rendered_image, _ = rasterize_to_pixels(
                    means2d=means2D_redistributed,
                    conics=conics_redistributed,
                    colors=colors_redistributed,
                    opacities=opacities_redistributed,
                    image_width=image_width,
                    image_height=image_height,
                    tile_size=tile_size,
                    isect_offsets=isect_offsets,
                    flatten_ids=flatten_ids,
                    backgrounds=backgrounds,
                    masks=compute_locally.unsqueeze(0),
                )

                rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()
            batched_rendered_image.append(rendered_image)
            batched_compute_locally.append(compute_locally)

            if timers is not None:
                timers.stop("forward_render_gaussians")
        utils.check_initial_gpu_memory_usage("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return batched_rendered_image, batched_compute_locally
