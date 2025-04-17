import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
    prefilter_voxel,
)
from torch.cuda import nvtx
from scene import Scene, GaussianModel, SceneDataset
from gaussian_renderer.workload_division import (
    start_strategy_final,
    finish_strategy_final,
    DivisionStrategyHistoryFinal,
)
from gaussian_renderer.loss_distribution import (
    load_camera_from_cpu_to_all_gpu,
    load_depth_from_cpu_to_all_gpu,
    load_gray_image_from_cpu_to_all_gpu,
    load_camera_from_cpu_to_all_gpu_for_eval,
    batched_loss_computation,
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification
import pdb
import torchvision
from os import makedirs
from utils.camera_utils import set_rays_od
import cv2
import numpy as np
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
def sync_model_with_rank0(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, 0)
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

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1

    # Init parameterized scene
    
    gaussians = GaussianModel(
        dataset_args.feat_dim, dataset_args.n_offsets, dataset_args.fork, dataset_args.use_feat_bank, dataset_args.appearance_dim, 
        dataset_args.add_opacity_dist, dataset_args.add_cov_dist, dataset_args.add_color_dist, dataset_args.add_level, 
        dataset_args.visible_threshold, dataset_args.dist2level, dataset_args.base_layer, dataset_args.progressive, dataset_args.extend
    )
    if utils.WORLD_SIZE > 1:
        sync_model_with_rank0(gaussians.mlp_color)
        sync_model_with_rank0(gaussians.mlp_cov)
        sync_model_with_rank0(gaussians.mlp_opacity)
    with torch.no_grad():
        scene = Scene(args, gaussians,args.load_iteration)
        gaussians.training_setup(opt_args)
        gaussians.set_coarse_interval(opt_args.coarse_iter, opt_args.coarse_factor)
        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0(
                "Restored from checkpoint: {}".format(args.start_checkpoint)
            )
            log_file.write(
                "Restored from checkpoint: {}\n".format(args.start_checkpoint)
            )

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # Init dataset
    # set_rays_od(scene.getTrainCameras())
    train_dataset = SceneDataset(scene.getTrainCameras())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.

    # Init distribution strategy history
    strategy_history = DivisionStrategyHistoryFinal(
        train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    # Init background
    background = None

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    ema_loss_for_log = 0
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # Step Initialization
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)
        num_trained_batches += 1
        timers.clear()
        if args.nsys_profile:
            nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # Prepare data: Pick random Cameras for training
        if args.local_sampling:
            assert (
                args.bsz % utils.WORLD_SIZE == 0
            ), "Batch size should be divisible by the number of GPUs."
            batched_cameras_idx = train_dataset.get_batched_cameras_idx(
                args.bsz // utils.WORLD_SIZE
            )
            batched_all_cameras_idx = torch.zeros(
                (utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int
            )
            batched_cameras_idx = torch.tensor(
                batched_cameras_idx, device="cuda", dtype=int
            )
            torch.distributed.all_gather_into_tensor(
                batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP
            )
            batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
            batched_cameras = train_dataset.get_batched_cameras_from_idx(
                batched_all_cameras_idx
            )
        else:
            batched_cameras, batched_nearest_cameras = train_dataset.get_batched_cameras(args.bsz)

        with torch.no_grad():
            # Prepare Workload division strategy
            timers.start("prepare_strategies")
            batched_strategies, gpuid2tasks = start_strategy_final(
                batched_cameras, strategy_history
            )
            # batched_strategies_nearest, gpuid2tasks_nearest = start_strategy_final(
            #     batched_nearest_cameras, strategy_history
            # )                                     # make sure don't divide the same camera twice  # gyy
            timers.stop("prepare_strategies")

            # Load ground-truth images to GPU
            timers.start("load_cameras")
            load_camera_from_cpu_to_all_gpu(
                batched_cameras, batched_strategies, gpuid2tasks
            )

            if iteration >  opt_args.multi_view_weight_from_iter and args.distributed_dataset_storage:

                load_gray_image_from_cpu_to_all_gpu(
                    batched_cameras, batched_strategies, gpuid2tasks
                                                    )
                load_gray_image_from_cpu_to_all_gpu(
                    batched_nearest_cameras, batched_strategies, gpuid2tasks)

            if iteration > opt_args.dpt_loss_from_iter and args.distributed_dataset_storage:
                load_depth_from_cpu_to_all_gpu(
                     batched_cameras, batched_strategies, gpuid2tasks
                )



            # load_camera_from_cpu_to_all_gpu(
            #     batched_nearest_cameras, batched_strategies, gpuid2tasks  # make sure don't divide the same camera twice  # gyy
            # )
            timers.stop("load_cameras")
        # pdb.set_trace()
        batched_voxel_mask = [] 
        # camera_t = []
        # cams=[]
        batched_nearest_voxel_mask = []
        for camera in batched_cameras:
            gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
            voxel_visible_mask = prefilter_voxel(camera, gaussians, pipe_args, background)
            batched_voxel_mask.append(voxel_visible_mask)
            # camera_t.append(camera.camera_center/torch.norm(camera.camera_center))
            # cams.append(camera)
        if iteration >  opt_args.multi_view_weight_from_iter:
            for camera in batched_nearest_cameras:
                if camera != None:
                    gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
                    voxel_visible_mask = prefilter_voxel(camera, gaussians, pipe_args, background)
                    batched_nearest_voxel_mask.append(voxel_visible_mask)
                else:
                    batched_nearest_voxel_mask.append(None)
        else:
            for camera in batched_nearest_cameras:
                batched_nearest_voxel_mask.append(None)
        retain_grad = (iteration < opt_args.update_until and iteration >= 0)

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipe_args,
            background,
            batched_voxel_mask = batched_voxel_mask,
            batched_nearest_cameras = batched_nearest_cameras,
            batched_nearest_voxel_mask = batched_nearest_voxel_mask,
            retain_grad=retain_grad,
            batched_strategies=batched_strategies,
            mode="train",
            return_plane = iteration > opt_args.single_view_weight_from_iter,
            iterations = iteration ,
        )
        batched_image, batched_compute_locally,  batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, batched_return_dict_nearest = render_final(batched_cameras,
            batched_screenspace_pkg, batched_strategies, gaussians, batched_cameras_nearest = batched_nearest_cameras
        )
        batch_statistic_collector = [
            cuda_args["stats_collector"]
            for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
        ]
        loss_sum, batched_losses = batched_loss_computation(
            batched_image,
            batched_return_dict,
            batched_cameras,
            # batched_compute_locally,
            batched_strategies,
            batch_statistic_collector,
            iterations=iteration,
            opt=opt_args,
            batched_return_dict_nearest = batched_return_dict_nearest,
            batched_nearest_cameras = batched_nearest_cameras
        )



        timers.start("backward")
        loss_sum.backward()
        timers.stop("backward")
        utils.check_initial_gpu_memory_usage("after backward")
        if utils.DEFAULT_GROUP.size() > 1:
        #reference to Momtumgs
            with torch.no_grad():
                # all reduce mlp grad
                for param in gaussians.mlp_opacity.parameters():
                    torch.distributed.all_reduce(param.grad)
                    torch.cuda.synchronize()
                    dist.barrier()
                    param.grad = param.grad / len(batched_cameras)

                for param in gaussians.mlp_color.parameters():
                    torch.distributed.all_reduce(param.grad)
                    torch.cuda.synchronize()
                    dist.barrier()
                    param.grad = param.grad / len(batched_cameras)

                for param in gaussians.mlp_cov.parameters():
                    torch.distributed.all_reduce(param.grad)
                    torch.cuda.synchronize()
                    dist.barrier()
                    param.grad = param.grad / len(batched_cameras)

        with torch.no_grad():
            # Adjust workload division strategy.
            globally_sync_for_timer()
            timers.start("finish_strategy_final")
            finish_strategy_final(
                batched_cameras,
                strategy_history,
                batched_strategies,
                batch_statistic_collector,
            )
            timers.stop("finish_strategy_final")

            # Sync losses in the batch
            timers.start("sync_loss_and_log")
            batched_losses = torch.tensor(batched_losses, device="cuda")
            if utils.DEFAULT_GROUP.size() > 1:
                dist.all_reduce(
                    batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                )
            batched_loss = (1.0 - args.lambda_dssim) * batched_losses[
                :, 0
            ] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
            batched_loss_cpu = batched_loss.cpu().numpy()
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            # Update Epoch Statistics
            train_dataset.update_losses(batched_loss_cpu)
            # Logging
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)
            timers.stop("sync_loss_and_log")

            # Evaluation
            end2end_timers.stop()
            training_report(
                iteration,
                l1_loss,
                args.test_iterations,
                scene,
                pipe_args,
                background,
                args.backend,
                dataset_args.model_path
            )
            end2end_timers.start()

            # Densification

            densification(iteration, scene, gaussians, batched_screenspace_pkg)

            # Save Gaussians
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

                if args.save_strategy_history:
                    with open(
                        args.log_folder
                        + "/strategy_history_ws="
                        + str(utils.WORLD_SIZE)
                        + "_rk="
                        + str(utils.GLOBAL_RANK)
                        + ".json",
                        "w",
                    ) as f:
                        json.dump(strategy_history.to_json(), f)
                end2end_timers.start()

            # Save Checkpoints
            if any(
                [
                    iteration <= checkpoint_iteration < iteration + args.bsz
                    for checkpoint_iteration in args.checkpoint_iterations
                ]
            ):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    if utils.DEFAULT_GROUP.size() > 1:
                        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                elif utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder
                    + "/chkpnt_ws="
                    + str(utils.WORLD_SIZE)
                    + "_rk="
                    + str(utils.GLOBAL_RANK)
                    + ".pth",
                )
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                timers.start("optimizer_step")

                if (
                    args.lr_scale_mode != "accumu"
                ):  # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz

                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timers.stop("optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")

        # Finish a iteration and clean up
        torch.cuda.synchronize()
        for (
            viewpoint_cam
        ) in batched_cameras:  # Release memory of locally rendered original_image
            viewpoint_cam.original_image = None
            if args.distributed_dataset_storage:
                viewpoint_cam.image_gray = None 
                viewpoint_cam.invdepthmap = None
        if iteration >  opt_args.multi_view_weight_from_iter and args.distributed_dataset_storage:
            for (
                nearest_camera
            ) in batched_nearest_cameras: 
                nearest_camera.image_gray = None
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    progress_bar.close()


def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend, model_path
):
    args = utils.get_args()
    log_file = utils.get_log_file()
    # Report test and samples of training set
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(
        iteration, utils.get_args().bsz, testing_iterations[0], 0
    ):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))

        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(), "num_cameras": len(scene.getTestCameras())},
            {
                "name": "train",
                "cameras": scene.getTrainCameras(),
                "num_cameras": max(20, args.bsz),
            },
        )
        

        # init workload division strategy
        for config in validation_configs:
            render_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "gt")
            depths_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "depths")
            render_normal_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "normals")
            rendered_distance_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "distance")
            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(depths_path, exist_ok=True)
            makedirs(render_normal_path, exist_ok=True)
            makedirs(rendered_distance_path, exist_ok=True)
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")
                ssim_test = torch.scalar_tensor(0.0, device="cuda")
                lpips_test = torch.scalar_tensor(0.0, device="cuda")
                # TODO: if not divisible by world size
                num_cameras = config["num_cameras"] 
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    if args.local_sampling:
                        # TODO: if not divisible by world size
                        batched_cameras_idx = eval_dataset.get_batched_cameras_idx(
                            args.bsz // utils.WORLD_SIZE
                        )
                        batched_all_cameras_idx = torch.zeros(
                            (utils.WORLD_SIZE, len(batched_cameras_idx)),
                            device="cuda",
                            dtype=int,
                        )
                        batched_cameras_idx = torch.tensor(
                            batched_cameras_idx, device="cuda", dtype=int
                        )
                        torch.distributed.all_gather_into_tensor(
                            batched_all_cameras_idx,
                            batched_cameras_idx,
                            group=utils.DEFAULT_GROUP,
                        )
                        batched_all_cameras_idx = (
                            batched_all_cameras_idx.cpu().numpy().squeeze()
                        )
                        batched_cameras, _ = eval_dataset.get_batched_cameras_from_idx(
                            batched_all_cameras_idx
                        )
                    else:
                        batched_cameras, _ = eval_dataset.get_batched_cameras(
                            num_camera_to_load , eval = True
                        )
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
                        scene.gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
                        voxel_visible_mask = prefilter_voxel(camera, scene.gaussians, pipe_args, background)
                        batched_voxel_mask.append(voxel_visible_mask)
                        batched_nearest_voxel_mask.append(None)
                        batched_nearest_cameras.append(None)
                    # retain_grad = (iteration < opt_args.update_until and iteration >= 0)

                    batched_screenspace_pkg = (
                        distributed_preprocess3dgs_and_all2all_final(
                            batched_cameras,
                            scene.gaussians,
                            pipe_args,
                            batched_voxel_mask = batched_voxel_mask,
                            bg_color = background,
                            batched_strategies=batched_strategies,
                            batched_nearest_cameras = batched_nearest_cameras,
                            batched_nearest_voxel_mask = batched_nearest_voxel_mask,
                            mode="test",
                        )
                    )
                    batched_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, _ = render_final(
                        batched_cameras, batched_screenspace_pkg, batched_strategies
                    )
                    for camera_id, (image, gt_camera, render_pkg) in enumerate(
                        zip(batched_image, batched_cameras, batched_return_dict)
                    ):
                        depth = render_pkg["plane_depth"]
                        normal = render_pkg["rendered_normal"]
                        rendered_distance = render_pkg["rendered_distance"]
                        if (
                            image is None or len(image.shape) == 0
                        ):  # The image is not rendered locally.
                            image = torch.zeros(
                                gt_camera.original_image.shape,
                                device="cuda",
                                dtype=torch.float32,
                            )
                        if (
                            depth is None or len(depth.shape) == 0
                        ):
                            depth = torch.zeros(
                                (1, gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]),
                                device="cuda",
                                dtype=torch.float32,
                            )
                            rendered_distance = torch.zeros(
                                (1, gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]),
                                device="cuda",
                                dtype=torch.float32,
                            )

                        if (
                            normal is None or len(normal.shape) == 0
                        ):
                            normal = torch.zeros(
                                (3, gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]),
                                device="cuda",
                                dtype=torch.float32,
                            )

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                depth, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                            torch.distributed.all_reduce(
                                rendered_distance, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        # if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )
                        # if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                normal, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(
                            gt_camera.original_image / 255.0, 0.0, 1.0
                        )

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                            # ssim_test += ssim(image, gt_image).mean().double()
                            # lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()


                        if utils.GLOBAL_RANK == 0:


                            depth = depth.detach().cpu().numpy().squeeze(0)
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                            

                            distance = rendered_distance.squeeze().detach().cpu().numpy()
                            distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                            distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                            distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)


                            # depth_RED = visualize_scalars(torch.log(depth + 1e-8).detach().cpu())

                            # plt.imsave(os.path.join(depths_path, 'depth-' +(gt_camera.image_name + '.png') ), depth_RED)

                            # torchvision.utils.save_image(
                            #     torch.tensor(depth_color).permute(2,0,1)/255.0,
                            #     os.path.join(depths_path, gt_camera.image_name + ".png"),
                            # )

                            normal = normal.permute(1,2,0)
                            normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                            normal = normal.detach().cpu().numpy()
                            normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                            # torchvision.utils.save_image(
                            #     torch.tensor(normal).permute(2,0,1)/255.0,
                            #     os.path.join(render_normal_path, gt_camera.image_name + ".png"),
                            # )
                            # if "MatrixCity" in args.source_path:
                            #     filename = gt_camera.image_name.split("/")[-1]  # 获取 "0068.png"
                            #     torchvision.utils.save_image(
                            #     image,
                            #     os.path.join(render_path, filename ),
                            #     )
                            #     torchvision.utils.save_image(
                            #         gt_image,
                            #         os.path.join(gts_path, filename ),
                            #     )
                            #     # cv2.imwrite(os.path.join(rendered_distance_path,  filename ), distance_color)
                            #     cv2.imwrite(os.path.join(depths_path,  filename ), depth_color)
                            #     # cv2.imwrite(os.path.join(render_normal_path,  filename), normal)
                            # else:
                            torchvision.utils.save_image(
                                image,
                                os.path.join(render_path, gt_camera.image_name + ".png"),
                            )
                            torchvision.utils.save_image(
                                gt_image,
                                os.path.join(gts_path, gt_camera.image_name + ".png"),
                            )
                            cv2.imwrite(os.path.join(rendered_distance_path,  gt_camera.image_name + ".png"), distance_color)
                            cv2.imwrite(os.path.join(depths_path,  gt_camera.image_name + ".png"), depth_color)
                            cv2.imwrite(os.path.join(render_normal_path,  gt_camera.image_name + ".png"), normal)

                        gt_camera.original_image = None
                psnr_test /= num_cameras
                lpips_test/= num_cameras
                ssim_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}, images {}, ".format(
                        iteration, config["name"], l1_test, psnr_test, num_cameras
                    ),
                    #  "\n[ITER {}] Evaluating {}: L1 {} SSIM {}".format(
                    #     iteration, config["name"], l1_test, ssim_test
                    # ),
                    #  "\n[ITER {}] Evaluating {}: L1 {} SSIM {}".format(
                    #     iteration, config["name"], l1_test, lpips_test
                    # )

                )
                log_file.write(
                    "[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(
                        iteration, config["name"], l1_test, psnr_test
                    ),
                    # "\n[ITER {}] Evaluating {}: L1 {} SSIM {}".format(
                    #     iteration, config["name"], l1_test, ssim_test
                    # ),
                    #  "\n[ITER {}] Evaluating {}: L1 {} SSIM {}".format(
                    #     iteration, config["name"], l1_test, lpips_test
                    # )

                )

        torch.cuda.empty_cache()
