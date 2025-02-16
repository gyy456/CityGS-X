import torch
import utils.general_utils as utils
import torch.distributed as dist




def intersect_lines(ray1_origin, ray1_dir, ray2_origin, ray2_dir):
    # Normalize direction vectors
    ray1_dir = ray1_dir / torch.norm(ray1_dir)
    ray2_dir = ray2_dir / torch.norm(ray2_dir)

    # Cross product of direction vectors
    cross_dir = torch.cross(ray1_dir, ray2_dir)
    cross_dir_norm = torch.norm(cross_dir)

    # Check if the rays are parallel
    if cross_dir_norm < 1e-6:
        return None  # Rays are parallel and do not intersect

    # Line between the origins
    origin_diff = ray2_origin - ray1_origin

    # Calculate the distance along the cross product direction
    t1 = torch.dot(torch.cross(origin_diff, ray2_dir), cross_dir) / (cross_dir_norm ** 2)
    t2 = torch.dot(torch.cross(origin_diff, ray1_dir), cross_dir) / (cross_dir_norm ** 2)

    # Closest points on each ray
    closest_point1 = ray1_origin + t1 * ray1_dir
    closest_point2 = ray2_origin + t2 * ray2_dir

    # Midpoint between the two closest points as the intersection point
    intersection_point = (closest_point1 + closest_point2) / 2.0

    return intersection_point









def densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        for radii, visibility_filter, screenspace_mean2D, opacity, offset_selection_mask, voxel_visible_mask in zip(
            batched_screenspace_pkg["batched_locally_preprocessed_radii"],
            batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"],
            batched_screenspace_pkg["batched_locally_preprocessed_mean2D"],
            batched_screenspace_pkg["batched_locally_opacity"],
            batched_screenspace_pkg["batched_locally_offset_mask"],
            batched_screenspace_pkg["batched_locally_voxel_mask"],
        ):
            # gaussians.max_radii2D[visibility_filter] = torch.max(
            #     gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            # )
            gaussians.training_statis(screenspace_mean2D, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
            
            # gaussians.add_densification_stats(screenspace_mean2D, visibility_filter)
        timers.stop("densification_update_stats")

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")

            # imgs = []
            # for camera_id, (image, gt_camera) in enumerate(
            #         zip(batched_image, batched_cameras)
            #     ):
            #     if (
            #         image is None or len(image.shape) == 0
            #     ):  # The image is not rendered locally.
            #         image = torch.zeros(
            #             gt_camera.original_image_backup.shape,
            #             device="cuda",
            #             dtype=torch.float32,
            #         )

            #     if utils.DEFAULT_GROUP.size() > 1:
            #         torch.distributed.all_reduce(
            #             image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
            #         )

            #     image = torch.clamp(image, 0.0, 1.0)
            #     gt_image = torch.clamp(
            #         gt_camera.original_image_backup / 255.0, 0.0, 1.0
            #     )
            #     imgs.append([image, gt_image])
            # boxes = []
            # losses = []
            # for img_pair in imgs:
            #     w_box = 32
            #     h_box = 16
            #     l1map = torch.abs(img_pair[1] - img_pair[0])
            #     c, h, w = l1map.shape
            #     loss_max = 0
            #     max_id = (0, 0)
            #     for i in range(0, h - h_box, h_box):
            #         for j in range(0, w - w_box, w_box):
            #             loss_region = torch.mean(
            #                 l1map[:, i:i + h_box, j:j + w_box])
            #             if loss_region > loss_max:
            #                 loss_max = loss_region
            #                 max_id = [i, j]
            #     losses.append(loss_max)
            #     i, j = max_id
            #     if loss_max > 0.3:
            #         box = [i, j, i+h_box, j+w_box]
            #     else :
            #         box = [i, j, i, j]
            #     boxes.append(box)
            #每张图loss最大的patch
            #这里是不是应该加一个loss的阈值 loss较小的时候就没必要翻倍点数了
            # sorted_indices = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)
            # boxes = [boxes[i] for i in sorted_indices]
            # cams = [batched_cameras[i] for i in sorted_indices]

            # box3ds = []
            # if cams is not None:
            #     for i, cam_0 in enumerate(cams):
            #         for j, cam_1 in enumerate(cams[i+1:]):
            #             ray0_o = cam_0.rayo
            #             ray0_d = cam_0.rayd
            #             box0 = boxes[i]

            #             ray0_o_topleft = ray0_o[0,:, box0[0], box0[1]]
            #             ray0_d_topleft = ray0_d[0,:, box0[0], box0[1]]

            #             ray0_o_bottomright = ray0_o[0,:, box0[2], box0[3]]
            #             ray0_d_bottomright = ray0_d[0,:, box0[2], box0[3]]

            #             ray0_o_bottomleft = ray0_o[0,:, box0[2], box0[1]]
            #             ray0_d_bottomleft = ray0_d[0,:, box0[2], box0[1]]

            #             ray0_o_topright = ray0_o[0, :, box0[0], box0[3]]
            #             ray0_d_topright = ray0_d[0, :, box0[0], box0[3]]

            #             ray1_o = cam_1.rayo
            #             ray1_d = cam_1.rayd
            #             box1 = boxes[j+i+1]

            #             ray1_o_topleft = ray1_o[0, :, box1[0], box1[1]]
            #             ray1_d_topleft = ray1_d[0, :, box1[0], box1[1]]

            #             ray1_o_bottomright = ray1_o[0, :, box1[2], box1[3]]
            #             ray1_d_bottomright = ray1_d[0, :, box1[2], box1[3]]

            #             ray1_o_bottomleft = ray1_o[0, :, box1[2], box1[1]]
            #             ray1_d_bottomleft = ray1_d[0, :, box1[2], box1[1]]

            #             ray1_o_topright = ray1_o[0, :, box1[0], box1[3]]
            #             ray1_d_topright = ray1_d[0, :, box1[0], box1[3]]


            #             topleft_intersect = intersect_lines(ray0_o_topleft, ray0_d_topleft, ray1_o_topleft, ray1_d_topleft)
            #             bottomright_intersect = intersect_lines(ray0_o_bottomright, ray0_d_bottomright, ray1_o_bottomright, ray1_d_bottomright)
            #             bottomleft_interset = intersect_lines(ray0_o_bottomleft, ray0_d_bottomleft, ray1_o_bottomleft, ray1_d_bottomleft)
            #             topright_intersect = intersect_lines(ray0_o_topright, ray0_d_topright, ray1_o_topright, ray1_d_topright)


            #             region3d = [topleft_intersect,bottomright_intersect, bottomleft_interset, topright_intersect]
            #             if len(region3d)==0 or None in region3d:
            #                 continue
            #             region3d = torch.vstack(region3d)
            #             x_min_3d = torch.min(region3d[:, 0])
            #             y_min_3d = torch.min(region3d[:, 1])
            #             z_min_3d = torch.min(region3d[:, 2])

            #             x_max_3d = torch.max(region3d[:, 0])
            #             y_max_3d = torch.max(region3d[:, 1])
            #             z_max_3d = torch.max(region3d[:, 2])



            #             box3d = [x_min_3d, y_min_3d, z_min_3d, x_max_3d, y_max_3d, z_max_3d]
            #             box3ds.append(box3d)
            #需要densification的offsets的空间点


            gaussians.adjust_anchor(
                        # box3ds,
                        iteration=iteration,
                        check_interval=100, 
                        success_threshold=0.8,
                        grad_threshold=0.0002,
                        update_ratio=0.2,
                        extra_ratio=0.25,
                        extra_up=0.01,
                        min_opacity= 0.005,
                    )
            timers.stop("densify_and_prune")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                # num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                num_3dgs_before_redistribute = gaussians.get_anchor.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()   #anchor
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write(
                    "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                        iteration,
                        iteration + args.bsz,
                        num_3dgs_before_redistribute,
                        num_3dgs_after_redistribute,
                    )
                )

            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=True
            )

            utils.inc_densify_iter()

        # if (
        #     utils.check_update_at_this_iter(
        #         iteration, args.bsz, args.opacity_reset_interval, 0
        #     )
        #     and iteration + args.bsz <= args.opacity_reset_until_iter
        # ):
        #     timers.start("reset_opacity")
        #     gaussians.reset_opacity()
        #     timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )


def gsplat_densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        image_width = batched_screenspace_pkg["image_width"]
        image_height = batched_screenspace_pkg["image_height"]
        batched_screenspace_mean2D_grad = batched_screenspace_pkg[
            "batched_locally_preprocessed_mean2D"
        ].grad
        for i, (radii, visibility_filter) in enumerate(
            zip(
                batched_screenspace_pkg["batched_locally_preprocessed_radii"],
                batched_screenspace_pkg[
                    "batched_locally_preprocessed_visibility_filter"
                ],
            )
        ):
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.gsplat_add_densification_stats(
                batched_screenspace_mean2D_grad[i],
                visibility_filter,
                image_width,
                image_height,
            )
        timers.stop("densification_update_stats")

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            gaussians.densify_and_prune(
                args.densify_grad_threshold,
                args.min_opacity,
                scene.cameras_extent,
                size_threshold,
            )
            timers.stop("densify_and_prune")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write(
                    "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                        iteration,
                        iteration + args.bsz,
                        num_3dgs_before_redistribute,
                        num_3dgs_after_redistribute,
                    )
                )

            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=True
            )

            utils.inc_densify_iter()

        if (
            utils.check_update_at_this_iter(
                iteration, args.bsz, args.opacity_reset_interval, 0
            )
            and iteration + args.bsz <= args.opacity_reset_until_iter
        ):
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )
