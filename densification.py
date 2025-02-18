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
