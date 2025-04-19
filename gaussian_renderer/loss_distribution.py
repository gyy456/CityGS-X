import torch
import utils.general_utils as utils
import torch.distributed as dist
from utils.loss_utils import pixelwise_l1_with_mask, pixelwise_ssim_with_mask
import time
import diff_gaussian_rasterization
import math
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.general_utils import get_expon_lr_func
from utils.image_utils import psnr, dilate, erode
import torch.nn.functional as F
from multi_view_precess import get_points_from_depth, get_points_depth_in_depth_map
import numpy as np

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)







def get_touched_tile_rect(touched_locally):
    nonzero_pos = touched_locally.nonzero()
    min_tile_y = nonzero_pos[:, 0].min().item()
    max_tile_y = nonzero_pos[:, 0].max().item() + 1
    min_tile_x = nonzero_pos[:, 1].min().item()
    max_tile_x = nonzero_pos[:, 1].max().item() + 1
    return min_tile_y, max_tile_y, min_tile_x, max_tile_x


def get_touched_pixels_rect(touched_locally=None, tile_rect=None):
    assert (
        touched_locally is not None or tile_rect is not None
    ), "Either touched_locally or tile_rect should be provided"
    if tile_rect is None:
        min_tile_y, max_tile_y, min_tile_x, max_tile_x = get_touched_tile_rect(
            touched_locally
        )
    else:
        min_tile_y, max_tile_y, min_tile_x, max_tile_x = tile_rect

    min_pixel_y = min_tile_y * utils.BLOCK_Y
    max_pixel_y = min(max_tile_y * utils.BLOCK_Y, utils.IMG_H)
    min_pixel_x = min_tile_x * utils.BLOCK_X
    max_pixel_x = min(max_tile_x * utils.BLOCK_X, utils.IMG_W)
    return min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x


def get_all_pos_send_to_j(compute_locally, touched_locally):
    timers = utils.get_timers()

    timers.start("[all_pos_send_to_j]all_gather_locally_compute")
    # allgather locally_compute. For 4K, the size of one locally_compute is 4000*4000 = 16MB. It is not a short time.
    # because bool tensor use 1 byte for 1 element: https://discuss.pytorch.org/t/why-does-each-torch-bool-value-take-up-an-entire-byte/183822
    # TODO: This could be optimized by using bitset. Divide communication volume by 8.
    all_locally_compute = torch.empty(
        (utils.MP_GROUP.size(),) + compute_locally.shape,
        dtype=torch.bool,
        device="cuda",
    )
    torch.distributed.all_gather_into_tensor(
        all_locally_compute, compute_locally, group=utils.MP_GROUP
    )
    # Suppose we are sending from i to j.
    pos_mask_to_recv_from_i = [None for _ in range(utils.MP_GROUP.size())]
    pos_recv_from_i = [None for _ in range(utils.MP_GROUP.size())]
    recv_from_i_size = [None for _ in range(utils.MP_GROUP.size())]
    for i in range(utils.MP_GROUP.size()):
        if i != utils.MP_GROUP.rank():
            pos_mask_to_recv_from_i[i] = torch.logical_and(
                all_locally_compute[i], touched_locally
            )
            pos_recv_from_i[i] = torch.nonzero(
                pos_mask_to_recv_from_i[i], as_tuple=False
            ).contiguous()  # (num_pos, 2); contiguous() is needed here.
            recv_from_i_size[i] = pos_recv_from_i[i].shape[0]
        else:
            pos_recv_from_i[i] = torch.zeros((0, 2), dtype=torch.long, device="cuda")
            recv_from_i_size[i] = pos_recv_from_i[i].shape[0]
    timers.stop("[all_pos_send_to_j]all_gather_locally_compute")

    timers.start(
        "[all_pos_send_to_j]all_gather_send_to_j_size"
    )  # NOTE: This is slow because all_gather_object involves cpu2gpu+gpu2gpu+gpu2cpu communication here.
    all_pos_recv_from_i = torch.cat(pos_recv_from_i, dim=0)
    j_recv_from_i_size = [
        None for _ in range(utils.MP_GROUP.size())
    ]  # jth row and ith column(j_recv_from_i_size[j][i]) is the size of sending from i to j.
    # each element should be a list of size of pos_recv_from_i[i] for all i. i.e. [None for _ in range(utils.MP_GROUP.size())]
    torch.distributed.all_gather_object(
        j_recv_from_i_size, recv_from_i_size, group=utils.MP_GROUP
    )
    send_to_j_size = [
        j_recv_from_i_size[j][utils.MP_GROUP.rank()]
        for j in range(utils.MP_GROUP.size())
    ]
    timers.stop("[all_pos_send_to_j]all_gather_send_to_j_size")

    timers.start("[all_pos_send_to_j]all_to_all_pos_send_to_j")
    # Use send_to_j_size[i] as the shape of the tensor
    pos_send_to_j = [None for _ in range(utils.MP_GROUP.size())]
    for j in range(utils.MP_GROUP.size()):
        pos_send_to_j[j] = torch.empty(
            (send_to_j_size[j], 2), dtype=torch.long, device="cuda"
        )
    torch.distributed.all_to_all(pos_send_to_j, pos_recv_from_i, group=utils.MP_GROUP)
    all_pos_send_to_j = torch.cat(pos_send_to_j, dim=0).contiguous()
    timers.stop("[all_pos_send_to_j]all_to_all_pos_send_to_j")

    return send_to_j_size, recv_from_i_size, all_pos_send_to_j, all_pos_recv_from_i


def get_remote_tiles(send_to_j_size, recv_from_i_size, all_tiles_send_to_j):
    # split all_tiles_send_to_j into tiles_send_to_j, according to the size of pos_send_to_j[j]
    tiles_send_to_j = [None for _ in range(utils.MP_GROUP.size())]
    tiles_recv_from_i = [None for _ in range(utils.MP_GROUP.size())]
    start = 0
    for j in range(utils.MP_GROUP.size()):
        end = start + send_to_j_size[j]
        tiles_send_to_j[j] = all_tiles_send_to_j[start:end].contiguous()
        start = end

        i = j
        tiles_recv_from_i[i] = torch.empty(
            (recv_from_i_size[i], 3, utils.BLOCK_Y, utils.BLOCK_X),
            dtype=torch.float32,
            device="cuda",
        )
        # XXX: Double check the empty behavior. Because the boundary condition is not clear.

    # all_to_all the pixels
    torch.distributed.nn.functional.all_to_all(
        tiles_recv_from_i, tiles_send_to_j, group=utils.MP_GROUP
    )  # The gradient successfully goes back.

    all_tiles_recv_from_i = torch.cat(tiles_recv_from_i, dim=0).contiguous()
    return all_tiles_recv_from_i


def general_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    timers = utils.get_timers()

    timers.start("[loss]prepare_for_distributed_loss_computation")

    # Get locally touched tiles and image rect.
    timers.start("[loss]get_touched_locally_and_local_image_rect")
    touched_locally = diff_gaussian_rasterization._C.get_touched_locally(  # touched_locally[i,j] is true if pixel (i,j) is touched during local loss computation.
        compute_locally,
        utils.IMG_H,
        utils.IMG_W,
        1,  # HACK: extension distance is currently only 1 here because window size is 11 which is less than BLOCK_X and BLOCK_Y. And we only have 1 layer of convolution.
    )
    min_tile_y, max_tile_y, min_tile_x, max_tile_x = get_touched_tile_rect(
        touched_locally
    )
    min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = get_touched_pixels_rect(
        tile_rect=(min_tile_y, max_tile_y, min_tile_x, max_tile_x)
    )
    touched_pixels_rect = [min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x]
    touched_tiles_rect = [min_tile_y, max_tile_y, min_tile_x, max_tile_x]
    # Get image_rect touched locally. shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    local_image_rect = image[
        :, min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x
    ].contiguous()
    timers.stop("[loss]get_touched_locally_and_local_image_rect")

    # Get positions of tiles to send/recv remotely.
    timers.start("[loss]get_all_pos_send_to_j")
    (
        send_to_j_size,
        recv_from_i_size,
        all_pos_send_to_j,
        all_pos_recv_from_i,
    ) = get_all_pos_send_to_j(compute_locally, touched_locally)
    timers.stop("[loss]get_all_pos_send_to_j")

    # Load local tiles to send remotely.
    timers.start("[loss]load_image_tiles_by_pos")
    all_tiles_send_to_j = diff_gaussian_rasterization.load_image_tiles_by_pos(
        local_image_rect,  # local image rect
        all_pos_send_to_j,  # in global coordinates.
        utils.IMG_H,
        utils.IMG_W,
        touched_pixels_rect,
        touched_tiles_rect,
    )
    timers.stop("[loss]load_image_tiles_by_pos")

    # Receive remote tiles to use locally.
    timers.start("[loss]get_remote_tiles")
    all_tiles_recv_from_i = get_remote_tiles(
        send_to_j_size, recv_from_i_size, all_tiles_send_to_j
    )
    timers.stop("[loss]get_remote_tiles")

    # Assemble the image from all the remote tiles, excluding the local ones. shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    timers.start("[loss]merge_local_tiles_and_remote_tiles")
    local_image_rect_from_remote_tiles = (
        diff_gaussian_rasterization.merge_image_tiles_by_pos(
            all_pos_recv_from_i,  # in global coordinates.
            all_tiles_recv_from_i,
            utils.IMG_H,
            utils.IMG_W,
            touched_pixels_rect,
            touched_tiles_rect,
        )
    )  # in local coordinates.
    local_image_rect_with_remote_tiles = (
        local_image_rect + local_image_rect_from_remote_tiles
    )  # in local coordinates.
    timers.stop("[loss]merge_local_tiles_and_remote_tiles")

    # Get pixels to compute locally. shape: (max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    # timers.start("[loss]get_pixels_compute_locally_and_in_rect")# very small time
    local_image_rect_pixels_compute_locally = (
        diff_gaussian_rasterization._C.get_pixels_compute_locally_and_in_rect(
            compute_locally,
            utils.IMG_H,
            utils.IMG_W,
            min_pixel_y,
            max_pixel_y,
            min_pixel_x,
            max_pixel_x,
        )
    )
    # timers.stop("[loss]get_pixels_compute_locally_and_in_rect")

    timers.stop("[loss]prepare_for_distributed_loss_computation")

    utils.check_initial_gpu_memory_usage(
        "after preparation for image loss distribution"
    )

    # Move image_gt to GPU. its shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000

    utils.check_initial_gpu_memory_usage("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
        utils.get_num_pixels() * 3
    )
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.

    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss


class _AddRemotePixelsToImage(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    ):
        (
            first_tile_y,
            first_tile_x,
            last_tile_y,
            last_tile_x,
            first_pixel_y,
            first_pixel_x,
            last_pixel_y_plus1,
            last_pixel_x_plus1,
            half_window_size,
        ) = configs

        ctx.configs = configs

        coverage_min_y = max(first_pixel_y - half_window_size, 0)
        coverage_max_y = min(last_pixel_y_plus1 + half_window_size, utils.IMG_H)
        image_with_remote_pixels = image[
            :, coverage_min_y:coverage_max_y, :
        ].contiguous()

        if utils.MP_GROUP.rank() != 0:
            if first_pixel_x == 0:
                image_with_remote_pixels[:, 0:half_window_size, :] = (
                    recv_from_rk_minus_1_part1
                )
            else:
                image_with_remote_pixels[
                    :,
                    (first_pixel_y + utils.BLOCK_Y - half_window_size)
                    - coverage_min_y : (first_pixel_y + utils.BLOCK_Y)
                    - coverage_min_y,
                    0:first_pixel_x,
                ] = recv_from_rk_minus_1_part1
                image_with_remote_pixels[
                    :,
                    (first_pixel_y - half_window_size)
                    - coverage_min_y : (
                        first_pixel_y + utils.BLOCK_Y - half_window_size
                    )
                    - coverage_min_y,
                    first_pixel_x - half_window_size : first_pixel_x,
                ] = recv_from_rk_minus_1_part2
                image_with_remote_pixels[
                    :,
                    (first_pixel_y - half_window_size)
                    - coverage_min_y : (first_pixel_y)
                    - coverage_min_y,
                    first_pixel_x : utils.IMG_W,
                ] = recv_from_rk_minus_1_part3

        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            if last_pixel_x_plus1 == utils.IMG_W:
                # recv from rank+1
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    :,
                ] = recv_from_rk_plus_1_part1
            else:
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    0:last_pixel_x_plus1,
                ] = recv_from_rk_plus_1_part1
                # recv_from_rk_plus_1_part2_shape = (3, utils.BLOCK_Y, min(half_window_size, utils.IMG_W - last_pixel_x_plus1))
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    last_pixel_x_plus1 : last_pixel_x_plus1
                    + min(half_window_size, utils.IMG_W - last_pixel_x_plus1),
                ] = recv_from_rk_plus_1_part2
                # recv_from_rk_plus_1_part3_shape = (3, half_window_size, utils.IMG_W-last_pixel_x_plus1)
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y)
                    - coverage_min_y : (
                        last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size
                    )
                    - coverage_min_y,
                    last_pixel_x_plus1 : utils.IMG_W,
                ] = recv_from_rk_plus_1_part3

        ctx.save_for_backward(image)

        return image_with_remote_pixels

    @staticmethod
    def backward(ctx, grad_image_with_remote_pixels):
        # grad_radii, grad_depths should be all None.

        (
            first_tile_y,
            first_tile_x,
            last_tile_y,
            last_tile_x,
            first_pixel_y,
            first_pixel_x,
            last_pixel_y_plus1,
            last_pixel_x_plus1,
            half_window_size,
        ) = ctx.configs

        (image,) = ctx.saved_tensors

        coverage_min_y = max(first_pixel_y - half_window_size, 0)
        coverage_max_y = min(last_pixel_y_plus1 + half_window_size, utils.IMG_H)

        if utils.MP_GROUP.rank() != 0:
            if first_pixel_x == 0:
                grad_recv_from_rk_minus_1_part1 = (
                    grad_image_with_remote_pixels[:, 0:half_window_size, :]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[:, 0:half_window_size, :] = 0

                grad_recv_from_rk_minus_1_part2 = None
                grad_recv_from_rk_minus_1_part3 = None
            else:
                grad_recv_from_rk_minus_1_part1 = (
                    grad_image_with_remote_pixels[
                        :,
                        (first_pixel_y + utils.BLOCK_Y - half_window_size)
                        - coverage_min_y : (first_pixel_y + utils.BLOCK_Y)
                        - coverage_min_y,
                        0:first_pixel_x,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (first_pixel_y + utils.BLOCK_Y - half_window_size)
                    - coverage_min_y : (first_pixel_y + utils.BLOCK_Y)
                    - coverage_min_y,
                    0:first_pixel_x,
                ] = 0

                grad_recv_from_rk_minus_1_part2 = (
                    grad_image_with_remote_pixels[
                        :,
                        (first_pixel_y - half_window_size)
                        - coverage_min_y : (
                            first_pixel_y + utils.BLOCK_Y - half_window_size
                        )
                        - coverage_min_y,
                        first_pixel_x - half_window_size : first_pixel_x,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (first_pixel_y - half_window_size)
                    - coverage_min_y : (
                        first_pixel_y + utils.BLOCK_Y - half_window_size
                    )
                    - coverage_min_y,
                    first_pixel_x - half_window_size : first_pixel_x,
                ] = 0

                grad_recv_from_rk_minus_1_part3 = (
                    grad_image_with_remote_pixels[
                        :,
                        (first_pixel_y - half_window_size)
                        - coverage_min_y : (first_pixel_y)
                        - coverage_min_y,
                        first_pixel_x : utils.IMG_W,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (first_pixel_y - half_window_size)
                    - coverage_min_y : (first_pixel_y)
                    - coverage_min_y,
                    first_pixel_x : utils.IMG_W,
                ] = 0
        else:
            grad_recv_from_rk_minus_1_part1 = None
            grad_recv_from_rk_minus_1_part2 = None
            grad_recv_from_rk_minus_1_part3 = None

        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            if last_pixel_x_plus1 == utils.IMG_W:
                # recv from rank+1
                grad_recv_from_rk_plus_1_part1 = (
                    grad_image_with_remote_pixels[
                        :,
                        (last_pixel_y_plus1)
                        - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                        - coverage_min_y,
                        :,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    :,
                ] = 0
                grad_recv_from_rk_plus_1_part2 = None
                grad_recv_from_rk_plus_1_part3 = None
            else:
                grad_recv_from_rk_plus_1_part1 = (
                    grad_image_with_remote_pixels[
                        :,
                        (last_pixel_y_plus1)
                        - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                        - coverage_min_y,
                        0:last_pixel_x_plus1,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    0:last_pixel_x_plus1,
                ] = 0

                grad_recv_from_rk_plus_1_part2 = (
                    grad_image_with_remote_pixels[
                        :,
                        (last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size)
                        - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                        - coverage_min_y,
                        last_pixel_x_plus1 : last_pixel_x_plus1
                        + min(half_window_size, utils.IMG_W - last_pixel_x_plus1),
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size)
                    - coverage_min_y : (last_pixel_y_plus1 + half_window_size)
                    - coverage_min_y,
                    last_pixel_x_plus1 : last_pixel_x_plus1
                    + min(half_window_size, utils.IMG_W - last_pixel_x_plus1),
                ] = 0

                grad_recv_from_rk_plus_1_part3 = (
                    grad_image_with_remote_pixels[
                        :,
                        (last_pixel_y_plus1 - utils.BLOCK_Y)
                        - coverage_min_y : (
                            last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size
                        )
                        - coverage_min_y,
                        last_pixel_x_plus1 : utils.IMG_W,
                    ]
                    .clone()
                    .contiguous()
                )
                grad_image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y)
                    - coverage_min_y : (
                        last_pixel_y_plus1 - utils.BLOCK_Y + half_window_size
                    )
                    - coverage_min_y,
                    last_pixel_x_plus1 : utils.IMG_W,
                ] = 0
        else:
            grad_recv_from_rk_plus_1_part1 = None
            grad_recv_from_rk_plus_1_part2 = None
            grad_recv_from_rk_plus_1_part3 = None

        grad_image = torch.zeros_like(image)
        grad_image[:, coverage_min_y:coverage_max_y, :] = grad_image_with_remote_pixels

        return (
            grad_image,
            grad_recv_from_rk_minus_1_part1,
            grad_recv_from_rk_minus_1_part2,
            grad_recv_from_rk_minus_1_part3,
            grad_recv_from_rk_plus_1_part1,
            grad_recv_from_rk_plus_1_part2,
            grad_recv_from_rk_plus_1_part3,
            None,
        )


def add_remote_pixels_to_image(
    image,
    recv_from_rk_minus_1_part1,
    recv_from_rk_minus_1_part2,
    recv_from_rk_minus_1_part3,
    recv_from_rk_plus_1_part1,
    recv_from_rk_plus_1_part2,
    recv_from_rk_plus_1_part3,
    configs,
):
    return _AddRemotePixelsToImage.apply(
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    )



class _AddRemotePixelsToImageLessComm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    ):
        (
            first_tile_y,
            first_tile_x,
            last_tile_y,
            last_tile_x,
            first_pixel_y,
            first_pixel_x,
            last_pixel_y_plus1,
            last_pixel_x_plus1,
            window_size,
        ) = configs

        ctx.configs = configs

        coverage_min_y = max(first_pixel_y - window_size, 0)
        coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)
        image_with_remote_pixels = image[
            :, coverage_min_y:coverage_max_y, :
        ].contiguous()

        if utils.MP_GROUP.rank() != 0:
            if first_pixel_x == 0:
                image_with_remote_pixels[:, 0:window_size, :] = (
                    recv_from_rk_minus_1_part1
                )
            else:
                image_with_remote_pixels[
                    :,
                    (first_pixel_y + utils.BLOCK_Y - window_size)
                    - coverage_min_y : (first_pixel_y + utils.BLOCK_Y)
                    - coverage_min_y,
                    0:first_pixel_x,
                ] = recv_from_rk_minus_1_part1
                image_with_remote_pixels[
                    :,
                    (first_pixel_y - window_size)
                    - coverage_min_y : (first_pixel_y + utils.BLOCK_Y - window_size)
                    - coverage_min_y,
                    first_pixel_x - window_size : first_pixel_x,
                ] = recv_from_rk_minus_1_part2
                image_with_remote_pixels[
                    :,
                    (first_pixel_y - window_size)
                    - coverage_min_y : (first_pixel_y)
                    - coverage_min_y,
                    first_pixel_x : utils.IMG_W,
                ] = recv_from_rk_minus_1_part3

        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            if last_pixel_x_plus1 == utils.IMG_W:
                # recv from rank+1
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + window_size)
                    - coverage_min_y,
                    :,
                ] = recv_from_rk_plus_1_part1
            else:
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1)
                    - coverage_min_y : (last_pixel_y_plus1 + window_size)
                    - coverage_min_y,
                    0:last_pixel_x_plus1,
                ] = recv_from_rk_plus_1_part1
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y + window_size)
                    - coverage_min_y : (last_pixel_y_plus1 + window_size)
                    - coverage_min_y,
                    last_pixel_x_plus1 : last_pixel_x_plus1
                    + min(window_size, utils.IMG_W - last_pixel_x_plus1),
                ] = recv_from_rk_plus_1_part2
                image_with_remote_pixels[
                    :,
                    (last_pixel_y_plus1 - utils.BLOCK_Y)
                    - coverage_min_y : (
                        last_pixel_y_plus1 - utils.BLOCK_Y + window_size
                    )
                    - coverage_min_y,
                    last_pixel_x_plus1 : utils.IMG_W,
                ] = recv_from_rk_plus_1_part3

        ctx.save_for_backward(image)

        return image_with_remote_pixels

    @staticmethod
    def backward(ctx, grad_image_with_remote_pixels):
        # grad_radii, grad_depths should be all None.

        (
            first_tile_y,
            first_tile_x,
            last_tile_y,
            last_tile_x,
            first_pixel_y,
            first_pixel_x,
            last_pixel_y_plus1,
            last_pixel_x_plus1,
            window_size,
        ) = ctx.configs

        (image,) = ctx.saved_tensors

        coverage_min_y = max(first_pixel_y - window_size, 0)
        coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)
        grad_image = torch.empty_like(image)
        grad_image[:, coverage_min_y:coverage_max_y, :] = grad_image_with_remote_pixels
        # NOTE: even if I do not clear other pixels' gradients,
        # it should not affect the correctness because backward render code only consider pixels that are computed locally.

        grad_recv_from_rk_minus_1_part1 = None
        grad_recv_from_rk_minus_1_part2 = None
        grad_recv_from_rk_minus_1_part3 = None
        grad_recv_from_rk_plus_1_part1 = None
        grad_recv_from_rk_plus_1_part2 = None
        grad_recv_from_rk_plus_1_part3 = None

        return (
            grad_image,
            grad_recv_from_rk_minus_1_part1,
            grad_recv_from_rk_minus_1_part2,
            grad_recv_from_rk_minus_1_part3,
            grad_recv_from_rk_plus_1_part1,
            grad_recv_from_rk_plus_1_part2,
            grad_recv_from_rk_plus_1_part3,
            None,
        )


def add_remote_pixels_to_image_less_comm(
    image,
    recv_from_rk_minus_1_part1,
    recv_from_rk_minus_1_part2,
    recv_from_rk_minus_1_part3,
    recv_from_rk_plus_1_part1,
    recv_from_rk_plus_1_part2,
    recv_from_rk_plus_1_part3,
    configs,
):
    return _AddRemotePixelsToImageLessComm.apply(
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    )


def fast_less_comm_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    # Compare to fast_distributed_loss_computation, this method get more remote pixels during forward and do replicated loss computation for pixels near border.
    # But it avoids another communication and associated memory movement during backward.
    # This method works when image resolution is small because we want to reduce the number of kernel launches.

    timers = utils.get_timers()

    assert (
        strategy is not None
    ), "strategy should not be None in fast_distributed_loss_computation"
    assert (
        utils.BLOCK_Y > 5
    ), "utils.BLOCK_Y should be greater than 5 to make sure fast_distributed_loss_computation works as expected."

    # Compare to fast_distributed_loss_computation.
    # We avoid the communication during backward.

    window_size = 11  # we only need to get pixels within the window size(11).
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[strategy.rank],
        strategy.division_pos[strategy.rank + 1],
    )

    # there are 3 parts to send and receive
    #                3
    #          ---------
    #         2|
    #  1       |
    # ----------

    first_tile_y, first_tile_x = (
        tile_ids_l // strategy.tile_x,
        tile_ids_l % strategy.tile_x,
    )
    first_pixel_y, first_pixel_x = (
        first_tile_y * utils.BLOCK_Y,
        first_tile_x * utils.BLOCK_X,
    )

    timers.start("[loss_distribution]prepare_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            # recv from rank-1
            recv_from_rk_minus_1_buffer = torch.empty(
                (3, window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            # send to rank-1
            send_to_rk_minus_1 = image[
                :, first_pixel_y : first_pixel_y + window_size, :
            ].contiguous()
        else:
            # recv from rank-1
            recv_from_rk_minus_1_part1_shape = (3, window_size, first_pixel_x)
            recv_from_rk_minus_1_part2_shape = (3, utils.BLOCK_Y, window_size)
            recv_from_rk_minus_1_part3_shape = (
                3,
                window_size,
                utils.IMG_W - first_pixel_x,
            )
            recv_from_rk_minus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_minus_1_part1_shape[1]
                        * recv_from_rk_minus_1_part1_shape[2]
                        + recv_from_rk_minus_1_part2_shape[1]
                        * recv_from_rk_minus_1_part2_shape[2]
                        + recv_from_rk_minus_1_part3_shape[1]
                        * recv_from_rk_minus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            # send to rank-1
            send_to_rk_minus_1_part1 = image[
                :,
                first_pixel_y
                + utils.BLOCK_Y : first_pixel_y
                + utils.BLOCK_Y
                + window_size,
                0:first_pixel_x,
            ]
            send_to_rk_minus_1_part2 = image[
                :,
                first_pixel_y
                + window_size : first_pixel_y
                + utils.BLOCK_Y
                + window_size,
                first_pixel_x : min(first_pixel_x + window_size, utils.IMG_W),
            ]
            send_to_rk_minus_1_part3 = image[
                :,
                first_pixel_y : first_pixel_y + window_size,
                first_pixel_x : utils.IMG_W,
            ]

            # flatten and concatenate them together
            send_to_rk_minus_1 = torch.cat(
                [
                    send_to_rk_minus_1_part1.flatten(),
                    send_to_rk_minus_1_part2.flatten(),
                    send_to_rk_minus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()

    last_tile_y, last_tile_x = (
        tile_ids_r // strategy.tile_x,
        tile_ids_r % strategy.tile_x,
    )
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y, utils.IMG_H),
            utils.IMG_W,
        )
        # NOTE: this is tricky here.
    else:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H),
            last_tile_x * utils.BLOCK_X,
        )

    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_buffer = torch.empty(
                (3, window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            send_to_rk_plus_1 = image[
                :, last_pixel_y_plus1 - window_size : last_pixel_y_plus1, :
            ].contiguous()
        else:
            recv_from_rk_plus_1_part1_shape = (3, window_size, last_pixel_x_plus1)
            recv_from_rk_plus_1_part2_shape = (
                3,
                utils.BLOCK_Y,
                min(window_size, utils.IMG_W - last_pixel_x_plus1),
            )
            recv_from_rk_plus_1_part3_shape = (
                3,
                window_size,
                utils.IMG_W - last_pixel_x_plus1,
            )

            recv_from_rk_plus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_plus_1_part1_shape[1]
                        * recv_from_rk_plus_1_part1_shape[2]
                        + recv_from_rk_plus_1_part2_shape[1]
                        * recv_from_rk_plus_1_part2_shape[2]
                        + recv_from_rk_plus_1_part3_shape[1]
                        * recv_from_rk_plus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            send_to_rk_plus_1_part1 = image[
                :,
                last_pixel_y_plus1 - window_size : last_pixel_y_plus1,
                0:last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part2 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - window_size : last_pixel_y_plus1
                - window_size,
                last_pixel_x_plus1 - window_size : last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part3 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - window_size : last_pixel_y_plus1
                - utils.BLOCK_Y,
                last_pixel_x_plus1 : utils.IMG_W,
            ]
            send_to_rk_plus_1 = torch.cat(
                [
                    send_to_rk_plus_1_part1.flatten(),
                    send_to_rk_plus_1_part2.flatten(),
                    send_to_rk_plus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()
    timers.stop("[loss_distribution]prepare_tensor_for_communication")

    timers.start("[loss_distribution]communication")
    communication_mode = "all2all"
    if communication_mode == "all2all":
        # a list of empty tensors of size 0
        send_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        recv_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        if utils.MP_GROUP.rank() != 0:
            recv_list[utils.MP_GROUP.rank() - 1] = recv_from_rk_minus_1_buffer
            send_list[utils.MP_GROUP.rank() - 1] = send_to_rk_minus_1
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_list[utils.MP_GROUP.rank() + 1] = recv_from_rk_plus_1_buffer
            send_list[utils.MP_GROUP.rank() + 1] = send_to_rk_plus_1

        torch.distributed.all_to_all(recv_list, send_list, group=utils.MP_GROUP)

        if utils.MP_GROUP.rank() != 0:
            recv_from_rk_minus_1 = recv_list[utils.MP_GROUP.rank() - 1]
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_from_rk_plus_1 = recv_list[utils.MP_GROUP.rank() + 1]
    else:
        raise NotImplementedError("grouped send/recv is not implemented yet.")
    timers.stop("[loss_distribution]communication")

    def n_of_elements(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    timers.start("[loss_distribution]extract_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1
            recv_from_rk_minus_1_part2 = None
            recv_from_rk_minus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part1_shape)
            ].view(*recv_from_rk_minus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part1_shape)
            recv_from_rk_minus_1_part2 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part2_shape)
            ].view(*recv_from_rk_minus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part2_shape)
            recv_from_rk_minus_1_part3 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part3_shape)
            ].view(*recv_from_rk_minus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part3_shape)
            assert (
                offset == recv_from_rk_minus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_minus_1"
    else:
        recv_from_rk_minus_1_part1 = None
        recv_from_rk_minus_1_part2 = None
        recv_from_rk_minus_1_part3 = None

    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1
            recv_from_rk_plus_1_part2 = None
            recv_from_rk_plus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part1_shape)
            ].view(*recv_from_rk_plus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part1_shape)
            recv_from_rk_plus_1_part2 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part2_shape)
            ].view(*recv_from_rk_plus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part2_shape)
            recv_from_rk_plus_1_part3 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part3_shape)
            ].view(*recv_from_rk_plus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part3_shape)
            assert (
                offset == recv_from_rk_plus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_plus_1"
    else:
        recv_from_rk_plus_1_part1 = None
        recv_from_rk_plus_1_part2 = None
        recv_from_rk_plus_1_part3 = None
    timers.stop("[loss_distribution]extract_tensor_for_communication")

    # add the received parts to the original image
    # first to make sure, result is correct;

    configs = (
        first_tile_y,
        first_tile_x,
        last_tile_y,
        last_tile_x,
        first_pixel_y,
        first_pixel_x,
        last_pixel_y_plus1,
        last_pixel_x_plus1,
        window_size,
    )

    timers.start("[loss_distribution]add_remote_pixels_to_image")
    local_image_rect_with_remote_tiles = add_remote_pixels_to_image_less_comm(
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    )
    timers.stop("[loss_distribution]add_remote_pixels_to_image")

    coverage_min_y = max(first_pixel_y - window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, :
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
        utils.get_num_pixels() * 3
    )
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.

    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss


def fast_less_comm_noallreduceloss_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    # Compare to fast_distributed_loss_computation, this method get more remote pixels during forward and do replicated loss computation for pixels near border.
    # But it avoids another communication and associated memory movement during backward.
    # This method works when image resolution is small because we want to reduce the number of kernel launches.

    timers = utils.get_timers()

    assert (
        strategy is not None
    ), "strategy should not be None in fast_distributed_loss_computation"
    assert (
        utils.BLOCK_Y > 5
    ), "utils.BLOCK_Y should be greater than 5 to make sure fast_distributed_loss_computation works as expected."

    # Compare to fast_distributed_loss_computation.
    # We avoid the communication during backward.

    window_size = 11  # we only need to get pixels within the window size(11).
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[strategy.rank],
        strategy.division_pos[strategy.rank + 1],
    )

    # there are 3 parts to send and receive
    #                3
    #          ---------
    #         2|
    #  1       |
    # ----------

    first_tile_y, first_tile_x = (
        tile_ids_l // strategy.tile_x,
        tile_ids_l % strategy.tile_x,
    )
    first_pixel_y, first_pixel_x = (
        first_tile_y * utils.BLOCK_Y,
        first_tile_x * utils.BLOCK_X,
    )

    timers.start("[loss_distribution]prepare_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            # recv from rank-1
            recv_from_rk_minus_1_buffer = torch.empty(
                (3, window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            # send to rank-1
            send_to_rk_minus_1 = image[
                :, first_pixel_y : first_pixel_y + window_size, :
            ].contiguous()
        else:
            # recv from rank-1
            recv_from_rk_minus_1_part1_shape = (3, window_size, first_pixel_x)
            recv_from_rk_minus_1_part2_shape = (3, utils.BLOCK_Y, window_size)
            recv_from_rk_minus_1_part3_shape = (
                3,
                window_size,
                utils.IMG_W - first_pixel_x,
            )
            recv_from_rk_minus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_minus_1_part1_shape[1]
                        * recv_from_rk_minus_1_part1_shape[2]
                        + recv_from_rk_minus_1_part2_shape[1]
                        * recv_from_rk_minus_1_part2_shape[2]
                        + recv_from_rk_minus_1_part3_shape[1]
                        * recv_from_rk_minus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            # send to rank-1
            send_to_rk_minus_1_part1 = image[
                :,
                first_pixel_y
                + utils.BLOCK_Y : first_pixel_y
                + utils.BLOCK_Y
                + window_size,
                0:first_pixel_x,
            ]
            send_to_rk_minus_1_part2 = image[
                :,
                first_pixel_y
                + window_size : first_pixel_y
                + utils.BLOCK_Y
                + window_size,
                first_pixel_x : min(first_pixel_x + window_size, utils.IMG_W),
            ]
            send_to_rk_minus_1_part3 = image[
                :,
                first_pixel_y : first_pixel_y + window_size,
                first_pixel_x : utils.IMG_W,
            ]

            # flatten and concatenate them together
            send_to_rk_minus_1 = torch.cat(
                [
                    send_to_rk_minus_1_part1.flatten(),
                    send_to_rk_minus_1_part2.flatten(),
                    send_to_rk_minus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()

    last_tile_y, last_tile_x = (
        tile_ids_r // strategy.tile_x,
        tile_ids_r % strategy.tile_x,
    )
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y, utils.IMG_H),
            utils.IMG_W,
        )
        # NOTE: this is tricky here.
    else:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H),
            last_tile_x * utils.BLOCK_X,
        )

    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_buffer = torch.empty(
                (3, window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            send_to_rk_plus_1 = image[
                :, last_pixel_y_plus1 - window_size : last_pixel_y_plus1, :
            ].contiguous()
        else:
            recv_from_rk_plus_1_part1_shape = (3, window_size, last_pixel_x_plus1)
            recv_from_rk_plus_1_part2_shape = (
                3,
                utils.BLOCK_Y,
                min(window_size, utils.IMG_W - last_pixel_x_plus1),
            )
            recv_from_rk_plus_1_part3_shape = (
                3,
                window_size,
                utils.IMG_W - last_pixel_x_plus1,
            )

            recv_from_rk_plus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_plus_1_part1_shape[1]
                        * recv_from_rk_plus_1_part1_shape[2]
                        + recv_from_rk_plus_1_part2_shape[1]
                        * recv_from_rk_plus_1_part2_shape[2]
                        + recv_from_rk_plus_1_part3_shape[1]
                        * recv_from_rk_plus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            send_to_rk_plus_1_part1 = image[
                :,
                last_pixel_y_plus1 - window_size : last_pixel_y_plus1,
                0:last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part2 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - window_size : last_pixel_y_plus1
                - window_size,
                last_pixel_x_plus1 - window_size : last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part3 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - window_size : last_pixel_y_plus1
                - utils.BLOCK_Y,
                last_pixel_x_plus1 : utils.IMG_W,
            ]
            send_to_rk_plus_1 = torch.cat(
                [
                    send_to_rk_plus_1_part1.flatten(),
                    send_to_rk_plus_1_part2.flatten(),
                    send_to_rk_plus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()
    timers.stop("[loss_distribution]prepare_tensor_for_communication")

    timers.start("[loss_distribution]communication")
    communication_mode = "all2all"
    if communication_mode == "all2all":
        # a list of empty tensors of size 0
        send_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        recv_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        if utils.MP_GROUP.rank() != 0:
            recv_list[utils.MP_GROUP.rank() - 1] = recv_from_rk_minus_1_buffer
            send_list[utils.MP_GROUP.rank() - 1] = send_to_rk_minus_1
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_list[utils.MP_GROUP.rank() + 1] = recv_from_rk_plus_1_buffer
            send_list[utils.MP_GROUP.rank() + 1] = send_to_rk_plus_1

        torch.distributed.all_to_all(recv_list, send_list, group=utils.MP_GROUP)

        if utils.MP_GROUP.rank() != 0:
            recv_from_rk_minus_1 = recv_list[utils.MP_GROUP.rank() - 1]
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_from_rk_plus_1 = recv_list[utils.MP_GROUP.rank() + 1]
    else:
        raise NotImplementedError("grouped send/recv is not implemented yet.")
    timers.stop("[loss_distribution]communication")

    def n_of_elements(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    timers.start("[loss_distribution]extract_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1
            recv_from_rk_minus_1_part2 = None
            recv_from_rk_minus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part1_shape)
            ].view(*recv_from_rk_minus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part1_shape)
            recv_from_rk_minus_1_part2 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part2_shape)
            ].view(*recv_from_rk_minus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part2_shape)
            recv_from_rk_minus_1_part3 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part3_shape)
            ].view(*recv_from_rk_minus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part3_shape)
            assert (
                offset == recv_from_rk_minus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_minus_1"
    else:
        recv_from_rk_minus_1_part1 = None
        recv_from_rk_minus_1_part2 = None
        recv_from_rk_minus_1_part3 = None

    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1
            recv_from_rk_plus_1_part2 = None
            recv_from_rk_plus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part1_shape)
            ].view(*recv_from_rk_plus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part1_shape)
            recv_from_rk_plus_1_part2 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part2_shape)
            ].view(*recv_from_rk_plus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part2_shape)
            recv_from_rk_plus_1_part3 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part3_shape)
            ].view(*recv_from_rk_plus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part3_shape)
            assert (
                offset == recv_from_rk_plus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_plus_1"
    else:
        recv_from_rk_plus_1_part1 = None
        recv_from_rk_plus_1_part2 = None
        recv_from_rk_plus_1_part3 = None
    timers.stop("[loss_distribution]extract_tensor_for_communication")

    # add the received parts to the original image
    # first to make sure, result is correct;

    configs = (
        first_tile_y,
        first_tile_x,
        last_tile_y,
        last_tile_x,
        first_pixel_y,
        first_pixel_x,
        last_pixel_y_plus1,
        last_pixel_x_plus1,
        window_size,
    )

    timers.start("[loss_distribution]add_remote_pixels_to_image")
    local_image_rect_with_remote_tiles = add_remote_pixels_to_image_less_comm(
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    )
    timers.stop("[loss_distribution]add_remote_pixels_to_image")

    coverage_min_y = max(first_pixel_y - window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, :
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    Ll1 = pixelwise_Ll1.sum() / (utils.get_num_pixels() * 3)
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    ssim_loss = pixelwise_ssim_loss.sum() / (utils.get_num_pixels() * 3)
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    timers.stop("local_loss_computation")

    return Ll1, ssim_loss


def functional_allreduce_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    # functional allreduce all pixels, we will have another allreduce during backward.
    # calculate the local loss, no replicated loss compute for pixels.

    timers = utils.get_timers()

    # Image allreduce
    timers.start("image_allreduce")
    if utils.MP_GROUP.size() > 1:
        torch.distributed.nn.functional.all_reduce(
            image, op=dist.ReduceOp.SUM, group=utils.MP_GROUP
        )
        # make sure non-local pixels are 0 instead of background, otherwise all_reduce sum will give 2*background.

    timers.stop("image_allreduce")

    timers.start("prepare_image_rect_and_mask")
    half_window_size = (
        5  # we only need to get pixels within half of the window size(11).
    )
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[strategy.rank],
        strategy.division_pos[strategy.rank + 1],
    )
    first_tile_y, first_tile_x = (
        tile_ids_l // strategy.tile_x,
        tile_ids_l % strategy.tile_x,
    )
    first_pixel_y, first_pixel_x = (
        first_tile_y * utils.BLOCK_Y,
        first_tile_x * utils.BLOCK_X,
    )
    last_tile_y, last_tile_x = (
        tile_ids_r // strategy.tile_x,
        tile_ids_r % strategy.tile_x,
    )
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y, utils.IMG_H),
            utils.IMG_W,
        )
    else:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H),
            last_tile_x * utils.BLOCK_X,
        )
    coverage_min_y = max(first_pixel_y - half_window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + half_window_size, utils.IMG_H)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = diff_gaussian_rasterization._C.get_pixels_compute_locally_and_in_rect(  # check this function.
        compute_locally,
        utils.IMG_H,
        utils.IMG_W,
        coverage_min_y,
        coverage_max_y,
        0,
        utils.IMG_W,
    )
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, :
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
        utils.get_num_pixels() * 3
    )
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.

    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss


def allreduce_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    # allreduce all pixels;
    # the the locally touched pixels.
    # replicated loss compute to avoid another allreduce during backward.

    timers = utils.get_timers()

    # Image allreduce
    timers.start("image_allreduce")
    if utils.MP_GROUP.size() > 1:
        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
        # make sure non-local pixels are 0 instead of background, otherwise all_reduce sum will give 2*background.
    timers.stop("image_allreduce")

    timers.start("prepare_image_rect_and_mask")
    window_size = 11
    half_window_size = (
        5  # we only need to get pixels within half of the window size(11).
    )
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[strategy.rank],
        strategy.division_pos[strategy.rank + 1],
    )
    first_tile_y, first_tile_x = (
        tile_ids_l // strategy.tile_x,
        tile_ids_l % strategy.tile_x,
    )
    first_pixel_y, first_pixel_x = (
        first_tile_y * utils.BLOCK_Y,
        first_tile_x * utils.BLOCK_X,
    )
    last_tile_y, last_tile_x = (
        tile_ids_r // strategy.tile_x,
        tile_ids_r % strategy.tile_x,
    )
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y, utils.IMG_H),
            utils.IMG_W,
        )
    else:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H),
            last_tile_x * utils.BLOCK_X,
        )

    # NOTE: here we need to locally compute larger area of pixels' loss, so that we could void communication during backward.
    coverage_min_y = max(first_pixel_y - window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    local_image_rect_gt = (
        viewpoint_cam.original_image[:, coverage_min_y:coverage_max_y, :]
        .cuda()
        .contiguous()
    )
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
        utils.get_num_pixels() * 3
    )
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.

    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss


loss_sum = 0
loss_cnt = 0


def avoid_pixel_all2all_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("prepare_image_rect_and_mask")
    window_size = 11
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[utils.MP_GROUP.rank()],
        strategy.division_pos[utils.MP_GROUP.rank() + 1],
    )
    first_tile_y = tile_ids_l // strategy.tile_x
    first_pixel_y = first_tile_y * utils.BLOCK_Y
    last_tile_y = tile_ids_r // strategy.tile_x
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1 = min(last_tile_y * utils.BLOCK_Y, utils.IMG_H)
    else:
        last_pixel_y_plus1 = min(
            last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H
        )

    coverage_min_y = max(first_pixel_y - window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, :
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    Ll1 = pixelwise_Ll1.sum() / (utils.get_num_pixels() * 3)
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    ssim_loss = pixelwise_ssim_loss.sum() / (utils.get_num_pixels() * 3)

    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.

    if args.get_global_exact_loss:
        # get the loss without redundant pixels compute, to make sure it runs correctly.
        # this is for debugging.
        with torch.no_grad():
            local_image_rect_pixels_compute_locally = diff_gaussian_rasterization._C.get_pixels_compute_locally_and_in_rect(  # check this function.
                compute_locally,
                utils.IMG_H,
                utils.IMG_W,
                coverage_min_y,
                coverage_max_y,
                0,
                utils.IMG_W,
            )
            pixelwise_Ll1 = pixelwise_l1_with_mask(
                local_image_rect,
                local_image_rect_gt,
                local_image_rect_pixels_compute_locally,
            )
            pixelwise_Ll1_sum = pixelwise_Ll1.sum()
            pixelwise_ssim_loss = pixelwise_ssim_with_mask(
                local_image_rect,
                local_image_rect_gt,
                local_image_rect_pixels_compute_locally,
            )
            pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
            two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
                utils.get_num_pixels() * 3
            )
            torch.distributed.all_reduce(
                two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP
            )
            loss = (1.0 - args.lambda_dssim) * two_losses[0] + args.lambda_dssim * (
                1.0 - two_losses[1]
            )
            log_file = utils.get_log_file()
            log_file.write(f"loss without redundant pixels compute: {loss.item()}\n")
            global loss_sum
            global loss_cnt
            loss_sum += loss.item()
            loss_cnt += 1
            if loss_cnt == 301:
                log_file.write(
                    f"epoch average loss without redundant pixels compute: {loss_sum/loss_cnt}\n"
                )
                loss_sum = 0
                loss_cnt = 0

    return Ll1, ssim_loss


def avoid_pixel_all2all_loss_computation_adjust_mode6(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("prepare_image_rect_and_mask")
    window_size = 11

    (
        (local_tile_y_l, local_tile_y_r),
        (local_tile_x_l, local_tile_x_r),
    ) = strategy.get_local_strategy()

    first_pixel_y = local_tile_y_l * utils.BLOCK_Y
    last_pixel_y_plus1 = local_tile_y_r * utils.BLOCK_Y
    first_pixel_x = local_tile_x_l * utils.BLOCK_X
    last_pixel_x_plus1 = local_tile_x_r * utils.BLOCK_X

    coverage_min_y = max(first_pixel_y - window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + window_size, utils.IMG_H)
    coverage_min_x = max(first_pixel_x - window_size, 0)
    coverage_max_x = min(last_pixel_x_plus1 + window_size, utils.IMG_W)

    local_image_rect = image[
        :, coverage_min_y:coverage_max_y, coverage_min_x:coverage_max_x
    ].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, coverage_max_x - coverage_min_x),
        dtype=torch.bool,
        device="cuda",
    )
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, coverage_min_x:coverage_max_x
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    Ll1 = pixelwise_Ll1.sum() / (utils.get_num_pixels() * 3)
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    ssim_loss = pixelwise_ssim_loss.sum() / (utils.get_num_pixels() * 3)

    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.

    return Ll1, ssim_loss


def fast_distributed_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    # This method is specific to current distribution strategy space: flatten 2D tiles to a sequence of tiles, and split a tiles sequence into sections, each allocated to a GPU.
    # Avoid redundant pixel communication and loss computation. In general distirbuted loss computation, we communicate at 16x16 tile level which include reduandant pixels.
    # Currently, I use all2all. Maybe in the future we could change to grouped send/recv if that is faster. (it seems that torch does not have this api as functional version).

    timers = utils.get_timers()

    assert (
        strategy is not None
    ), "strategy should not be None in fast_distributed_loss_computation"
    assert (
        utils.BLOCK_Y > 5
    ), "utils.BLOCK_Y should be greater than 5 to make sure fast_distributed_loss_computation works as expected."

    half_window_size = (
        5  # we only need to get pixels within half of the window size(11).
    )
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[strategy.rank],
        strategy.division_pos[strategy.rank + 1],
    )

    # there are 3 parts to send and receive
    #                3
    #          ---------
    #         2|
    #  1       |
    # ----------
    # The boundary condition is tricky here: Basically, part1 width+part3 width = utils.IMG_W; part2 height is utils.BLOCK_Y.

    first_tile_y, first_tile_x = (
        tile_ids_l // strategy.tile_x,
        tile_ids_l % strategy.tile_x,
    )
    first_pixel_y, first_pixel_x = (
        first_tile_y * utils.BLOCK_Y,
        first_tile_x * utils.BLOCK_X,
    )
    # print(f"rk: {utils.MP_GROUP.rank()}, tile_ids_l: {tile_ids_l}, tile_ids_r: {tile_ids_r}, strategy.tile_x: {strategy.tile_x}, first_tile_y: {first_tile_y}, first_tile_x: {first_tile_x}, first_pixel_y: {first_pixel_y}, first_pixel_x: {first_pixel_x}")

    timers.start("[loss_distribution]prepare_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            # recv from rank-1
            recv_from_rk_minus_1_buffer = torch.empty(
                (3, half_window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            # send to rank-1
            send_to_rk_minus_1 = image[
                :, first_pixel_y : first_pixel_y + half_window_size, :
            ].contiguous()
        else:
            # recv from rank-1
            recv_from_rk_minus_1_part1_shape = (3, half_window_size, first_pixel_x)
            recv_from_rk_minus_1_part2_shape = (3, utils.BLOCK_Y, half_window_size)
            recv_from_rk_minus_1_part3_shape = (
                3,
                half_window_size,
                utils.IMG_W - first_pixel_x,
            )
            recv_from_rk_minus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_minus_1_part1_shape[1]
                        * recv_from_rk_minus_1_part1_shape[2]
                        + recv_from_rk_minus_1_part2_shape[1]
                        * recv_from_rk_minus_1_part2_shape[2]
                        + recv_from_rk_minus_1_part3_shape[1]
                        * recv_from_rk_minus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            # send to rank-1
            send_to_rk_minus_1_part1 = image[
                :,
                first_pixel_y
                + utils.BLOCK_Y : first_pixel_y
                + utils.BLOCK_Y
                + half_window_size,
                0:first_pixel_x,
            ]
            send_to_rk_minus_1_part2 = image[
                :,
                first_pixel_y
                + half_window_size : first_pixel_y
                + utils.BLOCK_Y
                + half_window_size,
                first_pixel_x : min(first_pixel_x + half_window_size, utils.IMG_W),
            ]
            send_to_rk_minus_1_part3 = image[
                :,
                first_pixel_y : first_pixel_y + half_window_size,
                first_pixel_x : utils.IMG_W,
            ]

            # flatten and concatenate them together
            send_to_rk_minus_1 = torch.cat(
                [
                    send_to_rk_minus_1_part1.flatten(),
                    send_to_rk_minus_1_part2.flatten(),
                    send_to_rk_minus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()

    last_tile_y, last_tile_x = (
        tile_ids_r // strategy.tile_x,
        tile_ids_r % strategy.tile_x,
    )
    if tile_ids_r % strategy.tile_x == 0:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y, utils.IMG_H),
            utils.IMG_W,
        )
    else:
        last_pixel_y_plus1, last_pixel_x_plus1 = (
            min(last_tile_y * utils.BLOCK_Y + utils.BLOCK_Y, utils.IMG_H),
            last_tile_x * utils.BLOCK_X,
        )

    # print(f"rk: {utils.MP_GROUP.rank()}, last_tile_y {last_tile_y}, last_tile_x {last_tile_x}, first_pixel_y: {first_pixel_y}, first_pixel_x: {first_pixel_x}, last_pixel_y_plus1: {last_pixel_y_plus1}, last_pixel_x_plus1: {last_pixel_x_plus1}")
    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_buffer = torch.empty(
                (3, half_window_size, utils.IMG_W), dtype=torch.float32, device="cuda"
            )
            send_to_rk_plus_1 = image[
                :, last_pixel_y_plus1 - half_window_size : last_pixel_y_plus1, :
            ].contiguous()
        else:
            recv_from_rk_plus_1_part1_shape = (3, half_window_size, last_pixel_x_plus1)
            recv_from_rk_plus_1_part2_shape = (
                3,
                utils.BLOCK_Y,
                min(half_window_size, utils.IMG_W - last_pixel_x_plus1),
            )
            recv_from_rk_plus_1_part3_shape = (
                3,
                half_window_size,
                utils.IMG_W - last_pixel_x_plus1,
            )
            recv_from_rk_plus_1_buffer = torch.empty(
                (
                    3
                    * (
                        recv_from_rk_plus_1_part1_shape[1]
                        * recv_from_rk_plus_1_part1_shape[2]
                        + recv_from_rk_plus_1_part2_shape[1]
                        * recv_from_rk_plus_1_part2_shape[2]
                        + recv_from_rk_plus_1_part3_shape[1]
                        * recv_from_rk_plus_1_part3_shape[2]
                    ),
                ),
                dtype=torch.float32,
                device="cuda",
            )

            send_to_rk_plus_1_part1 = image[
                :,
                last_pixel_y_plus1 - half_window_size : last_pixel_y_plus1,
                0:last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part2 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - half_window_size : last_pixel_y_plus1
                - half_window_size,
                last_pixel_x_plus1 - half_window_size : last_pixel_x_plus1,
            ]
            send_to_rk_plus_1_part3 = image[
                :,
                last_pixel_y_plus1
                - utils.BLOCK_Y
                - half_window_size : last_pixel_y_plus1
                - utils.BLOCK_Y,
                last_pixel_x_plus1 : utils.IMG_W,
            ]
            send_to_rk_plus_1 = torch.cat(
                [
                    send_to_rk_plus_1_part1.flatten(),
                    send_to_rk_plus_1_part2.flatten(),
                    send_to_rk_plus_1_part3.flatten(),
                ],
                dim=0,
            ).contiguous()

    timers.stop("[loss_distribution]prepare_tensor_for_communication")

    timers.start("[loss_distribution]communication")
    communication_mode = "all2all"
    if communication_mode == "all2all":
        # a list of empty tensors of size 0
        send_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        recv_list = [
            torch.empty(0, dtype=torch.float32, device="cuda")
            for _ in range(utils.MP_GROUP.size())
        ]
        if utils.MP_GROUP.rank() != 0:
            recv_list[utils.MP_GROUP.rank() - 1] = recv_from_rk_minus_1_buffer
            send_list[utils.MP_GROUP.rank() - 1] = send_to_rk_minus_1
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_list[utils.MP_GROUP.rank() + 1] = recv_from_rk_plus_1_buffer
            send_list[utils.MP_GROUP.rank() + 1] = send_to_rk_plus_1

        torch.distributed.nn.functional.all_to_all(
            recv_list, send_list, group=utils.MP_GROUP
        )

        if utils.MP_GROUP.rank() != 0:
            recv_from_rk_minus_1 = recv_list[utils.MP_GROUP.rank() - 1]
        if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
            recv_from_rk_plus_1 = recv_list[utils.MP_GROUP.rank() + 1]
    else:
        raise NotImplementedError("grouped send/recv is not implemented yet.")
    timers.stop("[loss_distribution]communication")

    def n_of_elements(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    timers.start("[loss_distribution]extract_tensor_for_communication")
    if utils.MP_GROUP.rank() != 0:
        if first_pixel_x == 0:
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1
            recv_from_rk_minus_1_part2 = None
            recv_from_rk_minus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_minus_1_part1 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part1_shape)
            ].view(*recv_from_rk_minus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part1_shape)
            recv_from_rk_minus_1_part2 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part2_shape)
            ].view(*recv_from_rk_minus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part2_shape)
            recv_from_rk_minus_1_part3 = recv_from_rk_minus_1[
                offset : offset + n_of_elements(recv_from_rk_minus_1_part3_shape)
            ].view(*recv_from_rk_minus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_minus_1_part3_shape)
            assert (
                offset == recv_from_rk_minus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_minus_1"
    else:
        recv_from_rk_minus_1_part1 = None
        recv_from_rk_minus_1_part2 = None
        recv_from_rk_minus_1_part3 = None

    if utils.MP_GROUP.rank() != utils.MP_GROUP.size() - 1:
        if last_pixel_x_plus1 == utils.IMG_W:
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1
            recv_from_rk_plus_1_part2 = None
            recv_from_rk_plus_1_part3 = None
        else:
            offset = 0
            recv_from_rk_plus_1_part1 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part1_shape)
            ].view(*recv_from_rk_plus_1_part1_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part1_shape)
            recv_from_rk_plus_1_part2 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part2_shape)
            ].view(*recv_from_rk_plus_1_part2_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part2_shape)
            recv_from_rk_plus_1_part3 = recv_from_rk_plus_1[
                offset : offset + n_of_elements(recv_from_rk_plus_1_part3_shape)
            ].view(*recv_from_rk_plus_1_part3_shape)
            offset += n_of_elements(recv_from_rk_plus_1_part3_shape)
            assert (
                offset == recv_from_rk_plus_1.shape[0]
            ), "offset's final value should be equal to the total number of elements in recv_from_rk_plus_1"
    else:
        recv_from_rk_plus_1_part1 = None
        recv_from_rk_plus_1_part2 = None
        recv_from_rk_plus_1_part3 = None
    timers.stop("[loss_distribution]extract_tensor_for_communication")

    # add the received parts to the original image
    # first to make sure, result is correct;

    configs = (
        first_tile_y,
        first_tile_x,
        last_tile_y,
        last_tile_x,
        first_pixel_y,
        first_pixel_x,
        last_pixel_y_plus1,
        last_pixel_x_plus1,
        half_window_size,
    )

    timers.start("[loss_distribution]add_remote_pixels_to_image")
    local_image_rect_with_remote_tiles = add_remote_pixels_to_image(
        image,
        recv_from_rk_minus_1_part1,
        recv_from_rk_minus_1_part2,
        recv_from_rk_minus_1_part3,
        recv_from_rk_plus_1_part1,
        recv_from_rk_plus_1_part2,
        recv_from_rk_plus_1_part3,
        configs,
    )
    timers.stop("[loss_distribution]add_remote_pixels_to_image")

    coverage_min_y = max(first_pixel_y - half_window_size, 0)
    coverage_max_y = min(last_pixel_y_plus1 + half_window_size, utils.IMG_H)
    local_image_rect_pixels_compute_locally = diff_gaussian_rasterization._C.get_pixels_compute_locally_and_in_rect(  # check this function.
        compute_locally,
        utils.IMG_H,
        utils.IMG_W,
        coverage_min_y,
        coverage_max_y,
        0,
        utils.IMG_W,
    )

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    local_image_rect_gt = viewpoint_cam.original_image[
        :, coverage_min_y:coverage_max_y, :
    ].contiguous()
    local_image_rect_gt = torch.clamp(local_image_rect_gt / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect_with_remote_tiles,
        local_image_rect_gt,
        local_image_rect_pixels_compute_locally,
    )
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (
        utils.get_num_pixels() * 3
    )
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.

    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss





def replicated_loss_computation(
    image, viewpoint_cam, compute_locally, strategy, statistic_collector
):
    timers = utils.get_timers()

    # Image allreduce
    timers.start("image_allreduce")
    if utils.MP_GROUP.size() > 1:
        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
        # make sure non-local pixels are 0 instead of background, otherwise all_reduce sum will give 2*background.
    timers.stop("image_allreduce")

    # Move gt_image to gpu: if args.lazy_load_image is true, then the transfer will actually happen.
    timers.start("prepare_gt_image")
    if ("gt_image_comm_op" not in viewpoint_cam.__dict__) and (
        viewpoint_cam.gt_image_comm_op is not None
    ):
        viewpoint_cam.gt_image_comm_op.wait()
    gt_image = torch.clamp(viewpoint_cam.original_image / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")
    utils.check_initial_gpu_memory_usage("after prepare_gt_image")

    # Loss computation
    timers.start("loss")
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        image,
        gt_image,
        torch.ones((utils.IMG_H, utils.IMG_W), dtype=torch.bool, device="cuda"),
    )
    Ll1 = pixelwise_Ll1.mean()
    utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        image,
        gt_image,
        torch.ones((utils.IMG_H, utils.IMG_W), dtype=torch.bool, device="cuda"),
    )
    ssim_loss = pixelwise_ssim_loss.mean()
    utils.check_initial_gpu_memory_usage("after ssim_loss")
    timers.stop("loss")

    return Ll1, ssim_loss


########################## Create loss implementation ##########################

name2loss_implementation = {
    "replicated_loss_computation": replicated_loss_computation,
    "fast_distributed_loss_computation": fast_distributed_loss_computation,
    "general_distributed_loss_computation": general_distributed_loss_computation,
    "avoid_pixel_all2all_loss_computation": avoid_pixel_all2all_loss_computation,
    "avoid_pixel_all2all_loss_computation_adjust_mode6": avoid_pixel_all2all_loss_computation_adjust_mode6,
}


def loss_computation(
    image,
    viewpoint_cam,
    compute_locally,
    strategy,
    statistic_collector,
    image_distribution_mode,
):
    # HACK: if image is a scalar tensor, that implies there is no render. We return 0 to make sure the gradient is also 0.
    if len(image.shape) == 0:
        return image * 0, image * 0
    return name2loss_implementation[image_distribution_mode](
        image, viewpoint_cam, compute_locally, strategy, statistic_collector
    )


def get_coverage_y_min_max(tile_ids_l, tile_ids_r):
    return tile_ids_l * utils.BLOCK_Y, min(tile_ids_r * utils.BLOCK_Y, utils.IMG_H)


def get_coverage_y_min(tile_ids_l):
    return tile_ids_l * utils.BLOCK_Y


def get_coverage_y_max(tile_ids_r):
    return min(tile_ids_r * utils.BLOCK_Y, utils.IMG_H)


def load_depth_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks):
    timers = utils.get_timers()
    args = utils.get_args()

    # Asynchronously load ground-truth image to GPU
    timers.start("load_gt_image_to_gpu")

    def load_camera_from_cpu_to_gpu(first_task, last_task):
        coverage_min_max_y = {}
        coverage_min_y_first_task = get_coverage_y_min(first_task[1])
        coverage_max_y_last_task = get_coverage_y_max(last_task[2])
        for camera_id_in_batch in range(first_task[0], last_task[0] + 1):
            coverage_min_y = 0
            if camera_id_in_batch == first_task[0]:
                coverage_min_y = coverage_min_y_first_task
            coverage_max_y = utils.IMG_H
            if camera_id_in_batch == last_task[0]:
                coverage_max_y = coverage_max_y_last_task

            batched_cameras[camera_id_in_batch].invdepthmap = (
                batched_cameras[camera_id_in_batch]
                .invdepthmap_backup
                .cuda()
            )
            # if batched_cameras[camera_id_in_batch].depth_mask is not None:
            #     batched_cameras[camera_id_in_batch].depth_mask = (
            #         batched_cameras[camera_id_in_batch].depth_mask.cuda()
            #     )
            # batched_cameras[camera_id_in_batch].image_gray = (
            #     batched_cameras[camera_id_in_batch].image_gray.cuda()
            # )
            




            coverage_min_max_y[camera_id_in_batch] = (coverage_min_y, coverage_max_y)
        return coverage_min_max_y

    if args.distributed_dataset_storage:
        if args.local_sampling:
            # TODO: may preloaded
            first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
            last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
            _ = load_camera_from_cpu_to_gpu(first_task, last_task)
        elif utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.GLOBAL_RANK
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            first_task = gpuid2tasks[in_node_first_rank][0]
            last_task = gpuid2tasks[in_node_last_rank][-1]
            coverage_min_max_y_gpu0 = load_camera_from_cpu_to_gpu(first_task, last_task)
    else:
        first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
        last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
        _ = load_camera_from_cpu_to_gpu(first_task, last_task)

    timers.stop("load_gt_image_to_gpu")

    if args.local_sampling:
        return

    # Asynchronously send the original image from gpu0 to all GPUs in the same node.
    timers.start("scatter_gt_image")
    if args.distributed_dataset_storage:
        comm_ops = []
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            for rank in range(in_node_first_rank, in_node_last_rank + 1):
                if rank == utils.GLOBAL_RANK:
                    continue
                for task in gpuid2tasks[rank]:
                    camera_id = task[0]
                    coverage_min_y = get_coverage_y_min(task[1])
                    coverage_max_y = get_coverage_y_max(task[2])

                    coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                        camera_id
                    ]
                    if (
                        coverage_min_y == coverage_min_y_gpu0
                        and coverage_max_y == coverage_max_y_gpu0
                    ):
                        # less memory copy
                        op = torch.distributed.P2POp(
                            dist.isend,
                            batched_cameras[camera_id].invdepthmap.contiguous(),
                            rank,
                        )
                    else:
                        send_tensor = (
                            batched_cameras[camera_id]
                            .invdepthmap
                            .contiguous()
                        )
                        op = torch.distributed.P2POp(dist.isend, send_tensor, rank)
                    comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                camera_id = task[0]
                coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                    camera_id
                ]
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                batched_cameras[camera_id].invdepthmap = (
                    batched_cameras[camera_id]
                    .invdepthmap
                    .contiguous()
                )
        else:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            recv_buffer_list = []
            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                recv_buffer = torch.zeros(
                    (1, utils.IMG_H, utils.IMG_W),
                    dtype=torch.float32,
                    device="cuda",
                )
                recv_buffer_list.append(recv_buffer)
                op = torch.distributed.P2POp(
                    dist.irecv, recv_buffer, in_node_first_rank
                )
                comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for idx, task in enumerate(gpuid2tasks[utils.GLOBAL_RANK]):
                batched_cameras[task[0]].invdepthmap = recv_buffer_list[idx]

    timers.stop("scatter_gt_image")



def load_gray_image_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks):

    args = utils.get_args()


    def load_camera_from_cpu_to_gpu(first_task, last_task):
        coverage_min_max_y = {}
        coverage_min_y_first_task = get_coverage_y_min(first_task[1])
        coverage_max_y_last_task = get_coverage_y_max(last_task[2])
        for camera_id_in_batch in range(first_task[0], last_task[0] + 1):
            coverage_min_y = 0
            if camera_id_in_batch == first_task[0]:
                coverage_min_y = coverage_min_y_first_task
            coverage_max_y = utils.IMG_H
            if camera_id_in_batch == last_task[0]:
                coverage_max_y = coverage_max_y_last_task

            batched_cameras[camera_id_in_batch].image_gray = (
                batched_cameras[camera_id_in_batch]
                .image_gray_backup
                .cuda()
            )
            coverage_min_max_y[camera_id_in_batch] = (coverage_min_y, coverage_max_y)
        return coverage_min_max_y

    if args.distributed_dataset_storage:
        if args.local_sampling:
            # TODO: may preloaded
            first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
            last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
            _ = load_camera_from_cpu_to_gpu(first_task, last_task)
        elif utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.GLOBAL_RANK
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            first_task = gpuid2tasks[in_node_first_rank][0]
            last_task = gpuid2tasks[in_node_last_rank][-1]
            coverage_min_max_y_gpu0 = load_camera_from_cpu_to_gpu(first_task, last_task)
    else:
        first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
        last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
        _ = load_camera_from_cpu_to_gpu(first_task, last_task)



    if args.local_sampling:
        return

    # Asynchronously send the original image from gpu0 to all GPUs in the same node.

    if args.distributed_dataset_storage:
        comm_ops = []
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            for rank in range(in_node_first_rank, in_node_last_rank + 1):
                if rank == utils.GLOBAL_RANK:
                    continue
                for task in gpuid2tasks[rank]:
                    camera_id = task[0]
                    coverage_min_y = get_coverage_y_min(task[1])
                    coverage_max_y = get_coverage_y_max(task[2])

                    coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                        camera_id
                    ]
                    if (
                        coverage_min_y == coverage_min_y_gpu0
                        and coverage_max_y == coverage_max_y_gpu0
                    ):
                        # less memory copy
                        op = torch.distributed.P2POp(
                            dist.isend,
                            batched_cameras[camera_id].image_gray.contiguous(),
                            rank,
                        )
                    else:
                        send_tensor = (
                            batched_cameras[camera_id]
                            .image_gray
                            .contiguous()
                        )
                        op = torch.distributed.P2POp(dist.isend, send_tensor, rank)
                    comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                camera_id = task[0]
                coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                    camera_id
                ]
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                batched_cameras[camera_id].image_gray = (
                    batched_cameras[camera_id]
                    .image_gray
                    .contiguous()
                )
        else:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            recv_buffer_list = []
            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                recv_buffer = torch.zeros(
                    (1, utils.IMG_H, utils.IMG_W),
                    dtype=torch.float32,
                    device="cuda",
                )
                recv_buffer_list.append(recv_buffer)
                op = torch.distributed.P2POp(
                    dist.irecv, recv_buffer, in_node_first_rank
                )
                comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for idx, task in enumerate(gpuid2tasks[utils.GLOBAL_RANK]):
                batched_cameras[task[0]].image_gray = recv_buffer_list[idx]


def load_camera_from_cpu_to_all_gpu_for_eval(
    batched_cameras, batched_strategies, gpuid2tasks
):
    timers = utils.get_timers()
    args = utils.get_args()

    if args.distributed_dataset_storage:
        if args.local_sampling:
            for idx, camera in enumerate(batched_cameras):
                if camera.original_image_backup is not None:
                    camera.original_image = camera.original_image_backup.cuda()
                    scatter_list = [
                        camera.original_image for _ in range(utils.IN_NODE_GROUP.size())
                    ]
                    torch.distributed.scatter(
                        camera.original_image,
                        scatter_list=scatter_list,
                        src=utils.GLOBAL_RANK,
                        group=utils.IN_NODE_GROUP,
                    )
                else:
                    camera.original_image = torch.zeros(
                        (3, utils.IMG_H, utils.IMG_W), dtype=torch.uint8, device="cuda"
                    )
                    bsz_per_gpu = args.bsz // utils.WORLD_SIZE
                    torch.distributed.scatter(
                        camera.original_image,
                        scatter_list=None,
                        src=idx // bsz_per_gpu,
                        group=utils.IN_NODE_GROUP,
                    )
                torch.distributed.barrier(group=utils.DEFAULT_GROUP)
            return

        if utils.IN_NODE_GROUP.rank() == 0:
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
                scatter_list = [
                    camera.original_image for _ in range(utils.IN_NODE_GROUP.size())
                ]
                torch.distributed.scatter(
                    camera.original_image,
                    scatter_list=scatter_list,
                    src=utils.get_first_rank_on_cur_node(),
                    group=utils.IN_NODE_GROUP,
                )
        else:
            for camera in batched_cameras:
                camera.original_image = torch.zeros(
                    (3, utils.IMG_H, utils.IMG_W), dtype=torch.uint8, device="cuda"
                )
                torch.distributed.scatter(
                    camera.original_image,
                    scatter_list=None,
                    src=utils.get_first_rank_on_cur_node(),
                    group=utils.IN_NODE_GROUP,
                )
    else:
        for camera in batched_cameras:
            camera.original_image = camera.original_image_backup.cuda()


def load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks):
    timers = utils.get_timers()
    args = utils.get_args()

    # Asynchronously load ground-truth image to GPU
    timers.start("load_gt_image_to_gpu")

    def load_camera_from_cpu_to_gpu(first_task, last_task):
        coverage_min_max_y = {}
        coverage_min_y_first_task = get_coverage_y_min(first_task[1])
        coverage_max_y_last_task = get_coverage_y_max(last_task[2])
        for camera_id_in_batch in range(first_task[0], last_task[0] + 1):
            coverage_min_y = 0
            if camera_id_in_batch == first_task[0]:
                coverage_min_y = coverage_min_y_first_task
            coverage_max_y = utils.IMG_H
            if camera_id_in_batch == last_task[0]:
                coverage_max_y = coverage_max_y_last_task

            batched_cameras[camera_id_in_batch].original_image = (
                batched_cameras[camera_id_in_batch]
                .original_image_backup[:, coverage_min_y:coverage_max_y, :]
                .cuda()
            )
            coverage_min_max_y[camera_id_in_batch] = (coverage_min_y, coverage_max_y)
        return coverage_min_max_y

    if args.distributed_dataset_storage:
        if args.local_sampling:
            # TODO: may preloaded
            first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
            last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
            _ = load_camera_from_cpu_to_gpu(first_task, last_task)
        elif utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.GLOBAL_RANK
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            first_task = gpuid2tasks[in_node_first_rank][0]
            last_task = gpuid2tasks[in_node_last_rank][-1]
            coverage_min_max_y_gpu0 = load_camera_from_cpu_to_gpu(first_task, last_task)
    else:
        first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
        last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
        _ = load_camera_from_cpu_to_gpu(first_task, last_task)

    timers.stop("load_gt_image_to_gpu")

    if args.local_sampling:
        return

    # Asynchronously send the original image from gpu0 to all GPUs in the same node.
    timers.start("scatter_gt_image")
    if args.distributed_dataset_storage:
        comm_ops = []
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            for rank in range(in_node_first_rank, in_node_last_rank + 1):
                if rank == utils.GLOBAL_RANK:
                    continue
                for task in gpuid2tasks[rank]:
                    camera_id = task[0]
                    coverage_min_y = get_coverage_y_min(task[1])
                    coverage_max_y = get_coverage_y_max(task[2])

                    coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                        camera_id
                    ]
                    if (
                        coverage_min_y == coverage_min_y_gpu0
                        and coverage_max_y == coverage_max_y_gpu0
                    ):
                        # less memory copy
                        op = torch.distributed.P2POp(
                            dist.isend,
                            batched_cameras[camera_id].original_image.contiguous(),
                            rank,
                        )
                    else:
                        send_tensor = (
                            batched_cameras[camera_id]
                            .original_image[
                                :,
                                coverage_min_y
                                - coverage_min_y_gpu0 : coverage_max_y
                                - coverage_min_y_gpu0,
                                :,
                            ]
                            .contiguous()
                        )
                        op = torch.distributed.P2POp(dist.isend, send_tensor, rank)
                    comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                camera_id = task[0]
                coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                    camera_id
                ]
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                batched_cameras[camera_id].original_image = (
                    batched_cameras[camera_id]
                    .original_image[
                        :,
                        coverage_min_y
                        - coverage_min_y_gpu0 : coverage_max_y
                        - coverage_min_y_gpu0,
                        :,
                    ]
                    .contiguous()
                )
        else:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            recv_buffer_list = []
            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                recv_buffer = torch.zeros(
                    (3, coverage_max_y - coverage_min_y, utils.IMG_W),
                    dtype=torch.uint8,
                    device="cuda",
                )
                recv_buffer_list.append(recv_buffer)
                op = torch.distributed.P2POp(
                    dist.irecv, recv_buffer, in_node_first_rank
                )
                comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for idx, task in enumerate(gpuid2tasks[utils.GLOBAL_RANK]):
                batched_cameras[task[0]].original_image = recv_buffer_list[idx]

    timers.stop("scatter_gt_image")


def final_system_loss_computation(
    image, viewpoint_cam,  strategy, statistic_collector
):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("prepare_image_rect_and_mask")
    assert (
        utils.GLOBAL_RANK in strategy.gpu_ids
    ), "The current gpu must be used to render this camera."
    rank = strategy.gpu_ids.index(utils.GLOBAL_RANK)
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[rank],
        strategy.division_pos[rank + 1],
    )
    coverage_min_y, coverage_max_y = get_coverage_y_min_max(tile_ids_l, tile_ids_r)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    local_image_rect_gt = torch.clamp(viewpoint_cam.original_image / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    # print(viewpoint_cam.image_name)
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    Ll1 = pixelwise_Ll1.sum() / (utils.get_num_pixels() * 3)
    # utils.check_initial_gpu_memory_usage("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    ssim_loss = pixelwise_ssim_loss.sum() / (utils.get_num_pixels() * 3)

    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    # utils.check_initial_gpu_memory_usage("after ssim_loss")
    timers.stop(
        "local_loss_computation"
    )  # measure time before allreduce, so that we can get the real local time.

    return Ll1, ssim_loss


def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    points2 = depthmap2.reshape(1,-1) * rays_d
    return points1.reshape(3,H,W), points2.reshape(3,H,W)



def batched_loss_computation(
    batched_image,
    batched_return_dict,
    batched_cameras,
    # batched_compute_locally,
    batched_strategies,
    batched_statistic_collector,
    iterations,
    opt,
    batched_return_dict_nearest,
    batched_nearest_cameras,
):
    args = utils.get_args()
    timers = utils.get_timers()

    # Loss computation
    timers.start("loss_computation")
    batched_losses = []
    loss_sum = 0
    for idx, (
        image,
        render_pkg,
        camera,
        # compute_locally,
        strategy,
        statistic_collector,
        nearest_render_pkg,
        nearest_cam,
    ) in enumerate(
        zip(
            batched_image,
            batched_return_dict,
            batched_cameras,
            # batched_compute_locally,
            batched_strategies,
            batched_statistic_collector,
            batched_return_dict_nearest,
            batched_nearest_cameras,
        )
    ):
        if image is None:  # This image is not rendered locally.
            loss = 0

            batched_losses.append([0.0, 0.0])
        elif len(image.shape) == 0:  # This image is not rendered locally.
            loss = image * 0
            batched_losses.append([loss, 0.0])
        else:
            Ll1, ssim_loss = final_system_loss_computation(
                image, camera, strategy, statistic_collector
            )


            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
                1.0 - ssim_loss
            )
            if torch.isnan(loss) :
                print(f'NAN with rgb:{camera.image_name}' )
                continue
            batched_losses.append([Ll1, ssim_loss])
            # print(f"ssim_loss: {1-ssim_loss}")
            if iterations > opt.scale_loss_from_iter :
                visibility_filter = render_pkg["visibility_filter"]
                if visibility_filter is not None:
                    if visibility_filter.sum() > 0:
                        if iterations < opt.single_view_weight_from_iter * 2:
                            weight = 10.0
                        else:
                            weight = 4.0
                        # weight = 25
                        scale = (render_pkg["scales_redistributed"])[visibility_filter]
                        sorted_scale, _ = torch.sort(scale, dim=-1)
                        min_scale_loss = sorted_scale[...,0]
                        loss += weight * min_scale_loss.mean()
                            # if scale.shape[0] > 0:                        #防止高斯过大或过小
                            #     scaling_reg = scale.prod(dim=1).mean()
                            # else:
                            #     scaling_reg = torch.tensor(0.0, device="cuda")
                            # loss += 0.01 * scaling_reg # 0.01


            rank = strategy.gpu_ids.index(utils.GLOBAL_RANK)
            tile_ids_l, tile_ids_r = (
                strategy.division_pos[rank],
                strategy.division_pos[rank + 1],
            )
            coverage_min_y, coverage_max_y = get_coverage_y_min_max(tile_ids_l, tile_ids_r)

            if iterations > opt.single_view_weight_from_iter:
                weight = opt.single_view_weight
                normal = render_pkg["rendered_normal"]
                depth_normal = render_pkg["depth_normal"]
                # if normal is not None:
                gt_image = torch.clamp(camera.original_image / 255.0, 0.0, 1.0)

                image_weight = (1.0 - get_img_grad_weight(gt_image))
                image_weight = (image_weight).clamp(0,1).detach() ** 2
                # image_weight = erode(image_weight[None,None]).squeeze()
                try:
                    with torch.no_grad():
                        # normal_loss = (depth_normal - normal).abs().sum(0).mean()
                        # uncertainty_weight = (depth_normal[:, coverage_min_y:coverage_max_y, :] * normal[:, coverage_min_y:coverage_max_y, :]).abs().sum(0)
                    # loss += weight * (uncertainty_weight * (depth_normal[:, coverage_min_y:coverage_max_y, :] - normal[:, coverage_min_y:coverage_max_y, :]).abs().sum(0)).mean()    
                        normal_loss = weight * (image_weight * (((depth_normal[:, coverage_min_y:coverage_max_y, :] - normal[:, coverage_min_y:coverage_max_y, :])).abs().sum(0))).mean()

                        # normal_loss = weight * (((depth_normal[:, coverage_min_y:coverage_max_y, :] - normal[:, coverage_min_y:coverage_max_y, :])).abs().sum(0)).mean()
                    if torch.isnan(normal_loss) :
                        print(f'normal_loss loss is NaN, {camera.image_name}')
                        # continue
                    else:
                        loss += (normal_loss)
                    # loss += (normal_loss)
                except:
                    print(f'image_weight {image_weight.shape}, normal {normal.shape}, depth_normal {depth_normal.shape}, coverage_min_y {coverage_min_y},  coverage_max_y {coverage_max_y}')

            if iterations > opt.dpt_loss_from_iter:
                # if depth is not None:
                depth = render_pkg["plane_depth"]
                depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=args.dpt_end_iter)
                if depth_l1_weight(iterations) > 0 and camera.depth_reliable:
                    invDepth = 1/ depth
                    mono_invdepth = camera.invdepthmap.cuda()
                    # mask = camera.depth_mask.cuda()
                    mask = mono_invdepth != 0
                    # mask = mono_invdepth != 0
                    zero_mask_tensor = torch.zeros_like(invDepth, dtype=torch.bool)
                    zero_mask_tensor[:, coverage_min_y:coverage_max_y, :] = 1
                    # mask_tensor = zero_mask_tensor
                    mask_tensor = zero_mask_tensor & mask
                    Ll1depth_pure = torch.abs((invDepth[mask_tensor]  - mono_invdepth[mask_tensor]) ).mean()
                    Ll1depth = depth_l1_weight(iterations) * Ll1depth_pure 
                    # loss += Ll1depth
                    if torch.isnan(Ll1depth) :
                        print(f'Ll1depth oss is NaN, {camera.image_name}')
                        # continue
                    else:
                        loss += Ll1depth
                        # print('depth_loss:', Ll1depth)


            if nearest_render_pkg is not None and nearest_cam.image_name != camera.image_name:

                pixel_noise_th = opt.multi_view_pixel_noise_th
                # # total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)


                pts = get_points_from_depth(camera, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ camera.world_view_transform[:3,:3] + camera.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * camera.Fx / pts_in_view_cam[:,2] + camera.Cx,
                            pts_in_view_cam[:,1] * camera.Fy / pts_in_view_cam[:,2] + camera.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                mask = torch.zeros_like(render_pkg['plane_depth'][0], dtype=torch.bool)
                mask[coverage_min_y:coverage_max_y, :] = 1
                mask = mask.reshape(-1)

                ## sample mask
                if iterations < opt.multi_view_weight_from_iter * 1.2:
                    patch_size, sample_num, scale, pixel_noise_th = opt.multi_view_patch_size+4, 102400, 4, 2.0
                elif iterations < opt.multi_view_weight_from_iter * 1.5:
                    patch_size, sample_num, scale, pixel_noise_th = opt.multi_view_patch_size+2, 102400, 4, 1.0
                elif iterations < opt.multi_view_weight_from_iter * 2:
                    patch_size, sample_num, scale, pixel_noise_th = opt.multi_view_patch_size, 102400, 2, 1.0
                else:
                    patch_size, sample_num, scale, pixel_noise_th = opt.multi_view_patch_size, 102400, 1, 1.0

                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                    weights[~mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                d_mask = mask & d_mask
                if d_mask.sum() > 0:

                    with torch.no_grad():

                        sample_num = (coverage_max_y - coverage_min_y)*sample_num//H
                        # patch_size = 3
                        # sample_num = 102400
                        d_mask = d_mask.reshape(-1)
                        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                        if d_mask.sum() > sample_num:
                            index = np.random.choice(d_mask.sum().cpu().numpy(), round(sample_num), replace = False)
                            valid_indices = valid_indices[index]

                        weights = weights.reshape(-1)[valid_indices]
                        ## sample ref frame patch
                        pixels = pixels.reshape(-1,2)[valid_indices]
                        offsets = patch_offsets(patch_size, pixels.device)
                        ori_pixels_patch = pixels.reshape(-1, 1, 2) / 1 + offsets.float()
                        total_patch_size = (patch_size * 2 + 1) ** 2
                        gt_image_gray = camera.image_gray.cuda()
                        H, W = gt_image_gray.squeeze().shape
                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                        ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ camera.world_view_transform[:3,:3]
                        ref_to_neareast_t = -ref_to_neareast_r @ camera.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                    # ## compute Homography
                    ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                    ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                    ref_local_d = render_pkg['rendered_distance'].squeeze()
                    # rays_d = viewpoint_cam.get_rays()
                    # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                    # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                    # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                    H_ref_to_neareast = ref_to_neareast_r[None] - \
                        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                    H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                    H_ref_to_neareast = H_ref_to_neareast @camera.get_inv_k(camera.ncc_scale)
                    
                    ## compute neareast frame patch
                    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    nearest_image_gray = nearest_cam.image_gray.cuda()
                    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                    
                    ## compute loss
                    
                    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                    mask = ncc_mask.reshape(-1)
                    ncc = ncc.reshape(-1) * weights
                    ncc = ncc[mask].squeeze()

                    if mask.sum() > 0:
                        ncc_loss = ncc_weight * ncc.mean()
                        if torch.isnan(ncc_loss) or not torch.isfinite(ncc_loss):
                                ncc_loss = 0
                                print(f'[fuck ncc_loss] {camera.image_name}')
                        loss += ncc_loss 
                # try:
            timers.stop("loss_computation")


        loss_sum += loss


    assert loss_sum.dim() == 0, "The loss_sum must be a scalar tensor."
    return loss_sum * args.lr_scale_loss, batched_losses
