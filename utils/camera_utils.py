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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorch_1, get_args, get_log_file
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
from multiprocessing import shared_memory
import torch
from PIL import Image
from kornia import create_meshgrid

import cv2

import os




WARNED = False
def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def loadCam(args, id, cam_info, decompressed_image=None, return_image=False, depth_reliables = None, invdepthmaps = None , normal_mask = None, noraml_gt = None, depth_mask = None, resized_image_gray = None):
    orig_w, orig_h = cam_info.width, cam_info.height
    # assert (
    #     orig_w // args.resolution == utils.get_img_width() and orig_h // args.resolution == utils.get_img_height()
    # ), "All images should have the same size. "

    args = get_args()
    log_file = get_log_file()
    # resolution = orig_w, orig_h
    # NOTE: we do not support downsampling here.
    # if "MatrixCity" in cam_info.image_path:
    #     if orig_w > 1600:
    #         global WARNED
    #         if not WARNED:
    #             utils.print_rank_0("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                 "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #             WARNED = True
    #         global_down = orig_w / 1600   #原分辨率下降 global_down > 1
    #         scale = float(global_down) 

    #     resolution = (int(orig_w / scale), int(orig_h / scale))
    # else:
    resolution = round(orig_w/(args.resolution)), round(orig_h/(args.resolution))
    # may use cam_info.uid
    if (
        (
            args.local_sampling
            and args.distributed_dataset_storage
            and utils.GLOBAL_RANK == id % utils.WORLD_SIZE
        )
        or (
            not args.local_sampling
            and args.distributed_dataset_storage
            and utils.LOCAL_RANK == 0
        )
        or (not args.distributed_dataset_storage)
    ):
        if args.time_image_loading:
            start_time = time.time()
        image = Image.open(cam_info.image_path)
        # resized_image_gray = image .convert('L')
        resized_image_rgb = PILtoTorch(
            image, resolution, args, log_file, decompressed_image=decompressed_image
        )
        if args.time_image_loading:
            log_file.write(f"PILtoTorch image in {time.time() - start_time} seconds\n")

        # assert resized_image_rgb.shape[0] == 3, "Image should have exactly 3 channels!"
        gt_image = resized_image_rgb[:3, ...].contiguous()
        loaded_mask = None

        # Free the memory: because the PIL image has been converted to torch tensor, we don't need it anymore. And it takes up lots of cpu memory.
        image.close()
        image = None
        if return_image: 
            if "MatrixCity" in cam_info.image_path:
                depth_path = cam_info.image_path.replace("images/", "depth/", 1)
                mask_path = cam_info.image_path.replace('images','mask')
                mask_path = mask_path.replace('jpg','png')
            else:
                depth_path = cam_info.image_path.replace('rgbs','depths')
                mask_path = cam_info.image_path.replace('rgbs','mask')
                mask_path = mask_path.replace('jpg','png')

            depth_path = depth_path.replace('jpg','png')
            _normal_path = cam_info.image_path.replace('rgbs','normals')
            
            resolution = (round(orig_w/(args.resolution)), round(orig_h/(args.resolution)))

            if not args.not_use_multi_view_loss:
                image = Image.open(cam_info.image_path)
                resized_image_gray = image .convert('L')
                resized_image_gray = PILtoTorch_1(resized_image_gray, resolution)
                image.close()
                image = None
            else:
                resized_image_gray = None

            if os.path.exists(depth_path) and not args.not_use_dpt_loss:
                invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
                invdepthmap = cv2.resize(invdepthmap, resolution)
                invdepthmap[invdepthmap < 0] = 0
                depth_reliable = True
                depth_params = cam_info.depth_params
                if depth_params is not None:
                    # if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    #     utils.print_rank_0("cam_info.image_name")
                    # else:
                    #     utils.print_rank_0("False")
                    #     # self.depth_mask *= 0
                    if depth_params["scale"] > 0:
                        invdepthmap = invdepthmap * depth_params["scale"] + depth_params["offset"]  #统一尺度
                if invdepthmap.ndim != 2:
                    invdepthmap = invdepthmap[..., 0]
                invdepthmap = torch.from_numpy(invdepthmap[None])
            else:
                depth_reliable = None
                invdepthmap = None
            if os.path.exists(mask_path) and ~args.not_use_dpt_loss:
                mask_color = cv2.imread(mask_path, -1).astype(np.float32) 
                # mask = mask != 0
                mask = np.any(mask_color != [0, 0, 0], axis=-1)
                mask = torch.from_numpy(mask)
                invdepthmap[mask.unsqueeze(0)==0] = 0
            else:
                mask = None

            if os.path.exists(_normal_path):
                # _normal = Image.open(_normal_path)
                # resized_normal = PILtoTorch_1(_normal, resolution)
                # resized_normal = resized_normal[:3]
                # _normal = - (resized_normal * 2 - 1)
                # # normalize normal
                # _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
                # _normal = _normal.permute(2, 0, 1)
                # normal_norm = torch.norm(_normal, dim=0, keepdim=True)
                # normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
                # noraml_gt= _normal/ normal_norm
                normal_gt = None
                normal_mask = None
            

    else:
        gt_image = None
        loaded_mask = None
        depth_reliable = None
        invdepthmap = None
        mask = None 
        noraml_gt = None 
        normal_mask = None
        resized_image_gray =None

        # invdepthmap[mask.unsqueeze(0)] = 0
        # mask = None
    if return_image:
        return gt_image, depth_reliable, invdepthmap, mask, noraml_gt, normal_mask, resized_image_gray

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        image_height = round(orig_h/(args.resolution)),
        image_width = round(orig_w/(args.resolution)),
        uid=id,
        depth_params=cam_info.depth_params,
        image_path=cam_info.image_path,
        depth_reliables = depth_reliables, 
        invdepthmaps = invdepthmaps,
        normal_mask = normal_mask,
        noraml_gt =  noraml_gt,
        depth_mask = depth_mask,
        resized_image_gray = resized_image_gray
    )


def load_decompressed_image(params):
    args, id, cam_info = params
    return loadCam(args, id, cam_info, decompressed_image=None, return_image=True)


# Modify this code to support shared_memory.SharedMemory to make inter-process communication faster
def decompressed_images_from_camInfos_multiprocess(cam_infos, args):
    args = get_args()
    decompressed_images = []
    depth_reliables= []
    invdepthmaps = []
    depth_mask = []
    normal_masks = []
    noramls_gt = []
    resized_image_gray = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    with multiprocessing.Pool(processes=24) as pool:
        # Prepare data for processing
        tasks = [(args, id, cam_info) for id, cam_info in enumerate(cam_infos)]

        # Map load_camera_data to the tasks
        # results = pool.map(load_decompressed_image, tasks)
        results = list(
            tqdm(
                pool.imap(load_decompressed_image, tasks),
                total=total_cameras,
                disable=(utils.LOCAL_RANK != 0),
            )
        )
        # print(len(results))
        for id, result in enumerate(results):
            decompressed_images.append(result[0])
            depth_reliables.append(result[1])
            invdepthmaps.append(result[2])
            depth_mask.append(result[3])
            noramls_gt.append(result[4])
            normal_masks.append(result[5])
            resized_image_gray.append(result[6])

    return decompressed_images, depth_reliables, invdepthmaps, depth_mask, noramls_gt, normal_masks, resized_image_gray



def decompressed_images_from_camInfos_multiprocess_single_gpu(cam_infos, args):
    args = get_args()
    decompressed_images = []
    depth_reliables= []
    invdepthmaps = []
    depth_mask = []
    normal_masks = []
    noramls_gt = []
    resized_image_gray = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    # with multiprocessing.Pool(processes=24) as pool:
        # Prepare data for processing
    tasks = [(args, id, cam_info) for id, cam_info in enumerate(cam_infos)]

    # Map load_camera_data to the tasks
    # results = pool.map(load_decompressed_image, tasks)
    for task in tqdm(tasks, total=total_cameras):
        result = load_decompressed_image(task)
        decompressed_images.append(result[0])
        depth_reliables.append(result[1])
        invdepthmaps.append(result[2])
        depth_mask.append(result[3])
        noramls_gt.append(result[4])
        normal_masks.append(result[5])
        resized_image_gray.append(result[6])
    

    return decompressed_images, depth_reliables, invdepthmaps, depth_mask, noramls_gt, normal_masks, resized_image_gray


def decompress_and_scale_image(cam_info):
    pil_image = cam_info.image
    resolution = cam_info.image.size  # (w, h)
    # print("cam_info.image.size: ", cam_info.image.size)
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = np.array(resized_image_PIL)  # (h, w, 3)
    # print("resized_image.shape: ", resized_image.shape)
    if len(resized_image.shape) == 3:
        return resized_image.transpose(2, 0, 1)
    else:
        return resized_image[..., np.newaxis].transpose(2, 0, 1)


def load_decompressed_image_shared(params):
    shared_mem_name, args, id, cam_info, resolution_scale = params
    # Retrieve the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)

    # Assume each image will be stored as a flat array in shared memory
    # Example: using numpy for manipulation; adjust size and dtype as needed
    resolution_width, resolution_height = cam_info.image.size
    image_shape = (3, resolution_height, resolution_width)  # Set appropriate values
    dtype = np.uint8  # Adjust as per your image data type

    # Calculate the offset for this particular image
    offset = id * np.prod(image_shape)
    np_image_array = np.ndarray(
        image_shape, dtype=dtype, buffer=existing_shm.buf, offset=offset
    )

    # Decompress image into the numpy array directly
    decompressed_image = decompress_and_scale_image(cam_info)  # Implement this
    np_image_array[:] = decompressed_image

    # Clean up
    existing_shm.close()


def decompressed_images_from_camInfos_multiprocess_sharedmem(
    cam_infos, resolution_scale, args
):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Assume each image shape and dtype
    resolution_width, resolution_height = cam_infos[0].image.size
    image_shape = (
        3,
        resolution_height,
        resolution_width,
    )  # Define these as per your data
    dtype = np.uint8
    image_size = np.prod(image_shape) * np.dtype(dtype).itemsize

    # Create shared memory
    total_size = image_size * total_cameras
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    # Create a pool of processes
    with multiprocessing.Pool(64) as pool:
        # Prepare data for processing
        tasks = [
            (shm.name, args, id, cam_info, resolution_scale)
            for id, cam_info in enumerate(cam_infos)
        ]

        # print("Start Parallel loading...")
        # Map load_camera_data to the tasks
        list(
            tqdm(pool.imap(load_decompressed_image_shared, tasks), total=total_cameras)
        )

    # Read images from shared memory
    decompressed_images = []
    for id in range(total_cameras):
        offset = id * np.prod(image_shape)
        np_image_array = np.ndarray(
            image_shape, dtype=dtype, buffer=shm.buf, offset=offset
        )
        decompressed_images.append(
            torch.from_numpy(np_image_array)
        )  # Make a copy if necessary

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return decompressed_images




def set_rays_od(cams):
    for id, cam in enumerate(cams):
        rayd=1
        if rayd is not None:
            projectinverse = cam.projection_matrix.T.inverse()
            camera2wold = cam.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(cam.image_height, cam.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            xindx = pixgrid[:,:,0] # x
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, cam.image_height), pix2ndc(xindx, cam.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4
            projected = ndccamera @ projectinverse.T
            diretioninlocal = projected / projected[:,:,3:] #v
            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T
            # rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            rays_d = direction
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            cam.rayo = cam.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()
            cam.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()
        else :
            cam.rayo = None
            cam.rayd = None







def cameraList_from_camInfos(cam_infos, args):
    args = get_args()

    if utils.DEFAULT_GROUP.size() > 1 and args.multiprocesses_image_loading:
        decompressed_images, depth_reliables, invdepthmaps,  depth_masks, noramls_gt, normal_masks, resized_image_gray = decompressed_images_from_camInfos_multiprocess(
            cam_infos, args
        )
        # decompressed_images = decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args)
    else:
        decompressed_images, depth_reliables, invdepthmaps,  depth_masks, noramls_gt, normal_masks, resized_image_gray = decompressed_images_from_camInfos_multiprocess_single_gpu(
            cam_infos, args
        )
        # decompressed_images = [None for _ in cam_infos]
        # depth_reliables = [None for _ in cam_infos]
        # invdepthmaps = [None for _ in cam_infos]
        # normal_masks = [None for _ in cam_infos]
        # noramls_gt = [None for _ in cam_infos]
        

    camera_list = []
    for id, c in tqdm(
        enumerate(cam_infos), total=len(cam_infos), disable=(utils.LOCAL_RANK != 0)
    ):
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                decompressed_image=decompressed_images[id],
                return_image=False,
                depth_reliables = depth_reliables[id], 
                invdepthmaps = invdepthmaps[id],
                normal_mask = normal_masks[id],
                noraml_gt = noramls_gt[id],
                depth_mask = depth_masks[id], 
                resized_image_gray = resized_image_gray[id]
            )
        )

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
