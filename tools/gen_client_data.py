# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
import os
import random

import numpy as np
import torch

from tqdm import tqdm


def gen_client_data(c2ws: np.ndarray, n_data: int):
    """
    Args:
        c2ws (np.ndarray): camera extrinsic (camera2world) that is
                        an ndarray of shape (#cameras, 3, 4).
                        the coordinate system is following mega-nerf,
                        i.e., (down, right, backward)
        n_data (int): a number of data for a client
                        
    Returns:
        indices (np.ndarray): client's data indices
    """
    n_cameras = c2ws.shape[0]
    xyz_coord = c2ws[:, :3, -1] # (#cameras, 3)
    
    base_camera_idx = np.random.randint(0, n_cameras)
    center_xyz = xyz_coord[base_camera_idx]
    dists = np.sum(np.square(xyz_coord - center_xyz), -1)
    
    indices = np.argsort(dists, 0)[:n_data]
    return np.sort(indices)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d',
                        required=True,
                        type=str,
                        help='/path/to/dataset')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        type=str,
                        help='/path/to/output_dir')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed')
    parser.add_argument('--n-clients',
                        default=200,
                        type=int,
                        help='number of clients')
    parser.add_argument('--n-data-min', '-min',
                        default=100,
                        type=int,
                        help='minimum number of clients data')
    parser.add_argument('--n-data-max', '-max',
                        default=200,
                        type=int,
                        help='maximum number of clients data')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    fnames = sorted(os.listdir(os.path.join(args.dataset_dir, 'train/rgbs')))
    print('load metadatas')
    c2ws = []
    for fname in tqdm(fnames):
        c2ws.append(torch.load(os.path.join(args.dataset_dir, 'train/metadata', fname.split('.')[0] + '.pt'))['c2w'].numpy())
    c2ws = np.stack(c2ws)
    
    print('split data')
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n_clients):
        n_data = np.random.randint(args.n_data_min, args.n_data_max + 1)
        indices = gen_client_data(c2ws, n_data)
        training_image_names = [fnames[idx] for idx in indices]
        np.savetxt(os.path.join(args.output_dir, str(i).zfill(5) + '.txt'), training_image_names, fmt="%s")
