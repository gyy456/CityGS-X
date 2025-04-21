# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
from PIL import Image

import os

import numpy as np
import torch

from tqdm import tqdm

from database import COLMAPDatabase


RDF_TO_DRB = torch.Tensor([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, -1]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


@torch.inference_mode()
def load_cameras_and_create_txt(root, out_dir, index_file, database, scale=1):
    if index_file is None:
        image_fnames = sorted(os.listdir(os.path.join(root, 'rgbs')))
    else:
        image_fnames = np.loadtxt(index_file, dtype=str)

    image_txts = []
    camera_txts = []
    tvec_prior_txts = []
    Rs = []
    Ts = []
    Ks = []
    image_ids = []
    image_sizes = []

    # with open(os.path.join(path, transformsfile)) as json_file:

    # c2ws = np.array([frame["transform_matrix"] for frame in frames])

    for img_fname in tqdm(image_fnames):
        img = Image.open(os.path.join(root, 'rgbs', img_fname))        
        w, h = img.size
        image_sizes.append((w, h))
        findex = img_fname.split('.')[0]
        metadata = torch.load(os.path.join(root, 'metadata', f'{findex}.pt'))
        # add camera
        fx, fy, cx, cy = metadata['intrinsics']
        if isinstance(fx, torch.Tensor):
            fx = fx.item()
        if isinstance(fy, torch.Tensor):
            fy = fy.item()
        if isinstance(fx, torch.Tensor):
            cx = cx.item()
        if isinstance(fy, torch.Tensor):
            cy = cy.item()
        camera_id = database.add_camera(1, w, h, (fx, fy, cx, cy))
        # add image
        c2w = metadata['c2w'].reshape(3, 4)
        # inverse transform of https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py#L409
        c2w = torch.cat([-c2w[:, 1:2], c2w[:, :1], c2w[:, 2:]], 1)
        # inverse transform of https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py#L346-L349  #把mega-nerf反过来了
        c2w = torch.cat([RDF_TO_DRB.inverse() @ c2w[:3, :3] @ RDF_TO_DRB,
                         RDF_TO_DRB.inverse() @ c2w[:3, 3:]], -1)

        # camera-to-world to world-to-camera
        R = c2w[:3, :3].T
        t = - R @ c2w[:3, -1:]
        # world-to-camera do not need change anymore  #Blendmvs
        # R = c2w[:3, :3]
        # t = c2w[:3, -1:]

        Rs.append(R)
        Ts.append(t)
        # w2c = (torch.cat([c2w, torch.eye(4)[-1:]])).inverse()[:3, :4]
        # qx, qy, qz, qw = rot_mat_to_quaternion(R.cpu().numpy())
        q = rotmat2qvec(R.cpu().numpy())
        t = t[:, -1].tolist()
        print(img_fname)
        image_id = database.add_image(img_fname, camera_id, q, t)
        image_ids.append(image_id)

        K = torch.Tensor([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]])
        Ks.append(K)
        
        image_txts.append(f'{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {camera_id} {img_fname}')
        image_txts.append('')
        camera_txts.append(f'{camera_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}')
        tvec_prior_txts.append(f'{img_fname} {t[0]} {t[1]} {t[2]}')

    np.savetxt(os.path.join(out_dir, 'images.txt'), np.array(image_txts, dtype=str), fmt="%s")
    np.savetxt(os.path.join(out_dir, 'cameras.txt'), np.array(camera_txts, dtype=str), fmt="%s")
    np.savetxt(os.path.join(out_dir, 'tvec_priors.txt'), np.array(tvec_prior_txts, dtype=str), fmt="%s")
    # make an empty file
    f = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    f.close()

    Rs = torch.stack(Rs)
    Ts = torch.stack(Ts)
    Ks = torch.stack(Ks)
    return Rs, Ts, Ks, image_ids, image_sizes, database


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', required=True, type=str)
    parser.add_argument('--out-dir', '-out', default='./sparse_model', type=str)
    parser.add_argument('--index-file', default=None, type=str)
    parser.add_argument('--n-matched', default=20, type=int)
    parser.add_argument('--scale-factor', default=4, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max-keypoints', default=None, type=int)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'sparse'), exist_ok=True)

    if args.cuda:
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    db = COLMAPDatabase.connect(os.path.join(args.out_dir, 'database.db'))
    db.create_tables()
        
    print('load camera parameters')
    Rs, Ts, Ks, img_ids, img_sizes, database = load_cameras_and_create_txt(args.root,
                                                                           os.path.join(args.out_dir, 'sparse'),
                                                                           args.index_file,
                                                                           db,
                                                                           args.scale_factor)
    database.commit()
    database.close()
