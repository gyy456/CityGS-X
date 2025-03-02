import os.path,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import json
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
def load_ply(path):
    plydata = PlyData.read(path)
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_ply(path, anchor, offset, anchor_feat,  scaling, rotation, level, extra_level, opacity, voxel_size, standard_dist):
    l = []
    l.append('x')
    l.append('y')
    l.append('z')
    l.append('level')
    l.append('extra_level')
    l.append('info')
    for i in range(offset.shape[1]*offset.shape[2]):
        l.append('f_offset_{}'.format(i))
    for i in range(anchor_feat.shape[1]):
        l.append('f_anchor_feat_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    anchor = anchor.detach().cpu().numpy()
    levels = level.detach().cpu().numpy()
    extra_levels = extra_level.unsqueeze(dim=1).detach().cpu().numpy()
    infos = np.zeros_like(levels, dtype=np.float32)
    infos[0, 0] = voxel_size
    infos[1, 0] = standard_dist

    anchor_feats = anchor_feat.detach().cpu().numpy()
    offsets = offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scaling.detach().cpu().numpy()
    rots = rotation.detach().cpu().numpy()


    elements = np.empty(anchor.shape[0], dtype=dtype_full)
    attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_raw_ply(path):
    print("Loading ", path)
    plydata = PlyData.read(path)

    anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
    
    levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis]
    extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
    voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
    standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

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

    return anchor_feat, level, extra_level, offset, anchor, scaling, opacity, rotation, anchor_mask, levels, voxel_size ,standard_dist


def seamless_merge(source_path, model_path):


    catted_anchor_feat = []
    catted_offset = []
    catted_levels = []
    catted_extra_level = []
    catted_anchor = []
    catted_opacity = []
    catted_scaling = []
    catted_rotation = []


    for rk in range(4):
        one_checkpoint_path = (
            source_path + "/point_cloud_rk" + str(rk) + "_ws" + str(4) + ".ply"
        )
        anchor_feat, level, extra_level, offset, anchor, scaling, opacity, rotation, anchor_mask, levels,  voxel_size ,standard_dist = (
            load_raw_ply(one_checkpoint_path)
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
    anchor_feat = torch.tensor(catted_anchor_feat, dtype=torch.float, device="cuda")
    level = torch.tensor(catted_levels, dtype=torch.int, device="cuda")
    extra_level = torch.tensor(catted_extra_level, dtype=torch.float, device="cuda").squeeze(dim=1)
    offset = torch.tensor(catted_offset, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    anchor = torch.tensor(catted_anchor, dtype=torch.float, device="cuda")
    scaling = torch.tensor(catted_scaling, dtype=torch.float, device="cuda")
    opacity = torch.tensor(catted_opacity, dtype=torch.float, device="cuda")
    rotation = torch.tensor(catted_rotation, dtype=torch.float, device="cuda")
    anchor_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device="cuda")
    levels = torch.max(level) - torch.min(level) + 1
    levels = levels.int().item()




    save_ply(model_path, anchor, offset, anchor_feat,  scaling, rotation, level, extra_level, opacity, voxel_size, standard_dist)






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_model_path', '-m', required=True, help='Path to the pretrained model')
    parser.add_argument('--save_merge_dir', '-s', default=None, help='Path to save the PLY file')
    args = parser.parse_args()

    # local_model_path = 'results/device_PGSR/device_0'
    # block_config = 'datasets/rubble/fed_init/2_2/edge_0/edge_0/block.json'
    # save_merge_dir = f'results/device_PGSR/cloud_model_merge_device0.ply'
    seamless_merge(args.local_model_path,args.save_merge_dir)
