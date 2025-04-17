#!/bin/bash
#SBATCH -N 1 -n 12 --gres=gpu:1 -p gvlab -A gvlab

module load anaconda/2022.10
module load cuda/11.8
module load gcc/9.3.0
echo 111
source activate grendel-gyy 
echo 222

# torchrun --standalone --nnodes=1 --nproc-per-node=8  train.py --bsz 8 -s datasets/MatrixCity/aerial/small_city/aerial/train/block_all --resolution 1 --model_path output/Matrix_opensource --iterations 200000  \
#     --single_view_weight_from_iter 20000 --scale_loss_from_iter 0 --depth_l1_weight_final 0.01 --depth_l1_weight_init 1 --dpt_loss_from_iter 20000 --multi_view_weight_from_iter 80000 \
#     --multi_view_max_angle  30 --multi_view_min_dis  0.01 --multi_view_max_dis  25 --dpt_end_iter 100000 --default_voxel_size 0.0005 
        # --self.multi_view_min_dis  0.01
        # --self.multi_view_max_dis  1.5

python render.py --bsz 1 -s datasets/MatrixCity/aerial/small_city/aerial/test/block_all_test --resolution 1 \
    --model_path output/Matrix_opensource --images images --skip_train

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s /ailab/user/gaoyuanyuan_p/datasets/sci_art --resolution 4    --model_path output/sci_art_shuffle --iterations 100000 --images train/rgbsca
# python  multi_view_precess.py    -s datasets/sci_art --resolution 4    --model_path datasets/sci_art/mask_2 --iterations 100000 --images train/rgbs  
# python utils/make_depth_scale.py --base_dir datasets/sci_art/ --depths_dir datasets/sci_art/train/depths/
# python  normaldepth.py    -s datasets/sci_art --resolution 4    --model_path datasets/sci_art/train --iterations 100000 --images train/rgbs
# torchrun --standalone --nnodes=1 --nproc-per-node=4  train.py --bsz 4 -s huace_street --resolution 2    --model_path output/huace_street --iterations 100000


# python  train.py --bsz 1 -s datasets/rubble --resolution 4 --model_path output/rubble_1gpu_1bsz    --iterations 100000 --images train/rgbs \
#     --single_view_weight_from_iter 10000 --scale_loss_from_iter 0 --depth_l1_weight_final 0.01 --depth_l1_weight_init  --dpt_loss_from_iter 10000 --multi_view_weight_from_iter 100000


# torchrun --standalone --nnodes=1 --nproc-per-node=4  render_mesh.py -s datasets/rubble --model_path  output/rubble_w_geo  --skip_test  --bsz 4  --images train/rgbs\
#     --global_model_path dataset/BlendMVS/5b60fa0c764f146feef84df0_test25/device_0_dptloss/global_model.pth \
#     --resolution 4 \
#     --max_depth 0.5 \
#     --voxel_size 0.001 \
# merge.py 融合多块 