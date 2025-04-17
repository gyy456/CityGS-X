torchrun --standalone --nnodes=1 --nproc-per-node=8  train.py --bsz 8 -s datasets/MatrixCity/aerial/small_city/aerial/train/block_all --resolution 1 --model_path output/Matrix_opensource --iterations 150000  \
    --single_view_weight_from_iter 20000 --scale_loss_from_iter 0 --depth_l1_weight_final 0.01 --depth_l1_weight_init 0.5 --dpt_loss_from_iter 20000 --multi_view_weight_from_iter 80000 \
    --multi_view_max_angle  30 --multi_view_min_dis  0.01 --multi_view_max_dis  25 --dpt_end_iter 80000 --default_voxel_size 0.0005 

python render.py --bsz 1 -s datasets/MatrixCity/aerial/small_city/aerial/train/block_all --resolution 1 \
    --model_path output/Matrix_opensource --images images --skip_train --eval