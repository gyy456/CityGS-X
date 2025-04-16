torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_num>  train.py --bsz <bsz> \
    -s datasets/<scene_name> --resolution 4 --model_path output/<save_path> \
    --iterations 100000 --images train/rgbs --single_view_weight_from_iter 10000 \
    --depth_l1_weight_final 0.01 --depth_l1_weight_init 0.5 --dpt_loss_from_iter 10000\
    --dpt_end_iter 30000 --scale_loss_from_iter 0 \
    --multi_view_weight_from_iter 30000 --default_voxel_size 0.001