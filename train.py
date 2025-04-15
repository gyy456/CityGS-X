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

import os
import torch
import sys
import json
from utils.general_utils import safe_state, init_distributed
import utils.general_utils as utils
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import train_internal
import debugpy
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument('--not_use_dpt_loss', action='store_true', help='Do not use DPT loss')
    parser.add_argument('--not_use_single_view_loss', action='store_true', help='Do not use single view loss')
    parser.add_argument('--not_use_multi_view_loss', action='store_true', help='Do not use multi view loss')
    args = parser.parse_args(sys.argv[1:])

    # Set up distributed training


    if args.not_use_dpt_loss:
        args.dpt_loss_from_iter = args.iterations
        assert args.dpt_loss_from_iter >= args.iterations
    if args.not_use_multi_view_loss:
        args.multi_view_weight_from_iter = args.iterations
        assert args.multi_view_weight_from_iter >= args.iterations
    if args.not_use_single_view_loss:
        args.single_view_weight_from_iter = args.iterations
        assert args.single_view_weight_from_iter >= args.iterations

    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 在调用分布式初始化之前初始化调试器
    # port = 5690 + rank  # 每个进程使用不同的端口
    # if rank==0:
    #     debugpy.listen(('0.0.0.0', port))  # 启动调试器并监听不同的端口
    #     print(f"Process {rank} waiting for debugger to attach on port {port}...")
    #     debugpy.wait_for_client()  # 程序在这里暂停，直到调试器连接
    init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    args = utils.get_args()

    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(
            group=utils.DEFAULT_GROUP
        )  # log_folder is created before other ranks start writing log.
    if utils.GLOBAL_RANK == 0:
        with open(args.log_folder + "/args.json", "w") as f:
            json.dump(vars(args), f)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(
        args.log_folder
        + "/python_ws="
        + str(utils.WORLD_SIZE)
        + "_rk="
        + str(utils.GLOBAL_RANK)
        + ".log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    train_internal.training(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )

    # All done
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    utils.print_rank_0("\nTraining complete.")
