/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
// #define BLOCK_X 32
#define BLOCK_X 16
// #define BLOCK_X 8
#define BLOCK_Y 16
// #define BLOCK_Y 8
#define ONE_DIM_BLOCK_SIZE 256
// #define ONE_DIM_BLOCK_SIZE 128
// #define ONE_DIM_BLOCK_SIZE 64
#define NUM_ALL_MAP 5
// Currently, I need to change the block size here and rebuild the project to take effect;
// To rebuild, there is a environment setting bug: the compiler cannot notive my changes in the header file. 
// Therefore, I must delelte the build folder and rebuild the project from scratch. 
// TODO: fix the environment setting but and try to make it more flexible in the future.
#endif