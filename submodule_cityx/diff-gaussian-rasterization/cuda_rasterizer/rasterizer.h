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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <torch/extension.h>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// Either of the above two headers could make the compilation successful; 
// however, <pybind11/pybind11.h> will make the compilation very slow.

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);


		/////////////////////////////// Preprocess ///////////////////////////////

		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			bool debug);

		// Forward rendering procedure for differentiable rasterization
		// of Gaussians.
		static int preprocessForward(
			float2* means2D,
			float* depths,
			int* radii,
			float* cov3D,
			float4* conic_opacity,
			float* rgb,
			bool* clamped,//the above are all per-Gaussian intemediate results.
			const int P, int D, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float* rotations,
			const float* shs,
			const float* opacities,//3dgs parameters
			const float scale_modifier,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			bool debug,//raster_settings
			const pybind11::dict &args);

		static void preprocessBackward(
			const int* radii,
			const float* cov3D,
			const bool* clamped,//the above are all per-Gaussian intemediate results.
			const int P, int D, int M, int R,
			const int width, int height,//rasterization setting.
			const float* means3D,
			const float* scales,
			const float* rotations,
			const float* shs,//input of this operator
			const float scale_modifier,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,//rasterization setting.
			const float* dL_dmean2D,
			const float* dL_dconic,
			float* dL_dcolor,//gradients of output of this operator
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dsh,//gradients of input of this operator
			bool debug,
			const pybind11::dict &args);
		

		////////////////////// GetDistributionStrategy ////////////////////////

		static void getDistributionStrategy(
			std::function<char* (size_t)> distBuffer,
			const int P,
			const int width, int height,
			float2* means2D,
			int* radii,
			bool* compute_locally,
			bool debug,
			const pybind11::dict &args);



		/////////////////////////////// Render ///////////////////////////////




		static int renderForward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* background,
			const int width, int height,
			const float focal_x, const float focal_y,
			const float cx, const float cy,
			const float* viewmatrix,
			const float* cam_pos,
			const float* all_map,
			float2* means2D,//TODO: do I have to add const for it? However, internal means2D is not const type. 
			float* depths,
			int* radii,
			float4* conic_opacity,
			float* rgb,
			bool* compute_locally,
			float* out_color,
			int* n_render,// TODO: int* could not match with uint32_t*. error may occur, especially when the number is large.
			int* n_consider,// If your uint32_t array contains values higher than 2,147,483,647, they will overflow when converted to int.
			int* n_contrib,//array of results for this function. 
			int* out_observe,
			float* out_all_map,
			float* out_plane_depth,
			const bool render_geo,
			bool debug,
			const pybind11::dict &args);

		static void renderBackward(
			const int P, int R,
			const float* background,
			const int width, int height,//rasterization settings. 
			const float tan_fovx, float tan_fovy,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,//buffer that contains intermedia results
			bool* compute_locally,
			const float* dL_dpix,//gradient of output
			const float* dL_dout_all_map,  //gyy
			const float* dL_dout_plane_depth,
			float* dL_dmean2D,
			float* dL_dmean2D_abs, //gyy
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,//gradient of inputs
			float* dL_dall_map,
			const float2* means2D,
			const float4* conic_opacity,
			const float* rgb,//inputs
			const float* all_maps, //gyy
			const float* all_map_pixels,
			const bool render_geo,
			bool debug,
			const pybind11::dict &args);


	};
};

#endif