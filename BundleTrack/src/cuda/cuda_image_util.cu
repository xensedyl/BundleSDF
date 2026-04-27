/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_image_util.h"
#include "cudaUtil.h"
#include "common.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)


namespace cuda_image_util
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Depth to Camera Space Positions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convert_depth_to_camera_space_float4_kernel(
    float4* d_output, const float* dmap, float4x4 intrinsicsInv,
    unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    float depth = dmap[idx];

    // initialize output
    d_output[idx] = make_float4(0, 0, 0, 0);

    if (depth >= 0.1f)
    {
        float4 p_img = make_float4(float(x) * depth, float(y) * depth, depth, depth);
        float4 p_cam = intrinsicsInv * p_img;
        d_output[idx] = make_float4(p_cam.x, p_cam.y, p_cam.z, 1.0f);
    }
}

void convert_depth_to_camera_space_float4(
    float4* xyz_map,
    const float* dmap,
    const float4x4& intrinsicsInv,
    unsigned int w,
    unsigned int h)
{
    dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    dim3 gridSize((w + T_PER_BLOCK - 1) / T_PER_BLOCK,
                  (h + T_PER_BLOCK - 1) / T_PER_BLOCK);

	convert_depth_to_camera_space_float4_kernel<<<gridSize, blockSize>>>(
        xyz_map, dmap, intrinsicsInv, w, h);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void compute_normals_kernel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Out of bounds: set to zero and return
    if (x >= width || y >= height) {
        return;
    }

    d_output[y * width + x] = make_float4(0, 0, 0, 0);

    // Define a threshold for valid depth difference
    const float z_diff_thres = 0.02f;

    // Check bounds for safe neighbor access
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Fetch points for normal computation
        const float4 center = d_input[y * width + x];
		// If center depth is invalid, we can't compute normal
        if (center.z < 0.1f) return;
		float3 x_dir = make_float3(0, 0, 0);

        const float4 below = d_input[(y + 1) * width + x];
        const float4 above = d_input[(y - 1) * width + x];

		// X direction vector (vertical neighbors)
        if (below.z >= 0.1f && above.z >= 0.1f &&
            fabsf(below.z - center.z) <= z_diff_thres && fabsf(above.z - center.z) <= z_diff_thres)
        {
            x_dir = make_float3(below) - make_float3(above);
        }
        else if (below.z >= 0.1f && fabsf(below.z - center.z) <= z_diff_thres)
        {
            x_dir = make_float3(below) - make_float3(center);
        }
        else if (above.z >= 0.1f && fabsf(above.z - center.z) <= z_diff_thres)
        {
            x_dir = make_float3(above) - make_float3(center);
        }
        else
        {
            return;
        }

        const float4 left = d_input[y * width + (x - 1)];
		const float4 right = d_input[y * width + (x + 1)];

        

        
        float3 y_dir = make_float3(0, 0, 0);

        

        // Y direction vector (horizontal neighbors)
        if (right.z >= 0.1f && left.z >= 0.1f &&
            fabsf(right.z - center.z) <= z_diff_thres && fabsf(left.z - center.z) <= z_diff_thres)
        {
            y_dir = make_float3(right) - make_float3(left);
        }
        else if (right.z >= 0.1f && fabsf(right.z - center.z) <= z_diff_thres)
        {
            y_dir = make_float3(right) - make_float3(center);
        }
        else if (left.z >= 0.1f && fabsf(left.z - center.z) <= z_diff_thres)
        {
            y_dir = make_float3(left) - make_float3(center);
        }
        else
        {
            return;
        }

        float3 normal = cross(x_dir, y_dir);
        float normal_length = length(normal);

        if (normal_length > 0.0f) {
            normal = normal / normal_length;

            // Ensure normal points towards the camera
            if (dot(normal, make_float3(-center.x, -center.y, -center.z)) < 0) {
                normal = -normal;
            }

            d_output[y * width + x] = make_float4(normal, 0.0f);
        }
    }
}

void compute_normals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
    dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK,
                  (height + T_PER_BLOCK - 1) / T_PER_BLOCK);

    compute_normals_kernel<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Erode Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void erode_depthmap_kernel(float* d_output, const float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq, float zfar)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (x >= 0 && x < width && y >= 0 && y < height)
	{


		unsigned int count = 0;

		float oldDepth = d_input[y*width + x];
		if (oldDepth<=0.1f || oldDepth>zfar)
		{
			d_output[y*width + x] = 0;
			return;
		}
        int target = (int)ceil((float)(2 * structureSize + 1)*(2 * structureSize + 1)*fracReq);
		for (int i = -structureSize; i <= structureSize; i++)
		{
			for (int j = -structureSize; j <= structureSize; j++)
			{
				if (x + j >= 0 && x + j < width && y + i >= 0 && y + i < height)
				{
					float depth = d_input[(y + i)*width + (x + j)];
					if (depth == MINF || depth < 0.1f || fabs(depth - oldDepth) > dThresh)
					{
						target--;
					}
				}
			}
		}

		if (target<=0) {
			d_output[y*width + x] = 0;
		}
		else {
			d_output[y*width + x] = d_input[y*width + x];
		}
	}
}

void erode_depthmap(float* d_output, const float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq, float zfar)
{
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);

	erode_depthmap_kernel<<<gridSize, blockSize>>>(d_output, d_input, structureSize, width, height, dThresh, fracReq, zfar);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Gaussian_filter_depth_map_kernel(
    float* d_output, 
    const float* d_input, 
    int radius, 
    float sigmaD, 
    float sigmaR, 
    unsigned int width, 
    unsigned int height, 
    const float zfar)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;
    const int kernelSize = 2 * radius + 1;

    const float centerDepth = d_input[idx];
    d_output[idx] = 0.0f;

    // Compute local mean depth in window (excluding invalid depths)
    float mean_depth = 0.0f;
    int valid_count = 0;
    for (int ky = -radius; ky <= radius; ++ky) {
        int ny = y + ky;
        if (ny < 0 || ny >= height) continue;
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = x + kx;
            if (nx < 0 || nx >= width) continue;
            float d = d_input[ny * width + nx];
            if (d >= 0.1f && d <= zfar) {
                mean_depth += d;
                ++valid_count;
            }
        }
    }
    if (valid_count == 0)
        return;
    mean_depth /= (float)valid_count;

    // Apply bilateral (gaussian & range) kernel using only values near local mean
    float sum = 0.0f;
    float weightSum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        int ny = y + ky;
        if (ny < 0 || ny >= height) continue;
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = x + kx;
            if (nx < 0 || nx >= width) continue;
            float d = d_input[ny * width + nx];
            if (d >= 0.1f && d <= zfar && fabsf(d - mean_depth) < 0.01f) {
                float spatial = ((float)kx) * kx + ((float)ky) * ky;
                float range = (centerDepth - d) * (centerDepth - d);

                float weight = expf(-spatial / (2.0f * sigmaD * sigmaD) - range / (2.0f * sigmaR * sigmaR));
                sum += d * weight;
                weightSum += weight;
            }
        }
    }

    float total_kernel = (float)(kernelSize * kernelSize);
    if (weightSum > 0.0f && ((float)valid_count / total_kernel) > 0.0f) {
        d_output[idx] = sum / weightSum;
    }
}

void Gaussian_filter_dmap(
    float* d_output, 
    const float* d_input, 
    int radius, 
    float sigmaD, 
    float sigmaR, 
    unsigned int width, 
    unsigned int height, 
    const float zfar)
{
    dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, 
                  (height + T_PER_BLOCK - 1) / T_PER_BLOCK);

    Gaussian_filter_depth_map_kernel<<<gridSize, blockSize>>>(d_output, d_input, radius, sigmaD, sigmaR, width, height, zfar);

#ifdef DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter Depth Smoothed Edges
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Improved kernel with better readability, reusing intermediate variables, and clear structure
__global__ void filter_dmap_smoothed_edges_kernel(
    float* d_output,
    const float* d_input,
    const float4* d_normal,
    unsigned int width,
    unsigned int height,
    float angle_thres,
    float fx, float fy, float cx, float cy)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    float depth = d_input[idx];
    if (depth < 0.1f) return;

    // Compute view direction (normalized)
    float3 view_dir = make_float3(
        (u - cx) * depth / fx,
        (v - cy) * depth / fy,
        depth
    );
    view_dir = normalize(view_dir);

    // Compute normal direction (normalized)
    float4 nrm = d_normal[idx];
    float3 normal_dir = make_float3(nrm.x, nrm.y, nrm.z);
    normal_dir = normalize(normal_dir);

    // Compute angle between view direction and normal
    float dot_val = view_dir.x * normal_dir.x + view_dir.y * normal_dir.y + view_dir.z * normal_dir.z;
    float dot = fminf(fmaxf(dot_val, -1.0f), 1.0f); // Clamp for safety
    float angle = acosf(dot); // returns value in [0, pi]

    // Suppress depth on edges
    const float pi_2 = 1.5707963267948966f; // Precomputed pi/2
    if (fabsf(angle - pi_2) < angle_thres) {
        d_output[idx] = 0.0f;
    } else {
        d_output[idx] = depth;
    }
}

void filter_dmap_smoothed_edges(
    float* d_output,
    const float* d_input,
    const float4* d_normal,
    unsigned int width,
    unsigned int height,
    float angle_thres,
    float fx, float fy, float cx, float cy)
{
    dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    dim3 gridSize(
        (width + T_PER_BLOCK - 1) / T_PER_BLOCK,
        (height + T_PER_BLOCK - 1) / T_PER_BLOCK
    );

    filter_dmap_smoothed_edges_kernel<<<gridSize, blockSize>>>(
        d_output, d_input, d_normal, width, height,
        angle_thres, fx, fy, cx, cy
    );

#ifdef DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Covisibility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// More efficient, readable version using shared memory reduction for block-level partial sums

__global__ void compute_covisibility_reduction_kernel(
    int H, int W, int stride,
    Eigen::Matrix4f *cur_in_kfcam,
    float visible_angle_thres,
    const float4 *xyz_mapA,
    const float4 *normalA,
    unsigned int *visible_block_sum,
    unsigned int *total_block_sum)
{
    extern __shared__ unsigned int sdata[]; // 2*num threads per block: [visible][total]
    unsigned int* s_visible = sdata;
    unsigned int* s_total = &sdata[blockDim.x * blockDim.y];

    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    s_visible[tid] = 0;
    s_total[tid] = 0;

    const int w = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const int h = (blockIdx.y * blockDim.y + threadIdx.y) * stride;

    if (w < W && h < H) {
        const int idx = h * W + w;
        float4 ptA = xyz_mapA[idx];
        float4 normalA_tmp = normalA[idx];

        if (ptA.z >= 0.1f && (normalA_tmp.x != 0.f || normalA_tmp.y != 0.f || normalA_tmp.z != 0.f)) {
            Eigen::Vector3f ptA_ = (*cur_in_kfcam * Eigen::Vector4f(ptA.x, ptA.y, ptA.z, 1)).head(3);
            Eigen::Vector3f normalA_ = (*cur_in_kfcam).block(0,0,3,3) * Eigen::Vector3f(normalA_tmp.x, normalA_tmp.y, normalA_tmp.z);
            Eigen::Vector3f pt_to_eye = -ptA_;
            float dot_prod = pt_to_eye.normalized().dot(normalA_.normalized());

            s_total[tid] = 1;
            s_visible[tid] = (dot_prod > visible_angle_thres) ? 1 : 0;
        }
    }
    __syncthreads();

    // Reduce visible and total counts within block
    int threadsPerBlock = blockDim.x * blockDim.y;
    for (int stride = threadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_visible[tid] += s_visible[tid + stride];
            s_total[tid] += s_total[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        visible_block_sum[blockIdx.y * gridDim.x + blockIdx.x] = s_visible[0];
        total_block_sum[blockIdx.y * gridDim.x + blockIdx.x] = s_total[0];
    }
}

float compute_covisibility(
    const int H, const int W, int umin, int vmin, int umax, int vmax,
    const Eigen::Matrix3f &K,
    const Eigen::Matrix4f &cur_in_kfcam,
    const float visible_angle_thres,
    const float4 *normalA, const float *depthA)
{
    const int n_pixels = H * W;

    float4 *xyz_map_gpu = nullptr;
    cudaMalloc(&xyz_map_gpu, n_pixels * sizeof(float4));
    cudaMemset(xyz_map_gpu, 0, n_pixels * sizeof(float4));
    float4x4 K_inv_data;
    K_inv_data.setIdentity();
    Eigen::Matrix3f K_inv = K.inverse();
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            K_inv_data(row, col) = K_inv(row, col);

    cuda_image_util::convert_depth_to_camera_space_float4(xyz_map_gpu, depthA, K_inv_data, W, H);

    Eigen::Matrix4f *cur_in_kfcam_gpu = nullptr;
    cudaMalloc(&cur_in_kfcam_gpu, sizeof(Eigen::Matrix4f));
    cudaMemcpy(cur_in_kfcam_gpu, &cur_in_kfcam, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

    const int stride = 2;
    dim3 threads(16, 16);
    dim3 blocks(
        (W + threads.x * stride - 1) / (threads.x * stride),
        (H + threads.y * stride - 1) / (threads.y * stride)
    );
    int num_blocks = blocks.x * blocks.y;

    unsigned int *visible_block_sum_gpu = nullptr;
    unsigned int *total_block_sum_gpu = nullptr;
    cudaMalloc(&visible_block_sum_gpu, num_blocks * sizeof(unsigned int));
    cudaMalloc(&total_block_sum_gpu, num_blocks * sizeof(unsigned int));
    cudaMemset(visible_block_sum_gpu, 0, num_blocks * sizeof(unsigned int));
    cudaMemset(total_block_sum_gpu, 0, num_blocks * sizeof(unsigned int));

    size_t shared_mem_size = 2 * threads.x * threads.y * sizeof(unsigned int);

    compute_covisibility_reduction_kernel<<<blocks, threads, shared_mem_size>>>(
        H, W, stride, cur_in_kfcam_gpu, visible_angle_thres, xyz_map_gpu, normalA,
        visible_block_sum_gpu, total_block_sum_gpu);

    std::vector<unsigned int> visible_block_sum(num_blocks, 0);
    std::vector<unsigned int> total_block_sum(num_blocks, 0);

    cutilSafeCall(cudaMemcpy(visible_block_sum.data(), visible_block_sum_gpu, num_blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(total_block_sum.data(), total_block_sum_gpu, num_blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int n_visible = 0, n_total = 0;
    for (int i = 0; i < num_blocks; ++i) {
        n_visible += visible_block_sum[i];
        n_total += total_block_sum[i];
    }

    float visible = n_total > 0 ? float(n_visible) / n_total : 0.0f;

    cutilSafeCall(cudaFree(xyz_map_gpu));
    cutilSafeCall(cudaFree(visible_block_sum_gpu));
    cutilSafeCall(cudaFree(total_block_sum_gpu));
    cutilSafeCall(cudaFree(cur_in_kfcam_gpu));

    return visible;
}


};
