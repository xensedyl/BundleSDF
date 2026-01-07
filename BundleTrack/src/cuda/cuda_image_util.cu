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
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float centerDepth = d_input[y * width + x];
	if (centerDepth <= 0.1f || centerDepth > zfar)
	{
		d_output[y * width + x] = 0;
		return;
	}

	unsigned int count = 0;
	unsigned int ksize = 2*structureSize+1;
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			int nx = x + j - (ksize-1)/2;
			int ny = y + i - (ksize-1)/2;
			if (nx >= 0 && nx < width && ny >= 0 && ny < height)
			{
				float depth = d_input[ny * width + nx];
				if (depth == MINF || depth < 0.1f || fabsf(depth - centerDepth) > dThresh)
				{
					count++;
				}
			}
		}
	}
	float filter = (float)count / (float)(ksize*ksize) - fracReq;
	filter = filter > 0 ? 1.0f : 0.0f;
	d_output[y * width + x] = centerDepth * filter;
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
    float dot = fminf(fmaxf(dot(view_dir, normal_dir), -1.0f), 1.0f); // Clamp for safety
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

__global__ void computeCovisibilityKernel(const int H, const int W, const int stride, Eigen::Matrix4f *cur_in_kfcam, const float visible_angle_thres, const float4 *xyz_mapA, const float4 *normalA, int *n_visible, int *n_total_gpu)
{
	const int w = (blockIdx.x*blockDim.x + threadIdx.x) * stride;
	const int h = (blockIdx.y*blockDim.y + threadIdx.y) * stride;
	if (w >= W || h >= H) return;

	const int i_pix = h*W+w;

	float4 ptA = xyz_mapA[i_pix];
	if (ptA.z<0.1) return;
	float4 normalA_tmp = normalA[i_pix];
	if (normalA_tmp.x==0 && normalA_tmp.y==0 && normalA_tmp.z==0) return;

	Eigen::Vector3f ptA_ = (*cur_in_kfcam * Eigen::Vector4f(ptA.x, ptA.y, ptA.z, 1)).head(3);
	Eigen::Vector3f normalA_ = (*cur_in_kfcam).block(0,0,3,3) * Eigen::Vector3f(normalA_tmp.x, normalA_tmp.y, normalA_tmp.z);
	Eigen::Vector3f pt_to_eye = -ptA_;
	float dot_prod = pt_to_eye.normalized().dot(normalA_.normalized());

	atomicAdd(n_total_gpu, 1);

	if (dot_prod>visible_angle_thres)
	{
		atomicAdd(n_visible, 1);
	}

}


float computeCovisibility(const int H, const int W, int umin, int vmin, int umax, int vmax, const Eigen::Matrix3f &K, const Eigen::Matrix4f &cur_in_kfcam, const float visible_angle_thres, const float4 *normalA, const float *depthA)
{
  const int n_pixels = H*W;

  float4 *xyz_map_gpu;
  cudaMalloc(&xyz_map_gpu, n_pixels*sizeof(float4));
	cudaMemset(xyz_map_gpu, 0, n_pixels*sizeof(float4));
  float4x4 K_inv_data;
  K_inv_data.setIdentity();
  Eigen::Matrix3f K_inv = K.inverse();
  for (int row=0;row<3;row++)
  {
    for (int col=0;col<3;col++)
    {
      K_inv_data(row,col) = K_inv(row,col);
    }
  }
  cuda_image_util::convert_depth_to_camera_space_float4(xyz_map_gpu, depthA, K_inv_data, W, H);

	Eigen::Matrix4f *cur_in_kfcam_gpu;
	cudaMalloc(&cur_in_kfcam_gpu, sizeof(Eigen::Matrix4f));
	cudaMemcpy(cur_in_kfcam_gpu, &cur_in_kfcam, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

  int *n_visible_gpu, *n_total_gpu;
  cudaMalloc(&n_visible_gpu, sizeof(int));
	cudaMemset(n_visible_gpu, 0, sizeof(int));
	cudaMalloc(&n_total_gpu, sizeof(int));
	cudaMemset(n_total_gpu, 0, sizeof(int));
	const int stride = 2;
  dim3 threads = {32, 32};
  dim3 blocks = {divCeil(int(W/stride), threads.x), divCeil(int(H/stride), threads.y)};
  cuda_image_util::computeCovisibilityKernel<<<blocks, threads>>>(H, W, stride, cur_in_kfcam_gpu, visible_angle_thres, xyz_map_gpu, normalA, n_visible_gpu, n_total_gpu);
  int n_visible = 0, n_total = 0;
  cutilSafeCall(cudaMemcpy(&n_visible, n_visible_gpu, sizeof(int), cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(&n_total, n_total_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  float visible = float(n_visible)/n_total;

	cutilSafeCall(cudaFree(xyz_map_gpu));
	cutilSafeCall(cudaFree(n_visible_gpu));
	cutilSafeCall(cudaFree(n_total_gpu));
	cutilSafeCall(cudaFree(cur_in_kfcam_gpu));

  return visible;
}


};
