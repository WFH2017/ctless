#include "cuda/filter.h"
#include "cuda/def.h"
NMRECON_NAMESPACE_BEGIN

namespace common
{
    namespace cuda_filter
    {

        __constant__ float d_kernel[2 * MAX_KERNEL_RADIUS + 1];

        __global__ void gaussian_blur_kernel(
            const float *input, float *output,
            int W, int H, int D,
            int radius, int axis)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (x >= W || y >= H || z >= D)
                return;

            float val = 0.0f;
            for (int i = -radius; i <= radius; ++i)
            {
                int xx = x, yy = y, zz = z;
                if (axis == 0)
                    xx = min(max(x + i, 0), W - 1);
                else if (axis == 1)
                    yy = min(max(y + i, 0), H - 1);
                else if (axis == 2)
                    zz = min(max(z + i, 0), D - 1);
                val += input[(zz * H + yy) * W + xx] * d_kernel[i + radius];
            }
            output[(z * H + y) * W + x] = val;
        }

        void build_gaussian_kernel(float sigma, int &radius, std::vector<float> &h_kernel)
        {
            radius = std::min(int(std::ceil(3 * sigma)), MAX_KERNEL_RADIUS);
            int ksize = 2 * radius + 1;
            h_kernel.resize(ksize);
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
            {
                float v = std::exp(-0.5f * (i * i) / (sigma * sigma));
                h_kernel[i + radius] = v;
                sum += v;
            }
            for (auto &v : h_kernel)
                v /= sum;
            cudaMemcpyToSymbol(d_kernel, h_kernel.data(), ksize * sizeof(float));
        }

        void gaussian_filter_3d(
            const nmrecon::common::DeviceVector &input,
            nmrecon::common::DeviceVector &output,
            const nmrecon::common::Size &size,
            precision_t sigma)
        {
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            size_t W = size.x, H = size.y, D = size.z;
            int radius;
            std::vector<precision_t> kernel;
            build_gaussian_kernel(sigma, radius, kernel);

            thrust::device_vector<precision_t> temp1(input);
            thrust::device_vector<precision_t> temp2(W * H * D);

            dim3 block(8, 8, 8);
            dim3 grid((W + 7) / 8, (H + 7) / 8, (D + 7) / 8);

            float *ptr_in = thrust::raw_pointer_cast(temp1.data());
            float *ptr_out = thrust::raw_pointer_cast(temp2.data());

            // X, Y, Z 分别做三次一维卷积
            for (int axis = 0; axis < 3; ++axis)
            {
                gaussian_blur_kernel<<<grid, block, 0, stream>>>(ptr_in, ptr_out, W, H, D, radius, axis);
                std::swap(ptr_in, ptr_out);
            }

            output = temp1;
            cudaStreamDestroy(stream);
        }

        // CUDA核函数：对半频谱应用Butterworth低通滤波器
        __global__ void butterworth_kernel(cufftComplex *freq_data,
                                           int width, int height, int batch,
                                           float cutoff, int order)
        {
            int u = threadIdx.x + blockIdx.x * blockDim.x; // freq_width = width/2 +1
            int v = threadIdx.y + blockIdx.y * blockDim.y;
            int b = blockIdx.z;

            int freq_width = width / 2 + 1;
            if (u >= freq_width || v >= height || b >= batch)
                return;

            int idx = b * height * freq_width + v * freq_width + u;

            // 频率坐标计算（严格按 Python 非移位定义）
            float fu = float(u) / float(width); // 0 ~ 0.5

            // v 周期映射，非移位，和 Python np.where 等价
            float fv = (v < height / 2) ? float(v) / float(height) : float(v - height) / float(height);

            float D = sqrtf(fu * fu + fv * fv);

            float H = 1.0f / (1.0f + powf(D / cutoff, 2.0f * order));

            cufftComplex val = freq_data[idx];
            val.x *= H;
            val.y *= H;
            freq_data[idx] = val;
        }

        // 主函数示例，批量处理
        void apply_butterworth_filter(nmrecon::common::DeviceVector &d_input,
                                      int width, int height, int batch,
                                      float cutoff_freq, int order,
                                      nmrecon::common::DeviceVector &d_output)
        {
           int N = width * height;
    int freq_width = width / 2 + 1;
    int total_freq_size = batch * height * freq_width;

    if (d_output.size() != d_input.size())
        d_output.resize(d_input.size());

    cufftComplex* d_freq = nullptr;
    CHECK_CUDA(cudaMalloc(&d_freq, sizeof(cufftComplex) * total_freq_size));

    cufftHandle plan_fwd, plan_inv;
    int n[2] = { height, width };

    cufftPlanMany(&plan_fwd, 2, n,
                  NULL, 1, N,
                  NULL, 1, freq_width,
                  CUFFT_R2C, batch);

    cufftPlanMany(&plan_inv, 2, n,
                  NULL, 1, freq_width,
                  NULL, 1, N,
                  CUFFT_C2R, batch);

    cufftExecR2C(plan_fwd, (cufftReal*)thrust::raw_pointer_cast(d_input.data()), d_freq);

    dim3 block(16, 16);
    dim3 grid((freq_width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              batch);

    butterworth_kernel<<<grid, block>>>(d_freq, width, height, batch, cutoff_freq, order);
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftExecC2R(plan_inv, d_freq, (cufftReal*)thrust::raw_pointer_cast(d_output.data()));

    // 归一化
    int total_size = batch * N;
    thrust::device_ptr<float> out_ptr(d_output.data());
    thrust::transform(out_ptr, out_ptr + total_size, out_ptr,
                      [=] __device__(float val) { return val / N; });

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_freq);
        }
    }
}

NMRECON_NAMESPACE_END