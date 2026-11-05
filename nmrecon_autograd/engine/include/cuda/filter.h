#pragma once
// #include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include "common/config.h"
#include "common/basic_type.h"
NMRECON_NAMESPACE_BEGIN
namespace common
{

    namespace cuda_filter
    {

        // 核函数参数限制
        constexpr int MAX_KERNEL_RADIUS = 32;
        extern __constant__ float d_kernel[2 * MAX_KERNEL_RADIUS + 1];

        // 通用 3D 高斯卷积核：按 X / Y / Z 分离方向执行
        __global__ void gaussian_blur_kernel(const float *input, float *output,
                                             int W, int H, int D,
                                             int radius, int axis);

        // 构造高斯核并复制到常量内存
        void build_gaussian_kernel(float sigma, int &radius, std::vector<float> &h_kernel);

        // 主调函数：传入 device_vector，输出 device_vector（或原地处理）
        void gaussian_filter_3d(const nmrecon::common::DeviceVector &input,
                                nmrecon::common::DeviceVector &output,
                                const nmrecon::common::Size &size,
                                precision_t sigma);

        __global__ void butterworth_kernel(cufftComplex *freq_data,
                                                 int width, int height, int batch,
                                                 float cutoff_freq, int order);

        void apply_butterworth_filter(nmrecon::common::DeviceVector &d_input,
                                            int width, int height, int batch,
                                            float cutoff_freq, int order,
                                            nmrecon::common::DeviceVector &d_output);

    }
}

NMRECON_NAMESPACE_END