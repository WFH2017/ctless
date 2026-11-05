#include "cuda/io/utils.h"
#include "cuda/io/binary.h"
#include "cuda/filter.h"
#include <fstream>


NMRECON_NAMESPACE_BEGIN
namespace io
{
    void statistic(ProjectionValue &proj_value, const std::string &name = "")
    {
        thrust::device_vector<precision_t>::iterator max_it = thrust::max_element(proj_value.begin(), proj_value.end());
        precision_t max_value = *max_it;
        thrust::device_vector<precision_t>::iterator min_it = thrust::min_element(proj_value.begin(), proj_value.end());
        precision_t min_value = *min_it;
        // 使用thrust::transform_reduce计算总和
        precision_t sum = thrust::transform_reduce(proj_value.begin(), proj_value.end(),
                                                   thrust::identity<precision_t>(),
                                                   0.0f,
                                                   thrust::plus<precision_t>());

        // 计算平均值
        int num_elements = proj_value.size();
        precision_t average = sum / num_elements;

        // 输出结果
        std::cout << "Variable: " << name << ", max:" << max_value << ", average: " << average << ", min:" << min_value << std::endl;
    }

    void load_from_binary(const std::string &filepath, ProjectionValue &projection)
    {
        size_t num_elements = projection.size();
        size_t buffer_bytes = num_elements * sizeof(float);

        // 1. 在主机端分配零拷贝内存 (pinned memory)
        float *h_pinned_buffer = nullptr;
        cudaError_t cuda_status = cudaHostAlloc((void **)&h_pinned_buffer, buffer_bytes, cudaHostAllocMapped);
        if (cuda_status != cudaSuccess)
        {
            std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(cuda_status) << std::endl;
            std::abort();
        }

        std::ifstream ifs(filepath, std::ios::binary | std::ios::in);

        if (!ifs.is_open())
        {
            std::cerr << "File not found: " << filepath << std::endl;
            cudaFreeHost(h_pinned_buffer); // 记得释放分配的内存
            std::abort();
        }

        // 2. 将数据从文件读取到零拷贝内存
        ifs.read(reinterpret_cast<char *>(h_pinned_buffer), buffer_bytes);
        ifs.close();

        // 3. 使用 thrust::copy 将数据从零拷贝内存拷贝到 device_vector
        // thrust::copy 内部会识别出这是零拷贝内存，并利用优化传输
        thrust::copy(h_pinned_buffer, h_pinned_buffer + num_elements, projection.begin());

        // 4. 释放零拷贝内存
        cuda_status = cudaFreeHost(h_pinned_buffer);
        if (cuda_status != cudaSuccess)
        {
            std::cerr << "cudaFreeHost failed: " << cudaGetErrorString(cuda_status) << std::endl;
            // 不 abort，因为数据已经拷贝，但记录错误
        }
    }



    // 步骤5：将固定内存中的数据复制到GPU
    void copy_from_pinned_to_gpu(const HostPinnedBuffer &buffer,
                                 ProjectionValue &proj_value,
                                 ProjectionValue &scatter_value)
    {
        const size_t num_elements = buffer.size;
        precision_t *host_ptr = buffer.data;
        if (num_elements == proj_value.size() * 2)
        {
            size_t half_elements = num_elements / 2;

            if (proj_value.size() != half_elements)
                proj_value.resize(half_elements);
            if (scatter_value.size() != half_elements)
                scatter_value.resize(half_elements);

            cudaMemcpy(thrust::raw_pointer_cast(proj_value.data()),
                       host_ptr,
                       half_elements * sizeof(precision_t),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(thrust::raw_pointer_cast(scatter_value.data()),
                       host_ptr + half_elements,
                       half_elements * sizeof(precision_t),
                       cudaMemcpyHostToDevice);
        }
        else
        {
            if (proj_value.size() != num_elements)
                proj_value.resize(num_elements);

            cudaMemcpy(thrust::raw_pointer_cast(proj_value.data()),
                       host_ptr,
                       num_elements * sizeof(precision_t),
                       cudaMemcpyHostToDevice);
        }
    }

    nmrecon::common::Image3D dose_calibrated(const Image &image_in, precision_t dose_calibration_factor)
    {
        auto scaled_image = image_in * dose_calibration_factor;
        return scaled_image;
    }

    nmrecon::common::DeviceVector post_filtering(const Image &image_in, precision_t gaussian_sigma)
    {
        // scale the reconstructed image with dose calibration.

        if (gaussian_sigma > 0.0f)
        {
            nmrecon::common::DeviceVector d_float_image;
            nmrecon::common::cuda_filter::gaussian_filter_3d(image_in.get_data(), d_float_image,
                                                             image_in.get_size(), gaussian_sigma);
            return d_float_image;
        }
        else
        {
            return image_in.get_data();
        }
    }

    std::vector<uint16_t> convertFloatDeviceToUint16Host(const nmrecon::common::DeviceVector &d_input)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::vector<uint16_t> out_image(d_input.size());
        try
        {
            thrust::device_vector<uint16_t> d_output(d_input.size());

            thrust::transform(
                thrust::cuda::par.on(stream),
                d_input.begin(), d_input.end(),
                d_output.begin(),
                [] __host__ __device__(float x)
                {
                    return static_cast<uint16_t>(fminf(65535.0f, fmaxf(0.0f, x + 0.5f)));
                });

            cudaStreamSynchronize(stream);

            //  使用 host policy，不使用 stream
            thrust::copy(d_output.begin(), d_output.end(), out_image.begin());
        }
        catch (const thrust::system_error &e)
        {
            std::cerr << "Thrust error: " << e.what() << std::endl;
            out_image.clear();
        }
        cudaStreamDestroy(stream);
        return out_image;
    }
}
NMRECON_NAMESPACE_END
