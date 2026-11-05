#pragma once

#include "common/namespace.h"
#include "common/basic_type.h"
#include "cuda/image3d.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{
    typedef nmrecon::common::DeviceVector ProjectionValue;
    typedef nmrecon::common::Image3D Image;

    // 用于存储nifty和dicom文件中加载的临时数据
    struct HostPinnedBuffer
    {
        precision_t *data = nullptr;
        size_t size = 0;

        ~HostPinnedBuffer()
        {
            if (data)
            {
                cudaFreeHost(data); // 自动释放
                data = nullptr;
            }
        }
    };

    template <typename T>
    HostPinnedBuffer convert_to_pinned_buffer(const std::vector<T> &input);

}
NMRECON_NAMESPACE_END
