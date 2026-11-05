#include "io/common.h"
NMRECON_NAMESPACE_BEGIN
namespace io
{
    template <typename T>
    HostPinnedBuffer convert_to_pinned_buffer(const std::vector<T> &input)
    {
        HostPinnedBuffer buffer;
        buffer.size = input.size();
        cudaHostAlloc((void **)&buffer.data, buffer.size * sizeof(float), cudaHostAllocDefault);

        for (size_t i = 0; i < buffer.size; ++i)
        {
            buffer.data[i] = static_cast<float>(input[i]);
        }

        return buffer;
    }

    // ✅ 显式模板实例化
    template HostPinnedBuffer convert_to_pinned_buffer<uint16_t>(const std::vector<uint16_t> &);
    template HostPinnedBuffer convert_to_pinned_buffer<int16_t>(const std::vector<int16_t> &);
    template HostPinnedBuffer convert_to_pinned_buffer<float>(const std::vector<float> &);
}

NMRECON_NAMESPACE_END
