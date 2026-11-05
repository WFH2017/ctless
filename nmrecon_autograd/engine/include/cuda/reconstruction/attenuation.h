#pragma once

#include "common/config.h"
#include "common/basic_type.h"

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace correction
    {
        namespace attenuation
        {

            static constexpr precision_t k_mu_water = 0.0154;

            struct HuToMuMapFunctor
            {
                // 构造函数可以用来传入 mu_water，如果它不是全局常量
                // 这里的 mu_water 是一个成员，方便在 operator() 中使用
                float mu_water;

                __host__ __device__ // 允许在主机和设备代码中编译
                HuToMuMapFunctor(float water_mu_val = k_mu_water) : mu_water(water_mu_val)
                {
                    // printf("water_mu_val = %f\n", water_mu_val);
                }

                // 重载 operator()，定义转换逻辑
                template <typename T> // T 可以是 short, float 等，取决于输入的HU值类型
                __host__ __device__ float operator()(T hu_value) const
                {
                    float hu_float = static_cast<float>(hu_value); // 转换为浮点数进行计算
                    float mu_140;

                    if (hu_float < 0.0f)
                    {
                        mu_140 = mu_water * (hu_float / 1000.0f + 1.0f);
                    }
                    else
                    { // HU >= 0
                        mu_140 = mu_water * (0.665f * hu_float / 1000.0f + 1.0f);
                    }
                    return mu_140;
                }
            };
            nmrecon::common::DeviceVector hu_to_mu_map(const std::vector<int16_t>&);

        } // namespace attenuation
    } // namespace correction
} // namespace reconstruction
NMRECON_NAMESPACE_END
