
#include "cuda/reconstruction/attenuation.h"

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace correction
    {
        namespace attenuation
        {
            using nmrecon::common::DeviceVector;
            DeviceVector hu_to_mu_map(const std::vector<int16_t> &hu_map)
            {
                if (hu_map.empty())
                {
                    std::cerr << "警告: 输入的 HU map 为空，返回空的 DeviceVector。" << std::endl;
                    return DeviceVector(); // 返回一个空的 thrust::device_vector
                }

                // 1. 将主机 std::vector 的数据拷贝到设备 thrust::device_vector，同时进行类型转换
                // 使用 thrust::device_vector 的构造函数，它接受迭代器范围，并在此过程中进行隐式类型转换
                DeviceVector d_hu_values;
                try
                {
                    d_hu_values = DeviceVector(
                        hu_map.begin(),
                        hu_map.end());
                }
                catch (const thrust::system::system_error &e)
                {
                    std::cerr << "错误: 拷贝 HU map 到设备时发生 Thrust 错误: " << e.what() << std::endl;
                    return DeviceVector(); // 返回空 vector 表示失败
                }
                catch (const std::exception &e)
                {
                    std::cerr << "错误: 拷贝 HU map 到设备时发生标准库异常: " << e.what() << std::endl;
                    return DeviceVector(); // 返回空 vector 表示失败
                }

                // 2. 创建一个 thrust::device_vector 来存储转换后的 mu-map 值
                DeviceVector d_mu_map_values(d_hu_values.size());

                // 3. 在设备上应用 HuToMuMapFunctor 进行转换
                try
                {
                    // HuToMuMapFunctor 的构造函数接受 mu_water
                    thrust::transform(d_hu_values.begin(), d_hu_values.end(),
                                      d_mu_map_values.begin(), HuToMuMapFunctor());
                }
                catch (const thrust::system::system_error &e)
                {
                    std::cerr << "错误: 在设备上应用 mu-map functor 时发生 Thrust 错误: " << e.what() << std::endl;
                    return DeviceVector(); // 返回空 vector 表示失败
                }
                catch (const std::exception &e)
                {
                    std::cerr << "错误: 在设备上应用 mu-map functor 时发生标准库异常: " << e.what() << std::endl;
                    return DeviceVector(); // 返回空 vector 表示失败
                }

                std::cout << "成功将主机 HU map 转换为设备上的 mu-map。" << std::endl;
                
                return d_mu_map_values;
            }

        }

    } // namespace correction

}
NMRECON_NAMESPACE_END
