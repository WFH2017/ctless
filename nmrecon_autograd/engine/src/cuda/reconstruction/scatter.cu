#include "cuda/reconstruction/scatter.h"
#include "cuda/filter.h"
NMRECON_NAMESPACE_BEGIN

namespace reconstruction
{
    namespace correction
    {
        namespace scatter
        {
            bool smooth_projection(nmrecon::reconstruction::rawdata::Projection &projection,
                                   float cutoff = 0.3, int order = 5)
            {
                auto projection_value = projection.get_data();
                auto projection_size = projection.get_detector_size();
                size_t num_total_views = 0;
                for (auto &trajectory : projection.get_trajectory_list())
                {
                    num_total_views += trajectory.get_num_views();
                }
                int width = projection_size.z, height = projection_size.y;
                std::cout << "num_total_views: " << num_total_views << std::endl;
                std::cout << "width: " << width << " height: " << height << std::endl;

                auto smoothed_projection_value = nmrecon::common::DeviceVector(num_total_views * width * height, 0.0f);
                nmrecon::common::cuda_filter::apply_butterworth_filter(projection_value,
                                                                       width, height, num_total_views,
                                                                       cutoff, order, smoothed_projection_value);

                // 负值置零；
                thrust::transform(smoothed_projection_value.begin(), smoothed_projection_value.end(), smoothed_projection_value.begin(), [=] __device__(float x)
                                  { return x < 0.0f ? 0.0f : x; });
                projection.set_data(smoothed_projection_value);
                return true;
            }
        }
    }
}
NMRECON_NAMESPACE_END