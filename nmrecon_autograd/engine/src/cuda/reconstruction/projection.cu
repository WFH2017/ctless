#include "cuda/reconstruction/projection.h"
#include "cuda/device_arithmetic.h"
#include "cuda/host_arithmetic.h"

NMRECON_NAMESPACE_BEGIN

namespace reconstruction
{
    namespace rawdata
    {
        using namespace nmrecon::common::device_arithmetic;

        Projection::Projection()
            : trajectory_list(),
              detector_size(),
              projection_value() {}

        Projection::Projection(const Projection &projection)
            : trajectory_list(projection.trajectory_list),
              detector_size(projection.detector_size),
              projection_value(projection.projection_value) {}

        Projection::Projection(const TrajectoryList &trajectory_list_,
                               const Size &detector_size_)
            : trajectory_list(trajectory_list_),
              detector_size(detector_size_)
        {
            size_t total_num_rays = 0;
            size_t num_rays_per_view = detector_size.product();

            for (size_t i = 0; i < trajectory_list.size(); i++)
            {
                total_num_rays += trajectory_list[i].get_num_views() * num_rays_per_view;
            }
            projection_value = DeviceVector(total_num_rays, 0.0f);
        }

        Projection::Projection(const TrajectoryList &trajectory_list_,
                               const Size &detector_size_,
                               const DeviceVector &data_)
            : trajectory_list(trajectory_list_),
              detector_size(detector_size_),
              projection_value(data_)
        {

            size_t num_rays_per_view = detector_size.product();
            // detector_size.print();

            size_t total_num_rays = 0;
            // Check if the data size matches the number of rays
            for (size_t i = 0; i < trajectory_list.size(); i++)
            {
                total_num_rays += trajectory_list[i].get_num_views() * num_rays_per_view;
                // trajectory_list[i].print();
                // std::cout << "num_rays_per_view: " << num_rays_per_view << ", num_views: " << trajectory_list[i].get_num_views() << std::endl;
            }

            if (data_.size() != total_num_rays)
            {
                std::stringstream ss;
                ss << "Data size mismatch. Provided data size is " << data_.size()
                   << ", but expected " << total_num_rays << " rays based on trajectory and detector size.";
                // throw std::runtime_error(ss.str());
            }
        }

        Projection &Projection::operator=(const Projection &projection)
        {
            if (this != &projection)
            {
                trajectory_list = projection.trajectory_list;
                detector_size = projection.detector_size;
                projection_value = projection.projection_value;
            }
            return *this;
        }

        DeviceProjection Projection::to_device()
        {
            return {trajectory_list, detector_size, thrust::raw_pointer_cast(projection_value.data())};
        }

        DeviceVector Projection::get_data() const
        {
            return projection_value;
        }
        DeviceVector &Projection::get_data_ref()
        {
            return projection_value;
        }
        void Projection::set_data(const DeviceVector &data_)
        {
            projection_value = data_;
        }

        Projection Projection::operator+(const Projection &other) const
        {
            if (projection_value.size() != other.projection_value.size())
            {
                std::stringstream ss;
                ss << "Projection data size mismatch: "
                   << "this size = " << projection_value.size()
                   << ", other size = " << other.projection_value.size();
                throw std::runtime_error(ss.str());
            }
            DeviceVector result_data = projection_value + other.projection_value;
            return Projection(trajectory_list, detector_size, result_data);
        }
        
        Projection Projection::operator+(const precision_t &value) const
        {
            DeviceVector result_data = projection_value + value;
            return Projection(trajectory_list, detector_size, result_data);
        }

        Projection Projection::operator-(const Projection &other) const
        {
            if (projection_value.size() != other.projection_value.size())
            {
                std::stringstream ss;
                ss << "Projection data size mismatch: "
                   << "this size = " << projection_value.size()
                   << ", other size = " << other.projection_value.size();
                throw std::runtime_error(ss.str());
            }
            DeviceVector result_data = projection_value - other.projection_value;
            return Projection(trajectory_list, detector_size, result_data);
        }
        Projection Projection::operator-(const precision_t &value) const
        {
            DeviceVector result_data = projection_value - value;
            return Projection(trajectory_list, detector_size, result_data);
        }
        Projection Projection::operator*(const Projection &other) const
        {
            if (projection_value.size() != other.projection_value.size())
            {
                std::stringstream ss;
                ss << "Projection data size mismatch: "
                   << "this size = " << projection_value.size()
                   << ", other size = " << other.projection_value.size();
                throw std::runtime_error(ss.str());
            }
            DeviceVector result_data = projection_value * other.projection_value;
            return Projection(trajectory_list, detector_size, result_data);
        }
        Projection Projection::operator*(const precision_t &value) const
        {
            DeviceVector result_data = projection_value * value;
            return Projection(trajectory_list, detector_size, result_data);
        }
        Projection Projection::operator/(const Projection &other) const
        {
            if (projection_value.size() != other.projection_value.size())
            {
                std::stringstream ss;
                ss << "Projection data size mismatch: "
                   << "this size = " << projection_value.size()
                   << ", other size = " << other.projection_value.size();
                throw std::runtime_error(ss.str());
            }
            DeviceVector result_data = true_div(projection_value, other.projection_value);
            return Projection(trajectory_list, detector_size, result_data);
        }
        Projection Projection::operator/(const precision_t &value) const
        {
            if (value == 0)
            {
                throw std::runtime_error("Division by zero in Projection operator/");
            }
            DeviceVector result_data = projection_value / value;
            return Projection(trajectory_list, detector_size, result_data);
        }

        void Projection::fill(const precision_t &value)
        {
            thrust::fill(projection_value.begin(), projection_value.end(), value);
        }


        TrajectoryList Projection::get_trajectory_list() const
        {
            return trajectory_list;
        }

        Size Projection::get_detector_size() const
        {
            return detector_size;
        }

        /* deprecated
        __host__ host_vector<Projection> Projection::get_sub_projections_old(const size_t num_subsets) const
        {

            // Check if the number of sub projections is valid, if not, set to 1
            if (num_subsets < 1 || num_subsets > trajectory_list[0].get_num_views())
            {
                std::cerr << "[WARNING]: Invalid number " << num_subsets << " for sub projections, set to 1." << std::endl;
                DeviceVector p_value = this->get_data();
                auto det_size = this->get_detector_size();
                auto traj_list = this->get_trajectory_list();
                host_vector<Projection> sub_projections(1, {traj_list, det_size, p_value});
                return sub_projections;
            }
            if (trajectory_list[0].get_num_views() % num_subsets != 0)
            {
                std::cerr << "[WARNING]: The number of views " << trajectory_list[0].get_num_views() << " is not divisible by the number of sub projections " << num_subsets << std::endl;
                std::cerr << "Set the number of subsets again" << std::endl;
                std::exit(1);
            }

            host_vector<nmrecon::scanner::TrajectoryList> subset_traj_list(num_subsets);

            host_vector<Projection> sub_projections(num_subsets);

            int num_trajectories = trajectory_list.size();
            int num_views_per_subset = trajectory_list[0].get_num_views() / num_subsets;
            int num_elements_per_view = detector_size.product();
            int total_elements_per_subset = num_trajectories * num_views_per_subset * num_elements_per_view;
            // loop subsets
            for (int s = 0; s < num_subsets; s++)
            {
                host_vector<precision_t> subset_value = HostVector(total_elements_per_subset, 0.0f);
                int index = 0;
                // loop trajectory list
                for (int t = 0; t < num_trajectories; t++)
                {
                    // construct subset trajectory list
                    auto traj = trajectory_list[t];
                    // spilit the trajectory into subsets
                    auto subset_trajs = traj.get_sub_trajectories(num_subsets);
                    // reorganize the subset trajectory list
                    subset_traj_list[s].push_back(subset_trajs[s]);

                    auto current_subset_traj = subset_trajs[s];
                    for (int v = 0; v < num_views_per_subset; v++)
                    {
                        int start = (s + v * num_subsets + t * num_views_per_subset * num_subsets) * num_elements_per_view;
                        subset_value.insert(subset_value.begin() + index, projection_value.begin() + start, projection_value.begin() + start + num_elements_per_view);
                        index += num_elements_per_view;
                    }
                }
                DeviceVector subset_value_device = subset_value;
                sub_projections[s] = Projection{subset_traj_list[s], detector_size, subset_value_device};
            }

            return sub_projections;
        }
            */

        __host__ thrust::host_vector<Projection> Projection::get_sub_projections(const size_t num_subsets) const
        {
            // 假设成员变量已经更新为 `_host` 和 `_device` 后缀
            // 例如：`trajectory_list_host`, `detector_size_host`, `projection_value_device`

            // --- 输入验证 (改为抛出异常而不是退出程序，更鲁棒) ---
            if (num_subsets < 1)
            {
                std::stringstream ss;
                ss << "[ERROR]: 子集数量必须至少为1，但收到 " << num_subsets << "。";
                throw std::invalid_argument(ss.str());
            }

            if (trajectory_list.empty())
            {
                std::stringstream ss;
                ss << "[ERROR]: 轨迹列表为空。无法创建子集。";
                throw std::runtime_error(ss.str());
            }

            size_t num_views = trajectory_list[0].get_num_views();

            if (num_subsets > num_views)
            {
                std::stringstream ss;
                ss << "[ERROR]: 子集数量 (" << num_subsets << ") 不能超过总视图数 ("
                   << num_views<< ")。";
                throw std::invalid_argument(ss.str());
            }

            if (num_views % num_subsets != 0)
            {
                std::stringstream ss;
                ss << "[ERROR]: 视图数 (" << num_views
                   << ") 不能被子集数量 (" << num_subsets << ") 整除。";
                throw std::invalid_argument(ss.str());
            }
            // --- 输入验证结束 ---

            int num_trajectories = static_cast<int>(trajectory_list.size()); // 多个探头
            int num_views_per_subset = static_cast<int>(num_views / num_subsets); //每个探头多个视图
            int num_elements_per_view = static_cast<int>(detector_size.product());
            int total_elements_per_subset_data = num_trajectories * num_views_per_subset * num_elements_per_view;

            // 获取原始完整投影数据在设备上的裸指针
            auto full_device_data_ptr = thrust::raw_pointer_cast(projection_value.data());

            // 计算每个轨迹在完整投影数据 DeviceVector 中的起始偏移量
            std::vector<size_t> trajectory_global_offsets(num_trajectories);
            size_t current_offset = 0;
            for (int t = 0; t < num_trajectories; ++t)
            {
                trajectory_global_offsets[t] = current_offset;
                current_offset += trajectory_list[t].get_num_views() * num_elements_per_view;
            }

            // 存储每个子集划分后的 TrajectoryList (仍为主机端对象)
            host_vector<TrajectoryList> subset_traj_lists(num_subsets);
            // 最终要返回的 Projection 对象向量
            host_vector<Projection> sub_projections(num_subsets);

            // 遍历每个子集
            for (int s = 0; s < num_subsets; ++s)
            {
                // 为当前子集的数据创建一个新的 DeviceVector
                DeviceVector subset_value_device(total_elements_per_subset_data);
                auto current_subset_device_write_ptr = thrust::raw_pointer_cast(subset_value_device.data()); // 指向要写入的设备内存

                size_t write_offset_in_subset_device_vector = 0; // 在当前子集 DeviceVector 中的写入偏移

                // 遍历每个轨迹（例如：每个时间周期）
                for (int t = 0; t < num_trajectories; ++t)
                {
                    // 构造子集轨迹列表 (主机端)
                    auto full_traj_for_t = trajectory_list[t];                                   // 获取此轨迹的完整轨迹
                    auto subset_trajs_for_t = full_traj_for_t.get_sub_trajectories(num_subsets); // 将其划分为子轨迹

                    // 存储当前子集 's' 对应的子轨迹
                    if (s < subset_trajs_for_t.size())
                    {
                        subset_traj_lists[s].push_back(subset_trajs_for_t[s]);
                    }
                    else
                    {
                        // 这表示 get_sub_trajectories 或子集索引有问题
                        throw std::runtime_error("逻辑错误：子轨迹索引超出范围。");
                    }

                    // 拷贝属于当前子集 's' 的投影数据（来自此轨迹的所有视图）
                    for (int v = 0; v < num_views_per_subset; ++v)
                    {
                        // 计算在完整 DeviceVector 中的原始视图索引
                        int original_view_idx_in_trajectory_t = s + v * num_subsets;

                        // 计算源数据在 `full_device_data_ptr` 中的起始偏移
                        size_t src_start_offset = trajectory_global_offsets[t] +
                                                  static_cast<size_t>(original_view_idx_in_trajectory_t) * num_elements_per_view;

                        // 使用 Thrust 直接在设备端拷贝数据
                        thrust::copy(thrust::device_ptr<const precision_t>(full_device_data_ptr + src_start_offset),
                                     thrust::device_ptr<const precision_t>(full_device_data_ptr + src_start_offset + num_elements_per_view),
                                     thrust::device_ptr<precision_t>(current_subset_device_write_ptr + write_offset_in_subset_device_vector));

                        write_offset_in_subset_device_vector += num_elements_per_view;
                    }
                }
                // 将构建好的子集轨迹列表和设备数据创建为一个新的 Projection 对象
                // 假设 Projection 有一个构造函数可以直接接受 DeviceVector，例如：
                // Projection(const TrajectoryList &trajectory_list_, const Size &detector_size_, DeviceVector &&data_);
                // 或者你需要将 subset_value_device 转换为 HostVector 再传给现有构造函数 (不推荐)
                sub_projections[s] = Projection{subset_traj_lists[s], detector_size, std::move(subset_value_device)};
                // 使用 std::move 是为了进行移动语义，避免额外的 DeviceVector 拷贝
            }

            return sub_projections;
        }
    }
}

NMRECON_NAMESPACE_END
