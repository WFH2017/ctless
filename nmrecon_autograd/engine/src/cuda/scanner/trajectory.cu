#include <iostream>
#include "scanner/trajectory.h"

NMRECON_NAMESPACE_BEGIN

namespace scanner
{

  AffineList CircularTrajectory::get_affine_list() const
  {
    AffineList aff_list(num_views);
    thrust::counting_iterator<size_t> index_begin(0);
    thrust::transform(thrust::device, index_begin, index_begin + num_views,
                      aff_list.begin(), affine_functor(radius, start_angle, end_angle, num_views));
    return aff_list;
  }

  thrust::host_vector<CircularTrajectory> CircularTrajectory::get_sub_trajectories(const size_t num_subsets) const
  {
    // Check if the number of sub trajectories is valid, if not, set to 1
    if (num_subsets < 1 || num_views < num_subsets)
    {
      std::cerr << "[WARNING]: Invalid number " << num_subsets << " of sub trajectories, set to 1." << std::endl;
      thrust::host_vector<CircularTrajectory> sub_trajectories(1);
      sub_trajectories[0] = *this;
      return sub_trajectories;
    }
    if (num_views % num_subsets != 0)
    {
      std::cerr << "[WARNING]: The number of views " << num_views << " is not divisible by the number of sub trajectories " << num_subsets << std::endl;
      std::cerr << "Set the number of subsets again" << std::endl;
    }

    thrust::host_vector<CircularTrajectory> sub_trajectories(num_subsets);
    size_t sub_num_views = num_views / num_subsets;
    precision_t original_step_angle = get_step_angle();
    precision_t sub_step_angle = original_step_angle * num_subsets;

    // Calculate the start and end angle for each sub trajectory
    for (size_t i = 0; i < num_subsets; i++)
    {
      precision_t sub_start_angle = start_angle + i * original_step_angle;
      precision_t sub_end_angle = sub_start_angle + sub_step_angle * sub_num_views;
      sub_trajectories[i] = CircularTrajectory(radius, sub_start_angle, sub_end_angle, sub_num_views);
    }
    return sub_trajectories;
  }

  void CircularTrajectory::print() const
  {
    printf("CircularTrajectory: radius=%f, start_angle=%f, end_angle=%f, num_views=%zu, step=%f\n",
           radius, start_angle, end_angle, num_views, (end_angle - start_angle) / num_views);
  }

  thrust::device_vector<TrajectoryView> BodyContourTrajectory::get_trajectory_views() const
  {
    // 输入验证：检查 radius_vector 的大小是否与 num_views 匹配
    if (radius_vector.size() != num_views)
    {
      std::cerr << "[ERROR]: radius_vector size (" << radius_vector.size()
                << ") does not match num_views (" << num_views << ")." << std::endl;
      throw std::runtime_error("radius_vector size mismatch in get_trajectory_views.");
    }

    // 1. 创建一个临时的 thrust::host_vector 来在主机端准备数据
    thrust::host_vector<TrajectoryView> host_trajectory_views(num_views);

    // 2. 遍历每个视图，计算对应的角度和半径，并填充 host_trajectory_views
    for (size_t i = 0; i < num_views; ++i)
    {
      // 计算当前视图的角度。
      // 这部分逻辑与 affine_functor::operator() 中的角度计算相似。
      precision_t current_view_angle = start_angle;
      if (num_views > 1) // 避免当只有一个视图时除以零
      {
        //计算当前投影角度
        current_view_angle = start_angle + i * get_step_angle();
      }

      // 从主机端的 radius_vector 中获取对应的半径值
      precision_t current_radius = radius_vector[i];

      // 创建一个 TrajectoryView 对象，并将其赋值给 host_trajectory_views 的当前位置
      host_trajectory_views[i] = TrajectoryView(current_view_angle, current_radius);
    }

    // 3. 将主机端数据一次性拷贝到设备端
    // Thrust 会自动处理从 thrust::host_vector 到 thrust::device_vector 的高效拷贝
    thrust::device_vector<TrajectoryView> device_trajectory_views = host_trajectory_views;

    return device_trajectory_views;
  }

  thrust::host_vector<BodyContourTrajectory> BodyContourTrajectory::get_sub_trajectories(const size_t num_subsets) const
  {
    // --- 输入验证 ---
    // 检查 num_subsets <= 1 或 num_views < num_subsets 的情况
    // 你的原代码中已经处理了，这里保持一致
    if (num_subsets < 1 || num_views < num_subsets)
    {
      std::cerr << "[WARNING]: Invalid number " << num_subsets << " of sub trajectories. "
                << "Expected >= 1 and <= num_views (" << num_views << "). Returning full trajectory as single sub-trajectory." << std::endl;
      return thrust::host_vector<BodyContourTrajectory>(1, *this);
    }

    // 检查 num_views 是否能被 num_subsets 整除
    if (num_views % num_subsets != 0)
    {
      std::cerr << "[WARNING]: The number of views " << num_views << " is not divisible by the number of sub trajectories " << num_subsets << std::endl;
      std::cerr << "This may lead to uneven subset sizes or data distribution for radius_vector." << std::endl;
      // 注意：这里没有 std::exit(1)，而是继续执行，可能会导致最后一个子集的视图数量不同
      // 或者你可以选择抛出异常：throw std::invalid_argument("...");
    }

    thrust::host_vector<BodyContourTrajectory> sub_trajectories(num_subsets);
    // 每个子轨迹包含的视图数量
    size_t sub_num_views = num_views / num_subsets;
    // 如果 num_views 不能被 num_subsets 整除，某些子集的视图数量会少一个或多一个。
    // 为了简化，这里我们假设它能整除。如果不能整除，需要更复杂的逻辑来分配视图。

    // 计算总步长（整个圆周扫描的步长）
    // 假设 get_step_angle() 返回的是每个视图之间的角度步长
    precision_t global_step_angle = get_step_angle() * num_subsets; // 单个视图之间的角度步长

    // 遍历每个子集
    for (size_t i = 0; i < num_subsets; i++)
    {
      // 1. 角度计算：保持不变，因为角度是连续的
      // 子轨迹的起始角度是根据子集的索引 `i` 和全局步长 `global_step_angle` 来确定的
      // 这是该子集所包含的第一个视图的起始角度
      precision_t sub_start_angle = start_angle + i * get_step_angle();
      // 子轨迹的结束角度是该子集所包含的最后一个视图的结束角度
      precision_t sub_end_angle = sub_start_angle + global_step_angle * sub_num_views;

      // 2. radius_vector 的子集化：按照每隔 num_subsets 个元素取一个
      std::vector<precision_t> sub_radius_vector;
      sub_radius_vector.reserve(sub_num_views); // 预留空间以避免频繁重新分配

      // 选取当前子集对应的 radius 值
      // sub_idx 是在原始 radius_vector 中的索引
      for (size_t v_in_subset = 0; v_in_subset < sub_num_views; ++v_in_subset)
      {
        size_t original_radius_idx = i + v_in_subset * num_subsets;
        if (original_radius_idx < num_views)
        { // 确保不越界
          sub_radius_vector.push_back(radius_vector[original_radius_idx]);
        }
        else
        {
          // 如果出现这种情况，表示 num_views / num_subsets 计算有问题或者 num_views 不被整除
          // 或者 radius_vector 实际大小与 num_views 不符
          std::cerr << "[ERROR]: Index out of bounds when creating sub_radius_vector. "
                    << "original_radius_idx=" << original_radius_idx << ", num_views=" << num_views << std::endl;
          throw std::runtime_error("Radius vector indexing error in get_sub_trajectories.");
        }
      }

      // 3. 构建 BodyContourTrajectory 对象
      sub_trajectories[i] = BodyContourTrajectory(sub_radius_vector, sub_start_angle, sub_end_angle, sub_num_views);
    }

    return sub_trajectories;
  }

  AffineList BodyContourTrajectory::get_affine_list() const
  {
    auto trajectory_views = get_trajectory_views();
    AffineList aff_list(trajectory_views.size());
    thrust::transform(thrust::device, trajectory_views.begin(), trajectory_views.end(),
                      aff_list.begin(), affine_view_functor());
    return aff_list;
  }

  void BodyContourTrajectory::print() const
  {
    printf("BodyContourTrajectory: start_angle=%f, end_angle=%f, num_views=%zu\n",
           start_angle, end_angle, num_views);
    printf("Radius vector: ");
    for (const auto &radius : radius_vector)
    {
      printf("%f ", radius);
    }
    printf("\n");
  }

}

NMRECON_NAMESPACE_END
