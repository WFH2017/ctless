
/*!
  \file
  \ingroup
  \brief

  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author
*/

#pragma once
#include "common/config.h"
#include "common/basic_type.h"
#include "scanner/spect.h"

NMRECON_NAMESPACE_BEGIN

namespace reconstruction
{
  namespace rawdata
  {
    using namespace nmrecon::scanner;
    using namespace nmrecon::common;

    /// @brief DeviceProjection主要提供了一个封装，将Projection的数据裸指针，并复制其他相应信息，
    /// 便于GPU核函数中的计算。
    struct DeviceProjection
    {
      TrajectoryList trajectory_list;
      Size detector_size;
      // DeviceVector projection_value;
      precision_t *projection_value;
    };

    /// @brief Projection represent the projection data on the scanner.
    /// It contains the trajectory list, the size of the detector, and corresponding projection value.
    /// The projection value is a 1D array, which is indexed by the [detector size, trajectory views, trajectory list].
    class Projection
    {
    public:
      Projection();
      Projection(const Projection &projection);
      Projection(const TrajectoryList &trajectory_list_, const Size &detector_size_);
      Projection(const TrajectoryList &trajectory_list_, const Size &detector_size_, const DeviceVector &data_);
      Projection &operator=(const Projection &projection);

      /// @brief
      /// @return
      DeviceProjection to_device();

      void set_data(const DeviceVector &data_);

      DeviceVector get_data() const;
      DeviceVector &get_data_ref();
      TrajectoryList get_trajectory_list() const;
      Size get_detector_size() const;

      Projection operator+(const Projection &projection) const;
      Projection operator+(const precision_t &value) const;
      Projection operator-(const Projection &projection) const;
      Projection operator-(const precision_t &value) const;
      Projection operator*(const Projection &projection) const;
      Projection operator*(const precision_t &value) const;
      Projection operator/(const Projection &projection) const;
      Projection operator/(const precision_t &value) const;

      void fill(const precision_t &value);

      // reorganize the projection to subsets and move the data to the device.
      thrust::host_vector<Projection> get_sub_projections_old(const size_t num_subsets) const;
      thrust::host_vector<Projection> get_sub_projections(const size_t num_subsets) const;

    private:
      TrajectoryList trajectory_list;
      Size detector_size;
      DeviceVector projection_value;
    };
  } // end namespace rawdata
} // end namespace reconstruction

NMRECON_NAMESPACE_END
