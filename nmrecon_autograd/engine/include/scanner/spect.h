#pragma once

#include "common/config.h"
#include "common/basic_type.h"
#include "io/scan_info.h"
#include "scanner/collimator.h"
#include "scanner/trajectory.h"


NMRECON_NAMESPACE_BEGIN
namespace scanner
{
    typedef nmrecon::common::Value3<precision_t> F3;
    typedef nmrecon::common::Value3<size_t> S3;
    typedef nmrecon::common::Coordinate3D Coordinate3D;
    // typedef nmrecon::scanner::CircularTrajectory Trajectory;
    typedef nmrecon::scanner::BodyContourTrajectory Trajectory;
    typedef thrust::host_vector<Trajectory> TrajectoryList;
    
  class SPECT
  {

  public:


    SPECT(const nmrecon::io::ScanSetting &scan_setting);

    /**
     * @brief Construct a new SPECT object
     *
     * @param collimator_
     * @param trajectory_list_
     */
    

    SPECT(const ParallelCollimator &collimator_, const TrajectoryList &trajectory_list_)
        : collimator(collimator_), trajectory_list(trajectory_list_) {};

  
    /**
     * @brief Get the collimator object
     *
     * @return const Collimator&
     */
    const ParallelCollimator &get_collimator() const { return collimator; }

    /**
     * @brief Get the trajectory list object
     *
     * @return const TrajectoryList&
     */
    const TrajectoryList &get_trajectory_list() const { return trajectory_list; }

    /**
     * @brief Get a trajectory object by index
     *
     * @return const CircularTrajectory&
     */
    const Trajectory get_trajectory_by_index(const size_t &index) const;

    size_t get_total_views() const;

    size_t get_total_num_rays() const;

    void print() const;

  private:
    ParallelCollimator collimator;
    TrajectoryList trajectory_list;

  }; // end SPECT

  enum class SPECT_Type
  {
    NM670,
    BHP6608,
    Infinia,
    Test1024,
    Test512Simu,
    Test256Simu,
    Test256,
    Test128Simu,
    Test150Simu
  };

  precision_t to_radians(precision_t deg);
  SPECT make_scanner(SPECT_Type type);

} // end namespace scanner
NMRECON_NAMESPACE_END
