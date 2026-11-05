
/*!
  \file collimator.h
  \ingroup Scanner
  \brief Defines the commonly used collimators in SPECT.

  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author
*/

#pragma once

#include <string>
#include "common/config.h"
#include "common/basic_type.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner
{
  using namespace nmrecon::common;

  enum CollimatorType
  {
    Parallel,
    Pinhole,
    Fanbeam
  };

  /**
   * @brief basic Collimator class for the SPECT collimator components
   *
   */
  class Collimator
  {
  public:
    // const std::string &get_name() { return name; };
    // const CollimatorType &get_type() const { return collimator_type; };

  private:
    /**
     * @brief the name to specify the collimator
     *
     */
    // std::string name;               // name of the collimator

    // CollimatorType collimator_type; // enum type to specify the collimator (0: Parallel; 1: Pinhole; 2: Fanbeam)
    // const int num;
  }; // end Collimator

  struct hexagon_lookup_table_functor
  {

    const Origin origin;
    const Size size;
    const Spacing spacing;
    const Direction direction;

    hexagon_lookup_table_functor(
        const Origin &origin_,
        const Size &size_,
        const Spacing &spacing_,
        const Direction &direction_)
        : origin(origin_), size(size_), spacing(spacing_), direction(direction_) {}
    __device__ Vec3 operator()(const size_t &index) const;

  }; // end hexagon_lookup_table_functor

  class ParallelCollimator : public Collimator
  {

  public:
    ParallelCollimator():
      collimator_length(0.0), coordinate_prototype(Coordinate3D()) {}
    ParallelCollimator(const precision_t &collimator_length_,
                       const Coordinate3D &coord_)
        : collimator_length(collimator_length_), coordinate_prototype(coord_) {}

    ParallelCollimator& operator=(const ParallelCollimator &other)
    {
      if (this != &other)
      {
        this->collimator_length = other.collimator_length;
        this->coordinate_prototype = other.coordinate_prototype;
      }
      return *this;
    }

    const precision_t &get_collimator_length() const { return collimator_length; };
    const Coordinate3D &get_coordinate_prototype() const { return coordinate_prototype; };

    __host__ Point3DVector make_lookup_table() const;

    /**
     * @brief Get the num of rays of the collimator.
     *
     * @return size_t
     */
    size_t get_num_rays() const { return coordinate_prototype.size.product(); }

    void print() const
    {
      std::cout << "collimator length: " << collimator_length << std::endl;
      std::cout << "collimator coordinate prototype: " << std::endl;
      coordinate_prototype.print();
    }

  private:
    precision_t collimator_length; // the length of the collimator
    /**
     * @brief the coordinate prototype of the collimator, the origin is set to zero point and direction is identity matrix.
     *
     */
    Coordinate3D coordinate_prototype;

  }; // end ParallelCollimator

  class PinholeCollimator : public Collimator
  {
    // TODO: develop the pinhole collimator class
  public:
  private:
  }; // end PinholdCollimator

  class FanbeamCollimator : public Collimator
  {
  public:
    // TODO: setter and getter functions.
    FanbeamCollimator(
        const precision_t &focal_length_,
        const precision_t &collimator_to_detector_distance_,
        const precision_t &linear_factor_horizontal_,
        const precision_t &linear_factor_vertical_,
        const precision_t &thickness_,
        const precision_t &instrinsic_sigma_)
        : focal_length(focal_length_),
          collimator_to_detector_distance(collimator_to_detector_distance_),
          linear_factor_horizontal(linear_factor_horizontal_),
          linear_factor_vertical(linear_factor_vertical_),
          thickness(thickness_),
          instrinsic_sigma(instrinsic_sigma_) {};

  private:
    /**
     * @brief the focal length of the fanbeam collimator
     *
     */
    const precision_t focal_length;
    /**
     * @brief the collimator to detector distance(?)
     *
     */
    const precision_t collimator_to_detector_distance;

    /**
     * @brief linear factor for dependency of sigma on distance (fanbeam horizontal)
     *
     */
    const precision_t linear_factor_horizontal;

    /**
     * @brief linear factor for dependency of sigma on distance (fanbeam vertical)
     *
     */
    const precision_t linear_factor_vertical; //

    /**
     * @brief collimator
     *
     */
    const precision_t thickness; // collimator thickness?? (fanbeam)

    /**
     * @brief intrinsic sigma (cristal resolution) (fanbeam)
     *
     */
    const precision_t instrinsic_sigma; // intrinsic sigma (cristal resolution) (fanbeam)

  }; // end FanbeamCollimator

} // end namespace scanner
NMRECON_NAMESPACE_END
