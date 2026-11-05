#pragma once
#ifndef NMRECON_SCANNER_BLOCK_H_
#define NMRECON_SCANNER_BLOCK_H_

#include "common/config.h"
#include "cuda/coordinate.h"
#include "cuda/array.h"

NMRECON_NAMESPACE_BEGIN

namespace scanner {

/**
 * @brief The block is the basic unit of the detector, represented by coordinate and margin. 
 * A block is a spatial grid with coordinate (origin, size, spacing and direction), 
 * margin controls the distance between blocks.
 * 
 * The block is used for the construction of the detector, the detector is a collection of blocks.
 * The PET detectors are usually a hierarchical structure or blocks.
 * 
 * @tparam T 
 * @tparam Dimension: the dimension of the block, default is 3
 */

template <typename T=float, size_t Dimension=3,
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
class Block{

typedef nmrecon::common::Array<T, Dimension, Cont> Margin;
typedef nmrecon::common::Coordinate<T, Dimension, Cont> Coordinate;

public:

Block() = default;
/**
 * @brief Construct a new Block object
 * 
 * @param other 
 */
Block(const Block& other): coord(other.coord), margin(other.margin){}

/**
 * @brief Construct a new Block object
 * 
 * @param coord: the coordinate of the block
 * @param margin: the margin of the block
 */
Block(const Coordinate& coord_, const Margin& margin_): coord(coord_), margin(margin_){}

/**
 * @brief Get the coordinate of the block
 * 
 * @return const CoordinateType& 
 */
const Coordinate& get_coordinate() const {return coord;}

/**
 * @brief Get the margin of the block
 * 
 * @return const Margin& 
 */
const Margin& get_margin() const {return margin;}

/**
 * @brief Set the coordinate of the block
 * 
 * @param coord 
 */
void set_coordinate(const Coordinate& coord) {this->coord = coord;}

/**
 * @brief Set the margin of the block
 * 
 * @param margin 
 */
void set_margin(const Margin& margin) {this->margin = margin;}

void print() const {
  std::cout << "Block coordinate: " << std::endl;
  coord.print();
  std::cout << "Block margin: "  << std::endl;
  margin.print();
}

private:
/**
 * @brief the coordinate of the block
 * 
 */
Coordinate coord;

/**
 * @brief the margin of the block, used for the constructions of detector  
 * 
 */
Margin margin;

};


} // namespace scanner

NMRECON_NAMESPACE_END

#endif // !NMRECON_SCANNER_BLOCK_H_
