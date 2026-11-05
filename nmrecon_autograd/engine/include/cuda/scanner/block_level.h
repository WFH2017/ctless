
#pragma once
#ifndef NMRECON_SCANNER_BLOCK_LEVEL_H_
#define NMRECON_SCANNER_BLOCK_LEVEL_H_

#include <string>
#include "common/config.h"
#include "cuda/scanner/block.h"
#include "cuda/array.h"

NMRECON_NAMESPACE_BEGIN

namespace scanner
{

/**
 * @brief BlockLevel is a container for blocks at a given level of the hierarchy.
 * For PET scanner, the detector usually has a hierarchy structure, and each level can be 
 * taken as a block. The top block contains many smaller blocks with same size ,spacing or direction.
 * This inclusion relation go through the ensemble process until the basic crystal layer level.
 * BlockLevel is a container which contains a prototype block to represent the block characteristics at a given level.
 * A BlockLevel is used in the detector to generate all blocks at any level.
 * 
 * @tparam T: data type for the blocks
 * @tparam Levels: number of levels in the hierarchy.
 * @tparam Dimension: dimension of the blocks, default is 3 for practical use.
 */

template <typename T = float, std::size_t Levels = 1, std::size_t Dimension = 3, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Container=thrust::host_vector>
class BlockLevel
{
public:
  typedef nmrecon::scanner::Block<T, Dimension, Container> Block;
  // array of blocks
  typedef nmrecon::common::Array<Block, Levels, Container> BlockArray;

  // array of strings
  typedef nmrecon::common::Array<std::string, Levels, Container> StringArray;

  /**
   * @brief Construct a new Block Level object
   *
   */
  BlockLevel() : levels(Levels), blocks(){
    for (size_t i = 0; i < Levels; i++)
    {
      level_names.set(i, "level_" + std::to_string(i));
    }
  }

  /**
   * @brief copy constructor
   *
   * @param levels
   * @param blocks
   * @param level_names
   */
  BlockLevel(const BlockLevel &other) : levels(other.levels),
                                        blocks(other.blocks), level_names(other.level_names) {}
  
  /**
   * @brief assign constructor
   *
   */
  BlockLevel &operator=(const BlockLevel &other)
  {
    levels = other.levels;
    blocks = other.blocks;
    level_names = other.level_names;
    return *this;
  }

  /**
   * @brief get the number of levels
   *
   * @return size_t
   */
  size_t get_levels() const { return levels; }

  /**
   * @brief get the block array
   *
   * @return const BlockArray&
   */
  const BlockArray &get_blocks() { return blocks; }

  /**
   * @brief get the block of a given level 
   *
   * @return const Block&
   */
  const Block& get_block(size_t level) const { 
    if (level >= levels) {
      throw std::out_of_range("level is out of range");
    } 
    return blocks[level]; 
  }

  /**
   * @brief get the block of a given level_name
   *
   * @return cosnt Block&
   */
  const Block& get_block(const std::string & name ){
    for (size_t i = 0; i < levels; i++)
    {
      if (level_names[i] == name)
      {
        return blocks[i];
      }
    }
    throw std::out_of_range("level name is not found");
  }  

  /**
   * @brief set names of the levels
   *
   * @return
   */
  void set_level_names(const StringArray &names){
    for (size_t i = 0; i < levels; i++)
    {
      level_names.set(i, names[i]);
    }
  }

  /**
   * @brief print the block level with the name and blocks
   * 
   */
  void print() const{
    std::cout << "total number of levels is " <<blocks.size() << std::endl;
    for (size_t i = 0; i < levels; i++)
    {
      std::cout << level_names[i]<<":" << std::endl;
      blocks[i].print();
    }
  }

private:
  /**
   * @brief number of levels in the hierarchy
   *
   */
  size_t levels;

  /**
   * @brief array of blocks
   *
   */
  BlockArray blocks;

  /**
   * @brief name of each level
   *
   */
  StringArray level_names;

  /**
   * @brief check if the block level is valid
   * TODO
   * @return  
   */
  bool is_valid() const
  {
    if (levels != blocks.size())
    {
      return false;
    }

    return true;
  }
}; // class BlockLevel
} // namespace scanner

NMRECON_NAMESPACE_END

#endif // !NMRECON_SCANNER_BLOCK_LEVEL_H_