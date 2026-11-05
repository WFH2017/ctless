
/*!
  \file detector.h
  \ingroup Scanner
  \brief Detector is the crystal-involved component in PET or SPECT, 
  for 
  
  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author 
*/

#pragma once
#ifndef NMRECON_SCANNER_DETECTOR_H_
#define NMRECON_SCANNER_DETECTOR_H_


#include "common/config.h"
#include "cuda/coordinate.h"
#include "cuda/array.h"
#include "cuda/scanner/block.h"
#include "cuda/scanner/block_level.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner{


template<typename T=float, std::size_t Levels = 1, std::size_t Dimension=3>
class Detector
{

public:
  typedef Block<T, Dimension> Block;
  typedef Array<T, Dimension> Direction;
  typedef BlockLevel<T, Dimension> BlockLevel;

  Detector() = default;

private:
  BlockLevel block_level;
  Direction direction;




}; //class Detector

}//namespace scanner

NMRECON_NAMESPACE_END




#endif // !NMRECON_SCANNER_DETECTOR_H_