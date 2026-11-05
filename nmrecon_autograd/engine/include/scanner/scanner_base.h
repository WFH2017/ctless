#pragma once
#ifndef NMRECON_SCANNER_BASE_H_
#define NMRECON_SCANNER_BASE_H_

#include "nmrecon/common/config.h"

NMRECON_NAMESPACE_BEGIN

namespace scanner
{
/**
 * @brief Basic abtract scanner class to represent an nuclear medical scanner.
 * The ScannerBase can be derived to implement PET and SPECT scanners.
 */
class ScannerBase
{
public:
  static ScannerBase *as_parameters();

  enum Type
  {
    User_defined_scanner,
    Unknown_scanner
  };

  virtual ~ScannerBase() {}
  ScannerBase(Type scanner_type);



private:



  }; // end ScannerBase
} // end namespace scanner

NMRECON_NAMESPACE_END
#endif // NMRECON_SCANNER_BASE_H_
