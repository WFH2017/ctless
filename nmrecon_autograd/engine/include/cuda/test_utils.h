#pragma once
#ifndef NMRECON_TEST_UTILS_H_
#define NMRECON_TEST_UTILS_H_

#include "scanner/spect.h"
#include "common/basic_type.h"
#include "cuda/image3d.h"


typedef nmrecon::common::Image3D Image;

nmrecon::scanner::SPECT set_scanner();
nmrecon::scanner::SPECT set_single_view();

Image create_image(float value);
Image create_cylinder_phantom(void);

#endif // NMRECON_TEST_UTILS_H_