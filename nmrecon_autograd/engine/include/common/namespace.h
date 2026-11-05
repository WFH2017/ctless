/*
Copyright (C) 2023, Vplus Medical Technology Co., Ltd.
This file is part of NMRecon.
See NMRecon/LICENSE.txt for details.
*/

#pragma once

/*!
  \file namespace.h
  \ingroup Config
  \brief Utilities that allow `nmrecon::` to be placed inside an
 * application-specific namespace.

  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author
*/

/**
 * \def NMRECON_NAMESPACE_BEGIN
 * This macro is used to open a `nmrecon::` namespace block,
 * This macro is defined by NMRecon and may not be overridden.
 */
#define NMRECON_NAMESPACE_BEGIN \
  namespace nmrecon             \
  {

/**
 * \def NMRECON_NAMESPACE_END
 * This macro is used to close a `nmrecon::` namespace block,
 * This macro is defined by NMRecon and may not be overridden.
 */
#define NMRECON_NAMESPACE_END \
  } /* end namespace nmrecon */
