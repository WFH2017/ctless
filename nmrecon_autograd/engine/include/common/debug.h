/*
Copyright (C) 2023, Vplus Technology Co., Ltd.

This file is part of NMRecon.

See NMRecon/LICENSE.txt for details.
*/

/*!
  \file debug.h
  \ingroup Config
  \brief Defines debug macros
  
  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author 
*/


#pragma once

#ifndef NMRECON_DEBUG
#  ifndef NDEBUG
#    if defined(DEBUG) || defined(_DEBUG)
#      define NMRECON_DEBUG 1
#    endif // (DEBUG || _DEBUG)
#  endif // NDEBUG
#endif // NMRECON_DEBUG

#if NMRECON_DEBUG
#  ifndef __NMRECON_SYNCHRONOUS
#    define __NMRECON_SYNCHRONOUS 1
#  endif // __NMRECON_SYNCHRONOUS
#endif // NMRECON_DEBUG
