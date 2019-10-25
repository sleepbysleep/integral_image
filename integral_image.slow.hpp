/*
  Copyright (C) 2017 Hoyoung Lee

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <cassert>
#include <limits>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <type_traits>

#include <cpixmap.hpp>
#include <power_of_2.hpp>

#if !defined(USE_SIMD)

template <typename pixel_t, typename integral_t>
inline void integratePixmap(cpixmap<pixel_t>& pixmap, cpixmap<integral_t>& integral)
{
  assert(!(std::numeric_limits<pixel_t>::is_signed ^ std::numeric_limits<integral_t>::is_signed));
  assert(pixmap.isMatched(integral));

  if (std::numeric_limits<integral_t>::digits <
      (std::numeric_limits<pixel_t>::digits +
       ilog2(ceilPowerOf2((uint32_t)pixmap.getWidth())) +
       ilog2(ceilPowerOf2((uint32_t)pixmap.getHeight())))) {
    std::cout << "Warning!: Integral pixmap doesn't fully contain the result from image pixmap!" << std::endl;
  }

  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(integral_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    integral_t *sumLine = (integral_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      pixel_t *pixLine = pixmap.getLine(y, z);
      integral_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
      for (size_t x = 0; x < pixmap.getWidth(); ++x) {
	intLine[x] = sumLine[x] + (integral_t)pixLine[x];
      }
      sumLine = intLine;
    }
  }

  // Horizontally cumulative summation
  integral_t *intLine = (integral_t *)new uint8_t[bytes4integral];
  for (size_t z = 0; z < integral.getBands(); ++z) {
    integral_t *sumLine = (integral_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
      for (size_t y = 0; y < integral.getHeight(); ++y) {
	sumLine[y] = sumLine[y] + intLine[y];
      }
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }

  delete [] intLine;
  delete [] tempLine;
}

#else
# include "integral_image.slow.SIMD.hpp"
#endif

