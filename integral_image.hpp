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

// In general, non-SIMD integralPixmap is much faster than the SIMDed.

//#if !defined(USE_SIMD)
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
  
  size_t bytes4integral = ALIGN_BYTES((integral.getWidth() + 1) * sizeof(integral_t));
  
#pragma omp parallel for
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    uint8_t *temp1Line = new uint8_t[bytes4integral];
    std::memset(temp1Line, 0, bytes4integral);
    integral_t *prevLine = (integral_t *)temp1Line + 1;
    
    uint8_t *temp2Line = new uint8_t[bytes4integral];
    std::memset(temp2Line, 0, bytes4integral);
    integral_t *currLine = (integral_t *)temp2Line + 1;
    
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      pixel_t *pixLine = pixmap.getLine(y, z);
      //integral_t *intLine = integral.getLine(y, z);
      //integral.readHLine(currLine, integral.getWidth(), 0, y, z);
      for (size_t x = 0; x < pixmap.getWidth(); ++x) {
	currLine[(int)x] = prevLine[(int)x] - prevLine[(int)x-1] + currLine[(int)x-1] + (integral_t)pixLine[x];
      }
      integral.writeHLine(currLine, integral.getWidth(), 0, y, z);
      std::swap(prevLine, currLine);
    }
    delete [] temp1Line;
    delete [] temp2Line;
  }
}
/*
#else
# include "integral_image.slow.SIMD.hpp"
#endif
*/
