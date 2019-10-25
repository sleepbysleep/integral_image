/*
  Copyright (C) 2014 Hoyoung Lee

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

#include <iostream>
#include <sstream>
#include <cassert>
#include <limits>

inline size_t countLeadingZeros(uint8_t x)
{
  //assert(std::numeric_limits<unsigned long>::digits == 32);
  return __builtin_clzl((unsigned long)x) - (std::numeric_limits<unsigned long>::digits-8);
}

inline size_t countLeadingZeros(uint16_t x)
{
  //assert(std::numeric_limits<unsigned long>::digits == 32);
  return __builtin_clzl((unsigned long)x) - (std::numeric_limits<unsigned long>::digits-16);
}

inline size_t countLeadingZeros(uint32_t x)
{
  //assert(std::numeric_limits<unsigned long>::digits == 32);
  return __builtin_clzl((unsigned long)x) - (std::numeric_limits<unsigned long>::digits-32);
}

inline size_t countLeadingZeros(uint64_t x)
{
  //assert(std::numeric_limits<unsigned long long>::digits == 64);
  return __builtin_clzll((unsigned long long)x) - (std::numeric_limits<unsigned long long>::digits-64);
}

inline size_t countTrailingZeros(uint8_t x)
{
  //assert(std::numeric_limits<unsigned long>::digits == 32);
  return __builtin_ctzl((unsigned long)x);
}

inline size_t countTrailingZeros(uint16_t x)
{
  //assert(std::numeric_limits<unsigned long long>::digits == 64);
  return __builtin_ctzl((unsigned long)x);
}

inline size_t countTrailingZeros(uint32_t x)
{
  //assert(std::numeric_limits<unsigned long>::digits == 32);
  return __builtin_ctzl((unsigned long)x);
}

inline size_t countTrailingZeros(uint64_t x)
{
  //assert(std::numeric_limits<unsigned long long>::digits == 64);
  return __builtin_ctzll((unsigned long long)x);
}

inline size_t ilog2(uint8_t x)
{
  return ((std::numeric_limits<unsigned long>::digits - 1) - __builtin_clzl((unsigned long)x));
}

inline size_t ilog2(uint16_t x)
{
  return ((std::numeric_limits<unsigned long>::digits - 1) - __builtin_clzl((unsigned long)x));
}

inline size_t ilog2(uint32_t x)
{
  return ((std::numeric_limits<unsigned long>::digits - 1) - __builtin_clzl((unsigned long)x));
}

inline size_t ilog2(uint64_t x)
{
  return ((std::numeric_limits<unsigned long long>::digits - 1) - __builtin_clzll((unsigned long long)x));
}

// Reference: my.safaribooksonline.com/book/information-technology-and-software-development/0201914654/power-of-2-boundaries
// Hacker's Delight - Henry S. Warren, Jr.
inline uint8_t floorPowerOf2(uint8_t x) // aka flp2()
{
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);

  return x - (x>>1);
}

inline uint16_t floorPowerOf2(uint16_t x) // aka flp2()
{
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);
  
  return x - (x>>1);
}

inline uint32_t floorPowerOf2(uint32_t x) // aka flp2()
{
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);
  x = x | (x>>16);
  
  return x - (x>>1);
}

inline uint64_t floorPowerOf2(uint64_t x) // aka flp2()
{
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);
  x = x | (x>>16);
  x = x | (x>>32);
  
  return x - (x>>1);
}

inline uint8_t ceilPowerOf2(uint8_t x) // aka clp2
{
  x = x - 1;
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);

  return x+1;
}

inline uint16_t ceilPowerOf2(uint16_t x) // aka clp2
{
  x = x - 1;
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);

  return x+1;
}

inline uint32_t ceilPowerOf2(uint32_t x) // aka clp2
{
  x = x - 1;
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);
  x = x | (x>>16);

  return x+1;
}

inline uint64_t ceilPowerOf2(uint64_t x) // aka clp2
{
  x = x - 1;
  x = x | (x>>1);
  x = x | (x>>2);
  x = x | (x>>4);
  x = x | (x>>8);
  x = x | (x>>16);
  x = x | (x>>32);

  return x+1;
}

/*
template <typename T>
inline size_t countLeadingZeros(T x)
{
  assert(std::numeric_limits<T>::is_integer);
  assert(!std::numeric_limits<T>::is_signed);

  if (x == 0) return std::numeric_limits<T>::digits;

  const T bitmask = 1 << (std::numeric_limits<T>::digits - 1);
  size_t count;
  
  for (count = 0; (x & bitmask) == 0; ++count, x>>=1);
  
  return count;
}

template <typename T>
inline size_t countTrailingZeros(T x)
{
  assert(std::numeric_limits<T>::is_integer);
  assert(!std::numeric_limits<T>::is_signed);

  if (x == 0) return std::numeric_limits<T>::digits;

  const T bitmask = 1;
  size_t count;
  
  for (count = 0; (x & bitmask) == 0; ++count, x<<=1);
  
  return count;

}

template <typename T>
inline T floorPowerOf2(T x)
{
  assert(std::numeric_limits<T>::is_integer);
  assert(!std::numeric_limits<T>::is_signed);

  T y = 1 << (std::numeric_limits<T>::digits - 1);
  while (y > x) y >>= 1;

  return y;
}

template <typename T>
inline T ceilPowerOf2(T x)
{
  assert(std::numeric_limits<T>::is_integer);
  assert(!std::numeric_limits<T>::is_signed);

  T y = 1;
  while (y < x) y <<= 1;

  return y;
}
*/
