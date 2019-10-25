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
#include <iostream>
#include <cstring>
#include <cstdint>

#include <cpixmap.hpp>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
# define MAX_VECTOR_SIZE 512
# include <vectorclass/vectorclass.h>
# if INSTRSET < 2
#  error "Unsupported x86-SIMD! Please comment USE_SIMD on!"
# endif
#elif defined(__GNUC__) && defined (__ARM_NEON__)
# include <arm_neon.h>
#else
# error "Undefined SIMD!"
#endif

inline void integratePixmap(cpixmap<int8_t>& pixmap, cpixmap<int32_t>& integral)
{
  assert(pixmap.isMatched(integral));
  
  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(int32_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    int32_t *sumLine = (int32_t *)tempLine;

    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      int8_t *pixLine = pixmap.getLine(y, z);
      int32_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 16) {
	Vec16i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		      pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7],
		      pixLine[x+8], pixLine[x+9], pixLine[x+10], pixLine[x+11],
		      pixLine[x+12], pixLine[x+13], pixLine[x+14], pixLine[x+15]);
	Vec16i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		     pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	int8x8_t loadVec = vld1_s8((const int8_t *)(&pixLine[x]));
	int16x8_t tempVec = vmovl_s8(loadVec);
	// Lower 4 elements
	int16x4_t pixVec = vget_low_s16(tempVec);
	int32x4_t sumVec = vld1q_s32((const int32_t *)(&sumLine[x]));
	sumVec = vaddw_s16(sumVec, pixVec);
	vst1q_s32((int32_t *)(&intLine[x]), sumVec);
	// Higher 4 elements
	pixVec = vget_high_s16(tempVec);
	sumVec = vld1q_s32((const int32_t *)(&sumLine[x+4]));
	sumVec = vaddw_s16(sumVec, pixVec);
	vst1q_s32((int32_t *)(&intLine[x+4]), sumVec);
      }
#endif
      sumLine = intLine;
    }
  }

  // Horizontally cumulative summation
  int32_t *intLine = (int32_t *)new uint8_t[bytes4integral];
  for (size_t z = 0; z < integral.getBands(); ++z) {
    int32_t *sumLine = (int32_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);

    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 16) {
	Vec16i pixVec;
	pixVec.load(&intLine[y]);
	Vec16i sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8i pixVec;
	pixVec.load(&intLine[y]);
	Vec8i sumVec;      
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4i pixVec;
	pixVec.load(&intLine[y]);
	Vec4i sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < pixmap.getHeight(); y += 4) {
	int32x4_t pixVec = vld1q_s32((const int32_t *)&intLine[y]);
	int32x4_t sumVec = vld1q_s32((const int32_t *)&sumLine[y]);
	sumVec = vaddq_s32(sumVec, pixVec);
	vst1q_s32((int32_t *)&sumLine[y], sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }

  delete [] intLine;
  delete [] tempLine;
}

inline void integratePixmap(cpixmap<uint8_t>& pixmap, cpixmap<uint32_t>& integral)
{
  assert(pixmap.isMatched(integral));
  
  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(uint32_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    uint32_t *sumLine = (uint32_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      uint8_t *pixLine = pixmap.getLine(y, z);
      uint32_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 16) {
	Vec16ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		       pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7],
		       pixLine[x+8], pixLine[x+9], pixLine[x+10], pixLine[x+11],
		       pixLine[x+12], pixLine[x+13], pixLine[x+14], pixLine[x+15]);
	Vec16ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		      pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	uint8x8_t loadVec = vld1_u8((const uint8_t *)&pixLine[x]);
	uint16x8_t tempVec = vmovl_u8(loadVec);
	// Lower 4 elements
	uint16x4_t pixVec = vget_low_u16(tempVec);
	uint32x4_t sumVec = vld1q_u32((const uint32_t *)&sumLine[x]);
	sumVec = vaddw_u16(sumVec, pixVec);
	vst1q_u32((uint32_t *)&intLine[x], sumVec);
	// Higher 4 elements
	pixVec = vget_high_u16(tempVec);
	sumVec = vld1q_u32((const uint32_t *)&sumLine[x+4]);
	sumVec = vaddw_u16(sumVec, pixVec);
	vst1q_u32((uint32_t *)&intLine[x+4], sumVec);
      }
#endif
      sumLine = intLine;
    }
  }
  
  // Horizontally cumulative summation
  uint32_t *intLine = (uint32_t *)new uint8_t[bytes4integral];

  for (size_t z = 0; z < integral.getBands(); ++z) {
    uint32_t *sumLine = (uint32_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 16) {
	Vec16ui pixVec;
	pixVec.load(&intLine[y]);
	Vec16ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8ui pixVec;
	pixVec.load(&intLine[y]);
	Vec8ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4ui pixVec;
	pixVec.load(&intLine[y]);
	Vec4ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < pixmap.getHeight(); y += 4) {
	uint32x4_t pixVec = vld1q_u32((const uint16_t *)(&intLine[y]));
	uint32x4_t sumVec = vld1q_u32((const uint32_t *)(&sumLine[y]));
	sumVec = vaddq_u32(sumVec, pixVec);
	vst1q_u32((uint32_t *)(&sumLine[y]), sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }

  delete [] intLine;
  delete [] tempLine;
}

inline void integratePixmap(cpixmap<int16_t>& pixmap, cpixmap<int32_t>& integral)
{
  assert(pixmap.isMatched(integral));
  
  std::cout << "Warning!: Integral pixmap doesn't fully contain the result from image pixmap!" << std::endl;

  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(int32_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    int32_t *sumLine = (int32_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      int16_t *pixLine = pixmap.getLine(y, z);
      int32_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 16) {
	Vec16i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		      pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7],
		      pixLine[x+8], pixLine[x+9], pixLine[x+10], pixLine[x+11],
		      pixLine[x+12], pixLine[x+13], pixLine[x+14], pixLine[x+15]);
	Vec16i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		     pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4i pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4i sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	int16x4_t pixVec = vld1_s16((const int16_t *)&pixLine[x]);
	int32x4_t sumVec = vld1q_s32((const int32_t *)&sumLine[x]);
	sumVec = vaddw_s16(sumVec, pixVec);
	vst1q_s32((int32_t *)&intLine[x], sumVec);
      }
#endif
      sumLine = intLine;
    }
  }

  // Horizontally cumulative summation
  int32_t *intLine = (int32_t *)new uint8_t[bytes4integral];

  for (size_t z = 0; z < integral.getBands(); ++z) {
    int32_t *sumLine = (int32_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 16) {
	Vec16i pixVec;
	pixVec.load(&intLine[y]);
	Vec16i sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8i pixVec;
	pixVec.load(&intLine[y]);
	Vec8i sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4i pixVec;
	pixVec.load(&intLine[y]);
	Vec4i sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < pixmap.getHeight(); y += 4) {
	int32x4_t pixVec = vld1q_s32((const int32_t *)&intLine[y]);
	int32x4_t sumVec = vld1q_s32((const int32_t *)&sumLine[y]);
	sumVec = vaddq_s32(sumVec, pixVec);
	vst1q_s32((int32_t *)&sumLine[y], sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }

  delete [] intLine;
  delete [] tempLine;
}

inline void integratePixmap(cpixmap<uint16_t>& pixmap, cpixmap<uint32_t>& integral)
{
  assert(pixmap.isMatched(integral));

  std::cout << "Warning!: Integral pixmap doesn't fully contain the result from image pixmap!" << std::endl;

  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(uint32_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    uint32_t *sumLine = (uint32_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      uint16_t *pixLine = pixmap.getLine(y, z);
      uint32_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 16) {
	Vec16ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		       pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7],
		       pixLine[x+8], pixLine[x+9], pixLine[x+10], pixLine[x+11],
		       pixLine[x+12], pixLine[x+13], pixLine[x+14], pixLine[x+15]);
	Vec16ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		      pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
    
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4ui pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4ui sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	uint16x4_t pixVec = vld1_u16((const uint16_t *)(&pixLine[x]));
	uint32x4_t sumVec = vld1q_u32((const uint32_t *)(&sumLine[x]));
	sumVec = vaddw_u16(sumVec, pixVec);
	vst1q_u32((uint32_t *)(&intLine[x]), sumVec);
      }
#endif
      sumLine = intLine;
    }
  }

  // Horizontally cumulative summation
  uint32_t *intLine = (uint32_t *)new uint8_t[bytes4integral];

  for (size_t z = 0; z < integral.getBands(); ++z) {
    uint32_t *sumLine = (uint32_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 16) {
	Vec16ui pixVec;
	pixVec.load(&intLine[y]);
	Vec16ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8ui pixVec;
	pixVec.load(&intLine[y]);
	Vec8ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4ui pixVec;
	pixVec.load(&intLine[y]);
	Vec4ui sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	uint16x4_t pixVec = vld1q_u16((const uint32_t *)&intLine[y]);
	uint32x4_t sumVec = vld1q_u32((const uint32_t *)&sumLine[y]);
	sumVec = vaddq_u32(sumVec, pixVec);
	vst1q_u32((uint32_t *)&sumLine[y], sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }
  
  delete [] intLine;
  delete [] tempLine;
}

inline void integratePixmap(cpixmap<int16_t>& pixmap, cpixmap<int64_t>& integral)
{
  assert(pixmap.isMatched(integral));

  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(int64_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation  
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    int64_t *sumLine = (int64_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      int16_t *pixLine = pixmap.getLine(y, z);
      int64_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8q pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		     pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8q sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4q pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4q sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 2) {
	Vec2q pixVec(pixLine[x+0], pixLine[x+1]);
	Vec2q sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	int16x4_t loadVec = vld1_s16((const int16_t *)&pixLine[x]);
	int32x4_t tempVec = vmovl_s16(loadVec);
	// Lower 2 elements
	int32x2_t pixVec = vget_low_s32(tempVec);
	int64x2_t sumVec = vld1q_s64((const int64_t *)&sumLine[x]);
	sumVec = vaddw_s32(sumVec, pixVec);
	vst1q_s64((int64_t *)&intLine[x], sumVec);
	// Higher 2 elements
	pixVec = vget_high_s32(tempVec);
	sumVec = vld1q_s64((const int64_t *)&sumLine[x+2]);
	sumVec = vaddw_s32(sumVec, pixVec);
	vst1q_s64((int64_t *)&intLine[x+2], sumVec);
      }
#endif
      sumLine = intLine;
    }
  }

  // Horizontally cumulative summation
  int64_t *intLine = (int64_t *)new uint8_t[bytes4integral];

  for (size_t z = 0; z < integral.getBands(); ++z) {
    int64_t *sumLine = (int64_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8q pixVec;
	pixVec.load(&intLine[y]);
	Vec8q sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4q pixVec;
	pixVec.load(&intLine[y]);
	Vec4q sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 2) {
	Vec2q pixVec;
	pixVec.load(&intLine[y]);
	Vec2q sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < pixmap.getHeight(); y += 2) {
	int64x2_t pixVec = vld1q_s64((const int64_t *)&intLine[y]);
	int64x2_t sumVec = vld1q_s64((const int64_t *)&sumLine[y]);
	sumVec = vaddq_s64(sumVec, pixVec);
	vst1q_s64((int64_t *)&sumLine[y], sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }

  delete [] intLine;
  delete [] tempLine;
}

inline void integratePixmap(cpixmap<uint16_t>& pixmap, cpixmap<uint64_t>& integral)
{
  assert(pixmap.isMatched(integral));

  size_t bytes4integral = ALIGN_BYTES(std::max(pixmap.getWidth(), pixmap.getHeight()) * sizeof(uint64_t));
  uint8_t *tempLine = new uint8_t[bytes4integral];
  
  // Vertically cumulative summation  
  for (size_t z = 0; z < pixmap.getBands(); ++z) {
    std::memset(tempLine, 0, bytes4integral);
    uint64_t *sumLine = (uint64_t *)tempLine;
    for (size_t y = 0; y < pixmap.getHeight(); ++y) {
      uint16_t *pixLine = pixmap.getLine(y, z);
      uint64_t *intLine = integral.getLine(y, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 8) {
	Vec8uq pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3],
		      pixLine[x+4], pixLine[x+5], pixLine[x+6], pixLine[x+7]);
	Vec8uq sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	Vec4uq pixVec(pixLine[x+0], pixLine[x+1], pixLine[x+2], pixLine[x+3]);
	Vec4uq sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t x = 0; x < pixmap.getWidth(); x += 2) {
	Vec2uq pixVec(pixLine[x+0], pixLine[x+1]);
	Vec2uq sumVec;
	sumVec.load(&sumLine[x]);
	sumVec += pixVec;
	sumVec.store(&intLine[x]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t x = 0; x < pixmap.getWidth(); x += 4) {
	uint16x4_t loadVec = vld1_u16((const uint16_t *)&pixLine[x]);
	uint32x4_t tempVec = vmovl_u16(loadVec);
	// Lower 2 elements
	uint32x2_t pixVec = vget_low_u32(tempVec);
	uint64x2_t sumVec = vld1q_u64((const uint64_t *)&sumLine[x]);
	sumVec = vaddw_u32(sumVec, pixVec);
	vst1q_u64((uint64_t *)&intLine[x], sumVec);
	// Higher 2 elements
	pixVec = vget_high_u32(tempVec);
	sumVec = vld1q_u64((const uint64_t *)&sumLine[x+2]);
	sumVec = vaddw_u32(sumVec, pixVec);
	vst1q_u64((uint64_t *)&intLine[x+2], sumVec);
      }
#endif
      sumLine = intLine;
    }
  }
  // Horizontally cumulative summation
  uint64_t *intLine = (uint64_t *)new uint8_t[bytes4integral];

  for (size_t z = 0; z < integral.getBands(); ++z) {
    uint64_t *sumLine = (uint64_t *)tempLine;
    integral.readVLine(sumLine, integral.getHeight(), 0, 0, z);
    for (size_t x = 1; x < integral.getWidth(); ++x) {
      integral.readVLine(intLine, integral.getHeight(), x, 0, z);
#pragma omp parallel for
#if defined(__x86_64__) || defined(__i386__)
# if INSTRSET >= 9 // AVX512 - 512bits
      for (size_t y = 0; y < integral.getHeight(); y += 8) {
	Vec8uq pixVec;
	pixVec.load(&intLine[y]);
	Vec8uq sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 8 // AVX2 - 256bits
      for (size_t y = 0; y < integral.getHeight(); y += 4) {
	Vec4uq pixVec;
	pixVec.load(&intLine[y]);
	Vec4uq sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# elif INSTRSET >= 2 // SSE2 - 128bits
      for (size_t y = 0; y < integral.getHeight(); y += 2) {
	Vec2uq pixVec;
	pixVec.load(&intLine[y]);
	Vec2uq sumVec;
	sumVec.load(&sumLine[y]);
	sumVec += pixVec;
	sumVec.store(&sumLine[y]);
      }
# endif
#elif defined(__ARM_NEON__)
      for (size_t y = 0; y < pixmap.getHeight(); y += 2) {
	uint64x2_t pixVec = vld1q_u64((const uint64_t *)&intLine[y]);
	uint64x2_t sumVec = vld1q_u64((const uint64_t *)&sumLine[y]);
	sumVec = vaddq_u64(sumVec, pixVec);
	vst1q_u64((uint64_t *)&sumLine[y], sumVec);
      }
#endif
      integral.writeVLine(sumLine, integral.getHeight(), x, 0, z);
    }
  }
  
  delete [] intLine;
  delete [] tempLine;
}
