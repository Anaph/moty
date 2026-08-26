/* simd.h — backwards-compatibility wrapper.
 * All SIMD kernels are now in the hw/ layered architecture.
 * Include hw.h instead for new code. This file exists so existing
 * #include "util/simd.h" continues to work unchanged.
 */
#ifndef SIMD_H
#define SIMD_H
#include "hw/hw.h"
#endif
