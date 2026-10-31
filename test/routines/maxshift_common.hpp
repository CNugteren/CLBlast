
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_MAXSHIFT_COMMON_H_
#define CLBLAST_TEST_ROUTINES_MAXSHIFT_COMMON_H_

#include <algorithm>
#include <limits>

#include "clblast_half.h"
#include "test/routines/common.hpp"

namespace clblast {

// Compute the host reference in double precision, converting half values explicitly.
template <typename T>
double MaxshiftRead(const T value) {
  return static_cast<double>(value);
}
template <>
inline double MaxshiftRead<half>(const half value) {
  return HalfToFloat(value);
}

template <typename T>
T MaxshiftWrite(const double value) {
  return static_cast<T>(value);
}
template <>
inline half MaxshiftWrite<half>(const double value) {
  return FloatToHalf(static_cast<float>(value));
}

}  // namespace clblast
#endif
