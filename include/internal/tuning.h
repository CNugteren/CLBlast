
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the header for the tuner functions. This is only used for the optional
// and stand-alone tuner binaries and not part of the core of CLBlast. The convention used here is
// that X and Y are vectors, while A, B, and C are matrices.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_H_
#define CLBLAST_TUNING_H_

#include <vector>
#include <functional>

#include <cltune.h>

namespace clblast {
// =================================================================================================

// Functions with two or three OpenCL memory buffers
template <typename T>
using Tuner2 = std::function<void(const Arguments<T>&,
                                  const std::vector<T>&, std::vector<T>&,
                                  cltune::Tuner&)>;
template <typename T>
using Tuner3 = std::function<void(const Arguments<T>&,
                                  const std::vector<T>&, const std::vector<T>&, std::vector<T>&,
                                  cltune::Tuner&)>;

// As above, but now with an additional ID for the variation
template <typename T>
using Tuner3V = std::function<void(const Arguments<T>&, const size_t,
                                   const std::vector<T>&, const std::vector<T>&, std::vector<T>&,
                                   cltune::Tuner&)>;

// Tuner for vector-vector input
template <typename T>
void TunerXY(int argc, char* argv[], const Tuner2<T> &tune_function);

// Tuner for matrix-vector-vector input
template <typename T>
void TunerAXY(int argc, char* argv[], const size_t num_variations, const Tuner3V<T> &tune_function);

// Tuner for matrix-matrix input
template <typename T>
void TunerAB(int argc, char* argv[], const Tuner2<T> &tune_function);

// Tuner for matrix-matrix-matrix input
template <typename T>
void TunerABC(int argc, char* argv[], const Tuner3<T> &tune_function);

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_H_
#endif
