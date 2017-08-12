
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides declarations for the common test utility functions (performance clients and
// correctness testers).
//
// =================================================================================================

#ifndef CLBLAST_TEST_UTILITIES_H_
#define CLBLAST_TEST_UTILITIES_H_

#include <string>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// The client-specific arguments in string form
constexpr auto kArgCompareclblas = "clblas";
constexpr auto kArgComparecblas = "cblas";
constexpr auto kArgComparecublas = "cublas";
constexpr auto kArgStepSize = "step";
constexpr auto kArgNumSteps = "num_steps";
constexpr auto kArgWarmUp = "warm_up";

// The test-specific arguments in string form
constexpr auto kArgFullTest = "full_test";
constexpr auto kArgVerbose = "verbose";

// =================================================================================================

// Returns whether a scalar is close to zero
template <typename T> bool IsCloseToZero(const T value);

// =================================================================================================

// Structure containing all possible buffers for test clients
template <typename T>
struct Buffers {
  Buffer<T> x_vec;
  Buffer<T> y_vec;
  Buffer<T> a_mat;
  Buffer<T> b_mat;
  Buffer<T> c_mat;
  Buffer<T> ap_mat;
  Buffer<T> scalar;
};
template <typename T>
struct BuffersHost {
  std::vector<T> x_vec;
  std::vector<T> y_vec;
  std::vector<T> a_mat;
  std::vector<T> b_mat;
  std::vector<T> c_mat;
  std::vector<T> ap_mat;
  std::vector<T> scalar;
};

// =================================================================================================

// Converts a value (e.g. an integer) to a string. This also covers special cases for CLBlast
// data-types such as the Layout and Transpose data-types.
template <typename T>
std::string ToString(T value);

// =================================================================================================

// Copies buffers from the OpenCL device to the host
template <typename T, typename U>
void DeviceToHost(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names);

// Copies buffers from the host to the OpenCL device
template <typename T, typename U>
void HostToDevice(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names);

// =================================================================================================

// Conversion between half and single-precision
std::vector<float> HalfToFloatBuffer(const std::vector<half>& source);
void FloatToHalfBuffer(std::vector<half>& result, const std::vector<float>& source);

// As above, but now for OpenCL data-types instead of std::vectors
Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, cl_command_queue queue_raw);
void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, cl_command_queue queue_raw);

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_UTILITIES_H_
#endif
