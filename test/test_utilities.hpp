
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

#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>

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
constexpr auto kArgTunerFiles = "tuner_files";

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

template <typename T> T ComplexConjugate(const T value);

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
#ifdef OPENCL_API
  Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, RawCommandQueue queue_raw);
  void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, RawCommandQueue queue_raw);
#endif

// =================================================================================================

// Creates a buffer but don't test for validity. That's the reason this is not using the clpp11.h or
// cupp11.h interface.
template <typename T>
Buffer<T> CreateInvalidBuffer(const Context& context, const size_t size) {
  #ifdef OPENCL_API
    auto raw_buffer = clCreateBuffer(context(), CL_MEM_READ_WRITE, size * sizeof(T), nullptr, nullptr);
  #elif CUDA_API
    CUdeviceptr raw_buffer;
    cuMemAlloc(&raw_buffer, size * sizeof(T));
  #endif
  return Buffer<T>(raw_buffer);
}

// =================================================================================================

using BestParameters = std::unordered_map<std::string,size_t>;
using BestParametersCollection = std::unordered_map<std::string, BestParameters>;

void OverrideParametersFromJSONFiles(const std::vector<std::string>& file_names,
                                     const RawDeviceID device, const Precision precision);
void GetBestParametersFromJSONFile(const std::string& file_name,
                                   BestParametersCollection& all_parameters,
                                   const Precision precision);

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_UTILITIES_H_
#endif
