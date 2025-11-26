
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file implements a class with static methods to describe the Xminmax routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XMINMAX_H_
#define CLBLAST_TEST_ROUTINES_XMINMAX_H_

#include <clblast.h>
#include <clblast_half.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#include "test/test_utilities.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXminmax {
 public:
  // To test, it runs against the custom implementations of Xmin and Xmax provided by the CLBlast library
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN, kArgImaxOffset, kArgIminOffset, kArgXOffset, kArgXInc};
  }
  static std::vector<std::string> BuffersIn() { return {kBufVecX, kBufScalarUint, kBufSecondScalarUint}; }
  static std::vector<std::string> BuffersOut() { return {kBufScalarUint, kBufSecondScalarUint}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T>& args) { return (args.n * args.x_inc) + args.x_offset; }
  static size_t GetSizeImax(const Arguments<T>& args) { return args.imax_offset + 1; }
  static size_t GetSizeImin(const Arguments<T>& args) { return args.imin_offset + 1; }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T>& args, Queue& /*unused*/) {
    args.x_size = GetSizeX(args);
    args.scalar_size = GetSizeImax(args);
    args.second_scalar_size = GetSizeImin(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T>& /*unused*/) { return 1; }  // N/A for this routine
  static size_t DefaultLDB(const Arguments<T>& /*unused*/) { return 1; }  // N/A for this routine
  static size_t DefaultLDC(const Arguments<T>& /*unused*/) { return 1; }  // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes& /*unused*/) { return {}; }  // N/A for this routine
  static Transposes GetBTransposes(const Transposes& /*unused*/) { return {}; }  // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>& /*unused*/, Queue& /*unused*/, const int /*unused*/,
                          std::vector<T>& /*unused*/, std::vector<T>& /*unused*/, std::vector<T>& /*unused*/,
                          std::vector<T>& /*unused*/, std::vector<T>& /*unused*/, std::vector<T>& /*unused*/,
                          std::vector<T>& /*unused*/) {}  // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
#ifdef OPENCL_API
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Minmax<T>(args.n, buffers.scalar_uint(), args.imax_offset, buffers.second_scalar_uint(),
                            args.imin_offset, buffers.x_vec(), args.x_offset, args.x_inc, &queue_plain, &event);
    if (status == StatusCode::kSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
#elif CUDA_API
    auto status =
        Minmax<T>(args.n, buffers.scalar_uint(), args.imax_offset, buffers.second_scalar_uint(), args.imin_offset,
                  buffers.x_vec(), args.x_offset, args.x_inc, queue.GetContext()(), queue.getDevice()());
    cuStreamSynchronize(queue());
#endif
    return status;
  }

  // Note: No CBlas or CLBlas routine exists to compare against so instead it compares against the results of Max and
  // Min routines in CLBlast.
  static StatusCode RunReference1(const Arguments<float>& args, BuffersHost<float>& buffers, Queue& queue) {
    auto minmax =
        std::minmax_element(buffers.x_vec.begin() + args.x_offset, buffers.x_vec.begin() + args.x_offset + args.n);
    buffers.scalar_uint[args.imax_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.second));
    buffers.second_scalar_uint[args.imin_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.first));
    return StatusCode::kSuccess;
  }

  // Note: No CBlas or CLBlas routine exists to compare against so instead it compares against the results of Max and
  // Min routines in CLBlast.
  static StatusCode RunReference1(const Arguments<double>& args, BuffersHost<double>& buffers, Queue& queue) {
    auto minmax =
        std::minmax_element(buffers.x_vec.begin() + args.x_offset, buffers.x_vec.begin() + args.x_offset + args.n);
    buffers.scalar_uint[args.imax_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.second));
    buffers.second_scalar_uint[args.imin_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.first));
    return StatusCode::kSuccess;
  }

  // Note: No CBlas or CLBlas routine exists to compare against so instead it compares against the results of Max and
  // Min routines in CLBlast.
  static StatusCode RunReference1(const Arguments<half>& args, BuffersHost<half>& buffers, Queue& queue) {
    auto minmax =
        std::minmax_element(buffers.x_vec.begin() + args.x_offset, buffers.x_vec.begin() + args.x_offset + args.n,
                            [](half arg1, half arg2) { return HalfToFloat(arg1) < HalfToFloat(arg2); });
    buffers.scalar_uint[args.imax_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.second));
    buffers.second_scalar_uint[args.imin_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.first));
    return StatusCode::kSuccess;
  }

  // Note: No CBlas or CLBlas routine exists to compare against so instead it compares against the results of Max and
  // Min routines in CLBlast.
  static StatusCode RunReference1(const Arguments<T>& args, BuffersHost<T>& buffers, Queue& queue) {
    auto minmax = std::minmax_element(buffers.x_vec.begin() + args.x_offset,
                                      buffers.x_vec.begin() + args.x_offset + args.n, [](const T& arg1, const T& arg2) {
                                        return (std::fabs(arg1.real()) + std::fabs(arg1.imag())) <
                                               (std::fabs(arg2.real()) + std::fabs(arg2.imag()));
                                      });
    buffers.scalar_uint[args.imax_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.second));
    buffers.second_scalar_uint[args.imin_offset] =
        static_cast<unsigned int>(std::distance(buffers.x_vec.begin() + args.x_offset, minmax.first));
    return StatusCode::kSuccess;
  }

// Note: No CBlas routine exists to compare against so intead uses the RunReference1 method which works for both CBlas
// and CLBlas
#ifdef CLBLAST_REF_CBLAS
  static StatusCode RunReference2(const Arguments<T>& args, BuffersHost<T>& buffers_host, Queue& queue) {
    auto status = RunReference1(args, buffers_host, queue);
    return status;
  }
#endif

// Describes how to run the cuBLAS routine (for correctness/performance comparison)
#ifdef CLBLAST_REF_CUBLAS
  static StatusCode RunReference3(const Arguments<T>& args, BuffersCUDA<T>& buffers, Queue&) {
    return StatusCode::kUnknownError;
  }
#endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    std::vector<unsigned int> result_uint(args.scalar_size, 0);
    std::vector<unsigned int> second_result_uint(args.second_scalar_size, 0);
    buffers.scalar_uint.Read(queue, args.scalar_size, result_uint);
    buffers.second_scalar_uint.Read(queue, args.second_scalar_size, second_result_uint);
    // The result is an integer. However, since the test infrastructure assumes results of
    // type 'T' (float/double/float2/double2/half), we store the results into T instead.
    // The values might then become meaningless, but a comparison for testing should still
    // be valid to verify correctness.
    std::vector<T> result(args.scalar_size + args.second_scalar_size);
    result[0] = static_cast<T>(result_uint[args.imax_offset]);
    result[1] = static_cast<T>(second_result_uint[args.imin_offset]);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T>& /*unused*/) { return 1; }  // N/A for this routine
  static size_t ResultID2(const Arguments<T>& /*unused*/) { return 1; }  // N/A for this routine
  static size_t GetResultIndex(const Arguments<T>& args, const size_t /*unused*/, const size_t /*unused*/) {
    return args.imax_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T>& args) { return args.n * 2; }
  static size_t GetBytes(const Arguments<T>& args) { return ((args.n) + 1) * sizeof(T); }
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_TEST_ROUTINES_XAXPYBATCHED_H_
#endif
