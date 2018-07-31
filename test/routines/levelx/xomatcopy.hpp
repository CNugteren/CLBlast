
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xomatcopy routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XOMATCOPY_H_
#define CLBLAST_TEST_ROUTINES_XOMATCOPY_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
StatusCode RunReference(const Arguments<T> &args, BuffersHost<T> &buffers_host) {

  // Checking for invalid arguments
  const auto a_rotated = (args.layout == Layout::kRowMajor);
  const auto b_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                         (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
  const auto a_base = (a_rotated) ? args.a_ld*(args.m-1) + args.n : args.a_ld*(args.n-1) + args.m;
  const auto b_base = (b_rotated) ? args.b_ld*(args.m-1) + args.n : args.b_ld*(args.n-1) + args.m;
  if ((args.m == 0) || (args.n == 0)) { return StatusCode::kInvalidDimension; }
  if ((args.a_ld < args.m && !a_rotated) || (args.a_ld < args.n && a_rotated)) { return StatusCode::kInvalidLeadDimA; }
  if ((args.b_ld < args.m && !b_rotated) || (args.b_ld < args.n && b_rotated)) { return StatusCode::kInvalidLeadDimB; }
  if (buffers_host.a_mat.size() * sizeof(T) < (a_base + args.a_offset) * sizeof(T)) { return StatusCode::kInsufficientMemoryA; }
  if (buffers_host.b_mat.size() * sizeof(T) < (b_base + args.b_offset) * sizeof(T)) { return StatusCode::kInsufficientMemoryB; }

  // Matrix copy, scaling, and/or transpose
  for (auto id1 = size_t{0}; id1 < args.m; ++id1) {
    for (auto id2 = size_t{0}; id2 < args.n; ++id2) {
      const auto a_one = (a_rotated) ? id2 : id1;
      const auto a_two = (a_rotated) ? id1 : id2;
      const auto b_one = (b_rotated) ? id2 : id1;
      const auto b_two = (b_rotated) ? id1 : id2;
      const auto a_index = a_two * args.a_ld + a_one + args.a_offset;
      const auto b_index = b_two * args.b_ld + b_one + args.b_offset;
      auto a_value = buffers_host.a_mat[a_index];
      if (args.a_transpose == Transpose::kConjugate) { a_value = ComplexConjugate(a_value); }
      buffers_host.b_mat[b_index] = args.alpha * a_value;
    }
  }
  return StatusCode::kSuccess;
}

// Half-precision version calling the above reference implementation after conversions
template <>
StatusCode RunReference<half>(const Arguments<half> &args, BuffersHost<half> &buffers_host) {
  auto a_buffer2 = HalfToFloatBuffer(buffers_host.a_mat);
  auto b_buffer2 = HalfToFloatBuffer(buffers_host.b_mat);
  auto dummy = std::vector<float>(0);
  auto buffers2 = BuffersHost<float>{dummy, dummy, a_buffer2, b_buffer2, dummy, dummy, dummy};
  auto args2 = Arguments<float>();
  args2.a_size = args.a_size; args2.b_size = args.b_size;
  args2.a_ld = args.a_ld; args2.b_ld = args.b_ld; args2.m = args.m; args2.n = args.n;
  args2.a_offset = args.a_offset; args2.b_offset = args.b_offset;
  args2.layout = args.layout; args2.a_transpose = args.a_transpose;
  args2.alpha = HalfToFloat(args.alpha);
  auto status = RunReference(args2, buffers2);
  FloatToHalfBuffer(buffers_host.b_mat, buffers2.b_mat);
  return status;
}

// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXomatcopy {
 public:

  // The BLAS level: 4 for the extra routines
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN,
            kArgLayout, kArgATransp,
            kArgALeadDim, kArgBLeadDim,
            kArgAOffset, kArgBOffset,
            kArgAlpha};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatB}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<T> &args) {
    const auto a_rotated = (args.layout == Layout::kRowMajor);
    const auto a_two = (a_rotated) ? args.m : args.n;
    return a_two * args.a_ld + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) {
    const auto b_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                           (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    const auto b_two = (b_rotated) ? args.n : args.m;
    return b_two * args.b_ld + args.b_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDB(const Arguments<T> &args) { return args.m; }
  static size_t DefaultLDC(const Arguments<T> &) { return 1; } // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &all) { return all; }
  static Transposes GetBTransposes(const Transposes &) { return {}; } // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>&, Queue&, const int, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&) {} // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    #ifdef OPENCL_API
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = Omatcopy<T>(args.layout, args.a_transpose,
                                args.m, args.n, args.alpha,
                                buffers.a_mat(), args.a_offset, args.a_ld,
                                buffers.b_mat(), args.b_offset, args.b_ld,
                                &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = Omatcopy<T>(args.layout, args.a_transpose,
                                args.m, args.n, args.alpha,
                                buffers.a_mat(), args.a_offset, args.a_ld,
                                buffers.b_mat(), args.b_offset, args.b_ld,
                                queue.GetContext()(), queue.GetDevice()());
      cuStreamSynchronize(queue());
    #endif
    return status;
  }

  // Describes how to run a naive version of the routine (for correctness/performance comparison).
  // Note that a proper clBLAS or CPU BLAS comparison is not available for non-BLAS routines.
  static StatusCode RunReference1(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    auto buffers_host = BuffersHost<T>();
    DeviceToHost(args, buffers, buffers_host, queue, BuffersIn());
    const auto status = RunReference(args, buffers_host);
    HostToDevice(args, buffers, buffers_host, queue, BuffersOut());
    return status;
  }

  static StatusCode RunReference2(const Arguments<T> &args, BuffersHost<T> &buffers_host, Queue&) {
    return RunReference(args, buffers_host);
  }
  static StatusCode RunReference3(const Arguments<T> &, BuffersCUDA<T> &, Queue &) {
    return StatusCode::kUnknownError;
  }

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.b_size, static_cast<T>(0));
    buffers.b_mat.Read(queue, args.b_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.m; }
  static size_t ResultID2(const Arguments<T> &args) { return args.n; }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    const auto b_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                           (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    const auto b_one = (b_rotated) ? id2 : id1;
    const auto b_two = (b_rotated) ? id1 : id2;
    return b_two * args.b_ld + b_one + args.b_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return args.m*args.n;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return (2*args.m*args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XOMATCOPY_H_
#endif
