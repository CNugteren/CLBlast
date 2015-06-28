
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xsyrk routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XSYRK_H_
#define CLBLAST_TEST_ROUTINES_XSYRK_H_

#include <vector>
#include <string>

#include "wrapper_clblas.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXsyrk {
 public:

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN, kArgK,
            kArgLayout, kArgTriangle, kArgATransp,
            kArgALeadDim, kArgCLeadDim,
            kArgAOffset, kArgCOffset,
            kArgAlpha, kArgBeta};
  }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<T> &args) {
    auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                     (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
    auto a_two = (a_rotated) ? args.n : args.k;
    return a_two * args.a_ld + args.a_offset;
  }
  static size_t GetSizeC(const Arguments<T> &args) {
    return args.n * args.c_ld + args.c_offset;
  }

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, const Buffers &buffers,
                               CommandQueue &queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Syrk(args.layout, args.triangle, args.a_transpose,
                       args.n, args.k, args.alpha,
                       buffers.a_mat(), args.a_offset, args.a_ld, args.beta,
                       buffers.c_mat(), args.c_offset, args.c_ld,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  static StatusCode RunReference(const Arguments<T> &args, const Buffers &buffers,
                                 CommandQueue &queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXsyrk(static_cast<clblasOrder>(args.layout),
                              static_cast<clblasUplo>(args.triangle),
                              static_cast<clblasTranspose>(args.a_transpose),
                              args.n, args.k, args.alpha,
                              buffers.a_mat(), args.a_offset, args.a_ld, args.beta,
                              buffers.c_mat(), args.c_offset, args.c_ld,
                              1, &queue_plain, 0, nullptr, &event);
    clWaitForEvents(1, &event);
    return static_cast<StatusCode>(status);
  }

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers &buffers,
                                       CommandQueue &queue) {
    std::vector<T> result(args.c_size, static_cast<T>(0));
    buffers.c_mat.ReadBuffer(queue, args.c_size*sizeof(T), result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.n; }
  static size_t ResultID2(const Arguments<T> &args) { return args.n; }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return id1*args.c_ld + id2 + args.c_offset;
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XSYRK_H_
#endif
