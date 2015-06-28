
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xaxpy routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XAXPY_H_
#define CLBLAST_TEST_ROUTINES_XAXPY_H_

#include <vector>
#include <string>

#include "wrapper_clblas.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXaxpy {
 public:

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN,
            kArgXInc, kArgYInc,
            kArgXOffset, kArgYOffset,
            kArgAlpha};
  }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) {
    return args.n * args.x_inc + args.x_offset;
  }
  static size_t GetSizeY(const Arguments<T> &args) {
    return args.n * args.y_inc + args.y_offset;
  }

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, const Buffers &buffers,
                               CommandQueue &queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Axpy(args.n, args.alpha,
                       buffers.x_vec(), args.x_offset, args.x_inc,
                       buffers.y_vec(), args.y_offset, args.y_inc,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  static StatusCode RunReference(const Arguments<T> &args, const Buffers &buffers,
                                 CommandQueue &queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXaxpy(args.n, args.alpha,
                              buffers.x_vec(), args.x_offset, args.x_inc,
                              buffers.y_vec(), args.y_offset, args.y_inc,
                              1, &queue_plain, 0, nullptr, &event);
    clWaitForEvents(1, &event);
    return static_cast<StatusCode>(status);
  }

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers &buffers,
                                       CommandQueue &queue) {
    std::vector<T> result(args.y_size, static_cast<T>(0));
    buffers.y_vec.ReadBuffer(queue, args.y_size*sizeof(T), result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.n; }
  static size_t ResultID2(const Arguments<T> &args) { return 1; } // N/A for this routine
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return id1*args.y_inc + args.y_offset;
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XAXPY_H_
#endif
