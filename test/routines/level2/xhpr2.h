
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xhpr2 routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XHPR2_H_
#define CLBLAST_TEST_ROUTINES_XHPR2_H_

#include <vector>
#include <string>

#ifdef CLBLAST_REF_CLBLAS
  #include "wrapper_clblas.h"
#endif
#ifdef CLBLAST_REF_CBLAS
  #include "wrapper_cblas.h"
#endif

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXhpr2 {
 public:

  // The BLAS level: 1, 2, or 3
  static size_t BLASLevel() { return 2; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN,
            kArgLayout, kArgTriangle,
            kArgXInc, kArgYInc,
            kArgAPOffset, kArgXOffset, kArgYOffset,
            kArgAlpha};
  }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) {
    return args.n * args.x_inc + args.x_offset;
  }
  static size_t GetSizeY(const Arguments<T> &args) {
    return args.n * args.y_inc + args.y_offset;
  }
  static size_t GetSizeAP(const Arguments<T> &args) {
    return ((args.n*(args.n+1)) / 2) + args.ap_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args) {
    args.ap_size = GetSizeAP(args);
    args.x_size = GetSizeX(args);
    args.y_size = GetSizeY(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDB(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t DefaultLDC(const Arguments<T> &) { return 1; } // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &) { return {}; } // N/A for this routine
  static Transposes GetBTransposes(const Transposes &) { return {}; } // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Hpr2(args.layout, args.triangle,
                       args.n, args.alpha,
                       buffers.x_vec(), args.x_offset, args.x_inc,
                       buffers.y_vec(), args.y_offset, args.y_inc,
                       buffers.ap_mat(), args.ap_offset,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    return status;
  }

  // Describes how to run the clBLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CLBLAS
    static StatusCode RunReference1(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
      auto queue_plain = queue();
      auto event = cl_event{};
      auto status = clblasXhpr2(static_cast<clblasOrder>(args.layout),
                                static_cast<clblasUplo>(args.triangle),
                                args.n, args.alpha,
                                buffers.x_vec(), args.x_offset, args.x_inc,
                                buffers.y_vec(), args.y_offset, args.y_inc,
                                buffers.ap_mat(), args.ap_offset,
                                1, &queue_plain, 0, nullptr, &event);
      clWaitForEvents(1, &event);
      return static_cast<StatusCode>(status);
    }
  #endif

  // Describes how to run the CPU BLAS routine (for correctness/performance comparison)
  #ifdef CLBLAST_REF_CBLAS
    static StatusCode RunReference2(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
      std::vector<T> ap_mat_cpu(args.ap_size, static_cast<T>(0));
      std::vector<T> x_vec_cpu(args.x_size, static_cast<T>(0));
      std::vector<T> y_vec_cpu(args.y_size, static_cast<T>(0));
      buffers.ap_mat.Read(queue, args.ap_size, ap_mat_cpu);
      buffers.x_vec.Read(queue, args.x_size, x_vec_cpu);
      buffers.y_vec.Read(queue, args.y_size, y_vec_cpu);
      cblasXhpr2(convertToCBLAS(args.layout),
                 convertToCBLAS(args.triangle),
                 args.n, args.alpha,
                 x_vec_cpu, args.x_offset, args.x_inc,
                 y_vec_cpu, args.y_offset, args.y_inc,
                 ap_mat_cpu, args.ap_offset);
      buffers.ap_mat.Write(queue, args.ap_size, ap_mat_cpu);
      return StatusCode::kSuccess;
    }
  #endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.ap_size, static_cast<T>(0));
    buffers.ap_mat.Read(queue, args.ap_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.ap_size - args.ap_offset; }
  static size_t ResultID2(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t) {
    return id1 + args.ap_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return 5 * ((args.n*(args.n+1)) / 2);
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return ((args.n*(args.n+1)) + 2 * args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XHPR2_H_
#endif
