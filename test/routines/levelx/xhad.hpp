
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xhad routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XHAD_H_
#define CLBLAST_TEST_ROUTINES_XHAD_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
StatusCode RunReference(const Arguments<T> &args, BuffersHost<T> &buffers_host) {
  for (auto index = size_t{0}; index < args.n; ++index) {
    const auto x = buffers_host.x_vec[index * args.x_inc + args.x_offset];
    const auto y = buffers_host.y_vec[index * args.y_inc + args.y_offset];
    const auto z = buffers_host.c_mat[index]; // * args.z_inc + args.z_offset];
    buffers_host.c_mat[index] = args.alpha * x * y + args.beta * z;
  }
  return StatusCode::kSuccess;
}

// Half-precision version calling the above reference implementation after conversions
template <>
StatusCode RunReference<half>(const Arguments<half> &args, BuffersHost<half> &buffers_host) {
  auto x_buffer2 = HalfToFloatBuffer(buffers_host.x_vec);
  auto y_buffer2 = HalfToFloatBuffer(buffers_host.y_vec);
  auto c_buffer2 = HalfToFloatBuffer(buffers_host.c_mat);
  auto dummy = std::vector<float>(0);
  auto buffers2 = BuffersHost<float>{x_buffer2, y_buffer2, dummy, dummy, c_buffer2, dummy, dummy};
  auto args2 = Arguments<float>();
  args2.x_size = args.x_size; args2.y_size = args.y_size; args2.c_size = args.c_size;
  args2.x_inc = args.x_inc; args2.y_inc = args.y_inc; args2.n = args.n;
  args2.x_offset = args.x_offset; args2.y_offset = args.y_offset;
  args2.alpha = HalfToFloat(args.alpha); args2.beta = HalfToFloat(args.beta);
  auto status = RunReference(args2, buffers2);
  FloatToHalfBuffer(buffers_host.c_mat, buffers2.c_mat);
  return status;
}

// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXhad {
public:

  // The BLAS level: 4 for the extra routines (note: tested with matrix-size values for 'n')
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN,
            kArgXInc, kArgYInc,
            kArgXOffset, kArgYOffset,
            kArgAlpha, kArgBeta};
  }
  static std::vector<std::string> BuffersIn() { return {kBufVecX, kBufVecY, kBufMatC}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatC}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) {
    return args.n * args.x_inc + args.x_offset;
  }
  static size_t GetSizeY(const Arguments<T> &args) {
    return args.n * args.y_inc + args.y_offset;
  }
  static size_t GetSizeC(const Arguments<T> &args) { // used for 'vector z'
    return args.n; // * args.z_inc + args.z_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.x_size = GetSizeX(args);
    args.y_size = GetSizeY(args);
    args.c_size = GetSizeC(args); // used for 'vector z'
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t DefaultLDB(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t DefaultLDC(const Arguments<T> &) { return 1; } // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &) { return {}; } // N/A for this routine
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
      auto status = Had(args.n, args.alpha,
                        buffers.x_vec(), args.x_offset, args.x_inc,
                        buffers.y_vec(), args.y_offset, args.y_inc, args.beta,
                        buffers.c_mat(), 0, 1, // used for 'vector z'
                        &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
#elif CUDA_API
    auto status = Had(args.n, args.alpha,
                      buffers.x_vec(), args.x_offset, args.x_inc,
                      buffers.y_vec(), args.y_offset, args.y_inc, args.beta,
                      buffers.c_mat(), 0, 1, // used for 'vector z'
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
    std::vector<T> result(args.c_size, static_cast<T>(0));
    buffers.c_mat.Read(queue, args.c_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.n; }
  static size_t ResultID2(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t) {
    return id1; // * args.z_inc + args.z_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return 4 * args.n;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return (4 * args.n) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XHAD_H_
#endif
