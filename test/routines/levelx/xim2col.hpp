
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xim2col routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XIM2COL_H_
#define CLBLAST_TEST_ROUTINES_XIM2COL_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXim2col {
public:

  // The BLAS level: 4 for the extra routines
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgChannels, kArgHeight, kArgWidth, kArgKernelH, kArgKernelW, kArgPadH, kArgPadW,
            kArgStrideH, kArgStrideW, kArgDilationH, kArgDilationW,
            kArgAOffset, kArgBOffset};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatB}; }

  // Describes how to obtain the sizes of the buffers
  static size_t OutputHeight(const Arguments<T> &args) {
    const auto size = args.height + 2 * args.pad_h;
    const auto padding = args.dilation_h * (args.kernel_h - 1) + 1;
    if (size >= padding) { return (size - padding) / args.stride_h + 1; }
    return 1;
  }
  static size_t OutputWidth(const Arguments<T> &args) {
    const auto size = args.width + 2 * args.pad_w;
    const auto padding = args.dilation_w * (args.kernel_w - 1) + 1;
    if (size >= padding) { return (size - padding) / args.stride_w + 1; }
    return 1;
  }
  static size_t NumPatches(const Arguments<T> &args) {
    return OutputHeight(args) * OutputWidth(args) * args.channels;
  }
  static size_t GetSizeA(const Arguments<T> &args) {
    return args.height * args.width * args.channels + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) {
    return args.kernel_w * args.kernel_h * NumPatches(args) + args.b_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
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
      auto status = Im2col<T>(args.channels, args.height, args.width,
                              args.kernel_h, args.kernel_w,
                              args.pad_h, args.pad_w,
                              args.stride_h, args.stride_w,
                              args.dilation_h, args.dilation_w,
                              buffers.a_mat(), args.a_offset,
                              buffers.b_mat(), args.b_offset,
                              &queue_plain, &event);
      if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
    #elif CUDA_API
      auto status = Im2col<T>(args.channels, args.height, args.width,
                              args.kernel_h, args.kernel_w,
                              args.pad_h, args.pad_w,
                              args.stride_h, args.stride_w,
                              args.dilation_h, args.dilation_w,
                              buffers.a_mat(), args.a_offset,
                              buffers.b_mat(), args.b_offset,
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
  static size_t ResultID1(const Arguments<T> &args) { return args.kernel_h * args.kernel_w; }
  static size_t ResultID2(const Arguments<T> &args) { return NumPatches(args); }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return id1 + args.kernel_h * args.kernel_w * id2 + args.b_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    return 1;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    const auto input = args.channels * args.width * args.height; // possibly less with striding
    const auto output = args.kernel_h * args.kernel_w * NumPatches(args);
    return (input + output) * sizeof(T);
  }
};

// =================================================================================================

template <typename T>
StatusCode RunReference(const Arguments<T> &args, BuffersHost<T> &buffers_host) {
  const auto output_h = TestXim2col<T>::OutputHeight(args);
  const auto output_w = TestXim2col<T>::OutputWidth(args);
  for (auto c_id = size_t{0}; c_id < args.channels; ++c_id) { // input channels
    for (auto kh_id = size_t{0}; kh_id < args.kernel_h; ++kh_id) { // kernel height
      for (auto kw_id = size_t{0}; kw_id < args.kernel_w; ++kw_id) { // kernel width
        for (auto h_id = size_t{0}; h_id < output_h; ++h_id) { // image height
          for (auto w_id = size_t{0}; w_id < output_w; ++w_id) { // image width

            // Retrieves the input value
            const auto h_index = -args.pad_h + kh_id * args.dilation_h + args.stride_h * h_id;
            const auto w_index = -args.pad_w + kw_id * args.dilation_w + args.stride_w * w_id;
            auto val = ConstantZero<T>();
            if (h_index >= 0 && h_index < args.height &&
                w_index >= 0 && w_index < args.width) {
              const auto input_index = w_index + args.width * (h_index + args.height * c_id);
              val = buffers_host.a_mat[input_index + args.a_offset];
            }

            // Sets the output value
            const auto kernel_index = kw_id + args.kernel_w * kh_id;
            const auto patch_index = w_id + output_w * h_id;
            const auto output_index = patch_index + kernel_index * output_w * output_h +
                                      c_id * output_w * output_h * args.kernel_h * args.kernel_w;
            buffers_host.b_mat[output_index + args.b_offset] = val;
          }
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XIM2COL_H_
#endif
