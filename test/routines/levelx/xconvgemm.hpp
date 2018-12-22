
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xconvgemm routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XCONVGEMM_H_
#define CLBLAST_TEST_ROUTINES_XCONVGEMM_H_

#include "test/routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXconvgemm {
public:

  // The BLAS level: 4 for the extra routines
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgKernelMode,
            kArgChannels, kArgHeight, kArgWidth, kArgKernelH, kArgKernelW, kArgPadH, kArgPadW,
            kArgStrideH, kArgStrideW, kArgDilationH, kArgDilationW, kArgNumKernels, kArgBatchCount,
            kArgAOffset, kArgBOffset, kArgCOffset};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB, kBufMatC}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatC}; }

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
  static size_t GetSizeA(const Arguments<T> &args) { // 4D: NCHW == batch-channel-height-width
    return args.batch_count * args.channels * args.height * args.width + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) { // 4D: KCHW == kernel-channel-height-width
    return args.num_kernels * args.channels * args.kernel_h * args.kernel_w + args.b_offset;
  }
  static size_t GetSizeC(const Arguments<T> &args) { // 4D: NCHW == batch-channel-height-width
    return args.batch_count * args.num_kernels * OutputHeight(args) * OutputWidth(args) + args.c_offset;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
    args.c_size = GetSizeC(args);
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
    auto status = Convgemm<T>(args.kernel_mode,
                              args.channels, args.height, args.width,
                              args.kernel_h, args.kernel_w,
                              args.pad_h, args.pad_w,
                              args.stride_h, args.stride_w,
                              args.dilation_h, args.dilation_w,
                              args.num_kernels, args.batch_count,
                              buffers.a_mat(), args.a_offset,
                              buffers.b_mat(), args.b_offset,
                              buffers.c_mat(), args.c_offset,
                              &queue_plain, &event);
    if (status == StatusCode::kSuccess) { clWaitForEvents(1, &event); clReleaseEvent(event); }
#elif CUDA_API
    auto status = Convgemm<T>(args.kernel_mode,
                              args.channels, args.height, args.width,
                              args.kernel_h, args.kernel_w,
                              args.pad_h, args.pad_w,
                              args.stride_h, args.stride_w,
                              args.dilation_h, args.dilation_w,
                              args.num_kernels, args.batch_count,
                              buffers.a_mat(), args.a_offset,
                              buffers.b_mat(), args.b_offset,
                              buffers.c_mat(), args.c_offset,
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
  static size_t ResultID1(const Arguments<T> &args) { return OutputHeight(args) * OutputWidth(args); }
  static size_t ResultID2(const Arguments<T> &args) { return args.num_kernels * args.batch_count; }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return id1 + OutputHeight(args) * OutputWidth(args) * id2 + args.c_offset;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    const auto patch_size = args.kernel_h * args.kernel_w * args.channels;
    const auto num_patches = OutputHeight(args) * OutputWidth(args);
    return args.batch_count * 2 * num_patches * args.num_kernels * patch_size;
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return (GetSizeA(args) + GetSizeB(args) + GetSizeC(args)) * sizeof(T);
  }
};

// =================================================================================================

template <typename T>
StatusCode RunReference(const Arguments<T> &args, BuffersHost<T> &buffers_host) {
  const auto output_h = TestXconvgemm<T>::OutputHeight(args);
  const auto output_w = TestXconvgemm<T>::OutputWidth(args);
  for (auto batch_id = size_t{0}; batch_id < args.batch_count; ++batch_id) {
    for (auto co_id = size_t{0}; co_id < args.num_kernels; ++co_id) { // output channels == num-kernels
      for (auto ho_id = size_t{0}; ho_id < output_h; ++ho_id) { // image height
        for (auto wo_id = size_t{0}; wo_id < output_w; ++wo_id) { // image width
          auto result = ConstantZero<T>();

          // 3D convolution
          for (auto ci_id = size_t{0}; ci_id < args.channels; ++ci_id) { // input channels
            for (auto kh_id = size_t{0}; kh_id < args.kernel_h; ++kh_id) { // kernel height
              for (auto kw_id = size_t{0}; kw_id < args.kernel_w; ++kw_id) { // kernel width

                // Retrieves the value from the input image
                const auto hi_id = kh_id * args.dilation_h + args.stride_h * ho_id - args.pad_h;
                const auto wi_id = kw_id * args.dilation_w + args.stride_w * wo_id - args.pad_w;
                if (hi_id >= 0 && hi_id < args.height &&
                    wi_id >= 0 && wi_id < args.width) {
                  const auto input_index = wi_id + args.width * (
                                           hi_id + args.height * (
                                           ci_id + args.channels * (
                                           batch_id)));
                  const auto input_value = buffers_host.a_mat[input_index + args.a_offset];

                  // Multiplies with the kernel tensor
                  const auto kernel_index
                          = (args.kernel_mode == KernelMode::kConvolution)
                          ? (args.kernel_w - kw_id - 1) + args.kernel_w * (
                            (args.kernel_h - kh_id - 1) + args.kernel_h * (
                            ci_id + args.channels * (
                            co_id)))
                          : kw_id + args.kernel_w * (
                            kh_id + args.kernel_h * (
                            ci_id + args.channels * (
                            co_id)));
                  const auto kernel_value = buffers_host.b_mat[kernel_index + args.b_offset];
                  result += input_value * kernel_value;

                }
              }
            }
          }

          // Sets the output value (NCHW == batch-channel-height-width)
          const auto output_index = wo_id + output_w * (
                                    ho_id + output_h * (
                                    co_id + args.num_kernels * (
                                    batch_id)));
          buffers_host.c_mat[output_index + args.c_offset] = result;
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

// Half-precision version calling the above reference implementation after conversions
template <>
StatusCode RunReference<half>(const Arguments<half> &args, BuffersHost<half> &buffers_host) {
  auto a_buffer2 = HalfToFloatBuffer(buffers_host.a_mat);
  auto b_buffer2 = HalfToFloatBuffer(buffers_host.b_mat);
  auto c_buffer2 = HalfToFloatBuffer(buffers_host.c_mat);
  auto dummy = std::vector<float>(0);
  auto buffers2 = BuffersHost<float>{dummy, dummy, a_buffer2, b_buffer2, c_buffer2, dummy, dummy};
  auto args2 = Arguments<float>();
  args2.a_size = args.a_size; args2.b_size = args.b_size; args2.c_size = args.c_size;
  args2.kernel_mode = args.kernel_mode;
  args2.channels = args.channels; args2.height = args.height; args2.width = args.width;
  args2.kernel_h = args.kernel_h; args2.kernel_w = args.kernel_w;
  args2.pad_h = args.pad_h; args2.pad_w = args.pad_w;
  args2.stride_h = args.stride_h; args2.stride_w = args.stride_w;
  args2.dilation_h = args.dilation_h; args2.dilation_w = args.dilation_w;
  args2.num_kernels = args.num_kernels; args2.batch_count = args.batch_count;
  args2.a_offset = args.a_offset; args2.b_offset = args.b_offset; args2.c_offset = args.c_offset;
  auto status = RunReference(args2, buffers2);
  FloatToHalfBuffer(buffers_host.c_mat, buffers2.c_mat);
  return status;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XCONVGEMM_H_
#endif
