
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the ConvGemm kernels. These kernels are based on the GEMM
// direct kernel and will use those parameters, this tuner is just optional to use for advanced
// users.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Helper functions
template <typename T>
size_t OutputHeight(const Arguments<T> &args) {
  const auto size = args.height + 2 * args.pad_h;
  const auto padding = args.dilation_h * (args.kernel_h - 1) + 1;
  if (size >= padding) { return (size - padding) / args.stride_h + 1; }
  return 1;
}
template <typename T>
size_t OutputWidth(const Arguments<T> &args) {
  const auto size = args.width + 2 * args.pad_w;
  const auto padding = args.dilation_w * (args.kernel_w - 1) + 1;
  if (size >= padding) { return (size - padding) / args.stride_w + 1; }
  return 1;
}

// Settings for this kernel (default command-line arguments)
TunerDefaults XConvGemmGetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgChannels, kArgHeight, kArgWidth, kArgKernelH, kArgKernelW,
                      kArgNumKernels, kArgBatchCount, kArgFraction};
  settings.channels = 32;
  settings.height = 66;
  settings.width = 66;  // num_patches = 64x64 = 4096
  settings.kernel_h = 3;
  settings.kernel_w = 3;
  settings.num_kernels = 32;
  settings.default_batch_count = 16;
  settings.default_fraction = 1.0;
  settings.default_num_runs = 2;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings XConvGemmGetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "xconvgemm";
  settings.kernel_name = "XconvgemmNormal";
  settings.sources =
"#define ROUTINE_CONVGEMM"
#include "../src/kernels/level3/xgemm_direct_part1.opencl"
#include "../src/kernels/level3/xgemm_direct_part2.opencl"
#include "../src/kernels/level3/xgemm_direct_part3.opencl"
#include "../src/kernels/levelx/xconvgemm_part1.opencl"
#include "../src/kernels/levelx/xconvgemm_part2.opencl"
  ;

  // Helper variables
  const auto patch_size = args.kernel_h * args.kernel_w * args.channels;
  const auto num_patches = OutputHeight(args) * OutputWidth(args);

  // Buffer sizes
  settings.size_a = args.batch_count * args.channels * args.height * args.width;
  settings.size_b = args.num_kernels * args.channels * args.kernel_h * args.kernel_w;
  settings.size_c = args.batch_count * args.num_kernels * OutputHeight(args) * OutputWidth(args);

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {2, 3, 4};
  settings.outputs = {4};

  // Sets the base thread configuration
  settings.global_size = {num_patches, args.num_kernels, args.batch_count};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1, 1, 1};
  settings.local_size_ref = {8, 8, 1};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"MDIMCD", "NDIMCD"}};
  settings.mul_global = {{"MDIMCD", "NDIMCD"}};
  settings.div_global = {{"WGD", "WGD"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"WGD", {8, 16, 32}},
    {"MDIMCD", {8, 16, 32}},
    {"NDIMCD", {8, 16, 32}},
    {"MDIMAD", {8, 16, 32}},
    {"NDIMBD", {8, 16, 32}},
    {"KWID", {1}},
    {"VWMD", {1, 2, 4, 8}},
    {"VWND", {1, 2, 4, 8}},
    {"PADA", {0}},
    {"PADB", {0}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = args.batch_count * 2 * num_patches * args.num_kernels * patch_size;
  settings.performance_unit = "GFLOPS";

  return settings;
}

// Tests for valid arguments
template <typename T>
void XConvGemmTestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> XConvGemmSetConstraints(const int) {
  auto constraints = std::vector<Constraint>();
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
  auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };
  // Requirement for unrolling the WGD loop
  constraints.push_back({MultipleOfX, {"WGD", "KWID"}});
  // Required for integer MWID and NWID
  constraints.push_back({MultipleOfXMulY, {"WGD", "MDIMCD", "VWMD"}});
  constraints.push_back({MultipleOfXMulY, {"WGD", "NDIMCD", "VWND"}});
  // Required for integer MWIAD and NWIBD
  constraints.push_back({MultipleOfXMulY, {"WGD", "MDIMAD", "VWMD"}});
  constraints.push_back({MultipleOfXMulY, {"WGD", "NDIMBD", "VWND"}});
  // WGD has to be a multiple of KDIMAD = ((MDIMCD*NDIMCD)/(MDIMAD)) and KDIMBD = (...)
  constraints.push_back({MultipleOfXMulYDivZ, {"WGD", "MDIMCD", "NDIMCD", "MDIMAD"}});
  constraints.push_back({MultipleOfXMulYDivZ, {"WGD", "MDIMCD", "NDIMCD", "NDIMBD"}});

  return constraints;
}
template <typename T>
LocalMemSizeInfo XConvGemmComputeLocalMemSize(const int) {
  return {
      [] (std::vector<size_t> v) -> size_t {
          return GetBytes(PrecisionValue<T>()) * ((v[0]*(v[0] + v[1]) + v[0]*(v[0] + v[2])));
      },
      {"WGD", "PADA", "PADB"}
  };
}

// Sets the kernel's arguments
template <typename T>
void XConvGemmSetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  const auto output_h = OutputHeight(args);
  const auto output_w = OutputWidth(args);
  const auto patch_size = args.kernel_h * args.kernel_w * args.channels;
  const auto num_patches = output_h * output_w;
  const auto result_stride = args.num_kernels * output_h * output_w;
  kernel.SetArgument(0, static_cast<int>(num_patches));
  kernel.SetArgument(1, static_cast<int>(args.num_kernels));
  kernel.SetArgument(2, static_cast<int>(patch_size));
  kernel.SetArgument(3, buffers[3]()); // 3 == B matrix ==> kernel buffer
  kernel.SetArgument(4, 0); // kernel offset
  kernel.SetArgument(5, buffers[4]()); // 4 == C matrix ==> result buffer
  kernel.SetArgument(6, 0); // result offset
  kernel.SetArgument(7, static_cast<int>(result_stride));
  kernel.SetArgument(8, buffers[2]()); // 2 == A matrix ==> image buffer
  kernel.SetArgument(9, 0); // image offset
  kernel.SetArgument(10, static_cast<int>(args.height));
  kernel.SetArgument(11, static_cast<int>(args.width));
  kernel.SetArgument(12, static_cast<int>(args.channels));
  kernel.SetArgument(13, static_cast<int>(args.kernel_h));
  kernel.SetArgument(14, static_cast<int>(args.kernel_w));
  kernel.SetArgument(15, 0); // pad_h
  kernel.SetArgument(16, 0); // pad_w
  kernel.SetArgument(17, 1); // stride_h
  kernel.SetArgument(18, 1); // stride_w
  kernel.SetArgument(19, 1); // dilation_h
  kernel.SetArgument(20, 1); // dilation_w
  kernel.SetArgument(21, static_cast<int>(output_h));
  kernel.SetArgument(22, static_cast<int>(output_w));
}

// =================================================================================================
} // namespace clblast
