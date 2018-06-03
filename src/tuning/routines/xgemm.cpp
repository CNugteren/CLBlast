
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tunes the Xgemm routine at a high-level: choosing between the direct (single-kernel)
// and the in-direct (kernel plus pre/post-processing) methods.
//
// =================================================================================================

#include <exception>
#include <string>
#include <vector>
#include <iostream>

#include "utilities/utilities.hpp"
#include "test/test_utilities.hpp"
#include "tuning/routines/routine_tuner.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
void RunGemmRoutineMNK(const size_t m, const size_t n, const size_t k,
                       const Queue& queue, const std::vector<Buffer<T>>& buffers) {
  auto queue_plain = queue();
  auto event = cl_event{};
  auto status = Gemm(Layout::kRowMajor, Transpose::kNo, Transpose::kNo,
                     m, n, k, ConstantOne<T>(),
                     buffers[0](), 0, k,
                     buffers[1](), 0, n, ConstantOne<T>(),
                     buffers[2](), 0, n,
                     &queue_plain, &event);
  if (status != StatusCode::kSuccess) {
    throw RuntimeError("Gemm failed with status " + ToString(status));
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}
template <typename T>
void RunGemmRoutine(const size_t value, const Queue& queue, const std::vector<Buffer<T>>& buffers) {
  RunGemmRoutineMNK(value, value, value, queue, buffers);
}

template <typename T, size_t batch_count>
void RunGemmBatchedRoutine(const size_t value, const Queue& queue, const std::vector<Buffer<T>>& buffers) {
  auto offsets = std::vector<size_t>(batch_count);
  auto factors = std::vector<T>(batch_count);
  for (auto i = size_t{0}; i < batch_count; ++i) {
    offsets[i] = batch_count * value;
    factors[i] = ConstantOne<T>();
  }
  auto queue_plain = queue();
  auto event = cl_event{};
  auto status = GemmBatched(Layout::kRowMajor, Transpose::kNo, Transpose::kNo,
                            value, value, value, factors.data(),
                            buffers[0](), offsets.data(), value,
                            buffers[1](), offsets.data(), value, factors.data(),
                            buffers[2](), offsets.data(), value, batch_count,
                            &queue_plain, &event);
  if (status != StatusCode::kSuccess) {
    throw RuntimeError("GemmBatched failed with status " + ToString(status));
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}

template <typename T, size_t batch_count>
void RunGemmStridedBatchedRoutine(const size_t value, const Queue& queue, const std::vector<Buffer<T>>& buffers) {
  auto queue_plain = queue();
  auto event = cl_event{};
  auto status = GemmStridedBatched(Layout::kRowMajor, Transpose::kNo, Transpose::kNo,
                                   value, value, value, ConstantOne<T>(),
                                   buffers[0](), 0, value, value * value,
                                   buffers[1](), 0, value, value * value, ConstantOne<T>(),
                                   buffers[2](), 0, value, value * value, batch_count,
                                   &queue_plain, &event);
  if (status != StatusCode::kSuccess) {
    throw RuntimeError("Gemm failed with status " + ToString(status));
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}
// =================================================================================================

template <typename T>
void TuneGemmSingleSize(const Platform& platform, const Device& device, const Context& context, Queue& queue,
                        const size_t m, const size_t n, const size_t k, const size_t num_runs) {

  // Buffers
  auto buffers = std::vector<Buffer<T>>{
      Buffer<T>(context, m * k),
      Buffer<T>(context, k * n),
      Buffer<T>(context, m * n)
  };
  const auto FunctionToTune = [&]() { RunGemmRoutineMNK(m, n, k, queue, buffers); };

  // Collects the timings for two methods
  auto scores = std::vector<TuningResult>();
  const auto methods = std::vector<std::string>{"in-direct", "direct"};
  for (auto& method: methods) {

    printf("* Testing the %s routine\n", method.c_str());
    const auto limit = (method == "in-direct") ? 0 : std::max(std::max(m, n), k) + 1; // small or large number
    ForceSelectIndirectFrom<T>(limit, device, "GemmRoutine", "XGEMM_MIN_INDIRECT_SIZE");
    auto time_ms = -1.0;
    try {
      time_ms = TimeFunction(num_runs, FunctionToTune);
      printf("  --> %9.2lf ms\n", time_ms);
    }
    catch (...) {
      const auto status_code = DispatchExceptionCatchAll(true);
      printf("  --> error %-5d\n", static_cast<int>(status_code));
    }
    auto tuning_results = Configuration();
    tuning_results["XGEMM_MIN_INDIRECT_SIZE"] = limit;
    tuning_results["PRECISION"] = static_cast<size_t>(PrecisionValue<T>());
    scores.push_back(TuningResult{"gemm_kernel_selection_single_size", time_ms, tuning_results});
  }

  // Outputs the results as JSON to disk, including some meta-data
  const auto precision_string = std::to_string(static_cast<size_t>(PrecisionValue<T>()));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
      {"kernel_family", "gemm_routine_single_size"},
      {"precision", precision_string},
      {"arg_m", ToString(m)},
      {"arg_n", ToString(n)},
      {"arg_k", ToString(k)},
  };
  PrintTimingsToFileAsJSON("clblast_gemm_routine_single_size_" + precision_string + ".json",
                           device, platform, metadata, scores);
}

// =================================================================================================

template <typename T>
void TuneXgemm(int argc, char* argv[]) {
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  const auto platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  const auto precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  const auto num_runs    = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
  const auto arg_m       = GetArgument(command_line_args, help, kArgM, -1); // optional
  const auto arg_n       = GetArgument(command_line_args, help, kArgN, -1); // optional
  const auto arg_k       = GetArgument(command_line_args, help, kArgK, -1); // optional
  fprintf(stdout, "%s\n", help.c_str());

  // OpenCL initialisation
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  if (!PrecisionSupported<T>(device)) {
    printf("* Unsupported precision, skipping this tuning run\n");
    return;
  }
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Pre-load GEMM kernel tuning results if they exist
  printf("* The GEMM routine tuner requires already tuned kernels\n");
  printf("  Applying tuning results from disk if they exist...\n\n");
  const auto kernel_names = {"xgemm_1", "xgemm_direct_1", "copy", "pad", "transpose", "padtranspose"};
  for (const auto& kernel_name : kernel_names) {
    const auto tuner_file_name = "clblast_" + std::string{kernel_name} + "_" +
                                 ToString(static_cast<int>(precision)) + ".json";
    printf("* Looking for tuning results in the current folder: '%s'\n", tuner_file_name.c_str());
    if (std::ifstream(tuner_file_name)) { // Checks if the file exists on disk
      OverrideParametersFromJSONFiles({tuner_file_name}, device(), precision);
    }
    else {
      printf("  Not found: assuming the kernel '%s' is already tuned\n\n", kernel_name);
    }
  }

  // Test for only one m/n/k size
  if (arg_m != -1 || arg_n != -1 || arg_k != -1) {
    printf("* Tuning for one specific size: m=%d, n=%d, k=%d\n", arg_m, arg_n, arg_k);
    if (arg_m == -1 || arg_n == -1 || arg_k == -1) {
      printf("* Error: If one of m/n/k specified, please specify all three\n");
      return;
    }
    TuneGemmSingleSize<T>(platform, device, context, queue, static_cast<size_t>(arg_m),
                          static_cast<size_t>(arg_n), static_cast<size_t>(arg_k), num_runs);
  }

  else {
    // Run the tuners for the XGEMM routines
    TuneKernelSelection<T>(platform, device, context, queue, precision, RunGemmRoutine<T>,
                           64, 2048, 64, 1, num_runs,
                           "gemm", "GemmRoutine", "gemm_routine", "XGEMM_MIN_INDIRECT_SIZE");
    //TuneKernelSelection<T>(platform, device, context, queue, precision, RunGemmBatchedRoutine<T, 30>,
    //                       16, 128, 32, 30, num_runs,
    //                       "gemmbatched", "GemmRoutine", "gemm_routine_2", "XGEMMBATCHED_MIN_INDIRECT_SIZE");
    //TuneKernelSelection<T>(platform, device, context, queue, precision, RunGemmStridedBatchedRoutine<T, 30>,
    //                       16, 128, 32, 30, num_runs,
    //                       "gemmstridedbatched", "GemmRoutine", "gemm_routine_3", "XGEMMSTRIDEDBATCHED_MIN_INDIRECT_SIZE");
  }

  printf("* Completed tuning process\n");
  printf("\n");
}

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::TuneXgemm<half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::TuneXgemm<float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::TuneXgemm<double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::TuneXgemm<float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::TuneXgemm<double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
