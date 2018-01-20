
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

#include "utilities/utilities.hpp"
#include "tuning/routines/routine_tuner.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
void RunGemmRoutine(const size_t value, const Queue& queue, const std::vector<Buffer<T>>& buffers) {
  auto queue_plain = queue();
  auto event = cl_event{};
  auto status = Gemm(Layout::kRowMajor, Transpose::kNo, Transpose::kNo,
                     value, value, value, ConstantOne<T>(),
                     buffers[0](), 0, value,
                     buffers[1](), 0, value, ConstantOne<T>(),
                     buffers[2](), 0, value,
                     &queue_plain, &event);
  if (status != StatusCode::kSuccess) {
    throw RuntimeError("Gemm failed with status " + ToString(status));
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
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
void TuneXgemm(int argc, char* argv[]) {
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  const auto platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  const auto precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  const auto num_runs    = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
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
