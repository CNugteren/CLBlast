
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

  // Run the tuners for the XGEMM routine
  const auto scores = TuneKernelSelection<T>(device, context, queue, precision, RunGemmRoutine<T>,
                                             num_runs, "Gemm", "XGEMM_MIN_INDIRECT_SIZE");
  const auto xgemm_best = GetBestResult(scores);
  const auto xgemm_switching_point = xgemm_best.config.at("XGEMM_MIN_INDIRECT_SIZE");
  const auto xgemm_string = "XGEMM_MIN_INDIRECT_SIZE=" + ToString(xgemm_switching_point);

  // Outputs the results as JSON to disk, including some meta-data
  const auto precision_string = std::to_string(static_cast<size_t>(precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
      {"kernel_family", "gemm_routine"},
      {"precision", precision_string},
      {"best_kernel", xgemm_best.name},
      {"best_time", ToString(xgemm_best.score)},
      {"best_parameters", xgemm_string}
  };
  PrintTimingsToFileAsJSON("clblast_routine_gemm_" + precision_string + ".json",
                           device, platform, metadata, scores);

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
