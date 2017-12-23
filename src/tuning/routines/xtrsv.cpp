
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tunes the Xtrsv routine at a high-level: choosing an appropriate block size
//
// =================================================================================================

#include <exception>
#include <string>
#include <vector>
#include <limits>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

constexpr auto size = size_t{1024}; // 'n' argument

template <typename T>
void SetBlockSize(const size_t value, const Device &device) {
  const auto override_status = OverrideParameters(device(), "TrsvRoutine", PrecisionValue<T>(),
                                                  {{"TRSV_BLOCK_SIZE", value}});
  if (override_status != StatusCode::kSuccess) {
    throw RuntimeError("OverrideParameters failed with status " + ToString(override_status));
  }
}

template <typename T>
void RunTrsvRoutine(const size_t block_size, Queue& queue, const std::vector<Buffer<T>>& buffers) {
  SetBlockSize<T>(block_size, queue.GetDevice());
  auto queue_plain = queue();
  auto event = cl_event{};
  auto status = Trsv<T>(Layout::kRowMajor, Triangle::kLower, Transpose::kNo, Diagonal::kNonUnit,
                        size,
                        buffers[0](), 0, size, // A matrix
                        buffers[1](), 0, 1, // X vector
                        &queue_plain, &event);
  if (status != StatusCode::kSuccess) {
    throw RuntimeError("Trsv failed with status " + ToString(status));
  }
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}

template <typename T>
void TuneXtrsv(int argc, char* argv[]) {
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  const auto platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  const auto precision = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  const auto num_runs = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
  fprintf(stdout, "%s\n", help.c_str());

  // Values for the block size
  const auto from = size_t{8};
  const auto to = size_t{32 + 1};
  const auto step = size_t{8};

  // OpenCL initialisation
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  if (!PrecisionSupported<T>(device)) {
    printf("* Unsupported precision, skipping this tuning run\n");
    return;
  }
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Buffers
  auto buffers = std::vector<Buffer<T>>{
    Buffer<T>(context, size * size),
    Buffer<T>(context, size)
  };

  // Performance testing
  const auto results = TimeRoutine(from, to, step, num_runs, queue, buffers, RunTrsvRoutine<T>);

  // Stores the results in the expected format
  auto scores = std::vector<TuningResult>();
  for (const auto &result : results) {
    if (result.second != -1) {
      auto tuning_results = Configuration();
      tuning_results["TRSV_BLOCK_SIZE"] = result.first;
      tuning_results["PRECISION"] = static_cast<size_t>(precision);
      scores.emplace_back(TuningResult{"trsv_routine", result.second, tuning_results});
    }
  }

  // Computes the best result
  auto best_time = std::numeric_limits<double>::max();
  auto best_value = size_t{0};
  for (const auto &result : results) {
    if (result.second != -1 && result.second < best_time) {
      best_time = result.second;
      best_value = result.first;
    }
  }
  const auto best_string = "TRSV_BLOCK_SIZE=" + ToString(best_value);

  // Outputs the results as JSON to disk, including some meta-data
  const auto precision_string = std::to_string(static_cast<size_t>(precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", "trsv_routine"},
    {"precision", precision_string},
    {"arg_n", ToString(size)},
    {"best_kernel", "trsv_routine"},
    {"best_time", ToString(best_time)},
    {"best_parameters", best_string}
  };
  PrintTimingsToFileAsJSON("clblast_routine_xtrsv_" + precision_string + ".json",
                           device, platform, metadata, scores);

  printf("* Completed tuning process\n");
  printf("\n");
}

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kSingle: clblast::TuneXtrsv<float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::TuneXtrsv<double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::TuneXtrsv<float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::TuneXtrsv<double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
