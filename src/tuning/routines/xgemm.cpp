
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
#include <assert.h>

#include "utilities/utilities.hpp"
#include "utilities/timing.hpp"

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

template <typename T>
void ForceSelectIndirectFrom(const size_t minimum_size, const Device &device) {
  const auto override_status = OverrideParameters(device(), "GemmRoutine", PrecisionValue<T>(),
                                                  {{"XGEMM_MIN_INDIRECT_SIZE", minimum_size}});
  if (override_status != StatusCode::kSuccess) {
    throw RuntimeError("OverrideParameters failed with status " + ToString(override_status));
  }
}

template <typename T>
void TuneXgemm(int argc, char* argv[]) {
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  const auto platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  const auto precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  const auto num_runs    = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
  fprintf(stdout, "%s\n", help.c_str());

  // Values for m, n, and k
  const auto from = size_t{64};
  const auto to = size_t{2048};
  const auto step = size_t{64};

  // OpenCL initialisation
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  if (!PrecisionSupported<T>(device)) {
    printf("* Unsupported precision, skipping this tuning run\n\n");
    return;
  }
  const auto context = Context(device);
  const auto queue = Queue(context, device);

  // Buffers
  auto a_mat = Buffer<T>(context, to * to);
  auto b_mat = Buffer<T>(context, to * to);
  auto c_mat = Buffer<T>(context, to * to);
  auto buffers = std::vector<Buffer<T>>{a_mat, b_mat, c_mat};

  // In-direct version
  printf("[----------] Testing the in-direct GEMM routine for m=n=k\n");
  ForceSelectIndirectFrom<T>(0, device);
  const auto indirect = TimeRoutine(from, to, step, num_runs, queue, buffers, RunGemmRoutine<T>);

  // Direct version
  printf("[----------] Testing the direct GEMM routine for m=n=k\n");
  ForceSelectIndirectFrom<T>(to * to * to + 1, device);
  const auto direct = TimeRoutine(from, to, step, num_runs, queue, buffers, RunGemmRoutine<T>);

  // Determining final score and best kernel selection point
  assert(indirect.size() == direct.size());
  printf("[----------] Collecting results\n");
  auto ratios = std::vector<double>(indirect.size());
  for (auto i = size_t{0}; i < indirect.size(); ++i) {
    ratios[i] = indirect[i].second / direct[i].second;
  }
  auto scores = std::vector<TuningResult>(ratios.size());
  for (auto i = size_t{0}; i < scores.size(); ++i) {
    auto score = 0;
    for (auto j = size_t{0}; j < i; ++j) { score += (ratios[j] <= 1.0); }
    for (auto j = i + 1; j < ratios.size(); ++j) { score += (ratios[j] > 1.0); }
    const auto epsilon = (scores.size() - i) / 1e3; // favour later results over earlier ones
    const auto relative_score = static_cast<double>(score) / static_cast<double>(scores.size() - 1);
    scores[i] = TuningResult{
        "gemm_kernel_selection",
        (relative_score * relative_score) * 100 + epsilon,  // squared for proper default computation
        TuningParameters{
            TuningParameter{"XGEMM_MIN_INDIRECT_SIZE", indirect[i].first},
            TuningParameter{"PRECISION", static_cast<size_t>(precision)}
        }
    };
  }

  // Displaying results
  printf("[ -------> ]   value indirect   direct    score (lowest means best switching point)\n");
  for (auto i = size_t{0}; i < indirect.size(); ++i) {
    assert(indirect[i].first == direct[i].first);
    const auto value = indirect[i].first;
    if (indirect[i].second != -1 && direct[i].second != -1) {
      const auto gflops_indirect = (2 * value * value * value) / (indirect[i].second * 1.0e6);
      const auto gflops_direct = (2 * value * value * value) / (direct[i].second * 1.0e6);
      printf("[ -------> ] %7zu %8.2lf %8.2lf %8.2lf\n",
             value, gflops_indirect, gflops_direct, scores[i].score);
    }
  }

  // Outputs the results as JSON to disk, including some meta-data
  const auto precision_string = std::to_string(static_cast<size_t>(precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
      {"kernel_family", "gemm_routine"},
      {"arg_from", ToString(from)},
      {"arg_to", ToString(to)},
      {"arg_step", ToString(step)},
      {"precision", precision_string},
  };
  PrintTimingsToFileAsJSON("clblast_routine_gemm_" + precision_string + ".json",
                           device, platform, metadata, scores);

  printf("[  STATUS  ] All done\n");
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
