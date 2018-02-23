
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the invert OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults GetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgN, kArgM, kArgK};
  settings.default_n = 128; // dimension of input matrix 'n'
  settings.default_m = 64; // block size
  settings.default_k = 16; // current size
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings GetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "invert";
  settings.kernel_name = "TripleMatMul16Part1Lower";
  settings.sources =
"#define ROUTINE_INVERT"
#include "../src/kernels/level3/invert_diagonal_blocks_part1.opencl"
#include "../src/kernels/level3/invert_diagonal_blocks_part2.opencl"
  ;

  // Buffer sizes
  settings.size_a = args.n * args.n + args.a_offset;
  settings.size_b = Ceil(args.n, args.m) * args.m; // Ceil(n, block_size) * block_size

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {2, 3};
  settings.outputs = {3};

  // Sets the base thread configuration
  const auto num_pages = CeilDiv(args.n, args.k * 2); // CeilDiv(n, current_size*2)
  settings.global_size = {args.k / 4, num_pages * (args.k / 16) * 4};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1, 1};
  settings.local_size_ref = {4, 4};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"TMMWGSX", "TMMWGSY"}};
  settings.div_global = {{}};

  // Sets the tuning parameters and their possible values
  // TODO: Make these actually tunable, apart from LOCALPAD
  settings.parameters = {
    {"INTERNAL_BLOCK_SIZE", {16}},
    {"LOCALPAD", {0, 1}},
    {"TMMWGSX", {4}},
    {"TMMWGSY", {4}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = 1 * GetBytes(args.precision);
  settings.performance_unit = "N/A";

  return settings;
}

// Tests for valid arguments
template <typename T>
void TestValidArguments(const int, const Arguments<T> &args) {
  if (!(args.k == 16)) {
    throw std::runtime_error("'TripleMatMul16Part1Lower' requires 'k' to be 16");
  }
}
std::vector<Constraint> SetConstraints(const int) { return {}; }

// Sets the kernel's arguments
template <typename T>
void SetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  const auto num_pages = CeilDiv(args.n, args.k * 2); // CeilDiv(n, current_size*2)
  kernel.SetArgument(0, static_cast<int>(args.n)); // n
  kernel.SetArgument(1, buffers[2]()); // 2 == A matrix
  kernel.SetArgument(2, 0); // a_offset
  kernel.SetArgument(3, static_cast<int>(args.n)); // a_ld
  kernel.SetArgument(4, buffers[3]()); // 3 == B matrix
  kernel.SetArgument(5, static_cast<int>(args.k)); // current_size
  kernel.SetArgument(6, static_cast<int>(num_pages)); // num_pages
  kernel.SetArgument(7, static_cast<int>(args.m)); // block_size
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
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<half>, clblast::TestValidArguments<half>, clblast::SetConstraints, clblast::SetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float>, clblast::TestValidArguments<float>, clblast::SetConstraints, clblast::SetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double>, clblast::TestValidArguments<double>, clblast::SetConstraints, clblast::SetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float2>, clblast::TestValidArguments<float2>, clblast::SetConstraints, clblast::SetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double2>, clblast::TestValidArguments<double2>, clblast::SetConstraints, clblast::SetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
