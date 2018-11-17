
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the tests for the OverrideParameters function
//
// =================================================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>

#include "utilities/utilities.hpp"
#include "test/routines/level3/xgemm.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
size_t RunOverrideTests(int argc, char *argv[], const bool silent, const std::string &routine_name) {
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  auto errors = size_t{0};
  auto passed = size_t{0};
  auto example_routine = TestXgemm<0, T>();
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Determines the test settings
  const auto kernel_name = std::string{"Xgemm"};
  const auto precision = PrecisionValue<T>();
  const auto valid_settings = std::vector<std::unordered_map<std::string,size_t>>{
    { {"GEMMK",0}, {"KREG",1}, {"KWG",16}, {"KWI",2}, {"MDIMA",4}, {"MDIMC",4}, {"MWG",16}, {"NDIMB",4}, {"NDIMC",4}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} },
    { {"GEMMK",0}, {"KREG",1}, {"KWG",32}, {"KWI",2}, {"MDIMA",4}, {"MDIMC",4}, {"MWG",32}, {"NDIMB",4}, {"NDIMC",4}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} },
    { {"GEMMK",0}, {"KREG",1}, {"KWG",16}, {"KWI",2}, {"MDIMA",4}, {"MDIMC",4}, {"MWG",16}, {"NDIMB",4}, {"NDIMC",4}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} },
  };
  const auto invalid_settings = std::vector<std::unordered_map<std::string,size_t>>{
    { {"GEMMK",0}, {"KREG",1}, {"KWI",2}, {"MDIMA",4}, {"MDIMC",4}, {"MWG",16}, {"NDIMB",4}, {"NDIMC",4}, {"NWG",16}, {"SA",0} },
  };

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  auto args = Arguments<T>{};
  args.m = GetArgument(arguments, help, kArgM, size_t{256});
  args.n = GetArgument(arguments, help, kArgN, size_t{256});
  args.k = GetArgument(arguments, help, kArgK, size_t{256});
  args.a_ld = GetArgument(arguments, help, kArgALeadDim, args.k);
  args.b_ld = GetArgument(arguments, help, kArgBLeadDim, args.n);
  args.c_ld = GetArgument(arguments, help, kArgCLeadDim, args.n);
  args.a_offset = GetArgument(arguments, help, kArgAOffset, size_t{0});
  args.b_offset = GetArgument(arguments, help, kArgBOffset, size_t{0});
  args.c_offset = GetArgument(arguments, help, kArgCOffset, size_t{0});
  args.layout = GetArgument(arguments, help, kArgLayout, Layout::kRowMajor);
  args.a_transpose = GetArgument(arguments, help, kArgATransp, Transpose::kNo);
  args.b_transpose = GetArgument(arguments, help, kArgBTransp, Transpose::kNo);
  args.kernel_mode = GetArgument(arguments, help, kArgKernelMode, KernelMode::kCrossCorrelation);
  args.alpha = GetArgument(arguments, help, kArgAlpha, GetScalar<T>());
  args.beta  = GetArgument(arguments, help, kArgBeta, GetScalar<T>());

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Populate host matrices with some example data
  auto host_a = std::vector<T>(args.m * args.k);
  auto host_b = std::vector<T>(args.n * args.k);
  auto host_c = std::vector<T>(args.m * args.n);
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  PopulateVector(host_a, mt, dist);
  PopulateVector(host_b, mt, dist);
  PopulateVector(host_c, mt, dist);

  // Copy the matrices to the device
  auto device_a = Buffer<T>(context, host_a.size());
  auto device_b = Buffer<T>(context, host_b.size());
  auto device_c = Buffer<T>(context, host_c.size());
  auto device_temp = Buffer<T>(context, args.m * args.n * args.k); // just to be safe
  device_a.Write(queue, host_a.size(), host_a);
  device_b.Write(queue, host_b.size(), host_b);
  device_c.Write(queue, host_c.size(), host_c);
  auto dummy = Buffer<T>(context, 1);
  auto buffers = Buffers<T>{dummy, dummy, device_a, device_b, device_c, device_temp, dummy};

  // Loops over the valid combinations: run before and run afterwards
  fprintf(stdout, "* Testing OverrideParameters for '%s'\n", routine_name.c_str());
  for (const auto &override_setting : valid_settings) {
    const auto status_before = example_routine.RunRoutine(args, buffers, queue);
    if (status_before != StatusCode::kSuccess) { errors++; continue; }

    // Overrides the parameters
    const auto status = OverrideParameters(device(), kernel_name, precision, override_setting);
    if (status != StatusCode::kSuccess) { errors++; continue; } // error shouldn't occur

    const auto status_after = example_routine.RunRoutine(args, buffers, queue);
    if (status_after != StatusCode::kSuccess) { errors++; continue; }
    passed++;
  }

  // Loops over the invalid combinations: run before and run afterwards
  for (const auto &override_setting : invalid_settings) {
    const auto status_before = example_routine.RunRoutine(args, buffers, queue);
    if (status_before != StatusCode::kSuccess) { errors++; continue; }

    // Overrides the parameters
    const auto status = OverrideParameters(device(), kernel_name, precision, override_setting);
    if (status == StatusCode::kSuccess) { errors++; continue; } // error should occur

    const auto status_after = example_routine.RunRoutine(args, buffers, queue);
    if (status_after != StatusCode::kSuccess) { errors++; continue; }
    passed++;
  }

  // Prints and returns the statistics
  std::cout << "    " << passed << " test(s) passed" << std::endl;
  std::cout << "    " << errors << " test(s) failed" << std::endl;
  std::cout << std::endl;
  return errors;
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunOverrideTests<float>(argc, argv, false, "SGEMM");
  errors += clblast::RunOverrideTests<clblast::float2>(argc, argv, true, "CGEMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
