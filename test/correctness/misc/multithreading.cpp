
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains tests to verify thread safety of the library, even when running on multiple
// platforms and devices.
//
// =================================================================================================

#include "utilities/utilities.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "test/routines/level3/xgemm.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <future>
#include <random>

namespace clblast {
// =================================================================================================

template <typename T>
std::vector<StatusCode> RunExampleRoutine(const Arguments<T> args, const bool verbose,
                                          const size_t platform_id, const size_t device_id,
                                          const int num_runs) {
  auto example_routine = TestXgemm<T>();
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Initializes OpenCL
  if (verbose) { printf("* Running on platform %zu, device %zu\n", platform_id, device_id); }
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  if (verbose) { printf("  (device %s of vendor %s)\n", device.Name().c_str(), device.Vendor().c_str()); }
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Populate host matrices with some example data
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  auto host_a = std::vector<T>(args.m * args.k);
  auto host_b = std::vector<T>(args.n * args.k);
  auto host_c = std::vector<T>(args.m * args.n);
  PopulateVector(host_a, mt, dist);
  PopulateVector(host_b, mt, dist);
  PopulateVector(host_c, mt, dist);

  // Copy the matrices to the device
  auto device_a = Buffer<T>(context, host_a.size());
  auto device_b = Buffer<T>(context, host_b.size());
  auto device_c = Buffer<T>(context, host_c.size());
  device_a.Write(queue, host_a.size(), host_a);
  device_b.Write(queue, host_b.size(), host_b);
  device_c.Write(queue, host_c.size(), host_c);
  auto dummy = Buffer<T>(context, 1);
  auto buffers = Buffers<T>{dummy, dummy, device_a, device_b, device_c, dummy, dummy};

  // Runs a number of times
  auto statuses = std::vector<StatusCode>();
  for (auto t = 0; t < num_runs; ++t) {
    statuses.push_back(example_routine.RunRoutine(args, buffers, queue));
  }

  return statuses;
}

// =================================================================================================

template <typename T>
size_t RunMultithreadingTests(int argc, char *argv[], const bool silent, const std::string &routine_name) {
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  auto passed = size_t{0};
  auto skipped = size_t{0};
  auto errors = size_t{0};
  constexpr auto kNumRunsOuterLoop = 3;
  constexpr auto kNumRunsInnerLoop = 5;

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  auto args = Arguments<T>{};
  args.m = GetArgument(arguments, help, kArgM, size_t{128});
  args.n = GetArgument(arguments, help, kArgN, size_t{128});
  args.k = GetArgument(arguments, help, kArgK, size_t{128});
  args.a_ld = GetArgument(arguments, help, kArgALeadDim, args.k);
  args.b_ld = GetArgument(arguments, help, kArgBLeadDim, args.n);
  args.c_ld = GetArgument(arguments, help, kArgCLeadDim, args.n);
  args.a_offset = GetArgument(arguments, help, kArgAOffset, size_t{0});
  args.b_offset = GetArgument(arguments, help, kArgBOffset, size_t{0});
  args.c_offset = GetArgument(arguments, help, kArgCOffset, size_t{0});
  args.layout = GetArgument(arguments, help, kArgLayout, Layout::kRowMajor);
  args.a_transpose = GetArgument(arguments, help, kArgATransp, Transpose::kNo);
  args.b_transpose = GetArgument(arguments, help, kArgBTransp, Transpose::kNo);
  args.alpha = GetArgument(arguments, help, kArgAlpha, GetScalar<T>());
  args.beta  = GetArgument(arguments, help, kArgBeta, GetScalar<T>());
  const auto verbose = CheckArgument(arguments, help, kArgVerbose);

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Counts the number of platforms and devices
  const auto platforms = GetAllPlatforms();
  auto num_devices = size_t{0};
  for (const auto &platform : platforms) {
    num_devices += platform.NumDevices();
  }
  fprintf(stdout, "* Testing multithreading on %zu platform(s) with a total of %zu device(s) for '%s'\n",
          platforms.size(), num_devices, routine_name.c_str());

  // Runs an example routine asynchronously multiple times on each platform and device in the system
  auto futures = std::vector<std::future<std::vector<StatusCode>>>();
  for (auto platform_id = size_t{0}; platform_id < platforms.size(); ++platform_id) {
    for (auto device_id = size_t{0}; device_id < platforms[platform_id].NumDevices(); ++device_id) {
      for (auto run_id = 0; run_id < kNumRunsOuterLoop; ++run_id) {
        futures.push_back(std::async(std::launch::async, RunExampleRoutine<T>,
                                     args, verbose, platform_id, device_id, kNumRunsInnerLoop));
      }
    }
  }

  // Waits for everything to complete
  for (auto &future : futures) {
    const auto statuses = future.get();
    const auto num_passed = std::count(statuses.begin(), statuses.end(), StatusCode::kSuccess);
    const auto num_skipped = std::count(statuses.begin(), statuses.end(), StatusCode::kInvalidLocalThreadsDim); // CPU with Apple OpenCL
    passed += num_passed;
    skipped += num_skipped;
    errors += (kNumRunsInnerLoop - num_skipped - num_passed);
  }

  // Prints and returns the statistics
  fprintf(stdout, "    %zu test(s) passed\n", passed);
  fprintf(stdout, "    %zu test(s) skipped\n", skipped);
  fprintf(stdout, "    %zu test(s) failed\n", errors);
  fprintf(stdout, "\n");
  return errors;
}

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunMultithreadingTests<float>(argc, argv, false, "SGEMM");
  errors += clblast::RunMultithreadingTests<float2>(argc, argv, true, "CGEMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
