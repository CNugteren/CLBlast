
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the interface to the CLTune auto-tuner. This is only used for the optional
// and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_H_
#define CLBLAST_TUNING_H_

#include <vector>
#include <string>

#include <cltune.h>

namespace clblast {
// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for all types of kernel families. Note that this is a header-only function so
// that it is automatically compiled for the various kernels (given as the 'C' template argument).
template <typename C, typename T>
void Tuner(int argc, char* argv[]) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id   = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision   = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  for (auto &o: C::GetOptions()) {
    if (o == kArgM)        { args.m        = GetArgument(argc, argv, help, kArgM, C::DefaultM()); }
    if (o == kArgN)        { args.n        = GetArgument(argc, argv, help, kArgN, C::DefaultN()); }
    if (o == kArgK)        { args.k        = GetArgument(argc, argv, help, kArgK, C::DefaultK()); }
    if (o == kArgAlpha)    { args.alpha    = GetArgument(argc, argv, help, kArgAlpha, GetScalar<T>()); }
    if (o == kArgBeta)     { args.beta     = GetArgument(argc, argv, help, kArgBeta, GetScalar<T>()); }
    if (o == kArgFraction) { args.fraction = GetArgument(argc, argv, help, kArgFraction, C::DefaultFraction()); }
  }
  fprintf(stdout, "%s\n", help.c_str());

  // Tests validity of the given arguments
  C::TestValidArguments(args);

  // Tests for validity of the precision
  {
    auto platform = Platform(args.platform_id);
    auto device = Device(platform, args.device_id);
    if (!PrecisionSupported<T>(device)) {
      printf("* Unsupported precision, skipping this tuning run\n\n");
      return;
    }
  }

  // Creates input buffers with random data
  auto x_vec = std::vector<T>(C::GetSizeX(args));
  auto y_vec = std::vector<T>(C::GetSizeY(args));
  auto a_mat = std::vector<T>(C::GetSizeA(args));
  auto b_mat = std::vector<T>(C::GetSizeB(args));
  auto c_mat = std::vector<T>(C::GetSizeC(args));
  auto temp = std::vector<T>(C::GetSizeTemp(args));
  PopulateVector(x_vec);
  PopulateVector(y_vec);
  PopulateVector(a_mat);
  PopulateVector(b_mat);
  PopulateVector(c_mat);
  PopulateVector(temp);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use full-search to explore all parameter combinations or random-search to search only a part of
  // the parameter values. The fraction is set as a command-line argument.
  if (args.fraction == 1.0 || args.fraction == 0.0) {
    tuner.UseFullSearch();
  }
  else {
    tuner.UseRandomSearch(1.0/args.fraction);
  }

  // Loads the kernel sources and defines the kernel to tune
  auto sources = C::GetSources();
  auto id = tuner.AddKernelFromString(sources, C::KernelName(), C::GlobalSize(args), C::LocalSize());
  tuner.SetReferenceFromString(sources, C::KernelName(), C::GlobalSizeRef(args), C::LocalSizeRef());

  // Sets the tunable parameters and their possible values
  C::SetParameters(tuner, id);
  C::SetConstraints(tuner, id);
  C::SetLocalMemorySize(tuner, id, args);

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (both global and local) based on the parameters
  for (auto &parameters: C::MulLocal()) { tuner.MulLocalSize(id, parameters); }
  for (auto &parameters: C::DivLocal()) { tuner.DivLocalSize(id, parameters); }
  for (auto &parameters: C::MulGlobal()) { tuner.MulGlobalSize(id, parameters); }
  for (auto &parameters: C::DivGlobal()) { tuner.DivGlobalSize(id, parameters); }

  // Sets the function's arguments
  C::SetArguments(tuner, args, x_vec, y_vec, a_mat, b_mat, c_mat, temp);

  // Starts the tuning process
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s or GFLOPS
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms", time_ms);
    printf(" or %.1lf %s\n", C::GetMetric(args)/(time_ms*1.0e6), C::PerformanceUnit().c_str());
  }

  // Outputs the results as JSON to disk, including some meta-data
  auto precision_string = std::to_string(static_cast<size_t>(args.precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", C::KernelFamily()},
    {"precision", precision_string}
  };
  for (auto &o: C::GetOptions()) {
    if (o == kArgM) { metadata.push_back({"arg_m", std::to_string(args.m)}); }
    if (o == kArgN) { metadata.push_back({"arg_n", std::to_string(args.n)}); }
    if (o == kArgK) { metadata.push_back({"arg_k", std::to_string(args.k)}); }
  }
  tuner.PrintJSON("clblast_"+C::KernelFamily()+"_"+precision_string+".json", metadata);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_H_
#endif
