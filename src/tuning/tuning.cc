
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common auto-tuning code to interface with the CLTune library.
//
// =================================================================================================

#include <string>
#include <vector>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for vector-vector routines.
template <typename T>
void TunerXY(int argc, char* argv[], const Tuner2<T> &tune_function) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id   = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision   = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  args.n           = GetArgument(argc, argv, help, kArgN, size_t{4096*1024});
  args.alpha       = GetArgument(argc, argv, help, kArgAlpha, GetScalar<T>());
  fprintf(stdout, "%s\n", help.c_str());

  // Creates input buffers with random data
  auto x_vec = std::vector<T>(args.n);
  auto y_vec = std::vector<T>(args.n);
  PopulateVector(x_vec);
  PopulateVector(y_vec);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use full-search to explore all parameter combinations.
  tuner.UseFullSearch();

  // Configures the tuning parameters (kernel specific)
  tune_function(args, x_vec, y_vec, tuner);

  // Starts the tuning process
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s
  const auto mega_bytes = (3*args.n*GetBytes(args.precision)) * 1.0e-6;
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.1lf GB/s\n", time_ms, mega_bytes/time_ms);
  }
}

// Compiles the above function
template void TunerXY<float>(int, char**, const Tuner2<float>&);
template void TunerXY<double>(int, char**, const Tuner2<double>&);
template void TunerXY<float2>(int, char**, const Tuner2<float2>&);
template void TunerXY<double2>(int, char**, const Tuner2<double2>&);

// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for matrix-vector-vector routines.
template <typename T>
void TunerAXY(int argc, char* argv[], const Tuner3<T> &tune_function) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id   = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision   = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  args.m           = GetArgument(argc, argv, help, kArgM, size_t{1024});
  args.n           = GetArgument(argc, argv, help, kArgN, size_t{1024});
  args.alpha       = GetArgument(argc, argv, help, kArgAlpha, GetScalar<T>());
  args.beta        = GetArgument(argc, argv, help, kArgBeta, GetScalar<T>());
  args.layout      = GetArgument(argc, argv, help, kArgLayout, Layout::kColMajor);
  fprintf(stdout, "%s\n", help.c_str());

  // Creates input buffers with random data
  auto a_mat = std::vector<T>(args.m * args.n);
  auto x_vec = std::vector<T>(args.n);
  auto y_vec = std::vector<T>(args.m);
  PopulateVector(a_mat);
  PopulateVector(x_vec);
  PopulateVector(y_vec);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use full-search to explore all parameter combinations.
  tuner.UseFullSearch();

  // Configures the tuning parameters (kernel specific)
  tune_function(args, a_mat, x_vec, y_vec, tuner);

  // Starts the tuning process
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s and GFLOPS
  const auto mega_bytes = ((args.m*args.n + 2*args.m + args.n)*GetBytes(args.precision)) * 1.0e-6;
  const auto mega_flops = (2*args.m*args.n) * 1.0e-6;
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.1lf GB/s or %.1lf GFLOPS\n",
           time_ms, mega_bytes/time_ms, mega_flops/time_ms);
  }
}

// Compiles the above function
template void TunerAXY<float>(int, char**, const Tuner3<float>&);
template void TunerAXY<double>(int, char**, const Tuner3<double>&);
template void TunerAXY<float2>(int, char**, const Tuner3<float2>&);
template void TunerAXY<double2>(int, char**, const Tuner3<double2>&);

// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for matrix-matrix routines.
template <typename T>
void TunerAB(int argc, char* argv[], const Tuner2<T> &tune_function) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id   = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision   = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  args.m           = GetArgument(argc, argv, help, kArgM, size_t{1024});
  args.n           = GetArgument(argc, argv, help, kArgN, size_t{1024});
  args.fraction    = GetArgument(argc, argv, help, kArgFraction, 2048.0);
  fprintf(stdout, "%s\n", help.c_str());

  // Creates input buffers with random data
  auto a_mat = std::vector<T>(args.m * args.n);
  auto b_mat = std::vector<T>(args.m * args.n);
  PopulateVector(a_mat);
  PopulateVector(b_mat);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use full-search to explore all parameter combinations.
  tuner.UseFullSearch();

  // Configures the tuning parameters (kernel specific)
  tune_function(args, a_mat, b_mat, tuner);

  // Starts the tuning process
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s
  const auto mega_bytes = (2*args.m*args.n*GetBytes(args.precision)) * 1.0e-6;
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.1lf GB/s\n", time_ms, mega_bytes/time_ms);
  }
}

// Compiles the above function
template void TunerAB<float>(int, char**, const Tuner2<float>&);
template void TunerAB<double>(int, char**, const Tuner2<double>&);
template void TunerAB<float2>(int, char**, const Tuner2<float2>&);
template void TunerAB<double2>(int, char**, const Tuner2<double2>&);

// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for matrix-matrix-matrix routines.
template <typename T>
void TunerABC(int argc, char* argv[], const Tuner3<T> &tune_function) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id   = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision   = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  args.m           = GetArgument(argc, argv, help, kArgM, size_t{1024});
  args.n           = GetArgument(argc, argv, help, kArgN, size_t{1024});
  args.k           = GetArgument(argc, argv, help, kArgK, size_t{1024});
  args.alpha       = GetArgument(argc, argv, help, kArgAlpha, GetScalar<T>());
  args.beta        = GetArgument(argc, argv, help, kArgBeta, GetScalar<T>());
  args.fraction    = GetArgument(argc, argv, help, kArgFraction, 2048.0);
  fprintf(stdout, "%s\n", help.c_str());

  // Creates input buffers with random data
  auto a_mat = std::vector<T>(args.m * args.k);
  auto b_mat = std::vector<T>(args.n * args.k);
  auto c_mat = std::vector<T>(args.m * args.n);
  PopulateVector(a_mat);
  PopulateVector(b_mat);
  PopulateVector(c_mat);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use random-search to search only a part of the parameter values. The fraction of the search-
  // space to explore is set as a command-line argument.
  tuner.UseRandomSearch(1.0/args.fraction);

  // Configures the tuning parameters (kernel specific)
  tune_function(args, a_mat, b_mat, c_mat, tuner);

  // Starts the tuning process
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GFLOPS
  const auto mega_flops = (2*args.m*args.n*args.k) * 1.0e-6;
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.1lf GFLOPS\n", time_ms, mega_flops/time_ms);
  }
}

// Compiles the above function
template void TunerABC<float>(int, char**, const Tuner3<float>&);
template void TunerABC<double>(int, char**, const Tuner3<double>&);
template void TunerABC<float2>(int, char**, const Tuner3<float2>&);
template void TunerABC<double2>(int, char**, const Tuner3<double2>&);

// =================================================================================================
} // namespace clblast
