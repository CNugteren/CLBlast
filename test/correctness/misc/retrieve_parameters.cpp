
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the tests for the RetrieveParameters function
//
// =================================================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
size_t RunRetrieveParametersTests(int argc, char *argv[], const bool silent, const std::string &routine_name) {
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  auto errors = size_t{0};
  auto passed = size_t{0};

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  auto args = Arguments<T>{};

  // Determines the test settings
  const auto kernel_name = std::string{"Xgemm"};
  const auto expected_parameters = std::vector<std::string>{
    "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"
  };
  const auto expected_max_value = size_t{16384};

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);

  // Retrieves the parameters
  fprintf(stdout, "* Testing RetrieveParameters for '%s'\n", routine_name.c_str());
  auto parameters = std::unordered_map<std::string,size_t>();
  const auto status = RetrieveParameters(device(), kernel_name, PrecisionValue<T>(), parameters);
  if (status != StatusCode::kSuccess) { errors++; }

  // Verifies the parameters
  for (const auto &expected_parameter : expected_parameters) {
    if (parameters.find(expected_parameter) != parameters.end()) {
      const auto value = parameters[expected_parameter];
      if (value < expected_max_value) { passed++; } else { errors++; }
      //std::cout << expected_parameter << " = " << value << std::endl;
    }
    else { errors++; }
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
  errors += clblast::RunRetrieveParametersTests<float>(argc, argv, false, "SGEMM");
  errors += clblast::RunRetrieveParametersTests<clblast::float2>(argc, argv, true, "CGEMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
