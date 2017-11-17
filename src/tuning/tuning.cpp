
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the generic CLBlast auto-tuner (inspired by CLTune). This is only used for
//  the optional and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#include <vector>
#include <string>
#include <random>
#include <utility>
#include <algorithm>
#include <iostream>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

void PrintTimingsToFileAsJSON(const std::string &filename,
                              const Device& device, const Platform& platform,
                              const std::vector<std::pair<std::string,std::string>> &metadata,
                              const std::vector<TuningResult>& tuning_results) {
  auto num_results = tuning_results.size();
  printf("* Writing a total of %zu results to '%s'\n", num_results, filename.c_str());

  auto file = fopen(filename.c_str(), "w");
  fprintf(file, "{\n");
  for (auto &datum: metadata) {
    fprintf(file, "  \"%s\": \"%s\",\n", datum.first.c_str(), datum.second.c_str());
  }
  fprintf(file, "  \"platform_version\": \"%s\",\n", platform.Version().c_str());
  fprintf(file, "  \"clblast_device_name\": \"%s\",\n", GetDeviceName(device).c_str());
  fprintf(file, "  \"clblast_device_vendor\": \"%s\",\n", platform.Vendor().c_str());
  fprintf(file, "  \"clblast_device_type\": \"%s\",\n", device.Type().c_str());
  fprintf(file, "  \"clblast_device_architecture\": \"%s\",\n", GetDeviceArchitecture(device).c_str());
  fprintf(file, "  \"device_core_clock\": \"%zu\",\n", device.CoreClock());
  fprintf(file, "  \"device_compute_units\": \"%zu\",\n", device.ComputeUnits());
  fprintf(file, "  \"results\": [\n");

  // Loops over all results
  for (auto r = size_t{0}; r < num_results; ++r) {
    auto result = tuning_results[r];
    fprintf(file, "    {\n");
    fprintf(file, "      \"kernel\": \"%s\",\n", result.name.c_str());
    fprintf(file, "      \"time\": %.3lf,\n", result.score);

    // Loops over all the parameters for this result
    fprintf(file, "      \"parameters\": {");
    auto num_configs = result.config.size();
    auto p = size_t{0};
    for (const auto parameter : result.config) {
      fprintf(file, "\"%s\": %zu", parameter.first.c_str(), parameter.second);
      if (p < num_configs -1 ) { fprintf(file, ","); }
      ++p;
    }
    fprintf(file, "}\n");

    // The footer
    fprintf(file, "    }");
    if (r < num_results - 1) { fprintf(file, ","); }
    fprintf(file, "\n");
  }
  fprintf(file, "  ]\n");
  fprintf(file, "}\n");
  fclose(file);
}

void print_separator(const size_t parameters_size) {
  printf("x------x-------x");
  for (auto i = size_t{0}; i < parameters_size; ++i) { printf("-----"); }
  printf("-x----------x------------x--------x-------------------x\n");
}

// =================================================================================================
} // namespace clblast
