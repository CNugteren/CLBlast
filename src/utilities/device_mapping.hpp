
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file describes the mappings of extracted names from OpenCL (device, board, vendor, etc.) to
// more commonly used names to match devices from different vendors and platforms properly.
//
// =================================================================================================

#ifndef CLBLAST_UTILITIES_DEVICE_MAPPING_H_
#define CLBLAST_UTILITIES_DEVICE_MAPPING_H_

#include <string>
#include <unordered_map>

namespace clblast {
// A special namespace to hold all the global constant variables
namespace device_mapping {

// =================================================================================================

// Alternative names for some vendor names (top-level)
const std::unordered_map<std::string, std::string> kVendorNames {
  { "Intel(R) Corporation", "Intel" },
  { "GenuineIntel", "Intel" },
  { "Advanced Micro Devices, Inc.", "AMD" },
  { "NVIDIA Corporation", "NVIDIA" },
};

// Alternative names for some architectures (mid-level)
const std::unordered_map<std::string, std::string> kArchitectureNames {
  {"gfx803", "Fiji"},
  {"gfx900", "Vega"},
};

// Alternative names for some devices (low-level)
const std::unordered_map<std::string, std::string> kDeviceNames {
  // Empty
};

// Things to remove from device names (low-level)
const std::vector<std::string> kDeviceRemovals {
  "pthread-"
};

// =================================================================================================
} // namespace device_mapping
} // namespace clblast

// CLBLAST_UTILITIES_DEVICE_MAPPING_H_
#endif
