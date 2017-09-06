
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file describes the database storage structures.
//
// =================================================================================================

#ifndef CLBLAST_DATABASE_STRUCTURE_H_
#define CLBLAST_DATABASE_STRUCTURE_H_

#include <string>
#include <vector>
#include <unordered_map>

namespace clblast {
// A special namespace to hold all the global constant variables (including the database entries)
namespace database {

// =================================================================================================

// The OpenCL device types
const std::string kDeviceTypeCPU = "CPU";
const std::string kDeviceTypeGPU = "GPU";
const std::string kDeviceTypeAccelerator = "accelerator";
const std::string kDeviceTypeAll = "default";

// Type alias for the database parameters
using Parameters = std::unordered_map<std::string, size_t>;

// Structures for content inside the database
struct DatabaseDevice {
  std::string name;
  std::vector<size_t> parameters; // parameter values
};
struct DatabaseVendor {
  std::string type;
  std::string name;
  std::vector<DatabaseDevice> devices;
};
struct DatabaseEntry {
  std::string kernel;
  Precision precision;
  std::vector<std::string> parameter_names;
  std::vector<DatabaseVendor> vendors;
};

// =================================================================================================
} // namespace database
} // namespace clblast

// CLBLAST_DATABASE_STRUCTURE_H_
#endif
