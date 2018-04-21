
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

#ifndef CLBLAST_DATABASE_DATABASE_STRUCTURE_H_
#define CLBLAST_DATABASE_DATABASE_STRUCTURE_H_

#include <string>
#include <array>
#include <vector>
#include <map>

// Just needed for 'Precision'
#ifdef OPENCL_API
  #include "clblast.h"
#elif CUDA_API
  #include "clblast_cuda.h"
#endif

namespace clblast {
// A special namespace to hold all the global constant variables (including the database entries)
namespace database {

// =================================================================================================

// Type alias for the database storage (arrays for fast compilation/efficiency)
using Name = std::array<char, 51>; // name as stored in database (50 chars + string terminator)
using Params = std::array<size_t, 16>; // parameters as stored in database

// Type alias after extracting from the database (sorted map for improved code readability)
using Parameters = std::map<std::string, size_t>; // parameters after reading from DB

// The OpenCL device types
const std::string kDeviceTypeCPU = "CPU";
const std::string kDeviceTypeGPU = "GPU";
const std::string kDeviceTypeAccelerator = "accelerator";
const std::string kDeviceTypeAll = "default";
const Name kDeviceNameDefault = {"default                                           "};

struct DatabaseDevice {
  Name name;
  Params parameters; // parameter values

};
struct DatabaseArchitecture {
  std::string name;
  std::vector<DatabaseDevice> devices;
};
struct DatabaseVendor {
  std::string type;
  std::string name;
  std::vector<DatabaseArchitecture> architectures;
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

// CLBLAST_DATABASE_DATABASE_STRUCTURE_H_
#endif
