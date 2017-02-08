
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class, providing a static variable holding the actual database
// information. The class also provides utility functions to search the database and to access a
// found entry by parameter-key. The database itself is filled in the corresponding source-file and
// partially also by the database/xxxxx.h files, in which kernel-specific parameters are found.
//
// =================================================================================================

#ifndef CLBLAST_DATABASE_H_
#define CLBLAST_DATABASE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "utilities/utilities.hpp"
#include "clblast_plugin.h"

namespace clblast {
// =================================================================================================

// A special namespace to hold all the global constant variables (including the database entries)
namespace database {

  // The OpenCL device types
  const std::string kDeviceTypeCPU = "CPU";
  const std::string kDeviceTypeGPU = "GPU";
  const std::string kDeviceTypeAccelerator = "accelerator";
  const std::string kDeviceTypeAll = "default";

} // namespace database


// =================================================================================================

namespace plugin {

class Plugin;

} // namespace plugin

// =================================================================================================

// See comment at top of file for a description of the class
class Database : public plugin::Database {
 public:

  // These data structures are part of the plugin interface now
  using Parameters = plugin::database::Parameters;
  using ParametersPtr = const Parameters*;
  using DatabaseDevice = plugin::database::Device;
  using DatabaseVendor = plugin::database::Vendor;
  using DatabaseEntry = plugin::database::Routine;

  // The OpenCL device vendors
  static const std::string kDeviceVendorAll;

  // Alternative names for some OpenCL vendors
  static const std::unordered_map<std::string, std::string> kVendorNames;

  // The database consists of separate database entries, stored together in a vector
  static const std::vector<const DatabaseEntry*> database;

  Database() = default;

  // The constructor with a user-provided database overlay (potentially an empty vector)
  explicit Database(const Device &device, const std::vector<std::string> &routines,
                    const Precision precision, const plugin::Plugin &plugin);

  // Accessor of values by key
  size_t operator[](const std::string &key) const { return parameters_->find(key)->second; }

  // Obtain a list of OpenCL pre-processor defines based on the parameters
  std::string GetDefines() const;

 private:
  // Search method for a specified database, returning pointer (possibly a nullptr)
  ParametersPtr Search(const std::string &this_kernel, const std::string &this_type,
                       const std::string &this_vendor, const std::string &this_device,
                       const Precision this_precision,
                       const std::vector<const DatabaseEntry*> &db) const;

  // Found parameters suitable for this device/kernel
  std::shared_ptr<Parameters> parameters_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_DATABASE_H_
#endif
