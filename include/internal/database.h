
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

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Database {
 public:

  // Type alias for the database parameters
  using Parameters = std::unordered_map<std::string,size_t>;

  // Structures for content inside the database
  struct DatabaseDevice {
    const std::string name;
    const Parameters parameters;
  };
  struct DatabaseVendor {
    const cl_device_type type;
    const std::string name;
    const std::vector<DatabaseDevice> devices;
  };
  struct DatabaseEntry {
    const std::string kernel;
    const Precision precision;
    const std::vector<DatabaseVendor> vendors;
  };

  // The default vendor or device
  static constexpr auto kDefault = "Default";

  // The database consists of separate database entries, stored together in a vector
  static const DatabaseEntry XaxpySingle, XaxpyDouble, XaxpyComplexSingle, XaxpyComplexDouble;
  static const DatabaseEntry XgemmSingle, XgemmDouble, XgemmComplexSingle, XgemmComplexDouble;
  static const DatabaseEntry CopySingle, CopyDouble, CopyComplexSingle, CopyComplexDouble;
  static const DatabaseEntry PadSingle, PadDouble, PadComplexSingle, PadComplexDouble;
  static const DatabaseEntry TraSingle, TraDouble, TraComplexSingle, TraComplexDouble;
  static const DatabaseEntry PadTraSingle, PadTraDouble, PadTraComplexSingle, PadTraComplexDouble;
  static const std::vector<DatabaseEntry> database;

  // The constructor
  explicit Database(const CommandQueue &queue, const std::vector<std::string> &routines,
                    const Precision precision);

  // Accessor of values by key
  size_t operator[](const std::string key) const { return parameters_.find(key)->second; }

  // Obtain a list of OpenCL pre-processor defines based on the parameters
  std::string GetDefines() const;

 private:
  Parameters Search(const std::string &this_kernel, const cl_device_type this_type,
                    const std::string &this_vendor, const std::string &this_device,
                    const Precision this_precision) const;

  // Tests equality between a database-vendor string and an OpenCL vendor string
  bool VendorEqual(const std::string &db_vendor, const std::string &cl_vendor) const;

  // Found parameters suitable for this device/kernel
  Parameters parameters_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_DATABASE_H_
#endif
