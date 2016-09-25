
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

#include "utilities.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Database {
 public:

  // Type alias for the database parameters
  using Parameters = std::unordered_map<std::string,size_t>;
  using ParametersPtr = const Parameters*;

  // Structures for content inside the database
  struct DatabaseDevice {
    const std::string name;
    const Parameters parameters;
  };
  struct DatabaseVendor {
    const std::string type;
    const std::string name;
    const std::vector<DatabaseDevice> devices;
  };
  struct DatabaseEntry {
    const std::string kernel;
    const Precision precision;
    const std::vector<DatabaseVendor> vendors;
  };

  // The OpenCL device types
  static constexpr auto kDeviceTypeCPU = "CPU";
  static constexpr auto kDeviceTypeGPU = "GPU";
  static constexpr auto kDeviceTypeAccelerator = "accelerator";
  static constexpr auto kDeviceTypeAll = "default";

  // The OpenCL device vendors
  static constexpr auto kDeviceVendorAll = "default";

  // Alternative names for some OpenCL vendors
  const std::unordered_map<std::string,std::string> kVendorNames {
    {"Intel(R) Corporation", "Intel"},
    {"GenuineIntel", "Intel"},
    {"Advanced Micro Devices, Inc.", "AMD"},
    {"NVIDIA Corporation", "NVIDIA"},
  };

  // The database consists of separate database entries, stored together in a vector
  static const DatabaseEntry XaxpyHalf, XaxpySingle, XaxpyDouble, XaxpyComplexSingle, XaxpyComplexDouble;
  static const DatabaseEntry XdotHalf, XdotSingle, XdotDouble, XdotComplexSingle, XdotComplexDouble;
  static const DatabaseEntry XgemvHalf, XgemvSingle, XgemvDouble, XgemvComplexSingle, XgemvComplexDouble;
  static const DatabaseEntry XgemvFastHalf, XgemvFastSingle, XgemvFastDouble, XgemvFastComplexSingle, XgemvFastComplexDouble;
  static const DatabaseEntry XgemvFastRotHalf, XgemvFastRotSingle, XgemvFastRotDouble, XgemvFastRotComplexSingle, XgemvFastRotComplexDouble;
  static const DatabaseEntry XgerHalf, XgerSingle, XgerDouble, XgerComplexSingle, XgerComplexDouble;
  static const DatabaseEntry XgemmHalf, XgemmSingle, XgemmDouble, XgemmComplexSingle, XgemmComplexDouble;
  static const DatabaseEntry XgemmDirectHalf, XgemmDirectSingle, XgemmDirectDouble, XgemmDirectComplexSingle, XgemmDirectComplexDouble;
  static const DatabaseEntry CopyHalf, CopySingle, CopyDouble, CopyComplexSingle, CopyComplexDouble;
  static const DatabaseEntry PadHalf, PadSingle, PadDouble, PadComplexSingle, PadComplexDouble;
  static const DatabaseEntry TransposeHalf, TransposeSingle, TransposeDouble, TransposeComplexSingle, TransposeComplexDouble;
  static const DatabaseEntry PadtransposeHalf, PadtransposeSingle, PadtransposeDouble, PadtransposeComplexSingle, PadtransposeComplexDouble;
  static const std::vector<DatabaseEntry> database;

  // The constructor with a user-provided database overlay (potentially an empty vector)
  explicit Database(const Queue &queue, const std::vector<std::string> &routines,
                    const Precision precision, const std::vector<DatabaseEntry> &overlay);

  // Accessor of values by key
  size_t operator[](const std::string key) const { return parameters_.find(key)->second; }

  // Obtain a list of OpenCL pre-processor defines based on the parameters
  std::string GetDefines() const;

 private:
  // Search method for a specified database, returning pointer (possibly a nullptr)
  ParametersPtr Search(const std::string &this_kernel, const std::string &this_type,
                       const std::string &this_vendor, const std::string &this_device,
                       const Precision this_precision, const std::vector<DatabaseEntry> &db) const;

  // Found parameters suitable for this device/kernel
  Parameters parameters_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_DATABASE_H_
#endif
