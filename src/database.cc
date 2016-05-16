
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class (see the header for information about the class).
//
// =================================================================================================

#include "internal/database.h"
#include "internal/database/xaxpy.h"
#include "internal/database/xdot.h"
#include "internal/database/xgemv.h"
#include "internal/database/xger.h"
#include "internal/database/xgemm.h"
#include "internal/database/copy.h"
#include "internal/database/pad.h"
#include "internal/database/transpose.h"
#include "internal/database/padtranspose.h"

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// Initializes the database
const std::vector<Database::DatabaseEntry> Database::database = {
  XaxpyHalf, XaxpySingle, XaxpyDouble, XaxpyComplexSingle, XaxpyComplexDouble,
  XdotSingle, XdotDouble, XdotComplexSingle, XdotComplexDouble,
  XgemvSingle, XgemvDouble, XgemvComplexSingle, XgemvComplexDouble,
  XgerSingle, XgerDouble, XgerComplexSingle, XgerComplexDouble,
  XgemmSingle, XgemmDouble, XgemmComplexSingle, XgemmComplexDouble,
  CopyHalf, CopySingle, CopyDouble, CopyComplexSingle, CopyComplexDouble,
  PadHalf, PadSingle, PadDouble, PadComplexSingle, PadComplexDouble,
  TransposeHalf, TransposeSingle, TransposeDouble, TransposeComplexSingle, TransposeComplexDouble,
  PadtransposeHalf, PadtransposeSingle, PadtransposeDouble, PadtransposeComplexSingle, PadtransposeComplexDouble
};

// =================================================================================================

// Constructor, computing device properties and populating the parameter-vector from the database
Database::Database(const Queue &queue, const std::vector<std::string> &kernels,
                   const Precision precision):
  parameters_{} {

  // Finds information of the current device
  auto device = queue.GetDevice();
  auto device_type = device.Type();
  auto device_vendor = device.Vendor();
  auto device_name = device.Name();

  // Iterates over all kernels to include, and retrieves the parameters for each of them
  for (auto &kernel: kernels) {
    auto search_result = Search(kernel, device_type, device_vendor, device_name, precision);
    parameters_.insert(search_result.begin(), search_result.end());
  }
}

// =================================================================================================

// Returns a list of OpenCL pre-processor defines in string form
std::string Database::GetDefines() const {
  std::string defines{};
  for (auto &parameter: parameters_) {
    defines += "#define "+parameter.first+" "+ToString(parameter.second)+"\n";
  }
  return defines;
}

// =================================================================================================

// Searches the database for the right kernel and precision
Database::Parameters Database::Search(const std::string &this_kernel,
                                      const std::string &this_type,
                                      const std::string &this_vendor,
                                      const std::string &this_device,
                                      const Precision this_precision) const {
  // Set the short vendor name
  auto this_short_vendor = this_vendor;
  for (auto &combination : kVendorNames) {
    if (this_vendor == combination.first) {
      this_short_vendor = combination.second;
    }
  }

  // Selects the right kernel
  for (auto &db: database) {
    if (db.kernel == this_kernel && db.precision == this_precision) {

      // Searches for the right vendor and device type, or selects the default if unavailable. This
      // assumes that the default vendor / device type is last in the database.
      for (auto &vendor: db.vendors) {
        if ((vendor.name == this_short_vendor || vendor.name == kDeviceVendorAll) &&
            (vendor.type == this_type || vendor.type == kDeviceTypeAll)) {

          // Searches for the right device. If the current device is unavailable, selects the vendor
          // default parameters. This assumes the default is last in the database.
          for (auto &device: vendor.devices) {

            if (device.name == this_device || device.name == "default") {

              // Sets the parameters accordingly
              return device.parameters;
            }
          }
        }
      }
    }
  }

  // If we reached this point, something is wrong
  throw std::runtime_error("Database error, could not find a suitable entry");
}

// =================================================================================================
} // namespace clblast
