
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

#include "utilities.hpp"

#include "database/database.hpp"
#include "database/kernels/xaxpy.hpp"
#include "database/kernels/xdot.hpp"
#include "database/kernels/xgemv.hpp"
#include "database/kernels/xgemv_fast.hpp"
#include "database/kernels/xgemv_fast_rot.hpp"
#include "database/kernels/xger.hpp"
#include "database/kernels/xgemm.hpp"
#include "database/kernels/xgemm_direct.hpp"
#include "database/kernels/copy.hpp"
#include "database/kernels/pad.hpp"
#include "database/kernels/transpose.hpp"
#include "database/kernels/padtranspose.hpp"
#include "database/kernel_selection.hpp"

namespace clblast {
// =================================================================================================

// Initializes the database
const std::vector<Database::DatabaseEntry> Database::database = {
  XaxpyHalf, XaxpySingle, XaxpyDouble, XaxpyComplexSingle, XaxpyComplexDouble,
  XdotHalf, XdotSingle, XdotDouble, XdotComplexSingle, XdotComplexDouble,
  XgemvHalf, XgemvSingle, XgemvDouble, XgemvComplexSingle, XgemvComplexDouble,
  XgemvFastHalf, XgemvFastSingle, XgemvFastDouble, XgemvFastComplexSingle, XgemvFastComplexDouble,
  XgemvFastRotHalf, XgemvFastRotSingle, XgemvFastRotDouble, XgemvFastRotComplexSingle, XgemvFastRotComplexDouble,
  XgerHalf, XgerSingle, XgerDouble, XgerComplexSingle, XgerComplexDouble,
  XgemmHalf, XgemmSingle, XgemmDouble, XgemmComplexSingle, XgemmComplexDouble,
  XgemmDirectHalf, XgemmDirectSingle, XgemmDirectDouble, XgemmDirectComplexSingle, XgemmDirectComplexDouble,
  CopyHalf, CopySingle, CopyDouble, CopyComplexSingle, CopyComplexDouble,
  PadHalf, PadSingle, PadDouble, PadComplexSingle, PadComplexDouble,
  TransposeHalf, TransposeSingle, TransposeDouble, TransposeComplexSingle, TransposeComplexDouble,
  PadtransposeHalf, PadtransposeSingle, PadtransposeDouble, PadtransposeComplexSingle, PadtransposeComplexDouble,
  KernelSelectionHalf, KernelSelectionSingle, KernelSelectionDouble, KernelSelectionComplexSingle, KernelSelectionComplexDouble
};

// The OpenCL device types
const std::string Database::kDeviceTypeCPU = "CPU";
const std::string Database::kDeviceTypeGPU = "GPU";
const std::string Database::kDeviceTypeAccelerator = "accelerator";
const std::string Database::kDeviceTypeAll = "default";

// The OpenCL device vendors
const std::string Database::kDeviceVendorAll = "default";

// Alternative names for some OpenCL vendors
const std::unordered_map<std::string, std::string> Database::kVendorNames{
  { "Intel(R) Corporation", "Intel" },
  { "GenuineIntel", "Intel" },
  { "Advanced Micro Devices, Inc.", "AMD" },
  { "NVIDIA Corporation", "NVIDIA" },
};

// =================================================================================================

// Constructor, computing device properties and populating the parameter-vector from the database.
// This takes an optional overlay database in case of custom tuning or custom kernels.
Database::Database(const Queue &queue, const std::vector<std::string> &kernels,
                   const Precision precision, const std::vector<DatabaseEntry> &overlay):
  parameters_{} {

  // Finds information of the current device
  auto device = queue.GetDevice();
  auto device_type = device.Type();
  auto device_vendor = device.Vendor();
  auto device_name = device.Name();

  // Set the short vendor name
  for (auto &combination : kVendorNames) {
    if (device_vendor == combination.first) {
      device_vendor = combination.second;
    }
  }

  // Iterates over all kernels to include, and retrieves the parameters for each of them
  for (auto &kernel: kernels) {
    auto search_result = ParametersPtr{};

    for (auto db: { &overlay, &database }) {
      search_result = Search(kernel, device_type, device_vendor, device_name, precision, *db);
      if (search_result) {
        parameters_.insert(search_result->begin(), search_result->end());
        break;
      }
    }

    if (!search_result) { throw std::runtime_error("Database error, could not find a suitable entry"); }
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

// Searches a particular database for the right kernel and precision
Database::ParametersPtr Database::Search(const std::string &this_kernel,
                                         const std::string &this_type,
                                         const std::string &this_vendor,
                                         const std::string &this_device,
                                         const Precision this_precision,
                                         const std::vector<DatabaseEntry> &this_database) const {

  // Selects the right kernel
  for (auto &db: this_database) {
    if (db.kernel == this_kernel && db.precision == this_precision) {

      // Searches for the right vendor and device type, or selects the default if unavailable. This
      // assumes that the default vendor / device type is last in the database.
      for (auto &vendor: db.vendors) {
        if ((vendor.name == this_vendor || vendor.name == kDeviceVendorAll) &&
            (vendor.type == this_type || vendor.type == kDeviceTypeAll)) {

          // Searches for the right device. If the current device is unavailable, selects the vendor
          // default parameters. This assumes the default is last in the database.
          for (auto &device: vendor.devices) {

            if (device.name == this_device || device.name == "default") {

              // Sets the parameters accordingly
              return &device.parameters;
            }
          }
        }
      }
    }
  }

  // If we reached this point, the entry was not found in this database
  return nullptr;
}

// =================================================================================================
} // namespace clblast
