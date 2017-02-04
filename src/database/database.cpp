
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

#include "utilities/utilities.hpp"

#include "database/database.hpp"
#include "database/kernels/xaxpy.hpp"
#include "database/kernels/xdot.hpp"
#include "database/kernels/xgemv.hpp"
#include "database/kernels/xgemv_fast.hpp"
#include "database/kernels/xgemv_fast_rot.hpp"
#include "database/kernels/xger.hpp"
#include "database/kernels/xtrsv.hpp"
#include "database/kernels/xgemm.hpp"
#include "database/kernels/xgemm_direct.hpp"
#include "database/kernels/copy.hpp"
#include "database/kernels/pad.hpp"
#include "database/kernels/transpose.hpp"
#include "database/kernels/padtranspose.hpp"
#include "database/kernels/invert.hpp"
#include "database/kernel_selection.hpp"

namespace clblast {
// =================================================================================================

// Initializes the database
const std::vector<const Database::DatabaseEntry*> Database::database = {
  &database::XaxpyHalf, &database::XaxpySingle, &database::XaxpyDouble, &database::XaxpyComplexSingle, &database::XaxpyComplexDouble,
  &database::XdotHalf, &database::XdotSingle, &database::XdotDouble, &database::XdotComplexSingle, &database::XdotComplexDouble,
  &database::XgemvHalf, &database::XgemvSingle, &database::XgemvDouble, &database::XgemvComplexSingle, &database::XgemvComplexDouble,
  &database::XgemvFastHalf, &database::XgemvFastSingle, &database::XgemvFastDouble, &database::XgemvFastComplexSingle, &database::XgemvFastComplexDouble,
  &database::XgemvFastRotHalf, &database::XgemvFastRotSingle, &database::XgemvFastRotDouble, &database::XgemvFastRotComplexSingle, &database::XgemvFastRotComplexDouble,
  &database::XgerHalf, &database::XgerSingle, &database::XgerDouble, &database::XgerComplexSingle, &database::XgerComplexDouble,
  &database::XtrsvHalf, &database::XtrsvSingle, &database::XtrsvDouble, &database::XtrsvComplexSingle, &database::XtrsvComplexDouble,
  &database::XgemmHalf, &database::XgemmSingle, &database::XgemmDouble, &database::XgemmComplexSingle, &database::XgemmComplexDouble,
  &database::XgemmDirectHalf, &database::XgemmDirectSingle, &database::XgemmDirectDouble, &database::XgemmDirectComplexSingle, &database::XgemmDirectComplexDouble,
  &database::CopyHalf, &database::CopySingle, &database::CopyDouble, &database::CopyComplexSingle, &database::CopyComplexDouble,
  &database::PadHalf, &database::PadSingle, &database::PadDouble, &database::PadComplexSingle, &database::PadComplexDouble,
  &database::TransposeHalf, &database::TransposeSingle, &database::TransposeDouble, &database::TransposeComplexSingle, &database::TransposeComplexDouble,
  &database::PadtransposeHalf, &database::PadtransposeSingle, &database::PadtransposeDouble, &database::PadtransposeComplexSingle, &database::PadtransposeComplexDouble,
  &database::InvertHalf, &database::InvertSingle, &database::InvertDouble, &database::InvertComplexSingle, &database::InvertComplexDouble,
  &database::KernelSelectionHalf, &database::KernelSelectionSingle, &database::KernelSelectionDouble, &database::KernelSelectionComplexSingle, &database::KernelSelectionComplexDouble
};

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
                   const Precision precision, const std::vector<const DatabaseEntry*> &overlay):
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

    for (auto &db: { database, overlay}) {
      search_result = Search(kernel, device_type, device_vendor, device_name, precision, db);
      if (search_result) {
        parameters_.insert(search_result->begin(), search_result->end());
        break;
      }
    }

    if (!search_result) { throw RuntimeErrorCode(StatusCode::kDatabaseError); }
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
                                         const std::vector<const DatabaseEntry*> &this_database) const {

  // Selects the right kernel
  for (auto &db: this_database) {
    if (db->kernel == this_kernel && db->precision == this_precision) {

      // Searches for the right vendor and device type, or selects the default if unavailable. This
      // assumes that the default vendor / device type is last in the database.
      for (auto &vendor: db->vendors) {
        if ((vendor.name == this_vendor || vendor.name == kDeviceVendorAll) &&
            (vendor.type == this_type || vendor.type == database::kDeviceTypeAll)) {

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
