// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common (non-OpenCL-specific) functions of the CLBlast API.
//
// =================================================================================================

#include <string>

#include "utilities/utilities.hpp"
#include "cache.hpp"
#include "routines/routines.hpp"

namespace clblast {
// =================================================================================================

// Clears the cache of stored binaries
StatusCode ClearCache() {
  try {
    ProgramCache::Instance().Invalidate();
    BinaryCache::Instance().Invalidate();
  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

template <typename Real, typename Complex>
void FillCacheForPrecision(Queue &queue) {
  try {

    // Runs all the level 1 set-up functions
    Xswap<Real>(queue, nullptr); Xswap<Complex>(queue, nullptr);
    Xswap<Real>(queue, nullptr); Xswap<Complex>(queue, nullptr);
    Xscal<Real>(queue, nullptr); Xscal<Complex>(queue, nullptr);
    Xcopy<Real>(queue, nullptr); Xcopy<Complex>(queue, nullptr);
    Xaxpy<Real>(queue, nullptr); Xaxpy<Complex>(queue, nullptr);
    Xdot<Real>(queue, nullptr);
    Xdotu<Complex>(queue, nullptr);
    Xdotc<Complex>(queue, nullptr);
    Xnrm2<Real>(queue, nullptr); Xnrm2<Complex>(queue, nullptr);
    Xasum<Real>(queue, nullptr); Xasum<Complex>(queue, nullptr);
    Xsum<Real>(queue, nullptr); Xsum<Complex>(queue, nullptr);
    Xamax<Real>(queue, nullptr); Xamax<Complex>(queue, nullptr);
    Xmax<Real>(queue, nullptr); Xmax<Complex>(queue, nullptr);
    Xmin<Real>(queue, nullptr); Xmin<Complex>(queue, nullptr);

    // Runs all the level 2 set-up functions
    Xgemv<Real>(queue, nullptr); Xgemv<Complex>(queue, nullptr);
    Xgbmv<Real>(queue, nullptr); Xgbmv<Complex>(queue, nullptr);
    Xhemv<Complex>(queue, nullptr);
    Xhbmv<Complex>(queue, nullptr);
    Xhpmv<Complex>(queue, nullptr);
    Xsymv<Real>(queue, nullptr);
    Xsbmv<Real>(queue, nullptr);
    Xspmv<Real>(queue, nullptr);
    Xtrmv<Real>(queue, nullptr); Xtrmv<Complex>(queue, nullptr);
    Xtbmv<Real>(queue, nullptr); Xtbmv<Complex>(queue, nullptr);
    Xtpmv<Real>(queue, nullptr); Xtpmv<Complex>(queue, nullptr);
    Xger<Real>(queue, nullptr);
    Xgeru<Complex>(queue, nullptr);
    Xgerc<Complex>(queue, nullptr);
    Xher<Complex,Real>(queue, nullptr);
    Xhpr<Complex,Real>(queue, nullptr);
    Xher2<Complex>(queue, nullptr);
    Xhpr2<Complex>(queue, nullptr);
    Xsyr<Real>(queue, nullptr);
    Xspr<Real>(queue, nullptr);
    Xsyr2<Real>(queue, nullptr);
    Xspr2<Real>(queue, nullptr);

    // Runs all the level 3 set-up functions
    Xgemm<Real>(queue, nullptr); Xgemm<Complex>(queue, nullptr);
    Xsymm<Real>(queue, nullptr); Xsymm<Complex>(queue, nullptr);
    Xhemm<Complex>(queue, nullptr);
    Xsyrk<Real>(queue, nullptr); Xsyrk<Complex>(queue, nullptr);
    Xherk<Complex,Real>(queue, nullptr);
    Xsyr2k<Real>(queue, nullptr); Xsyr2k<Complex>(queue, nullptr);
    Xher2k<Complex,Real>(queue, nullptr);
    Xtrmm<Real>(queue, nullptr); Xtrmm<Complex>(queue, nullptr);

    // Runs all the non-BLAS set-up functions
    Xomatcopy<Real>(queue, nullptr); Xomatcopy<Complex>(queue, nullptr);

  } catch(const RuntimeErrorCode &e) {
    if (e.status() != StatusCode::kNoDoublePrecision &&
        e.status() != StatusCode::kNoHalfPrecision) {
      throw;
    }
  }
}

// Fills the cache with all binaries for a specific device
// TODO: Add half-precision FP16 set-up calls
StatusCode FillCache(const RawDeviceID device) {
  try {

    // Creates a sample context and queue to match the normal routine calling conventions
    auto device_cpp = Device(device);
    auto context = Context(device_cpp);
    auto queue = Queue(context, device_cpp);

    FillCacheForPrecision<float, float2>(queue);
    FillCacheForPrecision<double, double2>(queue);

  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Retrieves the current tuning parameters for this device-precision-kernel combination
StatusCode RetrieveParameters(const RawDeviceID device, const std::string &kernel_name,
                              const Precision precision,
                              std::unordered_map<std::string,size_t> &parameters) {
  try {

    // Retrieves the device name
    const auto device_cpp = Device(device);
    const auto platform_id = device_cpp.PlatformID();
    const auto device_name = GetDeviceName(device_cpp);

    // Retrieves the database values
    auto in_cache = false;
    auto database = DatabaseCache::Instance().Get(DatabaseKeyRef{platform_id, device, precision, kernel_name}, &in_cache);
    if (!in_cache) {
      log_debug("Searching database for kernel '" + kernel_name + "'");
      database = Database(device_cpp, kernel_name, precision, {});
    }

    // Retrieves the parameters
    for (const auto &parameter: database.GetParameters()) {
      parameters[parameter.first] = parameter.second;
    }

  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

// Overrides the tuning parameters for this device-precision-kernel combination
StatusCode OverrideParameters(const RawDeviceID device, const std::string &kernel_name,
                              const Precision precision,
                              const std::unordered_map<std::string,size_t> &parameters) {
  try {

    // Retrieves the device name
    const auto device_cpp = Device(device);
    const auto platform_id = device_cpp.PlatformID();
    const auto device_name = GetDeviceName(device_cpp);

    // Retrieves the current database values to verify whether the new ones are complete
    auto in_cache = false;
    auto current_database = DatabaseCache::Instance().Get(DatabaseKeyRef{platform_id, device, precision, kernel_name}, &in_cache);
    if (!in_cache) {
      log_debug("Searching database for kernel '" + kernel_name + "'");
      current_database = Database(device_cpp, kernel_name, precision, {});
    }

    // Verifies the parameters size
    const auto current_parameter_names = current_database.GetParameterNames();
    if (current_parameter_names.size() > parameters.size()) {
      return StatusCode::kMissingOverrideParameter;
    }

    // Retrieves the names and values separately and in the same order as the existing database
    auto parameter_values = database::Params{0};
    auto i = size_t{0};
    for (const auto &current_param : current_parameter_names) {
      if (parameters.find(current_param) == parameters.end()) {
        return StatusCode::kMissingOverrideParameter;
      }
      const auto parameter_value = parameters.at(current_param);
      parameter_values[i] = parameter_value;
      ++i;
    }

    // Creates a small custom database based on the provided parameters
    const auto database_device = database::DatabaseDevice{database::kDeviceNameDefault, parameter_values};
    const auto database_architecture = database::DatabaseArchitecture{"default", {database_device}};
    const auto database_vendor = database::DatabaseVendor{database::kDeviceTypeAll, "default", {database_architecture}};
    const auto database_entry = database::DatabaseEntry{kernel_name, precision, current_parameter_names, {database_vendor}};
    const auto database_entries = std::vector<database::DatabaseEntry>{database_entry};
    const auto database = Database(device_cpp, kernel_name, precision, database_entries);

    // Removes the old database entry and stores the new one in the cache
    DatabaseCache::Instance().Remove(DatabaseKey{platform_id, device, precision, kernel_name});
    DatabaseCache::Instance().Store(DatabaseKey{platform_id, device, precision, kernel_name}, Database(database));

  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace clblast
