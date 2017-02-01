
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file contains declarations for the plugin API. A plugin is a shared object loaded at runtime
// which provides potentially optimized device-specific versions of routines and database entries.
// A plugin must contain a public (exported) symbol `clblast::Base *clblast_plugin`, which must
// point to an object of type `clblast::plugin::Interface` (or descendant).
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_PLUGIN_H_
#define CLBLAST_CLBLAST_PLUGIN_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "clblast.h"

namespace clblast {

namespace plugin {
// =================================================================================================

// This is a base class for all objects returned from the plugin non-type-safely, which must be
// returned as pointers to this base class and then down-casted with dynamic_cast<>().
// This way, we can ensure ABI compatibility in a type-safe way.
// WARNING: if this definition is ever altered, the exported symbol name must be changed.
class Base {
public:
  virtual ~Base();
};

// =================================================================================================

// WARNING: if any definition inside this namespace is altered, the version must be incremented.
inline namespace version_1 {
// =================================================================================================

// Plugin interface to the database class
class Database {
 public:
  virtual ~Database();

  // Accessor of values by key
  virtual size_t operator[](const std::string &key) const = 0;
};

// =================================================================================================

// Plugin interface to clblast::RunKernel() using C types for ABI stability
// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(cl_kernel kernel, cl_command_queue queue, cl_device_id device,
               const std::vector<size_t> &global, const std::vector<size_t> &local,
               cl_event *event, const std::vector<cl_event> &waitForEvents = {});

// =================================================================================================

// Plugin interface to clblast::PadCopyTransposeMatrix() using C types for ABI stability
// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
template <typename T>
void PadCopyTransposeMatrix(cl_command_queue queue, cl_device_id device,
                            const Database &db,
                            cl_event *event, const std::vector<cl_event> &waitForEvents,
                            const size_t src_one, const size_t src_two,
                            const size_t src_ld, const size_t src_offset,
                            cl_mem src,
                            const size_t dest_one, const size_t dest_two,
                            const size_t dest_ld, const size_t dest_offset,
                            cl_mem dest,
                            const T alpha,
                            cl_program program, const bool do_pad,
                            const bool do_transpose, const bool do_conjugate,
                            const bool upper = false, const bool lower = false,
                            const bool diagonal_imag_zero = false);

// =================================================================================================

namespace database {

using Parameters = std::unordered_map<std::string, size_t>;

// Database tree leaf entry: describes a single device model
struct Device {
  // clGetDeviceInfo(CL_DEVICE_NAME) or "default"
  std::string name;

  // Actual parameters for this device and routine
  Parameters parameters;
};

// Database tree vendor entry: group of entries for a single device vendor
struct Vendor {
  // clGetDeviceInfo(CL_DEVICE_TYPE) as string: "CPU", "GPU", "accelerator" or "default"
  std::string type;

  // clGetDeviceInfo(CL_DEVICE_VENDOR) or "default"
  std::string name;

  // Entries for this vendor
  std::vector<Device> devices;
};

// Database tree routine entry: group of entries for a single routine tag
struct Routine {
  // Routine tag (see src/routines/*/*.cpp), multiple tags are used per each real BLAS routine
  std::string kernel;

  // Input data precision corresponding to these parameters
  Precision precision;

  // Entries for this routine
  std::vector<Vendor> vendors;
};

} // namespace database

// =================================================================================================

class Routine;

class Interface : public Base {
 public:
  virtual ~Interface();

  virtual const Routine *GetRoutine(cl_device_id device, const std::string &routine, Precision precision) const = 0;
};

// =================================================================================================

// The non-routine-specific part of a plugin entry.
// This base must be virtual in all routine-specific classes because we have routines such as Xhemm
// (which inherit from Xgemm).
class Routine : public Base {
 public:
  Routine();
  virtual ~Routine();

  // Custom database entries
  // NOTE: custom database entries completely override built-in ones on a per-routine basis
  // (i. e. if Xgemm/32 is overridden, then the built-in database is never looked up for Xgemm/32)
  std::vector<const database::Routine *> database;

  // Custom OpenCL kernel source, to be used alongside/instead of the built-in OpenCL kernel
  // NOTE: host<->device API/ABI is not considered stable, so avoid using built-in host routines
  // with custom kernels.
  std::string kernel;
  enum class KernelMode {
    // Custom kernel is ignored
    Default = 0,

    // Custom kernel is appended to built-in PadCopyTransform code
    NeedPCT,

    // Custom kernel replaces all built-in code
    Custom
  } kernel_mode;

  // Specifies whether to use custom host routine (provided via a corresponding virtual function
  // in a derived class).
  // NOTE: host<-> device API/ABI is not considered stable, so avoid calling built-in kernels
  // from custom routines.
  enum class RoutineMode {
    // Custom host routine is ignored
    Default = 0,

    // Custom host routine replaces the built-in one
    Custom
  } routine_mode;
};

// =================================================================================================
} // inline namespace version_X

} // namespace plugin

} // namespace clblast

// CLBLAST_CLBLAST_PLUGIN_H_
#endif
