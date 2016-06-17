
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================

#include <string>
#include <vector>

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// Constructor: not much here, because no status codes can be returned
template <typename T>
Routine<T>::Routine(Queue &queue, EventPointer event, const std::string &name,
                    const std::vector<std::string> &routines, const Precision precision):
    precision_(precision),
    routine_name_(name),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    device_name_(device_.Name()),
    db_(queue_, routines, precision_) {
}

// =================================================================================================

// Separate set-up function to allow for status codes to be returned
template <typename T>
StatusCode Routine<T>::SetUp() {

  // Queries the cache to see whether or not the program (context-specific) is already there
  if (ProgramIsInCache()) { return StatusCode::kSuccess; }

  // Queries the cache to see whether or not the binary (device-specific) is already there. If it
  // is, a program is created and stored in the cache
  if (BinaryIsInCache()) {
    try {
      auto& binary = cache::GetBinaryFromCache(device_name_, precision_, routine_name_);
      auto program = Program(device_, context_, binary);
      auto options = std::vector<std::string>();
      program.Build(device_, options);
      StoreProgramToCache(program);
    } catch (...) { return StatusCode::kBuildProgramFailure; }
    return StatusCode::kSuccess;
  }

  // Otherwise, the kernel will be compiled and program will be built. Both the binary and the
  // program will be added to the cache.

  // Inspects whether or not cl_khr_fp64 is supported in case of double precision
  const auto extensions = device_.Capabilities();
  if (precision_ == Precision::kDouble || precision_ == Precision::kComplexDouble) {
    if (extensions.find(kKhronosDoublePrecision) == std::string::npos) {
      return StatusCode::kNoDoublePrecision;
    }
  }

  // As above, but for cl_khr_fp16 (half precision)
  if (precision_ == Precision::kHalf) {
    if (extensions.find(kKhronosHalfPrecision) == std::string::npos) {
      return StatusCode::kNoHalfPrecision;
    }
  }

  // Loads the common header (typedefs and defines and such)
  std::string common_header =
    #include "kernels/common.opencl"
  ;

  // Collects the parameters for this device in the form of defines, and adds the precision
  auto defines = db_.GetDefines();
  defines += "#define PRECISION "+ToString(static_cast<int>(precision_))+"\n";

  // Adds the name of the routine as a define
  defines += "#define ROUTINE_"+routine_name_+"\n";

  // For specific devices, use the non-IEE754 compilant OpenCL mad() instruction. This can improve
  // performance, but might result in a reduced accuracy.
  if (device_.IsAMD() && device_.IsGPU()) {
    defines += "#define USE_CL_MAD 1\n";
  }

  // For specific devices, use staggered/shuffled workgroup indices.
  if (device_.IsAMD() && device_.IsGPU()) {
    defines += "#define USE_STAGGERED_INDICES 1\n";
  }

  // For specific devices add a global synchronisation barrier to the GEMM kernel to optimize
  // performance through better cache behaviour
  if (device_.IsARM() && device_.IsGPU()) {
    defines += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // Combines everything together into a single source string
  const auto source_string = defines + common_header + source_string_;

  // Compiles the kernel
  try {
    auto program = Program(context_, source_string);
    auto options = std::vector<std::string>();
    const auto build_status = program.Build(device_, options);

    // Checks for compiler crashes/errors/warnings
    if (build_status == BuildStatus::kError) {
      const auto message = program.GetBuildInfo(device_);
      fprintf(stdout, "OpenCL compiler error/warning: %s\n", message.c_str());
      return StatusCode::kBuildProgramFailure;
    }
    if (build_status == BuildStatus::kInvalid) { return StatusCode::kInvalidBinary; }

    // Store the compiled binary and program in the cache
    const auto binary = program.GetIR();
    StoreBinaryToCache(binary);
    StoreProgramToCache(program);
  } catch (...) { return StatusCode::kBuildProgramFailure; }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
StatusCode RunKernel(Kernel &kernel, Queue queue, const Device device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event, std::vector<Event>& waitForEvents) {

  // Tests for validity of the local thread sizes
  if (local.size() > device.MaxWorkItemDimensions()) {
    return StatusCode::kInvalidLocalNumDimensions; 
  }
  const auto max_work_item_sizes = device.MaxWorkItemSizes();
  for (auto i=size_t{0}; i<local.size(); ++i) {
    if (local[i] > max_work_item_sizes[i]) { return StatusCode::kInvalidLocalThreadsDim; }
  }
  auto local_size = size_t{1};
  for (auto &item: local) { local_size *= item; }
  if (local_size > device.MaxWorkGroupSize()) { return StatusCode::kInvalidLocalThreadsTotal; }

  // Make sure the global thread sizes are at least equal to the local sizes
  for (auto i=size_t{0}; i<global.size(); ++i) {
    if (global[i] < local[i]) { global[i] = local[i]; }
  }

  // Tests for local memory usage
  const auto local_mem_usage = kernel.LocalMemUsage(device);
  if (!device.IsLocalMemoryValid(local_mem_usage)) { return StatusCode::kInvalidLocalMemUsage; }

  // Launches the kernel (and checks for launch errors)
  try {
    kernel.Launch(queue, global, local, event, waitForEvents);
  } catch (...) { return StatusCode::kKernelLaunchError; }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// As above, but without an event waiting list
StatusCode RunKernel(Kernel &kernel, Queue queue, const Device device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event) {
  auto emptyWaitingList = std::vector<Event>();
  return RunKernel(kernel, queue, device, global, local, event, emptyWaitingList);
}

// =================================================================================================

// Compiles the templated class
template class Routine<half>;
template class Routine<float>;
template class Routine<double>;
template class Routine<float2>;
template class Routine<double2>;

// =================================================================================================
} // namespace clblast
