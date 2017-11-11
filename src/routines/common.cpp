
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common routine functions (see the header for more information).
//
// =================================================================================================

#include <vector>
#include <chrono>

#include "routines/common.hpp"

namespace clblast {
// =================================================================================================

// Compiles a program from source code
Program CompileFromSource(const std::string &source_string, const Precision precision,
                          const std::string &routine_name,
                          const Device& device, const Context& context,
                          std::vector<std::string>& options) {
  auto header_string = std::string{""};

  header_string += "#define PRECISION " + ToString(static_cast<int>(precision)) + "\n";

  // Adds the name of the routine as a define
  header_string += "#define ROUTINE_" + routine_name + "\n";

  // Not all OpenCL compilers support the 'inline' keyword. The keyword is only used for devices on
  // which it is known to work with all OpenCL platforms.
  if (device.IsNVIDIA() || device.IsARM()) {
    header_string += "#define USE_INLINE_KEYWORD 1\n";
  }

  // For specific devices, use the non-IEE754 compliant OpenCL mad() instruction. This can improve
  // performance, but might result in a reduced accuracy.
  if (device.IsAMD() && device.IsGPU()) {
    header_string += "#define USE_CL_MAD 1\n";
  }

  // For specific devices, use staggered/shuffled workgroup indices.
  if (device.IsAMD() && device.IsGPU()) {
    header_string += "#define USE_STAGGERED_INDICES 1\n";
  }

  // For specific devices add a global synchronisation barrier to the GEMM kernel to optimize
  // performance through better cache behaviour
  if (device.IsARM() && device.IsGPU()) {
    header_string += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // Optionally adds a translation header from OpenCL kernels to CUDA kernels
  #ifdef CUDA_API
    source_string +=
      #include "kernels/opencl_to_cuda.h"
    ;
  #endif

  // Loads the common header (typedefs and defines and such)
  header_string +=
    #include "kernels/common.opencl"
  ;

  // Prints details of the routine to compile in case of debugging in verbose mode
  #ifdef VERBOSE
    printf("[DEBUG] Compiling routine '%s-%s'\n",
           routine_name.c_str(), ToString(precision).c_str());
    const auto start_time = std::chrono::steady_clock::now();
  #endif

  // Compiles the kernel
  auto program = Program(context, header_string + source_string);
  try {
    program.Build(device, options);
  } catch (const CLCudaAPIBuildError &e) {
    if (program.StatusIsCompilationWarningOrError(e.status())) {
      fprintf(stdout, "OpenCL compiler error/warning: %s\n",
              program.GetBuildInfo(device).c_str());
    }
    throw;
  }

  // Prints the elapsed compilation time in case of debugging in verbose mode
  #ifdef VERBOSE
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
    printf("[DEBUG] Completed compilation in %.2lf ms\n", timing);
  #endif

  return program;
}

// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(Kernel &kernel, Queue &queue, const Device &device,
               std::vector<size_t> global, const std::vector<size_t> &local,
               EventPointer event, const std::vector<Event> &waitForEvents) {

  if (!local.empty()) {
    // Tests for validity of the local thread sizes
    if (local.size() > device.MaxWorkItemDimensions()) {
      throw RuntimeErrorCode(StatusCode::kInvalidLocalNumDimensions);
    }
    const auto max_work_item_sizes = device.MaxWorkItemSizes();
    for (auto i=size_t{0}; i<local.size(); ++i) {
      if (local[i] > max_work_item_sizes[i]) {
        throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim);
      }
    }
    auto local_size = size_t{1};
    for (auto &item: local) { local_size *= item; }
    if (local_size > device.MaxWorkGroupSize()) {
      throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsTotal);
    }

    // Make sure the global thread sizes are at least equal to the local sizes
    for (auto i=size_t{0}; i<global.size(); ++i) {
      if (global[i] < local[i]) { global[i] = local[i]; }
    }
  }

  // Tests for local memory usage
  const auto local_mem_usage = kernel.LocalMemUsage(device);
  if (!device.IsLocalMemoryValid(local_mem_usage)) {
    throw RuntimeErrorCode(StatusCode::kInvalidLocalMemUsage);
  }

  // Prints the name of the kernel to launch in case of debugging in verbose mode
  #ifdef VERBOSE
    queue.Finish();
    printf("[DEBUG] Running kernel '%s'\n", kernel.GetFunctionName().c_str());
    const auto start_time = std::chrono::steady_clock::now();
  #endif

  // Launches the kernel (and checks for launch errors)
  kernel.Launch(queue, global, local, event, waitForEvents);

  // Prints the elapsed execution time in case of debugging in verbose mode
  #ifdef VERBOSE
    queue.Finish();
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
    printf("[DEBUG] Completed kernel in %.2lf ms\n", timing);
  #endif
}

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(Queue &queue, const Device &device,
                const Program &program, const Databases &,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                const Buffer<T> &dest,
                const T constant_value) {
  auto kernel = Kernel(program, "FillMatrix");
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(ld));
  kernel.SetArgument(3, static_cast<int>(offset));
  kernel.SetArgument(4, dest());
  kernel.SetArgument(5, GetRealArg(constant_value));
  auto local = std::vector<size_t>{8, 8};
  auto global = std::vector<size_t>{Ceil(m, 8), Ceil(n, 8)};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// Compiles the above function
template void FillMatrix<half>(Queue&, const Device&, const Program&, const Databases&,
                               EventPointer, const std::vector<Event>&, const size_t, const size_t,
                               const size_t, const size_t, const Buffer<half>&, const half);
template void FillMatrix<float>(Queue&, const Device&, const Program&, const Databases&,
                                EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                const size_t, const size_t, const Buffer<float>&, const float);
template void FillMatrix<double>(Queue&, const Device&, const Program&, const Databases&,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const size_t, const Buffer<double>&, const double);
template void FillMatrix<float2>(Queue&, const Device&, const Program&, const Databases&,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const size_t, const Buffer<float2>&, const float2);
template void FillMatrix<double2>(Queue&, const Device&, const Program&, const Databases&,
                                  EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                  const size_t, const size_t, const Buffer<double2>&, const double2);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(Queue &queue, const Device &device,
                const Program &program, const Databases &,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t n, const size_t inc, const size_t offset,
                const Buffer<T> &dest,
                const T constant_value) {
  auto kernel = Kernel(program, "FillVector");
  kernel.SetArgument(0, static_cast<int>(n));
  kernel.SetArgument(1, static_cast<int>(inc));
  kernel.SetArgument(2, static_cast<int>(offset));
  kernel.SetArgument(3, dest());
  kernel.SetArgument(4, GetRealArg(constant_value));
  auto local = std::vector<size_t>{64};
  auto global = std::vector<size_t>{Ceil(n, 64)};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// Compiles the above function
template void FillVector<half>(Queue&, const Device&, const Program&, const Databases&,
                               EventPointer, const std::vector<Event>&, const size_t, const size_t,
                               const size_t, const Buffer<half>&, const half);
template void FillVector<float>(Queue&, const Device&, const Program&, const Databases&,
                                EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                const size_t, const Buffer<float>&, const float);
template void FillVector<double>(Queue&, const Device&, const Program&, const Databases&,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const Buffer<double>&, const double);
template void FillVector<float2>(Queue&, const Device&, const Program&, const Databases&,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const Buffer<float2>&, const float2);
template void FillVector<double2>(Queue&, const Device&, const Program&, const Databases&,
                                  EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                  const size_t, const Buffer<double2>&, const double2);

// =================================================================================================
} // namespace clblast
