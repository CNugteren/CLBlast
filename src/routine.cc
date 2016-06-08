
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
    max_work_item_dimensions_(device_.MaxWorkItemDimensions()),
    max_work_item_sizes_(device_.MaxWorkItemSizes()),
    max_work_group_size_(device_.MaxWorkGroupSize()),
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
  auto extensions = device_.Capabilities();
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

  // Determines whether this is a specific device
  const auto isAMD = device_.Vendor() == "AMD" || device_.Vendor() == "Advanced Micro Devices, Inc.";
  const auto isARM = device_.Vendor() == "ARM";
  const auto isGPU = device_.Type() == "GPU";

  // For specific devices, use the non-IEE754 compilant OpenCL mad() instruction. This can improve
  // performance, but might result in a reduced accuracy.
  if (isAMD && isGPU) {
    defines += "#define USE_CL_MAD 1\n";
  }

  // For specific devices, use staggered/shuffled workgroup indices.
  if (isAMD && isGPU) {
    defines += "#define USE_STAGGERED_INDICES 1\n";
  }

  // For specific devices add a global synchronisation barrier to the GEMM kernel to optimize
  // performance through better cache behaviour
  if (isARM && isGPU) {
    defines += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // Combines everything together into a single source string
  auto source_string = defines + common_header + source_string_;

  // Compiles the kernel
  try {
    auto program = Program(context_, source_string);
    auto options = std::vector<std::string>();
    auto build_status = program.Build(device_, options);

    // Checks for compiler crashes/errors/warnings
    if (build_status == BuildStatus::kError) {
      auto message = program.GetBuildInfo(device_);
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
template <typename T>
StatusCode Routine<T>::RunKernel(Kernel &kernel, std::vector<size_t> &global,
                                 const std::vector<size_t> &local, EventPointer event,
                                 std::vector<Event>& waitForEvents) {

  // Tests for validity of the local thread sizes
  if (local.size() > max_work_item_dimensions_) {
    return StatusCode::kInvalidLocalNumDimensions; 
  }
  for (auto i=size_t{0}; i<local.size(); ++i) {
    if (local[i] > max_work_item_sizes_[i]) { return StatusCode::kInvalidLocalThreadsDim; }
  }
  auto local_size = size_t{1};
  for (auto &item: local) { local_size *= item; }
  if (local_size > max_work_group_size_) { return StatusCode::kInvalidLocalThreadsTotal; }

  // Make sure the global thread sizes are at least equal to the local sizes
  for (auto i=size_t{0}; i<global.size(); ++i) {
    if (global[i] < local[i]) { global[i] = local[i]; }
  }

  // Tests for local memory usage
  auto local_mem_usage = kernel.LocalMemUsage(device_);
  if (!device_.IsLocalMemoryValid(local_mem_usage)) { return StatusCode::kInvalidLocalMemUsage; }

  // Launches the kernel (and checks for launch errors)
  try {
    kernel.Launch(queue_, global, local, event, waitForEvents);
  } catch (...) { return StatusCode::kKernelLaunchError; }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// As above, but without an event waiting list
template <typename T>
StatusCode Routine<T>::RunKernel(Kernel &kernel, std::vector<size_t> &global,
                                 const std::vector<size_t> &local, EventPointer event) {
  auto emptyWaitingList = std::vector<Event>();
  return RunKernel(kernel, global, local, event, emptyWaitingList);
}

// =================================================================================================

// Tests matrix A for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestMatrixA(const size_t one, const size_t two, const Buffer<T> &buffer,
                                   const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimA; }
  try {
    auto required_size = (ld*(two-1) + one + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }
  return StatusCode::kSuccess;
}

// Tests matrix B for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestMatrixB(const size_t one, const size_t two, const Buffer<T> &buffer,
                                   const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimB; }
  try {
    auto required_size = (ld*(two-1) + one + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryB; }
  } catch (...) { return StatusCode::kInvalidMatrixB; }
  return StatusCode::kSuccess;
}

// Tests matrix C for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestMatrixC(const size_t one, const size_t two, const Buffer<T> &buffer,
                                   const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimC; }
  try {
    auto required_size = (ld*(two-1) + one + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryC; }
  } catch (...) { return StatusCode::kInvalidMatrixC; }
  return StatusCode::kSuccess;
}

// Tests matrix AP for validity: checks for a valid OpenCL buffer and for a sufficient buffer size
template <typename T>
StatusCode Routine<T>::TestMatrixAP(const size_t n, const Buffer<T> &buffer,
                                    const size_t offset, const size_t data_size) {
  try {
    auto required_size = (((n*(n+1))/2) + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests vector X for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestVectorX(const size_t n, const Buffer<T> &buffer, const size_t offset,
                                   const size_t inc, const size_t data_size) {
  if (inc == 0) { return StatusCode::kInvalidIncrementX; }
  try {
    auto required_size = ((n-1)*inc + 1 + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryX; }
  } catch (...) { return StatusCode::kInvalidVectorX; }
  return StatusCode::kSuccess;
}

// Tests vector Y for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestVectorY(const size_t n, const Buffer<T> &buffer, const size_t offset,
                                   const size_t inc, const size_t data_size) {
  if (inc == 0) { return StatusCode::kInvalidIncrementY; }
  try {
    auto required_size = ((n-1)*inc + 1 + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryY; }
  } catch (...) { return StatusCode::kInvalidVectorY; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests vector dot for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestVectorDot(const size_t n, const Buffer<T> &buffer, const size_t offset,
                                     const size_t data_size) {
  try {
    auto required_size = (n + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryDot; }
  } catch (...) { return StatusCode::kInvalidVectorDot; }
  return StatusCode::kSuccess;
}

// Tests vector index for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
template <typename T>
StatusCode Routine<T>::TestVectorIndex(const size_t n, const Buffer<unsigned int> &buffer,
                                       const size_t offset, const size_t data_size) {
  try {
    auto required_size = (n + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryDot; }
  } catch (...) { return StatusCode::kInvalidVectorDot; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Copies or transposes a matrix and pads/unpads it with zeros
template <typename T>
StatusCode Routine<T>::PadCopyTransposeMatrix(EventPointer event, std::vector<Event>& waitForEvents,
                                              const size_t src_one, const size_t src_two,
                                              const size_t src_ld, const size_t src_offset,
                                              const Buffer<T> &src,
                                              const size_t dest_one, const size_t dest_two,
                                              const size_t dest_ld, const size_t dest_offset,
                                              const Buffer<T> &dest,
                                              const Program &program, const bool do_pad,
                                              const bool do_transpose, const bool do_conjugate,
                                              const bool upper, const bool lower,
                                              const bool diagonal_imag_zero) {

  // Determines whether or not the fast-version could potentially be used
  auto use_fast_kernel = (src_offset == 0) && (dest_offset == 0) && (do_conjugate == false) &&
                         (src_one == dest_one) && (src_two == dest_two) && (src_ld == dest_ld) &&
                         (upper == false) && (lower == false) && (diagonal_imag_zero == false);

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db_["TRA_WPT"]) &&
        IsMultiple(src_one, db_["TRA_WPT"]*db_["TRA_WPT"]) &&
        IsMultiple(src_two, db_["TRA_WPT"]*db_["TRA_WPT"])) {
      kernel_name = "TransposeMatrix";
    }
    else {
      use_fast_kernel = false;
      kernel_name = (do_pad) ? "PadTransposeMatrix" : "UnPadTransposeMatrix";
    }
  }
  else {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db_["COPY_VW"]) &&
        IsMultiple(src_one, db_["COPY_VW"]*db_["COPY_DIMX"]) &&
        IsMultiple(src_two, db_["COPY_WPT"]*db_["COPY_DIMY"])) {
      kernel_name = "CopyMatrix";
    }
    else {
      use_fast_kernel = false;
      kernel_name = (do_pad) ? "PadMatrix" : "UnPadMatrix";
    }
  }

  // Retrieves the kernel from the compiled binary
  try {
    auto kernel = Kernel(program, kernel_name);

    // Sets the kernel arguments
    if (use_fast_kernel) {
      kernel.SetArgument(0, static_cast<int>(src_ld));
      kernel.SetArgument(1, src());
      kernel.SetArgument(2, dest());
    }
    else {
      kernel.SetArgument(0, static_cast<int>(src_one));
      kernel.SetArgument(1, static_cast<int>(src_two));
      kernel.SetArgument(2, static_cast<int>(src_ld));
      kernel.SetArgument(3, static_cast<int>(src_offset));
      kernel.SetArgument(4, src());
      kernel.SetArgument(5, static_cast<int>(dest_one));
      kernel.SetArgument(6, static_cast<int>(dest_two));
      kernel.SetArgument(7, static_cast<int>(dest_ld));
      kernel.SetArgument(8, static_cast<int>(dest_offset));
      kernel.SetArgument(9, dest());
      if (do_pad) {
        kernel.SetArgument(10, static_cast<int>(do_conjugate));
      }
      else {
        kernel.SetArgument(10, static_cast<int>(upper));
        kernel.SetArgument(11, static_cast<int>(lower));
        kernel.SetArgument(12, static_cast<int>(diagonal_imag_zero));
      }
    }

    // Launches the kernel and returns the error code. Uses global and local thread sizes based on
    // parameters in the database.
    auto status = StatusCode::kSuccess;
    if (do_transpose) {
      if (use_fast_kernel) {
        auto global = std::vector<size_t>{dest_one / db_["TRA_WPT"],
                                          dest_two / db_["TRA_WPT"]};
        auto local = std::vector<size_t>{db_["TRA_DIM"], db_["TRA_DIM"]};
        status = RunKernel(kernel, global, local, event, waitForEvents);
      }
      else {
        auto global = std::vector<size_t>{Ceil(CeilDiv(dest_one, db_["PADTRA_WPT"]), db_["PADTRA_TILE"]),
                                          Ceil(CeilDiv(dest_two, db_["PADTRA_WPT"]), db_["PADTRA_TILE"])};
        auto local = std::vector<size_t>{db_["PADTRA_TILE"], db_["PADTRA_TILE"]};
        status = RunKernel(kernel, global, local, event, waitForEvents);
      }
    }
    else {
      if (use_fast_kernel) {
        auto global = std::vector<size_t>{dest_one / db_["COPY_VW"],
                                          dest_two / db_["COPY_WPT"]};
        auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
        status = RunKernel(kernel, global, local, event, waitForEvents);
      }
      else {
        auto global = std::vector<size_t>{Ceil(CeilDiv(dest_one, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                          Ceil(CeilDiv(dest_two, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
        auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
        status = RunKernel(kernel, global, local, event, waitForEvents);
      }
    }
    return status;
  } catch (...) { return StatusCode::kInvalidKernel; }
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
