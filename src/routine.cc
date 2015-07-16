
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

#include "internal/routine.h"

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// The cache of compiled OpenCL programs
std::vector<Routine::ProgramCache> Routine::program_cache_;

// Constructor: not much here, because no status codes can be returned
Routine::Routine(CommandQueue &queue, Event &event,
                 const std::vector<std::string> &routines, const Precision precision):
    precision_(precision),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    device_name_(device_.Name()),
    max_work_item_dimensions_(device_.MaxWorkItemDimensions()),
    max_work_item_sizes_(device_.MaxWorkItemSizes()),
    max_work_group_size_(device_.MaxWorkGroupSize()),
    db_(queue_, routines, precision_),
    routines_(routines) {
}

// =================================================================================================

// Separate set-up function to allow for status codes to be returned
StatusCode Routine::SetUp(const std::string &routine_source) {

  // Queries the cache to see whether or not the compiled kernel is already there. If not, it will
  // be built and added to the cache.
  if (!ProgramIsInCache()) {

    // Inspects whether or not cl_khr_fp64 is supported in case of double precision
    auto extensions = device_.Extensions();
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

    // Collects the parameters for this device in the form of defines, and adds the precision
    auto defines = db_.GetDefines();
    defines += "#define PRECISION "+ToString(static_cast<int>(precision_))+"\n";

    // For specific devices, use the non-IEE754 compilant OpenCL mad() instruction. This can improve
    // performance, but might result in a reduced accuracy.
    if (device_.Vendor() == "AMD") {
      defines += "#define USE_CL_MAD 1\n";
    }

    // Combines everything together into a single source string
    auto source_string = defines + common_header + routine_source;

    // Compiles the kernel
    try {
      auto program = Program(context_, source_string);
      auto options = std::string{};
      auto status = program.Build(device_, options);

      // Checks for compiler crashes/errors/warnings
      if (status == CL_BUILD_PROGRAM_FAILURE) {
        auto message = program.GetBuildInfo(device_);
        fprintf(stdout, "OpenCL compiler error/warning: %s\n", message.c_str());
        return StatusCode::kBuildProgramFailure;
      }
      if (status == CL_INVALID_BINARY) { return StatusCode::kInvalidBinary; }

      // Store the compiled program in the cache
      program_cache_.push_back({program, device_name_, precision_, routines_});
    } catch (...) { return StatusCode::kBuildProgramFailure; }
  }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
StatusCode Routine::RunKernel(const Kernel &kernel, std::vector<size_t> &global,
                              const std::vector<size_t> &local) {

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
  auto status = queue_.EnqueueKernel(kernel, global, local, event_);
  if (status != CL_SUCCESS) { return StatusCode::kKernelLaunchError; }

  // Waits for completion of the kernel
  status = event_.Wait();
  if (status != CL_SUCCESS) { return StatusCode::kKernelRunError; }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests matrix A for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
StatusCode Routine::TestMatrixA(const size_t one, const size_t two, const Buffer &buffer,
                                const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimA; }
  try {
    auto required_size = (ld*two + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }
  return StatusCode::kSuccess;
}

// Tests matrix B for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
StatusCode Routine::TestMatrixB(const size_t one, const size_t two, const Buffer &buffer,
                                const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimB; }
  try {
    auto required_size = (ld*two + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryB; }
  } catch (...) { return StatusCode::kInvalidMatrixB; }
  return StatusCode::kSuccess;
}

// Tests matrix C for validity: checks for a valid OpenCL buffer, a valid lead-dimension, and for a
// sufficient buffer size.
StatusCode Routine::TestMatrixC(const size_t one, const size_t two, const Buffer &buffer,
                                const size_t offset, const size_t ld, const size_t data_size) {
  if (ld < one) { return StatusCode::kInvalidLeadDimC; }
  try {
    auto required_size = (ld*two + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryC; }
  } catch (...) { return StatusCode::kInvalidMatrixC; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests vector X for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
StatusCode Routine::TestVectorX(const size_t n, const Buffer &buffer, const size_t offset,
                                const size_t inc, const size_t data_size) {
  if (inc == 0) { return StatusCode::kInvalidIncrementX; }
  try {
    auto required_size = (n*inc + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryX; }
  } catch (...) { return StatusCode::kInvalidVectorX; }
  return StatusCode::kSuccess;
}

// Tests vector Y for validity: checks for a valid increment, a valid OpenCL buffer, and for a
// sufficient buffer size.
StatusCode Routine::TestVectorY(const size_t n, const Buffer &buffer, const size_t offset,
                                const size_t inc, const size_t data_size) {
  if (inc == 0) { return StatusCode::kInvalidIncrementY; }
  try {
    auto required_size = (n*inc + offset)*data_size;
    auto buffer_size = buffer.GetSize();
    if (buffer_size < required_size) { return StatusCode::kInsufficientMemoryY; }
  } catch (...) { return StatusCode::kInvalidVectorY; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Copies or transposes a matrix and pads/unpads it with zeros
StatusCode Routine::PadCopyTransposeMatrix(const size_t src_one, const size_t src_two,
                                           const size_t src_ld, const size_t src_offset,
                                           const Buffer &src,
                                           const size_t dest_one, const size_t dest_two,
                                           const size_t dest_ld, const size_t dest_offset,
                                           const Buffer &dest,
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
        status = RunKernel(kernel, global, local);
      }
      else {
        auto global = std::vector<size_t>{Ceil(CeilDiv(dest_one, db_["PADTRA_WPT"]), db_["PADTRA_TILE"]),
                                          Ceil(CeilDiv(dest_two, db_["PADTRA_WPT"]), db_["PADTRA_TILE"])};
        auto local = std::vector<size_t>{db_["PADTRA_TILE"], db_["PADTRA_TILE"]};
        status = RunKernel(kernel, global, local);
      }
    }
    else {
      if (use_fast_kernel) {
        auto global = std::vector<size_t>{dest_one / db_["COPY_VW"],
                                          dest_two / db_["COPY_WPT"]};
        auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
        status = RunKernel(kernel, global, local);
      }
      else {
        auto global = std::vector<size_t>{Ceil(CeilDiv(dest_one, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                          Ceil(CeilDiv(dest_two, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
        auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
        status = RunKernel(kernel, global, local);
      }
    }
    return status;
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================

// Queries the cache and retrieves a matching program. Assumes that the match is available, throws
// otherwise.
const Program& Routine::GetProgramFromCache() const {
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(device_name_, precision_, routines_)) {
      return cached_program.program;
    }
  }
  throw std::runtime_error("Internal CLBlast error: Expected program in cache, but found none.");
}

// Queries the cache to see whether or not the compiled kernel is already there
bool Routine::ProgramIsInCache() const {
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(device_name_, precision_, routines_)) { return true; }
  }
  return false;
}

// =================================================================================================
} // namespace clblast
