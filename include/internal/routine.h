
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the basic functionality for the BLAS routines. This class serves as a
// base class for the actual routines (e.g. Xaxpy, Xgemm). It contains common functionality such as
// compiling the OpenCL kernel, connecting to the database, etc.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINE_H_
#define CLBLAST_ROUTINE_H_

#include <string>
#include <vector>

#include "internal/cache.h"
#include "internal/utilities.h"
#include "internal/database.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Routine {
 public:

  // Helper functions which check for errors in the status code
  static constexpr bool ErrorIn(const StatusCode s) { return (s != StatusCode::kSuccess); }

  // Base class constructor
  explicit Routine(Queue &queue, EventPointer event, const std::string &name,
                   const std::vector<std::string> &routines, const Precision precision);

  // Set-up phase of the kernel
  StatusCode SetUp();

 protected:
  
  // Runs a kernel given the global and local thread sizes
  StatusCode RunKernel(Kernel &kernel, std::vector<size_t> &global,
                       const std::vector<size_t> &local, EventPointer event,
                       std::vector<Event>& waitForEvents);

  // As above, but without an event waiting list
  StatusCode RunKernel(Kernel &kernel, std::vector<size_t> &global,
                       const std::vector<size_t> &local, EventPointer event);

  // Tests for valid inputs of matrices A, B, and C
  StatusCode TestMatrixA(const size_t one, const size_t two, const Buffer<T> &buffer,
                         const size_t offset, const size_t ld, const size_t data_size);
  StatusCode TestMatrixB(const size_t one, const size_t two, const Buffer<T> &buffer,
                         const size_t offset, const size_t ld, const size_t data_size);
  StatusCode TestMatrixC(const size_t one, const size_t two, const Buffer<T> &buffer,
                         const size_t offset, const size_t ld, const size_t data_size);
  StatusCode TestMatrixAP(const size_t n, const Buffer<T> &buffer,
                          const size_t offset, const size_t data_size);

  // Tests for valid inputs of vector X and Y
  StatusCode TestVectorX(const size_t n, const Buffer<T> &buffer, const size_t offset,
                         const size_t inc, const size_t data_size);
  StatusCode TestVectorY(const size_t n, const Buffer<T> &buffer, const size_t offset,
                         const size_t inc, const size_t data_size);

  // Tests for valid inputs of other vectors
  StatusCode TestVectorDot(const size_t n, const Buffer<T> &buffer, const size_t offset,
                           const size_t data_size);
  StatusCode TestVectorIndex(const size_t n, const Buffer<unsigned int> &buffer,
                             const size_t offset, const size_t data_size);

  // Copies/transposes a matrix and padds/unpads it with zeroes. This method is also able to write
  // to symmetric and triangular matrices through optional arguments.
  StatusCode PadCopyTransposeMatrix(EventPointer event, std::vector<Event>& waitForEvents,
                                    const size_t src_one, const size_t src_two,
                                    const size_t src_ld, const size_t src_offset,
                                    const Buffer<T> &src,
                                    const size_t dest_one, const size_t dest_two,
                                    const size_t dest_ld, const size_t dest_offset,
                                    const Buffer<T> &dest,
                                    const Program &program, const bool do_pad,
                                    const bool do_transpose, const bool do_conjugate,
                                    const bool upper = false, const bool lower = false,
                                    const bool diagonal_imag_zero = false);

  // Stores a newly compiled binary/program into the cache
  void StoreBinaryToCache(const std::string& binary) const {
    cache::StoreBinaryToCache(binary, device_name_, precision_, routine_name_);
  }
  void StoreProgramToCache(const Program& program) const {
    cache::StoreProgramToCache(program, context_, precision_, routine_name_);
  }

  // Queries the cache and retrieve either a matching binary/program or a boolean whether a match
  // exists. The first assumes that the binary/program is available in the cache and will throw an
  // exception otherwise.
  std::string GetBinaryFromCache() const {
    return cache::GetBinaryFromCache(device_name_, precision_, routine_name_);
  }
  Program GetProgramFromCache() const {
    return cache::GetProgramFromCache(context_, precision_, routine_name_);
  }
  bool BinaryIsInCache() const {
    return cache::BinaryIsInCache(device_name_, precision_, routine_name_);
  }
  bool ProgramIsInCache() const {
    return cache::ProgramIsInCache(context_, precision_, routine_name_);
  }

  // Non-static variable for the precision. Note that the same variable (but static) might exist in
  // a derived class.
  const Precision precision_;

  // The routine's name and its kernel-source in string form
  const std::string routine_name_;
  std::string source_string_;

  // The OpenCL objects, accessible only from derived classes
  Queue queue_;
  EventPointer event_;
  const Context context_;
  const Device device_;

  // OpenCL device properties
  const std::string device_name_;
  const size_t max_work_item_dimensions_;
  const std::vector<size_t> max_work_item_sizes_;
  const size_t max_work_group_size_;

  // Connection to the database for all the device-specific parameters
  const Database db_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINE_H_
#endif
