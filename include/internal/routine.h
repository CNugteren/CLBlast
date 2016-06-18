
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
#include "internal/buffer_test.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Routine {
 public:

  // Base class constructor
  explicit Routine(Queue &queue, EventPointer event, const std::string &name,
                   const std::vector<std::string> &routines, const Precision precision);

  // Set-up phase of the kernel
  StatusCode SetUp();

 protected:

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

  // Connection to the database for all the device-specific parameters
  const Database db_;
};

// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
StatusCode RunKernel(Kernel &kernel, Queue queue, const Device device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event, std::vector<Event>& waitForEvents);

// As above, but without an event waiting list
StatusCode RunKernel(Kernel &kernel, Queue queue, const Device device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event);

// =================================================================================================
} // namespace clblast

// Temporary fix: TODO place include in a more logical place
#include "internal/routines/common.h"

// CLBLAST_ROUTINE_H_
#endif
