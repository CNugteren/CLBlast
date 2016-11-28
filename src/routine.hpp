
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

#include "utilities/utilities.hpp"
#include "cache.hpp"
#include "utilities/buffer_test.hpp"
#include "database/database.hpp"
#include "routines/common.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Routine {
 public:

  // Base class constructor. The user database is an optional extra database to override the
  // built-in database.
  // All heavy preparation work is done inside this constructor.
  // NOTE: the caller must provide the same userDatabase for each combination of device, precision
  // and routine list, otherwise the caching logic will break.
  explicit Routine(Queue &queue, EventPointer event, const std::string &name,
                   const std::vector<std::string> &routines, const Precision precision,
                   const std::vector<const Database::DatabaseEntry*> &userDatabase,
                   std::initializer_list<const char *> source);

 private:

  // Initializes program_, fetching cached program or building one
  void InitProgram(std::initializer_list<const char *> source);

  // Initializes db_, fetching cached database or building one
  void InitDatabase(const std::vector<std::string> &routines,
                    const std::vector<const Database::DatabaseEntry*> &userDatabase);

 protected:

  // Non-static variable for the precision
  const Precision precision_;

  // The routine's name
  const std::string routine_name_;

  // The OpenCL objects, accessible only from derived classes
  Queue queue_;
  EventPointer event_;
  const Context context_;
  const Device device_;

  // OpenCL device properties
  const std::string device_name_;

  // Compiled program (either retrieved from cache or compiled in slow path)
  Program program_;

  // Connection to the database for all the device-specific parameters
  Database db_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINE_H_
#endif
