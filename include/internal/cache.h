
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the caching functionality of compiled binaries.
//
// =================================================================================================

#ifndef CLBLAST_CACHE_H_
#define CLBLAST_CACHE_H_

#include <string>
#include <vector>
#include <mutex>

#include "internal/utilities.h"

namespace clblast {
namespace cache {
// =================================================================================================

// The cache of compiled OpenCL programs, along with some meta-data
struct ProgramCache {
  Program program;
  std::string device_name;
  Precision precision;
  std::string routine_name_;

  // Finds out whether the properties match
  bool MatchInCache(const std::string &ref_device, const Precision &ref_precision,
                    const std::string &ref_routine) {
    return (device_name == ref_device &&
            precision == ref_precision &&
            routine_name_ == ref_routine);
  }
};

// The actual cache, implemented as a vector of the above data-type, and its mutex
static std::vector<ProgramCache> program_cache_;
static std::mutex program_cache_mutex_;

// =================================================================================================

// Stores the compiled program in the cache
void StoreProgramToCache(const Program& program, const std::string &device_name,
                         const Precision &precision, const std::string &routine_name);

// Queries the cache and retrieves a matching program. Assumes that the match is available, throws
// otherwise.
const Program& GetProgramFromCache(const std::string &device_name, const Precision &precision,
                                   const std::string &routine_name);

// Queries the cache to see whether or not the compiled kernel is already there
bool ProgramIsInCache(const std::string &device_name, const Precision &precision,
                      const std::string &routine_name);

// =================================================================================================

// Clears the cache of stored program binaries
StatusCode ClearCompiledProgramCache();

// =================================================================================================
} // namespace cache
} // namespace clblast

// CLBLAST_CACHE_H_
#endif
