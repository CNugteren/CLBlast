
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the caching functionality of compiled binaries and programs.
//
// =================================================================================================

#ifndef CLBLAST_CACHE_H_
#define CLBLAST_CACHE_H_

#include <string>
#include <vector>
#include <mutex>

#include "utilities.hpp"

namespace clblast {
// =================================================================================================

// The cache of compiled OpenCL binaries, along with some meta-data
struct BinaryCache {
  std::string binary;
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
static std::vector<BinaryCache> binary_cache_;
static std::mutex binary_cache_mutex_;

// =================================================================================================

// The cache of compiled OpenCL programs, along with some meta-data
struct ProgramCache {
  Program program;
  cl_context context;
  Precision precision;
  std::string routine_name_;

  // Finds out whether the properties match
  bool MatchInCache(const cl_context ref_context, const Precision &ref_precision,
                    const std::string &ref_routine) {
    return (context == ref_context &&
            precision == ref_precision &&
            routine_name_ == ref_routine);
  }
};

// The actual cache, implemented as a vector of the above data-type, and its mutex
static std::vector<ProgramCache> program_cache_;
static std::mutex program_cache_mutex_;

// =================================================================================================

// Stores the compiled binary or program in the cache
void StoreBinaryToCache(const std::string &binary, const std::string &device_name,
                        const Precision &precision, const std::string &routine_name);
void StoreProgramToCache(const Program &program, const Context &context,
                         const Precision &precision, const std::string &routine_name);

// Queries the cache and retrieves a matching binary or program. Assumes that the match is
// available, throws otherwise.
const std::string& GetBinaryFromCache(const std::string &device_name, const Precision &precision,
                                      const std::string &routine_name);
const Program& GetProgramFromCache(const Context &context, const Precision &precision,
                                   const std::string &routine_name);

// Queries the cache to see whether or not the compiled kernel is already there
bool BinaryIsInCache(const std::string &device_name, const Precision &precision,
                     const std::string &routine_name);
bool ProgramIsInCache(const Context &context, const Precision &precision,
                      const std::string &routine_name);

// =================================================================================================

// Clears the cache of stored binaries
void CacheClearAll();

// =================================================================================================
} // namespace clblast

// CLBLAST_CACHE_H_
#endif
