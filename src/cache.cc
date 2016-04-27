
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

#include <string>
#include <vector>
#include <mutex>

#include "internal/cache.h"

namespace clblast {
namespace cache {
// =================================================================================================

// Stores the compiled program in the cache
void StoreProgramToCache(const Program& program, const std::string &device_name,
                         const Precision &precision, const std::string &routine_name) {
  program_cache_mutex_.lock();
  program_cache_.push_back({program, device_name, precision, routine_name});
  program_cache_mutex_.unlock();
}

// Queries the cache and retrieves a matching program. Assumes that the match is available, throws
// otherwise.
const Program& GetProgramFromCache(const std::string &device_name, const Precision &precision,
                                   const std::string &routine_name) {
  program_cache_mutex_.lock();
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(device_name, precision, routine_name)) {
      program_cache_mutex_.unlock();
      return cached_program.program;
    }
  }
  program_cache_mutex_.unlock();
  throw std::runtime_error("Internal CLBlast error: Expected program in cache, but found none.");
}

// Queries the cache to see whether or not the compiled kernel is already there
bool ProgramIsInCache(const std::string &device_name, const Precision &precision,
                      const std::string &routine_name) {
  program_cache_mutex_.lock();
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(device_name, precision, routine_name)) {
      program_cache_mutex_.unlock();
      return true;
    }
  }
  program_cache_mutex_.unlock();
  return false;
}

// =================================================================================================

// Clears the cache of stored program binaries
StatusCode ClearCompiledProgramCache() {
  program_cache_mutex_.lock();
  program_cache_.clear();
  program_cache_mutex_.unlock();
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace cache
} // namespace clblast
