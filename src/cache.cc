
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

#include <string>
#include <vector>
#include <mutex>

#include "internal/cache.h"

namespace clblast {
// =================================================================================================

// Stores the compiled binary or IR in the cache
void StoreBinaryToCache(const std::string &binary, const std::string &device_name,
                        const Precision &precision, const std::string &routine_name) {
  binary_cache_mutex_.lock();
  binary_cache_.push_back(BinaryCache{binary, device_name, precision, routine_name});
  binary_cache_mutex_.unlock();
}

// Stores the compiled program in the cache
void StoreProgramToCache(const Program &program, const Context &context,
                         const Precision &precision, const std::string &routine_name) {
  program_cache_mutex_.lock();
  program_cache_.push_back(ProgramCache{program, context.pointer(), precision, routine_name});
  program_cache_mutex_.unlock();
}

// Queries the cache and retrieves a matching binary. Assumes that the match is available, throws
// otherwise.
const std::string& GetBinaryFromCache(const std::string &device_name, const Precision &precision,
                                      const std::string &routine_name) {
  binary_cache_mutex_.lock();
  for (auto &cached_binary: binary_cache_) {
    if (cached_binary.MatchInCache(device_name, precision, routine_name)) {
      binary_cache_mutex_.unlock();
      return cached_binary.binary;
    }
  }
  binary_cache_mutex_.unlock();
  throw std::runtime_error("Internal CLBlast error: Expected binary in cache, but found none.");
}

// Queries the cache and retrieves a matching program. Assumes that the match is available, throws
// otherwise.
const Program& GetProgramFromCache(const Context &context, const Precision &precision,
                                   const std::string &routine_name) {
  program_cache_mutex_.lock();
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(context.pointer(), precision, routine_name)) {
      program_cache_mutex_.unlock();
      return cached_program.program;
    }
  }
  program_cache_mutex_.unlock();
  throw std::runtime_error("Internal CLBlast error: Expected program in cache, but found none.");
}

// Queries the cache to see whether or not the compiled kernel is already there
bool BinaryIsInCache(const std::string &device_name, const Precision &precision,
                     const std::string &routine_name) {
  binary_cache_mutex_.lock();
  for (auto &cached_binary: binary_cache_) {
    if (cached_binary.MatchInCache(device_name, precision, routine_name)) {
      binary_cache_mutex_.unlock();
      return true;
    }
  }
  binary_cache_mutex_.unlock();
  return false;
}

// Queries the cache to see whether or not the compiled kernel is already there
bool ProgramIsInCache(const Context &context, const Precision &precision,
                      const std::string &routine_name) {
  program_cache_mutex_.lock();
  for (auto &cached_program: program_cache_) {
    if (cached_program.MatchInCache(context.pointer(), precision, routine_name)) {
      program_cache_mutex_.unlock();
      return true;
    }
  }
  program_cache_mutex_.unlock();
  return false;
}

// =================================================================================================

// Clears the cache of stored binaries and programs
StatusCode CacheClearAll() {
  binary_cache_mutex_.lock();
  binary_cache_.clear();
  binary_cache_mutex_.unlock();
  program_cache_mutex_.lock();
  program_cache_.clear();
  program_cache_mutex_.unlock();
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace clblast
