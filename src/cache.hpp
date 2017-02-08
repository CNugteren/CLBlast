
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
#include <mutex>
#include <map>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// The generic thread-safe cache. We assume that the Key may be a heavyweight struct that is not
// normally used by the caller, while the Value is either lightweight or ref-counted.
// Hence, searching by non-Key is supported (if there is a corresponding operator<()), and
// on Store() the Key instance is moved from the caller (because it will likely be constructed
// as temporary at the time of Store()).
template <typename Key, typename Value>
class Cache {
public:
  // Cached object is returned by-value to avoid racing with Invalidate().
  // Due to lack of std::optional<>, in case of a cache miss we return a default-constructed
  // Value and set the flag to false.
  template <typename U>
  Value Get(const U &key, bool *in_cache) const;

  // We do not return references to just stored object to avoid racing with Invalidate().
  // Caller is expected to store a temporary.
  void Store(Key &&key, Value &&value);
  void Invalidate();

  static Cache<Key, Value> &Instance();

private:
#if __cplusplus >= 201402L
  // The std::less<void> allows to search in cache by an object comparable with Key, without
  // constructing a temporary Key
  // (see http://en.cppreference.com/w/cpp/utility/functional/less_void,
  //      http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2013/n3657.htm,
  //      http://stackoverflow.com/questions/10536788/avoiding-key-construction-for-stdmapfind)
  std::map<Key, Value, std::less<void>> cache_;
#else
  std::vector<std::pair<Key, Value>> cache_;
#endif
  mutable std::mutex cache_mutex_;

  static Cache<Key, Value> instance_;
}; // class Cache

// =================================================================================================

// The key struct for the cache of compiled OpenCL binaries
// Order of fields: precision, routine_name, device_name (smaller fields first)
typedef std::tuple<Precision, std::string, std::string> BinaryKey;
typedef std::tuple<const Precision &, const std::string &, const std::string &> BinaryKeyRef;

typedef Cache<BinaryKey, std::string> BinaryCache;

extern template class Cache<BinaryKey, std::string>;
extern template std::string BinaryCache::Get(const BinaryKeyRef &, bool *) const;


// =================================================================================================

// The key struct for the cache of compiled OpenCL programs (context-dependent)
// Order of fields: context, precision, routine_name (smaller fields first)
typedef std::tuple<cl_context, Precision, std::string> ProgramKey;
typedef std::tuple<const cl_context &, const Precision &, const std::string &> ProgramKeyRef;

typedef Cache<ProgramKey, Program> ProgramCache;

extern template class Cache<ProgramKey, Program>;
extern template Program ProgramCache::Get(const ProgramKeyRef &, bool *) const;

// =================================================================================================

class Database;

// The key struct for the cache of database maps.
// Order of fields: precision, device_name, routines (smaller fields first)
typedef std::tuple<Precision, std::string, std::vector<std::string>> DatabaseKey;
typedef std::tuple<const Precision &, const std::string &, const std::vector<std::string> &> DatabaseKeyRef;

typedef Cache<DatabaseKey, Database> DatabaseCache;

extern template class Cache<DatabaseKey, Database>;
extern template Database DatabaseCache::Get(const DatabaseKeyRef &, bool *) const;

// =================================================================================================

namespace plugin {

class Library;

} // namespace plugin

typedef Cache<std::string, plugin::Library> PluginLibraryCache;

extern template class Cache<std::string, plugin::Library>;
extern template plugin::Library PluginLibraryCache::Get(const std::string &, bool *) const;

// =================================================================================================

namespace plugin {

class Plugin;

} // namespace plugin

// order of fields: precision, routine_name, device_name (smaller fields first)
typedef std::tuple<Precision, std::string, std::string> PluginKey;
typedef std::tuple<const Precision &, const std::string &, const std::string &> PluginKeyRef;

typedef Cache<PluginKey, plugin::Plugin> PluginCache;

extern template class Cache<PluginKey, plugin::Plugin>;
extern template plugin::Plugin PluginCache::Get(const PluginKeyRef &, bool *) const;

// =================================================================================================
} // namespace clblast

// CLBLAST_CACHE_H_
#endif
