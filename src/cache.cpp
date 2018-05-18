
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

#include "database/database.hpp"
#include "cache.hpp"

namespace clblast {
// =================================================================================================

template <typename Key, typename Value>
template <typename U>
Value Cache<Key, Value>::Get(const U &key, bool *in_cache) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

#if __cplusplus >= 201402L
  // generalized std::map::find() of C++14
  auto it = cache_.find(key);
#else
  // O(n) lookup in a vector
  auto it = std::find_if(cache_.begin(), cache_.end(), [&] (const std::pair<Key, Value> &pair) {
    return pair.first == key;
  });
#endif
  if (it == cache_.end()) {
    if (in_cache) {
      *in_cache = false;
    }
    return Value();
  }

  if (in_cache) {
    *in_cache = true;
  }
  return it->second;
}

template <typename Key, typename Value>
void Cache<Key, Value>::Store(Key &&key, Value &&value) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

#if __cplusplus >= 201402L
  // emplace() into a map
  auto r = cache_.emplace(std::move(key), std::move(value));
  if (!r.second) {
    throw LogicError("Cache::Store: object already in cache");
  }
#else
  // emplace_back() into a vector
  cache_.emplace_back(std::move(key), std::move(value));
#endif
}

template <typename Key, typename Value>
void Cache<Key, Value>::Remove(const Key &key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
#if __cplusplus >= 201402L
  cache_.erase(key);
#else
  auto it = cache_.begin();
  while (it != cache_.end()) {
    if ((*it).first == key) {
      it = cache_.erase(it);
    }
    else ++it;
  }
#endif
}

template <typename Key, typename Value>
template <int I1, int I2>
void Cache<Key, Value>::RemoveBySubset(const Key &key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = cache_.begin();
  while (it != cache_.end()) {
    const auto current_key = (*it).first;
    if ((std::get<I1>(key) == std::get<I1>(current_key)) &&
        (std::get<I2>(key) == std::get<I2>(current_key))) {
      it = cache_.erase(it);
    }
    else ++it;
  }
}

template <typename Key, typename Value>
void Cache<Key, Value>::Invalidate() {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  cache_.clear();
}

template <typename Key, typename Value>
Cache<Key, Value> &Cache<Key, Value>::Instance() {
  return instance_;
}

template <typename Key, typename Value>
Cache<Key, Value> Cache<Key, Value>::instance_;

// =================================================================================================

template class Cache<BinaryKey, std::string>;
template std::string BinaryCache::Get(const BinaryKeyRef &, bool *) const;

// =================================================================================================

template class Cache<ProgramKey, std::shared_ptr<Program>>;
template std::shared_ptr<Program> ProgramCache::Get(const ProgramKeyRef &, bool *) const;
template void ProgramCache::RemoveBySubset<1, 2>(const ProgramKey &); // precision and routine name

// =================================================================================================

template class Cache<DatabaseKey, Database>;
template Database DatabaseCache::Get(const DatabaseKeyRef &, bool *) const;

// =================================================================================================
} // namespace clblast
