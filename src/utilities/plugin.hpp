
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file implements the plugin loader for CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_PLUGIN_H_
#define CLBLAST_PLUGIN_H_

#include "utilities/utilities.hpp"
#include "clblast_plugin.h"

namespace clblast {

namespace plugin {
// =================================================================================================

// Implements a managing wrapper around a single plugin shared object or absence thereof
// (thus allowing a form of a negative cache).
class Library {
 public:
  Library() = default;

  // Loads a shared object by a platform-independent common name stem (such as "clblast_plugin"),
  // which is transformed into a platform-specific file name or path ("./libclblast_plugin.so").
  Library(const std::string &name);

  bool IsValid() const;

  const Interface &GetInterface() const;

 private:
  // It is pretty unfortunate that we have to do refcounting on top of refcounting, but well,
  // this seems to be the pattern for the rest of the library. Moreover, <dlfcn.h> does not
  // provide us with an explicit "retain" operation on an existing handle, and there is no
  // substitute on Windows, so here we go.
  // Note that we use a std::shared_ptr<void> (which holds a pointer of type void *) with a
  // custom deleter that calls dlclose(), thereby avoiding an allocation for the pointer itself.
  std::shared_ptr<void> handle_;
  const plugin::Interface *interface_;
};

// =================================================================================================

// Implements a managing wrapper around a single device- and routine-specific plugin entry.
// If none is found, a static stub entry is returned.
class Plugin {
 public:
  Plugin() = default;

  // Acquires a plugin object and entry for a specific device and routine.
  Plugin(const Device &device, const std::string &routine, const Precision precision);

  // Returns a suitably down-casted plugin entry (or a stub entry if there is no match).
  template <typename T>
  const T &GetRoutine() const {

    const T *routine = dynamic_cast<const T *>(routine_);
    if (routine != nullptr) {
      return *routine;
    }

    routine = dynamic_cast<const T *>(PickStubRoutine());
    if (routine != nullptr) {
      return *routine;
    }

    throw LogicError("plugin::Plugin: bad entry type requested");
  }

 private:

  // Returns a stub entry for given routine
  static const Routine *PickStubRoutine();

  // Initializes library_, fetching it from cache or opening a new one
  void InitLibrary(const std::string &library_name);

  Library library_;
  const Routine *routine_;
}; // class Plugin

// =================================================================================================

} // namespace plugin

} // namespace clblast

// CLBLAST_PLUGIN_H_
#endif
