
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file implements the plugin loader for CLBlast (POSIX-specific bits).
//
// =================================================================================================

#include <dlfcn.h>

#include "plugin.hpp"

namespace clblast {

namespace plugin {
// =================================================================================================

Library::Library(const std::string &name):
    interface_(nullptr) {

  std::string unix_name = "lib" + name + ".so";

  void *unix_handle = dlopen(unix_name.c_str(), RTLD_LAZY);
  if (unix_handle == nullptr) {
    fprintf(stderr, "CLBLast (plugin): could not load '%s': dlopen(): %s\n",
            unix_name.c_str(), dlerror());
    return;
  }

  std::shared_ptr<void> handle(unix_handle, [] (void *arg) { dlclose(arg); });

  Base *base = *reinterpret_cast<Base **>(dlsym(unix_handle, "clblast_plugin"));
  if (base == nullptr) {
    fprintf(stderr, "CLBlast (plugin): could not load '%s': dlsym(\"clblast_plugin\"): %s\n",
            unix_name.c_str(), dlerror());
    return;
  }

  Interface *interface = dynamic_cast<Interface *>(base);
  if (interface == nullptr) {
    fprintf(stderr, "CLBlast (plugin): could not load '%s': ABI mismatch\n",
            unix_name.c_str());
    return;
  }

  handle_ = handle;
  interface_ = interface;
}

bool Library::IsValid() const {
  return interface_ != nullptr;
}

const Interface &Library::GetInterface() const {
  if (!IsValid()) {
    throw RuntimeError("plugin::Library: attempt to use a negative cache entry");
  }
  return *interface_;
}

// =================================================================================================
} // namespace plugin

} // namespace clblast

