
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file implements the plugin loader for CLBlast (stubs for unsupported platforms).
//
// =================================================================================================

#include "plugin.hpp"

namespace clblast {

namespace plugin {
// =================================================================================================

Library::Library(const std::string &):
    interface_(nullptr) {
}

bool Library::IsValid() const {
  return false;
}

const Interface &Library::GetInterface() const {
  throw RuntimeError("plugin::Library: attempt to use a negative cache entry");
}

// =================================================================================================
} // namespace plugin

} // namespace clblast


