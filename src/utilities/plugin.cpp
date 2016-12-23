
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

#include "cache.hpp"
#include "plugin.hpp"

namespace clblast {

namespace plugin {
// =================================================================================================

Plugin::Plugin(const Device &device, const std::string &routine, const Precision precision):
    routine_(nullptr) {

  // TODO: maybe add certain device information to the plugin naming scheme to make it possible
  // to have multiple plugin libraries for different hardware at the same time
  static std::string library_name = "clblast_plugin";

  InitLibrary(library_name);

  if (library_.IsValid()) {
    routine_ = library_.GetInterface().GetRoutine(device(), routine, precision);
  }

  if (routine_ == nullptr) {
    routine_ = PickStubRoutine();
  }
}

void Plugin::InitLibrary(const std::string &library_name) {

  bool has_library;
  library_ = PluginLibraryCache::Instance().Get(library_name, &has_library);
  if (has_library) {
    return;
  }

  library_ = Library(library_name);
  PluginLibraryCache::Instance().Store(std::string{ library_name }, Library{ library_ });
}

// NOTE: Plugin::PickStubRoutine() is defined in clblast_plugin.cpp because it needs to
// instantiate per-routine stub methods, which are also defined there.

// =================================================================================================
} // namespace plugin

} // namespace clblast
