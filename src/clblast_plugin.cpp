
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file contains definitions for the plugin API.
//
// =================================================================================================

#include "utilities/plugin.hpp"
#include "clblast_plugin.h"

namespace clblast {

namespace plugin {
// =================================================================================================

Base::~Base() = default;

Interface::~Interface() = default;

Routine::~Routine() = default;

// =================================================================================================

Routine::Routine():
    kernel_mode(KernelMode::Default)
{}

// =================================================================================================

const Routine *Plugin::PickStubRoutine() {

  // This class must inherit from all device-specific entry subclasses
  struct RoutineStub : virtual public Routine {
  } static stub_routine;

  return &stub_routine;
}

// =================================================================================================
} // namespace plugin

} // namespace clblast

