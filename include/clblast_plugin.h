
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file contains declarations for the plugin API. A plugin is a shared object loaded at runtime
// which provides potentially optimized device-specific versions of routines and database entries.
// A plugin must contain a public (exported) symbol `clblast::Base *clblast_plugin`, which must
// point to an object of type `clblast::plugin::Interface` (or descendant).
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_PLUGIN_H_
#define CLBLAST_CLBLAST_PLUGIN_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "clblast.h"

namespace clblast {

namespace plugin {
// =================================================================================================

// This is a base class for all objects returned from the plugin non-type-safely, which must be
// returned as pointers to this base class and then down-casted with dynamic_cast<>().
// This way, we can ensure ABI compatibility in a type-safe way.
// WARNING: if this definition is ever altered, the exported symbol name must be changed.
class Base {
public:
  virtual ~Base();
};

// =================================================================================================

// WARNING: if any definition inside this namespace is altered, the version must be incremented.
inline namespace version_1 {

// =================================================================================================

class Routine;

class Interface : public Base {
 public:
  virtual ~Interface();

  virtual const Routine *GetRoutine(cl_device_id device, const std::string &routine, Precision precision) const = 0;
};

// =================================================================================================

// The non-routine-specific part of a plugin entry.
// This base must be virtual in all routine-specific classes because we have routines such as Xhemm
// (which inherit from Xgemm).
class Routine : public Base {
 public:
  virtual ~Routine();
};

// =================================================================================================
} // inline namespace version_X

} // namespace plugin

} // namespace clblast

// CLBLAST_CLBLAST_PLUGIN_H_
#endif
