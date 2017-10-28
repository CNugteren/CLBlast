
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file contains exception classes corresponding to 'clpp11.hpp'. It is also part of the
// CLCudaAPI project. See 'clpp11.hpp' for more details.
//
// =================================================================================================

#ifndef CLBLAST_CXPP11_COMMON_H_
#define CLBLAST_CXPP11_COMMON_H_

#include <cstring>   // strchr
#include <string>    // std::string
#include <stdexcept> // std::runtime_error

namespace clblast {
// =================================================================================================

// Basic exception class: represents an error happened inside our code
// (as opposed to an error in C++ runtime)
template <typename Base>
class Error : public Base {
 public:
  // Perfect forwarding of the constructor since "using Base::Base" is not supported by VS 2013
  template <typename... Args>
  Error(Args&&... args):
      Base(std::forward<Args>(args)...) {
  }
};

// =================================================================================================

// Represents a generic device-specific runtime error (returned by an OpenCL or CUDA API function)
class DeviceError : public Error<std::runtime_error> {
 public:
   // Perfect forwarding of the constructor since "using Error<std::runtime_error>::Error" is not
   // supported by VS 2013
   template <typename... Args>
   DeviceError(Args&&... args):
       Error<std::runtime_error>(std::forward<Args>(args)...) {
   }

  static std::string TrimCallString(const char *where) {
    const char *paren = strchr(where, '(');
    if (paren) {
      return std::string(where, paren);
    } else {
      return std::string(where);
    }
  }
};

// =================================================================================================

// Represents a generic runtime error (aka environmental problem)
class RuntimeError : public Error<std::runtime_error> {
 public:
  explicit RuntimeError(const std::string &reason):
      Error("Run-time error: " + reason) {
  }
};

// =================================================================================================

// Represents a generic logic error (aka failed assertion)
class LogicError : public Error<std::logic_error> {
 public:
  explicit LogicError(const std::string &reason):
      Error("Internal logic error: " + reason) {
  }
};

// =================================================================================================

// Internal exception base class with a status field and a subclass-specific "details" field
// which can be used to recreate an exception
template <typename Base, typename Status>
class ErrorCode : public Base {
 public:
  ErrorCode(Status status, const std::string &details, const std::string &reason):
      Base(reason),
      status_(status),
      details_(details) {
  }

  Status status() const {
    return status_;
  }

  const std::string& details() const {
    return details_;
  }

 private:
  const Status status_;
  const std::string details_;
};

// =================================================================================================

} // namespace clblast

// CLBLAST_CXPP11_COMMON_H_
#endif
