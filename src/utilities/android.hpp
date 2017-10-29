
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides macro's and definitions to make compilation work for Android. Note that this
// header should only be included when compiling for Android, e.g. when __ANDROID__ is defined.
//
// =================================================================================================

#ifndef CLBLAST_ANDROID_HPP_
#define CLBLAST_ANDROID_HPP_

// =================================================================================================

#include <cstdlib>
#include <string>
#include <sstream>

namespace std {

// No support for these standard library functions when compiling with the GNU C++ STL
template<typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}
inline double stod(const std::string& value) {
  return std::atof(value.c_str());
}
inline int stoi( const std::string& str, std::size_t* pos = 0, int base = 10) {
  char * p_end;
  const auto result = std::strtol(str.c_str(), &p_end, base);
  return result;
}

}

// =================================================================================================

// CLBLAST_ANDROID_HPP_
#endif
