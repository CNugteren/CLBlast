
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides macro's and definitions to make compilation work for Android. Note that this
// header should only be included when compiling for Android, e.g. when __ANDROID__ is defined.
//
// =================================================================================================

#ifndef CLBLAST_ANDROID_HPP_
#define CLBLAST_ANDROID_HPP_
#ifndef __clang__ // not to include custom impl to avoid ambiguous definition
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
#endif // clang header guard
// CLBLAST_ANDROID_HPP_
#endif
