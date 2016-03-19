
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides macro's to define the public API. This is needed when building a Windows DLL.
//
// =================================================================================================

#ifndef CLBLAST_PUBLIC_API_H_
#define CLBLAST_PUBLIC_API_H_

namespace clblast {
// =================================================================================================

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#ifdef _WIN32
  #define PUBLIC_API __declspec(dllexport)
#else
  #define PUBLIC_API
#endif

// =================================================================================================
} // namespace clblast

// CLBLAST_PUBLIC_API_H_
#endif