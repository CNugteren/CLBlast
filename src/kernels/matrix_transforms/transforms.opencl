
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for matrix-transform kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef PAD_DIMX
  #define PAD_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
  #define PAD_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
  #define PAD_WPTX 1      // Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
  #define PAD_WPTY 1      // Work per thread in the second dimension (y)
#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
