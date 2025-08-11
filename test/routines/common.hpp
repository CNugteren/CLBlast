
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common includes for the clients and tests
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_COMMON_H_
#define CLBLAST_TEST_ROUTINES_COMMON_H_

#include <vector>
#include <string>

#include "utilities/utilities.hpp"
#include "test/test_utilities.hpp"

#ifdef CLBLAST_REF_CLBLAS
  #include "test/wrapper_clblas.hpp"
#endif
#ifdef CLBLAST_REF_CBLAS
  #include "test/wrapper_cblas.hpp"
#endif
#include "test/wrapper_cuda.hpp"
#ifdef CLBLAST_REF_CUBLAS
  #include "test/wrapper_cublas.hpp"
#endif

// =================================================================================================

// CLBLAST_TEST_ROUTINES_COMMON_H_
#endif
