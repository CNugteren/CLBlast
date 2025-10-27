
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common includes for the clients and tests
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_COMMON_H_
#define CLBLAST_TEST_ROUTINES_COMMON_H_

#include "../test_utilities.hpp"

#ifdef CLBLAST_REF_CLBLAS
#include "test/wrapper_clblas.hpp"  // IWYU pragma: export
#endif
#ifdef CLBLAST_REF_CBLAS
#include "test/wrapper_cblas.hpp"  // IWYU pragma: export
#endif
#include "test/wrapper_cuda.hpp"  // IWYU pragma: export
#ifdef CLBLAST_REF_CUBLAS
#include "test/wrapper_cublas.hpp"  // IWYU pragma: export
#endif

// =================================================================================================

// CLBLAST_TEST_ROUTINES_COMMON_H_
#endif
