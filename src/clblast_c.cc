
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Minh Quan Ho
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// C wrapper files : 
//     * include/clblast_c.h 
//     * src/clblast_c.cc    (This file)
//
// This file implements all the C interface functions of CLBlast, using macros defined in 
// clblast_c.h.
//
// =================================================================================================

// Include C wrapper header
#include "clblast_c.h"

extern "C" {

// =================================================================================================
// BLAS level-1 (vector-vector) routines

// AXPY
DECLARE_FUNCTION(Saxpy, AXPY_SIGNATURE(float))    {AXPY_RETURN(float);}
DECLARE_FUNCTION(Daxpy, AXPY_SIGNATURE(double))   {AXPY_RETURN(double);}
DECLARE_FUNCTION(Caxpy, AXPY_SIGNATURE(float2))   {AXPY_RETURN(float2);}
DECLARE_FUNCTION(Zaxpy, AXPY_SIGNATURE(double2))  {AXPY_RETURN(double2);}

#undef AXPY_SIGNATURE
#undef AXPY_RETURN

// =================================================================================================
// BLAS level-2 (matrix-vector) routines

// GEMV
DECLARE_FUNCTION(Sgemv, GEMV_SIGNATURE(float))    {GEMV_RETURN(float);}
DECLARE_FUNCTION(Dgemv, GEMV_SIGNATURE(double))   {GEMV_RETURN(double);}
DECLARE_FUNCTION(Cgemv, GEMV_SIGNATURE(float2))   {GEMV_RETURN(float2);}
DECLARE_FUNCTION(Zgemv, GEMV_SIGNATURE(double2))  {GEMV_RETURN(double2);}

#undef GEMV_SIGNATURE
#undef GEMV_RETURN

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines

// GEMM
DECLARE_FUNCTION(Sgemm, GEMM_SIGNATURE(float))    {GEMM_RETURN(float);}
DECLARE_FUNCTION(Dgemm, GEMM_SIGNATURE(double))   {GEMM_RETURN(double);}
DECLARE_FUNCTION(Cgemm, GEMM_SIGNATURE(float2))   {GEMM_RETURN(float2);}
DECLARE_FUNCTION(Zgemm, GEMM_SIGNATURE(double2))  {GEMM_RETURN(double2);}

#undef GEMM_SIGNATURE
#undef GEMM_RETURN

// =================================================================================================

// SYMM
DECLARE_FUNCTION(Ssymm, SYMM_SIGNATURE(float))    {SYMM_RETURN(float);}
DECLARE_FUNCTION(Dsymm, SYMM_SIGNATURE(double))   {SYMM_RETURN(double);}
DECLARE_FUNCTION(Csymm, SYMM_SIGNATURE(float2))   {SYMM_RETURN(float2);}
DECLARE_FUNCTION(Zsymm, SYMM_SIGNATURE(double2))  {SYMM_RETURN(double2);}

#undef SYMM_SIGNATURE
#undef SYMM_RETURN

// =================================================================================================

// HEMM
DECLARE_FUNCTION(Chemm, HEMM_SIGNATURE(float2))   {HEMM_RETURN(float2);}
DECLARE_FUNCTION(Zhemm, HEMM_SIGNATURE(double2))  {HEMM_RETURN(double2);}

#undef HEMM_SIGNATURE
#undef HEMM_RETURN

// =================================================================================================

// SYRK
DECLARE_FUNCTION(Ssyrk, SYRK_SIGNATURE(float))    {SYRK_RETURN(float);}
DECLARE_FUNCTION(Dsyrk, SYRK_SIGNATURE(double))   {SYRK_RETURN(double);}
DECLARE_FUNCTION(Csyrk, SYRK_SIGNATURE(float2))   {SYRK_RETURN(float2);}
DECLARE_FUNCTION(Zsyrk, SYRK_SIGNATURE(double2))  {SYRK_RETURN(double2);}

#undef SYRK_SIGNATURE
#undef SYRK_RETURN

// =================================================================================================

// HERK
DECLARE_FUNCTION(Cherk, HERK_SIGNATURE(float))   {HERK_RETURN(float);}
DECLARE_FUNCTION(Zherk, HERK_SIGNATURE(double))  {HERK_RETURN(double);}

#undef HERK_SIGNATURE
#undef HERK_RETURN

// =================================================================================================

// SYR2K
DECLARE_FUNCTION(Ssyr2k, SYR2K_SIGNATURE(float))    {SYR2K_RETURN(float);}
DECLARE_FUNCTION(Dsyr2k, SYR2K_SIGNATURE(double))   {SYR2K_RETURN(double);}
DECLARE_FUNCTION(Csyr2k, SYR2K_SIGNATURE(float2))   {SYR2K_RETURN(float2);}
DECLARE_FUNCTION(Zsyr2k, SYR2K_SIGNATURE(double2))  {SYR2K_RETURN(double2);}

#undef SYR2K_SIGNATURE
#undef SYR2K_RETURN

// =================================================================================================

// HER2K
DECLARE_FUNCTION(Cher2k, HER2K_SIGNATURE(float2, float))   {HER2K_RETURN(float2, float);}
DECLARE_FUNCTION(Zher2k, HER2K_SIGNATURE(double2, double)) {HER2K_RETURN(double2, double);}

#undef HER2K_SIGNATURE
#undef HER2K_RETURN

// =================================================================================================

// TRMM
DECLARE_FUNCTION(Strmm, TRMM_SIGNATURE(float))    {TRMM_RETURN(float);}
DECLARE_FUNCTION(Dtrmm, TRMM_SIGNATURE(double))   {TRMM_RETURN(double);}
DECLARE_FUNCTION(Ctrmm, TRMM_SIGNATURE(float2))   {TRMM_RETURN(float2);}
DECLARE_FUNCTION(Ztrmm, TRMM_SIGNATURE(double2))  {TRMM_RETURN(double2);}

#undef TRMM_SIGNATURE
#undef TRMM_RETURN

// =================================================================================================

// TRSM
/*
DECLARE_FUNCTION(Strsm, TRSM_SIGNATURE(float))    {TRSM_RETURN(float);}
DECLARE_FUNCTION(Dtrsm, TRSM_SIGNATURE(double))   {TRSM_RETURN(double);}
DECLARE_FUNCTION(Ctrsm, TRSM_SIGNATURE(float2))   {TRSM_RETURN(float2);}
DECLARE_FUNCTION(Ztrsm, TRSM_SIGNATURE(double2))  {TRSM_RETURN(double2);}

#undef TRSM_SIGNATURE
#undef TRSM_RETURN
*/
// =================================================================================================

// Add more here ...


// =================================================================================================

// End wrapper, now undefine used macros
#undef CONVERT_LAYOUT
#undef CONVERT_TRANS
#undef CONVERT_TRIANGLE
#undef CONVERT_DIAG
#undef CONVERT_SIDE
#undef DECLARE_FUNCTION

// =================================================================================================
} // extern "C"
