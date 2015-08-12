
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Minh Quan Ho 
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the C interface to the CLBlast BLAS routines. It also contains C-translated 
// definitions of the returned status codes and the layout and transpose types. 
// This is the only header users of CLBlast should include and use for C code.
// (instead of clblast.h which are used for C++ code)
//
// C wrapper files : 
//     * include/clblast_c.h (This file)
//     * src/clblast_c.cc
//
// One who implement a backend BLAS C++ function, should/must complete this file and 
// the clblast_c.cc, following existing declaration patterns/macros, in order to expose his new 
// function to the C interface. 
// =================================================================================================

#ifndef CLBLAST_C_H_
#define CLBLAST_C_H_

// Include CLBlast header
#include <clblast.h>

#ifdef __cplusplus
extern "C" {
#endif

// =================================================================================================
// =================================================================================================
// PART 1 : 
// NORMALLY, YOU WILL NOT NEED TO EDIT THIS PART, OR YOU KNOWN WHAT YOU ARE DOING. 
// =================================================================================================
// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes. 
// Translated from C++ backend header. 
typedef enum StatusCode_e {

  // Status codes in common with the OpenCL standard
  kSuccess                   =   0, // CL_SUCCESS
  kTempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
  kBuildProgramFailure       = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
  kInvalidBinary             = -42, // CL_INVALID_BINARY
  kInvalidKernel             = -48, // CL_INVALID_KERNEL
  kInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  kInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  kInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
  kInvalidTempBufferSize     = -61, // CL_INVALID_BUFFER_SIZE

  // Status codes in common with the clBLAS library
  kNotImplemented            = -1024, // Routine or functionality not implemented yet
  kInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
  kInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
  kInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
  kInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
  kInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
  kInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
  kInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
  kInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
  kInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
  kInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
  kInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
  kInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
  kInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
  kInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
  kInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
  kInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  kKernelLaunchError         = -2048, // Problem occurred when enqueuing the kernel
  kKernelRunError            = -2047, // Problem occurred while running the kernel
  kInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
  kNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
  kNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
} StatusCode;

// Matrix layout and transpose types
typedef enum Layout_e { kRowMajor, kColMajor } Layout;
typedef enum Transpose_e { kNo, kYes, kConjugate } Transpose;
typedef enum Side_e { kLeft, kRight } Side;
typedef enum Triangle_e { kUpper, kLower } Triangle;
typedef enum Diagonal_e { kUnit, kNonUnit } Diagonal;

// Precision scoped enum (values in bits)
typedef enum Precision_e { kHalf = 16, kSingle = 32, kDouble = 64,
                       kComplexSingle = 3232, kComplexDouble = 6464 } Precision;

// =================================================================================================

// Define vector-type extension
typedef float  float2  __attribute__ ((vector_size (2*sizeof(float))));
typedef float  float4  __attribute__ ((vector_size (4*sizeof(float))));
typedef float  float8  __attribute__ ((vector_size (8*sizeof(float))));

typedef double double2 __attribute__ ((vector_size (2*sizeof(double))));
typedef double double4 __attribute__ ((vector_size (4*sizeof(double))));
typedef double double8 __attribute__ ((vector_size (8*sizeof(double))));

// =================================================================================================

// Inline function converting C++ status code to C status code 
static inline StatusCode convert_status(clblast::StatusCode status) {
  switch (status){
  case (clblast::StatusCode::kSuccess) : return kSuccess; break; 
  case (clblast::StatusCode::kTempBufferAllocFailure) : return kTempBufferAllocFailure; break; 
  case (clblast::StatusCode::kBuildProgramFailure) : return kBuildProgramFailure; break; 
  case (clblast::StatusCode::kInvalidBinary) : return kInvalidBinary; break; 
  case (clblast::StatusCode::kInvalidKernel) : return kInvalidKernel; break; 
  case (clblast::StatusCode::kInvalidLocalNumDimensions) : return kInvalidLocalNumDimensions; break; 
  case (clblast::StatusCode::kInvalidLocalThreadsTotal) : return kInvalidLocalThreadsTotal; break; 
  case (clblast::StatusCode::kInvalidLocalThreadsDim) : return kInvalidLocalThreadsDim; break; 
  case (clblast::StatusCode::kInvalidTempBufferSize) : return kInvalidTempBufferSize; break; 
  case (clblast::StatusCode::kNotImplemented) : return kNotImplemented; break; 
  case (clblast::StatusCode::kInvalidMatrixA) : return kInvalidMatrixA; break; 
  case (clblast::StatusCode::kInvalidMatrixB) : return kInvalidMatrixB; break; 
  case (clblast::StatusCode::kInvalidMatrixC) : return kInvalidMatrixC; break; 
  case (clblast::StatusCode::kInvalidVectorX) : return kInvalidVectorX; break; 
  case (clblast::StatusCode::kInvalidVectorY) : return kInvalidVectorY; break; 
  case (clblast::StatusCode::kInvalidDimension) : return kInvalidDimension; break; 
  case (clblast::StatusCode::kInvalidLeadDimA) : return kInvalidLeadDimA; break; 
  case (clblast::StatusCode::kInvalidLeadDimB) : return kInvalidLeadDimB; break; 
  case (clblast::StatusCode::kInvalidLeadDimC) : return kInvalidLeadDimC; break; 
  case (clblast::StatusCode::kInvalidIncrementX) : return kInvalidIncrementX; break; 
  case (clblast::StatusCode::kInvalidIncrementY) : return kInvalidIncrementY; break; 
  case (clblast::StatusCode::kInsufficientMemoryA) : return kInsufficientMemoryA; break; 
  case (clblast::StatusCode::kInsufficientMemoryB) : return kInsufficientMemoryB; break; 
  case (clblast::StatusCode::kInsufficientMemoryC) : return kInsufficientMemoryC; break; 
  case (clblast::StatusCode::kInsufficientMemoryX) : return kInsufficientMemoryX; break; 
  case (clblast::StatusCode::kInsufficientMemoryY) : return kInsufficientMemoryY; break; 
  case (clblast::StatusCode::kKernelLaunchError) : return kKernelLaunchError; break; 
  case (clblast::StatusCode::kKernelRunError) : return kKernelRunError; break; 
  case (clblast::StatusCode::kInvalidLocalMemUsage) : return kInvalidLocalMemUsage; break; 
  case (clblast::StatusCode::kNoHalfPrecision) : return kNoHalfPrecision; break; 
  case (clblast::StatusCode::kNoDoublePrecision) : return kNoDoublePrecision; break; 
  default : return kNotImplemented; break;
  }
}

// =================================================================================================
// Utility macros for clblast type binding from C to C++

#define CONVERT_LAYOUT(layout) \
  (layout == kRowMajor ? clblast::Layout::kRowMajor : clblast::Layout::kColMajor)

#define CONVERT_TRANS(trans) \
  (trans == kNo ? clblast::Transpose::kNo : \
      (trans == kYes ? clblast::Transpose::kYes : clblast::Transpose::kConjugate))

#define CONVERT_TRIANGLE(triangle) \
  (triangle == kUpper ? clblast::Triangle::kUpper : clblast::Triangle::kLower)

#define CONVERT_DIAG(diag) \
  (diag == kUnit ? clblast::Diagonal::kUnit : clblast::Diagonal::kNonUnit)

#define CONVERT_SIDE(side) \
  (side == kLeft ? clblast::Side::kLeft : clblast::Side::kRight)

// =================================================================================================
// Utility macros for function declaration
//                             Example :  StatusCode  clblastSgemm     (...)    
#define DECLARE_FUNCTION(NAME, SIGNATURE) StatusCode clblast ## NAME SIGNATURE



// =================================================================================================
// =================================================================================================
// PART 2 : 
// YOU HAVE IMPLEMENTED A NEW BLAS FUNCTION, BRAVO ! NOW ADD MORE SOME LINES IN THIS PART TO 
//         DECLARE ITS C-INTERFACE. DO NOT FORGET THE 'clblast_c.cc' as well.
// =================================================================================================
// =================================================================================================

// =================================================================================================
// BLAS level-1 (vector-vector) routines

// AXPY
#define AXPY_SIGNATURE(T)                                                         \
  (const size_t n, const T alpha,                                                 \
  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,               \
  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,                     \
  cl_command_queue* queue, cl_event* event)

#define AXPY_RETURN(T)                                               \
  return convert_status(clblast::Axpy<T>(                            \
          n, alpha,                                                  \
          x_buffer, x_offset, x_inc,                                 \
          y_buffer, y_offset, y_inc,                                 \
          queue, event));

DECLARE_FUNCTION(Saxpy, AXPY_SIGNATURE(float));
DECLARE_FUNCTION(Daxpy, AXPY_SIGNATURE(double));
DECLARE_FUNCTION(Caxpy, AXPY_SIGNATURE(float2));
DECLARE_FUNCTION(Zaxpy, AXPY_SIGNATURE(double2));

// =================================================================================================
// BLAS level-2 (matrix-vector) routines

// GEMV
#define GEMV_SIGNATURE(T)                                                         \
  (const Layout layout, const Transpose a_transpose,                              \
  const size_t m, const size_t n, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const T beta, \
  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,                     \
  cl_command_queue* queue, cl_event* event)

#define GEMV_RETURN(T)                                               \
  return convert_status(clblast::Gemv<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRANS(a_transpose),                                \
          m, n, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          x_buffer, x_offset, x_inc, beta,                           \
          y_buffer, y_offset, y_inc,                                 \
          queue, event));

DECLARE_FUNCTION(Sgemv, GEMV_SIGNATURE(float));
DECLARE_FUNCTION(Dgemv, GEMV_SIGNATURE(double));
DECLARE_FUNCTION(Cgemv, GEMV_SIGNATURE(float2));
DECLARE_FUNCTION(Zgemv, GEMV_SIGNATURE(double2));

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines

// GEMM
#define GEMM_SIGNATURE(T)                                                         \
  (const Layout layout, const Transpose a_transpose, const Transpose b_transpose, \
  const size_t m, const size_t n, const size_t k,                                 \
  const T alpha,                                                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,                \
  const T beta,                                                                   \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event)   

#define GEMM_RETURN(T)                                               \
  return convert_status(clblast::Gemm<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRANS(a_transpose),                                \
          CONVERT_TRANS(b_transpose),                                \
          m, n, k, alpha,                                            \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Sgemm, GEMM_SIGNATURE(float));
DECLARE_FUNCTION(Dgemm, GEMM_SIGNATURE(double));
DECLARE_FUNCTION(Cgemm, GEMM_SIGNATURE(float2));
DECLARE_FUNCTION(Zgemm, GEMM_SIGNATURE(double2));

// =================================================================================================

// SYMM
#define SYMM_SIGNATURE(T)                                            \
  (const Layout layout, const Side side, const Triangle triangle,    \
  const size_t m, const size_t n,                                    \
  const T alpha,                                                     \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,   \
  const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,   \
  const T beta,                                                      \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,         \
  cl_command_queue* queue, cl_event* event)

#define SYMM_RETURN(T)                                               \
  return convert_status(clblast::Symm<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_SIDE(side),                                        \
          CONVERT_TRIANGLE(triangle),                                \
          m, n, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Ssymm, SYMM_SIGNATURE(float));
DECLARE_FUNCTION(Dsymm, SYMM_SIGNATURE(double));
DECLARE_FUNCTION(Csymm, SYMM_SIGNATURE(float2));
DECLARE_FUNCTION(Zsymm, SYMM_SIGNATURE(double2));

// =================================================================================================

// HEMM
#define HEMM_SIGNATURE(T)                                                         \
  (const Layout layout, const Side side, const Triangle triangle,                 \
  const size_t m, const size_t n, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,  \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define HEMM_RETURN(T)                                               \
  return convert_status(clblast::Hemm<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_SIDE(side),                                        \
          CONVERT_TRIANGLE(triangle),                                \
          m, n, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Chemm, HEMM_SIGNATURE(float2));
DECLARE_FUNCTION(Zhemm, HEMM_SIGNATURE(double2));

// =================================================================================================

// SYRK
#define SYRK_SIGNATURE(T)                                                         \
  (const Layout layout, const Triangle triangle, const Transpose a_transpose,     \
  const size_t n, const size_t k, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const T beta,  \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define SYRK_RETURN(T)                                               \
  return convert_status(clblast::Syrk<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(a_transpose),                                \
          n, k, alpha,                                               \
          a_buffer, a_offset, a_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Ssyrk, SYRK_SIGNATURE(float));
DECLARE_FUNCTION(Dsyrk, SYRK_SIGNATURE(double));
DECLARE_FUNCTION(Csyrk, SYRK_SIGNATURE(float2));
DECLARE_FUNCTION(Zsyrk, SYRK_SIGNATURE(double2));

// =================================================================================================

// HERK
#define HERK_SIGNATURE(T)                                                         \
  (const Layout layout, const Triangle triangle, const Transpose a_transpose,     \
  const size_t n, const size_t k, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const T beta,  \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define HERK_RETURN(T)                                               \
  return convert_status(clblast::Herk<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(a_transpose),                                \
          n, k, alpha,                                               \
          a_buffer, a_offset, a_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Cherk, HERK_SIGNATURE(float));
DECLARE_FUNCTION(Zherk, HERK_SIGNATURE(double));

// =================================================================================================

// SYR2K
#define SYR2K_SIGNATURE(T)                                                        \
  (const Layout layout, const Triangle triangle, const Transpose ab_transpose,    \
  const size_t n, const size_t k, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,  \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event) 

#define SYR2K_RETURN(T)                                              \
  return convert_status(clblast::Syr2k<T>(                           \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(ab_transpose),                               \
          n, k, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Ssyr2k, SYR2K_SIGNATURE(float));
DECLARE_FUNCTION(Dsyr2k, SYR2K_SIGNATURE(double));
DECLARE_FUNCTION(Csyr2k, SYR2K_SIGNATURE(float2));
DECLARE_FUNCTION(Zsyr2k, SYR2K_SIGNATURE(double2));

// =================================================================================================

// HER2K
#define HER2K_SIGNATURE(T, U)                                                     \
  (const Layout layout, const Triangle triangle, const Transpose ab_transpose,    \
  const size_t n, const size_t k, const T alpha,                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const U beta,  \
  cl_mem c_buffer, const size_t c_offset, const size_t c_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define HER2K_RETURN(T, U)                                           \
  return convert_status(clblast::Her2k<T, U>(                        \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(ab_transpose),                               \
          n, k, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, beta,                            \
          c_buffer, c_offset, c_ld, queue, event));

DECLARE_FUNCTION(Cher2k, HER2K_SIGNATURE(float2, float));
DECLARE_FUNCTION(Zher2k, HER2K_SIGNATURE(double2, double));

// =================================================================================================

// TRMM
#define TRMM_SIGNATURE(T)                                                         \
  (const Layout layout, const Side side, const Triangle triangle,                 \
  const Transpose a_transpose, const Diagonal diagonal,                           \
  const size_t m, const size_t n,                                                 \
  const T alpha,                                                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  cl_mem b_buffer, const size_t b_offset, const size_t b_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define TRMM_RETURN(T)                                               \
  return convert_status(clblast::Trmm<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_SIDE(side),                                        \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(a_transpose),                                \
          CONVERT_DIAG(diagonal),                                    \
          m, n, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, queue, event));

DECLARE_FUNCTION(Strmm, TRMM_SIGNATURE(float));
DECLARE_FUNCTION(Dtrmm, TRMM_SIGNATURE(double));
DECLARE_FUNCTION(Ctrmm, TRMM_SIGNATURE(float2));
DECLARE_FUNCTION(Ztrmm, TRMM_SIGNATURE(double2));

// =================================================================================================

// TRSM
/*
#define TRSM_SIGNATURE(T)                                                         \
  (const Layout layout, const Side side, const Triangle triangle,                 \
  const Transpose a_transpose, const Diagonal diagonal,                           \
  const size_t m, const size_t n,                                                 \
  const T alpha,                                                                  \
  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,                \
  cl_mem b_buffer, const size_t b_offset, const size_t b_ld,                      \
  cl_command_queue* queue, cl_event* event)

#define TRSM_RETURN(T)                                               \
  return convert_status(clblast::Trsm<T>(                            \
          CONVERT_LAYOUT(layout),                                    \
          CONVERT_SIDE(side),                                        \
          CONVERT_TRIANGLE(triangle),                                \
          CONVERT_TRANS(a_transpose),                                \
          CONVERT_DIAG(diagonal),                                    \
          m, n, alpha,                                               \
          a_buffer, a_offset, a_ld,                                  \
          b_buffer, b_offset, b_ld, queue, event));


DECLARE_FUNCTION(Strsm, TRSM_SIGNATURE(float));
DECLARE_FUNCTION(Dtrsm, TRSM_SIGNATURE(double));
DECLARE_FUNCTION(Ctrsm, TRSM_SIGNATURE(float2));
DECLARE_FUNCTION(Ztrsm, TRSM_SIGNATURE(double2));
*/
// =================================================================================================

// Add more here ...


// =================================================================================================


#ifdef __cplusplus
}
#endif

// =================================================================================================
// CLBLAST_C_H_
#endif 
