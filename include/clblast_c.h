
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the plain C interface to the CLBlast BLAS routines, the counter-part of the
// normal 'clblast.h' C++ header.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_C_H_
#define CLBLAST_CLBLAST_C_H_

// Includes the normal OpenCL C header
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes. 
typedef enum StatusCode_ {

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
  kInvalidVectorDot          = -2043, // Vector dot is not a valid OpenCL buffer
  kInsufficientMemoryDot     = -2042, // Vector dot's OpenCL buffer is too small
} StatusCode;

// Matrix layout and transpose types
typedef enum Layout_ { kRowMajor, kColMajor } Layout;
typedef enum Transpose_ { kNo, kYes, kConjugate } Transpose;
typedef enum Side_ { kLeft, kRight } Side;
typedef enum Triangle_ { kUpper, kLower } Triangle;
typedef enum Diagonal_ { kUnit, kNonUnit } Diagonal;

// Precision scoped enum (values in bits)
typedef enum Precision_ { kHalf = 16, kSingle = 32, kDouble = 64,
                          kComplexSingle = 3232, kComplexDouble = 6464 } Precision;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP
StatusCode CLBlastSswap(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDswap(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCswap(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZswap(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL
StatusCode CLBlastSscal(const size_t n,
                        const float alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDscal(const size_t n,
                        const double alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCscal(const size_t n,
                        const cl_float2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZscal(const size_t n,
                        const cl_double2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY
StatusCode CLBlastScopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY
StatusCode CLBlastSaxpy(const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDaxpy(const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCaxpy(const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZaxpy(const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Dot product of two vectors: SDOT/DDOT
StatusCode CLBlastSdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors: CDOTU/ZDOTU
StatusCode CLBlastCdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
StatusCode CLBlastCdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// Generalized matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV
StatusCode CLBlastSgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
StatusCode CLBlastChemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZhemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV
StatusCode CLBlastSsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// Generalized matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM
StatusCode CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM
StatusCode CLBlastSsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
StatusCode CLBlastChemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK
StatusCode CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
StatusCode CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K
StatusCode CLBlastSsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
StatusCode CLBlastCher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM
StatusCode CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event);
StatusCode CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event);

// =================================================================================================

// CLBLAST_CLBLAST_C_H_
#endif
