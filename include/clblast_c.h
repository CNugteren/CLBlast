
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

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#ifdef _WIN32
  #define PUBLIC_API __declspec(dllexport)
#else
  #define PUBLIC_API
#endif

// The C interface
#ifdef __cplusplus
extern "C" {
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

// Generate plane rotation: SROTG/DROTG
StatusCode PUBLIC_API CLBlastSrotg(cl_mem SA_buffer, const size_t SA_offset,
                                   cl_mem SB_buffer, const size_t SB_offset,
                                   cl_mem C_buffer, const size_t C_offset,
                                   cl_mem S_buffer, const size_t S_offset,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDrotg(cl_mem SA_buffer, const size_t SA_offset,
                                   cl_mem SB_buffer, const size_t SB_offset,
                                   cl_mem C_buffer, const size_t C_offset,
                                   cl_mem S_buffer, const size_t S_offset,
                                   cl_command_queue* queue, cl_event* event);

// Apply plane rotation: SROT/DROT
StatusCode PUBLIC_API CLBlastSrot(const size_t n,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  const float C,
                                  const float S,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDrot(const size_t n,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  const double C,
                                  const double S,
                                  cl_command_queue* queue, cl_event* event);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP
StatusCode PUBLIC_API CLBlastSswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL
StatusCode PUBLIC_API CLBlastSscal(const size_t n,
                                   const float alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDscal(const size_t n,
                                   const double alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCscal(const size_t n,
                                   const cl_float2 alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZscal(const size_t n,
                                   const cl_double2 alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY
StatusCode PUBLIC_API CLBlastScopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDcopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCcopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZcopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY
StatusCode PUBLIC_API CLBlastSaxpy(const size_t n,
                                   const float alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDaxpy(const size_t n,
                                   const double alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCaxpy(const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZaxpy(const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Dot product of two vectors: SDOT/DDOT
StatusCode PUBLIC_API CLBlastSdot(const size_t n,
                                  cl_mem dot_buffer, const size_t dot_offset,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDdot(const size_t n,
                                  cl_mem dot_buffer, const size_t dot_offset,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors: CDOTU/ZDOTU
StatusCode PUBLIC_API CLBlastCdotu(const size_t n,
                                   cl_mem dot_buffer, const size_t dot_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZdotu(const size_t n,
                                   cl_mem dot_buffer, const size_t dot_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
StatusCode PUBLIC_API CLBlastCdotc(const size_t n,
                                   cl_mem dot_buffer, const size_t dot_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZdotc(const size_t n,
                                   cl_mem dot_buffer, const size_t dot_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV
StatusCode PUBLIC_API CLBlastSgemv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDgemv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const double beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCgemv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_float2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZgemv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_double2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV
StatusCode PUBLIC_API CLBlastSgbmv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n, const size_t kl, const size_t ku,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDgbmv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n, const size_t kl, const size_t ku,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const double beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCgbmv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n, const size_t kl, const size_t ku,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_float2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZgbmv(const Layout layout, const Transpose a_transpose,
                                   const size_t m, const size_t n, const size_t kl, const size_t ku,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_double2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
StatusCode PUBLIC_API CLBlastChemv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_float2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhemv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_double2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
StatusCode PUBLIC_API CLBlastChbmv(const Layout layout, const Triangle triangle,
                                   const size_t n, const size_t k,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_float2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhbmv(const Layout layout, const Triangle triangle,
                                   const size_t n, const size_t k,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_double2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
StatusCode PUBLIC_API CLBlastChpmv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_float2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhpmv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_double2 beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV
StatusCode PUBLIC_API CLBlastSsymv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsymv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const double beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV
StatusCode PUBLIC_API CLBlastSsbmv(const Layout layout, const Triangle triangle,
                                   const size_t n, const size_t k,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsbmv(const Layout layout, const Triangle triangle,
                                   const size_t n, const size_t k,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const double beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV
StatusCode PUBLIC_API CLBlastSspmv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const float alpha,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDspmv(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const double alpha,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const double beta,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV
StatusCode PUBLIC_API CLBlastStrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV
StatusCode PUBLIC_API CLBlastStbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV
StatusCode PUBLIC_API CLBlastStpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
StatusCode PUBLIC_API CLBlastStrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
StatusCode PUBLIC_API CLBlastStbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n, const size_t k,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
StatusCode PUBLIC_API CLBlastStpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t n,
                                   const cl_mem ap_buffer, const size_t ap_offset,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event);

// General rank-1 matrix update: SGER/DGER
StatusCode PUBLIC_API CLBlastSger(const Layout layout,
                                  const size_t m, const size_t n,
                                  const float alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDger(const Layout layout,
                                  const size_t m, const size_t n,
                                  const double alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);

// General rank-1 complex matrix update: CGERU/ZGERU
StatusCode PUBLIC_API CLBlastCgeru(const Layout layout,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZgeru(const Layout layout,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
StatusCode PUBLIC_API CLBlastCgerc(const Layout layout,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZgerc(const Layout layout,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian rank-1 matrix update: CHER/ZHER
StatusCode PUBLIC_API CLBlastCher(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const float alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZher(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const double alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
StatusCode PUBLIC_API CLBlastChpr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const float alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem ap_buffer, const size_t ap_offset,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhpr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const double alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem ap_buffer, const size_t ap_offset,
                                  cl_command_queue* queue, cl_event* event);

// Hermitian rank-2 matrix update: CHER2/ZHER2
StatusCode PUBLIC_API CLBlastCher2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZher2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
StatusCode PUBLIC_API CLBlastChpr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem ap_buffer, const size_t ap_offset,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhpr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem ap_buffer, const size_t ap_offset,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric rank-1 matrix update: SSYR/DSYR
StatusCode PUBLIC_API CLBlastSsyr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const float alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsyr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const double alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-1 matrix update: SSPR/DSPR
StatusCode PUBLIC_API CLBlastSspr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const float alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem ap_buffer, const size_t ap_offset,
                                  cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDspr(const Layout layout, const Triangle triangle,
                                  const size_t n,
                                  const double alpha,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem ap_buffer, const size_t ap_offset,
                                  cl_command_queue* queue, cl_event* event);

// Symmetric rank-2 matrix update: SSYR2/DSYR2
StatusCode PUBLIC_API CLBlastSsyr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const float alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsyr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const double alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2
StatusCode PUBLIC_API CLBlastSspr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const float alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem ap_buffer, const size_t ap_offset,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDspr2(const Layout layout, const Triangle triangle,
                                   const size_t n,
                                   const double alpha,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_mem ap_buffer, const size_t ap_offset,
                                   cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM
StatusCode PUBLIC_API CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                   const size_t m, const size_t n, const size_t k,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const float beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                   const size_t m, const size_t n, const size_t k,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const double beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                   const size_t m, const size_t n, const size_t k,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_float2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                                   const size_t m, const size_t n, const size_t k,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_double2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM
StatusCode PUBLIC_API CLBlastSsymm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const float beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const double beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_float2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_double2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
StatusCode PUBLIC_API CLBlastChemm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_float2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   const cl_double2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK
StatusCode PUBLIC_API CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const float beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const double beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_float2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const cl_double2 beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
StatusCode PUBLIC_API CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const float beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                                   const size_t n, const size_t k,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   const double beta,
                                   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                   cl_command_queue* queue, cl_event* event);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K
StatusCode PUBLIC_API CLBlastSsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const float alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const float beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const double alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const double beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const cl_float2 alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const cl_float2 beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const cl_double2 alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const cl_double2 beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
StatusCode PUBLIC_API CLBlastCher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const cl_float2 alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const float beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                                    const size_t n, const size_t k,
                                    const cl_double2 alpha,
                                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                    const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                    const double beta,
                                    cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                    cl_command_queue* queue, cl_event* event);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM
StatusCode PUBLIC_API CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
StatusCode PUBLIC_API CLBlastStrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastDtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastCtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);
StatusCode PUBLIC_API CLBlastZtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event);

// =================================================================================================

#ifdef __cplusplus
} // extern "C"
#endif

// CLBLAST_CLBLAST_C_H_
#endif
