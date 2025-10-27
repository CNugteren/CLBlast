
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the interface to the CLBlast BLAS routines. It also contains the definitions
// of the returned status codes and the layout and transpose types. This is the only header users
// of CLBlast should include and use.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_H_
#define CLBLAST_CLBLAST_H_

#include <cstdlib>        // For size_t
#include <string>         // For OverrideParameters function
#include <unordered_map>  // For OverrideParameters function

// Includes the normal OpenCL C header
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS  // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS  // to disable deprecation warnings
#elif defined(CL_TARGET_OPENCL_VERSION) && CL_TARGET_OPENCL_VERSION < 110
#pragma warning "OpenCL Version must be at least 1.1 (110) to use CLBlast, redefining"
#undef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS  // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS  // to disable deprecation warnings
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(CLBLAST_DLL)
#if defined(COMPILING_DLL)
#define PUBLIC_API __declspec(dllexport)
#else
#define PUBLIC_API __declspec(dllimport)
#endif
#else
#define PUBLIC_API
#endif

// Version numbering (v1.6.3)
#define CLBLAST_VERSION_MAJOR 1
#define CLBLAST_VERSION_MINOR 6
#define CLBLAST_VERSION_PATCH 3

namespace clblast {
// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes.
enum class StatusCode {

  // Status codes in common with the OpenCL standard
  kSuccess = 0,                      // CL_SUCCESS
  kOpenCLCompilerNotAvailable = -3,  // CL_COMPILER_NOT_AVAILABLE
  kTempBufferAllocFailure = -4,      // CL_MEM_OBJECT_ALLOCATION_FAILURE
  kOpenCLOutOfResources = -5,        // CL_OUT_OF_RESOURCES
  kOpenCLOutOfHostMemory = -6,       // CL_OUT_OF_HOST_MEMORY
  kOpenCLBuildProgramFailure = -11,  // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
  kInvalidValue = -30,               // CL_INVALID_VALUE
  kInvalidCommandQueue = -36,        // CL_INVALID_COMMAND_QUEUE
  kInvalidMemObject = -38,           // CL_INVALID_MEM_OBJECT
  kInvalidBinary = -42,              // CL_INVALID_BINARY
  kInvalidBuildOptions = -43,        // CL_INVALID_BUILD_OPTIONS
  kInvalidProgram = -44,             // CL_INVALID_PROGRAM
  kInvalidProgramExecutable = -45,   // CL_INVALID_PROGRAM_EXECUTABLE
  kInvalidKernelName = -46,          // CL_INVALID_KERNEL_NAME
  kInvalidKernelDefinition = -47,    // CL_INVALID_KERNEL_DEFINITION
  kInvalidKernel = -48,              // CL_INVALID_KERNEL
  kInvalidArgIndex = -49,            // CL_INVALID_ARG_INDEX
  kInvalidArgValue = -50,            // CL_INVALID_ARG_VALUE
  kInvalidArgSize = -51,             // CL_INVALID_ARG_SIZE
  kInvalidKernelArgs = -52,          // CL_INVALID_KERNEL_ARGS
  kInvalidLocalNumDimensions = -53,  // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  kInvalidLocalThreadsTotal = -54,   // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  kInvalidLocalThreadsDim = -55,     // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
  kInvalidGlobalOffset = -56,        // CL_INVALID_GLOBAL_OFFSET
  kInvalidEventWaitList = -57,       // CL_INVALID_EVENT_WAIT_LIST
  kInvalidEvent = -58,               // CL_INVALID_EVENT
  kInvalidOperation = -59,           // CL_INVALID_OPERATION
  kInvalidBufferSize = -61,          // CL_INVALID_BUFFER_SIZE
  kInvalidGlobalWorkSize = -63,      // CL_INVALID_GLOBAL_WORK_SIZE

  // Status codes in common with the clBLAS library
  kNotImplemented = -1024,       // Routine or functionality not implemented yet
  kInvalidMatrixA = -1022,       // Matrix A is not a valid OpenCL buffer
  kInvalidMatrixB = -1021,       // Matrix B is not a valid OpenCL buffer
  kInvalidMatrixC = -1020,       // Matrix C is not a valid OpenCL buffer
  kInvalidVectorX = -1019,       // Vector X is not a valid OpenCL buffer
  kInvalidVectorY = -1018,       // Vector Y is not a valid OpenCL buffer
  kInvalidDimension = -1017,     // Dimensions M, N, and K have to be larger than zero
  kInvalidLeadDimA = -1016,      // LD of A is smaller than the matrix's first dimension
  kInvalidLeadDimB = -1015,      // LD of B is smaller than the matrix's first dimension
  kInvalidLeadDimC = -1014,      // LD of C is smaller than the matrix's first dimension
  kInvalidIncrementX = -1013,    // Increment of vector X cannot be zero
  kInvalidIncrementY = -1012,    // Increment of vector Y cannot be zero
  kInsufficientMemoryA = -1011,  // Matrix A's OpenCL buffer is too small
  kInsufficientMemoryB = -1010,  // Matrix B's OpenCL buffer is too small
  kInsufficientMemoryC = -1009,  // Matrix C's OpenCL buffer is too small
  kInsufficientMemoryX = -1008,  // Vector X's OpenCL buffer is too small
  kInsufficientMemoryY = -1007,  // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  kInsufficientMemoryTemp = -2050,    // Temporary buffer provided to GEMM routine is too small
  kInvalidBatchCount = -2049,         // The batch count needs to be positive
  kInvalidOverrideKernel = -2048,     // Trying to override parameters for an invalid kernel
  kMissingOverrideParameter = -2047,  // Missing override parameter(s) for the target kernel
  kInvalidLocalMemUsage = -2046,      // Not enough local memory available on this device
  kNoHalfPrecision = -2045,           // Half precision (16-bits) not supported by the device
  kNoDoublePrecision = -2044,         // Double precision (64-bits) not supported by the device
  kInvalidVectorScalar = -2043,       // The unit-sized vector is not a valid OpenCL buffer
  kInsufficientMemoryScalar = -2042,  // The unit-sized vector's OpenCL buffer is too small
  kDatabaseError = -2041,             // Entry for the device was not found in the database
  kUnknownError = -2040,              // A catch-all error code representing an unspecified error
  kUnexpectedError = -2039,           // A catch-all error code representing an unexpected exception
  kInvalidVectorZ = -2038,            // Vector Z is not a valid OpenCL buffer
  kInvalidIncrementZ = -2037,         // Increment of vector Z cannot be zero
  kInsufficientMemoryZ = -2036,       // Vector Z's OpenCL buffer is too small
};

// Matrix layout and transpose types
enum class Layout { kRowMajor = 101, kColMajor = 102 };
enum class Transpose { kNo = 111, kYes = 112, kConjugate = 113 };
enum class Triangle { kUpper = 121, kLower = 122 };
enum class Diagonal { kNonUnit = 131, kUnit = 132 };
enum class Side { kLeft = 141, kRight = 142 };
enum class KernelMode { kCrossCorrelation = 151, kConvolution = 152 };

// Precision scoped enum (values in bits)
enum class Precision {
  kHalf = 16,
  kSingle = 32,
  kDouble = 64,
  kComplexSingle = 3232,
  kComplexDouble = 6464,
  kAny = -1
};

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
StatusCode Rotg(cl_mem sa_buffer, size_t sa_offset, cl_mem sb_buffer, size_t sb_offset, cl_mem sc_buffer,
                size_t sc_offset, cl_mem ss_buffer, size_t ss_offset, cl_command_queue* queue,
                cl_event* event = nullptr);

// Generate modified givens plane rotation: SROTMG/DROTMG
template <typename T>
StatusCode Rotmg(cl_mem sd1_buffer, size_t sd1_offset, cl_mem sd2_buffer, size_t sd2_offset, cl_mem sx1_buffer,
                 size_t sx1_offset, cl_mem sy1_buffer, size_t sy1_offset, cl_mem sparam_buffer, size_t sparam_offset,
                 cl_command_queue* queue, cl_event* event = nullptr);

// Apply givens plane rotation: SROT/DROT
template <typename T>
StatusCode Rot(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
               T cos, T sin, cl_command_queue* queue, cl_event* event = nullptr);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
StatusCode Rotm(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_mem sparam_buffer, size_t sparam_offset, cl_command_queue* queue,
                cl_event* event = nullptr);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
StatusCode Swap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
StatusCode Scal(size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                cl_event* event = nullptr);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
StatusCode Copy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
StatusCode Axpy(size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
StatusCode Dot(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
StatusCode Dotu(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
StatusCode Dotc(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
StatusCode Nrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
StatusCode Asum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
StatusCode Sum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_command_queue* queue, cl_event* event = nullptr);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
StatusCode Amax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
StatusCode Amin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
StatusCode Max(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_command_queue* queue, cl_event* event = nullptr);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
StatusCode Min(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_command_queue* queue, cl_event* event = nullptr);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
template <typename T>
StatusCode Gemv(Layout layout, Transpose a_transpose, size_t m, size_t n, T alpha, cl_mem a_buffer, size_t a_offset,
                size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
template <typename T>
StatusCode Gbmv(Layout layout, Transpose a_transpose, size_t m, size_t n, size_t kl, size_t ku, T alpha,
                cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
StatusCode Hemv(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
template <typename T>
StatusCode Hbmv(Layout layout, Triangle triangle, size_t n, size_t k, T alpha, cl_mem a_buffer, size_t a_offset,
                size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
StatusCode Hpmv(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem ap_buffer, size_t ap_offset,
                cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
StatusCode Symv(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
template <typename T>
StatusCode Sbmv(Layout layout, Triangle triangle, size_t n, size_t k, T alpha, cl_mem a_buffer, size_t a_offset,
                size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset,
                size_t y_inc, cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
StatusCode Spmv(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem ap_buffer, size_t ap_offset,
                cl_mem x_buffer, size_t x_offset, size_t x_inc, T beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
StatusCode Trmv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                cl_event* event = nullptr);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
template <typename T>
StatusCode Tbmv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, size_t k,
                cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
template <typename T>
StatusCode Tpmv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, cl_mem ap_buffer,
                size_t ap_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                cl_event* event = nullptr);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
StatusCode Trsv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                cl_event* event = nullptr);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
StatusCode Tbsv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, size_t k,
                cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_command_queue* queue, cl_event* event = nullptr);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
StatusCode Tpsv(Layout layout, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t n, cl_mem ap_buffer,
                size_t ap_offset, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                cl_event* event = nullptr);

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
StatusCode Ger(Layout layout, size_t m, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
               cl_command_queue* queue, cl_event* event = nullptr);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
StatusCode Geru(Layout layout, size_t m, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
StatusCode Gerc(Layout layout, size_t m, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
StatusCode Her(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
StatusCode Hpr(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem ap_buffer, size_t ap_offset, cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
StatusCode Her2(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
StatusCode Hpr2(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
StatusCode Syr(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
StatusCode Spr(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
               cl_mem ap_buffer, size_t ap_offset, cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
StatusCode Syr2(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
StatusCode Spr2(Layout layout, Triangle triangle, size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                cl_command_queue* queue, cl_event* event = nullptr);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
template <typename T>
StatusCode Gemm(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k, T alpha,
                cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, T beta,
                cl_mem c_buffer, size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event = nullptr,
                cl_mem temp_buffer = nullptr);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
template <typename T>
StatusCode Symm(Layout layout, Side side, Triangle triangle, size_t m, size_t n, T alpha, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, T beta, cl_mem c_buffer,
                size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
template <typename T>
StatusCode Hemm(Layout layout, Side side, Triangle triangle, size_t m, size_t n, T alpha, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, T beta, cl_mem c_buffer,
                size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
StatusCode Syrk(Layout layout, Triangle triangle, Transpose a_transpose, size_t n, size_t k, T alpha, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, T beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
template <typename T>
StatusCode Herk(Layout layout, Triangle triangle, Transpose a_transpose, size_t n, size_t k, T alpha, cl_mem a_buffer,
                size_t a_offset, size_t a_ld, T beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                cl_command_queue* queue, cl_event* event = nullptr);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
template <typename T>
StatusCode Syr2k(Layout layout, Triangle triangle, Transpose ab_transpose, size_t n, size_t k, T alpha, cl_mem a_buffer,
                 size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, T beta, cl_mem c_buffer,
                 size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
template <typename T, typename U>
StatusCode Her2k(Layout layout, Triangle triangle, Transpose ab_transpose, size_t n, size_t k, T alpha, cl_mem a_buffer,
                 size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, U beta, cl_mem c_buffer,
                 size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
StatusCode Trmm(Layout layout, Side side, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t m,
                size_t n, T alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset,
                size_t b_ld, cl_command_queue* queue, cl_event* event = nullptr);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
StatusCode Trsm(Layout layout, Side side, Triangle triangle, Transpose a_transpose, Diagonal diagonal, size_t m,
                size_t n, T alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer, size_t b_offset,
                size_t b_ld, cl_command_queue* queue, cl_event* event = nullptr);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
template <typename T>
StatusCode Had(size_t n, T alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset,
               size_t y_inc, T beta, cl_mem z_buffer, size_t z_offset, size_t z_inc, cl_command_queue* queue,
               cl_event* event = nullptr);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
StatusCode Omatcopy(Layout layout, Transpose a_transpose, size_t m, size_t n, T alpha, cl_mem a_buffer, size_t a_offset,
                    size_t a_ld, cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                    cl_event* event = nullptr);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
StatusCode Im2col(KernelMode kernel_mode, size_t channels, size_t height, size_t width, size_t kernel_h,
                  size_t kernel_w, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                  size_t dilation_w, cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                  cl_command_queue* queue, cl_event* event = nullptr);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
template <typename T>
StatusCode Col2im(KernelMode kernel_mode, size_t channels, size_t height, size_t width, size_t kernel_h,
                  size_t kernel_w, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                  size_t dilation_w, cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                  cl_command_queue* queue, cl_event* event = nullptr);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
template <typename T>
StatusCode Convgemm(KernelMode kernel_mode, size_t channels, size_t height, size_t width, size_t kernel_h,
                    size_t kernel_w, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                    size_t dilation_w, size_t num_kernels, size_t batch_count, cl_mem im_buffer, size_t im_offset,
                    cl_mem kernel_buffer, size_t kernel_offset, cl_mem result_buffer, size_t result_offset,
                    cl_command_queue* queue, cl_event* event = nullptr);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
StatusCode AxpyBatched(size_t n, const T* alphas, cl_mem x_buffer, const size_t* x_offsets, size_t x_inc,
                       cl_mem y_buffer, const size_t* y_offsets, size_t y_inc, size_t batch_count,
                       cl_command_queue* queue, cl_event* event = nullptr);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
template <typename T>
StatusCode GemmBatched(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k,
                       const T* alphas, cl_mem a_buffer, const size_t* a_offsets, size_t a_ld, cl_mem b_buffer,
                       const size_t* b_offsets, size_t b_ld, const T* betas, cl_mem c_buffer, const size_t* c_offsets,
                       size_t c_ld, size_t batch_count, cl_command_queue* queue, cl_event* event = nullptr);

// StridedBatched version of GEMM:
// SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
template <typename T>
StatusCode GemmStridedBatched(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k,
                              T alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, size_t a_stride, cl_mem b_buffer,
                              size_t b_offset, size_t b_ld, size_t b_stride, T beta, cl_mem c_buffer, size_t c_offset,
                              size_t c_ld, size_t c_stride, size_t batch_count, cl_command_queue* queue,
                              cl_event* event = nullptr);

// =================================================================================================

// Retrieves the required size of the temporary buffer for the GEMM kernel (optional)
template <typename T>
StatusCode GemmTempBufferSize(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k,
                              size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld, size_t c_offset, size_t c_ld,
                              cl_command_queue* queue, size_t& temp_buffer_size);

// =================================================================================================

// CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on
// for the same device. This cache can be cleared to free up system memory or in case of debugging.
StatusCode PUBLIC_API ClearCache();

// The cache can also be pre-initialized for a specific device with all possible CLBlast kernels.
// Further CLBlast routine calls will then run at maximum speed.
StatusCode PUBLIC_API FillCache(cl_device_id device);

// =================================================================================================

// Retrieves current tuning parameters for a specific device-precision-kernel combination
StatusCode PUBLIC_API RetrieveParameters(cl_device_id device, const std::string& kernel_name, Precision precision,
                                         std::unordered_map<std::string, size_t>& parameters);

// Overrides tuning parameters for a specific device-precision-kernel combination. The next time
// the target routine is called it will re-compile and use the new parameters from then on.
StatusCode PUBLIC_API OverrideParameters(cl_device_id device, const std::string& kernel_name, Precision precision,
                                         const std::unordered_map<std::string, size_t>& parameters);

// =================================================================================================

// Tunes the "Xaxpy" kernel, used for many level-1 routines such as XAXPY, XCOPY, and XSWAP
template <typename T>
StatusCode TuneXaxpy(cl_command_queue* queue, size_t n, double fraction,
                     std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Xdot" kernel, used for level-1 reduction routines such as XDOT, XMAX, and XSUM
template <typename T>
StatusCode TuneXdot(cl_command_queue* queue, size_t n, double fraction,
                    std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Xgemv" kernel, used for matrix-vector level-2 routines such as XGEMV, XGBMV, and XHEMV
template <typename T>
StatusCode TuneXgemv(cl_command_queue* queue, size_t m, size_t n, double fraction,
                     std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Xger" kernel, used for matrix update level-2 routines such as XGER, XHER, and XSYR2
template <typename T>
StatusCode TuneXger(cl_command_queue* queue, size_t m, size_t n, double fraction,
                    std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Xgemm" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TuneXgemm(cl_command_queue* queue, size_t m, size_t n, size_t k, double fraction,
                     std::unordered_map<std::string, size_t>& parameters);

// Tunes the "XgemmDiret" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TuneXgemmDirect(cl_command_queue* queue, size_t m, size_t n, size_t k, double fraction,
                           std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Copy" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TuneCopy(cl_command_queue* queue, size_t m, size_t n, double fraction,
                    std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Pad" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TunePad(cl_command_queue* queue, size_t m, size_t n, double fraction,
                   std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Transpose" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TuneTranspose(cl_command_queue* queue, size_t m, size_t n, double fraction,
                         std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Padtranspose" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode TunePadtranspose(cl_command_queue* queue, size_t m, size_t n, double fraction,
                            std::unordered_map<std::string, size_t>& parameters);

// Tunes the "Xgemm" kernel, used for the level-3 routine XTRSM
template <typename T>
StatusCode TuneInvert(cl_command_queue* queue, size_t m, size_t n, size_t k, double fraction,
                      std::unordered_map<std::string, size_t>& parameters);

// =================================================================================================

}  // namespace clblast

// CLBLAST_CLBLAST_H_
#endif
