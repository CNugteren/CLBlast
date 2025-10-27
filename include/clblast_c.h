
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the plain C interface to the CLBlast BLAS routines, the counter-part of the
// normal 'clblast.h' C++ header.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_C_H_
#define CLBLAST_CLBLAST_C_H_

#include <stddef.h>

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

// The C interface
#ifdef __cplusplus
extern "C" {
#endif

// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes.
typedef enum CLBlastStatusCode_ {

  // Status codes in common with the OpenCL standard
  CLBlastSuccess = 0,                      // CL_SUCCESS
  CLBlastOpenCLCompilerNotAvailable = -3,  // CL_COMPILER_NOT_AVAILABLE
  CLBlastTempBufferAllocFailure = -4,      // CL_MEM_OBJECT_ALLOCATION_FAILURE
  CLBlastOpenCLOutOfResources = -5,        // CL_OUT_OF_RESOURCES
  CLBlastOpenCLOutOfHostMemory = -6,       // CL_OUT_OF_HOST_MEMORY
  CLBlastOpenCLBuildProgramFailure = -11,  // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
  CLBlastInvalidValue = -30,               // CL_INVALID_VALUE
  CLBlastInvalidCommandQueue = -36,        // CL_INVALID_COMMAND_QUEUE
  CLBlastInvalidMemObject = -38,           // CL_INVALID_MEM_OBJECT
  CLBlastInvalidBinary = -42,              // CL_INVALID_BINARY
  CLBlastInvalidBuildOptions = -43,        // CL_INVALID_BUILD_OPTIONS
  CLBlastInvalidProgram = -44,             // CL_INVALID_PROGRAM
  CLBlastInvalidProgramExecutable = -45,   // CL_INVALID_PROGRAM_EXECUTABLE
  CLBlastInvalidKernelName = -46,          // CL_INVALID_KERNEL_NAME
  CLBlastInvalidKernelDefinition = -47,    // CL_INVALID_KERNEL_DEFINITION
  CLBlastInvalidKernel = -48,              // CL_INVALID_KERNEL
  CLBlastInvalidArgIndex = -49,            // CL_INVALID_ARG_INDEX
  CLBlastInvalidArgValue = -50,            // CL_INVALID_ARG_VALUE
  CLBlastInvalidArgSize = -51,             // CL_INVALID_ARG_SIZE
  CLBlastInvalidKernelArgs = -52,          // CL_INVALID_KERNEL_ARGS
  CLBlastInvalidLocalNumDimensions = -53,  // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  CLBlastInvalidLocalThreadsTotal = -54,   // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  CLBlastInvalidLocalThreadsDim = -55,     // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
  CLBlastInvalidGlobalOffset = -56,        // CL_INVALID_GLOBAL_OFFSET
  CLBlastInvalidEventWaitList = -57,       // CL_INVALID_EVENT_WAIT_LIST
  CLBlastInvalidEvent = -58,               // CL_INVALID_EVENT
  CLBlastInvalidOperation = -59,           // CL_INVALID_OPERATION
  CLBlastInvalidBufferSize = -61,          // CL_INVALID_BUFFER_SIZE
  CLBlastInvalidGlobalWorkSize = -63,      // CL_INVALID_GLOBAL_WORK_SIZE

  // Status codes in common with the clBLAS library
  CLBlastNotImplemented = -1024,       // Routine or functionality not implemented yet
  CLBlastInvalidMatrixA = -1022,       // Matrix A is not a valid OpenCL buffer
  CLBlastInvalidMatrixB = -1021,       // Matrix B is not a valid OpenCL buffer
  CLBlastInvalidMatrixC = -1020,       // Matrix C is not a valid OpenCL buffer
  CLBlastInvalidVectorX = -1019,       // Vector X is not a valid OpenCL buffer
  CLBlastInvalidVectorY = -1018,       // Vector Y is not a valid OpenCL buffer
  CLBlastInvalidDimension = -1017,     // Dimensions M, N, and K have to be larger than zero
  CLBlastInvalidLeadDimA = -1016,      // LD of A is smaller than the matrix's first dimension
  CLBlastInvalidLeadDimB = -1015,      // LD of B is smaller than the matrix's first dimension
  CLBlastInvalidLeadDimC = -1014,      // LD of C is smaller than the matrix's first dimension
  CLBlastInvalidIncrementX = -1013,    // Increment of vector X cannot be zero
  CLBlastInvalidIncrementY = -1012,    // Increment of vector Y cannot be zero
  CLBlastInsufficientMemoryA = -1011,  // Matrix A's OpenCL buffer is too small
  CLBlastInsufficientMemoryB = -1010,  // Matrix B's OpenCL buffer is too small
  CLBlastInsufficientMemoryC = -1009,  // Matrix C's OpenCL buffer is too small
  CLBlastInsufficientMemoryX = -1008,  // Vector X's OpenCL buffer is too small
  CLBlastInsufficientMemoryY = -1007,  // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  CLBlastInsufficientMemoryTemp = -2050,    // Temporary buffer provided to GEMM routine is too small
  CLBlastInvalidBatchCount = -2049,         // The batch count needs to be positive
  CLBlastInvalidOverrideKernel = -2048,     // Trying to override parameters for an invalid kernel
  CLBlastMissingOverrideParameter = -2047,  // Missing override parameter(s) for the target kernel
  CLBlastInvalidLocalMemUsage = -2046,      // Not enough local memory available on this device
  CLBlastNoHalfPrecision = -2045,           // Half precision (16-bits) not supported by the device
  CLBlastNoDoublePrecision = -2044,         // Double precision (64-bits) not supported by the device
  CLBlastInvalidVectorScalar = -2043,       // The unit-sized vector is not a valid OpenCL buffer
  CLBlastInsufficientMemoryScalar = -2042,  // The unit-sized vector's OpenCL buffer is too small
  CLBlastDatabaseError = -2041,             // Entry for the device was not found in the database
  CLBlastUnknownError = -2040,              // A catch-all error code representing an unspecified error
  CLBlastUnexpectedError = -2039,           // A catch-all error code representing an unexpected exception
  CLBlastInvalidVectorZ = -2038,            // Vector Z is not a valid OpenCL buffer
  CLBlastInvalidIncrementZ = -2037,         // Increment of vector Z cannot be zero
  CLBlastInsufficientMemoryZ = -2036,       // Vector Z's OpenCL buffer is too small
} CLBlastStatusCode;

// Matrix layout and transpose types
typedef enum CLBlastLayout_ { CLBlastLayoutRowMajor = 101, CLBlastLayoutColMajor = 102 } CLBlastLayout;
typedef enum CLBlastTranspose_ {
  CLBlastTransposeNo = 111,
  CLBlastTransposeYes = 112,
  CLBlastTransposeConjugate = 113
} CLBlastTranspose;
typedef enum CLBlastTriangle_ { CLBlastTriangleUpper = 121, CLBlastTriangleLower = 122 } CLBlastTriangle;
typedef enum CLBlastDiagonal_ { CLBlastDiagonalNonUnit = 131, CLBlastDiagonalUnit = 132 } CLBlastDiagonal;
typedef enum CLBlastSide_ { CLBlastSideLeft = 141, CLBlastSideRight = 142 } CLBlastSide;
typedef enum CLBlastKernelMode_ {
  CLBlastKernelModeCrossCorrelation = 151,
  CLBlastKernelModeConvolution = 152
} CLBlastKernelMode;

// Precision enum (values in bits)
typedef enum CLBlastPrecision_ {
  CLBlastPrecisionHalf = 16,
  CLBlastPrecisionSingle = 32,
  CLBlastPrecisionDouble = 64,
  CLBlastPrecisionComplexSingle = 3232,
  CLBlastPrecisionComplexDouble = 6464
} CLBlastPrecision;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
CLBlastStatusCode PUBLIC_API CLBlastSrotg(cl_mem sa_buffer, size_t sa_offset, cl_mem sb_buffer, size_t sb_offset,
                                          cl_mem sc_buffer, size_t sc_offset, cl_mem ss_buffer, size_t ss_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotg(cl_mem sa_buffer, size_t sa_offset, cl_mem sb_buffer, size_t sb_offset,
                                          cl_mem sc_buffer, size_t sc_offset, cl_mem ss_buffer, size_t ss_offset,
                                          cl_command_queue* queue, cl_event* event);

// Generate modified givens plane rotation: SROTMG/DROTMG
CLBlastStatusCode PUBLIC_API CLBlastSrotmg(cl_mem sd1_buffer, size_t sd1_offset, cl_mem sd2_buffer, size_t sd2_offset,
                                           cl_mem sx1_buffer, size_t sx1_offset, cl_mem sy1_buffer, size_t sy1_offset,
                                           cl_mem sparam_buffer, size_t sparam_offset, cl_command_queue* queue,
                                           cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotmg(cl_mem sd1_buffer, size_t sd1_offset, cl_mem sd2_buffer, size_t sd2_offset,
                                           cl_mem sx1_buffer, size_t sx1_offset, cl_mem sy1_buffer, size_t sy1_offset,
                                           cl_mem sparam_buffer, size_t sparam_offset, cl_command_queue* queue,
                                           cl_event* event);

// Apply givens plane rotation: SROT/DROT
CLBlastStatusCode PUBLIC_API CLBlastSrot(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                         size_t y_offset, size_t y_inc, float cos, float sin, cl_command_queue* queue,
                                         cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrot(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                         size_t y_offset, size_t y_inc, double cos, double sin, cl_command_queue* queue,
                                         cl_event* event);

// Apply modified givens plane rotation: SROTM/DROTM
CLBlastStatusCode PUBLIC_API CLBlastSrotm(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem sparam_buffer, size_t sparam_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotm(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem sparam_buffer, size_t sparam_offset,
                                          cl_command_queue* queue, cl_event* event);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
CLBlastStatusCode PUBLIC_API CLBlastSswap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDswap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCswap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZswap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHswap(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
CLBlastStatusCode PUBLIC_API CLBlastSscal(size_t n, float alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDscal(size_t n, double alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCscal(size_t n, cl_float2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZscal(size_t n, cl_double2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHscal(size_t n, cl_half alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
CLBlastStatusCode PUBLIC_API CLBlastScopy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDcopy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCcopy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZcopy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHcopy(size_t n, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
CLBlastStatusCode PUBLIC_API CLBlastSaxpy(size_t n, float alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDaxpy(size_t n, double alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCaxpy(size_t n, cl_float2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZaxpy(size_t n, cl_double2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHaxpy(size_t n, cl_half alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);

// Dot product of two vectors: SDOT/DDOT/HDOT
CLBlastStatusCode PUBLIC_API CLBlastSdot(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDdot(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHdot(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors: CDOTU/ZDOTU
CLBlastStatusCode PUBLIC_API CLBlastCdotu(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZdotu(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
CLBlastStatusCode PUBLIC_API CLBlastCdotc(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZdotc(size_t n, cl_mem dot_buffer, size_t dot_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
CLBlastStatusCode PUBLIC_API CLBlastSnrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDnrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScnrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDznrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHnrm2(size_t n, cl_mem nrm2_buffer, size_t nrm2_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
CLBlastStatusCode PUBLIC_API CLBlastSasum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDasum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScasum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDzasum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHasum(size_t n, cl_mem asum_buffer, size_t asum_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
CLBlastStatusCode PUBLIC_API CLBlastSsum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScsum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDzsum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsum(size_t n, cl_mem sum_buffer, size_t sum_offset, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
CLBlastStatusCode PUBLIC_API CLBlastiSamax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDamax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCamax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZamax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHamax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
CLBlastStatusCode PUBLIC_API CLBlastiSamin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDamin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCamin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZamin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHamin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                           size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
CLBlastStatusCode PUBLIC_API CLBlastiSmax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDmax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCmax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZmax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHmax(size_t n, cl_mem imax_buffer, size_t imax_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
CLBlastStatusCode PUBLIC_API CLBlastiSmin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDmin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCmin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZmin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHmin(size_t n, cl_mem imin_buffer, size_t imin_offset, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
CLBlastStatusCode PUBLIC_API CLBlastSgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, float beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, double beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_float2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_double2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_half beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
CLBlastStatusCode PUBLIC_API CLBlastSgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          size_t kl, size_t ku, float alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, float beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          size_t kl, size_t ku, double alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, double beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          size_t kl, size_t ku, cl_float2 alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_float2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          size_t kl, size_t ku, cl_double2 alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_double2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                          size_t kl, size_t ku, cl_half alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_half beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
CLBlastStatusCode PUBLIC_API CLBlastChemv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_float2 alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_float2 beta, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhemv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_double2 alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_double2 beta, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_command_queue* queue, cl_event* event);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
CLBlastStatusCode PUBLIC_API CLBlastChbmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, size_t k,
                                          cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_float2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhbmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, size_t k,
                                          cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_double2 beta,
                                          cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_command_queue* queue,
                                          cl_event* event);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
CLBlastStatusCode PUBLIC_API CLBlastChpmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_float2 alpha,
                                          cl_mem ap_buffer, size_t ap_offset, cl_mem x_buffer, size_t x_offset,
                                          size_t x_inc, cl_float2 beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_double2 alpha,
                                          cl_mem ap_buffer, size_t ap_offset, cl_mem x_buffer, size_t x_offset,
                                          size_t x_inc, cl_double2 beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
CLBlastStatusCode PUBLIC_API CLBlastSsymv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, float beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsymv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, double beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsymv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_half beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
CLBlastStatusCode PUBLIC_API CLBlastSsbmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, size_t k,
                                          float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, float beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsbmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, size_t k,
                                          double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, double beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsbmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, size_t k,
                                          cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_half beta, cl_mem y_buffer, size_t y_offset,
                                          size_t y_inc, cl_command_queue* queue, cl_event* event);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
CLBlastStatusCode PUBLIC_API CLBlastSspmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                          cl_mem ap_buffer, size_t ap_offset, cl_mem x_buffer, size_t x_offset,
                                          size_t x_inc, float beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                          cl_mem ap_buffer, size_t ap_offset, cl_mem x_buffer, size_t x_offset,
                                          size_t x_inc, double beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspmv(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                          cl_mem ap_buffer, size_t ap_offset, cl_mem x_buffer, size_t x_offset,
                                          size_t x_inc, cl_half beta, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
CLBlastStatusCode PUBLIC_API CLBlastStrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
CLBlastStatusCode PUBLIC_API CLBlastStbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
CLBlastStatusCode PUBLIC_API CLBlastStpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
CLBlastStatusCode PUBLIC_API CLBlastStrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
CLBlastStatusCode PUBLIC_API CLBlastStbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, size_t k, cl_mem a_buffer,
                                          size_t a_offset, size_t a_ld, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
CLBlastStatusCode PUBLIC_API CLBlastStpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          CLBlastDiagonal diagonal, size_t n, cl_mem ap_buffer, size_t ap_offset,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_command_queue* queue,
                                          cl_event* event);

// General rank-1 matrix update: SGER/DGER/HGER
CLBlastStatusCode PUBLIC_API CLBlastSger(CLBlastLayout layout, size_t m, size_t n, float alpha, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                         cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDger(CLBlastLayout layout, size_t m, size_t n, double alpha, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                         cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHger(CLBlastLayout layout, size_t m, size_t n, cl_half alpha, cl_mem x_buffer,
                                         size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                         cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                         cl_event* event);

// General rank-1 complex matrix update: CGERU/ZGERU
CLBlastStatusCode PUBLIC_API CLBlastCgeru(CLBlastLayout layout, size_t m, size_t n, cl_float2 alpha, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgeru(CLBlastLayout layout, size_t m, size_t n, cl_double2 alpha, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                          cl_event* event);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
CLBlastStatusCode PUBLIC_API CLBlastCgerc(CLBlastLayout layout, size_t m, size_t n, cl_float2 alpha, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgerc(CLBlastLayout layout, size_t m, size_t n, cl_double2 alpha, cl_mem x_buffer,
                                          size_t x_offset, size_t x_inc, cl_mem y_buffer, size_t y_offset, size_t y_inc,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_command_queue* queue,
                                          cl_event* event);

// Hermitian rank-1 matrix update: CHER/ZHER
CLBlastStatusCode PUBLIC_API CLBlastCher(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem a_buffer,
                                         size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem a_buffer,
                                         size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
CLBlastStatusCode PUBLIC_API CLBlastChpr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem ap_buffer,
                                         size_t ap_offset, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem ap_buffer,
                                         size_t ap_offset, cl_command_queue* queue, cl_event* event);

// Hermitian rank-2 matrix update: CHER2/ZHER2
CLBlastStatusCode PUBLIC_API CLBlastCher2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_float2 alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_double2 alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
CLBlastStatusCode PUBLIC_API CLBlastChpr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_float2 alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_double2 alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
CLBlastStatusCode PUBLIC_API CLBlastSsyr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem a_buffer,
                                         size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem a_buffer,
                                         size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem a_buffer,
                                         size_t a_offset, size_t a_ld, cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
CLBlastStatusCode PUBLIC_API CLBlastSspr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem ap_buffer,
                                         size_t ap_offset, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem ap_buffer,
                                         size_t ap_offset, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspr(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                         cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem ap_buffer,
                                         size_t ap_offset, cl_command_queue* queue, cl_event* event);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
CLBlastStatusCode PUBLIC_API CLBlastSsyr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
CLBlastStatusCode PUBLIC_API CLBlastSspr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, float alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, double alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspr2(CLBlastLayout layout, CLBlastTriangle triangle, size_t n, cl_half alpha,
                                          cl_mem x_buffer, size_t x_offset, size_t x_inc, cl_mem y_buffer,
                                          size_t y_offset, size_t y_inc, cl_mem ap_buffer, size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
CLBlastStatusCode PUBLIC_API CLBlastSgemm(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                          CLBlastTranspose b_transpose, size_t m, size_t n, size_t k, float alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, float beta, cl_mem c_buffer, size_t c_offset,
                                          size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemm(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                          CLBlastTranspose b_transpose, size_t m, size_t n, size_t k, double alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, double beta, cl_mem c_buffer, size_t c_offset,
                                          size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemm(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                          CLBlastTranspose b_transpose, size_t m, size_t n, size_t k, cl_float2 alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_float2 beta, cl_mem c_buffer,
                                          size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemm(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                          CLBlastTranspose b_transpose, size_t m, size_t n, size_t k, cl_double2 alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_double2 beta, cl_mem c_buffer,
                                          size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemm(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                          CLBlastTranspose b_transpose, size_t m, size_t n, size_t k, cl_half alpha,
                                          cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_half beta, cl_mem c_buffer, size_t c_offset,
                                          size_t c_ld, cl_command_queue* queue, cl_event* event);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
CLBlastStatusCode PUBLIC_API CLBlastSsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, float beta, cl_mem c_buffer,
                                          size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, double beta, cl_mem c_buffer,
                                          size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_float2 beta,
                                          cl_mem c_buffer, size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_double2 beta,
                                          cl_mem c_buffer, size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_half beta, cl_mem c_buffer,
                                          size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
CLBlastStatusCode PUBLIC_API CLBlastChemm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_float2 beta,
                                          cl_mem c_buffer, size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhemm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, size_t m,
                                          size_t n, cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_double2 beta,
                                          cl_mem c_buffer, size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                          cl_event* event);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
CLBlastStatusCode PUBLIC_API CLBlastSsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, float alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, float beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, double alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, double beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, cl_float2 alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_float2 beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, cl_double2 alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_double2 beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, cl_half alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, cl_half beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
CLBlastStatusCode PUBLIC_API CLBlastCherk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, float alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, float beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZherk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                                          size_t n, size_t k, double alpha, cl_mem a_buffer, size_t a_offset,
                                          size_t a_ld, double beta, cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
CLBlastStatusCode PUBLIC_API CLBlastSsyr2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, float alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, float beta, cl_mem c_buffer, size_t c_offset,
                                           size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, double alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, double beta, cl_mem c_buffer, size_t c_offset,
                                           size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsyr2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, cl_float2 alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, cl_float2 beta, cl_mem c_buffer,
                                           size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsyr2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, cl_double2 alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, cl_double2 beta, cl_mem c_buffer,
                                           size_t c_offset, size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, cl_half alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, cl_half beta, cl_mem c_buffer, size_t c_offset,
                                           size_t c_ld, cl_command_queue* queue, cl_event* event);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
CLBlastStatusCode PUBLIC_API CLBlastCher2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, cl_float2 alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, float beta, cl_mem c_buffer, size_t c_offset,
                                           size_t c_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher2k(CLBlastLayout layout, CLBlastTriangle triangle,
                                           CLBlastTranspose ab_transpose, size_t n, size_t k, cl_double2 alpha,
                                           cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                           size_t b_offset, size_t b_ld, double beta, cl_mem c_buffer, size_t c_offset,
                                           size_t c_ld, cl_command_queue* queue, cl_event* event);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
CLBlastStatusCode PUBLIC_API CLBlastStrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_command_queue* queue, cl_event* event);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
CLBlastStatusCode PUBLIC_API CLBlastStrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld, cl_mem b_buffer,
                                          size_t b_offset, size_t b_ld, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                          cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                                          CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, size_t m, size_t n,
                                          cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                          cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                          cl_event* event);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
CLBlastStatusCode PUBLIC_API CLBlastShad(size_t n, float alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                         cl_mem y_buffer, size_t y_offset, size_t y_inc, float beta, cl_mem z_buffer,
                                         size_t z_offset, size_t z_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDhad(size_t n, double alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                         cl_mem y_buffer, size_t y_offset, size_t y_inc, double beta, cl_mem z_buffer,
                                         size_t z_offset, size_t z_inc, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastChad(size_t n, cl_float2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                         cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_float2 beta,
                                         cl_mem z_buffer, size_t z_offset, size_t z_inc, cl_command_queue* queue,
                                         cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhad(size_t n, cl_double2 alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                         cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_double2 beta,
                                         cl_mem z_buffer, size_t z_offset, size_t z_inc, cl_command_queue* queue,
                                         cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHhad(size_t n, cl_half alpha, cl_mem x_buffer, size_t x_offset, size_t x_inc,
                                         cl_mem y_buffer, size_t y_offset, size_t y_inc, cl_half beta, cl_mem z_buffer,
                                         size_t z_offset, size_t z_inc, cl_command_queue* queue, cl_event* event);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
CLBlastStatusCode PUBLIC_API CLBlastSomatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                              float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                              cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                              cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDomatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                              double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                              cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                              cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastComatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                              cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                              cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                              cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZomatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                              cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                              cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                              cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHomatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, size_t m, size_t n,
                                              cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                              cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_command_queue* queue,
                                              cl_event* event);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
CLBlastStatusCode PUBLIC_API CLBlastSim2col(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDim2col(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCim2col(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZim2col(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHim2col(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem im_buffer, size_t im_offset, cl_mem col_buffer, size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
CLBlastStatusCode PUBLIC_API CLBlastScol2im(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDcol2im(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCcol2im(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZcol2im(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHcol2im(CLBlastKernelMode kernel_mode, size_t channels, size_t height, size_t width,
                                            size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                                            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                            cl_mem col_buffer, size_t col_offset, cl_mem im_buffer, size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
CLBlastStatusCode PUBLIC_API CLBlastSconvgemm(CLBlastKernelMode kernel_mode, size_t channels, size_t height,
                                              size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                              size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                                              size_t dilation_w, size_t num_kernels, size_t batch_count,
                                              cl_mem im_buffer, size_t im_offset, cl_mem kernel_buffer,
                                              size_t kernel_offset, cl_mem result_buffer, size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDconvgemm(CLBlastKernelMode kernel_mode, size_t channels, size_t height,
                                              size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                              size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                                              size_t dilation_w, size_t num_kernels, size_t batch_count,
                                              cl_mem im_buffer, size_t im_offset, cl_mem kernel_buffer,
                                              size_t kernel_offset, cl_mem result_buffer, size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHconvgemm(CLBlastKernelMode kernel_mode, size_t channels, size_t height,
                                              size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                              size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                                              size_t dilation_w, size_t num_kernels, size_t batch_count,
                                              cl_mem im_buffer, size_t im_offset, cl_mem kernel_buffer,
                                              size_t kernel_offset, cl_mem result_buffer, size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSaxpyBatched(size_t n, const float* alphas, cl_mem x_buffer,
                                                 const size_t* x_offsets, size_t x_inc, cl_mem y_buffer,
                                                 const size_t* y_offsets, size_t y_inc, size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDaxpyBatched(size_t n, const double* alphas, cl_mem x_buffer,
                                                 const size_t* x_offsets, size_t x_inc, cl_mem y_buffer,
                                                 const size_t* y_offsets, size_t y_inc, size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCaxpyBatched(size_t n, const cl_float2* alphas, cl_mem x_buffer,
                                                 const size_t* x_offsets, size_t x_inc, cl_mem y_buffer,
                                                 const size_t* y_offsets, size_t y_inc, size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZaxpyBatched(size_t n, const cl_double2* alphas, cl_mem x_buffer,
                                                 const size_t* x_offsets, size_t x_inc, cl_mem y_buffer,
                                                 const size_t* y_offsets, size_t y_inc, size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHaxpyBatched(size_t n, const cl_half* alphas, cl_mem x_buffer,
                                                 const size_t* x_offsets, size_t x_inc, cl_mem y_buffer,
                                                 const size_t* y_offsets, size_t y_inc, size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSgemmBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                 CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                 const float* alphas, cl_mem a_buffer, const size_t* a_offsets,
                                                 size_t a_ld, cl_mem b_buffer, const size_t* b_offsets, size_t b_ld,
                                                 const float* betas, cl_mem c_buffer, const size_t* c_offsets,
                                                 size_t c_ld, size_t batch_count, cl_command_queue* queue,
                                                 cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemmBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                 CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                 const double* alphas, cl_mem a_buffer, const size_t* a_offsets,
                                                 size_t a_ld, cl_mem b_buffer, const size_t* b_offsets, size_t b_ld,
                                                 const double* betas, cl_mem c_buffer, const size_t* c_offsets,
                                                 size_t c_ld, size_t batch_count, cl_command_queue* queue,
                                                 cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemmBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                 CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                 const cl_float2* alphas, cl_mem a_buffer, const size_t* a_offsets,
                                                 size_t a_ld, cl_mem b_buffer, const size_t* b_offsets, size_t b_ld,
                                                 const cl_float2* betas, cl_mem c_buffer, const size_t* c_offsets,
                                                 size_t c_ld, size_t batch_count, cl_command_queue* queue,
                                                 cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemmBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                 CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                 const cl_double2* alphas, cl_mem a_buffer, const size_t* a_offsets,
                                                 size_t a_ld, cl_mem b_buffer, const size_t* b_offsets, size_t b_ld,
                                                 const cl_double2* betas, cl_mem c_buffer, const size_t* c_offsets,
                                                 size_t c_ld, size_t batch_count, cl_command_queue* queue,
                                                 cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemmBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                 CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                 const cl_half* alphas, cl_mem a_buffer, const size_t* a_offsets,
                                                 size_t a_ld, cl_mem b_buffer, const size_t* b_offsets, size_t b_ld,
                                                 const cl_half* betas, cl_mem c_buffer, const size_t* c_offsets,
                                                 size_t c_ld, size_t batch_count, cl_command_queue* queue,
                                                 cl_event* event);

// StridedBatched version of GEMM:
// SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSgemmStridedBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        size_t a_stride, cl_mem b_buffer, size_t b_offset, size_t b_ld,
                                                        size_t b_stride, float beta, cl_mem c_buffer, size_t c_offset,
                                                        size_t c_ld, size_t c_stride, size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemmStridedBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        size_t a_stride, cl_mem b_buffer, size_t b_offset, size_t b_ld,
                                                        size_t b_stride, double beta, cl_mem c_buffer, size_t c_offset,
                                                        size_t c_ld, size_t c_stride, size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemmStridedBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        size_t a_stride, cl_mem b_buffer, size_t b_offset, size_t b_ld,
                                                        size_t b_stride, cl_float2 beta, cl_mem c_buffer,
                                                        size_t c_offset, size_t c_ld, size_t c_stride,
                                                        size_t batch_count, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemmStridedBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        size_t a_stride, cl_mem b_buffer, size_t b_offset, size_t b_ld,
                                                        size_t b_stride, cl_double2 beta, cl_mem c_buffer,
                                                        size_t c_offset, size_t c_ld, size_t c_stride,
                                                        size_t batch_count, cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemmStridedBatched(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        size_t a_stride, cl_mem b_buffer, size_t b_offset, size_t b_ld,
                                                        size_t b_stride, cl_half beta, cl_mem c_buffer, size_t c_offset,
                                                        size_t c_ld, size_t c_stride, size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);

// =================================================================================================
// General matrix-matrix multiplication with temporary buffer from user (optional, for advanced users):
// SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
CLBlastStatusCode PUBLIC_API CLBlastSgemmWithTempBuffer(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        float alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        cl_mem b_buffer, size_t b_offset, size_t b_ld, float beta,
                                                        cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastDgemmWithTempBuffer(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        double alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        cl_mem b_buffer, size_t b_offset, size_t b_ld, double beta,
                                                        cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastCgemmWithTempBuffer(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_float2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_float2 beta,
                                                        cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastZgemmWithTempBuffer(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_double2 alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_double2 beta,
                                                        cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastHgemmWithTempBuffer(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        cl_half alpha, cl_mem a_buffer, size_t a_offset, size_t a_ld,
                                                        cl_mem b_buffer, size_t b_offset, size_t b_ld, cl_half beta,
                                                        cl_mem c_buffer, size_t c_offset, size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);

// =================================================================================================
// Retrieves the required size of the temporary buffer for the GEMM kernel: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM (optional)
CLBlastStatusCode PUBLIC_API CLBlastSGemmTempBufferSize(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld,
                                                        size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastDGemmTempBufferSize(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld,
                                                        size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastCGemmTempBufferSize(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld,
                                                        size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastZGemmTempBufferSize(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld,
                                                        size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastHGemmTempBufferSize(CLBlastLayout layout, CLBlastTranspose a_transpose,
                                                        CLBlastTranspose b_transpose, size_t m, size_t n, size_t k,
                                                        size_t a_offset, size_t a_ld, size_t b_offset, size_t b_ld,
                                                        size_t c_offset, size_t c_ld, cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

// =================================================================================================

// CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on
// for the same device. This cache can be cleared to free up system memory or in case of debugging.
CLBlastStatusCode PUBLIC_API CLBlastClearCache();

// The cache can also be pre-initialized for a specific device with all possible CLBlast kernels.
// Further CLBlast routine calls will then run at maximum speed.
CLBlastStatusCode PUBLIC_API CLBlastFillCache(cl_device_id device);

// =================================================================================================

// Overrides tuning parameters for a specific device-precision-kernel combination. The next time
// the target routine is called it will re-compile and use the new parameters from then on.
CLBlastStatusCode PUBLIC_API CLBlastOverrideParameters(cl_device_id device, const char* kernel_name,
                                                       CLBlastPrecision precision, size_t num_parameters,
                                                       const char** parameters_names, const size_t* parameters_values);

// =================================================================================================

#ifdef __cplusplus
}  // extern "C"
#endif

// CLBLAST_CLBLAST_C_H_
#endif
