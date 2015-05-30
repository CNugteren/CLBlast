/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#ifndef CLAMDBLAS_H_
#define CLAMDBLAS_H_

/*! @file clAmdBlas.h
 * /note clAmdBlas.h is a deprecated header file.  
 * This header is provided to help projects that were written with the older clAmdBlas codebase, to help them 
 * port to the new API at their own schedule.  It will not be maintained or updated, and will be removed after 
 * a reasonable amount of time has passed.  All new code should be written against clFFT.h.  
 * Older projects should migrate to the new header at their earliest convenience.
 */

/**
 * @mainpage OpenCL BLAS
 *
 */

#include "clBLAS.h"

/* The following header defines a fixed version number as this header is deprecated and won't be updated */
#include "clAmdBlas.version.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup OVERVIEW Overview
 *
 * This library provides an implementation of the Basic Linear Algebra Subprograms levels 1, 2 and 3,
 * using OpenCL and optimized for AMD GPU hardware. It provides BLAS-1 functions
 * SWAP, SCAL, COPY, AXPY, DOT, DOTU, DOTC, ROTG, ROTMG, ROT, ROTM, iAMAX, ASUM and NRM2,
 * BLAS-2 functions GEMV, SYMV, TRMV, TRSV, HEMV, SYR, SYR2, HER, HER2, GER, GERU, GERC,
 * TPMV, SPMV, HPMV, TPSV, SPR, SPR2, HPR, HPR2, GBMV, TBMV, SBMV, HBMV and TBSV
 * and BLAS-3 functions GEMM, SYMM, TRMM, TRSM, HEMM, HERK, HER2K, SYRK and SYR2K.
 *
 * This library’s primary goal is to assist the end user to enqueue OpenCL
 * kernels to process BLAS functions in an OpenCL-efficient manner, while
 * keeping interfaces familiar to users who know how to use BLAS. All
 * functions accept matrices through buffer objects.
 *
 * @section deprecated
 * This library provided support for the creation of scratch images to achieve better performance
 * on older <a href="http://developer.amd.com/gpu/AMDAPPSDK/Pages/default.aspx">AMD APP SDK's</a>. 
 * However, memory buffers now give the same performance as buffers objects in the current SDK's. 
 * Scratch image buffers are being deprecated and users are advised not to use scratch images in
 * new applications.
 */

/**
 * @defgroup TYPES clAmdBlas types
 */
/*@{*/


/*	Since there is no method to inherit or extend an enum, clAmdBlasOrder is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasOrder and clAmdBlasOrder
	*/
#define clAmdBlasRowMajor clblasRowMajor
#define clAmdBlasColumnMajor clblasColumnMajor

typedef enum clblasOrder_ clAmdBlasOrder;

/*	Since there is no method to inherit or extend an enum, clAmdBlasTranspose is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasTranspose and clAmdBlasTranspose
	*/
#define clAmdBlasNoTrans clblasNoTrans
#define clAmdBlasTrans clblasTrans
#define clAmdBlasConjTrans clblasConjTrans

typedef enum clblasTranspose_ clAmdBlasTranspose;

/*	Since there is no method to inherit or extend an enum, clAmdBlasUplo is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasUplo and clAmdBlasUplo
	*/
#define clAmdBlasUpper clblasUpper
#define clAmdBlasLower clblasLower

typedef enum clblasUplo_ clAmdBlasUplo;

/*	Since there is no method to inherit or extend an enum, clAmdBlasDiag is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasDiag and clAmdBlasDiag
	*/
#define clAmdBlasUnit clblasUnit
#define clAmdBlasNonUnit clblasNonUnit

typedef enum clblasDiag_ clAmdBlasDiag;

/*	Since there is no method to inherit or extend an enum, clAmdBlasSide is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasSide and clAmdBlasSide
	*/
#define clAmdBlasLeft clblasLeft
#define clAmdBlasRight clblasRight

typedef enum clblasSide_ clAmdBlasSide;

/*	Since there is no method to inherit or extend an enum, clAmdBlasStatus is now a 
	set of macro's and typedefs that 'behave' like an enum.  The advantage is there
	is no need to cast between clblasStatus and clAmdBlasStatus
	*/
#define clAmdBlasSuccess clblasSuccess
#define clAmdBlasInvalidValue clblasInvalidValue
#define clAmdBlasInvalidCommandQueue clblasInvalidCommandQueue
#define clAmdBlasInvalidContext clblasInvalidContext
#define clAmdBlasInvalidMemObject clblasInvalidMemObject
#define clAmdBlasInvalidDevice clblasInvalidDevice
#define clAmdBlasInvalidEventWaitList clblasInvalidEventWaitList
#define clAmdBlasOutOfResources clblasOutOfResources
#define clAmdBlasOutOfHostMemory clblasOutOfHostMemory
#define clAmdBlasInvalidOperation clblasInvalidOperation
#define clAmdBlasCompilerNotAvailable clblasCompilerNotAvailable
#define clAmdBlasBuildProgramFailure clblasBuildProgramFailure

#define clAmdBlasNotImplemented clblasNotImplemented
#define clAmdBlasNotInitialized clblasNotInitialized
#define clAmdBlasInvalidMatA clblasInvalidMatA
#define clAmdBlasInvalidMatB clblasInvalidMatB
#define clAmdBlasInvalidMatC clblasInvalidMatC
#define clAmdBlasInvalidVecX clblasInvalidVecX
#define clAmdBlasInvalidVecY clblasInvalidVecY
#define clAmdBlasInvalidDim clblasInvalidDim
#define clAmdBlasInvalidLeadDimA clblasInvalidLeadDimA
#define clAmdBlasInvalidLeadDimB clblasInvalidLeadDimB
#define clAmdBlasInvalidLeadDimC clblasInvalidLeadDimC
#define clAmdBlasInvalidIncX clblasInvalidIncX
#define clAmdBlasInvalidIncY clblasInvalidIncY
#define clAmdBlasInsufficientMemMatA clblasInsufficientMemMatA
#define clAmdBlasInsufficientMemMatB clblasInsufficientMemMatB
#define clAmdBlasInsufficientMemMatC clblasInsufficientMemMatC
#define clAmdBlasInsufficientMemVecX clblasInsufficientMemVecX
#define clAmdBlasInsufficientMemVecY clblasInsufficientMemVecY

typedef enum clblasStatus_ clAmdBlasStatus;


/*@}*/

/**
 * @defgroup VERSION Version information
 */
/*@{*/

/**
 * @brief Get the clAmdBlas library version info.
 *
 * @param[out] major        Location to store library's major version.
 * @param[out] minor        Location to store library's minor version.
 * @param[out] patch        Location to store library's patch version.
 *
 * @returns always \b clAmdBlasSuccess.
 *
 * @ingroup VERSION
 */
__inline clAmdBlasStatus
clAmdBlasGetVersion( cl_uint* major, cl_uint* minor, cl_uint* patch )
{
	return clblasGetVersion( major, minor, patch );
}

/*@}*/

/**
 * @defgroup INIT Initialize library
 */
/*@{*/

/**
 * @brief Initialize the clAmdBlas library.
 *
 * Must be called before any other clAmdBlas API function is invoked.
 * @note This function is not thread-safe.
 *
 * @return
 *   - \b clAmdBlasSucces on success;
 *   - \b clAmdBlasOutOfHostMemory if there is not enough of memory to allocate
 *     library's internal structures;
 *   - \b clAmdBlasOutOfResources in case of requested resources scarcity.
 *
 * @ingroup INIT
 */
__inline clAmdBlasStatus
clAmdBlasSetup( )
{
	return clblasSetup( );
}

/**
 * @brief Finalize the usage of the clAmdBlas library.
 *
 * Frees all memory allocated for different computational kernel and other
 * internal data.
 * @note This function is not thread-safe.
 *
 * @ingroup INIT
 */
__inline void
clAmdBlasTeardown( )
{
	clblasTeardown( );
}

/*@}*/

/**
 * @defgroup MISC Miscellaneous
 */
/*@{*/

/**
 * @deprecated
 * @brief Create scratch image.
 *
 * Images created with this function can be used by the library to switch from
 * a buffer-based to an image-based implementation. This can increase
 * performance up to 2 or 3 times over buffer-objects-based ones on same systems.
 * To leverage the GEMM and TRMM kernels, it is necessary to create two images.
 *
 * The following description provides bounds for the width and height arguments
 * for functions that can use scratch images.
 *
 * Let \c type be the data type of the function in question.
 *
 * Let <tt>fl4RelSize(type) = sizeof(cl_float4) / sizeof(type)</tt>.
 *
 * Let \c width1 and \c width2 be the respective values of the width argument
 * passed into the function for the two images needed to activate the image-based
 * algorithm. Similarly, let \c height1 and \c height2 be the values for the
 * height argument.
 *
 * Let <tt>div_up(x,y) = (x + y – 1) / y</tt>.
 *
 * Let <tt>round_up(x,y) = div_up(x,y) * y</tt>.
 *
 * Let <tt>round_down(x,y) = (x / y) * y</tt>.
 *
 * Then:
 *
 * For \b xGEMM there should be 2 images satisfying the following requirements:
 *   - <tt>width1 >= round_up(K, 64) / fl4RelSize(type)</tt>,
 *   - <tt>width2 >= round_up(K, 64) / fl4RelSize(type)</tt>,
 *   - <tt>height >= 64M</tt>,
 *
 * for any transA, transB, and order.
 *
 * For \b xTRMM:
 *   - <tt>width1 >= round_up(T, 64) / fl4RelSize(type)</tt>,
 *   - <tt>width2 >= round_up(N, 64) / fl4RelSize(type)</tt>,
 *   - <tt>height >= 64</tt>,
 *
 * for any transA, transB and order, where
 *   - \c T = M, for \c side = clAmdBlasLeft, and
 *   - \c T = N, for \c side = clAmdBlasRight.
 *
 * For \b xTRSM:
 *   - <tt>round_down(width, 32) * round_down(height, 32) * fl4RelSize(type) >= 1/2 * (round_up(T, 32)^2 + div_up(T, 32) * 32^2)</tt>
 *
 * for any transA, transB and order, where
 *   - \c T = M, for \c side = clAmdBlasLeft, and
 *   - \c T = N, for \c side = clAmdBlasRight.
 *
 * A call to clAmdAddScratchImage with arguments \c width and \c height reserves
 * approximately <tt>width * height * 16</tt> bytes of device memory.
 *
 * @return A created image identifier.
 *
 * @ingroup MISC
 */
cl_ulong
clAmdBlasAddScratchImage(
    cl_context context,
    size_t width,
    size_t height,
    clAmdBlasStatus *status);

/**
 * @deprecated
 * @brief Release scratch image.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if an invalid image identified is passed.
 *
 * @ingroup MISC
 */
clAmdBlasStatus
clAmdBlasRemoveScratchImage(
    cl_ulong imageID);

/*@}*/

/**
 * @defgroup BLAS1 BLAS-1 functions
 *
 * The Level 1 Basic Linear Algebra Subprograms are functions that perform
 * vector-vector operations.
 */
/*@{*/
/*@}*/

/**
 * @defgroup SWAP SWAP  - Swap elements from 2 vectors
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief interchanges two vectors of float.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SWAP
 */
__inline clAmdBlasStatus
clAmdBlasSswap( 
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSswap( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sswap.c
 * Example of how to use the @ref clAmdBlasSswap function.
 */
 
 /**
 * @brief interchanges two vectors of double.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSswap() function otherwise.
 *
 * @ingroup SWAP
 */
__inline clAmdBlasStatus
clAmdBlasDswap( 
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDswap( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/**
 * @brief interchanges two vectors of complex-float elements.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSwap() function otherwise.
 *
 * @ingroup SWAP
 */
__inline clAmdBlasStatus
clAmdBlasCswap( 
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCswap( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief interchanges two vectors of double-complex elements.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasDwap() function otherwise.
 *
 * @ingroup SWAP
 */
__inline clAmdBlasStatus
clAmdBlasZswap( 
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZswap( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup SCAL SCAL  - Scales a vector by a constant
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scales a float vector by a float constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - \b incx zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SCAL
 */
__inline clAmdBlasStatus
clAmdBlasSscal(
    size_t N,
    cl_float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sscal.c
 * Example of how to use the @ref clAmdBlasSscal function.
 */
 
 /**
 * @brief Scales a double vector by a double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSscal() function otherwise.
 *
 * @ingroup SCAL
 */
__inline clAmdBlasStatus
clAmdBlasDscal(
    size_t N,
    cl_double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Scales a complex-float vector by a complex-float constant
 *
  *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSscal() function otherwise.
 *
 * @ingroup SCAL
 */
__inline clAmdBlasStatus
clAmdBlasCscal(
    size_t N,
    cl_float2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Scales a complex-double vector by a complex-double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasDscal() function otherwise.
 *
 * @ingroup SCAL
 */
__inline clAmdBlasStatus
clAmdBlasZscal(
    size_t N,
    cl_double2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/

/**
 * @defgroup SSCAL SSCAL  - Scales a complex vector by a real constant
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scales a complex-float vector by a float constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - \b incx zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SSCAL
 */
__inline clAmdBlasStatus
clAmdBlasCsscal(
    size_t N,
    cl_float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/

/**
 * @example example_csscal.c
 * Example of how to use the @ref clAmdBlasCsscal function.
 */
 
/**
 * @brief Scales a complex-double vector by a double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCsscal() function otherwise.
 *
 * @ingroup SSCAL
 */
__inline clAmdBlasStatus
clAmdBlasZdscal(
    size_t N,
    cl_double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZdscal( N, alpha, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

 /*@}*/
 
 
/**
 * @defgroup COPY COPY  - Copies elements from vector X to vector Y
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Copies float elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup COPY
 */
__inline clAmdBlasStatus
clAmdBlasScopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasScopy( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_scopy.c
 * Example of how to use the @ref clAmdBlasScopy function.
 */
 
 /**
 * @brief Copies double elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasScopy() function otherwise.
 *
 * @ingroup COPY
 */
__inline clAmdBlasStatus
clAmdBlasDcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDcopy( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Copies complex-float elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasScopy() function otherwise.
 *
 * @ingroup COPY
 */
__inline clAmdBlasStatus
clAmdBlasCcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCcopy( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Copies complex-double elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasDcopy() function otherwise.
 *
 * @ingroup COPY
 */
__inline clAmdBlasStatus
clAmdBlasZcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZcopy( N, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
 /*@}*/
 
/**
 * @defgroup AXPY AXPY  - Scale X and add to Y
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scale vector X of float elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup AXPY
 */
__inline clAmdBlasStatus
clAmdBlasSaxpy(
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSaxpy( N, alpha, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_saxpy.c
 * Example of how to use the @ref clAmdBlasSaxpy function.
 */
 
/**
 * @brief Scale vector X of double elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
__inline clAmdBlasStatus
clAmdBlasDaxpy(
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDaxpy( N, alpha, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Scale vector X of complex-float elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
__inline clAmdBlasStatus
clAmdBlasCaxpy(
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCaxpy( N, alpha, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief Scale vector X of double-complex elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasDaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
__inline clAmdBlasStatus
clAmdBlasZaxpy(
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZaxpy( N, alpha, X, offx, incx, Y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/


/**
 * @defgroup DOT DOT  - Dot product of two vectors
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief dot product of two vectors containing float elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, \b Y or \b dotProduct object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup DOT
 */
__inline clAmdBlasStatus
clAmdBlasSdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSdot( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sdot.c
 * Example of how to use the @ref clAmdBlasSdot function.
 */
 
/**
 * @brief dot product of two vectors containing double elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSdot() function otherwise.
 *
 * @ingroup DOT
 */
__inline clAmdBlasStatus
clAmdBlasDdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDdot( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 

/**
 * @brief dot product of two vectors containing float-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSdot() function otherwise.
 *
 * @ingroup DOT
 */
__inline clAmdBlasStatus
clAmdBlasCdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCdotu( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief dot product of two vectors containing double-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSdot() function otherwise.
 *
 * @ingroup DOT
 */

__inline clAmdBlasStatus
clAmdBlasZdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZdotu( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief dot product of two vectors containing float-complex elements conjugating the first vector
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSdot() function otherwise.
 *
 * @ingroup DOT
 */

__inline clAmdBlasStatus
clAmdBlasCdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCdotc( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief dot product of two vectors containing double-complex elements conjugating the first vector
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSdot() function otherwise.
 *
 * @ingroup DOT
 */
__inline clAmdBlasStatus
clAmdBlasZdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZdotc( N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup ROTG ROTG  - Constructs givens plane rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief construct givens plane rotation on float elements
 *
 * @param[out] SA           Buffer object that contains SA
 * @param[in] offSA         Offset to SA in \b SA buffer object.
 *                          Counted in elements.
 * @param[out] SB           Buffer object that contains SB
 * @param[in] offSB         Offset to SB in \b SB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidMemObject if either \b SA, \b SB, \b C or \b S object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTG
 */
__inline clAmdBlasStatus
clAmdBlasSrotg(
    cl_mem SA,
    size_t offSA,
    cl_mem SB,
    size_t offSB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSrotg( SA, offSA, SB, offSB, C, offC, S, offS, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_srotg.c
 * Example of how to use the @ref clAmdBlasSrotg function.
 */
 
/**
 * @brief construct givens plane rotation on double elements
 *
 * @param[out] DA           Buffer object that contains DA
 * @param[in] offDA         Offset to DA in \b DA buffer object.
 *                          Counted in elements.
 * @param[out] DB           Buffer object that contains DB
 * @param[in] offDB         Offset to DB in \b DB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSrotg() function otherwise.
 *
 * @ingroup ROTG
 */
__inline clAmdBlasStatus
clAmdBlasDrotg(
    cl_mem DA,
    size_t offDA,
    cl_mem DB,
    size_t offDB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDrotg( DA, offDA, DB, offDB, C, offC, S, offS, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief construct givens plane rotation on float-complex elements
 *
 * @param[out] CA           Buffer object that contains CA
 * @param[in] offCA         Offset to CA in \b CA buffer object.
 *                          Counted in elements.
 * @param[out] CB           Buffer object that contains CB
 * @param[in] offCB         Offset to CB in \b CB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C. C is real.
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSrotg() function otherwise.
 *
 * @ingroup ROTG
 */
__inline clAmdBlasStatus
clAmdBlasCrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCrotg( CA, offCA, CB, offCB, C, offC, S, offS, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief construct givens plane rotation on double-complex elements
 *
 * @param[out] CA           Buffer object that contains CA
 * @param[in] offCA         Offset to CA in \b CA buffer object.
 *                          Counted in elements.
 * @param[out] CB           Buffer object that contains CB
 * @param[in] offCB         Offset to CB in \b CB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C. C is real.
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasDrotg() function otherwise.
 *
 * @ingroup ROTG
 */
__inline clAmdBlasStatus
clAmdBlasZrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZrotg( CA, offCA, CB, offCB, C, offC, S, offS, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/

/**
 * @defgroup ROTMG ROTMG  - Constructs the modified givens rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief construct the modified givens rotation on float elements
 *
 * @param[out] SD1          Buffer object that contains SD1
 * @param[in] offSD1        Offset to SD1 in \b SD1 buffer object.
 *                          Counted in elements.
 * @param[out] SD2          Buffer object that contains SD2
 * @param[in] offSD2        Offset to SD2 in \b SD2 buffer object.
 *                          Counted in elements.
 * @param[out] SX1          Buffer object that contains SX1
 * @param[in] offSX1        Offset to SX1 in \b SX1 buffer object.
 *                          Counted in elements.
 * @param[in] SY1           Buffer object that contains SY1
 * @param[in] offSY1        Offset to SY1 in \b SY1 buffer object.
 *                          Counted in elements.
 * @param[out] SPARAM       Buffer object that contains SPARAM array of minimum length 5
                            SPARAM(0) = SFLAG
                            SPARAM(1) = SH11
                            SPARAM(2) = SH21
                            SPARAM(3) = SH12
                            SPARAM(4) = SH22
                            
 * @param[in] offSparam     Offset to SPARAM in \b SPARAM buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidMemObject if either \b SX1, \b SY1, \b SD1, \b SD2 or \b SPARAM object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTMG
 */
__inline clAmdBlasStatus
clAmdBlasSrotmg(
    cl_mem SD1,
    size_t offSD1,
    cl_mem SD2,
    size_t offSD2,
    cl_mem SX1,
    size_t offSX1,
    const cl_mem SY1,
    size_t offSY1,
    cl_mem SPARAM,
    size_t offSparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSrotmg( SD1, offSD1, SD2, offSD2, SX1, offSX1, SY1, offSY1, SPARAM, offSparam,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_srotmg.c
 * Example of how to use the @ref clAmdBlasSrotmg function.
 */
 
/**
 * @brief construct the modified givens rotation on double elements
 *
 * @param[out] DD1          Buffer object that contains DD1
 * @param[in] offDD1        Offset to DD1 in \b DD1 buffer object.
 *                          Counted in elements.
 * @param[out] DD2          Buffer object that contains DD2
 * @param[in] offDD2        Offset to DD2 in \b DD2 buffer object.
 *                          Counted in elements.
 * @param[out] DX1          Buffer object that contains DX1
 * @param[in] offDX1        Offset to DX1 in \b DX1 buffer object.
 *                          Counted in elements.
 * @param[in] DY1           Buffer object that contains DY1
 * @param[in] offDY1        Offset to DY1 in \b DY1 buffer object.
 *                          Counted in elements.
 * @param[out] DPARAM       Buffer object that contains DPARAM array of minimum length 5
                            DPARAM(0) = DFLAG
                            DPARAM(1) = DH11
                            DPARAM(2) = DH21
                            DPARAM(3) = DH12
                            DPARAM(4) = DH22
                            
 * @param[in] offDparam     Offset to DPARAM in \b DPARAM buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSrotmg() function otherwise.
 *
 * @ingroup ROTMG
 */
__inline clAmdBlasStatus
clAmdBlasDrotmg(
    cl_mem DD1,
    size_t offDD1,
    cl_mem DD2,
    size_t offDD2,
    cl_mem DX1,
    size_t offDX1,
    const cl_mem DY1,
    size_t offDY1,
    cl_mem DPARAM,
    size_t offDparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDrotmg( DD1, offDD1, DD2, offDD2, DX1, offDX1, DY1, offDY1, DPARAM, offDparam,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/


/**
 * @defgroup ROT ROT  - Apply givens rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief applies a plane rotation for float elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos.
 * @param[in] S         S specifies the sine, sin.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROT
 */
__inline clAmdBlasStatus
clAmdBlasSrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_float C,
    cl_float S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSrot( N, X, offx, incx, Y, offy, incy, C, S,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_srot.c
 * Example of how to use the @ref clAmdBlasSrot function.
 */
 
/**
 * @brief applies a plane rotation for double elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos.
 * @param[in] S         S specifies the sine, sin.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSrot() function otherwise.
 *
 * @ingroup ROT
 */
__inline clAmdBlasStatus
clAmdBlasDrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_double C,
    cl_double S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDrot( N, X, offx, incx, Y, offy, incy, C, S,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief applies a plane rotation for float-complex elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos. This number is real
 * @param[in] S         S specifies the sine, sin. This number is real
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSrot() function otherwise.
 *
 * @ingroup ROT
 */
__inline clAmdBlasStatus
clAmdBlasCsrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_float C,
    cl_float S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsrot( N, X, offx, incx, Y, offy, incy, C, S,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief applies a plane rotation for double-complex elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos. This number is real
 * @param[in] S         S specifies the sine, sin. This number is real
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSrot() function otherwise.
 *
 * @ingroup ROT
 */
__inline clAmdBlasStatus
clAmdBlasZdrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_double C,
    cl_double S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZdrot( N, X, offx, incx, Y, offy, incy, C, S,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/
 
/**
 * @defgroup ROTM ROTM  - Apply modified givens rotation for points in the plane
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief modified givens rotation for float elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] SPARAM    Buffer object that contains SPARAM array of minimum length 5
 *                      SPARAM(1)=SFLAG
 *                      SPARAM(2)=SH11
 *                      SPARAM(3)=SH21
 *                      SPARAM(4)=SH12
 *                      SPARAM(5)=SH22
 * @param[in] offSparam Offset of first element of array \b SPARAM in buffer object.
 *                      Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b X, \b Y or \b SPARAM object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTM
 */
__inline clAmdBlasStatus
clAmdBlasSrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem SPARAM,
    size_t offSparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSrotm( N, X, offx, incx, Y, offy, incy, SPARAM, offSparam,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_srotm.c
 * Example of how to use the @ref clAmdBlasSrotm function.
 */

/**
 * @brief modified givens rotation for double elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] DPARAM    Buffer object that contains SPARAM array of minimum length 5
 *                      DPARAM(1)=DFLAG
 *                      DPARAM(2)=DH11
 *                      DPARAM(3)=DH21
 *                      DPARAM(4)=DH12
 *                      DPARAM(5)=DH22
 * @param[in] offDparam Offset of first element of array \b DPARAM in buffer object.
 *                      Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSrotm() function otherwise.
 *
 * @ingroup ROTM
 */
__inline clAmdBlasStatus
clAmdBlasDrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem DPARAM,
    size_t offDparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDrotm( N, X, offx, incx, Y, offy, incy, DPARAM, offDparam,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup NRM2 NRM2  - Euclidean norm of a vector
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief computes the euclidean norm of vector containing float elements
 *
 *  NRM2 = sqrt( X' * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if any of \b X or \b NRM2 or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup NRM2
 */
__inline clAmdBlasStatus
clAmdBlasSnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_snrm2.c
 * Example of how to use the @ref clAmdBlasSnrm2 function.
 */
 
/**
 * @brief computes the euclidean norm of vector containing double elements
 *
 *  NRM2 = sqrt( X' * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSnrm2() function otherwise.
 *
 * @ingroup NRM2
 */
__inline clAmdBlasStatus
clAmdBlasDnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief computes the euclidean norm of vector containing float-complex elements
 *
 *  NRM2 = sqrt( X**H * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value.
 *                          Note that the answer of Scnrm2 is a real value.
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSnrm2() function otherwise.
 *
 * @ingroup NRM2
 */
__inline clAmdBlasStatus
clAmdBlasScnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasScnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief computes the euclidean norm of vector containing double-complex elements
 *
 *  NRM2 = sqrt( X**H * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value.
 *                          Note that the answer of Dznrm2 is a real value.
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSnrm2() function otherwise.
 *     executable.
 *
 * @ingroup NRM2
 */
__inline clAmdBlasStatus
clAmdBlasDznrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDznrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
 
/*@}*/

/**
 * @defgroup iAMAX iAMAX  - Index of max absolute value
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief index of max absolute value in a float array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if any of \b iMax \b X or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if the context, the passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup iAMAX 
 */
__inline clAmdBlasStatus
clAmdBlasiSamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasiSamax( N, iMax, offiMax, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_isamax.c
 * Example of how to use the @ref clAmdBlasiSamax function.
 */


/**
 * @brief index of max absolute value in a double array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasiSamax() function otherwise.
 *
 * @ingroup iAMAX 
 */
__inline clAmdBlasStatus
clAmdBlasiDamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasiDamax( N, iMax, offiMax, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief index of max absolute value in a complex float array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasiSamax() function otherwise.
 *
 * @ingroup iAMAX 
 */
__inline clAmdBlasStatus
clAmdBlasiCamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasiCamax( N, iMax, offiMax, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief index of max absolute value in a complex double array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasiSamax() function otherwise.
 *
 * @ingroup iAMAX 
 */
__inline clAmdBlasStatus
clAmdBlasiZamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasiZamax( N, iMax, offiMax, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup ASUM ASUM  - Sum of absolute values
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief absolute sum of values of a vector containing float elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absoule sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if any of \b X or \b asum or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ASUM
 */
__inline clAmdBlasStatus
clAmdBlasSasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSasum( N, asum, offAsum, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sasum.c
 * Example of how to use the @ref clAmdBlasSasum function.
 */

/**
 * @brief absolute sum of values of a vector containing double elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absoulte sum value
 * @param[in] offAsum       Offset to absoule sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSasum() function otherwise.
 *
 * @ingroup ASUM
 */
__inline clAmdBlasStatus
clAmdBlasDasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDasum( N, asum, offAsum, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief absolute sum of values of a vector containing float-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absolute sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSasum() function otherwise.
 *
 * @ingroup ASUM
 */
__inline clAmdBlasStatus
clAmdBlasScasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasScasum( N, asum, offAsum, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief absolute sum of values of a vector containing double-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absolute sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSasum() function otherwise.
 *
 * @ingroup ASUM
 */
__inline clAmdBlasStatus
clAmdBlasDzasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDzasum( N, asum, offAsum, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup BLAS2 BLAS-2 functions
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * matrix-vector operations.
 */
/*@{*/
/*@}*/


/**
 * @defgroup GEMV GEMV  - General matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. Must not be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b M or \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix size or the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b x, or \b y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasSgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSgemv( order, transA, M, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sgemv.c
 * Example of how to use the @ref clAmdBlasSgemv function.
 */

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasDgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDgemv( order, transA, M, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasCgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgemv( order, transA, M, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasZgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgemv( order, transA, M, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasSgemvEx(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSgemv( order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sgemv.c
 * This is an example of how to use the @ref clAmdBlasSgemvEx function.
 */

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of \b A in the buffer
 *                      object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasDgemvEx(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDgemv( order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasCgemvEx(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgemv( order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
__inline clAmdBlasStatus
clAmdBlasZgemvEx(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgemv( order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup SYMV SYMV  - Symmetric matrix-Vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsymvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b x, or \b y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYMV
 */
__inline clAmdBlasStatus
clAmdBlasSsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsymv( order, uplo, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssymv.c
 * This is an example of how to use the @ref clAmdBlasSsymv function.
 */

/**
 * @brief Matrix-vector product with a symmetric matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsymvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsymv() function otherwise.
 *
 * @ingroup SYMV
 */
__inline clAmdBlasStatus
clAmdBlasDsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsymv( order, uplo, N, alpha, A, 0, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a symmetric matrix and float elements.
 *        Extended version.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup SYMV
 */
__inline clAmdBlasStatus
clAmdBlasSsymvEx(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsymv( order, uplo, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssymv.c
 * This is an example of how to use the @ref clAmdBlasSsymv function.
 */

/**
 * @brief Matrix-vector product with a symmetric matrix and double elements.
 *        Extended version.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSsymv() function otherwise.
 *
 * @ingroup SYMV
 */
__inline clAmdBlasStatus
clAmdBlasDsymvEx(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsymv( order, uplo, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup HEMV HEMV  - Hermitian matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMV
 */
__inline clAmdBlasStatus
clAmdBlasChemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasChemv( order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChemv() function otherwise.
 *
 * @ingroup HEMV
 */
__inline clAmdBlasStatus
clAmdBlasZhemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZhemv( order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_zhemv.cpp
 * Example of how to use the @ref clAmdBlasZhemv function.
 */
/*@}*/



/**
 * @defgroup TRMV TRMV  - Triangular matrix vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRMV
 */
__inline clAmdBlasStatus
clAmdBlasStrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrmv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_strmv.c
 * Example of how to use the @ref clAmdBlasStrmv function.
 */

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrmv() function otherwise.
 *
 * @ingroup TRMV
 */
__inline clAmdBlasStatus
clAmdBlasDtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrmv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStrmv() function.
 * @ingroup TRMV
 */
__inline clAmdBlasStatus
clAmdBlasCtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrmv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtrmv() function.
 * @ingroup TRMV
 */
__inline clAmdBlasStatus
clAmdBlasZtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrmv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/*@}*/

/**
 * @defgroup TRSV TRSV  - Triangular matrix vector Solve
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRSV
 */
__inline clAmdBlasStatus
clAmdBlasStrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrsv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_strsv.c
 * Example of how to use the @ref clAmdBlasStrsv function.
 */


/**
 * @brief solving triangular matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrsv() function otherwise.
 *
 * @ingroup TRSV
 */
__inline clAmdBlasStatus
clAmdBlasDtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrsv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief solving triangular matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStrsv() function.
 *
 * @ingroup TRSV
 */
__inline clAmdBlasStatus
clAmdBlasCtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrsv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief solving triangular matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtrsv() function.
 *
 * @ingroup TRSV
 */
__inline clAmdBlasStatus
clAmdBlasZtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrsv( order, uplo, trans, diag, N, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup GER GER   - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GER
 */
__inline clAmdBlasStatus
clAmdBlasSger(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSger( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sger.c
 * Example of how to use the @ref clAmdBlasSger function.
 */


/**
 * @brief vector-vector product with double elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSger() function otherwise.
 *
 * @ingroup GER
 */
__inline clAmdBlasStatus
clAmdBlasDger(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDger( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/

/**
 * @defgroup GERU GERU  - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERU
 */
__inline clAmdBlasStatus
clAmdBlasCgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgeru( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		   Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCgeru() function otherwise.
 *
 * @ingroup GERU
 */
__inline clAmdBlasStatus
clAmdBlasZgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgeru( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/

/**
 * @defgroup GERC GERC  - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERC
 */
__inline clAmdBlasStatus
clAmdBlasCgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgerc( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCgerc() function otherwise.
 *
 * @ingroup GERC
 */
__inline clAmdBlasStatus
clAmdBlasZgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgerc( order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/*@}*/

/**
 * @defgroup SYR SYR   - Symmetric rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR
 */
__inline clAmdBlasStatus
clAmdBlasSsyr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasSsyr( order, uplo, N, alpha, X, offx, incx, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyr() function otherwise.
 *
 * @ingroup SYR
 */
__inline clAmdBlasStatus
clAmdBlasDsyr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasDsyr( order, uplo, N, alpha, X, offx, incx, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup HER HER   - Hermitian rank 1 operation 
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER
 */
__inline clAmdBlasStatus
clAmdBlasCher(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasCher( order, uplo, N, alpha, X, offx, incx, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_cher.c
 * Example of how to use the @ref clAmdBlasCher function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar double value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher() function otherwise.
 *
 * @ingroup HER
 */
__inline clAmdBlasStatus
clAmdBlasZher(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasZher( order, uplo, N, alpha, X, offx, incx, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup SYR2 SYR2  - Symmetric rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */
__inline clAmdBlasStatus
clAmdBlasSsyr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasSsyr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */
__inline clAmdBlasStatus
clAmdBlasDsyr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasDsyr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup HER2 HER2  - Hermitian rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER2
 */
__inline clAmdBlasStatus
clAmdBlasCher2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasCher2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
* @brief Hermitian rank 2 operation with a general triangular matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher2() function otherwise.
 *
 * @ingroup HER2
 */
__inline clAmdBlasStatus
clAmdBlasZher2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasZher2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda, 
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_zher2.c
 * Example of how to use the @ref clAmdBlasZher2 function.
 */

/*@}*/

/**
 * @defgroup TPMV TPMV  - Triangular packed matrix-vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPMV
 */
__inline clAmdBlasStatus
clAmdBlasStpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStpmv( order, uplo, trans, diag, N, AP, offa, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_stpmv.c
 * Example of how to use the @ref clAmdBlasStpmv function.
 */

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStpmv() function otherwise.
 *
 * @ingroup TPMV
 */
__inline clAmdBlasStatus
clAmdBlasDtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtpmv( order, uplo, trans, diag, N, AP, offa, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
  * @brief Matrix-vector product with a packed triangular matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStpmv() function.
 * @ingroup TPMV
 */
__inline clAmdBlasStatus
clAmdBlasCtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtpmv( order, uplo, trans, diag, N, AP, offa, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtpmv() function.
 * @ingroup TPMV
 */
__inline clAmdBlasStatus
clAmdBlasZtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtpmv( order, uplo, trans, diag, N, AP, offa, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/



/**
 * @defgroup TPSV TPSV  - Triangular packed matrix vector solve 
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular packed matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */
__inline clAmdBlasStatus
clAmdBlasStpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStpsv( order, uplo, trans, diag, N, A, offa, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_stpsv.c
 * Example of how to use the @ref clAmdBlasStpsv function.
 */

/**
 * @brief solving triangular packed matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */
__inline clAmdBlasStatus
clAmdBlasDtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtpsv( order, uplo, trans, diag, N, A, offa, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief solving triangular packed matrix problems with float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */
__inline clAmdBlasStatus
clAmdBlasCtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtpsv( order, uplo, trans, diag, N, A, offa, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief solving triangular packed matrix problems with double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */
__inline clAmdBlasStatus
clAmdBlasZtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtpsv( order, uplo, trans, diag, N, A, offa, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup SPMV SPMV  - Symmetric packed matrix vector multiply
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPMV
 */
__inline clAmdBlasStatus
clAmdBlasSspmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSspmv( order, uplo, N, alpha, AP, offa, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sspmv.c
 * This is an example of how to use the @ref clAmdBlasSspmv function.
 */

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspmv() function otherwise.
 *
 * @ingroup SPMV
 */
__inline clAmdBlasStatus
clAmdBlasDspmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDspmv( order, uplo, N, alpha, AP, offa, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/



/**
 * @defgroup HPMV HPMV  - Hermitian packed matrix-vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a packed hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPMV
 */
__inline clAmdBlasStatus
clAmdBlasChpmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasChpmv( order, uplo, N, alpha, AP, offa, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_chpmv.c
 * This is an example of how to use the @ref clAmdBlasChpmv function.
 */


/**
 * @brief Matrix-vector product with a packed hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpmv() function otherwise.
 *
 * @ingroup HPMV
 */
__inline clAmdBlasStatus
clAmdBlasZhpmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZhpmv( order, uplo, N, alpha, AP, offa, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup SPR SPR   - Symmetric packed matrix rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR
 */
__inline clAmdBlasStatus
clAmdBlasSspr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasSspr( order, uplo, N, alpha, X, offx, incx, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
	
/**
 * @example example_sspr.c
 * Example of how to use the @ref clAmdBlasSspr function.
 */

/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspr() function otherwise.
 *
 * @ingroup SPR
 */
__inline clAmdBlasStatus
clAmdBlasDspr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasDspr( order, uplo, N, alpha, X, offx, incx, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

	/*@}*/

/**
 * @defgroup HPR HPR   - Hermitian packed matrix rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR
 */
__inline clAmdBlasStatus
clAmdBlasChpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasChpr( order, uplo, N, alpha, X, offx, incx, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_chpr.c
 * Example of how to use the @ref clAmdBlasChpr function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpr() function otherwise.
 *
 * @ingroup HPR
 */
__inline clAmdBlasStatus
clAmdBlasZhpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasZhpr( order, uplo, N, alpha, X, offx, incx, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

	/*@}*/

/**
 * @defgroup SPR2 SPR2  - Symmetric packed matrix rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR2
 */
__inline clAmdBlasStatus
clAmdBlasSspr2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasSspr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sspr2.c
 * Example of how to use the @ref clAmdBlasSspr2 function.
 */

/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspr2() function otherwise.
 *
 * @ingroup SPR2
 */
__inline clAmdBlasStatus
clAmdBlasDspr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasDspr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup HPR2 HPR2  - Hermitian packed matrix rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR2
 */
__inline clAmdBlasStatus
clAmdBlasChpr2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasChpr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpr2() function otherwise.
 *
 * @ingroup HPR2
 */
__inline clAmdBlasStatus
clAmdBlasZhpr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	return clblasZhpr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_zhpr2.c
 * Example of how to use the @ref clAmdBlasZhpr2 function.
 */
/*@}*/



/**
 * @defgroup GBMV GBMV  - General banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b M or \b N is zero, or
 *     - KL is greater than \b M - 1, or
 *     - KU is greater than \b N - 1, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix size or the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GBMV
 */
__inline clAmdBlasStatus
clAmdBlasSgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sgbmv.c
 * Example of how to use the @ref clAmdBlasSgbmv function.
 */


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgbmv() function otherwise.
 *
 * @ingroup GBMV
 */
__inline clAmdBlasStatus
clAmdBlasDgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasSgbmv() function.
 *
 * @ingroup GBMV
 */    
__inline clAmdBlasStatus
clAmdBlasCgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDgbmv() function.
 *
 * @ingroup GBMV
 */
__inline clAmdBlasStatus
clAmdBlasZgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup TBMV TBMV  - Triangular banded matrix vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBMV
 */
__inline clAmdBlasStatus
clAmdBlasStbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/**
 * @example example_stbmv.c
 * Example of how to use the @ref clAmdBlasStbmv function.
 */


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStbmv() function otherwise.
 *
 * @ingroup TBMV
 */
__inline clAmdBlasStatus
clAmdBlasDtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clAmdBlasStbmv() function.
 *
 * @ingroup TBMV
 */
__inline clAmdBlasStatus
clAmdBlasCtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clAmdBlasDtbmv() function.
 *
 * @ingroup TBMV
 */
__inline clAmdBlasStatus
clAmdBlasZtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/*@}*/


/**
 * @defgroup SBMV SBMV  - Symmetric banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a symmetric banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SBMV
 */
__inline clAmdBlasStatus
clAmdBlasSsbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/**
 * @example example_ssbmv.c
 * This is an example of how to use the @ref clAmdBlasSsbmv function.
 */
 
 
/**
 * @brief Matrix-vector product with a symmetric banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsbmv() function otherwise.
 *
 * @ingroup SBMV
 */
__inline clAmdBlasStatus
clAmdBlasDsbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup HBMV HBMV  - Hermitian banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HBMV
 */
__inline clAmdBlasStatus
clAmdBlasChbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasChbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_chbmv.c
 * This is an example of how to use the @ref clAmdBlasChbmv function.
 */
 
 
/**
 * @brief Matrix-vector product with a hermitian banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChbmv() function otherwise.
 *
 * @ingroup HBMV
 */
__inline clAmdBlasStatus
clAmdBlasZhbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZhbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup TBSV TBSV  - Solving triangular banded matrix
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular banded matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBSV
 */
__inline clAmdBlasStatus
clAmdBlasStbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
/**
 * @example example_stbsv.c
 * This is an example of how to use the @ref clAmdBlasStbsv function.
 */
 
 
/**
 * @brief solving triangular banded matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStbsv() function otherwise.
 *
 * @ingroup TBSV
 */
__inline clAmdBlasStatus
clAmdBlasDtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief solving triangular banded matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStbsv() function.
 *
 * @ingroup TBSV
 */
__inline clAmdBlasStatus
clAmdBlasCtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}
    
/**
 * @brief solving triangular banded matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtbsv() function.
 *
 * @ingroup TBSV
 */
__inline clAmdBlasStatus
clAmdBlasZtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup BLAS3 BLAS-3 functions
 *
 * The Level 3 Basic Linear Algebra Subprograms are funcions that perform
 * matrix-matrix operations.
 */
/*@{*/
/*@}*/

/**
 * @defgroup GEMM GEMM - General matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b K when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b K
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSgemmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasSgemm( 
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSgemm( order, transA, transB, M, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sgemm.c
 * This is an example of how to use the @ref clAmdBlasSgemm function.
 */

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDgemmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasDgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDgemm( order, transA, transB, M, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 * complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCgemmEx() instead.
 *
 * @return The same result as the clAmdBlasSgemm() function.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasCgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgemm( order, transA, transB, M, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 * complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZgemmEx() instead.
 *
 * @return The same result as the clAmdBlasDgemm() function.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasZgemm(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgemm( order, transA, transB, M, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b K when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b K
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in]  offC     Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as clAmdBlasSgemm() otherwise.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasSgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSgemm( order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_sgemm.c
 * This is an example of how to use the @ref clAmdBlasSgemmEx function.
 */

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasDgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDgemm( order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        complex elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasCgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCgemm( order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        complex elements. Exteneded version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
__inline clAmdBlasStatus
clAmdBlasZgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZgemm( order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/*@}*/

/**
 * @defgroup TRMM TRMM - Triangular matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Multiplying a matrix by a triangular matrix with float elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when it is set
 *                      to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or not less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasStrmmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N, or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasStrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrmm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @example example_strmm.c
 * This is an example of how to use the @ref clAmdBlasStrmm function.
 */

/**
 * @brief Multiplying a matrix by a triangular matrix with double elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDtrmmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasDtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrmm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Multiplying a matrix by a triangular matrix with float complex
 * elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCtrmmEx() instead.
 *
 * @return The same result as the clAmdBlasStrmm() function.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasCtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrmm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Multiplying a matrix by a triangular matrix with double complex
 * elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZtrmmEx() instead.
 *
 * @return The same result as the clAmdBlasDtrmm() function.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasZtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrmm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Multiplying a matrix by a triangular matrix with float elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when it is set
 *                      to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or not less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasStrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrmm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_strmm.c
 * This is an example of how to use the @ref clAmdBlasStrmmEx function.
 */

/**
 * @brief Multiplying a matrix by a triangular matrix with double elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasDtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrmm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Multiplying a matrix by a triangular matrix with float complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasCtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrmm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Multiplying a matrix by a triangular matrix with double complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
__inline clAmdBlasStatus
clAmdBlasZtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrmm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup TRSM TRSM - Solving triangular systems of equations
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and float elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N
 *                      when it is set to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasStrsmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasStrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrsm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @example example_strsm.c
 * This is an example of how to use the @ref clAmdBlasStrsm function.
 */

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and double elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDtrsmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasDtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrsm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and float complex elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCtrsmEx() instead.
 *
 * @return The same result as the clAmdBlasStrsm() function.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasCtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrsm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and double complex elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZtrsmEx() instead.
 *
 * @return The same result as the clAmdBlasDtrsm() function.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasZtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrsm( order, side, uplo, transA, diag, M, N, alpha, A, 0, lda, B, 0, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N
 *                      when it is set to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasStrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasStrsm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}


/**
 * @example example_strsm.c
 * This is an example of how to use the @ref clAmdBlasStrsmEx function.
 */

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasDtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDtrsm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasCtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCtrsm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise
 *
 * @ingroup TRSM
 */
__inline clAmdBlasStatus
clAmdBlasZtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZtrsm( order, side, uplo, transA, diag, M, N, alpha, A, offA, lda, B, offB, ldb,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup SYRK SYRK - Symmetric rank-k update of a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-k update of a symmetric matrix with float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasSsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsyrk( order, uplo, transA, N, K, alpha, A, 0, lda, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssyrk.c
 * This is an example of how to use the @ref clAmdBlasSsyrk function.
 */

/**
 * @brief Rank-k update of a symmetric matrix with double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasDsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsyrk( order, uplo, transA, N, K, alpha, A, 0, lda, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-k update of a symmetric matrix with complex float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasCsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsyrk( order, uplo, transA, N, K, alpha, A, 0, lda, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-k update of a symmetric matrix with complex double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasZsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZsyrk( order, uplo, transA, N, K, alpha, A, 0, lda, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasSsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsyrk( order, uplo, transA, N, K, alpha, A, offA, lda, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssyrk.c
 * This is an example of how to use the @ref clAmdBlasSsyrkEx function.
 */

/**
 * @brief Rank-k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasDsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsyrk( order, uplo, transA, N, K, alpha, A, offA, lda, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasCsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsyrk( order, uplo, transA, N, K, alpha, A, offA, lda, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *         point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
__inline clAmdBlasStatus
clAmdBlasZsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZsyrk( order, uplo, transA, N, K, alpha, A, offA, lda, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/

/**
 * @defgroup SYR2K SYR2K - Symmetric rank-2k update to a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-2k update of a symmetric matrix with float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be less
 *                       than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be less
 *                       less than \b K if \b B matches to the op(\b B) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b B or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasSsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsyr2k( order, uplo, transAB, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssyr2k.c
 * This is an example of how to use the @ref clAmdBlasSsyr2k function.
 */

/**
 * @brief Rank-2k update of a symmetric matrix with double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasDsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsyr2k( order, uplo, transAB, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-2k update of a symmetric matrix with complex float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasCsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsyr2k( order, uplo, transAB, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-2k update of a symmetric matrix with complex double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasZsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZsyr2k( order, uplo, transAB, N, K, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-2k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be less
 *                       than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be less
 *                       less than \b K if \b B matches to the op(\b B) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasSsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsyr2k( order, uplo, transAB, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssyr2k.c
 * This is an example of how to use the @ref clAmdBlasSsyr2kEx function.
 */

/**
 * @brief Rank-2k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasDsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsyr2k( order, uplo, transAB, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-2k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasCsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsyr2k( order, uplo, transAB, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Rank-2k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
__inline clAmdBlasStatus
clAmdBlasZsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZsyr2k( order, uplo, transAB, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup SYMM SYMM  - Symmetric matrix-matrix multiply
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with float
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYMM
 */
__inline clAmdBlasStatus
clAmdBlasSsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasSsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_ssymm.c
 * This is an example of how to use the @ref clAmdBlasSsymm function.
 */


/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with double
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsymm() function otherwise.
 *
 * @ingroup SYMM
 */
__inline clAmdBlasStatus
clAmdBlasDsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasDsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasSsymm() function.
 *
 * @ingroup SYMM
 */
__inline clAmdBlasStatus
clAmdBlasCsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDsymm() function.
 *
 * @ingroup SYMM
 */
__inline clAmdBlasStatus
clAmdBlasZsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup HEMM HEMM  - Hermitian matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMM
 */
__inline clAmdBlasStatus
clAmdBlasChemm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasChemm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_chemm.cpp
 * This is an example of how to use the @ref clAmdBlasChemm function.
 */


/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChemm() function otherwise.
 *
 * @ingroup HEMM
 */
__inline clAmdBlasStatus
clAmdBlasZhemm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZhemm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup HERK HERK  - Hermitian rank-k update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HERK
 */
__inline clAmdBlasStatus
clAmdBlasCherk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCherk( order, uplo, transA, N, K, alpha, A, offa, lda, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_cherk.cpp
 * This is an example of how to use the @ref clAmdBlasCherk function.
 */


/**
 * @brief Rank-k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCherk() function otherwise.
 *
 * @ingroup HERK
 */
__inline clAmdBlasStatus
clAmdBlasZherk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZherk( order, uplo, transA, N, K, alpha, A, offa, lda, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/


/**
 * @defgroup HER2K HER2K  - Hermitian rank-2k update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-2k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if \b B is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A , \b B or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HER2K
 */
__inline clAmdBlasStatus
clAmdBlasCher2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasCher2k( order, uplo, trans, N, K, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/**
 * @example example_cher2k.c
 * This is an example of how to use the @ref clAmdBlasCher2k function.
 */


/**
 * @brief Rank-2k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if B is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher2k() function otherwise.
 *
 * @ingroup HER2K
 */
__inline clAmdBlasStatus
clAmdBlasZher2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	return clblasZher2k( order, uplo, trans, N, K, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
			numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

/*@}*/




#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* CLAMDBLAS_H_ */
