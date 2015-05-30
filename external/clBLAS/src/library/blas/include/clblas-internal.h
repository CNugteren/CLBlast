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


#ifndef CLBLAS_INTERNAL_H_
#define CLBLAS_INTERNAL_H_

#include <defbool.h>
#include <blas_mempat.h>
#include <devinfo.h>
#include <trace_malloc.h>

#include "blas_funcs.h"
#include "kernel_extra.h"

#if defined(_MSC_VER)
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct SolutionStep;

typedef struct CLBlasSolvers {
    MemoryPattern memPatterns[MEMPAT_PER_BLASFN];
    unsigned int nrPatterns;
    int defaultPattern;         /*   -1 -- select among all available patterns
                                 * >= 0 -- index for memPatterns[]
                                 */
} CLBlasSolvers;

extern int clblasInitialized;

extern CLBlasSolvers clblasSolvers[BLAS_FUNCTIONS_NUMBER];

extern struct KernelCache *clblasKernelCache;

typedef union ArgMultiplier {
    cl_float argFloat;
    cl_double argDouble;
    FloatComplex argFloatComplex;
    DoubleComplex argDoubleComplex;
} ArgMultiplier;

typedef union LeadingDimention {
    size_t matrix;  /**< Positive ld value for matrixes */
    int vector;     /**< Integer offset value for vectors */
} LeadingDimention;

typedef enum reductionType {
    REDUCE_BY_SUM,
    REDUCE_BY_MAX,
    REDUCE_BY_MIN,
    REDUCE_MAX_WITH_INDEX,
    REDUCE_BY_HYPOT,
    REDUCE_BY_SSQ,
    REDUCE_MAX_WITH_INDEX_ATOMICS
} reductionType;

/**
 * @internal
 * @brief Kernel arguments for solver methods
 * @ingroup SUBMIT_PROBLEM
 */
typedef struct CLBlasKargs {
    BlasFunctionID pigFuncID; // FuncID piggy backing on this call. Used by Blas-3 routines to take advantage of GEMM code
    /** Kernel type to pass the arguments for */
    CLBlasKernelType kernType;
    DataType dtype;             /**< Data type */
    clblasOrder order;          /**< Row/column order */
    clblasSide side;            /**< Matrix A side */
    clblasUplo uplo;            /**< Matrix A is upper/lower */
    clblasTranspose transA;     /**< Operation to be applied to matrix A */
    clblasTranspose transB;     /**< Operation to be applied to matrix B */
    clblasDiag diag;            /**< Matrix A diagonality */
    size_t M;                   /**< Problem size in M dimension */
    size_t N;                   /**< Problem size in N dimension */
    size_t K;                   /**< Problem size in K dimension, or number of diagonals in a banded-matrix */
    ArgMultiplier alpha;        /**< Alpha multiplier */
    cl_mem A;                   /**< Matrix A data */
    LeadingDimention lda;       /**< Matrix A leading dimension */
    cl_mem B;                   /**< Matrix B data */
    LeadingDimention ldb;       /**< Matrix B or vector X leading dimension */
    ArgMultiplier beta;         /**< Beta multiplier */
    cl_mem C;                   /**< Matrix C data */
    LeadingDimention ldc;       /**< Matrix C or vector Y leading dimension */
    cl_mem D;                   /**< Extra cl_mem buffer. For scratch usage or other purpose */
    cl_mem E;                   /**< Extra buffer.. Needed for blas 1 functions */
    int addrBits;               /**< Number of device address bits */
    /** Problem start offset in M dimension to process from */
    size_t offsetM;
    /** Problem start offset in N dimension to process from */
    size_t offsetN;
    /** Problem start offset in K dimension to process from */
    size_t offsetK;
    cl_mem scimage[2];          /**< Scratch images */
    size_t offA;                /**< Offset of first element of matrix A */
    /** Offset of first element of matrix B or vector X */
    size_t offBX;
    /**< Offset of first element of matrix C or vector Y */
    size_t offCY;
	size_t offa;				/**< Offset of first element of Matrix A */
	size_t offb;				/**< Offset of first element of Matrix B */
	size_t offc;				/**< Offset of first element of Matrix C */
    size_t offd;                /**< Offset of first element of buffer D */
    size_t offe;                /**< Offset of first element of buffer E */
	cl_int startRow;				/**< Triangular Solver - Identify where the triangle starts */
	cl_int endRow;					/**< Triangular Solver - Identify where the triangle ends */
	size_t tailStartM;			// Tail Kernel for GEMM2
	size_t tailStartN;			// Tail Kernel for GEMM2
    size_t KL;                  // Number of sub-diagonals in a banded-matrix
    size_t KU;                  // Number of super-diagonals in a banded-matrix
    reductionType redctnType;   // To store kind of reduction for reduction-framewrok to handle -- enum
} CLBlasKargs;

static __inline bool
areKernelsCacheable(void)
{
    return (clblasKernelCache != NULL);
}

/*
 * Assign a scalar multiplied on a matrix as a kernel argument
 */
void
assignScalarKarg(KernelArg *arg, const void *value, DataType dtype);

/**
 * calculate amount of global threads needed to compute all the problem
 *
 * @wgDim: Subproblem dimension at the level where the previous level subproblem
 *        is distributed among different work groups
 * @M: problem size in dimension M before the distributing
 * @N: problem size in dimension N before the distributing
 */
void
calcGlobalThreads(
    size_t globalThreads[2],
    const SubproblemDim *wgDim,
    const PGranularity *pgran,
    size_t M,
    size_t N);

/**
 * @internal
 * @brief Get the context associated with kernel.
 *
 * @param[in] kernel Kernel object being queried.
 * @param[out] context The context.
 *
 * @return clGetKernelInfo() return code.
 */
cl_int
getKernelContext(
    cl_kernel kernel,
    cl_context *context);

/**
 * @brief Get the context associated with queue.
 *
 * @param[in] queue Queue being queried.
 * @param[out] context The context.
 *
 * @return clGetCommandQueueInfo() return code.
 */
cl_int
getQueueContext(
    cl_command_queue queue,
    cl_context *context);

/**
 * @internal
 * @brief Get the device specified when the command-queue is created.
 *
 * @param[in] queue Queue being queried.
 * @param[out] device The device.
 *
 * @return clGetCommandQueueInfo() return code.
 */
cl_int
getQueueDevice(
    cl_command_queue queue,
    cl_device_id *device);

/**
 * @internal
 * @brief Get the currently specified properties for the command-queue.
 *
 * @param[in] queue Queue being queried.
 * @param[out] props Properties.
 *
 * @return clGetCommandQueueInfo() return code.
 */
cl_int
getQueueProperties(
    cl_command_queue queue,
    cl_command_queue_properties *props);

Kernel
*makeKernel(
    cl_device_id device,
    cl_context context,
    SolverKgen kernelGenerator,
    const SubproblemDim *dims,
    const PGranularity *pgran,
    const CLBLASKernExtra *extra,
    const char *buildOpts,
    cl_int *error);

Kernel
*loadKernel( const unsigned char** buffer,
             size_t sizeBuffer,
             KernelKey *key,
             const CLBLASKernExtra *extra,
             cl_int *error);

/*
 * TODO: doxygen style comments
 */
void
setupBuildOpts(
    char opts[BUILD_OPTS_MAXLEN],
    cl_device_id devID,
    MemoryPattern *mempat);

void addBuildOpt(
    char * opts,
    size_t len,
    const char * option);

// Internal scatter image API

int
initSCImages(void);

void
releaseSCImages(void);

/**
 * Request an image appropriating the most to perform a user API request
 *
 * @ctx: context containing images
 * @devID: id of device the image will used for
 * @bestSize: size of image, i. e. minWidth*bestHeight of the image that should
 *            be enough to solve a problem in single step
 * @minSize: minimal size of image image, i. e. minWidth*minHeight
 * @minWidth: minimal image width
 *
 * Returns memory object of the most appropriate image. If there are
 * not images available for the device or not enough memory, to allocate
 * some internal structures to save a usage info the function returns NULL.
 */
cl_mem
getSCImage(
    cl_context ctx,
    cl_device_id devID,
    cl_ulong bestSize,
    cl_ulong minSize,
    size_t minWidth);

void
putSCImage(cl_device_id devID, cl_mem image);

char
*sprintfGranulation(char *buf, const SubproblemDim *dim, int level);

const char
*kernelTypeString(CLBlasKernelType ktype);

#ifdef DUMP_CLBLAS_KERNELS

void
dumpKernel(
    const struct SolutionStep *step,
    CLBlasKernelType ktype);

#else       /* DUMP_CLBLAS_KERNEL */

// stub, does nothing
#define dumpKernel(step, ktype)

#endif      /* !DUMP_CLBLAS_KERNEL */

static __inline solver_id_t
makeSolverID(int fid, int mpat)
{
    return (solver_id_t)(fid * MEMPAT_PER_BLASFN + mpat);
}

static __inline int
solverFunctionID(solver_id_t sid)
{
    return (sid / MEMPAT_PER_BLASFN);
}

static __inline int
solverPattern(solver_id_t sid)
{
    return (sid % MEMPAT_PER_BLASFN);
}

typedef enum ErrorCodeSet {
     A_MAT_ERRSET,
     B_MAT_ERRSET,
     C_MAT_ERRSET,
     X_VEC_ERRSET,
     Y_VEC_ERRSET,
     END_ERRSET
} ErrorCodeSet;

clblasStatus
checkMatrixSizes(
    DataType dtype,
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_mem A,
    size_t offA,
    size_t lda,
    ErrorCodeSet err );

clblasStatus
checkBandedMatrixSizes(
    DataType dtype,
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_mem A,
    size_t offA,
    size_t lda,
    ErrorCodeSet err );

clblasStatus
checkVectorSizes(
    DataType dtype,
    size_t N,
    cl_mem x,
    size_t offx,
    int incx,
    ErrorCodeSet err );

clblasStatus
checkMemObjects(
    cl_mem A,
    cl_mem B,
    cl_mem C,
    bool checkC,
    ErrorCodeSet errA,
    ErrorCodeSet errB,
    ErrorCodeSet errC );

/**
 * @brief Set preferred function internal implementation.
 *
 * Some BLAS functions are implemented in several different ways internally.
 * By default the library tries to select the most suitable implementation for
 * given problem. Using this function user can force library to use specific one.
 *
 * @return \b clblasSuccess on success, \b clblasInvalidValue if an
 * unknown implementation id was passed.
 */
clblasStatus
clblasSelectImplementation(
    clblasImplementation impl);

/**
 * @brief Set preferred implementation according to environment variable.
 */
void
parseEnvImplementation(void);

/**
 * @brief Check whether it is allowed to use scratch images
 */
int
scratchImagesEnabled(void);


#ifdef __cplusplus
}
#endif

#endif /* CLBLAS_INTERNAL_H_ */
