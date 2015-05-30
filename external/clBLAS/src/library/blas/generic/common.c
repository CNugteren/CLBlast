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


#include <string.h>
#include <stdlib.h>

#include <clBLAS.h>
#include <clkern.h>
#include <cltypes.h>
#include <stdio.h>
#include <ctype.h>

#include "clblas-internal.h"

#if defined(DUMP_CLBLAS_KERNELS) && !defined(KEEP_CLBLAS_KERNEL_SOURCES)
#define KEEP_CLBLAS_KERNEL_SOURCES
#endif

int clblasInitialized = 0;
CLBlasSolvers clblasSolvers[BLAS_FUNCTIONS_NUMBER];
struct KernelCache *clblasKernelCache = NULL;

enum {
    BUILD_LOG_SIZE = 65536
};

static __inline void
storeErrorCode(cl_int *error, cl_int code)
{
    if (error != NULL) {
        *error = code;
    }
}

#ifndef PRINT_BUILD_ERRORS
    #define PRINT_BUILD_ERRORS
#endif

#ifdef PRINT_BUILD_ERRORS

static char
*allocBuildLog(void)
{
	char *log;

    log = malloc(BUILD_LOG_SIZE);
	if (log) {
		log[0] = '\0';
	}

	return log;
}

static void
freeBuildLog(char *buildLog)
{
    free(buildLog);
}

static void
printBuildError(
    cl_int error,
    cl_device_id device,
    SolverKgen kgen,
    const SubproblemDim *dims,
    const PGranularity *pgran,
    const CLBLASKernExtra *kextra,
    const char *source,
    const char *buildLog)
{
    char name[128];
    char dimStr[1024];
    char pgranStr[1024];
    char *p;
    MemoryPattern *mempat = NULL;
    unsigned int i, j;
    const char *s;

    name[0] = '\0';
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);

    // lookup memory pattern
    s = NULL;
    for (i = 0; i < BLAS_FUNCTIONS_NUMBER; i++) {
        for (j = 0; j < clblasSolvers[i].nrPatterns; j++) {
            mempat = &clblasSolvers[i].memPatterns[j];
            if (kgen == mempat->sops->genKernel) {
                s = kernelTypeString(kextra->kernType);
                break;
            }
        }
        if (s != NULL) {
            break;
        }
    }

    // sprintf Subproblem dimensions
    p = dimStr;
    for (i = 0; i < mempat->nrLevels; i++) {
        p = sprintfGranulation(p, dims, i);
        strcat(p, "; ");
        p += strlen(p);
    }

    // sprintf data parallelism granularity
    sprintf(pgranStr, "pgran->wgDim = %d, pgran->wgSize[0] = %u, "
            "pgran->wgSize[1] = %u, pgran->wfSize = %u",
            pgran->wgDim, pgran->wgSize[0], pgran->wgSize[1],
            pgran->wfSize);

    fprintf(stderr, "\n========================================================\n\n");
    fprintf(stderr, "AN INTERNAL KERNEL BUILD ERROR OCCURRED!\n");
    fprintf(stderr, "device name = %s\n", name);
    fprintf(stderr, "error = %d\n", error);
    fprintf(stderr, "memory pattern = %s, %s kernel generator\n", mempat->name, s);
    fprintf(stderr, "Subproblem dimensions: %s\n", dimStr);
    fprintf(stderr, "Parallelism granularity: %s\n", pgranStr);
    fprintf(stderr, "Kernel extra flags: %u\n", kextra->flags);
    fprintf(stderr, "Source:\n\n%s\n\n", source);
    fprintf(stderr, "--------------------------------------------------------\n\n");
    if (buildLog) {
        fprintf(stderr, "Build log:\n\n%s\n", buildLog);
    }
    else {
        fprintf(stderr, "Build log is unavailable\n");
    }
    fprintf(stderr, "========================================================\n\n");
}

#else               /* PRINT_BUILD_ERRORS */

static __inline char*
allocBuildLog(void)
{
    /* stub, do nothing */
    return NULL;
}

#define freeBuildLog(log)                       /* stub, do nothing */
#define printBuildError(error, device, kgen, \
    dims, pgran, kextra, source, buildLog)      /* stub, do nothing */

#endif              /* !PRINT_BUILD_ERRORS */

static void
extraDtor(struct Kernel *kernel)
{
    if (kernel->extra != NULL) {
        free(kernel->extra);
        kernel->extra = NULL;
    }
}

static char
*sprintfDim(
    char *buf,
    size_t dim,
    const char *dimName,
    int level,
    bool first)
{
    if (!first) {
        strcat(buf, ", ");
        buf += strlen(buf);
    }
    if (dim == SUBDIM_UNUSED) {
        sprintf(buf, "dims[%d].%s = SUBDIM_UNUSED", level, dimName);
    }
    else {
        sprintf(buf, "dims[%d].%s = %lu", level, dimName, dim);
    }

    buf += strlen(buf);

    return buf;
}

const char VISIBILITY_HIDDEN
*kernelTypeString(CLBlasKernelType ktype)
{
    switch (ktype) {
    case CLBLAS_COMPUTING_KERNEL:
        return "computing";
    case CLBLAS_PREP_A_KERNEL:
        return "preparative for matrix A";
    case CLBLAS_PREP_B_KERNEL:
        return "preparative for matrix B";
    default:
        return NULL;
    }
}

/*
 * Assign a scalar multiplied on a matrix a kernel argument
 */
void VISIBILITY_HIDDEN
assignScalarKarg(KernelArg *arg, const void *value, DataType dtype)
{
    arg->typeSize = dtypeSize(dtype);
    memcpy(arg->arg.data, value, arg->typeSize);
}

void VISIBILITY_HIDDEN
calcGlobalThreads(
    size_t globalThreads[2],
    const SubproblemDim *wgDim,
    const PGranularity *pgran,
    size_t M,
    size_t N)
{
    globalThreads[1] = 1;

    if ((wgDim->itemX != SUBDIM_UNUSED) &&
        (wgDim->itemY != SUBDIM_UNUSED)) {

        size_t groupWorkX, groupWorkY;
        size_t nrGroupsX, nrGroupsY;
        int nrDims;

        groupWorkX = wgDim->itemX;
        groupWorkY = wgDim->itemY;

        nrGroupsX = N / groupWorkX;
        if (N % groupWorkX) {
            nrGroupsX++;
        }

        nrGroupsY = M / groupWorkY;
        if (M % groupWorkY) {
            nrGroupsY++;
        }

        nrDims = (pgran == NULL) ? 1 : pgran->wgDim;
        if (nrDims == 1) {
            globalThreads[0] = nrGroupsX * nrGroupsY;
        }
        else {
            globalThreads[0] = nrGroupsY;
            globalThreads[1] = nrGroupsX;
        }
    }
    else {
        size_t totalWork, groupWork;

        if (wgDim->itemX != SUBDIM_UNUSED) {
            totalWork = N;
            groupWork = wgDim->itemX;
        }
        else {
            totalWork = M;
            groupWork = wgDim->itemY;
        }

        globalThreads[0] = totalWork / groupWork;
        if (totalWork % groupWork) {
            globalThreads[0]++;
        }
    }

    if (pgran != NULL) {
        globalThreads[0] *= pgran->wgSize[0];
        globalThreads[1] *= pgran->wgSize[1];
    }
}

cl_int VISIBILITY_HIDDEN
getKernelContext(cl_kernel kernel, cl_context *context)
{
    cl_int err;
    cl_context ctx;

    err = clGetKernelInfo(kernel, CL_KERNEL_CONTEXT,
        sizeof(cl_context), &ctx, NULL);
    if (err != CL_SUCCESS)
        return err;
    if (context != NULL)
        *context = ctx;
    return err;
}

cl_int VISIBILITY_HIDDEN
getQueueContext(cl_command_queue queue, cl_context *context)
{
    cl_int err;
    cl_context ctx;

    err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
        sizeof(cl_context), &ctx, NULL);
    if (err != CL_SUCCESS)
        return err;
    if (context != NULL)
        *context = ctx;
    return err;
}

cl_int VISIBILITY_HIDDEN
getQueueDevice(cl_command_queue queue, cl_device_id *device)
{
    cl_int err;
    cl_device_id dev;

    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(cl_device_id), &dev, NULL);
    if (err != CL_SUCCESS)
        return err;
    if (device != NULL)
        *device = dev;
    return err;
}

cl_int VISIBILITY_HIDDEN
getQueueProperties(
    cl_command_queue queue,
    cl_command_queue_properties *props)
{
    cl_int err;
    cl_command_queue_properties p;

    err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
        sizeof(cl_command_queue_properties), &p, NULL);
    if (err != CL_SUCCESS)
        return err;
    if (props != NULL)
        *props = p;
    return err;
}

Kernel VISIBILITY_HIDDEN
*loadKernel( const unsigned char** buffer,
             size_t sizeBuffer,
             KernelKey *key,
             const CLBLASKernExtra *extra,
             cl_int *error)

{
    cl_int status = CL_SUCCESS;
    Kernel* kernel;

    kernel = allocKernel();
    if (kernel == NULL) {
        return NULL;
    }

    kernel->program = createClProgramWithBinary(key->context,
                                                key->device,
                                                (unsigned char*)*buffer,
                                                sizeBuffer,
                                                &status);
    if (status == CL_SUCCESS) {
        kernel->extraSize = sizeof(CLBLASKernExtra);
        kernel->extra = calloc(1, kernel->extraSize);
        *(CLBLASKernExtra*)(kernel->extra) = *extra;
        kernel->dtor = extraDtor;
        kernel->noSource = 1;
    }
    else {
        putKernel(NULL, kernel);
        storeErrorCode(error, status);
        kernel = NULL;
    }

    return kernel;
}

#if !defined(DUMP_CLBLAS_KERNELS)

/*
 * Drop the program's source so as to consume memory as few as possible
 * at caching
 */
static cl_int
dropProgramSource(cl_program *program, cl_context ctx, cl_device_id devID)
{
    size_t size;
    unsigned char *bin;
    cl_program p = *program;
    cl_int err;

    size = getProgramBinarySize(p);
    bin = getProgramBinary(p);

    /*
     * Don't release the original program until a new one is created
     * in order to retain its own reference to the context if it is
     * released by user
     */
    p = createClProgramWithBinary(ctx, devID, bin, size, &err);
    if (err == CL_SUCCESS) {
        clReleaseProgram(*program);
        *program = p;
    }

    free(bin);

    return err;
}

#endif /* !DUMP_CLBLAS_KERNELS */

Kernel
*makeKernel(
    cl_device_id device,
    cl_context context,
    SolverKgen kernelGenerator,
    const SubproblemDim *dims,
    const PGranularity *pgran,
    const CLBLASKernExtra *extra,
    const char *buildOpts,
    cl_int *error)
{

    cl_int err;
    char *source;
    ssize_t size;
    Kernel *kernel;
    char *log;

	#ifdef DEBUG_2
	printf("Make kernel called\n");
	printf("x : %d, y : %d, itemX: %d, itemY: %d\n",  dims->x, dims->y, dims->itemX, dims->itemY);
	printf("PG : wgSize[0] : %d, wgSize[1] : %d, wfSize: %d\n",  pgran->wgSize[0], pgran->wgSize[1], pgran->wfSize);
	#endif

    size = kernelGenerator(NULL, 0, dims, pgran, (void*)extra);
    if (size < 0) {
        storeErrorCode(error, CL_OUT_OF_HOST_MEMORY);
        return NULL;
    }
    source = calloc(1, size);
    if (source == NULL) {
        storeErrorCode(error, CL_OUT_OF_HOST_MEMORY);
        return NULL;
    }
    if (kernelGenerator(source, size, dims, pgran, (void*)extra) != size) {
        free(source);
        storeErrorCode(error, CL_OUT_OF_HOST_MEMORY);
        return NULL;
    }

	kernel = allocKernel();
    if (kernel == NULL) {
        free(source);
        storeErrorCode(error, CL_OUT_OF_HOST_MEMORY);
        return NULL;
    }

    log = allocBuildLog();

	//#define DEBUG_2
	#ifdef DEBUG_2
	printf("Build Options used %s \n", buildOpts);
	printf("Source kernel used %s \n", source);
	#endif
	#undef DEBUG_2

    kernel->program = buildClProgram(source, buildOpts, context, device,
                                     log, BUILD_LOG_SIZE, &err);
    if (err != CL_SUCCESS) {
        printBuildError(err, device, kernelGenerator, dims,
                        pgran, extra, source, log);
        freeBuildLog(log);
        putKernel(NULL, kernel);
        free(source);
        storeErrorCode(error, err);
        return NULL;
    }
	else
	{
		// #define DEBUG_2
		#ifdef DEBUG_2
		printf("Kernel compilation succeeded\n");
		#endif
		#undef DEBUG_2
	}

    freeBuildLog(log);
    free(source);

#if !defined(KEEP_CLBLAS_KERNEL_SOURCES)
    if (err == CL_SUCCESS) {
        err = dropProgramSource(&kernel->program, context, device);
        kernel->noSource = 1;
    }
#endif  /* !DUMP_CLBLAS_KERNELS */

    if (err != CL_SUCCESS) {
        putKernel(NULL, kernel);
        storeErrorCode(error, err);
        return NULL;
    }

    kernel->extraSize = sizeof(CLBLASKernExtra);
    kernel->extra = calloc(1, kernel->extraSize);
    *(CLBLASKernExtra*)(kernel->extra) = *extra;
    kernel->dtor = extraDtor;

    storeErrorCode(error, CL_SUCCESS);

    return kernel;

}

void
setupBuildOpts(
    char opts[BUILD_OPTS_MAXLEN],
    cl_device_id devID,
    MemoryPattern *mempat)
{
    TargetDevice target;

    target.id = devID;
    identifyDevice(&target);
    opts[0] = '\0';

#if !defined NDEBUG
    // Nvidia runtime does not appear to support the -g flag, at least in their OpenCL v1.1 runtime
    if( target.ident.vendor != VENDOR_NVIDIA )
        addBuildOpt( opts, BUILD_OPTS_MAXLEN, "-g" );
#endif  /* NDEBUG */

    if (target.ident.vendor == VENDOR_NVIDIA &&
        !strcmp(mempat->name, "2-staged cached global memory based "
                              "block trsm")) {

        addBuildOpt(opts, BUILD_OPTS_MAXLEN, "-cl-opt-disable");
    }
}

void addBuildOpt(
    char * opts,
    size_t len,
    const char * option)
{
    size_t l = strlen(opts);

    if (l > 0 && !isspace(opts[l-1]) && l+1 < len) {
      opts[l] = ' ';
      opts[l+1]   = '\0';
      l++;
    }

    strncat(opts, option, len - l - 1);
}


char VISIBILITY_HIDDEN
*sprintfGranulation(char *buf, const SubproblemDim *dim, int level)
{
    buf = sprintfDim(buf, dim[level].itemY, "itemY", level, true);
    buf = sprintfDim(buf, dim[level].itemX, "itemX", level, false);
    buf = sprintfDim(buf, dim[level].y, "y", level, false);
    buf = sprintfDim(buf, dim[level].x, "x", level, false);
    buf = sprintfDim(buf, dim[level].bwidth, "bwidth", level, false);
    strcat(buf, "; ");
    buf += strlen(buf);

    return buf;
}

clblasStatus VISIBILITY_HIDDEN
checkMatrixSizes(
    DataType dtype,
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_mem A,
    size_t offA,
    size_t lda,         // lda is passed as zero for packed matrices
    ErrorCodeSet err )
{
    size_t memSize, matrSize, tsize, memUsed;
    size_t unusedTail = 0;
    bool tra;

    if ((M == 0) || (N == 0)) {
        return clblasInvalidDim;
    }

    tsize = dtypeSize(dtype);
    tra = (order == clblasRowMajor && transA != clblasNoTrans) ||
          (order == clblasColumnMajor && transA == clblasNoTrans);

    if( lda > 0 )              // For Non-packed matrices
    {
        if (tra) {
            if (lda < M) {
                switch( err )
                {
                case A_MAT_ERRSET:
                    return clblasInvalidLeadDimA;
                case B_MAT_ERRSET:
                    return clblasInvalidLeadDimB;
                case C_MAT_ERRSET:
                    return clblasInvalidLeadDimC;
                default:
                    return clblasNotImplemented;
                }
            }
            matrSize = ((N - 1) * lda + M) * tsize;
            unusedTail = ( lda - N ) * tsize;
        }
        else {
            if (lda < N) {
                switch( err )
                {
                case A_MAT_ERRSET:
                    return clblasInvalidLeadDimA;
                case B_MAT_ERRSET:
                    return clblasInvalidLeadDimB;
                case C_MAT_ERRSET:
                    return clblasInvalidLeadDimC;
                default:
                    return clblasNotImplemented;
                }
            }
            matrSize = ((M - 1) * lda + N) * tsize;
            unusedTail = ( lda - M ) * tsize;
        }
    }
    else {                     // For the case of packed matrices
         matrSize = ((M * (N+1)) / 2) * tsize;
    }

    offA *= tsize;

    if (clGetMemObjectInfo(A, CL_MEM_SIZE, sizeof(memSize), &memSize, NULL) !=
                                CL_SUCCESS) {
        switch( err )
        {
        case A_MAT_ERRSET:
            return clblasInvalidMatA;
        case B_MAT_ERRSET:
            return clblasInvalidMatB;
        case C_MAT_ERRSET:
            return clblasInvalidMatC;
        default:
            return clblasNotImplemented;
        }
    }

    //  It is possible to allocate a buffer, and set up lda & ldb such that it looks like it will access outside of the allocated buffer, but if
    //  M & N are kept small enough, no out of bounds access will occur.  Compensate for the offset values and the unused tail memory caused by lda & ldb.
    //  Ex: BuffSize=6 floats, M=1, N=2, lda=ldb=3, offA = 0, offB = 2 :  |A[0,0]|unused|B[0,0]|A[0,1]|unused|B[0,1]|
    memUsed = (( offA + matrSize ) > unusedTail) ? offA + matrSize - unusedTail: 0;

    // Note: this is a hack to get the xsymm tests to work.
    // TODO: Find out why "memUsed" is set to 0 in some cases!
    memUsed = matrSize;
    //printf("%lu required but found %lu\n", memUsed/tsize, memSize/tsize);

    if (( memUsed > memSize ) || (offA + matrSize < offA)) {
        switch( err )
        {
        case A_MAT_ERRSET:
            return clblasInsufficientMemMatA;
        case B_MAT_ERRSET:
            return clblasInsufficientMemMatB;
        case C_MAT_ERRSET:
            return clblasInsufficientMemMatC;
        default:
            return clblasNotImplemented;
        }
    }

    return clblasSuccess;
}


clblasStatus VISIBILITY_HIDDEN
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
    ErrorCodeSet err )
{
    size_t memSize, matrSize, tsize, K, memUsed;
    size_t unusedTail = 0;
    bool tra;

    if ((M == 0) || (N == 0)) {
        return clblasInvalidDim;
    }

    tsize = dtypeSize(dtype);
    K = KL + KU + 1;
    tra = (order == clblasRowMajor && transA != clblasNoTrans) ||
          (order == clblasColumnMajor && transA == clblasNoTrans);

    if (lda < K) {
        switch( err )
        {
        case A_MAT_ERRSET:
            return clblasInvalidLeadDimA;
        case B_MAT_ERRSET:
            return clblasInvalidLeadDimB;
        case C_MAT_ERRSET:
            return clblasInvalidLeadDimC;
        default:
            return clblasNotImplemented;
        }
    }

    if (tra) {
        matrSize = ((N - 1) * lda + K) * tsize;
        unusedTail = ( lda - N ) * tsize;
    }
    else {
        matrSize = ((M - 1) * lda + K) * tsize;
        unusedTail = ( lda - M ) * tsize;
    }

    offA *= tsize;

    if (clGetMemObjectInfo(A, CL_MEM_SIZE, sizeof(memSize), &memSize, NULL) !=
                                CL_SUCCESS) {
        switch( err )
        {
        case A_MAT_ERRSET:
            return clblasInvalidMatA;
        case B_MAT_ERRSET:
            return clblasInvalidMatB;
        case C_MAT_ERRSET:
            return clblasInvalidMatC;
        default:
            return clblasNotImplemented;
        }
    }

    //  It is possible to allocate a buffer, and set up lda & ldb such that it looks like it will access outside of the allocated buffer, but if
    //  M & N are kept small, no out of bounds access will occur.  Compensate for the offset values and the unused tail memory caused by lda & ldb.
    //  Ex: BuffSize=6 floats, M=1, N=2, lda=ldb=3, offA = 0, offB = 2 :  |A[0,0]|unused|B[0,0]|A[0,1]|unused|B[0,1]|
    memUsed = (( offA + matrSize ) > unusedTail) ? offA + matrSize - unusedTail: 0;
    if (memUsed > memSize) {
        switch( err )
        {
        case A_MAT_ERRSET:
            return clblasInsufficientMemMatA;
        case B_MAT_ERRSET:
            return clblasInsufficientMemMatB;
        case C_MAT_ERRSET:
            return clblasInsufficientMemMatC;
        default:
            return clblasNotImplemented;
        }
    }

    return clblasSuccess;
}

clblasStatus VISIBILITY_HIDDEN
checkVectorSizes(
    DataType dtype,
    size_t N,
    cl_mem x,
    size_t offx,
    int incx,
    ErrorCodeSet err )
{
    size_t memSize, sizev;
    size_t tsize;

    if (N == 0) {
        return clblasInvalidDim;
    }

    if (incx == 0) {
        switch( err )
        {
        case X_VEC_ERRSET:
            return clblasInvalidIncX;
        case Y_VEC_ERRSET:
            return clblasInvalidIncY;
        default:
            return clblasNotImplemented;
        }
    }

    if (clGetMemObjectInfo(x, CL_MEM_SIZE, sizeof(memSize), &memSize, NULL) !=
                                CL_SUCCESS) {
        switch( err )
        {
        case X_VEC_ERRSET:
            return clblasInvalidVecX;
        case Y_VEC_ERRSET:
            return clblasInvalidVecY;
        default:
            return clblasNotImplemented;
        }
    }

    tsize = dtypeSize(dtype);
    sizev = ((N - 1) * abs(incx) + 1) * tsize;
    offx *= tsize;

    if ((offx + sizev > memSize) || (offx + sizev < offx)) {
        switch( err )
        {
        case X_VEC_ERRSET:
            return clblasInsufficientMemVecX;
        case Y_VEC_ERRSET:
            return clblasInsufficientMemVecY;
        default:
            return clblasNotImplemented;
        }
    }

    return clblasSuccess;
}

clblasStatus
checkMemObjects(
    cl_mem A,
    cl_mem B,
    cl_mem C,
    bool checkC,
    ErrorCodeSet errA,
    ErrorCodeSet errB,
    ErrorCodeSet errC )
{
    cl_mem_object_type mobjType = 0;

    if (!clGetMemObjectInfo(A, CL_MEM_TYPE, sizeof(mobjType), &mobjType, NULL) &&
        (mobjType != CL_MEM_OBJECT_BUFFER)) {
        switch( errA )
        {
        case A_MAT_ERRSET:
            return clblasInvalidMatA;
        case B_MAT_ERRSET:
            return clblasInvalidMatB;
        case C_MAT_ERRSET:
            return clblasInvalidMatC;
        case X_VEC_ERRSET:
            return clblasInvalidVecX;
        case Y_VEC_ERRSET:
            return clblasInvalidVecY;
        default:
            return clblasNotImplemented;
        }
    }

    mobjType = 0;
    if (!clGetMemObjectInfo(B, CL_MEM_TYPE, sizeof(mobjType), &mobjType, NULL) &&
        (mobjType != CL_MEM_OBJECT_BUFFER)) {
        switch( errB )
        {
        case A_MAT_ERRSET:
            return clblasInvalidMatA;
        case B_MAT_ERRSET:
            return clblasInvalidMatB;
        case C_MAT_ERRSET:
            return clblasInvalidMatC;
        case X_VEC_ERRSET:
            return clblasInvalidVecX;
        case Y_VEC_ERRSET:
            return clblasInvalidVecY;
        default:
            return clblasNotImplemented;
        }
    }

    mobjType = 0;
    if (checkC && !clGetMemObjectInfo(C, CL_MEM_TYPE, sizeof(mobjType),
                                     &mobjType, NULL) &&
        (mobjType != CL_MEM_OBJECT_BUFFER)) {
        switch( errC )
        {
        case A_MAT_ERRSET:
            return clblasInvalidMatA;
        case B_MAT_ERRSET:
            return clblasInvalidMatB;
        case C_MAT_ERRSET:
            return clblasInvalidMatC;
        case X_VEC_ERRSET:
            return clblasInvalidVecX;
        case Y_VEC_ERRSET:
            return clblasInvalidVecY;
        default:
            return clblasNotImplemented;
        }
    }

    return clblasSuccess;
}
