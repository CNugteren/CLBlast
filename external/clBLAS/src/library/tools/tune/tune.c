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


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// #include "fileio.h"
#include "toolslib.h"
#include "tune.h"
#include "devinfo.h"
#include "assert.h"
#include "solution_seq.h"
#include "matrix_dims.h"

#include "subdim.h"

#if defined(_MSC_VER)
#include "Windows.h"
#elif defined(__APPLE__)
#include <stdint.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#else
#include "time.h"
#endif

#define EXIT_COD_OK                         0x0000
#define EXIT_COD_CL_ERROR                   0x0100
#define EXIT_COD_UNKNOWN_DATATYPE           0x0101
#define EXIT_COD_NO_DATA                    0x0102
#define EXIT_COD_NO_ENVIRONMENT_VARIABLE    0x0103
#define EXIT_COD_BAD_ENVIRONMENT_VARIABLE   0x0104

#define TYPE_NUMBER 4
#define MAX_RUN_KERNEL 3

typedef  int KMASK;

//////////////////////////////////////////////////////////////////
#if defined(_MSC_VER)

typedef unsigned long long nano_time_t;
#define NANOTIME_MAX (~0ULL - 1)

#define fmin min
#define fmax max

nano_time_t
conv2nanosec(nano_time_t t)
{
    LARGE_INTEGER count;

    if (QueryPerformanceFrequency(&count) == FALSE) {
        return 0;
    }
    t = (t * 1000000)/count.QuadPart;

    return (nano_time_t)(t * 1000);
}

nano_time_t
getCurrentTime(void)
{
     LARGE_INTEGER count;

     if (QueryPerformanceCounter(&count) == FALSE) {
         return 0;
     }
     return (nano_time_t)count.QuadPart;
}

#elif defined(__APPLE__)

typedef uint64_t nano_time_t;
#define NANOTIME_MAX UINT64_MAX

nano_time_t
conv2nanosec(nano_time_t t)
{
  static mach_timebase_info_data_t timebase_info = {0};

    if (timebase_info.denom == 0)
    {
        (void)mach_timebase_info(&timebase_info);
    }

    /* Let's hope we don't overflow */
    return (t * timebase_info.denom) / timebase_info.numer;
}

nano_time_t
getCurrentTime(void)
{
    return mach_absolute_time();
}

#else

typedef unsigned long nano_time_t;
#define NANOTIME_MAX (~0UL - 1)

nano_time_t
conv2nanosec(nano_time_t t)
{
    /* clock_... functions measure time in nanoseconds */
    return t;
}

nano_time_t
getCurrentTime(void)
{
    int err;
    struct timespec t;

    err = clock_gettime(CLOCK_REALTIME, &t);
    if (err == 0) {
        return (t.tv_sec * 1000000000UL + t.tv_nsec);
    }
    return 0;
}

#endif  /* defined(_MCS_VER) */
//////////////////////////////////////////////////////////////////

cl_int
waitForSuccessfulFinish(
    cl_command_queue commandQueues,
    cl_event *event)
{
    cl_int err, status;

    err = clFinish(commandQueues);
    if (err != CL_SUCCESS) {
        return err;
    }

    if (event == NULL || *event == NULL) {
        return CL_SUCCESS;
    }

    status = CL_COMPLETE;
    err = clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(status), &status, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }
    if (status < 0) {
        return -status;
    }
    return CL_SUCCESS;
}

cl_int
flushAll(cl_command_queue commandQueue)
{
    cl_int err;

    err = clFlush(commandQueue);
    if (err != CL_SUCCESS) {
        return err;
    }
    return CL_SUCCESS;
}


enum {
    MASK_KERNEL_COMP = 0x01,
    MASK_KERNEL_A    = 0x02,
    MASK_KERNEL_B    = 0x04
};

const char *FILE_PATH = NULL;

FILE* logStream;

int globalDim = 0;
enum {
    DEVNAME_MAXLEN = 64
};

#ifdef TEST_LOG
#include <time.h>
typedef unsigned long nano_time_t;
nano_time_t
getCurrentTime(void)
{
    int err;
    struct timespec t;


    err = clock_gettime(CLOCK_REALTIME, &t);
    if (err == 0) {
        return (t.tv_sec * 1000000000UL + t.tv_nsec);
    }
    return 0;
}
double globalTime = 0;
double globalFastTime = 0;

#endif

extern int getDataTypeSize(DataType dataType);
extern void writeStorageCache(TargetDevice* devID);
extern BlasFunctionInfo* getBlasFunctionInfo(TargetDevice* devID, int func);
extern void checkFILE(TargetDevice* devID, BlasFunctionInfo* fiArr);
extern char* getDevName(TargetDevice* tdev);

const unsigned int uiNONE = (unsigned int)-1;

// float types based unified pointer
typedef union FPtr {
    void *v;
    cl_float *f;
    cl_double *d;
    cl_float2 *f2;
    cl_double2 *d2;
} FPtr;



typedef struct GParam {

    int count;
    char            name[65];
    SubproblemDim   dims[MAX_SUBDIMS];
    PGranularity    pgran;
    unsigned int    vecLen;

    cl_ulong        time;
    // For each kernel the binaries are created
    Kernel*         kernel;
    //cl_kernel       clkern;
    size_t          binary_sizes;
    char*           binaries;

    Kernel          *kernelPrepA;
    Kernel          *kernelPrepB;

    size_t          binary_sizesA;
    char*           binariesA;

    size_t          binary_sizesB;
    char*           binariesB;

//    POSFILE         fbin[MAX_CLBLAS_KERNELS_PER_STEP];
} GParam;

typedef struct MatrixInfo {
    DataType        dtype;
    unsigned int    sizeDType;

    unsigned int M;
    unsigned int N;
    unsigned int K;

    cl_mem clA;
    cl_mem clB;
    cl_mem clC;

    FPtr A;
    FPtr B;
    FPtr C;

    cl_mem clImgA;
    cl_mem clImgB;
    void *imgA;
    void *imgB;
}MatrixInfo;

enum Command {
    C_DEFAULT,
    C_REBUILD,
    C_GENKERNEL,
    C_ADD,
};

struct GeneratorInfoRec {

    cl_platform_id      platform;       // ID of platform
    cl_device_type      devType;
    cl_context          ctx;
    cl_command_queue    queue;
    //
    unsigned int        numDevices;     // number of Devices

    TargetDevice 		targetDevice;	//
    DeviceInfo          deviceInfos;    // Todo delete this member. Use TargetDevice.
    char                *deviceName;    //

    bool       aFunc[BLAS_FUNCTIONS_NUMBER];    //  True/false value if the corresponding function should be tuned
    int        aPattern;
    bool       aDType[TYPE_NUMBER]; //  True false value if the precision should be tuned; s/d/c/z
    int        aFlag;
    int        aCommand;
    bool       aIsKernel;   // True/false value to store binary kernels into the kernel database
    int        aMaxparam;
    bool       aExtendedOutput;
    bool       aAll;

    double next;
    double last;
    const char* patternName;
} genInfo;

char *
genParamStr(char* name, int w, size_t data)
{
    char format[5];

    sprintf(format,"%%%uu ", w);
    if (data != (size_t)-1) {
        char format[5];

        sprintf(format,"%%%uu ", w);
        sprintf(name, format, (unsigned)data);
    }
    else {
        char format[5];

        sprintf(format,"%%%us ", w);
        sprintf(name, format, "SU");
    }
    return name + w + 1;
}

char *
genParamsStr(SubproblemDim* dim, char* name, int w)
{
    char* n = name;

        n = genParamStr(n, w, dim->x);
        n = genParamStr(n, w, dim->y);
        n = genParamStr(n, w, dim->bwidth);
    sprintf(n,":");
    return n + 1;
}

void
createGParamName(GParam* param)
{
    char* n = param->name;

    if (param->dims[2].itemX > 0) {
        n = genParamsStr(&param->dims[0], n, 3);
        n = genParamsStr(&param->dims[1], n, 3);
        n = genParamsStr(&param->dims[2], n, 2);
    }
    else {
        n = genParamsStr(&param->dims[0], n, 3);
        n = genParamsStr(&param->dims[1], n, 2);
    }

    sprintf(n,"%3dx%-2d", param->pgran.wgSize[0],
                          param->pgran.wgSize[1]);
}

static int
patternUseImages(MemoryPattern *pattern)
{
    const CLBLASMpatExtra *extra = (const CLBLASMpatExtra*)pattern->extra;

    if (extra == NULL) {
        return 0;
    }
    if ((extra->mobjA == CLMEM_IMAGE)  ||
        (extra->mobjB == CLMEM_IMAGE) ) {
            return 1;
    }
    return 0;
}


void
initGeneratorInfoRec(void)
{
    int i;

    memset(&genInfo, 0, sizeof(struct GeneratorInfoRec));
    genInfo.devType = CL_DEVICE_TYPE_GPU;

    genInfo.aCommand = C_DEFAULT;
    for (i=0; i < TYPE_NUMBER; ++i) {
        genInfo.aDType[i] = false;
    }
    genInfo.aFlag    = -1;
    for (i=0; i < BLAS_FUNCTIONS_NUMBER; ++i) {
        genInfo.aFunc[i] = false;
    }
    genInfo.aPattern = -1;
    genInfo.aIsKernel = false;
    genInfo.aMaxparam = 5000;

    genInfo.aExtendedOutput = false;
}

void
destroyKernels(GParam *param)
{
    if (param->kernel != NULL) {
        putKernel(NULL, param->kernel);
        param->kernel = NULL;
    }
    if (param->kernelPrepA != NULL) {
        putKernel(NULL, param->kernelPrepA);
        param->kernelPrepA = NULL;
    }
    if (param->kernelPrepB != NULL) {
        putKernel(NULL, param->kernelPrepB);
        param->kernelPrepB = NULL;
    }
}

void
destroyGenInfo(void)
{
    free (genInfo.deviceName);
    genInfo.deviceName = NULL;
    clReleaseCommandQueue(genInfo.queue);
    clReleaseContext(genInfo.ctx);
//    destroyData(genInfo.functionInfo);
}

void
checkErrorFunc(char* funcName, cl_int status)
{
    if (status != CL_SUCCESS) {
        char * ret = "UNKNOWN";

        switch (status) {
        case CL_OUT_OF_RESOURCES:
            ret = "CL_OUT_OF_RESOURCES";    // -5
            break;
        case CL_BUILD_PROGRAM_FAILURE:      // -11
            ret = "CL_BUILD_PROGRAM_FAILURE";
            break;
        case CL_INVALID_VALUE:              // - 30
            ret = "CL_INVALID_VALUE";
            break;
        case CL_INVALID_KERNEL_ARGS:        // - 52
            ret = "CL_INVALID_KERNEL_ARGS";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:    // - 54
            ret = "CL_INVALID_WORK_GROUP_SIZE";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:     // - 55
            ret = "CL_INVALID_WORK_ITEM_SIZE";
            break;
       case CL_INVALID_BUFFER_SIZE:         // - 61
            ret = "CL_INVALID_BUFFER_SIZE";
            break;

        }

        fprintf(logStream, "%s() failed with %d(%s)\n", funcName, status, ret);
        fflush(logStream);
        destroyGenInfo();
        exit(EXIT_COD_CL_ERROR);
    }
}

void
initOpenCl(void)
{
    cl_int status = 0;
    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErrorFunc("clGetPlatformIDs", status);

    if (numPlatforms > 0) {
        unsigned int i;
        cl_platform_id* platforms =
                (cl_platform_id *)malloc(numPlatforms*sizeof(cl_platform_id));

        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        checkErrorFunc("clGetPlatformIDs", status);

        for(i=0; i < numPlatforms; ++i) {
            char pbuff[100];
            status = clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR,
                    sizeof(pbuff), pbuff, NULL);

            checkErrorFunc("clGetPlatformInfo", status);
            genInfo.platform = platforms[i];
            if(!strcmp(pbuff, "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        free(platforms);
    }

    // Init Device count
    status = clGetDeviceIDs(genInfo.platform, genInfo.devType, 0, 0,
            (cl_uint*)&genInfo.numDevices);
    checkErrorFunc("clGetDeviceIDs", status);
}

void
initDevice(int dev)
{
    cl_int status = 0;
    cl_uint num_devices;
    cl_device_id* deviceIDs =
           (cl_device_id *)calloc(genInfo.numDevices, sizeof(cl_device_id));

    status = clGetDeviceIDs(genInfo.platform, genInfo.devType,
        genInfo.numDevices,  deviceIDs, &num_devices);
    checkErrorFunc("clGetDeviceIDs", status);

    genInfo.targetDevice.id = deviceIDs[dev];
    identifyDevice(&genInfo.targetDevice);
    genInfo.deviceName = getDevName(&genInfo.targetDevice);
    initCLDeviceInfoRec(&genInfo.targetDevice, &genInfo.deviceInfos);
}

void
getContext(void)
{
    cl_int status = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_device_id device = genInfo.targetDevice.id;

    props[1] = (cl_context_properties)genInfo.platform;

    genInfo.ctx = clCreateContext(props, 1, &device,
        NULL, NULL, &status);
    checkErrorFunc("clCreateContext", status);

    genInfo.queue = clCreateCommandQueue(genInfo.ctx,
    	device,
        CL_QUEUE_PROFILING_ENABLE,
        &status);
    checkErrorFunc("clCreateCommandQueue",status);
}

int
bitcount (unsigned int n)  {
    int count = 1 ;

    while (n)  {
        count  *= 2;
        n &= (n - 1) ;
    }
    return count ;
}

bool
genKernel(GParam *param, CLBLASKernExtra* extra, MemoryPattern *pattern)
{
    cl_int status;
    SolverKgen genKernel;
    bool ret = false;
    cl_device_id device;
    char bopts[BUILD_OPTS_MAXLEN];

    genKernel = pattern->sops->genKernel;
    device = genInfo.targetDevice.id;

    setupBuildOpts(bopts, device, pattern);
    param->kernel = makeKernel(device, genInfo.ctx, genKernel,
                               param->dims, &param->pgran, extra, bopts, NULL);
    if (param->kernel != NULL) {
        status = clGetProgramInfo(param->kernel->program, CL_PROGRAM_BINARY_SIZES,
                                  sizeof(size_t), &param->binary_sizes, NULL);

        checkErrorFunc("clGetProgramInfo", status);
        param->binaries = (char *)malloc(sizeof(char)*param->binary_sizes);
        status = clGetProgramInfo(param->kernel->program,
                CL_PROGRAM_BINARIES,
                sizeof(char *),
                &param->binaries,
                NULL);
        checkErrorFunc("clGetProgramInfo", status);
        ret = true;
    }

    return ret;
}

void
convKExtraFlagToArg(KernelExtraFlags flags, CLBlasKargs* args)
{
    args->order = (flags & KEXTRA_COLUMN_MAJOR)?clblasColumnMajor: clblasRowMajor;
    args->side  = (flags & KEXTRA_SIDE_RIGHT)? clblasRight: clblasLeft;
    args->uplo  = (flags & KEXTRA_UPPER_TRIANG)?clblasUpper: clblasLower;

    args->transA = (flags & KEXTRA_TRANS_A)? clblasTrans: clblasNoTrans;
    args->transB = (flags & KEXTRA_TRANS_B)? clblasTrans: clblasNoTrans;
    if (isComplexType(args->dtype))
    {
        args->transA = (flags & KEXTRA_CONJUGATE_A)?clblasConjTrans: args->transA;
        args->transB = (flags & KEXTRA_CONJUGATE_B)?clblasConjTrans: args->transB;
    }
    args->diag = (flags & KEXTRA_UNIT_DIAGONAL)? clblasUnit: clblasNonUnit;

}

void
initCLBlasKArgDim(CLBlasKargs *args, MatrixInfo* mi, KernelExtraFlags extra)
{
    cl_int status;
    float beta = ((extra & KEXTRA_BETA_ZERO) != 0)? 0.0f : 1.0f;

    memset( args, 0, sizeof(CLBlasKargs) );
    convKExtraFlagToArg( extra, args );
    args->dtype = mi->dtype;

    switch (mi->dtype)
    {
    case TYPE_FLOAT:
        args->alpha.argFloat = 1.0;
        args->beta.argFloat = beta;
        break;
    case TYPE_DOUBLE:
        args->alpha.argDouble = 1.0;
        args->beta.argFloat = beta;
        break;
    case TYPE_COMPLEX_FLOAT:
        args->alpha.argFloatComplex.s[0] = 1.0;
        args->alpha.argFloatComplex.s[1] = 0.0;
        args->beta.argFloatComplex.s[0] = beta;
        args->beta.argFloatComplex.s[1] = 0.0;
        break;

    case TYPE_COMPLEX_DOUBLE:
        args->alpha.argDoubleComplex.s[0] = 1.0;
        args->alpha.argDoubleComplex.s[1] = 0.0;
        args->beta.argDoubleComplex.s[0] = beta;
        args->beta.argDoubleComplex.s[1] = 0.0;
        break;
    }
    args->M = mi->M;
    args->N = mi->N;
    args->K = mi->K;

    args->A  = clCreateBuffer(genInfo.ctx, CL_MEM_READ_ONLY,
        args->N * args->M * mi->sizeDType, NULL, &status);
    checkErrorFunc("clCreateBuffer",status);
    mi->clA = args->A;

    status = clEnqueueWriteBuffer(genInfo.queue, args->A, CL_TRUE, 0,
        args->N * args->M * mi->sizeDType, mi->A.v, 0, NULL, NULL);
    checkErrorFunc("clEnqueueWriteBuffer",status);

    args->lda.matrix = args->K;
    args->ldb.matrix = args->K;
    args->ldc.matrix = args->M;

    args->B = clCreateBuffer(genInfo.ctx, CL_MEM_READ_ONLY ,
        args->K * args->N * mi->sizeDType, NULL, &status);
    checkErrorFunc("clCreateBuffer",status);
    mi->clB = args->B;

    status = clEnqueueWriteBuffer(genInfo.queue, args->B, CL_TRUE, 0,
        args->K * args->N * mi->sizeDType, mi->B.v, 0, NULL, NULL);
    checkErrorFunc("clEnqueueWriteBuffer",status);


    args->C = clCreateBuffer(genInfo.ctx, CL_MEM_WRITE_ONLY ,
        args->M * args->K * mi->sizeDType, NULL, &status);
    checkErrorFunc("clCreateBuffer",status);

    mi->clC = args->C;
    args->addrBits = genInfo.deviceInfos.addressBits;
    args->offsetM = 0;
    args->offsetN = 0;
    args->offA = 0;
    args->offBX = 0;
    args->offCY = 0;
    args->scimage[0] = mi->clImgA;
    args->scimage[1] = mi->clImgB;

}

void
initKernelArg(
    MemoryPattern *pattern,
    CLBlasKargs args,
    cl_kernel kernel,
    CLBlasKernelType kernType,
    const CLBLASKernExtra *kextra)
{
    unsigned int ind;
    unsigned int nrArgs;
    cl_int   status;
    KernelArg karg[MAX_KERNEL_ARGS];

    memset(karg, 0, sizeof(KernelArg) * MAX_KERNEL_ARGS);

    args.kernType = kernType;
    pattern->sops->assignKargs(karg, &args, kextra);

    status = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS,
        sizeof(nrArgs), &nrArgs, NULL);

    for (ind = 0; ((ind < nrArgs) && (status == CL_SUCCESS)); ind++) {
        status = clSetKernelArg(kernel, ind, karg[ind].typeSize,
            karg[ind].arg.data);
    }
}

double
runKernel(
    cl_kernel kernel,
    cl_device_id device,
    MemoryPattern *pattern,
    const GParam *param,
    //unsigned int dim,
    CLBlasKargs *args,
    const void *extra,
    unsigned int funcID)
{
    unsigned int nrComputeUnits;
    size_t globalWorkSize[2];
    size_t localWorkSize[3];
    cl_event evt = NULL;
    cl_int  status;
    double ret;

    status = clGetDeviceInfo(device,
        CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl_uint),
        (void*)&nrComputeUnits,
        NULL);
    checkErrorFunc("clGetDeviceInfo",status);

    //////////////////////////////////////////////////////////////////////////
    //calcWorkGroups();

    if (pattern->sops->calcThreads) {
        pattern->sops->calcThreads(globalWorkSize, param->dims,
                                   &param->pgran, args, extra);
    }
    else {
        /////
        SubproblemDim globDim;
        SubproblemDim sd[MAX_SUBDIMS];

        kargsToProbDims(&globDim, funcID, args, false);
        sd[0] = param->dims[0];
        sd[1] = param->dims[1];

        if ((param->pgran.wgDim == 2) && pattern->sops->innerDecompositionAxis) {
            if (pattern->sops->innerDecompositionAxis(args) ==
                DECOMP_AXIS_X) {

                /*
                 * these dimensions will not used more anywhere, so we can
                 * just swap them
                 */
                swapDimXY(&(sd[0]));
                swapDimXY(&(sd[1]));
                swapDimXY(&globDim);
            }
        }
        calcGlobalThreads(globalWorkSize, &(sd[0]),
                          &param->pgran, globDim.y, globDim.x);
    }

    localWorkSize[0] = param->pgran.wgSize[0];
    localWorkSize[1] = param->pgran.wgSize[1];
    localWorkSize[2] = 0;

    fflush(stdout);
    status = clEnqueueNDRangeKernel(genInfo.queue, kernel, param->pgran.wgDim,
                                    NULL, globalWorkSize, localWorkSize,
                                    0, NULL, &evt);
    clReleaseKernel(kernel);
    checkErrorFunc("clEnqueueNDRangeKernel",status);

#if 0
    {
        cl_ulong start, end;

        status = clFinish(genInfo.queue);

        checkErrorFunc("clFinish", status);
        status = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong), &start, NULL);

        checkErrorFunc("clGetEventProfilingInfo",status);
        status = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong), &end, NULL);
        checkErrorFunc("clGetEventProfilingInfo",status);

        end -= start;
        end /= 1000;
        ret = (double)end/1000;
    }
#else
    {
        nano_time_t time;

        status = flushAll(genInfo.queue);
        checkErrorFunc("flushAll", status);

        time = getCurrentTime();
        status = waitForSuccessfulFinish(genInfo.queue, &evt);
        checkErrorFunc("waitForSuccessfulFinish", status);
        time = getCurrentTime() - time;

        ret = (double)conv2nanosec(time)/1000;
        ret /= 1000;
    }

#endif
    clReleaseEvent(evt);
    return ret;
}

void
subInitMatrixInfo(
                  MatrixInfo *matrixInfo,
                  DataType dt,
                  unsigned int sizeType)
{
    matrixInfo->dtype = dt;
    matrixInfo->sizeDType = sizeType;

    matrixInfo->A.v = malloc(matrixInfo->N * matrixInfo->M * sizeType);
    matrixInfo->B.v = malloc(matrixInfo->N * matrixInfo->K * sizeType);
    matrixInfo->C.v = malloc(matrixInfo->M * matrixInfo->K * sizeType);
}

void
initMatrixFloat(FPtr* m, int maxi)
{
    int i;
    for (i = 0; i < maxi; ++i) {
        m->f[i] = 1.0;
    }
}


void
initMatrixInfo(
    MatrixInfo *mi,
    DataType dt,
    DeviceInfo* di,
    BlasExtraInfo* bExtra
    )
{
    unsigned int nDim;
    BlasFunctionInfo* bFunc = bExtra->parent->parent;

    for (nDim = 0; nDim < bExtra->numParam; ++nDim, mi++) {
        unsigned int i;
        unsigned int dimension = getDimension(nDim, dt, di, bFunc->funcNo);

        if (bFunc != NULL && bFunc->initKNM != NULL) {
            bFunc->initKNM(mi, dimension);
        }
        else {
            mi->K = dimension;
            mi->N = dimension;
            mi->M = dimension;
        }


        switch (dt)
        {
        case TYPE_FLOAT:
            subInitMatrixInfo(mi, dt, sizeof(cl_float));
            initMatrixFloat(&mi->A, mi->K * mi->M);
            initMatrixFloat(&mi->B, mi->N * mi->K);
            break;
        case TYPE_DOUBLE:
            subInitMatrixInfo(mi, dt, sizeof(cl_double));
            for (i = 0; i < mi->K * mi->M; ++i) {
                mi->A.d[i] = 1.0;
            }
            for (i = 0; i < mi->N * mi->K; ++i) {
                mi->B.d[i] = 1.0;
            }
            break;
        case TYPE_COMPLEX_FLOAT:
            subInitMatrixInfo(mi, dt, sizeof(cl_float2));
            for (i = 0; i < mi->K * mi->M; ++i) {
                mi->A.f2[i].s[0] = 1.0;
                mi->A.f2[i].s[1] = 0.0;
            }
            for (i = 0; i < mi->N * mi->K; ++i) {
                mi->B.f2[i].s[0] = 1.0;
                mi->B.f2[i].s[1] = 0.0;
            }
            break;
        case TYPE_COMPLEX_DOUBLE:
            subInitMatrixInfo(mi, dt, sizeof(cl_double2));
            for (i = 0; i < mi->K * mi->M; ++i) {
                mi->A.d2[i].s[0] = 1.0;
                mi->A.d2[i].s[1] = 0.0;
            }
            for (i = 0; i < mi->N * mi->K; ++i) {
                mi->B.d2[i].s[0] = 1.0;
                mi->B.d2[i].s[1] = 0.0;
            }
            break;
        default:
            exit (EXIT_COD_UNKNOWN_DATATYPE);
        }

        mi->clA = NULL;
        mi->clB = NULL;
        mi->clC = NULL;

        mi->clImgA = NULL;
        mi->clImgB = NULL;
        mi->imgA = NULL;
        mi->imgB = NULL;

    }
}

void
releaseMemObjOne(MatrixInfo * mi)
{
    clReleaseMemObject(mi->clA);
    clReleaseMemObject(mi->clB);
    clReleaseMemObject(mi->clC);

    mi->clA = NULL;
    mi->clB = NULL;
    mi->clC = NULL;
    mi->clImgA = NULL;
    mi->imgA = NULL;
    mi->clImgB = NULL;
    mi->imgB = NULL;
}

void
releaseMemObjAll(MatrixInfo * mi, BlasExtraInfo* bExtra)
{
    unsigned int nDim;

    for (nDim = 0; nDim < bExtra->numParam; ++nDim, mi++) {
        releaseMemObjOne(mi);
    }
}

void
destroyMatrixInfo(MatrixInfo* mi, BlasExtraInfo* bExtra)
{
    unsigned int nDim;

    for (nDim = 0; nDim < bExtra->numParam; ++nDim, mi++) {
        free(mi->A.v);
        free(mi->B.v);
        free(mi->C.v);
    }
}

void
logBest(
        unsigned int * bestParam,
        unsigned int nDim,
        GParam * gp,
        double * bestTime)
{
    fprintf(logStream,  "        %d  %s = %f\n",bestParam[nDim],
            gp->name,  bestTime[nDim]);
    fflush(logStream);
}
void
logCheckError(int dim)
{
    fprintf(logStream,  " [%5d]:  NOT FOUND\n", dim);
}
void
logCheck(
         int dim,
         SubproblemDim* sdim,
         PGranularity* pgran,
         double t,
         double oldt,
         bool kern)
{
    GParam gp;

    gp.dims[0] = sdim[0];
    gp.dims[1] = sdim[1];
    gp.dims[2] = sdim[2];

    gp.pgran = *pgran;
    createGParamName(&gp);
    if (genInfo.aExtendedOutput) {
        if (oldt == 0) {
            fprintf(logStream,  " [%5d]:  %s  - %7g ",dim, gp.name, t);
            oldt = t;
        }
        if (fabs(t - oldt) < 0.0001) {
            fprintf(logStream, (kern) ? "* " : "+ ");
        }
        else {
            fprintf(logStream,  "- ");
        }
    }
    fflush(logStream);
}


void
logParamName(GParam * params, int cur, int max)
{
    if (genInfo.aExtendedOutput) {

        fprintf(logStream, "%3i/%-3i, %s :", cur, max, params->name);
        fflush(logStream);


    /*  For Debug GEMM, Memmory pattern #4

        fprintf(logStream,
                "%3i/%-3i; wg: %dx%d; iB: %lux%lu; gB: %lux%lu; bw: %lu",
                cur,
                max,
                params->pgran.wgSize[1],
                params->pgran.wgSize[0],
                params->dims[1].x,
                params->dims[1].y,
                params->dims[0].x,
                params->dims[0].y,
                params->dims[0].bwidth);
    */
        fflush(logStream);

    }
    else {
        if (cur > 0) {
            fprintf(logStream, "\b\b\b\b\b\b\b");
        }
        fprintf(logStream, "%5.2f%% ", genInfo.last
                + (genInfo.next - genInfo.last)*cur/max);
        fflush(logStream);
    }

}

void
logTime(double time)
{
    if (genInfo.aExtendedOutput) {
        fprintf(logStream, " %7.2f", time);
        fflush(logStream);
    }
}

void
logKernalGen(void)
{
    if (genInfo.aExtendedOutput) {
        fprintf(logStream, " *");
        fflush(logStream);
    }
}

void
logPattern(const char * patternName)
{
    if ( genInfo.aExtendedOutput || genInfo.patternName != patternName ) {
        fprintf(logStream, "%s is being tuned, progress: ", patternName);
        if (genInfo.aExtendedOutput) {
            fprintf(logStream, "\n");
        }else {
            fprintf(logStream, "       ");
        }
        fflush(logStream);
        genInfo.patternName = patternName;
    }

}
void
logEndString(void)
{
    if (genInfo.aExtendedOutput) {
        fprintf(logStream, "\n");
        fflush(logStream);
    }
}

void
logExtraFlag(
        KernelExtraFlags flags,
        KernelExtraFlags flag,
        const char * trueName,
        const char * falseName
        )
{
    if ((flags & flag) > 0) {
        fprintf(logStream, "%s", trueName);
    }
    else {
        fprintf(logStream, "%s", falseName);
    }

}
void
logEndPattern(unsigned int func, unsigned int patt)
{
    //bool isFunc = (genInfo.aFunc == -1 || genInfo.aFunc == (int)func);
    bool isFunc = genInfo.aFunc[func];
    bool isPattern = (genInfo.aPattern == -1 || genInfo.aPattern == (int)patt);

    if (!(isFunc && isPattern)) {
            return;
    }

    if (!genInfo.aExtendedOutput) {
        fprintf(logStream, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
        fprintf(logStream, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
        fprintf(logStream, " tuning is complete.                     \n");
    }
    fprintf(logStream, "\n");
    fflush(logStream);
}

void
logExtra(BlasExtraInfo* bExtra)
{

    const char*  strType = "";
    const char*  strTrans = "";

    KernelExtraFlags flags = bExtra->flags;

    if (!genInfo.aExtendedOutput) {

        fprintf(logStream, "\b\b\b\b\b\b\b\b %5.2f%% ", genInfo.last);
    }
    else {
        fprintf(logStream, "   Flag (%d):(clblas*)", flags);

        strTrans = (flags & KEXTRA_TRANS_A)? " Trans": " NoTrans";
        logExtraFlag(flags, KEXTRA_CONJUGATE_A,        " ConjTrans", strTrans);
        fprintf(logStream, "(A)");
        strTrans = (flags & KEXTRA_TRANS_B)? " Trans": " NoTrans";
        logExtraFlag(flags, KEXTRA_CONJUGATE_B,        " ConjTrans", strTrans);
        fprintf(logStream, "(B)");
        logExtraFlag(flags, KEXTRA_COLUMN_MAJOR,    " ColumnMajor", " RowMajor");
        logExtraFlag(flags, KEXTRA_UPPER_TRIANG,    " Upper", " Lower");
        logExtraFlag(flags, KEXTRA_SIDE_RIGHT,      " Right", " Left");

        fprintf(logStream, " \n");

        switch (bExtra->dtype)
        {
        case TYPE_FLOAT:            strType = "FLOAT"; break;
        case TYPE_DOUBLE:           strType = "DOUBLE"; break;
        case TYPE_COMPLEX_FLOAT:    strType = "COMPLEX_FLOAT"; break;
        case TYPE_COMPLEX_DOUBLE:   strType = "COMPLEX_DOUBLE"; break;
        }
        fprintf(logStream, "   TYPE = %s:", strType);
    }


    fflush(logStream);
    logEndString();
}

void
logError(void)
{
    fprintf(logStream, " An internal kernel build error occurred!\n");
    fflush(logStream);
}

static void
releaseSCImage(void** buf, cl_mem* clImg)
{
    if (*clImg != NULL) {
        clReleaseMemObject(*clImg);
        *clImg = NULL;
        free(*buf);
        *buf = NULL;
    }
}

static cl_int
createSCImage(
    void **buf,
    cl_mem *image)
{
    cl_image_format format = { CL_RGBA, CL_FLOAT };
    size_t width, height, maxWidth, maxHeight;
    cl_int status;
    cl_ulong memSize;
    cl_device_id device;
    cl_int err;

    err = clGetContextInfo(genInfo.ctx, CL_CONTEXT_DEVICES,
                           sizeof(device), &device, NULL);

    if (err != CL_SUCCESS) {
        return err;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(memSize), &memSize, NULL);

    if (err != CL_SUCCESS) {
        return err;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
            sizeof(maxWidth), &maxWidth, NULL);

    if (err != CL_SUCCESS) {
        return err;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
            sizeof(maxHeight), &maxHeight, NULL);

    if (err != CL_SUCCESS) {
        return err;
    }
    // some functions need 2 scratch images
    memSize /= 2;

    height = (size_t)sqrt((double)memSize / sizeof(cl_float));
    width = height / 4;
    if (height > maxHeight) {
        height = maxHeight;
    }

    if (width > maxWidth) {
        width = maxWidth;
    }

    *buf = calloc(width * height, 4 * sizeof(cl_float));
    if (buf == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    *image = clCreateImage2D(genInfo.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        &format, width, height, 0, *buf, &status);
    if (*image == NULL) {
        free(*buf);
        *buf = NULL;
        return status;
    }
    return CL_SUCCESS;
}

static void
generatePrepKernel(
    cl_device_id device,
    MemoryPattern *pattern,
    GParam * param,
    CLBlasKargs *args,
    CLBLASKernExtra *extra,
    CLBlasKernelType kernType)
{
    PGranularity pgran;
    Kernel *k = NULL;
    size_t bSize;
    char*  bin;
    cl_int status;
    cl_ulong ldsSize;
    CLBlasKernelType kernTypeOld = extra->kernType;

    DUMMY_ARG_USAGE(args);

    extra->kernType = kernType;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong), &ldsSize, NULL);

    pgran = param->pgran;

    k = makeKernel(
        device,
        genInfo.ctx,
        pattern->sops->genKernel,
        param->dims,
        &pgran,
        extra,
        NULL,
        NULL);

    status = clGetProgramInfo(k->program, CL_PROGRAM_BINARY_SIZES,
            sizeof(size_t), &bSize, NULL);
    checkErrorFunc("clGetProgramInfo", status);

    bin = (char *)malloc(sizeof(char)*bSize);
    status = clGetProgramInfo(k->program, CL_PROGRAM_BINARIES, sizeof(char*),
            &bin, NULL);
    checkErrorFunc("clGetProgramInfo", status);

    if (kernType == CLBLAS_PREP_A_KERNEL) {
        param->binariesA = bin;
        param->binary_sizesA = bSize;
        param->kernelPrepA = k;
    }

    if (kernType == CLBLAS_PREP_B_KERNEL) {
        param->binariesB = bin;
        param->binary_sizesB = bSize;
        param->kernelPrepB = k;
    }

    extra->kernType = kernTypeOld;
}

void
delGParam(GParam * gp)
{
    if (gp != NULL) {
        gp->count --;
        if (gp->count ==0){
            destroyKernels(gp);
            free(gp->binaries);
            free(gp->binariesA);
            free(gp->binariesB);
            free(gp);
            gp = NULL;
        }
    }
}

void
setFlagsDependentOnDevice(
        CLBlasKargs* args,
        CLBLASKernExtra* extra,
        GParam* parCur,
        unsigned int func,
        unsigned int patt
        )
{
    SolutionStep step;
    cl_int status;

    step.args = *args;
    step.cmdQueue = genInfo.queue;
    step.extraFlags = extra->flags;
    step.funcID = func;
    step.kernels[0] = NULL;
    step.kernels[1] = NULL;
    step.kernels[2] = NULL;
    //step.node = NULL;
    step.numEventsInWaitList = 0;
    step.patternID = patt;
    step.pgran = parCur->pgran;
    step.subdims[0] = parCur->dims[0];
    step.subdims[1] = parCur->dims[1];
    step.subdims[2] = parCur->dims[2];

    step.device.id = genInfo.targetDevice.id;
    status = identifyDevice(&step.device);
    checkErrorFunc("identifyDevice", status);

    if (step.device.ident.vendor == VENDOR_AMD) {
        extra->flags |= (KEXTRA_VENDOR_AMD | KEXTRA_ENABLE_MAD);
    }
    selectVectorization(&step, extra);
}

bool
genAllKernel(
        CLBlasKargs* args,
        CLBLASKernExtra extra,
        GParam* parCur,
        MemoryPattern * pattern,
        unsigned int func,
        unsigned int patt
        )
{
    bool ret;
    cl_device_id device = genInfo.targetDevice.id;

    if (func == (unsigned int)CLBLAS_SYRK ||
            func == (unsigned int)CLBLAS_SYR2K) {
        extra.flags |= KEXTRA_SYRK_SEPARATE_DIAGONAL;
    }

    setFlagsDependentOnDevice(args, &extra, parCur, func, patt);

    // fixup work group size in respect with desired work dispatch order
    if ((parCur->pgran.wgDim == 2) && pattern->sops->innerDecompositionAxis) {
        if (pattern->sops->innerDecompositionAxis(args) == DECOMP_AXIS_X) {
            unsigned int u;

            u = parCur->pgran.wgSize[0];
            parCur->pgran.wgSize[0] = parCur->pgran.wgSize[1];
            parCur->pgran.wgSize[1] = u;
        }
    }

    if (pattern->sops->fixupArgs) {
        pattern->sops->fixupArgs(args, parCur->dims, &extra);
    }

    ret = genKernel(parCur, &extra, pattern);

    if (patternUseImages(pattern)) {
        generatePrepKernel(device, pattern, parCur, args, &extra,
                CLBLAS_PREP_A_KERNEL);

        generatePrepKernel(device, pattern, parCur, args, &extra,
                CLBLAS_PREP_B_KERNEL);
    }
    return ret;
}

double
runAllKernel(
             MemoryPattern * pattern,
             CLBlasKargs *args,
             GParam* parCur,
             unsigned int funcId,
             double bestTime)
{
    double time;
    double minTime = 1e30;
    int i;
    cl_device_id device = genInfo.targetDevice.id;
    int max_run_kernel = MAX_RUN_KERNEL + (funcBlasLevel(funcId) == 2 ? 7 : 0);


    cl_int status;
    cl_kernel kernel;
    if (patternUseImages(pattern)) {
        /////////////// A //////////////
        cl_kernel kPrepA;
        cl_kernel kPrepB;

        status = clCreateKernelsInProgram(
                parCur->kernelPrepA->program, 1, &kPrepA, NULL);
        checkErrorFunc("clGetProgramInfo", status);

        initKernelArg(pattern, *args, kPrepA, CLBLAS_PREP_A_KERNEL,
                      parCur->kernelPrepA->extra);

        args->kernType = CLBLAS_PREP_A_KERNEL;
        time = runKernel(kPrepA, device, pattern, parCur,
                args, parCur->kernelPrepA->extra, funcId);

        /////////////// B //////////////
        status = clCreateKernelsInProgram(
                parCur->kernelPrepB->program, 1, &kPrepB, NULL);
        checkErrorFunc("clGetProgramInfo", status);


        initKernelArg(pattern, *args, kPrepB, CLBLAS_PREP_B_KERNEL,
                      parCur->kernelPrepB->extra);

        args->kernType = CLBLAS_PREP_B_KERNEL;
        time = runKernel(kPrepB, device, pattern, parCur,
                args, parCur->kernelPrepB->extra, funcId);
        args->kernType = CLBLAS_COMPUTING_KERNEL;
    }

    for (i = 0; i < max_run_kernel; ++i) {
        status = clCreateKernelsInProgram(parCur->kernel->program, 1, &kernel, NULL);
        checkErrorFunc("clGetProgramInfo", status);

        initKernelArg(pattern, *args, kernel, CLBLAS_COMPUTING_KERNEL,
                      parCur->kernel->extra);

        time = runKernel(kernel, device, pattern, parCur, args,
                         parCur->kernel->extra, funcId);
        minTime = fmin(time, minTime);
        if (minTime > bestTime*2 && i >= max_run_kernel/2 && minTime > 2) {
            break;
        }
    }
    return minTime;
}

GParam*
createParCur(SubDimInfo *sdi)
{
    GParam*   parCur   = calloc(1, sizeof(GParam));
    parCur->count ++;

    parCur->dims[0] = sdi->sdim[0];
    parCur->dims[1] = sdi->sdim[1];
    parCur->dims[2] = sdi->sdim[2];
    parCur->pgran   = sdi->pgran;

    createGParamName(parCur);
    return parCur;
}

GParam*
createParCur2(GParam* sdi)
{
    GParam*   parCur   = calloc(1, sizeof(GParam));
    parCur->count = 1;

    parCur->dims[0] = sdi->dims[0];
    parCur->dims[1] = sdi->dims[1];
    parCur->dims[2] = sdi->dims[2];
    parCur->pgran   = sdi->pgran;
    parCur->vecLen  = sdi->vecLen;
    parCur->binaries  = NULL;
    parCur->binariesA = NULL;
    parCur->binariesB = NULL;
    parCur->binary_sizes = 0;
    parCur->binary_sizesA = 0;
    parCur->binary_sizesB = 0;
    parCur->time = sdi->time;

    createGParamName(parCur);
    return parCur;
}

static void
setParam(BlasParamInfo* bParam, double time, GParam* parCur)
{
    bParam->time = time;
    bParam->pGran = parCur->pgran;
    bParam->sDim[0]  = parCur->dims[0];
    bParam->sDim[1]  = parCur->dims[1];
    bParam->sDim[2]  = parCur->dims[2];
    //
    if (genInfo.aIsKernel) {
        bParam->kSize[0] = (unsigned int)parCur->binary_sizes;
        bParam->kSize[1] = (unsigned int)parCur->binary_sizesA;
        bParam->kSize[2] = (unsigned int)parCur->binary_sizesB;
    }
    else {
        bParam->kSize[0] = 0;
        bParam->kSize[1] = 0;
        bParam->kSize[2] = 0;
    }
}

int VISIBILITY_HIDDEN
comp(const void *i, const void *j)
{
    return *(double *)i < *(double *)j;
}

void VISIBILITY_HIDDEN
initCLBLASExtra(CLBLASKernExtra* extra, BlasExtraInfo* bExtra)
{
    memset( extra, 0, sizeof(CLBLASKernExtra) );
//    if (bExtra) {
        extra->dtype = bExtra->dtype;
        extra->flags = bExtra->flags;
//        extra->vecLen = bExtra->vecLen;
//    }
}
#ifdef TEST_LOG

typedef struct LOG_FILE
{
    FILE* f;
    bool readElem;
    double t1, t2, t3;
    double tall;
}LOG_FILE;

typedef struct LOG_STAT
{
    int count;
    double minTime;
    double maxTime;
    double midleTime;
}LOG_STAT;

void
openLogFile(LOG_FILE* lf, char* fileName)
{
    if ((lf->f = fopen(fileName, "a+")) != NULL) {

    }

}
void
closeLogFile(LOG_FILE* lf)
{
    fclose(lf->f);
}
bool
readElemLogFile(LOG_FILE* lf, SubproblemDim* sd, unsigned int vecLen)
{
    unsigned int l0x, l0y, l0w, l1x, l1y, l1w, vl;
    double t1, t2, t3, tall;
    fscanf(lf->f, "%u %u %u %u %u %u %u - %lf %lf %lf %lf\n",
           &l0x, &l0y, &l0w, &l1x, &l1y, &l1w, &vl, &t1, &t2, &t3, &tall);
    if ( l0x == sd[0].x && l0y == sd[0].y && l0w == sd[0].bwidth
       &&  l1x == sd[1].x && l1y == sd[1].y && l1w == sd[1].bwidth
       && vl == vecLen) {
        lf->t1 = t1;
        lf->t2 = t2;
        lf->t3 = t3;
        lf->tall = tall;
        return true;
    }
    return false;
}

double
readLogFile(LOG_FILE* lf, SubproblemDim* sd, unsigned  int vecLen)
{
    lf->t3 = 0;
    lf->readElem = readElemLogFile(lf, sd, vecLen);
    if (!lf->readElem) {
        rewind(lf->f);
        while (!lf->readElem && !feof(lf->f)) {
            lf->readElem =  readElemLogFile(lf, sd, vecLen);
        }
    }
    return lf->t3;
}

double
saveLogFile(LOG_FILE* lf,
        SubproblemDim* sd,
        unsigned int vecLen,
        double* time,
        double timeAll)
{
    if (!lf->readElem) {
        fprintf(lf->f, "%u %u %u %u %u %u %u - %lf %lf %lf %lf\n",
               (unsigned int)sd[0].x, (unsigned int)sd[0].y,
               (unsigned int)sd[0].bwidth,
               (unsigned int)sd[1].x, (unsigned int)sd[1].y,
               (unsigned int)sd[1].bwidth,
               vecLen,
               time[0], time[1], time[2], timeAll);
    }
    return lf->t3;
}

void
getBestVariant(LOG_FILE* lf)
{
    rewind(lf->f);
    lf->tall = 0;
    while (!feof(lf->f)) {
        unsigned int l0x, l0y, l0w, l1x, l1y, l1w, vl;
        double t1, t2, t3, tall;
        fscanf(lf->f, "%u %u %u %u %u %u %u - %lf %lf %lf %lf\n",
               &l0x, &l0y, &l0w, &l1x, &l1y, &l1w, &vl, &t1, &t2, &t3, &tall);
        lf->t1 = fmin(t1, lf->t1);
        lf->t2 = fmin(t2, lf->t2);
        lf->t3 = fmin(t3, lf->t3);
        lf->tall += tall;
    }
}

#endif

static void
findBestParams(
    MemoryPattern *pattern,
    unsigned int func,
    unsigned int patt,
    bool isEnvPattSelected,
    BlasExtraInfo* bExtra,
    GParam*     bestParam[DIMARRAYCOUNT])
{
    unsigned int nDim;
    SolutionStep step;
    MatrixInfo mi [DIMARRAYCOUNT];
    //cl_kernel kernel_old[MAX_CLBLAS_KERNELS_PER_STEP];
    double time[DIMARRAYCOUNT];
    CLBLASKernExtra extra;
    SubDimInfo sdi;
    void*  imgA  = NULL;
    cl_mem clImgA = NULL;
    void*  imgB = NULL;
    cl_mem clImgB = NULL;
    int curStep;
    unsigned int dimension;

#ifdef TEST_LOG
    LOG_FILE lf;
    double all_time = 0;
    double step_time;
    char str[1000];
#endif

    memset(time, 0, sizeof(time));
    initCLBLASExtra(&extra, bExtra);

#ifdef TEST_LOG
    sprintf(str, "test_%d_%d_%d_%d.log",func, patt, extra.dtype, extra.flags);
    openLogFile(&lf,str);
#endif

    // create images
    if (patternUseImages(pattern)) {
        cl_int status;
        // Init Image
        status = createSCImage(&imgA, &clImgA);
        checkErrorFunc("createSCImage", status);
        status = createSCImage(&imgB, &clImgB);
        checkErrorFunc("createSCImage", status);
    }

    initSubDimInfo(&sdi, pattern, &genInfo.deviceInfos, func, patt,
                    extra.dtype, extra.flags);

    initMatrixInfo(mi,  extra.dtype, &genInfo.deviceInfos, bExtra);
    resetSubdim(&sdi);

    curStep = 0;
    while (nextSubdim(&sdi, genInfo.aMaxparam, time[bExtra->numParam - 1]))
    {
        GParam* parCur;
        GParam* lastbest[DIMARRAYCOUNT];
        bool isKernelValid;
        if (bExtra) {
            parCur   = createParCur(&sdi);
        }

        globalDim++;
        curStep++;
        logParamName(parCur, curStep, sdi.varCount);

#ifdef TEST_LOG
        step_time = getCurrentTime();
        time[DIMARRAY_BIG] = readLogFile(&lf, sdi.sdim, sdi.vecLen);
        if (!lf.readElem) {
#endif

#ifdef TEST_LOG
        }
        else {
            time[DIMARRAY_SMALL] = lf.t1;
            time[DIMARRAY_MIDDLE] = lf.t2;
            time[DIMARRAY_BIG] = lf.t3;
            step_time = lf.tall;
        }
#endif

        for (nDim = 0; nDim < bExtra->numParam; nDim++) {
             if (bExtra){
                lastbest[nDim] = NULL;
             }
        }

        isKernelValid = 0;
        for (nDim = 0; nDim < bExtra->numParam; nDim++) {

            BlasParamInfo* bParam;
            // can current combination of flags be handled by selected pattern
            bool isProbSupported = false;

            dimension = getDimension(nDim, extra.dtype,
                                     &genInfo.deviceInfos,
                                     bExtra->parent->parent->funcNo);

            // setup kernel arguments
            if (patternUseImages(pattern)) {
                // Init Image
                mi[nDim].imgA = imgA;
                mi[nDim].clImgA =clImgA;
                mi[nDim].imgB = imgB;
                mi[nDim].clImgB =clImgB;
            }

            // Incorrect subdimension for a given size of the matrix
            if ( dimension < sdi.sdim[0].x ||
                 dimension % sdi.sdim[0].x != 0 ||
                 dimension < sdi.sdim[0].y ||
                 dimension % sdi.sdim[0].y != 0 ||
                 dimension < sdi.sdim[0].bwidth ||
                 dimension % sdi.sdim[0].bwidth != 0
                 ) {

                releaseMemObjOne(mi + nDim);
                if (genInfo.aExtendedOutput) {
                    fprintf(logStream, "        ");
                }
                // write dummy data
                time[nDim] = -1;
                continue;
            }

            step.extraFlags = extra.flags;
            step.funcID = func;
            initCLBlasKArgDim( &step.args, mi + nDim, extra.flags );

            // assuming that all
            // "old-fashioned" patterns, providing no performance estimation
            // function can handle any set of arguments/flags
            if ( NULL == pattern->sops->getPatternPerf ||
                pattern->sops->getPatternPerf( step.extraFlags,
                (void*)&step.args ) >= 0 ) {

                isProbSupported = true;
            }
            else {
                isProbSupported = false;
            }

            // if current flags and dimensions are not optimal for current
            // pattern - skip building and running kernel.
            // But if the pattern is selected by environment
            // and can handle current problem - tune it anyway.
            if ( (patt != selectPattern( &step, 0 ) &&
                 (!isEnvPattSelected || !isProbSupported)) ) {

                releaseMemObjOne(mi + nDim);

                // write dummy data
                time[nDim] = -1;
                bestParam[nDim] = NULL;
                continue;
            }

            if ( 0 == isKernelValid ) {
                isKernelValid = genAllKernel(
                    &step.args,
                    extra,
                    parCur,
                    pattern,
                    func,
                    patt);

                logKernalGen();
            }

            if ( 0 == isKernelValid ) {

                releaseMemObjOne(mi + nDim);
                logError();
                break;
            }
            bParam = &(bExtra->param[nDim]);

#ifdef TEST_LOG
            if (!lf.readElem) {
#endif
                time[nDim] = runAllKernel(pattern, &step.args, parCur,
                                          func, bParam->time);
                releaseMemObjOne(mi + nDim);
#ifdef TEST_LOG
            }
#endif
            logTime(time[nDim]);
            if (bParam->time > time[nDim]) {
                if (bExtra) {
                    BlasParamInfo* bParamNT = &(bExtra->param[nDim]);
                    setParam(bParamNT, time[nDim], parCur);
                    lastbest[nDim] = bestParam[nDim];
                    bestParam[nDim] = parCur;
                    parCur->count++;
                }
            }
        }
        for (nDim = 0; nDim < bExtra->numParam; nDim++) {
            if (bExtra) {
                delGParam(lastbest[nDim]);
                lastbest[nDim] = NULL;
            }
        }

#ifdef TEST_LOG
        step_time = ((double)(getCurrentTime()) - step_time)/1000000;
        saveLogFile(&lf, sdi.sdim, sdi.vecLen, time, step_time);
        if (lf.readElem) {
            step_time = lf.tall;
        }
        logTime(step_time);
        all_time += step_time;
#endif
        logEndString();
        releaseMemObjAll(mi, bExtra);
        if (bExtra) {
            delGParam(parCur);
        }
    }

#ifdef TEST_LOG
    // Show log
    resetSubdim(&sdi);
    double t;
    double all = 0;
    int count = 0;

    time[DIMARRAY_SMALL] = 5000.0;
    time[DIMARRAY_MIDDLE] = 5000.0;
    time[DIMARRAY_BIG] = 5000.0;

    sdi.returnAll = true;
    do
    {
        t = readLogFile(&lf, sdi.sdim, sdi.vecLen);
        if (lf.readElem) {
            time[DIMARRAY_SMALL] = fmin(lf.t1, time[DIMARRAY_SMALL]);
            time[DIMARRAY_MIDDLE] = fmin(lf.t2, time[DIMARRAY_MIDDLE]);
            time[DIMARRAY_BIG] = fmin(lf.t3, time[DIMARRAY_BIG]);
            all+= lf.tall;
            count++;
        }
        else {
            printf ("^");
        }

    } while (nextSubdim(&sdi, genInfo.aMaxparam, t));

#ifdef TEST_LOG
    getBestVariant(&lf);
#endif

    lf.t1 = time[DIMARRAY_SMALL];
    lf.t2 = time[DIMARRAY_MIDDLE];
    lf.t3 = time[DIMARRAY_BIG];
    lf.tall = all;

    fprintf(logStream, "---------------------------------------------------\n");
    fprintf(logStream, "            steps  time1  time2  time3     AllTime \n");
    int tmin = (int)(lf.tall/1000/60);
    int tsec = (int)(lf.tall/1000) - tmin*60;
    fprintf(logStream, " --> Best %5d %7.2lf %7.2lf %7.2lf    %2d:%2d  \n",
            count, lf.t1, lf.t2, lf.t3, tmin, tsec);
    tmin = (int)(all_time/1000/60);
    tsec = (int)(all_time/1000) - tmin*60;
    fprintf(logStream, " --> Fast %5d %7.2lf %7.2lf %7.2lf    %2d:%2d\n",
            curStep,
            bExtra->param[DIMARRAY_SMALL].time,
            bExtra->param[DIMARRAY_MIDDLE].time,
            bExtra->param[DIMARRAY_BIG].time,
            tmin,tsec);

    globalFastTime += all_time;
    globalTime += lf.tall;

    closeLogFile(&lf);
#endif



    logEndString();
     // Release image
     releaseSCImage(&imgA, &clImgA);
     releaseSCImage(&imgB, &clImgB);

     destroyMatrixInfo(mi, bExtra);
}

double
checkData(
		  TargetDevice* devID,
          const MemoryPattern * pattern,
          DataType dtype,
          KernelExtraFlags flags,
          int dim,
          double oldt)
{
    SubproblemDim sdim[MAX_SUBDIMS];
    PGranularity  pgran;
    double time;
    int i;
    unsigned char* buffer[MAX_CLBLAS_KERNELS_PER_STEP];
    size_t sizeBuffer[MAX_CLBLAS_KERNELS_PER_STEP];

    int status;

    for (i = 0; i < MAX_CLBLAS_KERNELS_PER_STEP; ++i) {
        buffer[i] = NULL;
        sizeBuffer[i] = 0;
    }

    status = getGranularityInfo(devID, pattern->name, dtype, flags, dim, sdim,
            &pgran, &time);


    if (status == GF_SUCCESS) {
        status = getKernelInfo(devID, pattern->name, dtype, flags, dim, buffer,
                sizeBuffer);

        logCheck(dim, sdim, &pgran, time, oldt, buffer[0] != NULL);
    }
    else {
        logCheckError(dim);
    }
    free(buffer[0]);
    free(buffer[1]);
    free(buffer[2]);
    return time;
}

void
logDimension(BlasFunctionInfo* bFunc)
{
    int func = bFunc->funcNo;
    int i;

    if (genInfo.aExtendedOutput) {
        printf("FLOAT           ");
        for (i = 0; i < DIMARRAYCOUNT; ++i) {
            printf(" %6u",
                   getDimension(i, TYPE_FLOAT, &genInfo.deviceInfos, func));
        }
        printf("\n");
        printf("DOUBLE          ");
        for (i = 0; i < DIMARRAYCOUNT; ++i) {
            printf(" %6u",
                   getDimension(i, TYPE_DOUBLE, &genInfo.deviceInfos, func));
        }
        printf("\n");
        printf("COMPLEX FLOAT   ");
        for (i = 0; i < DIMARRAYCOUNT; ++i) {
            printf(" %6u",
                   getDimension(i, TYPE_COMPLEX_FLOAT, &genInfo.deviceInfos, func));
        }
        printf("\n");
        printf("COMPLEX DOUBLE  ");
        for (i = 0; i < DIMARRAYCOUNT; ++i) {
            printf(" %6u",
                   getDimension(i, TYPE_COMPLEX_DOUBLE, &genInfo.deviceInfos, func));
        }
        printf("\n");

    }

}

void
calcExtraCount(int index, int indexCount)
{

    genInfo.last = (double)index/indexCount*100;
    genInfo.next = (double)(index + 1)/indexCount*100;
}

int
isFlag(BlasExtraInfo* info, KernelExtraFlags flag)
{
    return (info->flags & flag) == flag;
}

int
isNoFlag(BlasExtraInfo* info, KernelExtraFlags flag)
{
    return (info->flags & flag) == 0;
}

/* Check if current set of flags and datatype should be evaluated
 for current function and pattern.
 It may be skipped due to compiler/runtime bugs and if it is
 considered slow for any of the checked problem sizes

 If skipSlowPatt parameter is set to false - pattern considered slow
 is tuned anyway, otherwise, it is skipped */
bool
skipFlags(BlasExtraInfo* info,
    int patt,
    int func,
    DeviceInfo* pDI,
    bool skipSlowPatt)
{
    bool b = false;
    int i;
    bool shouldTunePatt = false;
    SolutionStep step;

    memset( &step, 0, sizeof(SolutionStep) );

    step.funcID = func;
    step.patternID = patt;
    step.extraFlags = info->flags;

    (void)func;
    (void)patt;

    /* evaluate problem sizes */
    /* skip pattern, if it is not optimal for any of the dimensions
    for current flags */
    if ( skipSlowPatt ) {

        for ( i = 0; i < DIMARRAYCOUNT; i++ ) {

            step.args.M = getDimension( i, info->dtype, pDI, func );
            step.args.N = getDimension( i, info->dtype, pDI, func );
            step.args.K = getDimension( i, info->dtype, pDI, func );

            if ( selectPattern( &step, 0 ) == (unsigned int)patt ){
                shouldTunePatt = true;
            }
        }

        if( false == shouldTunePatt ){
            return true;
        }
    }

    b |= (func == CLBLAS_SYMV) && (info->dtype == TYPE_COMPLEX_FLOAT);
    b |= (func == CLBLAS_SYMV) && (info->dtype == TYPE_COMPLEX_DOUBLE);

    /*
      * WORKAROUND for WINDOWS: Now, for many subproblem dimensions,
      *                         when tuning  TRMM, SYRK, SYR2K functions
      *                         for complex-double type, gives BSoD.
      */

#if defined(_WIN32)
    b |= (func == CLBLAS_TRSM) && (info->dtype == TYPE_COMPLEX_DOUBLE);
    b |= (func == CLBLAS_SYRK) && (info->dtype == TYPE_COMPLEX_DOUBLE);
    b |= (func == CLBLAS_SYR2K) && (info->dtype == TYPE_COMPLEX_DOUBLE);
#endif
    b |= !info->isUseForTunning;
    return b;
}

bool
isFilter(BlasExtraInfo* info, int patt, int func)
{
    int dType =  (int)info->dtype;
    int flag = (int)info->flags;

    bool isFunc = genInfo.aFunc[func];
    bool isPattern = (genInfo.aPattern == -1 || genInfo.aPattern == patt);
    bool isDataType = genInfo.aDType[dType];
    bool isFlag = (genInfo.aFlag == -1 || genInfo.aFlag == flag);

    return (!(isFunc && isPattern && isDataType && isFlag));
}

void
initParamsTime(BlasExtraInfo* bExtra)
{
    unsigned int nDim;

    for (nDim = 0; nDim < bExtra->numParam; nDim++) {
        if (bExtra){
            bExtra->param[nDim].time += 1e50;
        }
    }
}

void
saveBestParams(
    BlasExtraInfo* bExtra,
    GParam*  bestParam[DIMARRAYCOUNT])
{
    unsigned int nDim;

    for (nDim = 0; nDim < bExtra->numParam; nDim++) {
    	if (bExtra){
    		BlasParamInfo* bParam = &bExtra->param[nDim];

    		if (bestParam[nDim] != NULL){
    			saveBestParam(&genInfo.targetDevice, bParam);
            }
        }
    }
}

void
deleteGParams (BlasExtraInfo* bExtra, GParam*  bestParam[DIMARRAYCOUNT])
{
    unsigned int nDim;

    for (nDim = 0; nDim < bExtra->numParam; nDim++) {
        if (bExtra){
            delGParam(bestParam[nDim]);
        }
    }
}

void
checkDatas(BlasExtraInfo* bExtra, const MemoryPattern* pattern)
{
    unsigned int nDim;
    double t;
    unsigned int dimension;
    int func = bExtra->parent->parent->funcNo;

    for (nDim = 0; nDim < bExtra->numParam; nDim++) {
        t = 0;
        if (bExtra) {
            dimension = getDimension(nDim, bExtra->dtype,
                                     &genInfo.deviceInfos, func);
            if(nDim == BANK_ALIGNED_CASE_RECORD_IDX) {
                dimension = 0;
            }
            // TODO add implementation checkData
            (void) pattern;
            t = checkData(&genInfo.targetDevice, pattern, bExtra->dtype,
                          bExtra->flags, dimension, t);
        }
        logEndString();
   }
}

void
generateKernelForOthersFlag( BlasExtraInfo* bExtra,
    GParam*  bestParam[DIMARRAYCOUNT],
    MemoryPattern* pattern)
{
    unsigned int nExtra;
    BlasPatternInfo*  bPatt = bExtra->parent;
    BlasFunctionInfo*  bFunc = bPatt->parent;
    BlasExtraInfo* bExtraOther;
    CLBLASKernExtra extra;
    GParam*  bestParamOther[DIMARRAYCOUNT];
    unsigned int nDim;
    CLBlasKargs args;

    memset( bestParamOther, 0, sizeof(GParam*)*DIMARRAYCOUNT );

    for (nExtra = 0; nExtra < bPatt->numExtra; ++nExtra) {

        bool isMaskFlag;
        bool isEqFlag;
        bool isDataType;
        unsigned int mask;

        bExtraOther = &(bPatt->extra[nExtra]);

        mask = bExtraOther->flags & bFunc->maskForTuningsKernel;
        isMaskFlag =  mask == bExtra->flags;
        isEqFlag  = bExtraOther->flags == bExtra->flags;
        isDataType = bExtra->dtype == bExtraOther->dtype;

        if (isDataType && isMaskFlag && !isEqFlag) {

            for (nDim = 0; nDim < bExtra->numParam; nDim++) {
                if (bestParam[nDim] == NULL) {
                    continue;
                }
                bestParamOther[nDim] = createParCur2(bestParam[nDim]);
            }

            for (nDim = 0; nDim < bExtra->numParam; nDim++) {
                unsigned int nd;

                if (bestParam[nDim] == NULL) {
                    continue;
                }
                for (nd = 0; nd < nDim; ++nd) {
                    if (bestParam[nDim] == bestParam[nd]) {
                            bestParamOther[nDim] = bestParamOther[nd];
                            bestParamOther[nDim]->count++;
                    }
                }

                //  If the user selected that they want to store the kernel binaries to disk,
                //  and we do not have those binaries, generate them again
                if (genInfo.aIsKernel && bestParamOther[nDim]->kernel == NULL) {
                    MatrixInfo mi [DIMARRAYCOUNT];
                    unsigned int func = bFunc->funcNo;
                    unsigned int patt = bPatt->pattNo;

                    //  Initialize resources to generate kernels in genAllKernel
                    initCLBLASExtra(&extra, bExtra);
                    initMatrixInfo( mi, extra.dtype, &genInfo.deviceInfos, bExtra );
                    initCLBlasKArgDim( &args, mi, extra.flags );

                    genAllKernel(&args, extra, bestParamOther[nDim], pattern, func, patt);

                    //  Free those resources when finished
                    releaseMemObjAll( mi, bExtra );
                    destroyMatrixInfo( mi, bExtra );

                    logKernalGen( );
                }

                //  This stores the kernel binaries to disk
                saveBestParams(bExtraOther, bestParamOther);
            }
            deleteGParams(bExtraOther, bestParamOther);
         }
     }
}

BlasPatternInfo*
getPattern(BlasFunctionID fid, int pid)
{
	BlasFunctionInfo* pFunc =  getBlasFunctionInfo(&genInfo.targetDevice, fid);
    return &pFunc->pattInfo[pid];
}

void
configurePattern(void)
{
    // Initialization specific to the handler function.
    //getPattern(CLBLAS_XXXX, 0)->isPGValid = ;
    //getPattern(CLBLAS_XXXX, 0)->initSubdim = ;
}

bool
isRebuild(BlasExtraInfo* bExtra)
{
    unsigned int nDim;
    bool ret = genInfo.aCommand != C_DEFAULT;

    for (nDim = 0; nDim < bExtra->numParam; ++nDim) {
        BlasParamInfo* bParam = &bExtra->param[nDim];

        ret |= bParam->sstatus == SS_NOLOAD;
        if (bParam->offset == 0 ) {
            printf("*****\n");
        }
    }
    return ret;
}


void
createFile(void)
{
    unsigned int funcId;
    unsigned int pattId = 0;
    unsigned int envPattId = 0;
    bool isEnvPattSelected = false;
    unsigned int dev;

    //  This intializes global genInfo with either the last detected platform, or the
    //  first AMD platform it finds.  It records the number of devices in that platform.
    initOpenCl( );

    // For each devices
    for (dev = 0; dev < genInfo.numDevices; dev++) {
    	initDevice(dev);

        //  The following creates the .kdb file on disk according to the set environment variable
        writeStorageCache(&genInfo.targetDevice);

        //  The following creates the OpenCL context and commanqueue for the first device in global genInfo struct
        getContext( );

        //  Does nothing; nop
        configurePattern( );

        // for each function
        for (funcId = 0; funcId < BLAS_FUNCTIONS_NUMBER; funcId++) {

            char *pRest = NULL;
            BlasFunctionInfo *funcInfo = getBlasFunctionInfo(
                &genInfo.targetDevice,
                funcId );

            if (funcInfo->envImplementation != NULL) {
                const char *envImpl;

                envImpl = getenv(funcInfo->envImplementation);
                if (envImpl != NULL) {

                    envPattId = strtoul( envImpl, &pRest, 10 );
                    //wrong value of env. variable AMD_CLBLAS_X_IMPLEMENTATION
                    if( 0 == strlen( envImpl ) ||
                        pRest != envImpl + strlen(envImpl) ){

                        isEnvPattSelected = false;
                    }
                    else{

                        isEnvPattSelected = true;
                    }
                }
                else{

                    isEnvPattSelected = false;
                }

            }

            // if pattern is selected by environment - tune it
            // otherwise - start from the pattern number 0
            if( true == isEnvPattSelected ){
                pattId = envPattId;
            }
            else{
                pattId = 0;
            }

            //logPattern( funcInfo->name );
            do
            {
                unsigned int nExtra;
                unsigned int nTuneExtra = 0;
                BlasPatternInfo * bPatt;
                MemoryPattern* pattern;

                bPatt = &(funcInfo->pattInfo[pattId]);
                pattern = &(funcInfo->pattern[pattId]);

                //if select a new trsm memory pattern (#3), then skip it
                if ( funcId == CLBLAS_TRSM && pattId == 3) {
                    pattId++;
                    continue;
                }

                for (nExtra = 0; nExtra < bPatt->numExtra; ++nExtra) {
                    bool isRebuildRequired;

                    BlasExtraInfo* bExtra;
                    bExtra = &(bPatt->extra[nExtra]);
                    genInfo.last = 0;

                    //  This evaluates whether the current combination of parameters from the given function should be tuned or not
                    //  If skipFlags returns 1, then the this combination is skipped
                    //  This checks for hardcoded combinations which are skipped because of known runtime bugs.  
                    if ( skipFlags(bExtra,
                            pattId,
                            funcId,
                            &genInfo.deviceInfos,
                            !isEnvPattSelected ) ) {
                        continue;
                    }

                    //  Similar logic to skipFlags, but this mostly filters out cases that were specified on the command line
                    if (isFilter(bExtra, pattId, funcId)) {
                        continue;
                    }
                    logPattern( funcInfo->name );

                    calcExtraCount(nTuneExtra, bPatt->numTuneExtra);
                    nTuneExtra++;

                    logDimension(funcInfo);
                    logExtra(bExtra);

                    isRebuildRequired = isRebuild(bExtra);

                    if (isRebuildRequired) {
                        size_t bestPatamSize = sizeof(GParam*)*DIMARRAYCOUNT;

                        GParam* bestParam[DIMARRAYCOUNT];

                        memset(bestParam, 0, bestPatamSize);

                        initParamsTime(bExtra);

                        findBestParams( pattern,
                            funcId,
                            pattId,
                            isEnvPattSelected,
                            bExtra,
                            bestParam);

                        saveBestParams(bExtra, bestParam);

                        generateKernelForOthersFlag( bExtra,
                            bestParam,
                            pattern);

                        deleteGParams(bExtra, bestParam);
                    }
                    checkDatas(bExtra, pattern);
                } /* extra */
                //logEndPattern(funcId, pattId);

                pattId++;
            /* patt */
            }while( false == isEnvPattSelected &&
                    pattId < clblasSolvers[funcId].nrPatterns );

        } /* func */
    } /* dev */
    destroyGenInfo();
}

void
parseArg(int argc, char*  argv[])
{

    static char* help=  "clblasTune - automatically tune the clblas "
                        "library for specific hardware.\n"
                        "\n"
                        "clblas function related parameters:\n"
                        "   --gemm\n"
                        "       Tune kernels for the GEMM function family.\n"
                        "   --trmm\n"
                        "       Tune kernels for the TRMM function family.\n"
                        "   --trsm\n"
                        "       Tune kernels for the TRSM function family.\n"
                        "   --gemv\n"
                        "       Tune kernels for the GEMV function family.\n"
                        "   --symv\n"
                        "       Tune kernels for the SYMV function family.\n"
                        "   --syrk\n"
                        "       Tune kernels for the SYRK function family.\n"
                        "   --syr2k\n"
                        "       Tune kernels for the SYR2K function family.\n"
                        "\n"
                        "   You can specify the parameters of "
                        "several alternatives simultaneously.\n"
                        "\n"
                        "   If any of these parameters is not specified the "
                        "tool tries kernels for all the functions.\n"
                        "\n"
                        " Used data types:\n"
                        "   --float\n"
                        "       Single precision version of functions.\n"
                        "   --double\n"
                        "       Double precision version of functions.\n"
                        "   --complex\n"
                        "       Single complex precision version of functions.\n"
                        "   --double-complex\n"
                        "       Double complex precision version of functions.\n"
                        "\n"
                        "   You can specify the parameters of "
                        "several alternatives simultaneously.\n"
                        "\n"
                        "   If any of these parameters is not specified the "
                        "tool tries kernels for all the data types.\n"
                        "\n"
                        "Management:\n"
                        "   --fast\n"
                        "       Using this option allows you to accelerate "
                        "tuning in up to 2-3 times. Achieving an optimal result "
                        "is not guaranteed.\n"
                        "   --rebuild\n"
                        "       Re-tuning the fastest OpenCL kernels. Can be "
                        "used after the driver update.\n"
                        "   --store-kernels\n"
                        "       Store found best kernels into a database file\n"
                        "       WARNING! The file can be very large.\n"
                        "\n"

                        ;


    static char* args[] = { "--gemm",               // 0
                            "--trmm",               // 1
                            "--trsm",               // 2
                            "--buffers",            // 3
                            "--images",             // 4
                            "--float",              // 5
                            "--double",             // 6
                            "--complex",            // 7
                            "--double-complex",     // 8
                            "--store-kernels",      // 9
                            "--rebuild",            // 10
#if defined(_EXTENDED_TUNE_ARG)
                            "--e",                  // 11
                            "--max",                // 12
                            "--extended-output",    // 13
#else
                            "",
                            "",
                            "",
#endif
                            "--gemv",               // 14
                            "--symv",               // 15
                            "--syrk",               // 20
                            "--syr2k",              // 17
                            "--fast",               // 18
                            "--caches",             // 19
                            "--help"                // 20
                            };
    int i;
    unsigned int j;
    bool isSetFunction = false;
    bool isSetType = false;

    genInfo.aAll = true;

    for (i = 1; i < argc; ++i) {
        char * arg = argv[i];
        bool b = true;
        for (j = 0; j < sizeof(args)/sizeof(char*); ++ j){
            if (strcmp(arg, args[j]) == 0){
#if defined(_EXTENDED_TUNE_ARG)
                int argi = 0;
#endif
                switch (j){
                    case 0 :
                        genInfo.aFunc[CLBLAS_GEMM] = true;
                        isSetFunction = true;
                        break;
                    case 1 :
                        genInfo.aFunc[CLBLAS_TRMM] = true;
                        isSetFunction = true;
                        break;
                    case 2 :
                        genInfo.aFunc[CLBLAS_TRSM] = true;
                        isSetFunction = true;
                        break;
                    case 3 : genInfo.aPattern = 0;                  break;
                    case 4 : genInfo.aPattern = 1;                  break;
                    case 5 :
                        genInfo.aDType[TYPE_FLOAT] = true;
                        isSetType = true;
                        break;
                    case 6 :
                        genInfo.aDType[TYPE_DOUBLE] = true;
                        isSetType = true;
                        break;
                    case 7 :
                        genInfo.aDType[TYPE_COMPLEX_FLOAT] = true;
                        isSetType = true;
                        break;
                    case 8 :
                        genInfo.aDType[TYPE_COMPLEX_DOUBLE] = true;
                        isSetType = true;
                        break;
                    case 9 : genInfo.aIsKernel = true;              break;
                    case 10: genInfo.aCommand = C_REBUILD;          break;
#if defined(_EXTENDED_TUNE_ARG)
                    case 11:
                        i++;
                        argi = atoi(argv[i]);
                        genInfo.aFlag = argi;
                        break;
                    case 12:
                        i++;
                        argi = atoi(argv[i]);
                        genInfo.aMaxparam = argi;
                        break;
                    case 13:
                        genInfo.aExtendedOutput = true;
                        break;
#endif
                    case 14:
                        genInfo.aFunc[CLBLAS_GEMV] = true;
                        isSetFunction = true;
                        break;
                    case 15:
                        genInfo.aFunc[CLBLAS_SYMV] = true;
                        isSetFunction = true;
                        break;
                    case 16:
                        genInfo.aFunc[CLBLAS_SYRK] = true;
                        isSetFunction = true;
                        break;
                    case 17:
                        genInfo.aFunc[CLBLAS_SYR2K] = true;
                        isSetFunction = true;
                        break;
                    case 18: genInfo.aAll  = false;                break;
                    case 19: genInfo.aPattern = 2;                 break;
                    case 20:
                        printf ("%s", help);
                        exit(0);
                        break;
                }
                b = false;
            }
        }
        if (b) {
            fprintf(stdout, "Unknown argument %s\n", arg);
        }
    }
    if (!isSetFunction) {
        for (i=0; i < BLAS_FUNCTIONS_NUMBER; ++i) {
            genInfo.aFunc[i] = 1;
        }
    }
    if (!isSetType) {
        for (i=0; i < TYPE_NUMBER; ++i) {
            genInfo.aDType[i] = 1;
        }
    }
}

int
main(int argc, char*  argv[])
{
    FILE_PATH = getenv(ENV_FILE_PATH);

    //  This clears and initializes the global GeneratorInfoRec genInfo struct
    initGeneratorInfoRec( );
    parseArg(argc, argv);

    //  This will
    //  Set up the global clblasSolvers for all function families supported within blas, including initializing memory patterns
    //  Identify all recognized devices in the system
    clblasSetup();

    if (!FILE_PATH){
        printf("The environment variable 'CLBLAS_STORAGE_PATH' is not defined\n");
        exit(EXIT_COD_NO_ENVIRONMENT_VARIABLE);
    }

    logStream = stdout;
    createFile();

#ifdef TEST_LOG

    int h = (int)(globalTime/1000/60/60);
    int m = (int)(globalTime/1000/60) - h*60;
    int c = (int)(globalTime/1000) - m*60 - h*60*60;
    fprintf(logStream, " --> All  time : %2d:%2d:%2d  \n",h, m,c);

    h = (int)(globalFastTime/1000/60/60);
    m = (int)(globalFastTime/1000/60) - h*60;
    c = (int)(globalFastTime/1000) - m*60 - h*60*60;
    fprintf(logStream, " --> Fast time : %2d:%2d:%2d  \n",h, m,c);
#endif
}

char*
getDeviceName(cl_device_id devID, int * status)
{
    char* devName;
    size_t size;
    *status = clGetDeviceInfo(devID, CL_DEVICE_NAME, 0, NULL, &size);
    checkErrorFunc("clGetDeviceInfo", *status);

    devName = malloc(size * sizeof(char));

    *status = clGetDeviceInfo(devID, CL_DEVICE_NAME, size, devName, NULL);
    checkErrorFunc("clGetDeviceInfo", *status);
    return devName;
}

