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


#ifndef STORAGEDATA_H_
#define STORAGEDATA_H_

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <trace_malloc.h>

#include "toolslib.h"
#include "solution_seq.h"
#include "matrix_dims.h"

//
typedef unsigned int OFFSET;

/* Device information needed for tuning CLBLAS kernels. */
typedef struct CLDeviceInfoRec {
    cl_uint       nrComputeUnits;   /* CL_DEVICE_MAX_COMPUTE_UNITS */
    unsigned int  nrStreamCores;    /* Number of stream cores per Compute Unit */
    cl_ulong      globalSize;       /* CL_DEVICE_GLOBAL_MEM_SIZE */
    cl_ulong      maxMemAllocSize;  /* CL_DEVICE_MAX_MEM_ALLOC_SIZE */
    cl_ulong      ldsSize;          /* CL_DEVICE_LOCAL_MEM_SIZE */
    unsigned int  wavefront;        /* Number of work-items executed in parallel on hardware */
    cl_uint       alignment;        /* CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE */
    unsigned int  addressBits;      /* CL_DEVICE_ADDRESS_BITS */
    size_t        workItemSizes[3]; /* CL_DEVICE_MAX_WORK_ITEM_SIZES */
    cl_uint       workItemSizesDim; /* CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS */
    size_t        workGroupSizes;   /* CL_DEVICE_MAX_WORK_GROUP_SIZE */
    bool          nativeDouble;     /* Specifies whether device supports double precision float */
    bool          nativeComplex;    /* Specifies whether device supports complex float */
    TargetDevice* tdev;
} DeviceInfo;


typedef enum Dimensions{
    DIMARRAY_SMALL,
    DIMARRAY_SHORT,
    DIMARRAY_MIDDLE,
    DIMARRAY_BIG,
    DIMARRAY_HUGE,
    DIMARRAY_BANK_CONFLICT,
    DIMARRAYCOUNT,      //
}Dimensions;

//
struct SubDimInfo;
struct BlasFunctionInfo;
struct BlasPatternInfo;
struct BlasExtraInfo;
struct MatrixInfo;

typedef enum SynchStatus
{
    SS_NOLOAD,
    SS_CORRECT_DATA,
    SS_INCORRECT_DATA,
}SynchStatus;


typedef struct BlasParamInfo
{
    int             dim;
    SubproblemDim   sDim[MAX_SUBDIMS];
    PGranularity    pGran;

    OFFSET          kernel[MAX_CLBLAS_KERNELS_PER_STEP];
    unsigned int    kSize[MAX_CLBLAS_KERNELS_PER_STEP];
    double          time;

    OFFSET      offset;
    size_t      size;
    SynchStatus sstatus;
} BlasParamInfo;


typedef struct BlasExtraInfo
{
    struct BlasPatternInfo* parent;

    unsigned int      numParam;
    DataType          dtype;
    KernelExtraFlags  flags;
    unsigned int      vecLen;
    bool              isUseForTunning;

    BlasParamInfo* param;

    OFFSET      offset;
    size_t      size;
    SynchStatus sstatus;
} BlasExtraInfo;

typedef struct BlasPatternInfo
{
    struct BlasFunctionInfo* parent;

    unsigned int   numExtra;
    unsigned int   numTuneExtra;
    BlasExtraInfo* extra;
    const char   * name;

    OFFSET      offset;
    size_t      size;
    SynchStatus sstatus;

    unsigned int pattNo;
    bool (*isPGValid) (struct SubDimInfo* sdi);
    void (*initSubdim)(struct SubDimInfo* sdi);

} BlasPatternInfo;

typedef struct BlasFunctionInfo
{
    unsigned int      numPatterns;
    int               funcNo;
    unsigned int      maskForTuningsKernel;
    unsigned int      maskForUniqueKernels;
    const char*       envImplementation;
    int               defaultPattern;
    const char*       name;
    //

    bool (*isValidFlag) (DataType curType, unsigned int  flags);
    void (*initFunctionInfo) (struct BlasFunctionInfo* bFunc);
    void (*initKNM) (struct MatrixInfo*, unsigned int baseDim);

    BlasPatternInfo   pattInfo[MEMPAT_PER_BLASFN];
    MemoryPattern     pattern[MEMPAT_PER_BLASFN];

} BlasFunctionInfo;


typedef struct StorageCacheImpl
{
    char* fpath;
    char* fpath_tmp;
    bool isInit;     //
    bool isPopulate; // The cache has been initialized,
                     // but does not contain data
    BlasFunctionInfo functionInfo[BLAS_FUNCTIONS_NUMBER];
    DeviceIdent  devIdent;

    OFFSET endFile;
} StorageCacheImpl;

/*
 * The 'force' argument set to true means returning a cache object even
 * if the file on disk doesn't exist
 */
StorageCacheImpl* getStorageCache(TargetDevice* devID, bool force);

BlasParamInfo*  findParam(StorageCacheImpl* cache,
                          const char* pattName, const DataType dt,
                          const KernelExtraFlags kflag, int dim);

void loadKernelsFromFile(StorageCacheImpl* cache, BlasParamInfo* bParam,
                unsigned char** buffer, size_t* sizeBuffer);

void loadDataFromFile(StorageCacheImpl* cache);

char * createFullPatch(const char * name, bool tmp);

OFFSET calcOffset(BlasFunctionInfo* functionInfo);

BlasPatternInfo * getPatternInfo(StorageCacheImpl* cache, unsigned int func,
                                 unsigned int patt);

void  nextPattern(StorageCacheImpl* cache, unsigned int* func,
                  unsigned int* patt);
void saveBestParam(TargetDevice* tdev, BlasParamInfo* bParam);

unsigned int getDimension(int idx, DataType dt, DeviceInfo* di, int func);
bool initReadingData(StorageCacheImpl* cacheImpl, TargetDevice* devID );
void initBlasFuncionData(BlasFunctionInfo* fInfo);
void initCacheData (BlasFunctionInfo* bFunc, DeviceInfo* defInfo);
void initCLDeviceInfoRec(TargetDevice* tdev, DeviceInfo *devInfo);
void destroyData(BlasFunctionInfo* fInfo);

#endif /* STORAGEDATA_H_ */
