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
#include <signal.h>

#include "storage_data.h"
#include "toolslib.h"
#include "devinfo.h"
#include "assert.h"
#include "clblas_stddef.h"
#include "mutex.h"


// The array size is the total number devices on all platforms
static StorageCacheImpl* storageCacheArray = NULL;
// Number of items in storage cache array
// is the number of unique devices.
static unsigned int    storageCacheArrayCount = 0;

static mutex_t *storageCacheLock = NULL;


static void
clearPatternsNumber(BlasFunctionInfo *funcInfo)
{
    int i;

    for (i = 0; i < BLAS_FUNCTIONS_NUMBER; i++) {
        funcInfo[i].numPatterns = 0;
    }
}

char*
getDevName(TargetDevice* tdev)
{
    size_t size;
    char* name;

    clGetDeviceInfo(tdev->id, CL_DEVICE_NAME, 0, NULL, &size);
    name = malloc(size * sizeof(char));
    clGetDeviceInfo(tdev->id, CL_DEVICE_NAME, size, name, NULL);

    return name;
}

void
initCLDeviceInfoRec(TargetDevice* tdev, DeviceInfo *devInfo)
{
    cl_int status = 0;
    cl_uint bDouble;
    cl_device_id devID = tdev->id;
    devInfo->tdev = tdev;

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl_uint),
        &(devInfo->nrComputeUnits),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(cl_ulong),
        &(devInfo->globalSize),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(cl_ulong),
        &(devInfo->ldsSize),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(cl_ulong),
        &(devInfo->maxMemAllocSize),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
        sizeof(cl_uint),
        &(devInfo->alignment),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
        sizeof(cl_uint),
        &(devInfo->workItemSizesDim),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(size_t) * devInfo->workItemSizesDim,
        &(devInfo->workItemSizes),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(size_t) ,
        &(devInfo->workGroupSizes),
        NULL);

    status = clGetDeviceInfo(devID,
        CL_DEVICE_ADDRESS_BITS,
        sizeof(cl_uint),
        &(devInfo->addressBits),
        NULL);



    status = clGetDeviceInfo(devID,
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
        sizeof(cl_uint),
        &bDouble,
        NULL);
    devInfo->nativeDouble = deviceHasNativeDouble(devID, &status);

    // Values are put randomly.
    //TODO  To use the correct data
    devInfo->nrStreamCores = 1;     /* Number of stream cores per Compute Unit */
    devInfo->wavefront = 64;        /* Number of work-items executed in parallel on hardware */
    devInfo->nativeComplex = true;  /* Specifies whether device supports complex float */
}

bool
initReadingData(StorageCacheImpl* cacheImpl, TargetDevice* tdev )
{
    char* devName;
    DeviceInfo  defInf;

    initCLDeviceInfoRec(tdev, &defInf);
    initBlasFuncionData(cacheImpl->functionInfo);
    initCacheData(cacheImpl->functionInfo, &defInf);

    cacheImpl->endFile = calcOffset(cacheImpl->functionInfo);
    devName = getDevName(tdev);
    cacheImpl->fpath = createFullPatch(devName, false);
    cacheImpl->fpath_tmp = createFullPatch(devName, true);
    free(devName);
    if (cacheImpl->fpath == NULL) {
        return false;
    }

    return true;
}

int
getGranularityInfo(
    // In
    TargetDevice* tdev,
    const char* pattName,
    const DataType dt,
    const KernelExtraFlags kflag,
    int dim,  //
    // Out
    SubproblemDim *sdim,
    PGranularity *pgran,
    double *time)
{
    BlasParamInfo* bParam;
    int ret = GF_ERROR;
    int r;
    StorageCacheImpl* cache = getStorageCache(tdev, false);

    if (cache == NULL) {
        return ret;
    }

    bParam = findParam(cache, pattName, dt, kflag, dim);
    if (bParam != NULL) {
        r = bParam->sstatus != SS_CORRECT_DATA;
        if (!r) {
            memcpy(sdim, bParam->sDim, sizeof(SubproblemDim)* MAX_SUBDIMS);
            memcpy(pgran,&bParam->pGran, sizeof(PGranularity));
            *time = bParam->time;
            ret = GF_SUCCESS;
        }
        else if (r == -1) {
            ret = GF_CORRUPT_FILE;
            //printCorruptionError(devID);
        }
    }

    return ret;
}

int
getKernelInfo(
    TargetDevice* devID,
    const char* pattName,
    const DataType dt,
    const KernelExtraFlags kflag,
    int dim,
    unsigned char** buffer,
    size_t* sizeBuffer)
{
    BlasParamInfo* bParam;
    int ret = GF_ERROR;
    StorageCacheImpl* cache = getStorageCache(devID, false);

    if (cache == NULL) {
        return ret;
    }

    memset(buffer, 0, sizeof(char*) * MAX_CLBLAS_KERNELS_PER_STEP);
    memset(sizeBuffer, 0, sizeof(size_t) * MAX_CLBLAS_KERNELS_PER_STEP);
    if (cache->isPopulate) {
        bParam = findParam(cache, pattName, dt, kflag, dim);
        if (bParam != NULL) {
            loadKernelsFromFile(cache, bParam, buffer, sizeBuffer);
            if (buffer[0] == NULL) {
                ret = GF_SUCCESS;
            }
        }
    }
    return ret;
}
/******************************************************************************/

void
destroyStorageCache(void)
{
    unsigned int i;
    StorageCacheImpl*  curCache;

    if(storageCacheArray != NULL) {
        for (i = 0; i < storageCacheArrayCount; i++) {
            curCache = &storageCacheArray[i];

            if (curCache != NULL) {
                destroyData(curCache->functionInfo);

                if (curCache->fpath != NULL) {
                    free(curCache->fpath);
                }
                if (curCache->fpath_tmp != NULL) {
                    free(curCache->fpath_tmp);
                }

                curCache->isPopulate = false;
            }
        }

        storageCacheArrayCount = 0;

        mutexDestroy(storageCacheLock);
        storageCacheLock = NULL;

        free(storageCacheArray);
        storageCacheArray = NULL;
    }
}

BlasFunctionInfo*
getBlasFunctionInfo(TargetDevice* tdev, int func)
{
    StorageCacheImpl*  impl = getStorageCache(tdev, false);
    BlasFunctionInfo* ret = NULL;

    if (impl == NULL) {
        return NULL;
    }

    if (func >= 0 && func < BLAS_FUNCTIONS_NUMBER) {
        ret = &impl->functionInfo[func];
    }
    return ret;
}


#define CHECK_(X) \
        res = X; \
        if (!res) { \
            printf("ERROR %s\n", #X); \
            /*raise(SIGTRAP);*/ \
        }

void checkFILE(TargetDevice* tdev, BlasFunctionInfo* fiArr)
{
    StorageCacheImpl*  impl;
    bool res;
    int func;
    unsigned int patt;
    unsigned int extra;
    unsigned int param;

    impl = getStorageCache(tdev, false);
    if (impl == NULL) {
        return;
    }

    for (func = 0; func < BLAS_FUNCTIONS_NUMBER; func++) {
        BlasFunctionInfo* cfi = &impl->functionInfo[func];
        BlasFunctionInfo* fi = &fiArr[func];

        CHECK_(cfi->funcNo == fi->funcNo);
        CHECK_(cfi->numPatterns == fi->numPatterns);
        CHECK_(cfi->maskForTuningsKernel == fi->maskForTuningsKernel);
        CHECK_(cfi->maskForUniqueKernels == fi->maskForUniqueKernels);
        CHECK_(cfi->defaultPattern == fi->defaultPattern);
        CHECK_(cfi->defaultPattern == fi->defaultPattern);
        CHECK_(strcmp(cfi->name, fi->name) == 0);
        //CHECK_(cfi-> == fi->)
        for (patt = 0; patt < fi->numPatterns; ++patt) {
            BlasPatternInfo* cpi = &cfi->pattInfo[patt];
            BlasPatternInfo* pi = &fi->pattInfo[patt];
            MemoryPattern*  cmp = &cfi->pattern[patt];
            MemoryPattern*  mp = &fi->pattern[patt];

            CHECK_(cpi->numExtra == pi->numExtra );
            CHECK_(cpi->numTuneExtra == pi->numTuneExtra);
            CHECK_(cpi->offset == pi->offset);
            CHECK_(cpi->size == pi->size);
            //CHECK_(cpi->sstatus == pi->sstatus);
            CHECK_(cpi->pattNo == pi->pattNo);
            CHECK_(strcmp(cpi->name, pi->name) == 0);

            CHECK_(cmp->nrLevels == mp->nrLevels );
            CHECK_(cmp->cuLevel == mp->cuLevel );
            CHECK_(cmp->thLevel == mp->thLevel );
            CHECK_(cmp->sops == mp->sops );
            CHECK_(cmp->extra == mp->extra );
            CHECK_(strcmp(cmp->name, mp->name) == 0);

            for (extra = 0; extra < pi->numExtra; ++extra) {
                BlasExtraInfo* cei = &cpi->extra[extra];
                BlasExtraInfo* ei = &pi->extra[extra];

                CHECK_(cei->numParam == ei->numParam);
                CHECK_(cei->dtype == ei->dtype);
                CHECK_(cei->flags == ei->flags);
                CHECK_(cei->vecLen == ei->vecLen);
                CHECK_(cei->isUseForTunning == ei->isUseForTunning);

                CHECK_(cei->offset == ei->offset);
                CHECK_(cei->size == ei->size);
                CHECK_(cei->sstatus == ei->sstatus);


                for (param = 0; param < ei->numParam; ++param) {
                    BlasParamInfo* cpri = &cei->param[param];
                    BlasParamInfo* pri = &ei->param[param];

                    CHECK_(cpri->dim == pri->dim);
                    CHECK_(cpri->pGran.wfSize == pri->pGran.wfSize);
                    CHECK_(cpri->pGran.wgDim == pri->pGran.wgDim);
                    CHECK_(cpri->pGran.wgSize[0] == pri->pGran.wgSize[0]);
                    CHECK_(cpri->pGran.wgSize[1] == pri->pGran.wgSize[1]);
                    CHECK_(cpri->sDim[0].bwidth == pri->sDim[0].bwidth);
                    CHECK_(cpri->sDim[0].itemX== pri->sDim[0].itemX);
                    CHECK_(cpri->sDim[0].itemY== pri->sDim[0].itemY);
                    CHECK_(cpri->sDim[0].x == pri->sDim[0].x);
                    CHECK_(cpri->sDim[0].y == pri->sDim[0].y);
                    CHECK_(cpri->sDim[1].bwidth == pri->sDim[1].bwidth);
                    CHECK_(cpri->sDim[1].itemX== pri->sDim[1].itemX);
                    CHECK_(cpri->sDim[1].itemY== pri->sDim[1].itemY);
                    CHECK_(cpri->sDim[1].x == pri->sDim[1].x);
                    CHECK_(cpri->sDim[1].y == pri->sDim[1].y);
                    CHECK_(cpri->sDim[2].bwidth == pri->sDim[2].bwidth);
                    CHECK_(cpri->sDim[2].itemX== pri->sDim[2].itemX);
                    CHECK_(cpri->sDim[2].itemY== pri->sDim[2].itemY);
                    CHECK_(cpri->sDim[2].x == pri->sDim[2].x);
                    CHECK_(cpri->sDim[2].y == pri->sDim[2].y);
                    CHECK_(cpri->time == pri->time);
                    CHECK_(cpri->offset == pri->offset);
                    CHECK_(cpri->size == pri->size);
                    CHECK_(cpri->sstatus == pri->sstatus);
                }
            }
        }
    }
}

bool
isDeviceEQ(DeviceIdent* dev1, DeviceIdent* dev2)
{
    bool ret = true;

    ret &= dev1->chip == dev2->chip;
    ret &= dev1->family == dev2->family;
    ret &= dev1->vendor == dev2->vendor;

    return ret;
}

StorageCacheImpl*
getStorageCache(TargetDevice* tdev, bool force)
{
    unsigned int k;
    StorageCacheImpl* curCache = NULL;

    assert(storageCacheArray != NULL);
    assert(storageCacheLock != NULL);

    for (k = 0; k < storageCacheArrayCount; ++k) {
        if (isDeviceEQ(&tdev->ident, &storageCacheArray[k].devIdent) ) {
            curCache  = &storageCacheArray[k];
        }
    }

    assert (curCache != NULL);

    // Read data from file can be only one thread
    // Work with the cached data can all threads in parallel
    if (!curCache->isInit) {
        mutexLock(storageCacheLock);                // LOCK

        if (!curCache->isInit) {
            curCache->isPopulate = false;

            if (initReadingData(curCache, tdev)) {
                loadDataFromFile(curCache);
            }

            curCache->isInit = true;
        }
        mutexUnlock(storageCacheLock);              // UNLOCK
    }

    // if storage cashe is empty then return NULL
    if (!(curCache->isPopulate || force)) {
        curCache = NULL;
    }

    return curCache;
}

unsigned int
getPlatforms(cl_platform_id **platforms)
{
    cl_int ret;
    cl_uint numberPlatform;

    ret = clGetPlatformIDs(0, NULL, &numberPlatform);

    if (ret != CL_SUCCESS || numberPlatform == 0) {
        return  0;
    }

    *platforms = calloc(numberPlatform, sizeof(cl_platform_id));

    if (*platforms == NULL) {
        return 0;
    }

    ret = clGetPlatformIDs(numberPlatform, *platforms, NULL);
    return numberPlatform;
}

void
initStorageCache(void)
{
    cl_uint numberPlatform = 0;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    StorageCacheImpl* cur = NULL;
    cl_int ret;

    unsigned int deviceCount = 0;
    unsigned int i, j, k;

    assert (storageCacheLock == NULL);
    assert (storageCacheArray == NULL);
    assert (storageCacheArrayCount == 0);

    storageCacheLock = mutexInit();
    numberPlatform = getPlatforms(&platforms);

    if (numberPlatform ==0) {
        return;
    }

    for (i =0; i < numberPlatform; ++i) {
        cl_uint dc;

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &dc);
        if (ret == CL_SUCCESS) {
            deviceCount += dc;
        }
    }

    storageCacheArray = calloc(deviceCount, sizeof(*storageCacheArray));

    for (i =0; i < numberPlatform; ++i) {
        cl_uint dc;

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &dc);
        if (ret != CL_SUCCESS) {
            continue;
        }

        devices = calloc(dc, sizeof(*devices));

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, dc, devices, NULL);

        for (j = 0; j < dc; ++ j) {
            TargetDevice td;
            bool isUnique = true;

            td.id = devices[j];
            identifyDevice(&td);

            for (k = 0; k < storageCacheArrayCount; ++k) {
                if (isDeviceEQ(&td.ident, &storageCacheArray[k].devIdent) ) {
                    isUnique = false;
                }
            }

            if (isUnique) {
                cur = &storageCacheArray[storageCacheArrayCount];

                clearPatternsNumber(cur->functionInfo);
                cur->isInit = false;
                cur->devIdent.chip = td.ident.chip;
                cur->devIdent.family = td.ident.family;
                cur->devIdent.vendor = td.ident.vendor;

                storageCacheArrayCount++;
            }
        }
        free(devices);
    }
    free (platforms);
}
