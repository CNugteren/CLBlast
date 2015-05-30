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


#include "storage_data.h"
#include "assert.h"

BlasParamInfo*
findParam(
    StorageCacheImpl* cacheImpl,
    const char* pattName,
    const DataType dt,
    const KernelExtraFlags kflag,
    int dim)
{
    unsigned int func;
    BlasFunctionInfo *functionInfo = cacheImpl->functionInfo;
    //unsigned int mask[BLAS_FUNCTIONS_NUMBER];

    //initMask(mask);
    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func) {
        unsigned int patt;

        BlasFunctionInfo* bFunc = &functionInfo[func];
        for (patt =0; patt < bFunc->numPatterns; ++ patt) {
            unsigned int extra;

            BlasPatternInfo* bPatt = &bFunc->pattInfo[patt];
            if (strcmp(bPatt->name, pattName) == 0) {
                KernelExtraFlags flag  = kflag & bFunc->maskForTuningsKernel;
                for (extra =0; extra < bPatt->numExtra; ++ extra) {
                    BlasExtraInfo* bExtra = &bPatt->extra[extra];
                    if (bExtra->dtype == dt && bExtra->flags == flag)
                    {
                        unsigned int param;
                        BlasParamInfo* bestParam    = NULL;
                        unsigned int bestDimDelta = 50000;

                        if (dim == 0) {
                            //leading dimension banks aligned case
                            bestParam =
                                &bExtra->param[BANK_ALIGNED_CASE_RECORD_IDX];
                        }
                        else {
                            for (param = 0; param < bExtra->numParam; ++param) {
                                BlasParamInfo* bParam = &bExtra->param[param];
                                unsigned int dimDelta = abs(dim - bParam->dim);

                                if (param == BANK_ALIGNED_CASE_RECORD_IDX) {
                                    continue;
                                }

                                if (dimDelta < bestDimDelta){
                                    bestDimDelta = dimDelta;
                                    bestParam    = bParam;
                                }
                            }
                        }
                        return bestParam;
                    }
                }
            }
        }
    }
    return NULL;
}


BlasPatternInfo *
getPatternInfo(StorageCacheImpl* cache, unsigned int func, unsigned int patt)
{
	BlasPatternInfo* bPatt = NULL;

	if (func != BLAS_FUNCTIONS_NUMBER) {
		BlasFunctionInfo* bFunc = &cache->functionInfo[func];

		bPatt = &bFunc->pattInfo[patt];
	}
	return bPatt;
}


void
nextPattern(StorageCacheImpl* cache, unsigned int* func, unsigned int* patt)
{
	BlasFunctionInfo* bFunc = &cache->functionInfo[*func];

	(*patt)++;
	if (bFunc->numPatterns == *patt) {
		(*func)++;
		*patt = 0;
	}
}

////////////////////////////////////////////////////////////////////////////////
bool
isValidFlagMatrix(DataType curType, unsigned int  flags)
{
    bool ret;
    // todo Make refactoring expressions.

    ret = !isComplexType(curType)
             && ( (flags & KEXTRA_CONJUGATE_A) || (flags & KEXTRA_CONJUGATE_B));
    //  The flag KEXTRA_CONJUGATE_X can be set TRUE only when the flag KEXTRA_TRANS_X is TRUE.
    ret = ret || (flags & (KEXTRA_TRANS_A | KEXTRA_CONJUGATE_A))
    		== KEXTRA_CONJUGATE_A;
    ret = ret || (flags & (KEXTRA_TRANS_B | KEXTRA_CONJUGATE_B))
    		== KEXTRA_CONJUGATE_B;

    return ret;
}

size_t
getDTypeArray(DataType * dTypes, size_t dtypeCount, DeviceInfo* defInf )
{
    if (dtypeCount < 4) {
        return 0;
    }
    if (defInf->nativeDouble) {
        if (defInf->nativeComplex) {
            dTypes[0] =  TYPE_FLOAT;
            dTypes[1] =  TYPE_COMPLEX_FLOAT;
            dTypes[2] =  TYPE_DOUBLE;
            dTypes[3] =  TYPE_COMPLEX_DOUBLE;
            dtypeCount = 4;
        }
        else {
            dTypes[0] =  TYPE_FLOAT;
            dTypes[1] =  TYPE_DOUBLE;
            dtypeCount = 2;
        }
    }
    else {
        if (defInf->nativeComplex) {
            dTypes[0] =  TYPE_FLOAT;
            dTypes[1] =  TYPE_COMPLEX_FLOAT;
            dtypeCount = 2;
        }
        else {
            dTypes[0] =  TYPE_FLOAT;
            dtypeCount = 1;
        }
    }
    return dtypeCount;
}

void
initParamData (BlasParamInfo* bParam, int dim)
{
    memset(bParam->sDim,        0, sizeof(SubproblemDim) * MAX_SUBDIMS);
    memset(&bParam->pGran,      0, sizeof(PGranularity) );
    memset(bParam->kernel,      0, sizeof(OFFSET) * MAX_CLBLAS_KERNELS_PER_STEP);
    memset(bParam->kSize,       0, sizeof(size_t)* MAX_CLBLAS_KERNELS_PER_STEP);

    bParam->time = 1e50; // any large number;
    bParam->dim  = dim;

    bParam->offset = 0;
    bParam->size   = 0;
    bParam->sstatus = SS_NOLOAD;
}

void
initExtraData(BlasExtraInfo* bExtra, DataType dTypes, unsigned int flags, DeviceInfo* di)
{
    unsigned int param;
    int func = bExtra->parent->parent->funcNo;

    assert(bExtra->param == 0);

    bExtra->dtype = dTypes;
    bExtra->flags = flags;

    if (isComplexType(dTypes)) {
        bExtra->vecLen = 2;
    }
    else {
        bExtra->vecLen = 4;
    }

    bExtra->numParam = getDimensionCount(di->tdev, func);

    bExtra->offset = 0;
    bExtra->size   = 0;
    bExtra->sstatus = SS_NOLOAD;

    bExtra->param = calloc( bExtra->numParam, sizeof(BlasParamInfo));
    for (param = 0; param < bExtra->numParam; ++param) {
        BlasParamInfo* bParam = &bExtra->param[param];
        initParamData(bParam, getDimension(param, bExtra->dtype, di, func));
     }
}

int
genExtraDatasForPattern(
    BlasPatternInfo* bPatt,
    unsigned int tuningsMask,
    unsigned int uniqueMask,
    DeviceInfo* defInf)
{
    size_t dtypeCount;
    size_t ndt;
    unsigned int  flags;
    unsigned int index;
    DataType  dTypes[4];
    BlasExtraInfo* extra;
    BlasFunctionInfo* bFunc;
    unsigned int extraCount;

    bFunc = bPatt->parent;
    extra = bPatt->extra;
    extraCount = bPatt->numExtra;
    bPatt->numTuneExtra = 0;

    dtypeCount = getDTypeArray(dTypes, 4, defInf);
    index = 0;
    for (flags = 0; flags <= uniqueMask; flags++) {
        unsigned int m = flags & (~uniqueMask);
        if (!m){
            for (ndt = 0; ndt < dtypeCount; ++ndt) {
                DataType curType = dTypes[ndt];
                if ( bFunc->isValidFlag != NULL
                     && bFunc->isValidFlag(curType, flags)) {
                    continue;
                }

                if (extra != NULL) {
                    unsigned int tm;
                    if (index == extraCount) {
                        return index;
                    }

                    extra[index].parent = bPatt;
                    initExtraData(&extra[index], dTypes[ndt], flags, defInf);
                    tm = flags & (~tuningsMask);
                    extra[index].isUseForTunning = tm == 0;
                    if (extra[index].isUseForTunning) {
                        bPatt->numTuneExtra++;
                    }
                }
                ++index;
            }
        }
        else {
            m = (m&(m-1))^m;
            flags = flags + m - 1;
        }
    }
    return index;
}

void
initPatternData (BlasPatternInfo*  bPatt, DeviceInfo* defInf)
{
    unsigned int tuningsMask = bPatt->parent->maskForTuningsKernel;
    unsigned int uniqueMask = bPatt->parent->maskForUniqueKernels;

    assert(bPatt->numExtra == 0);
    assert(bPatt->extra == 0);

    bPatt->numExtra = genExtraDatasForPattern(bPatt, tuningsMask,
                uniqueMask, defInf);

    bPatt->offset = 0;
    bPatt->size   = 0;
    bPatt->sstatus = SS_NOLOAD;

    bPatt->extra = calloc( bPatt->numExtra, sizeof(BlasExtraInfo));
    genExtraDatasForPattern(bPatt, tuningsMask, uniqueMask, defInf);
}

void
initFuncData (BlasFunctionInfo* bFunc, DeviceInfo* defInf)
{
    unsigned int patt;
    bFunc->isValidFlag = isValidFlagMatrix;

    if (bFunc->initFunctionInfo != NULL) {
        bFunc->initFunctionInfo(bFunc);
    }

    for (patt = 0 ; patt < bFunc->numPatterns; ++patt) {
        BlasPatternInfo*  bPatt = &bFunc->pattInfo[patt];
        bPatt->parent = bFunc;
        bPatt->name = bFunc->pattern[patt].name;
        bPatt->pattNo = patt;
        initPatternData (bPatt, defInf);
    }
}

void
initCacheData (BlasFunctionInfo* bFuncs, DeviceInfo* defInf)
{
    unsigned int func;

    for (func=0; func < BLAS_FUNCTIONS_NUMBER; ++func) {
        BlasFunctionInfo* bFunc = &bFuncs[func];
        bFunc->funcNo = func;
        initFuncData(bFunc, defInf);
    }
}

void
destroyParamData(BlasParamInfo* bParam)
{
    int k;

    for (k=0; k < MAX_CLBLAS_KERNELS_PER_STEP; ++k) {
        bParam->kSize[0] = 0;
    }
}

void
destroyExtraData(BlasExtraInfo* bExtra)
{
    unsigned int param;

    if (bExtra == NULL) {
        return;
    }

    for (param = 0; param < bExtra->numParam; ++param) {
        BlasParamInfo* bParam = &bExtra->param[param];
        destroyParamData(bParam);
    }
    free(bExtra->param);
}

void
destroyPatternData(BlasPatternInfo*  bPatt)
{
    unsigned int extra;

    for (extra = 0 ; extra < bPatt->numExtra; ++extra){
        BlasExtraInfo*  bExtra = &bPatt->extra[extra];
        destroyExtraData (bExtra);
    }
    free (bPatt->extra);
}

void
destroyFuncData(BlasFunctionInfo* bFunc)
{
    unsigned int patt;

    for (patt = 0 ; patt < bFunc->numPatterns; ++patt) {
        BlasPatternInfo*  bPatt = &bFunc->pattInfo[patt];
        destroyPatternData (bPatt);
    }
}

void
destroyData(BlasFunctionInfo* fInfo)
{
    unsigned int func;

    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func){
        destroyFuncData( &fInfo[func]);
    }
}

