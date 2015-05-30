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


#include <stdlib.h>
#include <assert.h>

#include <blas_mempat.h>
#include <blas_kgen.h>
#include <clblas-internal.h>
#include <kern_cache.h>
#include <blas_funcs.h>

#include "fileio.h"
#include "toolslib.h"

#include "tune.h"
#include "subdim.h"
#include <math.h>

#if defined(_MSC_VER)
#define fmin min
#define fmax max
#endif

#define isLdsUsed(pattern)                                          \
    (checkMatrixMemLevelSet(pattern, MATRIX_A, CLMEM_LEVEL_LDS) ||  \
     checkMatrixMemLevelSet(pattern, MATRIX_B, CLMEM_LEVEL_LDS))

int VISIBILITY_HIDDEN
getDataTypeSize(DataType dataType)
{
    int dataTypeSize = 0;

    switch (dataType) {
        case TYPE_FLOAT:
            dataTypeSize = 4;
            break;
        case TYPE_DOUBLE:
        case TYPE_COMPLEX_FLOAT:
            dataTypeSize = 8;
            break;
        case TYPE_COMPLEX_DOUBLE:
            dataTypeSize = 16;
            break;
    }
    return dataTypeSize;
}
/*
*  Checks current dimensionality on a validity
*/
bool VISIBILITY_HIDDEN
isSubDimValid(SubDimInfo* sd)
{
    int j;
    size_t wgX = sd->pgran.wgSize[0];
    size_t wgY = sd->pgran.wgSize[1];
    SubproblemDim l0 = sd->sdim[0];
    SubproblemDim l1 = sd->sdim[1];
    size_t dataTypeSize = getDataTypeSize(sd->dtype);
    size_t dataFloatSize = getDataTypeSize(TYPE_FLOAT);
    int maxRegistr = 64;
    bool ret = true;
    bool inv;
    IgnoreItem* ii = sd->first;

    // if pattern-based validation is available
    if( NULL != sd->pattern->sops->checkCalcDecomp ){

        return sd->pattern->sops->checkCalcDecomp(
            &sd->pgran,
            sd->sdim,
            2,
            sd->dtype,
            PGRAN_CHECK );
    }

    ret = ret && (l1.y >= 4*dataFloatSize/dataTypeSize);

    if (sd->blasLevel == 3) {
        if (!isMatrixAccessColMaj(sd->func, sd->flag, MATRIX_A) ||
                !isMatrixAccessColMaj(sd->func, sd->flag, MATRIX_B)) {
            /* Avoid small bwidth and big x0, y0 for cases other than
             * column major access to both matrixes */
            ret = ret && (l1.bwidth >= 4*dataFloatSize/dataTypeSize);
            ret = ret && (l0.y < 128);
            ret = ret && (l0.x < 128);
        }
    }

    if ( 0 == l1.bwidth ){
        return false;
    }
    else{
        ret = ret && ((l0.bwidth % l1.bwidth) == 0);
        ret = ret && (wgX*wgY == 64);
    }
    //ret = ret && (wgX*wgY < sd->workGroupSizes);
    //ret = ret && (wgX*wgY > 16);
    if (sd->blasLevel == 2) {
        ret = ret && (l0.y > l1.y);
    }
    else {
        ret = ret && (l0.x > l1.x);
        ret = ret && (l0.y > l1.y);
        ret = ret && (l1.x >= 4*dataFloatSize/dataTypeSize);
    }
    if (sd->is2D) {
        bool r = ret;
        ret = ret && (wgY * l1.itemX == l0.x);
        ret = ret && (wgX * l1.itemY == l0.y);
        if (r != ret) {
            return ret;
        }
    }

    if (ret && sd->isSquareBlock) {
        ret = ret &&  (l0.x == l0.y && l0.x == l0.bwidth);
    }

    //if (!(isLdsUsed(sd->pattern) || (sd->isSquareBlock && sd->nrLevel == 2))) {
    //    ret = ret &&  l0.bwidth == l1.bwidth;
    //}

    if (ret) {
        int r ;
        r = (int)(l1.x*l1.bwidth + l1.y*l1.bwidth + l1.x*l1.y);

        r = r * (int)dataTypeSize / sizeof(cl_float4);

        if (r > maxRegistr) {
            return false;
        }
    }

    if  (ret &&  sd->pattern->sops->isFitToLDS != NULL) {
        bool isFitToLDS;
        CLBlasKargs args;

        convKExtraFlagToArg(sd->flag, &args);

        isFitToLDS = sd->pattern->sops->isFitToLDS(sd->sdim, sd->dtype,
                                               sd->ldsSize, &args);
        if (!isFitToLDS)
            return false;
    }

    // Skip ignored dimension
    for (;ii != NULL; ii = ii->next) {
        inv = true;
        for(j = 0; j < V_COUNT; ++j) {
            int v1 = ii->var[j];
            int v2 = get(&sd->var[j]);
            if (v1 == -1) {
                continue;
            }
            if (v1 == v2) {
                continue;
            }
            inv = false;
            break;
        }
        if (inv) {
            ret = false;
        }
    }

    return ret;
}

/*
 * Set invalid SubDimension.
 * Invalid SubDimensions will be skipped.
 */
void VISIBILITY_HIDDEN
setInvalid(SubDimInfo* sdi, int l0x, int l0y, int l0w,
                            int l1x, int l1y, int l1w)
{
    IgnoreItem* ii = malloc(sizeof(IgnoreItem));
    ii->var[V_L0_X]  = l0x;
    ii->var[V_L0_Y]  = l0y;
    ii->var[V_L0_BW] = l0w;
    ii->var[V_L1_X]  = l1x;
    ii->var[V_L1_Y]  = l1y;
    ii->var[V_L1_BW] = l1w;
    ii->next = sdi->first;
    sdi->first = ii;
}

void VISIBILITY_HIDDEN
initVector(SubDimInfo* sd)
{                //0 1 2 3  4  5  6   7   8   9   10   11
    int dim  [] = {1,2,4,8,16,32,64,128,256,512,1024,2048, 4096};
    if (sd->blasLevel == 2 ) {
        setVariable(sd, V_L0_X,  1, &dim[0]);
        setVariable(sd, V_L0_Y,  6, &dim[4]);
        setVariable(sd, V_L0_BW, 10, &dim[0]);
        setVariable(sd, V_L1_X,  1, &dim[0]);
        setVariable(sd, V_L1_Y,  6, &dim[1]);
        setVariable(sd, V_L1_BW, 6, &dim[0]);
    }
    else {
        setVariable(sd, V_L0_X,  4, &dim[4]);
        setVariable(sd, V_L0_Y,  4, &dim[4]);
        setVariable(sd, V_L0_BW, 6, &dim[0]);
        setVariable(sd, V_L1_X,  6, &dim[0]);
        setVariable(sd, V_L1_Y,  6, &dim[0]);
        setVariable(sd, V_L1_BW, 6, &dim[0]);
    }
}

void VISIBILITY_HIDDEN
initKNMVector(
        SubDimInfo* sd,
        unsigned int baseDim,
        unsigned int* K,
        unsigned int* N,
        unsigned int* M
        )
{
    if (sd->blasLevel == 2 ) {
        *K = 1;
        *N = baseDim * 2;
        *M = baseDim * 2;
    } else
    {
        *K = baseDim;
        *N = baseDim;
        *M = baseDim;
    }
}

int VISIBILITY_HIDDEN
get(SubDimItem* sd)
{
    return sd->data[sd->curId];
}

void VISIBILITY_HIDDEN
calcPGranularity (SubDimInfo* sd)
{
    SubproblemDim* dim = sd->sdim;
    PGranularity* pgran = &sd->pgran;
    //int level = sd->cuLevel;

    pgran->wgDim = 2;
    pgran->wfSize = 64;
	pgran->maxWorkGroupSize = sd->workGroupSizes;

    // if pattern provides granularity calculation
    // call the pattern function
    if( NULL != sd->pattern->sops->checkCalcDecomp ){

        sd->pattern->sops->checkCalcDecomp(
            pgran,
            dim,
            2,
            sd->dtype,
            PGRAN_CALC );
    }
    else{
        pgran->wgSize[1] =  (unsigned int)(dim[0].x / dim[1].itemX);
        pgran->wgSize[0] =  (unsigned int)(dim[0].y / dim[1].itemY);

        if (!sd->is2D) {
            pgran->wgDim = 1;
            pgran->wgSize[0] *= pgran->wgSize[1];
            pgran->wgSize[1] = 1;
        }
    }

}

void VISIBILITY_HIDDEN
calcParam(SubDimInfo* sd)
{
    SubproblemDim* dim = sd->sdim;

    int dataTypeSize = getDataTypeSize(sd->dtype);

    memset(dim, 0, sizeof(sd->sdim));

    dim[0].x      = get(&sd->var[V_L0_X]);
    dim[0].itemX  = get(&sd->var[V_L0_X]);
    dim[0].y      = get(&sd->var[V_L0_Y]);
    dim[0].itemY  = get(&sd->var[V_L0_Y]);
    dim[0].bwidth = get(&sd->var[V_L0_BW]);

    dim[1].x      = get(&sd->var[V_L1_X]);
    dim[1].itemX  = get(&sd->var[V_L1_X]);
    dim[1].y      = get(&sd->var[V_L1_Y]);
    dim[1].itemY  = get(&sd->var[V_L1_Y]);
    dim[1].bwidth = get(&sd->var[V_L1_BW])
            / (dataTypeSize / getDataTypeSize(TYPE_FLOAT));

    if (funcHasTriangMatrix((BlasFunctionID)sd->func) && !sd->is2D) {
        dim[0].itemY  = SUBDIM_UNUSED;
    }

    if (sd->blasLevel == 2) {
        size_t xBlocks;

        xBlocks = dim[0].x / dim[1].x;
        dim[0].x = 1;
        dim[1].itemX = 1;
        dim[1].x = 1;
        if( NULL == sd->pattern->sops->checkCalcDecomp ){
            dim[0].bwidth = dim[1].bwidth * xBlocks;
        }
    }

    calcPGranularity(sd);
}

bool VISIBILITY_HIDDEN
next(SubDimItem var[V_COUNT])
{
    int i = V_COUNT - 1;
    bool next;
    do {
        next = false;
        var[i].curId ++;
        if (var[i].curId >= var[i].maxId) {
            var[i].curId = 0;
            next = true;
            -- i;
        }
    } while (next && i >= 0 );
    return (next && i < 0);
}

void VISIBILITY_HIDDEN
findValidSubdimInit(SubDimInfo* sd)
{
    bool n = false;
    do {
        n = false;
        calcParam(sd);
        sd->valid = sd->isValid(sd);
        if (!sd->valid) {
            n = !next(sd->var);
            sd->valid = false;
        }
    } while (n);
}

bool
nextSubdimElem(SubDimInfo* sd)
{
    bool n = false;

    // !!! DEBUG
    if (sd->count > 500) {
        abort();
    }

    sd->count ++;
    if (sd->valid == false) {
        return false;
    }

    if (sd->init != NULL) {
        sd->valid = false;
        n = !next(sd->var);
        if (n)
            findValidSubdimInit(sd);
    }
    return sd->valid;
}

/*
 * The variant included of the group.
 */
bool
isMemberOfGroup(GroupStatInfo* gsi,  Variant* vi)
{
    bool res = true;
    res &= gsi->var[V_L0_X]  == -1 || vi->var[V_L0_X]  == gsi->var[V_L0_X];
    res &= gsi->var[V_L0_Y]  == -1 || vi->var[V_L0_Y]  == gsi->var[V_L0_Y];
    res &= gsi->var[V_L0_BW] == -1 || vi->var[V_L0_BW] == gsi->var[V_L0_BW];
    res &= gsi->var[V_L1_X]  == -1 || vi->var[V_L1_X]  == gsi->var[V_L1_X];
    res &= gsi->var[V_L1_Y]  == -1 || vi->var[V_L1_Y]  == gsi->var[V_L1_Y];
    res &= gsi->var[V_L1_BW] == -1 || vi->var[V_L1_BW] == gsi->var[V_L1_BW];
    return res;
}

/*
 * Calculate the minimum expected run time.
 */

double
calcMinExpectedTimeForGroup(GroupStatInfo* gsi)
{
    /*
     * K_INCREASE - Expected range of time values in the group
     * K_GLOBAL -
     */
    const double K_INCREASE = 1.5;
    const double K_GLOBAL = 0.97;

    /* Number of variants in group */
    double m = gsi->allCount;
    /* Number of variants in group for whom time is measured*/
    double i = gsi->count;

    /*
     *  k - Reflects the expected spread of values in the group,
     *  depending on the number of measurements
     *  decreases with increasing i
     *  if i == 1 then k K_INCREASE
     *  if i == m then k = 1
     */

    double ki = 1/ ((K_INCREASE + K_INCREASE/(m+i) -1)/(i) + (m-K_INCREASE)/(m+1));
    double averageTime = (gsi->allTime / m);

    /*
     * kdelta - Reflects the expected spread of values in the group,
     * depending on the spread of values of the measured variations
     */

    double kdelta = (gsi->minTime*3)/((gsi->minTime*2) + averageTime);
    double t = K_GLOBAL * kdelta * ki * gsi->minTime;

    /*
     * Select the minimum time between the minimum time for the current group
     * and the minimum time for the previous groups
     */
    return t;
}

bool
nextSubdim(SubDimInfo* sd, int maxParam, double time)
{
    int i;
    int j;
    double minW = -5000;
    int vari = 0;
    double midTime;
    int iCount = 0;
    double maxTime;
    const int MAX_WEIGHT = 99;

    Variant* v0 = sd->curVar;   // Current variant
    Variant* varNext = NULL;    // Next Variant

    if (sd->count >= maxParam) {
        return false;
    }

    if (sd->returnAll) {
        bool ret = nextSubdimElem (sd);
        calcParam(sd);
        sd->curVarID = sd->count;
        return ret;
    }

    v0->time = time;
    sd->sumTime += time;

    midTime = sd->sumTime/(sd->count + 1);

    if (time > 0)  {
        sd->minTime = fmin(sd->minTime, (float)time);
    }

    maxTime = fmax(2.1*midTime - sd->minTime,  sd->minTime*5);

    /* Initialize all groups */
    for (j = 0; j < sd->infoCount; j++ ) {
        GroupStatInfo* si = &sd->info[j];
        si->allTime = 0;
        si->count   = 0;
        si->minTime = 1e9;
    }

    /* Calculate an estimate for the groups */
    for (i = 0; i < sd->varCount; ++i) {
        Variant* vi = &sd->allVariant[i];
        /* If time for variant is measured*/
        if (vi->time > 0) {
            for (j = 0; j < sd->infoCount; j++ ) {
                GroupStatInfo* gsi = &sd->info[j];
                // For each group, if variant is member this group
                if (isMemberOfGroup(gsi, vi)) {
                    gsi->minTime = fmin(gsi->minTime, vi->time);
                    gsi->allTime += fmin(vi->time, maxTime);
                    gsi->count ++;
                    gsi->minTime = calcMinExpectedTimeForGroup(gsi);
                }
           }
        }
        vi->minTime = 0;
        vi->maxTime = 5000;
        vi->weight  = MAX_WEIGHT;
    }

    /*
     * Calculate the estimate run-time variant
     */
    for (i = 0; i < sd->varCount; ++i) {
        Variant* vi = &sd->allVariant[i];

        vi->weight = MAX_WEIGHT;
        if (vi->time == 0) {
            double kgroup = 1.0;

            for (j = 0; j < sd->infoCount; j++ ) {
                GroupStatInfo* gsi = &sd->info[j];
                // if the variant included of the group
                if (isMemberOfGroup(gsi, vi)) {
                    if (gsi->count > 0) {
                        vi->minTime = fmax(vi->minTime, gsi->minTime);
                        vi->weight  = sd->minTime/vi->minTime;
                    }
                    else {
                        /*
                         * If variant don't included of the group
                         * then to reduce estimated time
                         */
                        kgroup *= 1.1;
                    }
                }
            }
            vi->weight *= kgroup;
            vi->minTime /= kgroup;
        }
    }

    /* Find variant with minimal run time */

    for (i = 0; i < sd->varCount; ++i)
    {
        Variant* vi = &sd->allVariant[i];
        if (vi->time == 0 && vi->weight >= 0.01 ) {
            iCount ++;

            if (minW < vi->weight) {
                minW = vi->weight;
                varNext = vi;
                vari = i;
            }
        }
    }

    //
    if (varNext == NULL) {
        return false;
    }

    sd->curVar =  varNext;
    sd->curVarID = vari;
#ifdef TEST_LOG
    printf ("%4d %6.2f [%6.2f:%5.2f ]",iCount, sd->minTime,
            sd->curVar->minTime, sd->curVar->weight);
#endif

    for(j = 0; j < V_COUNT; ++j) {
        sd->var[j].curId = varNext->var[j];
    }

    calcParam(sd);
    sd->count++;
    return true;
}

void
resetSubdim(SubDimInfo* sd)
{
    int i;
    for (i=0; i< V_COUNT; ++i) {
        sd->var[i].curId = 0;
    }

    sd->count = 0;

    sd->valid = false;
    if (sd->init != NULL) {
        sd->init(sd);
        findValidSubdimInit(sd);

        assert(sd->valid);
    }
}

/*
 * Groups variants in nonzero parameters.
 *
 * Example: l0x = 1 and remaining parameters = 0;
 * At different variants the parameter l0x accepts values 16, 32, 64.
 * At the first stage creates are 3 groups (a set of groups).
 * At the second stage all variants are arranged on these groups.
 *
 * The each variant included one group of the set of group.
 * The each variant included in each set of group.
 * In set of group can be only one group
 */

void setGroup(SubDimInfo* sd,
         int l0x, int l0y, int l0w,
         int l1x, int l1y, int l1w,
         int pg)
{
    int i, j;
    int start = sd->infoCount;
    int end   = sd->infoCount;

    (void) pg;

    //For each variant
    for (i = 0; i < sd->varCount; ++i) {
        Variant* vi = &sd->allVariant[i];
        int  id = -1;
        // For each group of the set of group
        for (j = start; j < end; j++ ) {
            bool bj = true;
            bj &= l0x == 0 || vi->var[V_L0_X] == sd->info[j].var[V_L0_X];
            bj &= l0y == 0 || vi->var[V_L0_Y] == sd->info[j].var[V_L0_Y];
            bj &= l0w == 0 || vi->var[V_L0_BW] == sd->info[j].var[V_L0_BW];
            bj &= l1x == 0 || vi->var[V_L1_X] == sd->info[j].var[V_L1_X];
            bj &= l1y == 0 || vi->var[V_L1_Y] == sd->info[j].var[V_L1_Y];
            bj &= l1w == 0 || vi->var[V_L1_BW] == sd->info[j].var[V_L1_BW];
            // if the variant belongs to group
            if (bj) {
                id = j;
                break;
            }
        }
        /*
         * if the variant doesn't belong to any group create new group
         */

        if (id == -1) {
            sd->info[end].var[V_L0_X]  = (l0x == 1)? vi->var[V_L0_X]  : -1;
            sd->info[end].var[V_L0_Y]  = (l0y == 1)? vi->var[V_L0_Y]  : -1;
            sd->info[end].var[V_L0_BW] = (l0w == 1)? vi->var[V_L0_BW] : -1;
            sd->info[end].var[V_L1_X]  = (l1x == 1)? vi->var[V_L1_X]  : -1;
            sd->info[end].var[V_L1_Y]  = (l1y == 1)? vi->var[V_L1_Y]  : -1;
            sd->info[end].var[V_L1_BW] = (l1w == 1)? vi->var[V_L1_BW] : -1;
            sd->info[end].pg = 0;

            sd->info[end].allTime = 0;
            sd->info[end].allCount = 1;

            end++;
            sd->infoCount++;
        }
        else {
            sd->info[id].allCount++;
        }
    }
}

void
initSubDimInfo(SubDimInfo* sd,
               MemoryPattern* mempatt,
               DeviceInfo* devinfo,
               unsigned int func,
               unsigned int patt,
               DataType dtype,
               KernelExtraFlags flag)
{
    int i = 0;

    memset(sd, 0, sizeof(SubDimInfo));

    sd->func = func;
    sd->patt = patt;
    sd->dtype = dtype;
    sd->flag  = flag;
    sd->pattern = mempatt;
    sd->first = NULL;

    sd->is2D  = (sd->pattern->sops->getFlags() & SF_WSPACE_2D)?true:false;
    sd->isSquareBlock = ((sd->pattern->sops->getFlags() &
                          SF_TOP_INPUT_SQUARE_BLOCKS) != 0);
    sd->blasLevel = funcBlasLevel(sd->func);
    sd->nrLevel   = sd->pattern->nrLevels;

    sd->ldsSize = devinfo->ldsSize;
    sd->workGroupSizes = devinfo->workGroupSizes;

    // Virtual function
    sd->isValid = isSubDimValid;
    sd->init = initVector;

    resetSubdim(sd);

    i = 0;
    do {
        i++;
    } while (nextSubdimElem(sd));
    sd->allVariant = malloc(i* sizeof(Variant));

    resetSubdim(sd);
    sd->varCount = i;

    for (i = 0; i < sd->varCount; ++i) {
        int j;
        int gpx;
        int gpy;

        for(j = 0; j < V_COUNT; ++j) {
            sd->allVariant[i].var[j]  = sd->var[j].curId;
        }

        sd->allVariant[i].minTime = 0.0;
        sd->allVariant[i].probableTime = 0.0;
        sd->allVariant[i].maxTime = 5000.0;
        sd->allVariant[i].weight = 10;
        sd->allVariant[i].time = 0;

        gpx = get(&sd->var[V_L0_X])/ get(&sd->var[V_L1_X]);
        gpy = get(&sd->var[V_L0_Y])/ get(&sd->var[V_L1_Y]);
        sd->allVariant[i].pg =  gpx * 1000 + gpy;

        nextSubdimElem(sd);
    }
    resetSubdim(sd);

    sd->minTime = 9999;
    sd->curVar = &sd->allVariant[0];
    sd->curVarID = 0;


    // Initializing group
    sd->infoMaxCount = 5000;
    sd->infoCount  = 0;
    sd->info = malloc(sd->infoMaxCount * sizeof(GroupStatInfo) );

    //           L0       L1       PG
    //           x  y  w  x  y  w
    setGroup(sd, 1, 1, 0, 0, 0, 0, 0);
    setGroup(sd, 1, 1, 1, 0, 0, 0, 0);
    setGroup(sd, 0, 0, 0, 1, 1, 1, 0);
    setGroup(sd, 1, 1, 0, 1, 1, 0, 0);
}

void
setVariable(struct SubDimInfo* sdi, SubDimVariable var, int dcount, int* dim)
{
    size_t size =  dcount*sizeof(int);

    sdi->var[var].curId = 0;
    sdi->var[var].maxId = dcount;

    if (sdi->var[var].data != NULL) {
        free (sdi->var[var].data);
        sdi->var[var].data = NULL;
    }
    sdi->var[var].data = malloc(size);
    memcpy(sdi->var[var].data, dim, size);
}


