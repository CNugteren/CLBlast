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


#include <math.h>       // sqrt()

#include "toolslib.h"
#include "clblas_stddef.h"
#include "storage_data.h"


unsigned int  DimensionsArrayL3[]=  {7, 13, 32, 48, 64, 64};
unsigned int  DimensionsArrayL2[]=  {768/4, 1792/4, 3328/4, 5248/4, 6784/4,
                                     3*1024/4};

int getDimensionCount(TargetDevice* tdev, int func)
{
	(void)tdev;
	(void)func;
	return DIMARRAYCOUNT;
}

//
dimension
getDimensionID(TargetDevice* tdev, int func, size_t M, size_t N, size_t K)
{
    (void)tdev;
	(void)func;
	(void)M;
	(void)N;
	(void)K;

	return 0;
}

#include <assert.h>

unsigned int
getDimension(int idx, DataType dt, DeviceInfo *devInfo, int func)
{
    unsigned int dim;
    // bas - banks aligned size, in bytes, should be
    // number of banks * number of channels * bytes per channel
    // here it is set to 8*256 = 2048 = 512 floats
    size_t bas = 8*256;
    unsigned int tsize;

    // The minimum step for which the tails are not.
    size_t noTailStep;

    float step;

    (void) func;

    tsize = dtypeSize(dt);
    noTailStep = 256 * sizeof(cl_float) / tsize;

    // !!! DEBUG
    //printf("[%s, line %d]: devInfo->globalSize = %lu\n",
    //        __func__, __LINE__, devInfo->globalSize);

    /*
     * Skip the smallest size, it does not provide sufficient
     * device payload anyway
     */
    //i = (idx == DIMARRAYCOUNT - 1) ? (DIMARRAYCOUNT - 1) : (idx + 1);

//    dim = DimensionsArray2[i];
//    dim *= devInfo->nrComputeUnits;
    step = (float)umin(devInfo->nrComputeUnits, funcBlasLevel(func) == 2 ? 1 : 24);

    switch (dt) {
        case TYPE_FLOAT:
            step *= 4;
            break;
        case TYPE_DOUBLE:
        case TYPE_COMPLEX_FLOAT:
            step = 2.8f * step;
            break;
        case TYPE_COMPLEX_DOUBLE:
#if defined(_WIN32) && defined(FORCE_BSOD)
            if (func != CLBLAS_SYRK && func != CLBLAS_SYR2K) {
                step *= 2;
            }
#else
            step *= 2;
#endif
            break;
    }

    if (funcBlasLevel(func) == 2) {
        dim = (unsigned int)(step * DimensionsArrayL2[idx]);
    }
    else {
        dim = (unsigned int)(step * DimensionsArrayL3[idx]);
    }

    if (dim * dim * tsize > devInfo->maxMemAllocSize) {
        dim = (unsigned int)sqrt((double)(devInfo->maxMemAllocSize / tsize));
    }

    assert(devInfo->globalSize);
    if (dim * dim * tsize >= devInfo->globalSize / 3) {
        dim = (unsigned int)sqrt((double)devInfo->globalSize / 3 / tsize);
    }

    dim = (unsigned int)roundUp(dim - (noTailStep/2), noTailStep);
    if (idx == BANK_ALIGNED_CASE_RECORD_IDX) {
        // force size to be banks aligned
        if (dim * dtypeSize(dt) % bas != 0) {
            dim = (unsigned int)roundUp(dim, bas / dtypeSize(dt));
        }
    }
    else {
        // avoid banks aligned size adding maximal base dimension
        if (dim * dtypeSize(dt) % bas == 0) {
//            dim += DimensionsArray2[DIMARRAYCOUNT - 1] /
//                   (dtypeSize(dt) / sizeof(cl_float));
            dim += (unsigned int)noTailStep;
        }
    }
	return dim;
}
