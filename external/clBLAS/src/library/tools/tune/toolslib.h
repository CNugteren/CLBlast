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


#ifndef TOOLSLIB_H__
#define TOOLSLIB_H__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <defbool.h>
#include <devinfo.h>
#include <cltypes.h>

#include <granulation.h>
#include <kernel_extra.h>

// Interface to access to saved data

#define GF_SUCCESS          0
#define GF_ERROR            1
#define GF_INVALID_CACHE    2
#define GF_CORRUPT_FILE     3
#define GF_KERNEL_NOT_FOUND 4


/*
 * FIXME: It's a kludge to dedicated processing a case when matrix leading
 *        dimension is aligned on the bank size
 */
#define BANK_ALIGNED_CASE_RECORD_IDX 5


typedef int dimension;

void
initStorageCache(void);

void
destroyStorageCache(void);

int
getGranularityInfo (
    TargetDevice* tdev,
    const char* pattName,
    const DataType dt,
    const KernelExtraFlags kflag,
    dimension dim,
    SubproblemDim* sdim,
    PGranularity*
    pgran,
    double* time);

int
getKernelInfo (
    TargetDevice* tdev,
    const char* pattName,
    const DataType dt,
    const KernelExtraFlags kflag,
    dimension dim,
    unsigned char** bufer,
    size_t* sizeBufer);

int getDimensionCount(TargetDevice* tdev, int func);

dimension
getDimensionID (
    TargetDevice* tdev,
    int func,
    size_t M,
    size_t N,
    size_t K);

#endif /* TOOLSLIB_H__ */

