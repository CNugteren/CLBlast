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

#ifndef SUBGROUP_H
#define SUBGROUP_H

#include <clBLAS.h>

#include <cltypes.h>
#include <kerngen.h>
#include <mempat.h>
#include <dblock_kgen.h>

#include <blas_funcs.h>
#include <matrix_props.h>
#include "blas_kgen.h"

#include "tile.h"
#include "fetch.h"

typedef int
(*UpresProcPtr)( struct KgenContext*,
    BlasFunctionID,
    const BlasGenSettings *,
    UpdateResultFlags,
    const char *,
    const char *,
    const char *);

/**
*/
typedef struct SubgVarNames {

    const char* subgCoord;  // 2-vector of subgroup ID by X and Y
    const char* itemId;     // 2-vector of subgroup item id/subgroupID
} SubgVarNames;

/**
*/
int
mergeUpdateResult( struct KgenContext* pCtx,
    BlasFunctionID funcID,
    struct BlasGenSettings* pGSet,
    SubgVarNames* pSubgVNames,
    UpdateResultFlags upResFlags,
    UpresProcPtr upresProcPtr );

/**
*/
int
subgGetDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    void* pArgs );

#endif
