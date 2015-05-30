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


#ifndef XXMV_COMMON_H_
#define XXMV_COMMON_H_

#include "blas_kgen.h"
#include "gen_helper.h"

/* Fetch part of vector x into tile b */
void
genFetchX(
    struct KgenContext *ctx,
    Tile *tile,
    unsigned int vecLen,
    DataType dtype,
    const KernelVarNames *varNames,
    TileMulFlags tflags,
    KernelExtraFlags kflags);

void
setResultPos(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    const char *axVar);

void
updateResultVectorTiled(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    unsigned int vecLen,
    Tile *tile);

void
genIncPointers(
    struct KgenContext *ctx,
    KernelExtraFlags kflags);

void
genStoreLocalResult(
    struct KgenContext *ctx,
    Tile *tile,
    const char *lid);

void
genAddLocalResult(
    struct KgenContext *ctx,
    Tile *tile,
    const char *lid,
    unsigned int cLocal,
    unsigned int bStep);

/* Store partial result to private result buffer */
void
genMergeResults(
    struct KgenContext *ctx,
    Tile *result,
    Tile *source);

#endif /* XXMV_COMMON_H_ */
