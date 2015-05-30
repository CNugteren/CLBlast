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


#ifndef TRXM_COMMON_LEGACY_H_
#define TRXM_COMMON_LEGACY_H_

#include "../gen_helper.h"

/*
 * Declare local variables for LDS based version
 * of TRXM kernels.
 *
 * It provides the names typical for another generators as well:
 *
 * lid, gid - local and global ID.
 * m0, k0 - top level counters over M and N
 * currM, currN - current block coordinates over M and N at the top level
 * tempA, tempB - blocks of matrix A and B located in the local memory
 * tempC - block of matrix C located in the local memory; declared if
 *      the 'useLocalC' argument is set
 * c - matrix C tile located in registers; declared if the 'useLocalC'
 *      argument is not set
 * x, y - auxiliary variables to evaluate size of read/write blocks
 *
 * TRXM specific variables:
 *
 * startM, endM - starting and end coordinate over rows a kernel can access
 */
void
declareLdsBasedTrxmVariables(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims,
    const PGranularity *pgran,
    bool useLocalC);

/*
 * NOTE: the all following functions generate a code
 *       using local variables declared with the
 *       'declareTrxmLocalVariables' function
 */

void
genPrepareTrxmBlockA(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags,
    const char *nameM);

void
genPrepareTrxmBlockB(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags);

/*
 * Triangulate matrix block. The decision to triangulate is
 * made based on the current coordinates.
 */
void
genTriangMatrBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    KernelExtraFlags kflags);


#endif /* TRXM_COMMON_LEGACY_H_ */
