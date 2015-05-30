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


#ifndef TRXM_BUFS_COMMON_H_
#define TRXM_BUFS_COMMON_H_

#include "blas_kgen.h"
#include "gen_helper.h"
#include "blas_funcs.h"

/*
 * COMMON NOTES:
 * To use the functions the caller must guarantee kernel argument
 * naming and subproblem dimensions independent on the side.
 * That means size of A must be named as 'M'. The 'y' field of dimensions
 * must be a step over rows of the matrix A in case of the left side, and over
 * columns of the matrix otherwise. Similarly the 'x' field must be a step
 * over columns of the matrix B in case of the left side, and over rows of
 * the matrix otherwise. Both 'A' and 'B' are passed in global buffers.
 */

void
declareTrxmKernel(
    struct KgenContext *ctx,
    DataType dtype,
    const PGranularity *pgran,
    KernelExtraFlags kflags,
    BlasFunctionID funcID,
    const char *nameSuffix,
    bool declareC,
    bool restrictPointers);

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

void
genUpdateTrxmResult(
    struct KgenContext *ctx,
    const SubproblemDim *dims,
    char *fnName,
    char *genericFnName,
    KernelExtraFlags kflags);

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

/*
 * Move matrix B start pointer according to offsetM, offsetN.
 */
void
genTrxmBMatrShift(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    bool useC);

void
fixupTrxmKargs(CLBlasKargs *kargs);

/* Setting to zero upper/lower triangle elements and optionally set diagonal
 * elements to one after fetching */
int
genTrxmPostFetchZero(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv);

#endif /* TRXM_BUFS_COMMON_H_ */
