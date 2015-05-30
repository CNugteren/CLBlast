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


/*
 * LDS based generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "../init.h"
#include "blas_kgen_legacy.h"
#include "gen_helper_legacy.h"
#include "../gen_helper.h"
#include "../trxm_common.h"
#include "trxm_common_legacy.h"

static CLBLASMpatExtra mpatExtra;

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static void
assignKargs(KernelArg *args, const void *params, const void *extra);

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs);

static SolverFlags
solverFlags(void);

static int
getPerf( unsigned int kflags,
    const void *args);

static SolverOps solverOps = {
    generator,
    assignKargs,
    isFitToLDS,
    getPerf,
    NULL,
    NULL,
    NULL,
    solverFlags,
    NULL, //fixupKargs
    NULL, //getDefaultDecomp
    NULL, //getDecompList
    NULL,
    NULL
};

static void
genPrepareBlockC(
    struct KgenContext *ctx,
    const ZeroFuncs *zeroFuncs)
{
    char tmp[2048];

    sprintf(tmp, "%s((__local float4*)tempC);\n", zeroFuncs->names[MATRIX_C]);
    kgenAddStmt(ctx, tmp);
}

static void
genWriteBlockB(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    KernelExtraFlags kflags)
{
    char tmp[1024];
    size_t pitch;
    const char *coordName[2] = {"currM", "currN"};
    int trb;

    trb = isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_C);
    pitch = matrBlockPitch(dim, MATRIX_C, dtype, clblasLeft);

    if (!(kflags & (KEXTRA_TAILS_N | KEXTRA_TAILS_M))) {
        sprintf(tmp, "%s((GPtr)B, (LPtr)tempC, %s, %s, ldb);\n",
                copyFuncs->write, coordName[trb], coordName[1 - trb]);
    }
    else {
        sprintf(tmp,
                "y = (currM + %lu <= M) ? %lu : M - currM;\n"
                "x = (currN + %lu <= N) ? %lu : N - currN;\n"
                "if ((y == %lu) && (x == %lu)) {\n"
                     // fast rwrite
                "    %s((GPtr)B, (LPtr)tempC, %s, %s, ldb);\n"
                "}\n"
                "else {\n"
                     // slow write
                "    %s((GPtr)B, (LPtr)tempC, %s, %s, y, x, ldb, %lu);\n"
                "}\n\n",
                dim->y, dim->y, dim->x, dim->x, dim->y, dim->x,
                copyFuncs->write, coordName[trb], coordName[1 - trb],
                copyFuncs->writeGeneric, coordName[trb],
                coordName[1 - trb], pitch);
    }

    kgenAddStmt(ctx, tmp);
}

static void
genInitCurrM(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags)
{
    char tmp[1024];

    if (isMatrixUpper(kflags)) {
        strcpy(tmp, "currM = 0;\n");
    }
    else {
        sprintf(tmp, "currM = (M - 1) / %lu * %lu;\n", dim->y, dim->y);
    }

    kgenAddStmt(ctx, tmp);
    kgenAddBlankLine(ctx);
}

static void
genInternalLoopCtl(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags)
{
    char tmp[1024];

    if (isMatrixUpper(kflags)) {
        if (!(kflags & KEXTRA_TAILS_M)) {
            sprintf(tmp, "for (k0 = M - %lu; (k0 + %lu > currM) && (k0 < M); "
                         "k0 -= %lu)",
                    dim->bwidth, dim->bwidth, dim->bwidth);
        }
        else {
            sprintf(tmp, "for (k0 = (M - 1) / %lu * %lu; k0 + %lu > currM; "
                              "k0 -= %lu)",
                    dim->bwidth, dim->bwidth, dim->bwidth, dim->bwidth);
        }
    }
    else {
        sprintf(tmp, "for (k0 = 0; (k0 < currM + %lu) && (k0 < M); "
                          "k0 += %lu)",
                dim->y, dim->bwidth);
    }

    kgenBeginBranch(ctx, tmp);
}

static void
initKernelVarNames(KernelVarNames *kvars,  KernelExtraFlags kflags)
{
    kvars->A = "A";
    kvars->B = "B";
    if (isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_A)) {
        kvars->coordA = "coordA.x";
    }
    else {
        kvars->coordA = "coordA.y";
    }
    if (isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_B)) {
        kvars->coordB = "coordB.x";
    }
    else {
        kvars->coordB = "coordB.y";
    }
    kvars->sizeM = "M";
    kvars->sizeN = "N";
    kvars->sizeK = "origM";
}

static ssize_t
generator(
    char *buf,
    size_t buflen,
    const struct SubproblemDim *subdims,
    const struct PGranularity *pgran,
    void *extra)
{
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    char tmp[2048];
    char blkmul[128];
    char updateResFn[FUNC_NAME_MAXLEN];
    char updateResGenericFn[FUNC_NAME_MAXLEN];
    CopyBufFuncs copyFuncs;
    ZeroFuncs zeroFuncs;
    DataType dtype = kextra->dtype;
    ssize_t ret;
    BlasGenSettings gset;
    BlkMulOpts mulOpts;
    size_t pitchAB, pitchC;
    bool b;
    KernelExtraFlags kflags = kextra->flags;
    const char *outTypeName;
    unsigned int nrRegs;
    bool useLocalC;
    unsigned int vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    int tra, trb;
    unsigned int l1Pans;
    char vect[2] = {'y', 'x'};

    if (pgran->wgDim != 1) {
        return -EINVAL;
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    /* Code that updates block of B matrix using local registers or use mad's
     * doesn't work on some GPUs. As a workaround use buffer in local memory
     * for unaligned matrix sizes */
    useLocalC = (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N));

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.pgran = pgran;
    gset.kextra = kextra;

    initKernelVarNames(&gset.varNames, kflags);

    // at first, generate needed declarations and auxiliary functions

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);
    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_TRMM, &gset,
                         BCHF_MATRIX_A | BCHF_MATRIX_B | BCHF_WRITE_OUTPUT);
    if (useLocalC) {
        generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                ZF_MATRIX_A | ZF_MATRIX_B | ZF_MATRIX_C);
    }
    else {
        generateUpresFuncs(ctx, CLBLAS_TRMM, &gset, updateResFn,
                           updateResGenericFn);
        generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                ZF_MATRIX_A | ZF_MATRIX_B);
    }
    kgenAddBlankLine(ctx);

    // block multiplication function
    mulOpts.aMobj = CLMEM_BUFFER;
    mulOpts.bMobj = CLMEM_BUFFER;

    if (useLocalC) {
        mulOpts.flags = BLKMUL_SKEW_COLUMN;
    }
    else {
        mulOpts.flags = BLKMUL_OUTPUT_PRIVATE | BLKMUL_SKEW_COLUMN;
    }
    // BLKMUL_MAD doesn't work here on all cards so use SEPARATE_MULADD always
    // as a workaround
    mulOpts.core = BLKMUL_SEPARATE_MULADD;
    ret = blkMulGen(ctx, subdims, dtype, &mulOpts);
    if (ret) {
        destroyKgenContext(ctx);
        return -EOVERFLOW;
    }

    kgenAddBlankLine(ctx);
    kgenGetLastFuncName(blkmul, sizeof(blkmul), ctx);

    // now, generate the kernel
    declareTrxmKernel(ctx, dtype, pgran, kflags, CLBLAS_TRMM, NULL, false,
                      false);
    ret = kgenBeginFuncBody(ctx);

    /*
     * Calculate local buffer pitches, and then insert the
     * preparative code
     */
    pitchAB = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);
    pitchC = matrBlockPitch(subdims, MATRIX_C, dtype, clblasLeft);

    getResultGPRsInfo(dtype, &subdims[1], vecLen, &nrRegs, &outTypeName);
    declareLdsBasedTrxmVariables(ctx, dtype, subdims, pgran, useLocalC);

    /*
    * B matrix is divided on panels, each work group
    * multiply such a panel on the whole matrix A.
    */
    sprintf(tmp, "currN = gid * %lu;\n", subdims->x);
    kgenAddStmt(ctx, tmp);
    genInitCurrM(ctx, subdims, kflags);
    if (((kflags & (KEXTRA_SIDE_RIGHT | KEXTRA_STARTM_NOT_ZERO)) ==
          KEXTRA_STARTM_NOT_ZERO) ||
        ((kflags & (KEXTRA_SIDE_RIGHT | KEXTRA_STARTN_NOT_ZERO)) ==
                   (KEXTRA_SIDE_RIGHT | KEXTRA_STARTN_NOT_ZERO))) {

        kgenAddStmt(ctx, "A += lda * offsetM + offsetM;\n");
    }
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    genTrxmBMatrShift(ctx, kflags, false);

    tra = isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_B);
    l1Pans = (unsigned int)subdims[0].x / (unsigned int)subdims[1].x;

    sprintf(tmp, "coordB.%c = currN + lid %% %u * %lu;\n"
                 "coordB.%c = 0;\n\n",
            vect[trb], l1Pans, subdims[1].x, vect[1 - trb]);
    kgenAddStmt(ctx, tmp);

    // loop over M
    sprintf(tmp, "for (m0 = 0; m0 < M; m0 += %lu)", subdims->y);
    kgenBeginBranch(ctx, tmp);

    sprintf(tmp, "coordA.%c = currM + lid / %u * %lu;\n"
                 "coordA.%c = 0;\n\n",
            vect[tra], l1Pans, subdims[1].y, vect[1 - tra]);
    kgenAddStmt(ctx, tmp);

    if (useLocalC) {
        genPrepareBlockC(ctx, &zeroFuncs);
    }
    else {
        // zero work item C block
        sprintf(tmp, "for (k0 = 0; k0 < %u; k0++) {\n"
                     "    c[k0] = 0;\n"
                     "}\n\n", nrRegs);
        kgenAddStmt(ctx, tmp);
    }

    /*
     * In the first pass the part without triangle blocks is processed,
     * and in the second one only triangle blocks are processed
     */
    genInternalLoopCtl(ctx, subdims, kflags);

    genPrepareTrxmBlockA(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                         kflags, "M");
    genPrepareTrxmBlockB(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                         kflags);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);

    genTriangMatrBlock(ctx, subdims, dtype, kflags);

    // and eventually multiply the blocks and update the matrix C block
    if (useLocalC) {
        sprintf(tmp, "%s(alpha, (LPtr)(tempA + (lid / %u * %lu) * %lu), \n"
                "                    (LPtr)(tempB + (lid %% %u * %lu) * %lu),\n"
                "                    (LPtr)(tempC + (lid / %u * %lu) * %lu + \n"
                "                    (lid %% %u * %lu)), lid);\n",
                blkmul, l1Pans, subdims[1].y, pitchAB,
                l1Pans, subdims[1].x, pitchAB,
                l1Pans, subdims[1].y, pitchC, l1Pans, subdims[1].x);
    }
    else {
        sprintf(tmp, "%s((LPtr)(tempA + (lid / %u * %lu) * %lu), "
                     "(LPtr)(tempB + (lid %% %u * %lu) * %lu), c, lid);\n",
                   blkmul, l1Pans, subdims[1].y, pitchAB, l1Pans,
                   subdims[1].x, pitchAB);
    }
    kgenAddStmt(ctx, tmp);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    genInternalLoopEnd(ctx);                             // loop over K
    kgenAddBlankLine(ctx);

    // write back the block, it's evaluated
    if (useLocalC) {
        genWriteBlockB(ctx, subdims, dtype, &copyFuncs, kflags);
        kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);
    }
    else {
        if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N)) {
            sprintf(tmp, "if ((coordA.%c < M) && (coordB.%c < N))",
                    vect[tra], vect[trb]);
            kgenBeginBranch(ctx, tmp);
        }

        generateResultUpdateOld(ctx, CLBLAS_TRMM, &gset, updateResFn,
                             updateResGenericFn);

        if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N)) {
           kgenEndBranch(ctx, tmp);
        }
    }

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "currM += %lu;\n", subdims[0].y);
    }
    else {
        sprintf(tmp, "currM -= %lu;\n", subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    kgenEndBranch(ctx, NULL);                                 // loop over M

    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

static void
assignKargs(KernelArg *args, const void *params, const void *extra)
{
    const CLBlasKargs *blasArgs = (const CLBlasKargs*)params;
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    int idx = 7;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
    initMemobjKarg(&args[3], blasArgs->A, NULL, 0, 0);
    initSizeKarg(&args[4], blasArgs->lda.matrix);
    initMemobjKarg(&args[5], blasArgs->B, NULL, 0, 0);
    initSizeKarg(&args[6], blasArgs->ldb.matrix);
    if (kflags & KEXTRA_STARTM_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offsetM);
    }
    if (kflags & KEXTRA_STARTN_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offsetN);
    }
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offBX);
    }
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    const CLBlasKargs *kargs = (const CLBlasKargs*)kernelArgs;
    cl_ulong size;

    size = matrBlockSize(dim, MATRIX_A, dtype, kargs->side);
    size += matrBlockSize(dim, MATRIX_B, dtype, kargs->side);
    size += matrBlockSize(dim, MATRIX_C, dtype, kargs->side);

    return (size * dtypeSize(dtype) <= ldsSize);
}

static SolverFlags
solverFlags(void)
{
    return ((unsigned int)SF_WSPACE_1D);
}

void
initTrmmLdsPattern(MemoryPattern *mempat)
{
    mempat->name = "LDS based block trmm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &solverOps;

    mpatExtra.aMset = CLMEM_LEVEL_LDS;
    mpatExtra.bMset = CLMEM_LEVEL_LDS;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

static int
getPerf( unsigned int kflags,
    const void *args)
{
    DUMMY_ARG_USAGE(kflags);
    DUMMY_ARG_USAGE(args);

    return PPERF_POOR;
}
