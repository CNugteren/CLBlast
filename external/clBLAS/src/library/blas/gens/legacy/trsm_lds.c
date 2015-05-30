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
 * LDS based trsm generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <matrix_dims.h>

#include "../init.h"
#include "blas_kgen_legacy.h"
#include "gen_helper_legacy.h"
#include "trsm_kgen_legacy.h"
#include "../trxm_common.h"
#include "../trsm_kgen.h"

static CLBLASMpatExtra mpatExtra;

/*
 *  template for memory object based trsm preparation part
 *  for one dimensional work space
 */
static const char *trsmPrep1D =
    "uint m0, k0;\n"
    "__local %s tempA[%lu];\n"
    "__local %s tempC[%lu];\n"
    "%s c[%u];\n"
    "int lid, gid;\n"
    "%s"                                    // groups per Panel variable
    "uint currM, currN;\n"
    "uint x, y;\n"
    "uint2 coordA, coordB;\n"
    "\n"
    "lid = get_local_id(0);\n"
    "gid = get_global_id(0) / %u;\n"       // group ID
    "\n";

static const char *readSquareBlock =
    "y = (currM + %lu <= M) ? %lu : M - currM;\n"
    "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
    "if ((y == %lu) && (x == %lu)) {\n"
    // just read with an optimized function
    "    %s((LPtr)temp%c, (GPtr)A, currM, k0, lda);\n"
    "}\n"
    "else {\n"
    "    %s((__local float4*)temp%c);\n"           // zeroing
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    %s((LPtr)temp%c, (GPtr)A, currM, k0, y, x, %lu, lda);\n"
    "}\n\n";

static const char *readSquareBlockOpt =
    // just read with an optimized function
    "%s((LPtr)temp%c, (GPtr)A, currM, k0, lda);\n";

static const char *readSquareBlockTrans =
    "y = (currM + %lu <= M) ? %lu : M - currM;\n"
    "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
    "if ((y == %lu) && (x == %lu)) {\n"
    // read and transpose with an optimized function
    "    %s((LPtr)temp%c, (GPtr)A, k0, currM, lda);\n"
    "}\n"
    "else {\n"
    "    %s((__local float4*)temp%c);\n"           // zeroing
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    // read and transpose with slow function
    "    %s((LPtr)temp%c, (GPtr)A, k0, currM, x, y, %lu, lda);\n"
    "}\n\n";

static const char *readSquareBlockTransOpt =
    // read and transpose with an optimized function
    "%s((LPtr)temp%c, (GPtr)A, k0, currM, lda);\n";

static const char *readRectBlock =
    "y = (currN + %lu <= N) ? %lu : N - currN;\n"
    "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
    "if ((y == %lu) && (x == %lu)) {\n"
    // just read with an optimized function
    "    %s((LPtr)temp%c, (GPtr)B, currN, k0, ldb);\n"
    "}\n"
    "else {\n"
    "    %s((__local float4*)temp%c);\n"           // zeroing
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    %s((LPtr)temp%c, (GPtr)B, currN, k0, y, x, %lu, ldb);\n"
    "}\n\n";

static const char *readRectBlockOpt =
    // just read with an optimized function
    "%s((LPtr)temp%c, (GPtr)B, currN, k0, ldb);\n";

static const char *readRectBlockTrans =
    "y = (currN + %lu <= N) ? %lu : N - currN;\n"
    "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
    "if ((y == %lu) && (x == %lu)) {\n"
    // read and transpose with an optimized function
    "    %s((LPtr)temp%c, (GPtr)B, k0, currN, ldb);\n"
    "}\n"
    "else {\n"
    "    %s((__local float4*)temp%c);\n"           // zeroing
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    // read and transpose with slow function
    "    %s((LPtr)temp%c, (GPtr)B, k0, currN, x, y, %lu, ldb);\n"
    "}\n\n";

static const char *readRectBlockTransOpt =
    // read and transpose with an optimized function
    "%s((LPtr)temp%c, (GPtr)B, k0, currN, ldb);\n";

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

static SolverOps solverOps = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL,
    NULL,
    NULL,
    NULL,
    solverFlags,
    NULL, //fixupArgs
    NULL, //getDefaultDecomp
   	NULL, //getDecompList
   	NULL,
   	NULL
};

static void
genZeroResult(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims)
{
    unsigned int n;
    char tmp[1024];
    unsigned int vecLen = sizeof(cl_float4) / dtypeSize(dtype);

    getResultGPRsInfo(dtype, &dims[1], vecLen, &n, NULL);

    sprintf(tmp, "for (x = 0; x < %u; x++) {\n"
                 "    c[x] = 0;\n"
                 "}\n\n", n);

    kgenAddStmt(ctx, tmp);
}

static void
genPrepareSquareBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags kflags,
    char c)
{
    char tmp[1024];
    size_t pitch;
    const char *readBlock;
    bool tra;

    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    pitch = matrBlockPitch(dim, MATRIX_A, dtype, clblasLeft);

    if (!(kflags & KEXTRA_TAILS_M)) {
        readBlock = (tra) ? readSquareBlockTransOpt : readSquareBlockOpt;
        sprintf(tmp, readBlock, copyFuncs->read[MATRIX_A], c);
    }
    else {
        readBlock = (tra) ? readSquareBlockTrans : readSquareBlock;
        sprintf(tmp, readBlock, dim->y, dim->y, dim->bwidth, dim->bwidth,
                dim->y, dim->bwidth, copyFuncs->read[MATRIX_A], c,
                zeroFuncs->names[MATRIX_A], c,
                copyFuncs->readGeneric[MATRIX_A], c, pitch);
    }
    kgenAddStmt(ctx, tmp);
}

static void
genPrepareRectBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags kflags,
    char c)
{
    char tmp[1024];
    size_t pitch;
    const char *readBlock;
    bool trb;

    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);
    pitch = matrBlockPitch(dim, MATRIX_B, dtype, clblasLeft);

    if (!(kflags & (KEXTRA_TAILS_N | KEXTRA_TAILS_M))) {
        readBlock = (trb) ? readRectBlockTransOpt : readRectBlockOpt;
        sprintf(tmp, readBlock, copyFuncs->read[MATRIX_B], c);
    }
    else {
        readBlock = (trb) ? readRectBlockTrans : readRectBlock;
        sprintf(tmp, readBlock, dim->x, dim->x, dim->bwidth, dim->bwidth,
                dim->x, dim->bwidth, copyFuncs->read[MATRIX_B], c,
                zeroFuncs->names[MATRIX_B], c,
                copyFuncs->readGeneric[MATRIX_B], c, pitch);
    }
    kgenAddStmt(ctx, tmp);
}

static void
genZeroBlockA(
    struct KgenContext *ctx,
    const ZeroFuncs *zeroFuncs)
{
    char tmp[1024];
    sprintf(tmp, "%s((__local float4*)tempA);\n", zeroFuncs->names[MATRIX_A]);
    kgenAddStmt(ctx, tmp);
}

/*
 * Generate control block of the loop over K
 * Two kind of loops: without triangle block and only triangle block
 */
static void
genInternalLoopCtl(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags,
    bool triangPart)
{
    char tmp[1024];

    (void)triangPart;

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "for (k0 = currM + %lu; k0 < M; k0 += %lu)",
                dim->bwidth, dim->bwidth);
    }
    else {
        sprintf(tmp, "for (k0 = 0; k0 < currM; k0 += %lu)",
                dim->bwidth);
    }

    kgenBeginBranch(ctx, tmp);
}

static void
genInitCurrM(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags)
{
    char tmp[1024];

    if (isMatrixUpper(kflags)) {
        /* start from the last block */
        sprintf(tmp, "currM = ((M - 1) / %lu) * %lu;\n", dim->y, dim->y);
        kgenAddStmt(ctx, tmp);
    }
    else {
        kgenAddStmt(ctx, "currM = 0;\n");
    }
}

static void
initKernelVarNames(KernelVarNames *kvars, KernelExtraFlags kflags)
{
    kvars->A = "A";
    kvars->B = "B";

    if (isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A)) {
        kvars->coordA = "coordA.x";
    }
    else {
        kvars->coordA = "coordA.y";
    }
    if (isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B)) {
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
    KernelExtraFlags kflags = kextra->flags;
    char tmp[1024];
    char blkmul[FUNC_NAME_MAXLEN];
    char updateResFn[FUNC_NAME_MAXLEN];
    char updateResGenericFn[FUNC_NAME_MAXLEN];
    char updateResFnRev[FUNC_NAME_MAXLEN];
    char updateResGenericFnRev[FUNC_NAME_MAXLEN];
    char copyPLFn[FUNC_NAME_MAXLEN];
    char *s1 = "";
    const char *typeName;
    CopyBufFuncs copyFuncs;
    ZeroFuncs zeroFuncs;
    DataType dtype = kextra->dtype;
    ssize_t ret;
    BlasGenSettings gset;
    BlkMulOpts mulOpts;
    size_t pitchAB, pitchC;
    bool b;
    const char *outTypeName;
    unsigned int nrRegs;
    unsigned int vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    int tra, trb;
    unsigned int l1Pans;
    char vect[2] = {'y', 'x'};
    UpdateResultFlags upFlags;

    if (pgran->wgDim != 1) {
        return -EINVAL;
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations and auxiliary functions

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;

    initKernelVarNames(&gset.varNames, kflags);

    if (isComplexType(dtype)) {
        genComplexMathOperators(ctx, dtype);
    }

    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_TRSM, &gset,
                         BCHF_MATRIX_A | BCHF_MATRIX_B | BCHF_WRITE_OUTPUT);

    generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                         ZF_MATRIX_A | ZF_MATRIX_B);
    getResultGPRsInfo(dtype, &subdims[1], vecLen, &nrRegs, &outTypeName);

    // functions updating result
    // for the final result
    generateUpresFuncs(ctx, CLBLAS_TRSM, &gset, updateResFn,
                       updateResGenericFn);
    // for intermediate result after blocks modification
    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= UPRES_WITH_BETA | UPRES_PRIV_DEST;
    genUpresFuncsWithFlags(ctx, &gset, upFlags, updateResFnRev,
                           updateResGenericFnRev);
    // for heaping before multiplying on inverted block
    updateResultGenOld(ctx, &gset, UPRES_SET,
                    UPRES_COLUMN_MAJOR | UPRES_USE_LDS, NULL);
    kgenGetLastFuncName(copyPLFn, FUNC_NAME_MAXLEN, ctx);
    kgenAddBlankLine(ctx);

    // block multiplication function
    mulOpts.aMobj = CLMEM_BUFFER;
    mulOpts.bMobj = CLMEM_BUFFER;
    mulOpts.flags = BLKMUL_SKEW_COLUMN | BLKMUL_OUTPUT_PRIVATE;
    mulOpts.core = BLKMUL_SEPARATE_MULADD;
    ret = blkMulGen(ctx, subdims, dtype, &mulOpts);
    if (ret) {
        destroyKgenContext(ctx);

        return -EOVERFLOW;
    }

    kgenAddBlankLine(ctx);
    kgenGetLastFuncName(blkmul, sizeof(blkmul), ctx);

    //matrix inversion function
    genInvertingBlockFunc(ctx, subdims[0].bwidth, dtype, kflags);

    typeName = dtypeBuiltinType(dtype);

    // now, generate the kernel
    declareTrxmKernel(ctx, dtype, pgran, kflags, CLBLAS_TRSM, NULL, false,
                      false);
    ret = kgenBeginFuncBody(ctx);

    /*
     * Calculate local buffer pitches, and then insert the
     * preparative code
     */
    pitchAB = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);
    pitchC = matrBlockPitch(subdims, MATRIX_C, dtype, clblasLeft);
    sprintf(tmp, trsmPrep1D, typeName, pitchAB * subdims[0].y,
            typeName,
            ((pitchC > pitchAB) ? pitchC : pitchAB) * subdims[0].y,
            outTypeName, nrRegs, s1, pgran->wgSize[0]);
    ret = kgenAddStmt(ctx, tmp);

   /*
    * B matrix is divided on panels, each work group
    * multiply such a panel on the whole matrix A.
    */

    sprintf(tmp, "currN = gid * %lu;\n", subdims[0].x);
    kgenAddStmt(ctx, tmp);

    genInitCurrM(ctx, subdims, kflags);
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    genTrxmBMatrShift(ctx, kflags, false);

    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);

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

    genZeroResult(ctx, dtype, subdims);

    genInternalLoopCtl(ctx, subdims, kflags, false);   // loop over K

    genPrepareSquareBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                          kflags, 'A');
    genPrepareRectBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                        kflags, 'C');
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    // multiplication for the step-by-step block updating
    sprintf(tmp, "%s((LPtr)(tempA + (lid / %u * %lu) * %lu), \n"
                 "    (LPtr)(tempC + (lid %% %u * %lu) * %lu),\n"
                 "    (%s*)c, lid %% %lu);\n",
                blkmul, l1Pans, subdims[1].y, pitchAB,
                l1Pans, subdims[1].x, pitchAB, outTypeName, subdims[1].y);
    ret = kgenAddStmt(ctx, tmp);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    genInternalLoopEnd(ctx);                             // loop over K
    kgenAddBlankLine(ctx);

    kgenAddStmt(ctx, "k0 = currM;\n");
    genPrepareSquareBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                          kflags, 'C');
    genZeroBlockA(ctx, &zeroFuncs);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    if (kflags & KEXTRA_UNIT_DIAGONAL) {
        sprintf(tmp, "if (lid < %lu) {\n"
                     "    tempC[lid * %lu + lid] = %s;\n"
                     "}\n",
                subdims[0].bwidth, pitchAB, strOne(dtype));
        kgenAddStmt(ctx, tmp);
        kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
        kgenAddBlankLine(ctx);
    }

    sprintf(tmp, "if (lid < %lu)", subdims[0].bwidth);
    kgenBeginBranch(ctx, tmp);
    sprintf(tmp, "invert(tempC, tempA, lid, (currM + %lu > M) ? "
                         "M - currM : %lu);\n",
            subdims[0].y, subdims[0].y);
    kgenAddStmt(ctx, tmp);
    kgenEndBranch(ctx, NULL);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);

    genUpdateIntermTrsmResult(ctx, &gset, updateResFnRev,
                              updateResGenericFnRev, true);

    genHeapTrsmResultToLDS(ctx, &gset, copyPLFn, "tempC");
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    genZeroResult(ctx, dtype, subdims);

    // multypling on an inverted block
    sprintf(tmp, "%s((LPtr)(tempA + (lid / %u * %lu) * %lu), \n"
                 "    (LPtr)(tempC + (lid %% %u * %lu) * %lu),\n"
                 "    (%s*)c, lid %% %lu);\n\n",
            blkmul, l1Pans, subdims[1].y, pitchAB,
            l1Pans, subdims[1].x, pitchAB, outTypeName, subdims[1].y);
    ret = kgenAddStmt(ctx, tmp);

    // write back the tile evaluated
    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= UPRES_EXCEED_PROBLEM_CONDITION;
    genResultUpdateWithFlagsOld(ctx, CLBLAS_TRSM, &gset, upFlags, updateResFn,
                                updateResGenericFn, NULL);

    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "currM -= %lu;\n", subdims[0].y);
    }
    else {
        sprintf(tmp, "currM += %lu;\n", subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    kgenEndBranch(ctx, NULL);                       // loop over M

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
    KernelExtraFlags kflags = ((CLBLASKernExtra*)extra)->flags;
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
    cl_ulong sizeA, sizeB, size;
    const CLBlasKargs *kargs = (const CLBlasKargs*)kernelArgs;

    /*
     * It's needed one block for each matrix A and B,
     * and one block of size maximal of this one for
     * matrix B and matrix C
     */

    sizeA = matrBlockSize(dim, MATRIX_A, dtype, kargs->side);
    sizeB = matrBlockSize(dim, MATRIX_B, dtype, kargs->side);
    size = matrBlockSize(dim, MATRIX_C, dtype, kargs->side);
    if (sizeB > size) {
        size = sizeB;
    }
    size += sizeA + sizeB;

    return (size * dtypeSize(dtype) <= ldsSize);
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D | SF_TOP_INPUT_SQUARE_BLOCKS);
}

void
initTrsmLdsPattern(MemoryPattern *mempat)
{
    mempat->name = "LDS based block trsm";
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
