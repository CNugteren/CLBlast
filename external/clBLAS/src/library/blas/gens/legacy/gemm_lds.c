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
 * LDS based block GEMM generator
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
ldsGetPerf(
    unsigned int kflags,
    const void *args);

static SolverOps solverOps = {
    generator,
    assignKargs,
    isFitToLDS,
    ldsGetPerf,
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
declareKernel(
    struct KgenContext *ctx,
    DataType dtype,
    const PGranularity *pgran)
{
    char tmp[1024];
    char fpref;
    const char *typeName;

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    sprintf(tmp, "__attribute__((reqd_work_group_size(%u, %u, 1)))\n"
                 "void __kernel\n"
                 "%cgemm(\n"
                 "    uint M,\n"
                 "    uint N,\n"
                 "    uint K,\n"
                 "    %s alpha,\n"
                 "    __global %s *A,\n"
                 "    uint lda,\n"
                 "    __global %s *B,\n"
                 "    uint ldb,\n"
                 "    %s beta,\n"
                 "    __global %s *C,\n"
                 "    uint ldc,\n"
                 "    const uint offA,\n"
                 "    const uint offB,\n"
                 "    const uint offC)\n",
            pgran->wgSize[0], pgran->wgSize[1],
            fpref, typeName, typeName,
            typeName, typeName, typeName);

    kgenDeclareFunction(ctx, tmp);
}

static void
declareLocalVariables(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims,
    const PGranularity *pgran)
{
    char tmp[1024];
    const char *inTypeName, *outTypeName;
    size_t pitchAB;
    unsigned int nrRegs;
    unsigned int vecLen;

    inTypeName = dtypeBuiltinType(dtype);
    pitchAB = matrBlockPitch(dims, MATRIX_A, dtype, clblasLeft);
    vecLen = sizeof(cl_float4) / dtypeSize(dtype);

    getResultGPRsInfo(dtype, &dims[1], vecLen, &nrRegs, &outTypeName);

    sprintf(tmp, "uint m0, k0;\n"
                 "__local %s tempA[%lu];\n"
                 "__local %s tempB[%lu];\n"
                 "%s c[%u];\n"
                 "uint currM, currN, groupsPan;\n"
                 "uint2 coordA, coordB;\n"
                 "uint x, y;\n",
             inTypeName, pitchAB * dims[0].y,
             inTypeName, pitchAB * dims[0].x,
             outTypeName, nrRegs);

    kgenAddStmt(ctx, tmp);
    kgenDeclareLocalID(ctx, "lid", pgran);
    kgenDeclareGroupID(ctx, "gid", pgran);
    kgenAddBlankLine(ctx);
}

static void
genPrepareBlockA(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags)
{
    char tmp[1024];
    size_t pitch;
    const char *coordName[2] = {"currM", "k0"};
    const char *sizeName[2] = {"y", "x"};
    size_t bsize[2] = {dim->y, dim->bwidth};
    int tra;

    tra = isMatrixAccessColMaj(CLBLAS_GEMM, flags, MATRIX_A);
    pitch = matrBlockPitch(dim, MATRIX_A, dtype, clblasLeft);

    /*
     * If the (sub)problem is integrally divisible,
     * skip any checks, and just read with optimal blocks,
     * otherwise check for tails and then read with a
     * fast function in the case of optimal blocks, and with
     * the slow one in the case of tails respectively
     */

    if (!(flags & (KEXTRA_TAILS_M | KEXTRA_TAILS_K))) {
        sprintf(tmp, "%s((LPtr)tempA, (GPtr)A, %s, %s, lda);\n",
                copyFuncs->read[MATRIX_A], coordName[tra], coordName[1 - tra]);
    }
    else {
        sprintf(tmp,
                "y = (currM + %lu <= M) ? %lu : M - currM;\n"
                "x = (k0 + %lu <= K) ? %lu : K - k0;\n"
                "if ((y == %lu) && (x == %lu)) {\n"
                     // fast read
                "    %s((LPtr)tempA, (GPtr)A, %s, %s, lda);\n"
                "}\n"
                "else {\n"
                "    %s((__local float4*)tempA);\n"           // zeroing
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                     // slow read
                "    %s((LPtr)tempA, (GPtr)A, %s, %s, %s, %s, %lu, lda);\n"
                "}\n\n",
                bsize[0], bsize[0], bsize[1], bsize[1], bsize[0], bsize[1],
                copyFuncs->read[MATRIX_A], coordName[tra], coordName[1 - tra],
                zeroFuncs->names[MATRIX_A],
                copyFuncs->readGeneric[MATRIX_A], coordName[tra],
                coordName[1 - tra], sizeName[tra], sizeName[1 - tra],
                pitch);
    }

    kgenAddStmt(ctx, tmp);
}

static void
genPrepareBlockB(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags)
{
    char tmp[1024];
    size_t pitch;
    const char *coordName[2] = {"currN", "k0"};
    const char *sizeName[2] = {"y", "x"};
    size_t bsize[2] = {dim->x, dim->bwidth};
    int trb;

    pitch = matrBlockPitch(dim, MATRIX_B, dtype, clblasLeft);
    trb = isMatrixAccessColMaj(CLBLAS_GEMM, flags, MATRIX_B);

    if (!(flags & (KEXTRA_TAILS_N | KEXTRA_TAILS_K))) {
        sprintf(tmp, "%s((LPtr)tempB, (GPtr)B, %s, %s, ldb);\n",
                copyFuncs->read[MATRIX_B], coordName[trb],
                coordName[1 - trb]);
    }
    else {
        sprintf(tmp,
                "y = (currN + %lu <= N) ? %lu : N - currN;\n"
                "x = (k0 + %lu <= K) ? %lu : K - k0;\n"
                "if ((y == %lu) && (x == %lu)) {\n"
                     // fast read
                "    %s((LPtr)tempB, (GPtr)B, %s, %s, ldb);\n"
                "}\n"
                "else {\n"
                "    %s((__local float4*)tempB);\n"           // zeroing
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                             // slow read
                "    %s((LPtr)tempB, (GPtr)B, %s, %s, %s, %s, %lu, ldb);\n"
                "}\n\n",
                bsize[0], bsize[0], bsize[1], bsize[1], bsize[0], bsize[1],
                copyFuncs->read[MATRIX_B], coordName[trb], coordName[1 - trb],
                zeroFuncs->names[MATRIX_B],
                copyFuncs->readGeneric[MATRIX_B], coordName[trb],
                coordName[1 - trb], sizeName[trb], sizeName[1 - trb],
                pitch);
    }

    kgenAddStmt(ctx, tmp);
}

static void
genZeroResult(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims)
{
    unsigned int n;
    char tmp[1024];
    unsigned int vecLen;

    vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    getResultGPRsInfo(dtype, &dims[1], vecLen, &n, NULL);

    sprintf(tmp, "\n"
                 "for (x = 0; x < %u; x++) {\n"
                 "    c[x] = 0;\n"
                 "}\n\n", n);

    kgenAddStmt(ctx, tmp);
}

static void
initKernelVarNames(KernelVarNames *kvars, KernelExtraFlags kflags)
{
    kvars->A = "A";
    kvars->B = "B";
    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_A)) {
        kvars->coordA = "coordA.x";
    }
    else {
        kvars->coordA = "coordA.y";
    }
    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_B)) {
        kvars->coordB = "coordB.x";
    }
    else {
        kvars->coordB = "coordB.y";
    }
    kvars->sizeM = "M";
    kvars->sizeN = "N";
    kvars->sizeK = "K";
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
    char blkmul[128];
    char updateResFn[FUNC_NAME_MAXLEN];
    char updateResGenericFn[FUNC_NAME_MAXLEN];
    CopyBufFuncs copyFuncs;
    ZeroFuncs zeroFuncs;
    DataType dtype = kextra->dtype;
    ssize_t ret;
    BlasGenSettings gset;
    BlkMulOpts mulOpts;
    size_t pitchAB;
    const char *s;
    bool b;
    int tra, trb;
    unsigned int l1Pans;
    unsigned int vecLen;
    char vect[2] = {'y', 'x'};
    UpdateResultFlags upFlags;

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations and auxiliary functions

    pitchAB = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);
    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    // generator settings initialization
    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;
    initKernelVarNames(&gset.varNames, kflags);

    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_GEMM, &gset,
                         BCHF_MATRIX_A | BCHF_MATRIX_B);

    generateUpresFuncs(ctx, CLBLAS_GEMM, &gset, updateResFn,
                       updateResGenericFn);

    generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                         ZF_MATRIX_A | ZF_MATRIX_B);

    // block multiplication function
    mulOpts.aMobj = CLMEM_BUFFER;
    mulOpts.bMobj = CLMEM_BUFFER;
    mulOpts.flags = BLKMUL_OUTPUT_PRIVATE | BLKMUL_SKEW_COLUMN;
    if (isComplexType(dtype)) {
        mulOpts.core = BLKMUL_SEPARATE_MULADD;
    }
    else {
        mulOpts.core = BLKMUL_MAD;
    }
    ret = blkMulGen(ctx, subdims, dtype, &mulOpts);
    if (ret) {
        destroyKgenContext(ctx);

        return -EOVERFLOW;
    }

    kgenAddBlankLine(ctx);
    kgenGetLastFuncName(blkmul, sizeof(blkmul), ctx);

    // now, generate the kernel
    declareKernel(ctx, dtype, pgran);
    kgenBeginFuncBody(ctx);
    declareLocalVariables(ctx, dtype, subdims, pgran);

    // Shift matrices' origins according to offsetM and offsetN.
    kgenAddBlankLine(ctx);
    tmp[0] = '\0';
    strcat(tmp, "A += offA;\n");
    strcat(tmp, "B += offB;\n");
    strcat(tmp, "C += offC;\n");

    kgenAddStmt(ctx, tmp);
    kgenAddBlankLine(ctx);

    /*
     * Output matrix is divided into squares, each work group
     * gets such a square. Get current panel coordinates
     * depending on which matrix must be outer.
     * Assign different inner matrix's panels processed
     * at the same time to different work groups in order to
     * reduce global memory bank conflicts. Use cyclic
     * addressing for this purpose
     */
    sprintf(tmp, // number of outer panels
                 "groupsPan = N / %lu;\n"
                 "if (N %% %lu) {\n"
                 "    groupsPan++;\n"
                 "}\n"
                 "x = gid %% groupsPan;\n"  // outer panel number
                 "y = gid / groupsPan;\n"   // outer inner number
                 "currN = x * %lu;\n"
                 "\n"
                 // number of inner panels
                 "groupsPan = M / %lu;\n"
                 "if (M %% %lu) {\n"
                 "    groupsPan++;\n"
                 "}\n"
                 // inner panel number using cyclic addressing
                 "y = (x + y) %% groupsPan;\n"
                 "currM = y * %lu;\n"
                 "\n",
            subdims[0].itemX, subdims[0].itemX, subdims[0].itemX,
            subdims[0].itemY, subdims[0].itemY, subdims[0].itemY);
        ret = kgenAddStmt(ctx, tmp);

    tra = isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_B);
    sprintf(tmp, "coordA.%c = currM;\n"
                 "coordA.%c = 0;\n"
                 "coordB.%c = currN;\n"
                 "coordB.%c = 0;\n\n",
            vect[tra], vect[1 - tra], vect[trb], vect[1 - trb]);
    kgenAddStmt(ctx, tmp);

    genZeroResult(ctx, dtype, subdims);

    // loop over K
    sprintf(tmp, "for (k0 = 0; k0 < K; k0 += %lu)", subdims[0].bwidth);
    kgenBeginBranch(ctx, tmp);

    genPrepareBlockA(ctx, subdims, dtype, &copyFuncs,
                     &zeroFuncs, kflags);
    genPrepareBlockB(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                     kflags);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    l1Pans = (unsigned int)subdims[0].x / (unsigned int)subdims[1].x;

    vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    // and eventually multiply the blocks and update the current result
    getResultGPRsInfo(dtype, &subdims[1], vecLen, NULL, &s);
    sprintf(tmp, "%s((LPtr)(tempA + (lid / %u * %lu) * %lu),\n"
                 "   (LPtr)(tempB + (lid %% %u * %lu) * %lu),\n"
                 "   (%s*)c, lid);\n",
            blkmul, l1Pans, subdims[1].y, pitchAB, l1Pans,
            subdims[1].x, pitchAB, s);
    kgenAddStmt(ctx, tmp);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    kgenEndBranch(ctx, NULL);       // loop over K

    // update result logic
    sprintf(tmp, "coordA.%c += lid / %u * %lu;\n"
                 "coordB.%c += lid %% %u * %lu;\n",
            vect[tra], l1Pans, subdims[1].y, vect[trb], l1Pans, subdims[1].x);
    kgenAddStmt(ctx, tmp);
    if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N)) {
        sprintf(tmp, "if (coordA.%c >= M || coordB.%c >= N) {\n"
                     "  return;\n"
                     "}\n", vect[tra], vect[trb]);
        kgenAddStmt(ctx, tmp);
    }
    kgenAddBlankLine(ctx);

    upFlags = kextraToUpresFlags(CLBLAS_GEMM, kflags);
    upFlags |= UPRES_EXCEED_PROBLEM_CONDITION;
    genResultUpdateWithFlagsOld(ctx, CLBLAS_GEMM, &gset, upFlags, updateResFn,
                                updateResGenericFn, NULL);

    ret = kgenEndFuncBody(ctx);
    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

static void
assignKargs(KernelArg *args, const void *params, const void *extra)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    (void)extra;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    initSizeKarg(&args[2], blasArgs->K);
    assignScalarKarg(&args[3], &(blasArgs->alpha), blasArgs->dtype);
    initMemobjKarg(&args[4], blasArgs->A, NULL, 0, 0);
    initSizeKarg(&args[5], blasArgs->lda.matrix);
    initMemobjKarg(&args[6], blasArgs->B, NULL, 0, 0);
    initSizeKarg(&args[7], blasArgs->ldb.matrix);
    assignScalarKarg(&args[8], &(blasArgs->beta), blasArgs->dtype);
    initMemobjKarg(&args[9], blasArgs->C, NULL, 0, 0);
    initSizeKarg(&args[10], blasArgs->ldc.matrix);
    initSizeKarg(&args[11], blasArgs->offA);
    initSizeKarg(&args[12], blasArgs->offBX);
    initSizeKarg(&args[13], blasArgs->offCY);
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    cl_ulong size;

    (void)kernelArgs;

    size = matrBlockSize(dim, MATRIX_A, dtype, clblasLeft);
    size += matrBlockSize(dim, MATRIX_B, dtype, clblasLeft);
    size += matrBlockSize(dim, MATRIX_C, dtype, clblasLeft);

    return (size * dtypeSize(dtype) <= ldsSize);
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_2D);
}

void
initGemmLdsPattern(MemoryPattern *mempat)
{
    mempat->name = "LDS based block gemm";
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
ldsGetPerf(
    unsigned int kflags,
    const void *args)
{
    (void)args;
    (void)kflags;

    return PPERF_POOR;
}
