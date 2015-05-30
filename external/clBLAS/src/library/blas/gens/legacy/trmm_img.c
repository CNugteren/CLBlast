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
 * trmm image based generator
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#include <clBLAS.h>
#include <matrix_dims.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <dis_warning.h>

#include "blas_kgen_legacy.h"
#include "../gen_helper.h"
#include "gen_helper_legacy.h"
#include "trxm_common_legacy.h"

static CLBLASMpatExtra mpatExtra;

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static ssize_t
preparator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static ssize_t
genWrapper(
    char *buf,
    size_t buflen,
    const struct SubproblemDim *subdims,
    const struct PGranularity *pgran,
    void *extra)
{
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    if (kextra->kernType == CLBLAS_COMPUTING_KERNEL) {
        return generator(buf, buflen, subdims, pgran, extra);
    }
    else {
        return preparator(buf, buflen, subdims, pgran, extra);
    }
}

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

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);

static int getPerf(
    unsigned int kflags,
    const void *args);

static SolverOps imgSops = {
    genWrapper,
    assignKargs,
    isFitToLDS,
    getPerf,
    NULL,
    calcNrThreads,
    NULL,
    solverFlags,
    NULL, //fixupKargs
    NULL, //getDefaultDecomp
    NULL, //getDecompList
    NULL,
    NULL
};

static void
imgToCopyBufFuncs(
    CopyBufFuncs *bufFuncs,
    const CopyImgFuncs *imgFuncs,
    KernelExtraFlags kflags)
{
    memcpy(bufFuncs->write, imgFuncs->localToImage, FUNC_NAME_MAXLEN);
    if (isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_A)) {
        memcpy(bufFuncs->read[MATRIX_A],
               imgFuncs->globalToLocalTransposed[MATRIX_A], FUNC_NAME_MAXLEN);
        memcpy(bufFuncs->readGeneric[MATRIX_A],
               imgFuncs->globalToLocalTransposedGeneric[MATRIX_A],
               FUNC_NAME_MAXLEN);
    }
    else {
        memcpy(bufFuncs->read[MATRIX_A],
               imgFuncs->globalToLocal[MATRIX_A], FUNC_NAME_MAXLEN);
        memcpy(bufFuncs->readGeneric[MATRIX_A],
               imgFuncs->globalToLocalGeneric[MATRIX_A],
               FUNC_NAME_MAXLEN);
    }
}

static void
genPrepKernelA(
    struct KgenContext *ctx,
    const SubproblemDim *subdims,
    KernelExtraFlags kflags,
    DataType dtype,
    CopyImgFuncs *copyImgFuncs,
    const PGranularity *pgran)
{
    char tmp[4096];
    bool isBranch = false;
    size_t localBufSize;
    unsigned int tsize, vecLen;
    const char *typeName;
    CopyBufFuncs copyBufFuncs;
    char fpref;

    fpref = dtypeToBlasPrefix(dtype);
    typeName = dtypeBuiltinType(dtype);
    tsize = dtypeSize(dtype);
    vecLen = sizeof(cl_float4) / tsize;
    localBufSize = subdims[1].y * fl4RowWidth(subdims[1].bwidth, tsize);
    localBufSize *= vecLen;
    imgToCopyBufFuncs(&copyBufFuncs, copyImgFuncs, kflags);

    sprintf(tmp, "void __kernel\n"
                 "%cprepareImageA(\n"
                 "    uint M,\n"
                 "    __global %s *A,\n"
                 "    uint lda,\n"
                 "    __write_only image2d_t imgA,\n"
                 "    uint startM,\n"
                 "    uint origM,\n"
                 "    uint offA)\n",
            fpref, typeName);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    kgenDeclareGroupID(ctx, "gid", pgran);
    kgenDeclareLocalID(ctx, "lid", pgran);
    sprintf(tmp, "const uint bpr = (origM + %lu) / %lu;\n"
                 "uint currM = startM + (gid / bpr) * %lu;\n"
                 "uint k0 = (gid %% bpr) * %lu;\n"
                 "uint x, y;\n"
                 "__local %s tempA[%lu];\n"
                 "bool processed = false;\n\n",
            subdims[1].bwidth - 1, subdims[1].bwidth, subdims[1].y,
            subdims[1].bwidth, typeName, localBufSize);
    kgenAddStmt(ctx, tmp);

    kgenAddStmt(ctx, "A += offA;\n");
    if (!(isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_A) ||
          isMatrixConj(kflags, MATRIX_A))) {

        if (isMatrixUpper(kflags)) {
            sprintf(tmp, "if (k0 >= currM + %lu)", subdims[1].y);
        }
        else {
            sprintf(tmp, "if (k0 + %lu <= currM)", subdims[1].bwidth);
        }
        kgenBeginBranch(ctx, tmp);
        sprintf(tmp, "if ((currM + %lu <= M + startM) && "
                         "(k0 + %lu <= origM) && %d) {\n"
                     // write directly to an image from the global memory
                     "    %s(imgA, k0 / %u, currM - startM, (GPtr)A, "
                            "currM, k0, lda);\n"
                     "    processed = true;\n"
                     "}\n",
                subdims[1].y, subdims[1].bwidth,
                (kflags & KEXTRA_NO_COPY_VEC_A) == 0,
                copyImgFuncs->globalToImage[MATRIX_A], vecLen);

        kgenAddStmt(ctx, tmp);
        kgenEndBranch(ctx, NULL);

        kgenBeginBranch(ctx, "if (!processed)");
        isBranch = true;
    }

    // now, zeroing blocks entirely located in the "other" triangle
    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "if (k0 + %lu <= currM) {\n"
                     "    %s((__local float4*)tempA);\n"
                     "}\n",
                subdims[1].bwidth, copyImgFuncs->zeroBlock[MATRIX_A]);
    }
    else {
        sprintf(tmp, "if (k0 >= currM + %lu) {\n"
                     "    %s((__local float4*)tempA);\n"
                     "}\n",
                subdims[1].y, copyImgFuncs->zeroBlock[MATRIX_A]);
    }
    kgenAddStmt(ctx, tmp);

    // useful block path, reading data from the global memory to the local one
    kgenBeginBranch(ctx, "else");
    kgenAddStmt(ctx, "M += startM;\n");
    genPrepareTrxmBlockA(ctx, subdims, dtype, &copyBufFuncs,
                         (ZeroFuncs*)copyImgFuncs->zeroBlock,
                         kflags, "origM");
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddStmt(ctx, "M -= startM;\n");
    genTriangMatrBlock(ctx, subdims, dtype, kflags);
    kgenEndBranch(ctx, NULL);

    // and write to the image
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    sprintf(tmp, "%s(imgA, k0 / %u, currM - startM, (LPtr)tempA);\n",
            copyImgFuncs->localToImage[MATRIX_A], vecLen);
    kgenAddStmt(ctx, tmp);
    if (isBranch) {
        kgenEndBranch(ctx, NULL);
    }

    kgenEndFuncBody(ctx);
}

static void
genPrepKernelB(
    struct KgenContext *ctx,
    const SubproblemDim *subdims,
    DataType dtype,
    CopyImgFuncs *copyImgFuncs,
    const PGranularity *pgran,
    KernelExtraFlags kflags)
{
    char tmp[4096];
    size_t localBufSize;
    unsigned int tsize, vecLen;
    const char *typeName;
    char fpref;

    const char *funcHead =
        "bool trb, aligned;\n"
        "const uint bpr = (origM + %lu) / %lu;\n"
        "const uint n = startN + (gid / bpr) * %lu;\n"
        "const uint k = (gid %% bpr) * %lu;\n"
        "uint x, y;\n"
        "__local %s temp[%lu];\n"
        "\n"
        "B += offB;\n"
        "trb = (order == clblasRowMajor) ^ (side == clblasRight);\n"
        "N += startN;\n";

    const char *funcBody =
        "//copy matrix B block\n"
        "y = n + %u <= N ? %u : N - n;\n"
        "x = k + %u <= origM ? %u : origM - k;\n"
        "aligned = (x == %u) && (y == %u) && %d;\n"
        "if (aligned && !trb) {\n"
        "    %s(imgB, k / %u, n - startN, (GPtr)B, n, k, ldb);\n"
        "}\n"
        "else {\n"
        "    if (n >= N) {\n"
                // just zero, this is padding related part
        "        %s((__local float4*)temp);\n"
        "    }\n"
        "    else if (!aligned) {\n"
        "        // zero local memory\n"
        "        %s((__local float4*)temp);\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "        if (trb) {\n"
        "            // generic transposed global to local\n"
        "            %s((LPtr)temp, (GPtr)B, k, n, x, y, %u, ldb);\n"
        "        }\n"
        "        else {\n"
        "            // generic global to local\n"
        "            %s((LPtr)temp, (GPtr)B, n, k, y, x, %u, ldb);\n"
        "        }\n"
        "    }\n"
        "    else {\n"
        "        if (trb) {//transposed, aligned\n"
        "            // optimized transposed global to local\n"
        "            %s((LPtr)temp, (GPtr)B, k, n, ldb);\n"
        "        }\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    %s(imgB, k / %u, n - startN, (LPtr)temp);\n"
        "}\n"
        "\n";

    fpref = dtypeToBlasPrefix(dtype);
    typeName = dtypeBuiltinType(dtype);
    tsize = dtypeSize(dtype);
    vecLen = sizeof(cl_float4) / tsize;
    localBufSize = subdims[1].x * fl4RowWidth(subdims[1].bwidth, tsize);
    localBufSize *= vecLen;

    sprintf(tmp, "void __kernel\n"
                 "%cprepareImageB(\n"
                 "    clblasOrder order,\n"
                 "    clblasSide side,\n"
                 "    uint N,\n"
                 "    __global %s *B,\n"
                 "    uint ldb,\n"
                 "    __write_only image2d_t imgB,\n"
                 "    uint startN,\n"
                 "    uint origM,\n"
                 "    uint offB)\n",
            fpref, typeName);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    kgenDeclareGroupID(ctx, "gid", pgran);
    sprintf(tmp, funcHead,
            subdims[1].bwidth - 1, subdims[1].bwidth,
            subdims[1].x, subdims[1].bwidth,
            typeName, localBufSize);
    kgenAddStmt(ctx, tmp);

    sprintf(tmp, funcBody,
            subdims[1].x, subdims[1].x, // y = n + dy <= N ?...
            subdims[1].bwidth,
            subdims[1].bwidth, // x = k + bw <= M ?...
            subdims[1].bwidth,
            subdims[1].x, // aligned = (x==bw1)&&(y==dx1)
            (kflags & KEXTRA_NO_COPY_VEC_B) == 0,
            copyImgFuncs->globalToImage[MATRIX_B],
            vecLen,
            copyImgFuncs->zeroBlock[MATRIX_B],
            copyImgFuncs->zeroBlock[MATRIX_B],
            copyImgFuncs->globalToLocalTransposedGeneric[MATRIX_B],
            subdims[1].bwidth,
            copyImgFuncs->globalToLocalGeneric[MATRIX_B],
            subdims[1].bwidth,
            copyImgFuncs->globalToLocalTransposed[MATRIX_B],
            copyImgFuncs->localToImage[MATRIX_B],
            vecLen);
    kgenAddStmt(ctx, tmp);

    kgenEndFuncBody(ctx);
}

static void
declareMainKernel(
    struct KgenContext *ctx,
    DataType dtype,
    KernelExtraFlags kflags,
    const PGranularity *pgran)
{
    char tmp[4048];
    char fpref;
    const char *typeName;
    char coordNames[2] = {'M', 'N'};
    int side = ((kflags & KEXTRA_SIDE_RIGHT) != 0);

    fpref = dtypeToBlasPrefix(dtype);
    typeName = dtypeBuiltinType(dtype);
    sprintf(tmp, "__attribute__((reqd_work_group_size(%u, %u, 1)))\n"
                 "void __kernel\n"
                 "%ctrmmImg(\n"
                 "    uint %c,\n"
                 "    uint %c,\n"
                 "    const %s alpha,\n"
                 "    const __read_only image2d_t A,\n"
                 "    const __read_only image2d_t B,\n"
                 "    __global %s *C,\n"
                 "    uint ldb,\n"
                 "    const uint start%c,\n"
                 "    const uint start%c,\n"
                 "    const uint origM,\n"
                 "    const uint offB)\n",
             pgran->wgSize[0], pgran->wgSize[1],  fpref, coordNames[side],
             coordNames[1 - side], typeName, typeName, coordNames[side],
             coordNames[1 - side]);

    kgenDeclareFunction(ctx, tmp);
}

// Preparation function for images based kernel generator
static ssize_t
preparator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    CopyImgFuncs copyImgFuncs;
    BlasGenSettings gset;
    ssize_t ret;
    bool b;

    memset(&copyImgFuncs, 0, sizeof(copyImgFuncs));
    memset(&gset, 0, sizeof(gset));

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    b = isDoubleBasedType(kextra->dtype);
    kgenDeclareUptrs(ctx, b);
    if (kextra->kernType == CLBLAS_PREP_B_KERNEL) {
        declareBlasEnums(ctx);
    }

    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;

    // generate necessary memory to image copying functions
    generateImageCopyFuncs(&copyImgFuncs, ctx, CLBLAS_TRMM, &gset);
    kgenAddBlankLine(ctx);

    if (kextra->kernType == CLBLAS_PREP_A_KERNEL) {
        genPrepKernelA(ctx, subdims, kextra->flags, kextra->dtype,
                       &copyImgFuncs, pgran);
    }
    else {
        genPrepKernelB(ctx, subdims, kextra->dtype, &copyImgFuncs, pgran,
                       kextra->flags);
    }

    ret = kgenAddBlankLine(ctx);
    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }
    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

static void
initKernelVarNames(KernelVarNames *kvars, KernelExtraFlags kflags)
{
    kvars->A = "imgA";
    kvars->B = "imgB";
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
    char tmp[4096], tmp1[4096];
    char *p;
    // is the iteration over N, N at the top level
    const char *typeName;
    DataType dtype = kextra->dtype;
    ssize_t ret;
    BlasGenSettings gset;
    BlkMulOpts mulOpts;
    unsigned int tsize;
    unsigned int vecLen, outVecLen;
    bool b;
    const char *outTypeName;
    unsigned int i;
    unsigned int nrRegs, regPitch;
    KernelExtraFlags kflags = kextra->flags;
    int tra, trb;
    char coordNames[2] = {'M', 'N'};
    char vect[2] = {'y', 'x'};

    const char *coordConstants =
        "const uint workItemM = startM + get_global_id(0) * %lu;\n"
        "const uint workItemN = startN + get_global_id(1) * %lu;\n"
        "const int2 skewRow = (int2)(0, get_local_id(0) %% %lu);\n"
        "uint vectK = (origM + %u) / %u;\n";

    /*
     *  template for image based trmm preparation part
     *  for two dimensional work space
     */
    const char *localVariables =
        "uint k0;\n"
        "int2 coordA = (int2)(0, workItemM - startM);\n"
        "int2 coordB = (int2)(0, workItemN - startN);\n"
        "%s c[%u];\n\n";

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;
    initKernelVarNames(&gset.varNames, kflags);

    tsize = dtypeSize(dtype);
    vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    if (isComplexType(dtype)) {
        regPitch = (unsigned int)subdims[1].x;
    }
    else {
        regPitch = (unsigned int) fl4RowWidth(subdims[1].x, tsize) *
                                             sizeof(cl_float4) / tsize;
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations and auxiliary functions
    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    typeName = dtypeBuiltinType(dtype);

    // now, generate the kernel
    declareMainKernel(ctx, dtype, kflags, pgran);
    ret = kgenBeginFuncBody(ctx);

    // constants
    sprintf(tmp, coordConstants,
            subdims[1].y, subdims[1].x, subdims[1].y,
            vecLen - 1, vecLen);
    kgenAddStmt(ctx, tmp);

    /*
     * Calculate local buffer pitches, and then declare local
     * variables
     */
    getResultGPRsInfo(dtype, &subdims[1], vecLen, &nrRegs, &outTypeName);

    sprintf(tmp, localVariables, outTypeName, nrRegs);
    kgenAddStmt(ctx, tmp);

    // check if offset exceeds matrix
    kgenAddStmt(ctx, "if ((workItemM >= startM + M) ||"
                         "(workItemN >= startN + N)) {\n"
                     "    return;\n"
                     "}\n");

    // zero C block
    sprintf(tmp, "for (k0 = 0; k0 < %u; k0++) {\n"
                 "    c[k0] = 0;\n"
                 "}\n\n",
            nrRegs);
    kgenAddStmt(ctx, tmp);

    // loop over K
    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "coordA.x = vectK - %lu;\n"
                     "coordB.x = coordA.x;\n",
                subdims[1].bwidth / vecLen);
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "for (k0 = ((workItemM/%lu)*%lu)/%u; "
                          "k0 < vectK; k0 += %lu)",
                subdims[0].bwidth, subdims[0].bwidth, vecLen,
                subdims[1].bwidth / vecLen);
    }
    else {
        size_t dk;

        dk = (subdims[1].y > subdims[1].bwidth) ? subdims[1].y :
                                                  subdims[1].bwidth;
        dk = dk / vecLen + 1;
        sprintf(tmp, "for (k0 = 0; "
                          "k0 < min((workItemM+%u)/%u + %lu, vectK); "
                          "k0 += %lu)",
                vecLen - 1, vecLen, dk, subdims[1].bwidth / vecLen);
    }
    kgenBeginBranch(ctx, tmp);

    mulOpts.aMobj = CLMEM_IMAGE;
    mulOpts.bMobj = CLMEM_IMAGE;
    mulOpts.flags = BLKMUL_OUTPUT_PRIVATE | BLKMUL_SKEW_ROW | BLKMUL_INLINE |
                    BLKMUL_AVOID_AND;
    if (isComplexType(dtype)) {
        mulOpts.core = BLKMUL_SEPARATE_MULADD;
    }
    else {
        mulOpts.core = BLKMUL_MAD;
    }
    mulOpts.argNames.coordA = "coordA";
    mulOpts.argNames.coordB = "coordB";
    mulOpts.argNames.skewCol = "skewCol";
    mulOpts.argNames.skewRow = "skewRow";
    mulOpts.argNames.k = "k0";
    mulOpts.argNames.vectBoundK = "vectK";
    ret = blkMulGen(ctx, subdims, dtype, &mulOpts);
    if (ret) {
        destroyKgenContext(ctx);
        return -EOVERFLOW;
    }

    // update image coordinates
    if (isMatrixUpper(kflags)) {
        // In this case loop is inverted to avoid 'random' skews
        sprintf(tmp, "\ncoordA.x -= %lu;\n"
                     "coordB.x -= %lu;\n",
                subdims[1].bwidth / vecLen, subdims[1].bwidth / vecLen);
    }
    else {
        sprintf(tmp, "\ncoordA.x += %lu;\n"
                     "coordB.x += %lu;\n",
                subdims[1].bwidth / vecLen, subdims[1].bwidth / vecLen);
    }
    kgenAddStmt(ctx, tmp);

    kgenEndBranch(ctx, NULL);
    // reorder the given solution
    outVecLen = isComplexType(dtype) ? 1 : vecLen;
    p = tmp1;
    for (i = 0; i < regPitch / outVecLen; i++) {
        unsigned int k = (unsigned int)(subdims[1].y - 1)
                                         * regPitch / outVecLen + i;

        sprintf(p,  "\n"
                    "    tmp = c[%u];\n"
                    "    for (j = %lu; j >= 0; j--) {\n"
                    "        c[(j+1) * %u + %u] = c[j * %u + %u];\n"
                    "    }\n"
                    "    c[%u] = tmp;\n",
                k, subdims[1].y - 2, regPitch / outVecLen,
                i, regPitch / outVecLen, i, i);
        p += strlen(p);
    }
    sprintf(tmp, "\n"
                 "for (k0 = 0; k0 < skewRow.y; k0++) {\n"
                 "    int j;\n"
                 "    %s tmp;\n"
                 "%s"
                 "}\n"
                 "\n",
                 outTypeName, tmp1);
    kgenAddStmt(ctx, tmp);

    // write back the tile evaluated
    tra = isMatrixAccessColMaj(CLBLAS_TRMM, kextra->flags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRMM, kextra->flags, MATRIX_B);
    sprintf(tmp, "coordA.%c = workItemM - startM;\n"
                 "coordB.%c = workItemN - startN;\n\n",
            vect[tra], vect[trb]);
    kgenAddStmt(ctx, tmp);
    kgenBeginBranch(ctx, NULL);
    trb = isMatrixAccessColMaj(CLBLAS_TRMM, kextra->flags, MATRIX_C);
    sprintf(tmp, "__global %s *B = C + offB + start%c * ldb + start%c;\n\n",
            typeName, coordNames[trb], coordNames[1 - trb]);

    kgenAddStmt(ctx, tmp);
    generateResultUpdateOld(ctx, CLBLAS_TRMM, &gset, NULL, NULL);
    kgenEndBranch(ctx, NULL);
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
    int side = (blasArgs->side == clblasRight);
    size_t sizes[2] = {blasArgs->M, blasArgs->N};
    size_t offs[2] = {blasArgs->offsetM, blasArgs->offsetN};

    (void)extra;

    switch (blasArgs->kernType) {
    case CLBLAS_COMPUTING_KERNEL:
        initSizeKarg(&args[0], blasArgs->M);
        initSizeKarg(&args[1], blasArgs->N);
        assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
        INIT_KARG(&args[3], blasArgs->scimage[0]);
        INIT_KARG(&args[4], blasArgs->scimage[1]);
        initMemobjKarg(&args[5], blasArgs->B, NULL, 0, 0);
        initSizeKarg(&args[6], blasArgs->ldb.matrix);
        initSizeKarg(&args[7], blasArgs->offsetM);
        initSizeKarg(&args[8], blasArgs->offsetN);
        initSizeKarg(&args[9], blasArgs->K);
        initSizeKarg(&args[10], blasArgs->offBX);
        break;
    case CLBLAS_PREP_A_KERNEL:
        initSizeKarg(&args[0], sizes[side]);
        initMemobjKarg(&args[1], blasArgs->A, NULL, 0, 0);
        initSizeKarg(&args[2], blasArgs->lda.matrix);
        INIT_KARG(&args[3], blasArgs->scimage[0]);
        initSizeKarg(&args[4], offs[side]);
        initSizeKarg(&args[5], blasArgs->K);
        initSizeKarg(&args[6], blasArgs->offA);
        break;
    case CLBLAS_PREP_B_KERNEL:
        INIT_KARG(&args[0], blasArgs->order);
        INIT_KARG(&args[1], blasArgs->side);
        initSizeKarg(&args[2], sizes[1 - side]);
        initMemobjKarg(&args[3], blasArgs->B, NULL, 0, 0);
        initSizeKarg(&args[4], blasArgs->ldb.matrix);
        INIT_KARG(&args[5], blasArgs->scimage[1]);
        initSizeKarg(&args[6], offs[1 - side]);
        initSizeKarg(&args[7], blasArgs->K);
        initSizeKarg(&args[8], blasArgs->offBX);
        break;
    default:
        //this should not happen
        break;
    }
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    cl_ulong size;
    const CLBlasKargs *kargs = (const CLBlasKargs*)kernelArgs;
    size = matrBlockSize(&dim[1], MATRIX_C, dtype, kargs->side);
    return (size * dtypeSize(dtype) <= ldsSize);
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    const CLBlasKargs *kargs = args;
    size_t m, n, k;
    (void)extra;

    //form inner subdims with respect of multiplication side
    if (kargs->side == clblasRight) {
        m = kargs->N;
        n = kargs->M;
        //original N was stored in K
        k = kargs->K;
    }
    else {
        m = kargs->M;
        n = kargs->N;
        //original M was stored in K
        k = kargs->K;
    }

    if (kargs->kernType != CLBLAS_COMPUTING_KERNEL) {
        size_t whole, part;
        size_t nrGroups;

        // each thread gets one block
        if (kargs->kernType == CLBLAS_PREP_A_KERNEL) {
            whole = m;
            part = subdims[0].itemY;
        }
        else {
            whole = n;
            part = subdims[0].itemX;
        }

        nrGroups = whole / part + (whole % part != 0);
        nrGroups *= (k / subdims[0].bwidth +
                    (k % subdims[0].bwidth != 0));
        threads[0] = pgran->wgSize[0] * nrGroups;
        threads[1] = pgran->wgSize[1];
    }
    else {
        calcGlobalThreads(threads, &subdims[0], pgran, m, n);
    }
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_2D);
}

void
initTrmmImgPattern(MemoryPattern *mempat)
{
    mempat->name = "Image based block trmm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &imgSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS;
    mpatExtra.mobjA = CLMEM_IMAGE;
    mpatExtra.mobjB = CLMEM_IMAGE;
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
