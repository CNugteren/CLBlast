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
 * gemm image based generators
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <clBLAS.h>
#include <matrix_dims.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <dis_warning.h>

#include "blas_kgen_legacy.h"
#include "../gen_helper.h"
#include "gen_helper_legacy.h"

static CLBLASMpatExtra mpatExtra;

static const char *prepareImagesGemmDeclA =
    "void __kernel\n"
    "%cprepareImageA(\n"
    "    clblasOrder order,\n"
    "    clblasTranspose transA,\n"
    "    uint M,\n"
    "    uint K,\n"
    "    __global %s *A,\n"
    "    uint lda,\n"
    "    __write_only image2d_t imgA,\n"
    "    uint offsetA)\n";

static const char *prepareImagesGemmDeclB =
    "void __kernel\n"
    "%cprepareImageB(\n"
    "    clblasOrder order,\n"
    "    clblasTranspose transB,\n"
    "    uint N,\n"
    "    uint K,\n"
    "    __global %s *B,\n"
    "    uint ldb,\n"
    "    __write_only image2d_t imgB,\n"
    "    uint offsetB)\n";


static const char *imgGemmDecl =
    "__attribute__((reqd_work_group_size(%lu, %lu, 1)))\n"
    "void __kernel\n"
    "%cgemmImg(\n"
    "    const uint M,\n"
    "    const uint N,\n"
    "    const uint K,\n"
    "    const %s alpha,\n"
    "    const __read_only image2d_t A,\n"
    "    const __read_only image2d_t B,\n"
    "    const %s beta,\n"
    "    __global %s *C,\n"
    "    const uint ldc,\n"
    "    const uint offsetC)\n";

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

static int
imgGetPerf(
    unsigned int kflags,
    const void *args);

static SolverOps imgSops = {
    genWrapper,
    assignKargs,
    isFitToLDS,
    imgGetPerf,
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
    char tmp[4096], conjStr[1024];
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    CopyImgFuncs copyImgFuncs;
    DataType dtype = kextra->dtype;
    BlasGenSettings gset;
    unsigned int vecLen;
    unsigned int tsize;
    const char *typeName;
    char fpref;
    bool b;
    size_t localBufSize;
    ssize_t ret;
    const char *conjCond;

    const char *functionHeadA =
        "int tra, aligned;\n"
        "const uint bpr = (K + %lu) / %lu;\n"
        "uint m = (gid / bpr) * %lu;\n"
        "uint k = (gid %% bpr) * %lu;\n"
        "uint x, y;\n"
        "__local %s temp[%lu];\n"
        "\n"
        "A += offsetA;\n"
        "tra = (!transA && order == clblasColumnMajor) ||\n"
        "      (transA && order == clblasRowMajor);\n"
        "if (m >= M) {\n"
        "     return;\n"
        "}\n";

    const char *functionHeadB =
        "int trb, aligned;\n"
        "const uint bpr = (K + %lu) / %lu;\n"
        "const uint n = (gid / bpr) * %lu;\n"
        "const uint k = (gid %% bpr) * %lu;\n"
        "uint x, y;\n"
        "__local %s temp[%lu];\n"
        "\n"
        "B += offsetB;\n"
        "trb = (!transB && order == clblasRowMajor) ||\n"
        "      (transB && order == clblasColumnMajor);\n"
        "if (n >= N) {\n"
        "    return;\n"
        "}\n";

    // Distribute blocks across compute units and copy matrix A to image.
    // Transposition and filling with zeros in unaligned cases is made using
    // buffer in local memory.
    const char *copyToImageA =
        "//copy matrix A block\n"
        "y = m + %u <= M ? %u : M - m;\n"
        "x = k + %u <= K ? %u : K - k;\n"
        "aligned = (x == %u) && (y == %u) && %d;\n"
        "int atcase = aligned * 10 + tra;\n"
        "%s" // conjugated check
        "if (atcase != 10) {\n"
        "    %s((__local float4*)temp);\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
        "switch(atcase) {\n"
        "case 10: //aligned, not transposed\n"
        "    %s(imgA, k / %u, m, (GPtr)A, m, k, lda);\n"
        "    break;\n"
        "%s" // conjugated case
        "case 1: //not aligned, transposed\n"
        "    // generic transposed global to local\n"
        "    %s((LPtr)temp, (GPtr)A, k, m, x, y, %u, lda);\n"
        "    break;\n"
        "case 0: //not aligned, not transposed\n"
        "    // generic global to local\n"
        "    %s((LPtr) temp, (GPtr)A, m, k, y, x, %u, lda);\n"
        "    break;\n"
        "case 11: //aligned, transposed\n"
        "    // optimized transposed global to local\n"
        "    %s((LPtr) temp, (GPtr)A, k, m, lda);\n"
        "    break;\n"
        "}\n"
        "if (atcase != 10) {\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    %s(imgA, k / %u, m, (LPtr) temp);\n"
        "}\n"
        "\n";

    const char *copyToImageB =
            "//copy matrix B block\n"
            "y = n + %u <= N ? %u : N - n;\n"
            "x = k + %u <= K ? %u : K - k;\n"
            "aligned = (x == %u) && (y == %u) && %d;\n"
            "int atcase = aligned * 10 + trb;\n"
            "%s" // conjugated check
            "if (atcase != 10) {\n"
            "    %s((__local float4*)temp);\n"
            "    barrier(CLK_LOCAL_MEM_FENCE);\n"
            "}\n"
            "switch (atcase) {\n"
            "case 10: //aligned, not transposed\n"
            "    %s(imgB, k / %u, n, (GPtr)B, n, k, ldb);\n"
            "    break;\n"
            "%s" // conjugated case
            "case 1: //not aligned, transposed\n"
            "    // generic transposed global to local\n"
            "    %s((LPtr)temp, (GPtr)B, k, n, x, y, %u, ldb);\n"
            "    break;\n"
            "case 0: //not aligned, not transposed\n"
            "    // generic global to local\n"
            "    %s((LPtr)temp, (GPtr)B, n, k, y, x, %u, ldb);\n"
            "    break;\n"
            "case 11: //transposed, aligned\n"
            "    // optimized transposed global to local\n"
            "    %s((LPtr)temp, (GPtr)B, k, n, ldb);\n"
            "    break;\n"
            "}\n"
            "if (atcase != 10) {\n"
            "    barrier(CLK_LOCAL_MEM_FENCE);\n"
            "    %s(imgB, k / %u, n, (LPtr)temp);\n"
            "}\n"
            "\n";

    memset(&copyImgFuncs, 0, sizeof(copyImgFuncs));
    memset(&gset, 0, sizeof(gset));

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    tsize = dtypeSize(dtype);

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);
    declareBlasEnums(ctx);

    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;

    // generate necessary memory to image copying functions
    generateImageCopyFuncs(&copyImgFuncs, ctx, CLBLAS_GEMM, &gset);

    kgenAddBlankLine(ctx);
    vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    if (kextra->kernType == CLBLAS_PREP_A_KERNEL) {
        sprintf(tmp, prepareImagesGemmDeclA, fpref, typeName, typeName);
        kgenDeclareFunction(ctx, tmp);
        ret = kgenBeginFuncBody(ctx);

        // same local buffer is used for both matrix A and matrix B blocks
        localBufSize = subdims[1].y * fl4RowWidth(subdims[1].bwidth, tsize);
        localBufSize *= vecLen;

        kgenDeclareGroupID(ctx, "gid", pgran);
        sprintf(tmp, functionHeadA,
                subdims[1].bwidth - 1, subdims[1].bwidth,
                subdims[1].y, subdims[1].bwidth,
                typeName, localBufSize);
        kgenAddStmt(ctx, tmp);

        if (isComplexType(dtype)) {
            conjCond = "atcase += ((atcase == 10) && "
                    "(transA == clblasConjTrans)) ? 100 : 0;\n";
            sprintf(conjStr, "case 110: //conjugated, not transposed, aligned\n"
                             "    %s((LPtr)temp, (GPtr)A, m, k, lda);\n"
                             "    break;\n",
                    copyImgFuncs.globalToLocal[MATRIX_A]);
        }
        else {
            conjCond = "";
            strcpy(conjStr, "");
        }

        sprintf(tmp, copyToImageA,
                subdims[1].y, subdims[1].y, // y = m + dy <= M ?...
                subdims[1].bwidth, subdims[1].bwidth, // x = k + bw <= K ?...
                subdims[1].bwidth, subdims[1].y, // aligned = (x==bw1)&&(y==dy1)
                (kextra->flags & KEXTRA_NO_COPY_VEC_A) == 0,
                conjCond,
                copyImgFuncs.zeroBlock[MATRIX_A],
                copyImgFuncs.globalToImage[MATRIX_A],
                vecLen,
                conjStr,
                copyImgFuncs.globalToLocalTransposedGeneric[MATRIX_A],
                subdims[1].bwidth,
                copyImgFuncs.globalToLocalGeneric[MATRIX_A],
                subdims[1].bwidth,
                copyImgFuncs.globalToLocalTransposed[MATRIX_A],
                copyImgFuncs.localToImage[MATRIX_A],
                vecLen);
        kgenAddStmt(ctx, tmp);
    }
    else { // PREP_B
        sprintf(tmp, prepareImagesGemmDeclB, fpref, typeName, typeName);
        kgenDeclareFunction(ctx, tmp);
        ret = kgenBeginFuncBody(ctx);

        // same local buffer is used for both matrix A and matrix B blocks
        localBufSize = subdims[1].x * fl4RowWidth(subdims[1].bwidth, tsize);
        localBufSize *= vecLen;

        kgenDeclareGroupID(ctx, "gid", pgran);
        sprintf(tmp, functionHeadB,
                subdims[1].bwidth - 1, subdims[1].bwidth,
                subdims[1].x, subdims[1].bwidth,
                typeName, localBufSize);
        kgenAddStmt(ctx, tmp);

        if (isComplexType(dtype)) {
            conjCond = "atcase += ((atcase == 10) && "
                    "(transB == clblasConjTrans)) ? 100 : 0;\n";
            sprintf(conjStr, "case 110: //conjugated, not transposed, aligned\n"
                             "    %s((LPtr)temp, (GPtr)B, n, k, ldb);\n"
                             "    break;\n",
                    copyImgFuncs.globalToLocal[MATRIX_B]);
        }
        else {
            conjCond = "";
            strcpy(conjStr, "");
        }

        sprintf(tmp, copyToImageB,
                subdims[1].x, subdims[1].x, // y = n + dy <= N ?...
                subdims[1].bwidth, subdims[1].bwidth, // x = k + bw <= K ?...
                subdims[1].bwidth, subdims[1].x, // aligned = (x==bw1)&&(y==dx1)
                (kextra->flags & KEXTRA_NO_COPY_VEC_B) == 0,
                conjCond,
                copyImgFuncs.zeroBlock[MATRIX_B],
                copyImgFuncs.globalToImage[MATRIX_B],
                vecLen,
                conjStr,
                copyImgFuncs.globalToLocalTransposedGeneric[MATRIX_B],
                subdims[1].bwidth,
                copyImgFuncs.globalToLocalGeneric[MATRIX_B],
                subdims[1].bwidth,
                copyImgFuncs.globalToLocalTransposed[MATRIX_B],
                copyImgFuncs.localToImage[MATRIX_B],
                vecLen);
        kgenAddStmt(ctx, tmp);
    }

    kgenEndFuncBody(ctx);

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

// global memory based kernel generator
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
    char fpref;
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
    int tra, trb;
    char vect[2] = {'y', 'x'};

    const char *coordConstants =
        "const uint workItemM = get_global_id(0) * %lu;\n"
        "const uint workItemN = get_global_id(1) * %lu;\n"
        "const int2 skewRow = (int2)(0, get_local_id(0) %% %lu);\n"
        "uint vectK = (K + %u) / %u;\n";

    /*
     *  template for image based gemm preparation part
     *  for two dimensional work space
     */
    const char *localVariables =
        "uint k0;\n"
        "int2 coordA = (int2)(0, workItemM);\n"
        "int2 coordB = (int2)(0, workItemN);\n"
        "%s c[%u];\n\n";

    tsize = dtypeSize(dtype);
    vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    if (isComplexType(dtype)) {
        regPitch = (unsigned int)subdims[1].x;
    }
    else {
        regPitch = (unsigned int) fl4RowWidth(subdims[1].x, tsize) *
                    sizeof(cl_float4) / tsize;
    }

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;
    initKernelVarNames(&gset.varNames, kextra->flags);

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations and auxiliary functions
    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    // now, generate the kernel

    sprintf(tmp, imgGemmDecl, pgran->wgSize[0], pgran->wgSize[1], fpref,
            typeName, typeName, typeName);
    kgenDeclareFunction(ctx, tmp);
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
    kgenAddStmt(ctx, "if ((workItemM >= M) ||"
                         "(workItemN >= N)) {\n"
                     "    return;\n"
                     "}\n");

    kgenAddStmt(ctx, "C += offsetC;\n");

    // zero C block
    sprintf(tmp, "for (k0 = 0; k0 < %u; k0++) {\n"
                 "    c[k0] = 0;\n"
                 "}\n\n",
            nrRegs);
    kgenAddStmt(ctx, tmp);

    // block multiplication inlined function
    sprintf(tmp, "for (k0 = 0; k0 < vectK; k0 += %lu)",
            subdims[1].bwidth / vecLen);
    kgenBeginBranch(ctx, tmp);

    mulOpts.aMobj = CLMEM_IMAGE;
    mulOpts.bMobj = CLMEM_IMAGE;
    mulOpts.flags = BLKMUL_OUTPUT_PRIVATE | BLKMUL_SKEW_ROW | BLKMUL_INLINE;
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
    sprintf(tmp, "\ncoordA.x += %lu;\n"
                 "coordB.x += %lu;\n",
            subdims[1].bwidth / vecLen, subdims[1].bwidth / vecLen);
    kgenAddStmt(ctx, tmp);

    kgenEndBranch(ctx, NULL);

    // reorder the given solution
    outVecLen = isComplexType(dtype) ? 1 : vecLen;
    p = tmp1;
    for (i = 0; i < regPitch / outVecLen; i++) {
        unsigned int k = (unsigned int)(subdims[1].y - 1) *
                         regPitch / outVecLen + i;

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

    tra = isMatrixAccessColMaj(CLBLAS_GEMM, kextra->flags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_GEMM, kextra->flags, MATRIX_B);
    sprintf(tmp, "coordA.%c = workItemM;\n"
                 "coordB.%c = workItemN;\n\n",
            vect[tra], vect[trb]);
    kgenAddStmt(ctx, tmp);

    // write back the tile evaluated
    generateResultUpdateOld(ctx, CLBLAS_GEMM, &gset, NULL, NULL);

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

    (void)extra;

    switch (blasArgs->kernType) {
    case CLBLAS_COMPUTING_KERNEL:
        // arguments for computational kernel
        initSizeKarg(&args[0], blasArgs->M);
        initSizeKarg(&args[1], blasArgs->N);
        initSizeKarg(&args[2], blasArgs->K);
        assignScalarKarg(&args[3], &(blasArgs->alpha), blasArgs->dtype);
        INIT_KARG(&args[4], blasArgs->scimage[0]);
        INIT_KARG(&args[5], blasArgs->scimage[1]);
        assignScalarKarg(&args[6], &(blasArgs->beta), blasArgs->dtype);
        initMemobjKarg(&args[7], blasArgs->C, NULL, 0, 0);
        initSizeKarg(&args[8], blasArgs->ldc.matrix);
        initSizeKarg(&args[9], blasArgs->offCY);
        break;
    case CLBLAS_PREP_A_KERNEL:
        INIT_KARG(&args[0], blasArgs->order);
        INIT_KARG(&args[1], blasArgs->transA);
        initSizeKarg(&args[2], blasArgs->M);
        initSizeKarg(&args[3], blasArgs->K);
        initMemobjKarg(&args[4], blasArgs->A, NULL, 0, 0);
        initSizeKarg(&args[5], blasArgs->lda.matrix);
        INIT_KARG(&args[6], blasArgs->scimage[0]);
        initSizeKarg(&args[7], blasArgs->offA);
        break;
    case CLBLAS_PREP_B_KERNEL:
        INIT_KARG(&args[0], blasArgs->order);
        INIT_KARG(&args[1], blasArgs->transB);
        initSizeKarg(&args[2], blasArgs->N);
        initSizeKarg(&args[3], blasArgs->K);
        initMemobjKarg(&args[4], blasArgs->B, NULL, 0, 0);
        initSizeKarg(&args[5], blasArgs->ldb.matrix);
        INIT_KARG(&args[6], blasArgs->scimage[1]);
        initSizeKarg(&args[7], blasArgs->offBX);
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
    (void)extra;

    if (kargs->kernType != CLBLAS_COMPUTING_KERNEL) {
        const size_t *whole, *part;
        size_t nrGroups;

        // each thread gets one block

        if (kargs->kernType == CLBLAS_PREP_A_KERNEL) {
            whole = &kargs->M;
            part = &subdims[0].itemY;
        }
        else {
            whole = &kargs->N;
            part = &subdims[0].itemX;
        }

        nrGroups = *whole / *part + (*whole % *part != 0);
        nrGroups *= (kargs->K / subdims[0].bwidth +
                    (kargs->K % subdims[0].bwidth != 0));
        threads[0] = pgran->wgSize[0] * nrGroups;
        threads[1] = pgran->wgSize[1];
    }
    else {
        calcGlobalThreads(threads, &subdims[0], pgran, kargs->M, kargs->N);
    }
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_2D);
}

void
initGemmImgPattern(MemoryPattern *mempat)
{
    mempat->name = "Image based block gemm";
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
imgGetPerf(
    unsigned int kflags,
    const void *args)
{
    (void)args;
    (void)kflags;

    return PPERF_POOR;
}
