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
 * Image based trsm generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>

#include <matrix_dims.h>

#include "blas_kgen_legacy.h"
#include "gen_helper_legacy.h"
#include "trsm_kgen_legacy.h"
#include "../gen_helper.h"
#include "../trsm_kgen.h"
#include <dis_warning.h>

static const char *trsmImDecl =
    "__attribute__((reqd_work_group_size(%lu, %lu, 1)))\n"
    "void __kernel\n"
    "%ctrsmIm(\n"
    "    uint %c,\n"
    "    uint %c,\n"
    "    %s alpha,\n"
    "    __read_only image2d_t A,\n"
    "    __global %s *B,\n"
    "    uint ldb,\n"
    "    uint startRow,\n"
    "    uint finishRow,\n"
    "    uint offB)\n";

/*
 *  template for memory object based trsm preparation part
 *  for one dimensional work space
 */
static const char *trsmImPrep1D =
    "uint m0, k0;\n"
    "__local %s tempC[%lu];\n"
    "%s c[%u];\n"
    "const int lid = get_local_id(0);\n"
    "const int skew = lid %% %lu;\n"
    "%s"                                    // groups per Panel variable
    "uint blockN;\n"
    "uint x, y, imx, imy;\n"
    "uint2 coordA, coordB;\n"
    "\n"
    "const uint currN = get_global_id(0) / %u * %lu;\n"       // group ID
    "\n";

static const char *readRectBlock =
    "y = (currN + %lu <= N) ? %lu : N - currN;\n"
    "x = (k0 + %lu <= finishRow) ? %lu : finishRow - k0;\n"
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
    "x = (k0 + %lu <= finishRow) ? %lu : finishRow - k0;\n"
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
wrapper(
    char *buf,
    size_t buflen,
    const struct SubproblemDim *subdims,
    const struct PGranularity *pgran,
    void *extra);

static ssize_t
generator(
    char *buf,
    size_t buflen,
    const struct SubproblemDim *subdims,
    const struct PGranularity *pgran,
    void *extra);

static ssize_t
prepGenerator(
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

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *dims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);


static void
imgPackMode(
    const void *extra,
    const SubproblemDim *dims,
    int dataID,
    unsigned int *packRate,
    clblasOrder *packOrder);

static SolverFlags
solverFlags(void);

static SolverOps solverOps = {
    wrapper,
    assignKargs,
    isFitToLDS,
    NULL,
    NULL,
    calcNrThreads,
    imgPackMode,
    solverFlags,
    NULL, //fixupArgs
    NULL, //getDefaultDecomp
   	NULL, //getDecompList
   	NULL,
   	NULL
};

static CLBLASMpatExtra mpatExtra;

/* Prepare A kernel begin */

static const char *trsmPrepDecl =
    "void __kernel\n"
    "%ctrsmPrepare(\n"
    "    uint %c,\n"
    "    __global %s *A,\n"
    "    uint lda,\n"
    "    __write_only image2d_t imA,\n"
    "    uint startRow,\n"
    "    uint offA)\n";

/*
 * template for memory object based trsm preparation part
 * for one dimensional work space
 */
static const char *trsmPrep1D =
    "__local %s tempA[%lu];\n"
    "__local %s tempC[%lu];\n"
    "int lid, gid;\n"
    "uint currM, k0;\n"
    "uint x, y, imx, imy;\n"
    "\n"
    "lid = get_local_id(0);\n"
    "gid = get_global_id(0) / %u;\n"      // group ID
    "A += offA;\n"
    "\n";

static const char *readSquareBlock =
    "y = (currM + %lu <= M) ? %lu : M - currM;\n"
    "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
    "if ((y == %lu) && (x == %lu)) {\n"
    // just read with an optimized function
    "    %s((LPtr)temp%c, (GPtr)A, currM, k0, lda);\n"
    "}\n"
    "else {\n"
    "    %s((__local float4*)temp%c);\n"          // zeroing
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
    "    %s((__local float4*)temp%c);\n"          // zeroing
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    // read and transpose with slow function
    "    %s((LPtr)temp%c, (GPtr)A, k0, currM, x, y, %lu, lda);\n"
    "}\n\n";

static const char *readSquareBlockTransOpt =
    // read and transpose with an optimized function
    "%s((LPtr)temp%c, (GPtr)A, k0, currM, lda);\n";


static bool
useTransposedMul(const SubproblemDim *dims, DataType dtype, bool trb)
{
    unsigned int vecLen;

    vecLen = sizeof(cl_float4) / dtypeSize(dtype);

    return (!(trb || isComplexType(dtype) || (dims[1].x % vecLen)));
}

static size_t
calcPitchB(const SubproblemDim *dim, DataType dtype, bool transpMul)
{
    size_t ret;
    size_t tsize;

    tsize = dtypeSize(dtype);
    ret = (transpMul) ? dim->x : dim->bwidth;
    ret = fl4RowWidth(ret, tsize) * sizeof(cl_float4) / tsize;

    return ret;
}

static void
genPrepareSquareBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    bool tra,
    char c,
    bool opt)
{
    char tmp[1024];
    size_t pitch;
    const char *readBlock;

    pitch = matrBlockPitch(dim, MATRIX_A, dtype, clblasLeft);
    if (opt) {
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
genPrepZeroBlockC(
    struct KgenContext *ctx,
    const ZeroFuncs *zeroFuncs)
{
    char tmp[1024];
    sprintf(tmp, "%s((__local float4*)tempC);\n", zeroFuncs->names[MATRIX_A]);
    kgenAddStmt(ctx, tmp);
}

static void
genWriteBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const CopyBufFuncs *copyFuncs)
{
    char tmp[1024];

    sprintf(tmp, "%s(imA, imx, imy, (LPtr)tempC, %lu, %lu, %lu);\n",
        copyFuncs->write, dim[0].y, dim[0].y, dim[0].y);
    kgenAddStmt(ctx, tmp);
}

static void
getBufferPos(struct KgenContext *ctx, bool isU) //n -> x,y buffer
{
    kgenDeclareFunction(ctx, "void\ngetBufferPos(uint n, uint startRow, "
                                                "uint width, uint *y, "
                                                "uint *x)\n");
    kgenBeginFuncBody(ctx);
    if (isU) {
        //n from beginning
        kgenAddStmt(ctx, "n += (2 * width - startRow + 1) * (startRow) / 2;\n");
        kgenAddStmt(ctx, "*y = trunc((2 * width + 1) - "
                               "sqrt((2 * width + 1) *"
                               "(2 * width + 1) - 8 * n)) / 2;\n");
        kgenAddStmt(ctx, "*x = *y + n - (2 * width - *y + 1) * (*y) / 2;\n");
    }
    else {
        //n from beginning
        kgenAddStmt(ctx, "n += startRow * (startRow + 1) / 2;\n");
        kgenAddStmt(ctx, "*y = trunc((-0.5 + sqrt(2.0 * n + 0.25)));\n");
        kgenAddStmt(ctx, "*x = n - (*y) * (*y + 1) / 2;\n");
    }
    kgenEndFuncBody(ctx);

    kgenAddBlankLine(ctx);
}

static void
genGetImagePos(
    struct KgenContext *ctx,
    const SubproblemDim *subdims,
    DataType dtype,
    const char *blockName,
    bool tra) //n -> x,y image
{
    char tmp[1024];
    const char *parName;
    const char *op[2] = {"/", "%"};

    parName = (tra) ? "bpc" : "bpr";

    sprintf(tmp, "imy = %s %s %s * %lu;\n"
                 "imx = (%s %s %s) * %lu;\n",
            blockName, op[tra], parName, subdims[0].y,
            blockName, op[1 - tra], parName,
            subdims[0].y * dtypeSize(dtype) / sizeof(cl_float4));
    kgenAddStmt(ctx, tmp);
}

// global memory to image converter
static ssize_t
prepGenerator(
    char *buf,
    size_t buflen,
    const struct SubproblemDim *subdims,
    const struct PGranularity *pgran,
    void *extra)
{
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    char tmp[1024];
    const char *typeName;
    CopyBufFuncs copyFuncs;
    ZeroFuncs zeroFuncs;
    char fpref;
    DataType dtype = kextra->dtype;
    KernelExtraFlags kflags = kextra->flags;
    ssize_t ret;
    size_t pitchAB;
    bool b;
    bool tra, trb, isU, transpMul;
    BlasGenSettings gset;

    if (pgran->wgDim != 1) {
        return -EINVAL;
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);
    isU = isMatrixUpper(kflags);

    // at first, generate needed declarations and auxiliary functions

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    if (isComplexType(dtype)) {
        genComplexMathOperators(ctx, dtype);
    }

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;

    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_TRSM, &gset,
                         BCHF_MATRIX_A | BCHF_WRITE_OUTPUT | BCHF_IMAGE_WRITE);
    generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                         ZF_MATRIX_A);

    //matrix inversion function
    genInvertingBlockFunc(ctx, (unsigned int)subdims[0].bwidth, dtype, isU);

    //coordinates calculation
    getBufferPos(ctx, isU);

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    // now, generate the kernel

    sprintf(tmp, trsmPrepDecl, fpref, 'M', typeName,
        typeName, typeName, typeName);

    kgenDeclareFunction(ctx, tmp);
    ret = kgenBeginFuncBody(ctx);

    transpMul = useTransposedMul(subdims, dtype, trb);
    if (!transpMul) {
        sprintf(tmp, "const int bpr = get_image_width(imA) / %lu;\n",
                subdims[0].y / (sizeof(cl_float4) / dtypeSize(dtype)));
    }
    else {
        sprintf(tmp, "const int bpc = get_image_height(imA) / %lu;\n",
                subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    /*
     * Calculate local buffer pitches, and then insert the
     * preparative code
     */
    pitchAB = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);
    sprintf(tmp, trsmPrep1D, typeName, pitchAB * subdims[0].y,
            typeName, pitchAB * subdims[0].y, pgran->wgSize[0]);
    ret = kgenAddStmt(ctx, tmp);

    sprintf(tmp, "getBufferPos(gid, startRow / %lu, (M + %lu) / %lu, &currM, &k0);\n",
            subdims[0].y, subdims[0].y - 1, subdims[0].y);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "currM *= %lu;\n"
            "k0 *= %lu;\n", subdims[0].y, subdims[0].y);
    kgenAddStmt(ctx, tmp);

    genGetImagePos(ctx, subdims, dtype, "gid", transpMul);

    kgenBeginBranch(ctx, "if (currM == k0)");
    genPrepareSquareBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                          tra, 'A', !(kextra->flags & KEXTRA_TAILS_M));
    genPrepZeroBlockC(ctx, &zeroFuncs);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    if (kextra->flags & KEXTRA_UNIT_DIAGONAL) {
        sprintf(tmp, "if (lid < %lu) {\n"
                     "    tempA[lid * %lu + lid] = %s;\n"
                     "}\n",
                subdims[0].bwidth, pitchAB, strOne(dtype));
        kgenAddStmt(ctx, tmp);
        kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
        kgenAddBlankLine(ctx);
    }

    sprintf(tmp, "if (lid < %lu)", subdims[0].bwidth);
    kgenBeginBranch(ctx, tmp);
    sprintf(tmp, "invert(tempA, tempC, lid, (currM + %lu > M) ? "
                                            "M - currM : %lu);\n",
            subdims[0].y, subdims[0].y);
    kgenAddStmt(ctx, tmp);
    kgenEndBranch(ctx, NULL);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenEndBranch(ctx, NULL);

    kgenBeginBranch(ctx, "else");
    genPrepareSquareBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs, tra,
                          'C', !(kextra->flags & KEXTRA_TAILS_M));
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenEndBranch(ctx, NULL);

    genWriteBlock(ctx, subdims, &copyFuncs);
    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

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
genPrepareRectBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    bool trb,
    char c,
    bool opt)
{
    char tmp[1024];
    size_t pitch;
    const char *readBlock;
    size_t bsizes[2] = {dim->bwidth, dim->x};

    /*
     * NOTE: in case of accessing to B in the non transposed way
     *       block multiplication is done with transposed block B
     */
    pitch = calcPitchB(dim, dtype, !trb);
    if (opt) {
        readBlock = (trb) ? readRectBlockTransOpt : readRectBlockOpt;
        sprintf(tmp, readBlock, copyFuncs->read[MATRIX_B], c);
    }
    else {
        readBlock = (trb) ? readRectBlockTrans : readRectBlock;
        sprintf(tmp, readBlock, bsizes[trb], bsizes[trb], bsizes[1 - trb],
                bsizes[1 - trb], bsizes[trb], bsizes[1 - trb],
                copyFuncs->read[MATRIX_B], c, zeroFuncs->names[MATRIX_B], c,
                copyFuncs->readGeneric[MATRIX_B], c, pitch);
    }
    kgenAddStmt(ctx, tmp);
}

static void
getNblock(struct KgenContext *ctx, bool isU) //x, y -> n
{
    kgenDeclareFunction(ctx, "void\ngetNBlock(uint y, uint x, uint startRow, "
        "uint width, uint *n)\n");
    kgenBeginFuncBody(ctx);
    if (isU) {
        kgenAddStmt(ctx, "*n = ((2 * width - y + 1) * y - "
            "(2 * width - startRow + 1) * startRow) / 2 + x - y;\n");
    }
    else {
        kgenAddStmt(ctx, "*n = (y * (y + 1) - startRow * (startRow + 1)) / 2 + x;\n");
    }
    kgenEndFuncBody(ctx);
    kgenAddBlankLine(ctx);
}

static void
genMultiplication(
    struct KgenContext *ctx,
    const SubproblemDim *dims,
    DataType dtype,
    const char *blkmulName,
    BlkMulFlags mulFlags)
{
    char tmp[1024];
    size_t u;
    unsigned int l1Pans;

    l1Pans = (unsigned int)(dims[0].x / dims[1].x);
    if (mulFlags & BLKMUL_TRANSPOSED_B) {
        u = 1;
    }
    else {
        u = matrBlockPitch(dims, MATRIX_B, dtype, clblasLeft);
    }

    // find image position and invoke the multiplier
    sprintf(tmp, "getNBlock(m0 / %lu, k0 / %lu, startRow / %lu, "
                           "(M + %lu) / %lu, &blockN);\n",
            dims[0].y, dims[0].y, dims[0].y, dims[0].y - 1, dims[0].y);
    kgenAddStmt(ctx, tmp);
    genGetImagePos(ctx, dims, dtype, "blockN", (mulFlags & BLKMUL_TRANSPOSED_B) != 0);
    sprintf(tmp, "%s(A, (int2)(imx, imy + lid / %u * %lu), \n"
                  "   (LPtr)(tempC + (lid %% %u * %lu) * %lu),\n"
                  "   c, skew);\n",
            blkmulName, l1Pans, dims[1].y, l1Pans, dims[1].x, u);
    kgenAddStmt(ctx, tmp);
}

static void
genReorderSolution(
    struct KgenContext *ctx,
    const SubproblemDim *subdims,
    const char *outTypeName,
    unsigned int colRegs)
{
    char tmp[1024], tmp1[1024];
    char *p;
    unsigned i;

    sprintf(tmp, "void\n"
                 "reorderResult(%s *c, int skew)",
            outTypeName);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    sprintf(tmp, "%s tmp;\n"
                 "int i, j;\n",
           outTypeName);
    kgenAddStmt(ctx, tmp);

    p = tmp1;
    for (i = 0; i < colRegs; i++) {
        unsigned int k = (unsigned int)(subdims[1].y - 1) * colRegs + i;

        sprintf(p,  "\n"
                    "    tmp = c[%u];\n"
                    "    for (j = %lu; j >= 0; j--) {\n"
                    "        c[(j+1) * %u + %u] = c[j * %u + %u];\n"
                    "    }\n"
                    "    c[%u] = tmp;\n",
                k, subdims[1].y - 2, colRegs, i, colRegs, i, i);
        p += strlen(p);
    }

    sprintf(tmp, "\n"
                 "for (i = 0; i < skew; i++) {\n"
                 "%s"
                 "}\n"
                 "\n",
            tmp1);
    kgenAddStmt(ctx, tmp);

    kgenEndFuncBody(ctx);
    kgenAddBlankLine(ctx);
}

static void
initKernelVarNames(KernelVarNames *kvars, KernelExtraFlags kflags)
{
    kvars->A = "imgA";
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

// image based kernel generator
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
    CLBLASKernExtra kextraTmp = *kextra;
    char tmp[1024], tmp1[1024];
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
    char fpref;
    DataType dtype = kextra->dtype;
    ssize_t ret;
    BlasGenSettings gset;
    BlkMulOpts mulOpts;
    BlkMulFlags mulFlags;
    size_t pitchAB;
    size_t u;
    bool b;
    bool isU;
    bool areTails;
    const char *outTypeName;
    unsigned int nrRegs, colRegs;
    KernelExtraFlags kflags = kextra->flags;
    size_t tsize;
    unsigned int vecLen = sizeof(cl_float4) / dtypeSize(dtype);
    UpdateResultFlags upFlags;
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

    tsize = dtypeSize(dtype);
    areTails = (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N));
    isU = isMatrixUpper(kflags);

    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);
    l1Pans = (unsigned int)subdims[0].x / (unsigned int)subdims[1].x;

    /*
     * Force generation of the transposed version of the block
     * reading function with following multiplication with transposed
     * block B to decrease LDS bank conflicts without column skew using.
     * Reverse temporarily the flag of the column-major order for that
     */
    if (useTransposedMul(subdims, dtype, trb)) {
        if (kflags & KEXTRA_COLUMN_MAJOR) {
            kflags &= ~KEXTRA_COLUMN_MAJOR;
        }
        else {
            kflags |= KEXTRA_COLUMN_MAJOR;
        }
        mulFlags = BLKMUL_SKEW_ROW | BLKMUL_TRANSPOSED_B;
        u = subdims[1].y;
    }
    else {
        mulFlags = BLKMUL_SKEW_COLUMN;
        u = subdims[0].y / (sizeof(cl_float4) / dtypeSize(dtype));
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations and auxiliary functions

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    kextraTmp.flags = kflags;
    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = &kextraTmp;
    gset.pgran = pgran;
    initKernelVarNames(&gset.varNames, kextra->flags);

    if (isComplexType(dtype)) {
        genComplexMathOperators(ctx, dtype);
    }

    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_TRSM, &gset, BCHF_MATRIX_B);
    /*
     * Temporary kernel extra has been needed to produce inverted block B read.
     * Restore the original one, and restore kflags as well
     */
    gset.kextra = kextra;
    kflags = kextra->flags;

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
    upFlags = UPRES_USE_LDS;
    if (!(mulFlags & BLKMUL_TRANSPOSED_B)) {
        upFlags |= UPRES_COLUMN_MAJOR;
    }
    updateResultGenOld(ctx, &gset, UPRES_SET, upFlags, NULL);
    kgenGetLastFuncName(copyPLFn, FUNC_NAME_MAXLEN, ctx);
    kgenAddBlankLine(ctx);

    generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                         ZF_MATRIX_B | ZF_MATRIX_C);

    // block multiplication function
    mulOpts.aMobj = CLMEM_IMAGE;
    mulOpts.bMobj = CLMEM_BUFFER;
    mulOpts.flags = BLKMUL_OUTPUT_PRIVATE | mulFlags;
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

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    // block number calculation
    getNblock(ctx, isU);

    getResultGPRsInfo(dtype, &subdims[1], vecLen, &nrRegs, &outTypeName);
    if (isComplexType(dtype)) {
        colRegs = (unsigned int)subdims[1].x;
    }
    else {
        colRegs = (unsigned int)fl4RowWidth(subdims[1].x, tsize);
    }

    if (mulFlags & BLKMUL_SKEW_ROW) {
        genReorderSolution(ctx, subdims, outTypeName, colRegs);
    }

    // now, generate the kernel

    if (kflags & KEXTRA_SIDE_RIGHT) {
        sprintf(tmp, trsmImDecl, pgran->wgSize[0], pgran->wgSize[1],
            fpref, 'N', 'M', typeName, typeName, typeName, typeName);
    }
    else {
        sprintf(tmp, trsmImDecl, pgran->wgSize[0], pgran->wgSize[1],
            fpref, 'M', 'N', typeName, typeName, typeName, typeName);
    }

    kgenDeclareFunction(ctx, tmp);
    ret = kgenBeginFuncBody(ctx);

    if (!(mulFlags & BLKMUL_TRANSPOSED_B)) {
        sprintf(tmp, "const int bpr = get_image_width(A) / %lu;\n",
                subdims[0].y / (sizeof(cl_float4) / tsize));
    }
    else {
        sprintf(tmp, "const int bpc = get_image_height(A) / %lu;\n",
                subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    /*
     * Calculate local buffer pitches, and then insert the
     * preparative code
     */
    pitchAB = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);

    sprintf(tmp, trsmImPrep1D, typeName, pitchAB * subdims[0].x,
        outTypeName, nrRegs, u, s1, pgran->wgSize[0], subdims[0].itemX);
    kgenAddStmt(ctx, tmp);
    kgenAddBlankLine(ctx);

    kgenAddStmt(ctx, "B += offB;\n");
    sprintf(tmp, "coordB.%c = currN + lid %% %u * %lu;\n"
                 "coordB.%c = 0;\n\n",
            vect[trb], l1Pans, subdims[1].x, vect[1 - trb]);
    kgenAddStmt(ctx, tmp);

   /*
    * B matrix is divided on panels, each work group
    * multiply such a panel on the whole matrix A.
    */

    // top level loop over M
    if (isU) {
        sprintf(tmp1, "(((finishRow - 1) / %lu) * %lu)", subdims[0].y,
                subdims[0].y); //last block start
        sprintf(tmp, "for (m0 = %s; m0 + %lu != startRow; m0 -= %lu)",
                tmp1, subdims[0].y, subdims[0].y);
        ret = kgenBeginBranch(ctx, tmp);
    }
    else {
        sprintf(tmp, "for (m0 = startRow; m0 < finishRow; m0 += %lu)",
                subdims[0].y);
        ret = kgenBeginBranch(ctx, tmp);
    }

    sprintf(tmp, "coordA.%c = m0 + lid / %u * %lu;\n"
                 "coordA.%c = 0;\n\n",
            vect[tra], l1Pans, subdims[1].y, vect[1 - tra]);
    kgenAddStmt(ctx, tmp);

    genZeroResult(ctx, dtype, subdims);

    // loop over K
    if (isU) {
        sprintf(tmp, "for (k0 = m0 + %lu; k0 < M; k0 += %lu)",
            subdims[0].bwidth, subdims[0].bwidth);
    }
    else {
        sprintf(tmp, "for (k0 = 0; k0 < m0; k0 += %lu)",
            subdims[0].bwidth);
    }
    ret = kgenBeginBranch(ctx, tmp);

    genPrepareRectBlock(ctx, subdims, dtype, &copyFuncs, &zeroFuncs,
                        trb, 'C', !areTails);

    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    // multiplication in the adjusting loop
    genMultiplication(ctx, subdims, dtype, blkmul, mulFlags);

    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenEndBranch(ctx, NULL); // loop over K
    kgenAddBlankLine(ctx);

    if (mulFlags & BLKMUL_SKEW_ROW) {
        kgenAddStmt(ctx, "reorderResult(c, skew);\n");
    }
    kgenAddStmt(ctx, "k0 = m0;\n");

    genUpdateIntermTrsmResult(ctx, &gset, updateResFnRev,
                                  updateResGenericFnRev, true);

    genHeapTrsmResultToLDS(ctx, &gset, copyPLFn, "tempC");
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    genZeroResult(ctx, dtype, subdims);

    // multiplication on the inverted block
    genMultiplication(ctx, subdims, dtype, blkmul, mulFlags);
    if (mulFlags & BLKMUL_SKEW_ROW) {
        kgenAddStmt(ctx, "reorderResult(c, skew);\n");
    }

    // write back the tile evaluated
    upFlags = UPRES_EXCEED_PROBLEM_CONDITION;
    if (isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_C)) {
        upFlags |= UPRES_COLUMN_MAJOR;
    }
    genResultUpdateWithFlagsOld(ctx, CLBLAS_TRSM, &gset, upFlags, updateResFn,
                                updateResGenericFn, NULL);

    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    // end external loops over panels of matrix A
    kgenEndBranch(ctx, NULL);
    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

static ssize_t
wrapper(
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
        return prepGenerator(buf, buflen, subdims, pgran, extra);
    }
}

static void
assignKargs(KernelArg *args, const void *params, const void *extra)
{
    const CLBlasKargs *blasArgs = (const CLBlasKargs*)params;

    (void)extra;

    if (blasArgs->kernType == CLBLAS_COMPUTING_KERNEL) {
        if (blasArgs->side == clblasLeft) {
           initSizeKarg(&args[0], blasArgs->K);
           initSizeKarg(&args[1], blasArgs->N);
        }
        else {
           initSizeKarg(&args[0], blasArgs->M);
           initSizeKarg(&args[1], blasArgs->K);
        }
        assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
        initMemobjKarg(&args[3], blasArgs->scimage[0], NULL, 0, 0);
        initMemobjKarg(&args[4], blasArgs->B, NULL, 0, 0);
        initSizeKarg(&args[5], blasArgs->ldb.matrix);
        if (blasArgs->side == clblasLeft) {
            initSizeKarg(&args[6], blasArgs->offsetM);
            initSizeKarg(&args[7], blasArgs->M + blasArgs->offsetM);
        }
        else {
            initSizeKarg(&args[6], blasArgs->offsetN);
            initSizeKarg(&args[7], blasArgs->N + blasArgs->offsetN);
        }
        initSizeKarg(&args[8], blasArgs->offBX);
    }
    else {
        if (blasArgs->side == clblasLeft) {
            initSizeKarg(&args[0], blasArgs->M);
        }
        else {
            initSizeKarg(&args[0], blasArgs->N);
        }
        initMemobjKarg(&args[1], blasArgs->A, NULL, 0, 0);
        initSizeKarg(&args[2], blasArgs->lda.matrix);
        initMemobjKarg(&args[3], blasArgs->scimage[0], NULL, 0, 0);
        if (blasArgs->side == clblasLeft) {
            initSizeKarg(&args[4], blasArgs->offsetM);
        }
        else {
            initSizeKarg(&args[4], blasArgs->offsetN);
        }
        initSizeKarg(&args[5], blasArgs->offA);
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
     * For prepare kernel two square local blocks required.
     * For main kernel two rectangular blocks required.
     * Maximum of these two values checked.
     */

    sizeA = matrBlockSize(dim, MATRIX_A, dtype, kargs->side);
    sizeB = matrBlockSize(dim, MATRIX_B, dtype, kargs->side);
    size = (sizeA > sizeB) ? sizeA : sizeB;

    return (2 * size * dtypeSize(dtype) <= ldsSize);
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *dims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    SubproblemDim globDim, offDim;
    const CLBlasKargs *kargs = (const CLBlasKargs*)args;
    size_t width, startBlock, finishBlock;
    bool isU = (kargs->uplo == clblasUpper) ^
        (kargs->transA != clblasNoTrans) ^ (kargs->side == clblasRight);

    (void)extra;

    width = kargs->K;
    width = (width + dims[0].bwidth - 1) / dims[0].bwidth;
    kargsToProbDims(&globDim, CLBLAS_TRSM, kargs, false);
    kargsToProbDims(&offDim, CLBLAS_TRSM, kargs, true);

    startBlock = offDim.y / dims[0].bwidth;
    finishBlock = (globDim.y + offDim.y + dims[0].bwidth - 1) / dims[0].bwidth;

    if (kargs->kernType == CLBLAS_PREP_A_KERNEL) {
        if (isU) {
            threads[0] = ((2 * width - startBlock - finishBlock + 1) *
                (finishBlock - startBlock) / 2) * pgran->wgSize[0];
        }
        else {
            threads[0] = ((1 + finishBlock + startBlock) *
                (finishBlock - startBlock) / 2) * pgran->wgSize[0];
        }
        threads[1] = 0;
    }
    else {
        calcGlobalThreads(threads, dims, pgran, globDim.y, globDim.x);
    }
}

static void
imgPackMode(
    const void *extra,
    const SubproblemDim *dims,
    int dataID,
    unsigned int *packRate,
    clblasOrder *packOrder)
{
    bool trb;
    const CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;

    (void)dataID;

    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kextra->flags, MATRIX_B);
    if (trb || isComplexType(kextra->dtype)) {
        *packOrder = clblasRowMajor;
        *packRate = (unsigned int)dims[0].y;
    }
    else {
        *packOrder = clblasColumnMajor;
        *packRate = (unsigned int)dims[0].y;
    }
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D | SF_TOP_INPUT_SQUARE_BLOCKS);
}

void
initTrsmImgPattern(MemoryPattern *mempat)
{
    mempat->name = "Image based block trsm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &solverOps;
    mpatExtra.aMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS;
    mpatExtra.bMset = CLMEM_LEVEL_LDS;
    mpatExtra.mobjA = CLMEM_IMAGE;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}
