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
 * TRSM generator with support of cached reads from the global memory
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <matrix_props.h>
#include <matrix_dims.h>

#include "../blas_kgen.h"
#include "../trxm_common.h"
#include "trsm_kgen_legacy.h"
#include "gen_helper_legacy.h"
#include "../trsm_kgen.h"

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

static CLBLASMpatExtra mpatExtra;

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs);

static SolverFlags
solverFlags(void);

static void
assignKargs(KernelArg *args, const void *params, const void *extra);

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static SolverOps trsmSops = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL,
    NULL,
    NULL,
    NULL,
    solverFlags,
    fixupArgs,
    NULL, //getDefaultDecomp
   	NULL, // getDecompList
   	NULL,
   	NULL
};

static TileMulFlags
getCyclicFlags(
    const SubproblemDim *dim,
    KernelExtraFlags kflags,
    bool tailPass,
    unsigned int vecLen)
{
    TileMulFlags mflags = TILEMUL_NO_FLAGS;

    if (tailPass && !isMatrixUpper(kflags)) {
        mflags |= TILEMUL_GLOBAL_CYCLIC_A;
    }

    if (isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B) &&
        (kflags & KEXTRA_TAILS_N) && (dim->x > vecLen)) {

        mflags |= TILEMUL_GLOBAL_CYCLIC_B;
    }

    return mflags;
}

static void
initTiles(BlasGenSettings *gset)
{
    unsigned int nrRows, nrCols;
    unsigned int vecLen;
    const SubproblemDim *dim = &gset->subdims[1];
    const CLBLASKernExtra *kextra = gset->kextra;
    DataType dtype = kextra->dtype;
    bool tra;

    // the tile A should be able to fit rectangular and square tiles
    nrCols = (unsigned int)szmax(dim->y, dim->bwidth);
    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kextra->flags, MATRIX_A);
    vecLen = getVecLen(gset, CLBLAS_TRSM, MATRIX_A);
    initTile(&gset->tileA, "a", (unsigned int)dim->y, nrCols, vecLen,
             dtype, PRIV_STORAGE_ARRAY, tra, false);

    /*
     * tile B should be able to fit tiles of the matrix B and of the
     * intermediate result. That result will be always transposed
     * from the point of view of tile multiplication
     */
    tra = !isMatrixAccessColMaj(CLBLAS_TRSM, kextra->flags, MATRIX_B);
    if (tra) {
        nrRows = (unsigned int)szmax(dim->bwidth, dim->y);
        nrCols = (unsigned int)dim->x;
    }
    else {
        nrRows = (unsigned int)szmax(dim->bwidth, dim->x);
        nrCols = (unsigned int)szmax(dim->x, dim->y);
    }
    vecLen = getVecLen(gset, CLBLAS_TRSM, MATRIX_B);
    initTile(&gset->tileBX, "b", nrRows, nrCols, vecLen, dtype,
             PRIV_STORAGE_ARRAY, tra, false);

    initTile(&gset->tileCY, "c", (unsigned int)dim->y, (unsigned int)dim->x,
             vecLen, dtype, PRIV_STORAGE_ARRAY, false, false);
}

static void
prepareTilesForMainLoop(BlasGenSettings *gset)
{
    const SubproblemDim *dim = &gset->subdims[1];

    gset->tileA.nrCols = (unsigned int)dim->bwidth;
    gset->tileBX.nrRows = (unsigned int)dim->bwidth;
    gset->tileBX.nrCols = (unsigned int)dim->x;
}

static void
declareLocalVariables(
    struct KgenContext *ctx,
    const BlasGenSettings *gset)
{
    char tmp[1024];
    const char *elemType;
    const SubproblemDim *dims = gset->subdims;
    DataType dtype = gset->kextra->dtype;
    size_t pitchAC, heightC;

    elemType = dtypeBuiltinType(dtype);
    pitchAC = matrBlockPitch(dims, MATRIX_C, dtype, clblasRight);
    heightC = szmax(dims[0].y, dims[0].x);

    declareTileStorages(ctx, gset);
    sprintf(tmp, "const int lid = get_local_id(0);\n"
                 "const int gid = get_group_id(0);\n"
                 "const uint2 skewRow = 0, skewCol = 0;\n\n"
                 "GPtr uA, uB;\n"
                 "uint coordA, coordB, k;\n"
                 "uint x, y;\n"
                 "__local %s tempA[%lu], tempC[%lu];\n"
                 "LPtr utmpA, utmpC;\n"
                 "uint m0 = 0, k0, currM, currN;\n",
            elemType, pitchAC * dims[0].y, pitchAC * heightC);
    kgenAddStmt(ctx, tmp);
}

static void
genReadDiagBlock(
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
genZeroResult(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims,
    unsigned int vecLen)
{
    unsigned int n;
    char tmp[1024];

    getResultGPRsInfo(dtype, &dims[1], vecLen, &n, NULL);

    sprintf(tmp, "for (x = 0; x < %u; x++) {\n"
                 "    c[x] = 0;\n"
                 "}\n\n", n);

    kgenAddStmt(ctx, tmp);
}

static void
genInternalLoopCtl(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags)
{
    char tmp[1024];

    if (isMatrixUpper(kflags)) {
        if (kflags & KEXTRA_TAILS_M) {
            sprintf(tmp, "for (k0 = currM + %lu; k0 < M / %lu * %lu; "
                               "k0 += %lu)",
                    dim[0].bwidth, dim[1].bwidth, dim[1].bwidth, dim[1].bwidth);
        }
        else {
            sprintf(tmp, "for (k0 = currM + %lu; k0 < M; k0 += %lu)",
                    dim[0].bwidth, dim[1].bwidth);
        }
    }
    else {
        sprintf(tmp, "for (k0 = 0; k0 < currM; k0 += %lu)",
                dim[1].bwidth);
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
initKernelVarNames(KernelVarNames *kvars)
{
    kvars->A = "uA";
    kvars->B = "uB";
    kvars->coordA = "coordA";
    kvars->coordB = "coordB";
    kvars->k = "k";
    kvars->sizeM = "M";
    kvars->sizeN = "N";
    kvars->sizeK = "M";
    kvars->lda = "lda";
    kvars->ldb = "ldb";
}

/*
 * Generate a code copying tile between LDS and private location.
 */
static void
genLdsCopy(
    struct KgenContext *ctx,
    const BlasGenSettings *gset)
{
    char pitchStr[16];
    char coordY[128], coordX[128];
    size_t pitch;
    UpresVarNames uvars;
    UpdateResultFlags upFlags = UPRES_INLINE | UPRES_USE_LDS |
                                UPRES_WITHOUT_ALPHA | UPRES_COLUMN_MAJOR;
    const SubproblemDim *dims = gset->subdims;
    unsigned int l1Pans = (unsigned int)(dims[0].x / dims[1].x);

    memset(&uvars, 0, sizeof(uvars));

    pitch = matrBlockPitch(dims, MATRIX_C, gset->kextra->dtype, clblasRight);
    sprintf(pitchStr, "%lu", pitch);
    sprintf(coordY, "lid / %u * %lu", l1Pans, dims[1].y);
    sprintf(coordX, "lid %% %u * %lu", l1Pans, dims[1].x);
    uvars.result = "tempC";
    uvars.ld = pitchStr;
    uvars.startRow = coordY;
    uvars.startCol = coordX;
    uvars.nrRows = NULL;
    uvars.nrCols = NULL;

    kgenBeginBranch(ctx, NULL);

    updateResultGen(ctx,
        gset,
        CLBLAS_TRSM,
        UPRES_SET,
        upFlags,
        &uvars);

    kgenEndBranch(ctx, NULL);

    kgenAddBlankLine(ctx);
}

static void
genZeroResultTrash(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const CLBLASKernExtra *kextra)
{
    char tmp[1024];
    unsigned int vecLen, pitch;
    unsigned int i;

    vecLen = (isComplexType(kextra->dtype)) ? 1 : kextra->vecLen;
    pitch = (unsigned int)roundUp(dim->x, vecLen);
    sprintf(tmp, "if (coordA + %lu > M)", dim->y);
    kgenBeginBranch(ctx, tmp);
    sprintf(tmp, "int i = (coordA >= M) ? %lu : (%lu - M %% %lu);\n\n",
            dim->y, dim->y, dim->y);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "for (; i > 0; i--)");
    kgenBeginBranch(ctx, tmp);

    for (i = 0; i < pitch / vecLen; i++) {
        sprintf(tmp, "c[(%lu - i) * %u + %u] = 0;\n",
                dim->y, pitch / vecLen, i);
        kgenAddStmt(ctx, tmp);
    }

    kgenEndBranch(ctx, NULL);
    kgenEndBranch(ctx, NULL);
}

static void
setupVdepUpresFlags(KernelExtraFlags kflags, UpdateResultFlags* upFlags)
{
    bool forceBug = false;

    unsigned int bugFlag1 = KEXTRA_NO_COPY_VEC_A
                          | KEXTRA_TAILS_K
                          | KEXTRA_TAILS_M;
    unsigned int bugFlag2 = bugFlag1
                          | KEXTRA_UPPER_TRIANG
                          | KEXTRA_TRANS_A;
    unsigned int bugFlag3 = bugFlag1
                          | KEXTRA_SIDE_RIGHT
                          | KEXTRA_COLUMN_MAJOR;
    unsigned int bugFlag4 = bugFlag3
                          | KEXTRA_TRANS_A;
    unsigned int bugFlag5 = bugFlag3
                          | KEXTRA_UPPER_TRIANG;
    unsigned int bugFlag6 = KEXTRA_NO_COPY_VEC_A
                          | KEXTRA_NO_COPY_VEC_B
                          | KEXTRA_NO_COPY_VEC_C
                          | KEXTRA_TAILS_K
                          | KEXTRA_TAILS_M;
    unsigned int bugFlag7 = bugFlag6
                          | KEXTRA_COLUMN_MAJOR;
    unsigned int bugFlag8 = bugFlag6
                          | KEXTRA_SIDE_RIGHT
                          | KEXTRA_UPPER_TRIANG;
    unsigned int bugFlag9 = bugFlag6
                          | KEXTRA_UPPER_TRIANG
                          | KEXTRA_TRANS_A
                          | KEXTRA_TAILS_N;
    unsigned int bugFlag10 = bugFlag7
                           | KEXTRA_SIDE_RIGHT
                           | KEXTRA_TRANS_A
                           | KEXTRA_TAILS_N;
    unsigned int bugFlag11 = bugFlag9
                           | KEXTRA_UNIT_DIAGONAL;
    unsigned int bugFlag12 = bugFlag6
                           | KEXTRA_TAILS_N
                           | KEXTRA_SIDE_RIGHT
                           | KEXTRA_UNIT_DIAGONAL
                           | KEXTRA_COLUMN_MAJOR
                           | KEXTRA_TRANS_A;

    /*
     * WORKAROUND for AMD GPU: Now, we avoid optimizing the case when
     *                         matrix B is not divided on block size and
     *                         since it leads to a hang up at code seeming
     *                         correct.
     */
    if (kflags & KEXTRA_VENDOR_AMD) {
        forceBug = (kflags & KEXTRA_TAILS_N) != 0;
    }
    else {
        forceBug = (kflags != bugFlag1
            && kflags != bugFlag2 && kflags != bugFlag4 &&  kflags != bugFlag5
            && kflags != bugFlag7 && kflags != bugFlag8 &&  kflags != bugFlag9
            && kflags != bugFlag10 && kflags != bugFlag11
            && kflags != bugFlag12);
    }

    if (!forceBug) {
        *upFlags |= UPRES_INDEXING_WITH_CONSTANTS;
    }
}

static void
genSetupCoordinates(
    struct KgenContext *ctx,
    const SubproblemDim *dims,
    KernelExtraFlags kflags)
{
    char tmp[1024];
    unsigned int l1Pans = (unsigned int)(dims[0].x / dims[1].x);

    sprintf(tmp, "coordA = currM + lid / %u * %lu;\n", l1Pans, dims[1].y);
    kgenAddStmt(ctx, tmp);
    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "k = currM + %lu;\n", dims[0].y);
    }
    else {
        strcpy(tmp, "k = 0;\n");
    }
    kgenAddStmt(ctx, tmp);
}

static void
genInvertDiagBlock(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const ZeroFuncs *zeroFuncs)
{
    char tmp[1024];
    const CLBLASKernExtra *kextra = gset->kextra;
    const SubproblemDim *subdims = gset->subdims;
    size_t pitchA;

    pitchA = matrBlockPitch(subdims, MATRIX_A, kextra->dtype, clblasLeft);

    sprintf(tmp, "%s((__local float4*)tempA);\n", zeroFuncs->names[MATRIX_A]);
    kgenAddStmt(ctx, tmp);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    if (kextra->flags & KEXTRA_UNIT_DIAGONAL) {
        sprintf(tmp, "if (lid < %lu) {\n"
                     "    tempC[lid * %lu + lid] = %s;\n"
                     "}\n",
                subdims[0].bwidth, pitchA, strOne(kextra->dtype));
        kgenAddStmt(ctx, tmp);
        kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
        kgenAddBlankLine(ctx);
    }

    sprintf(tmp, "if (lid < %lu)", subdims[0].y);
    kgenBeginBranch(ctx, tmp);
    sprintf(tmp, "invert(tempC, tempA, lid, (currM + %lu > M) ? "
                         "M - currM : %lu);\n",
            subdims[0].y, subdims[0].y);
    kgenAddStmt(ctx, tmp);
    kgenEndBranch(ctx, NULL);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);
}

static void
genMulOnDiagBlock(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    const TileMulOpts *mulOpts)
{
    char tmp[1024];
    const SubproblemDim *dims = gset->subdims;
    const CLBLASKernExtra *kextra = gset->kextra;
    unsigned int l1Pans = (unsigned int)(dims[0].x / dims[1].x);
    TileMulOpts optsNew;
    size_t pitchAC;
    const char *ptrName;
    Tile *tile;
    BlasGenSettings gsetNew;

    pitchAC = matrBlockPitch(dims, MATRIX_C, kextra->dtype, clblasRight);
    ptrName = dtypeUPtrField(kextra->dtype);

    memcpy(&optsNew, mulOpts, sizeof(optsNew));
    optsNew.memA = CLMEM_LOCAL_MEMORY;
    optsNew.memB = CLMEM_LOCAL_MEMORY;
    optsNew.flags &= ~(TILEMUL_TRA | TILEMUL_GLOBAL_CYCLIC | TILEMUL_CONJA);
    optsNew.flags |= TILEMUL_TRB;
    optsNew.memA = CLMEM_LOCAL_MEMORY;
    optsNew.memB = CLMEM_LOCAL_MEMORY;
    gset->varNames.A = "utmpA";
    gset->varNames.B = "utmpC";

    sprintf(tmp, "utmpA.%s = tempA + lid / %u * %lu;\n"
                 "utmpC.%s = tempC + lid %% %u * %lu;\n\n",
            ptrName, l1Pans, pitchAC * dims[1].y,
            ptrName, l1Pans, pitchAC * dims[1].x);
    kgenAddStmt(ctx, tmp);

    memcpy(&gsetNew, gset, sizeof(gsetNew));
    gsetNew.subdims[1].bwidth = dims[1].y;

    // Configure the tile descriptors to deal with tile of needed sizes.
    tile = &gsetNew.tileA;
    tile->nrRows = (unsigned int)dims[1].y;
    tile->nrCols = (unsigned int)dims[1].y;
    tile->trans = false;
    tile = &gsetNew.tileBX;
    tile->nrRows = (unsigned int)dims[1].y;
    tile->nrCols = (unsigned int)dims[1].x;
    tile->trans = true;
    tileMulGen(ctx, &gsetNew, &optsNew);

    gset->varNames.A = "uA";
    gset->varNames.B = "uB";
}

static void
genOneTrsmPass(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    const char *updateResFnRev,
    const char *updateResGenericFnRev,
    CopyBufFuncs *copyFuncs,
    ZeroFuncs *zeroFuncs,
    bool isTailPass)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    CLBLASKernExtra kextraTmp;
    KernelExtraFlags kflags = kextra->flags;
    char tmp[1024];
    DataType dtype = kextra->dtype;
    unsigned int vecLen = gset->kextra->vecLen;
    SubproblemDim *subdims = gset->subdims;
    int tra, trb;
    UpdateResultFlags upFlags;
    TilePostFetchPrivate pfpriv;
    TileMulOpts mulOpts;
    TailFetch tf;
    TailStatus tailStatus = 0;

    memset(&pfpriv, 0, sizeof(pfpriv));

    // multiply options
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;
    mulOpts.core = TILEMUL_MAD;//TILEMUL_MULADD;
    mulOpts.postFetch = NULL;
    mulOpts.flags = kextraToTilemulFlags(CLBLAS_TRSM, kflags);
    mulOpts.flags |= TILEMUL_EXTERN_RDECL;
    mulOpts.flags |= getCyclicFlags(subdims, kflags, isTailPass, vecLen);

    tra = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    trb = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);

    tf = checkForTailFetches(CLBLAS_TRSM, &subdims[1], kextra, MATRIX_B,
                             false, false);
    if (trb) {
        tf &= ~FETCH_TAIL_COL;
    }

    /*
     * For lower triangular matrix we proceed upto the diagonal, so we
     * can't exceed matrix bound and zeroing is not needed
     */
    if (isMatrixUpper(kflags)) {
        tf |= checkForTailFetches(CLBLAS_TRSM, &subdims[1], kextra,
                                  MATRIX_A, false, false);
        if (tra && trb) {
            tf &= ~FETCH_TAIL_COL;
        }
    }

    if (tf != FETCH_NO_TAILS) {
        memset(&pfpriv, 0, sizeof(pfpriv));
        pfpriv.funcID = CLBLAS_TRSM;
        pfpriv.gset = gset;
    }

    // loop over M
    if (!isTailPass) {
        sprintf(tmp, "for (m0 = 0; m0 < M / %lu * %lu; m0 += %lu)",
                subdims->y, subdims->y, subdims->y);
        kgenBeginBranch(ctx, tmp);
    }

    genSetupCoordinates(ctx, subdims, kflags);
    genZeroResult(ctx, dtype, subdims, vecLen);

    if (!isMatrixUpper(kflags) && isTailPass) {
        // skip update loop is the matrix consist of the single block
        sprintf(tmp, "if (M > %lu)", subdims->y);
        kgenBeginBranch(ctx, tmp);
    }

    // Avoid tail adjusting along M.

    memcpy(&kextraTmp, kextra, sizeof(kextraTmp));
    kextraTmp.flags &= ~(KEXTRA_TAILS_M | KEXTRA_TAILS_M_LOWER);

    // update loop is not needed for tail of an upper triangular matrix
    if (!(isTailPass && isMatrixUpper(kflags))) {
        if (isTailPass || (kflags & KEXTRA_TAILS_N)) {
            kgenBeginBranch(ctx, "if (coordB < N)");
        }

        gset->kextra = &kextraTmp;
        tailStatus = checkGenAdjustTailCoords(ctx, CLBLAS_TRSM, gset, NULL);
        gset->kextra = kextra;

        genInternalLoopCtl(ctx, subdims, kflags);           // loop over K

        // multiplication for the step-by-step block updating
        subdims[0].bwidth = subdims[1].bwidth;
        tileMulGen(ctx, gset, &mulOpts);
        subdims[0].bwidth = subdims[0].y;

        genInternalLoopEnd(ctx);                             // loop over K
        kgenAddBlankLine(ctx);

        // invoke once again, in order to process tails along K
        if (isMatrixUpper(kflags) && (tf != FETCH_NO_TAILS)) {
            subdims[0].bwidth = subdims[1].bwidth;

            if (!(tra && trb)) {
                mulOpts.flags |= TILEMUL_WRAP_AROUND_TAIL;
            }
            mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_K;

            mulOpts.postFetchPriv = &pfpriv;
            mulOpts.postFetch = defaultTilePostFetch;

            subdims[0].bwidth = subdims[1].bwidth;
            tileMulGen(ctx, gset, &mulOpts);
            subdims[0].bwidth = subdims[0].y;

            mulOpts.postFetch = NULL;
            mulOpts.postFetchPriv = NULL;
        }

        gset->kextra = &kextraTmp;
        checkGenRestoreTailCoords(ctx, gset, tailStatus);
        gset->kextra = kextra;

        if (isTailPass || (kflags & KEXTRA_TAILS_N)) {
            kgenEndBranch(ctx, NULL);
        }
    }
    else if (!trb && (kflags & KEXTRA_TAILS_N)) {
        tailStatus |= TAIL_B_RAISED;
    }

    mulOpts.flags &= ~(TILEMUL_WRAP_AROUND_TAIL | TILEMUL_GLOBAL_CYCLIC_A |
                       TILEMUL_GLOBAL_CYCLIC_K);

    if (!isMatrixUpper(kflags) && isTailPass) {
        /*
         * end of branch for non single block tail processing of
         * the lower triangular matrix
         */
        kgenEndBranch(ctx, NULL);
    }

    /*
     * Final phase: update the accumulated result, multiply on an inverted
     *              block and write back the result
     */
    if (isMatrixUpper(kflags) || ((kflags & KEXTRA_VENDOR_AMD) != 0)) {
        kgenAddStmt(ctx, "k0 = currM;\n");
    }
    else {
        kgenAddStmt(ctx, "k0 = m0;\n");
    }

    genReadDiagBlock(ctx, subdims, dtype, copyFuncs, zeroFuncs,
                     kflags, 'C');
    genInvertDiagBlock(ctx, gset, zeroFuncs);

    // Avoid generating not executed non optimal path
    gset->kextra = &kextraTmp;
    if (isTailPass) {
        kextraTmp.flags |= (KEXTRA_TAILS_M | KEXTRA_TAILS_M_LOWER);
    }
    genUpdateIntermTrsmResult(ctx, gset, updateResFnRev,
                              updateResGenericFnRev, true);
    gset->kextra = kextra;

    /*
     * Heap to LDS.
     * Zero unuseful part along columns since it will have an influence
     * on the result at multiplication on an inverted block
     */
    if (isTailPass) {
        genZeroResultTrash(ctx, &subdims[1], kextra);
    }
    genLdsCopy(ctx, gset);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    genZeroResult(ctx, dtype, subdims, vecLen);

    genMulOnDiagBlock(ctx, gset, &mulOpts);

    // write back the tile evaluated
    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= tailStatusToUpresFlags(tailStatus);
    upFlags |= UPRES_EXCEED_PROBLEM_CONDITION;
    setupVdepUpresFlags(kflags, &upFlags);

    gset->kextra = &kextraTmp;

    genResultUpdateWithFlags(ctx, CLBLAS_TRSM, gset, upFlags,
                             NULL, NULL, NULL);
    gset->kextra = kextra;

    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "currM -= %lu;\n", subdims[0].y);
    }
    else {
        sprintf(tmp, "currM += %lu;\n", subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    if (!isTailPass) {
        kgenEndBranch(ctx, NULL);                       // loop over M
    }
}

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    char tmp[1024];
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    KernelExtraFlags kflags = kextra->flags;
    DataType dtype = kextra->dtype;
    BlasGenSettings gset;
    char updateResFnRev[FUNC_NAME_MAXLEN];
    char updateResGenericFnRev[FUNC_NAME_MAXLEN];
    CopyBufFuncs copyFuncs;
    ZeroFuncs zeroFuncs;
    UpdateResultFlags upFlags;
    const char *ptrName;
    bool b;
    ssize_t ret;
    unsigned int l1Pans = (unsigned int)(subdims[0].x / subdims[1].x);
    bool tailMarker[2] = {false, true};
    int triang;
    int i;

    if (pgran->wgDim != 1) {
        return -EINVAL;
    }

    if (kflags & KEXTRA_TAILS_M) {
        kflags |= KEXTRA_TAILS_M_LOWER;
    }
    if (kflags & KEXTRA_TAILS_N) {
        kflags |= KEXTRA_TAILS_N_LOWER;
    }
    if (kflags & KEXTRA_TAILS_K) {
        kflags |= KEXTRA_TAILS_K_LOWER;
    }
    kextra->flags = kflags;

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    triang = isMatrixUpper(kflags);

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.kextra = kextra;
    gset.pgran = pgran;

    initKernelVarNames(&gset.varNames);

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);
    if (isComplexType(dtype)) {
        genComplexMathOperators(ctx, dtype);
    }

    /*
     * For intermediate result after blocks modification.
     * Take into account tails adjusting
     */
    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= UPRES_WITH_BETA | UPRES_PRIV_DEST;

    if (!isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B) &&
        (kflags & KEXTRA_TAILS_N)) {

        upFlags |= UPRES_TAIL_COL;
    }

    setupVdepUpresFlags(kflags, &upFlags);
    initTiles(&gset);
    genUpresFuncsWithFlags(ctx, &gset, upFlags, updateResFnRev,
                           updateResGenericFnRev);

    generateBufCopyFuncs(&copyFuncs, ctx, CLBLAS_TRSM, &gset, BCHF_MATRIX_A);
    generateZeroingFuncs(&zeroFuncs, ctx, &subdims[0], pgran, dtype,
                         ZF_MATRIX_A);

    //matrix inversion function
    genInvertingBlockFunc(ctx, subdims[0].bwidth, dtype, kflags);
    kgenAddBlankLine(ctx);

    // now, generate the kernel
    declareTrxmKernel(ctx, dtype, pgran, kflags, CLBLAS_TRSM, "Cached", false,
                      true);
    ret = kgenBeginFuncBody(ctx);

    declareLocalVariables(ctx, &gset);
    prepareTilesForMainLoop(&gset);

    sprintf(tmp, "currN = gid * %lu;\n", subdims[0].x);
    kgenAddStmt(ctx, tmp);
    genInitCurrM(ctx, subdims, kflags);

    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    genTrxmBMatrShift(ctx, kflags, false);

    ptrName = dtypeUPtrField(dtype);
    sprintf(tmp, "uA.%s = A;\n"
                 "uB.%s = B;\n\n",
            ptrName, ptrName);
    kgenAddStmt(ctx, tmp);

    /*
     * B matrix is divided on panels, each work group
     * multiply such a panel on the whole matrix A.
     */

    sprintf(tmp, "coordB = gid * %lu + lid %% %u * %lu;\n",
            subdims[0].x, l1Pans, subdims[1].x);
    kgenAddStmt(ctx, tmp);

    for (i = 0; i < 2; i++) {
        b = (i) ? tailMarker[1 - triang] : tailMarker[triang];
        if (!b || (kflags & KEXTRA_TAILS_M)) {
            genOneTrsmPass(ctx, &gset, updateResFnRev, updateResGenericFnRev,
                           &copyFuncs, &zeroFuncs, b);
        }
    }

    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    cl_ulong sizeA, sizeC;
    const CLBlasKargs *kargs = (const CLBlasKargs*)kernelArgs;

    /*
     * It's needed one block for matrix A,
     * and one block of size maximal of this one for
     * matrix A and matrix C
     */

    sizeA = matrBlockSize(dim, MATRIX_A, dtype, kargs->side);
    sizeC = matrBlockSize(dim, MATRIX_B, dtype, kargs->side);
    if (sizeA > sizeC) {
        sizeC = sizeA;
    }

    return ((sizeA + sizeC) * dtypeSize(dtype) <= ldsSize);
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D | SF_TOP_INPUT_SQUARE_BLOCKS);
}

static void
assignKargs(KernelArg *args, const void *params, const void *extra)
{
    const CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    int idx = 7;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
    initMemobjKarg(&args[3], blasArgs->A, NULL, 0, 0);
    initSizeKarg(&args[4], blasArgs->lda.matrix);
    initMemobjKarg(&args[5], blasArgs->B, NULL, 0, 0);
    initSizeKarg(&args[6], blasArgs->ldb.matrix);
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offBX);
    }
}

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    (void)extra;
    (void)subdims;

    fixupTrxmKargs((CLBlasKargs*)args);
}

void
initTrsmCachedPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block trsm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 0;
    mempat->sops = &trsmSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}
