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
#include <stdlib.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <matrix_props.h>
#include <matrix_dims.h>

#include "dblock_kgen.h"
#include "kerngen.h"
#include "blas_kgen.h"
#include "gen_helper.h"
#include "trxm_common.h"
#include "trsm_kgen.h"
#include "legacy/blas_kgen_legacy.h"

typedef enum LdsUseFlags {
    LDS_NO_USE = 0,
    LDS_USE_LARGE = 0x1,
    LDS_USE_DIAGONAL = 0x2
} LdsUseFlags;

typedef struct TrsmExtraParams {
    int unrollingFactor;
    unsigned int unrolledTail;
    LdsUseFlags ldsUse;
} TrsmExtraParams;

enum TrsmStage {
    BLOCK_UPDATE,
    TILE_UPDATE
};

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

static bool
checkCalcDecompDedicated(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

#if 0
static int
getDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void * pArgs);
#endif

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
    NULL,//getDefaultDecomp
    checkCalcDecompDedicated,
    NULL,
    NULL
};

// The struct for storage tails
typedef struct TileSet
{
    Tile rectA;     // The rectangular tile A for the update loop at stage 1
    Tile squareA;   // The square tile for the stage 2
    Tile origB;     // The rectangular tile B for the update loop at the stage 1
    Tile bStage2;   // The rectangular tile B for the update loop at thestage 2
    Tile bAsSqA;    // Descriptor for holding square tile A in the storage of B
    Tile bAsC;      // Descriptor for holding tile C in the storage of B
    // the entire tile A matching the storage declared in the kernel
    Tile A;
    // the entire tile B matching the storage declared in the kernel
    Tile B;
} TileSet;


static bool
useSkewedFetchB(const BlasGenSettings *gset)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    TrsmExtraParams *extraParams = (TrsmExtraParams*)gset->kextra->solverPriv;
    bool ret = false;

    if (extraParams->ldsUse & LDS_USE_LARGE) {
        ret = !isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);
    }

    return ret;
}

static void
restoreTile(Tile* dst, const Tile* src)
{
    dst->baseName = src->baseName;
    dst->vecLen = src->vecLen;
    dst->storType = src->storType;
}

static Tile
substituteTile(Tile* dst, const Tile* src)
{
    Tile tmp;

    restoreTile(&tmp, dst);
    restoreTile(dst, src);

    return tmp;
}

static void
sprintfInvertedElement(
    Kstring *elem,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    unsigned int len,
    bool isU)
{
    if (isU) {
        row = tile->nrRows - row - 1;
        col = tile->nrCols - col - len;
    }

    sprintfTileElement(elem, tile, row, col, len);
}

static void
genTileInverting(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileSet *tileSet)
{
    char tmp[1024];
    const CLBLASKernExtra *kextra = gset->kextra;
    KernelExtraFlags kflags = kextra->flags;
    DataType dtype = kextra->dtype;
    const SubproblemDim *dim = &gset->subdims[1];
    unsigned int accLen;
    unsigned int i, j, k;
    Tile srcTile;
    Tile dstTile;
    bool isU, isComplex;
    bool isInlined = gset->flags & BGF_EXPLICIT_INLINE;
    const char* typeNameA;
    const char* typeNameB;

    memcpy(&srcTile, &tileSet->bAsSqA, sizeof(srcTile));
    memcpy(&dstTile, &tileSet->squareA, sizeof(dstTile));

    getVectorTypeName(kextra->dtype, dstTile.vecLen, &typeNameA, NULL);
    getVectorTypeName(kextra->dtype, srcTile.vecLen, &typeNameB, NULL);
    isU = isMatrixUpper(kflags);
    isComplex = isComplexType(dtype);

    if (isComplex || dstTile.trans) {
        accLen = 1;
    }
    else {
        accLen = umin(srcTile.vecLen, dstTile.vecLen);
        accLen = umin(accLen, srcTile.nrCols);
    }

    if (!isInlined) {
        dstTile.baseName = "a";
        srcTile.baseName = "b";
        sprintf(tmp, "void\n"
                     "invertTile(%s *a, %s *b)\n",
                typeNameA, typeNameB);
        kgenDeclareFunction(ctx, tmp);
        kgenBeginFuncBody(ctx);
    }
    else {
        kgenAddStmt(ctx, "// Invert tile\n");
    }

    // made destination block unit
    genZeroTile(ctx, &dstTile);
    for (i = 0; i < dim->y; i++) {
        genSetUnitInTile(ctx, &dstTile, i, i);
    }
    kgenAddBlankLine(ctx);

    for (i = 0; i < dim->y; i++) {
        Kstring src, srcDiag, dst, dstLast;

        // current source diagonal element
        sprintfInvertedElement(&srcDiag, &srcTile, i, i, 1, isU);
        for (j = i; j < dim->y; j++) {
            // current source non diagonal element
            if (i) {
                sprintfInvertedElement(&src, &srcTile, j, i - 1, 1, isU);
            }

            for (k = 0; k < dim->y; k += accLen) {
                // current updated vectorized element
                sprintfInvertedElement(&dst, &dstTile, j, k, accLen, isU);

                // update
                if (i) {
                    // last updated vectorized element
                    sprintfInvertedElement(&dstLast, &dstTile, i - 1, k,
                                           accLen, isU);
                    if (isComplex) {
                        sprintf(tmp, "%s -= mul(%s, %s);\n",
                                dst.buf, dstLast.buf, src.buf);
                    }
                    else {
                        sprintf(tmp, "%s -= %s * %s;\n",
                                dst.buf, dstLast.buf, src.buf);
                    }
                    kgenAddStmt(ctx, tmp);
                }

                // divide on the diagonal element
                if (j == i) {
                    if (isComplex) {
                        sprintf(tmp, "%s = div(%s, %s);\n",
                                dst.buf, dst.buf, srcDiag.buf);
                    }
                    else {
                        sprintf(tmp, "%s /= %s;\n", dst.buf, srcDiag.buf);
                    }
                    kgenAddStmt(ctx, tmp);
                }
            }
        }
        if (i != dim->y - 1) {
            kgenAddBlankLine(ctx);
        }
    }

    if (!isInlined) {
        kgenEndFuncBody(ctx);
    }
    kgenAddBlankLine(ctx);

}

static void
declareLocalVariables(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    Tile* parTile,
    TrsmExtraParams * extraParams)
{
    char tmp[1024];
    const SubproblemDim *dims = gset->subdims;
    const char* parTileTypeName = NULL;
    bool trb = isMatrixAccessColMaj(CLBLAS_TRSM, gset->kextra->flags,
                                   MATRIX_B);
    unsigned int locWidth;
    unsigned int tsize;
    unsigned int parTileSize;
    unsigned int l1Pans;
    unsigned int step;

    kgenAddStmt(ctx,
                 "const int lid = get_local_id(0);\n"
                 "const int gid = get_group_id(0);\n"
                 "GPtr uA, uB;\n"
                 "uint coordA, coordB;\n"
                 "uint m0 = 0, k0, m1;\n");

    if (isMatrixUpper(gset->kextra->flags)) {
        sprintf(tmp, "uint currM = (M - 1) / %lu * %lu;\n",
                dims[0].y, dims[0].y);
        kgenAddStmt(ctx, tmp);
    }

    /*
     * Declare private blocks.
     * The region 'b' stores in different time tiles of both
     * the input matrices and the result
     */

    declareTileStorages(ctx, gset);

    *parTile = gset->tileBX;

    if (extraParams->ldsUse) {
        tsize = dtypeSize(gset->kextra->dtype);
        l1Pans = (unsigned int)(dims[0].x / dims[1].x);

        parTile->vecLen = (trb) ? (unsigned int)dims[1].x
                                : (unsigned int)dims[1].bwidth;
        parTile->vecLen = umin(parTile->vecLen, sizeof(cl_float4) / tsize);
        parTile->trans = trb;

       /*
        * Allocate enough space in the local area to fit several tiles
        * at the stage1 (according to the unrolled factor) and one tile
        * at the stage2
        */

        locWidth = (unsigned int)dims[1].bwidth * extraParams->unrollingFactor;
        if (extraParams->ldsUse & LDS_USE_DIAGONAL) {
            locWidth = umax(locWidth, (unsigned int)dims[1].y);
        }
        if (trb) {
            parTile->nrRows = locWidth;
            parTile->nrCols = (unsigned int)dims[0].x;
            step = (unsigned int)dims[1].x / parTile->vecLen;
        }
        else {
            parTile->nrRows = (unsigned int)dims[0].x;
            parTile->nrCols = locWidth;
            step = (unsigned int)dims[1].x * locWidth / parTile->vecLen;
        }

        parTileSize = tileVectorsNum(parTile);

        getVectorTypeName(gset->kextra->dtype, parTile->vecLen,
                          &parTileTypeName, NULL);

        sprintf(tmp, "__local %s tmpB[%i];\n"
                     "LPtr lB;\n"
                     "LPtr lBMain = {(__local float*)(tmpB + lid %% %u * %u)};\n",
                parTileTypeName, parTileSize, l1Pans, step);
        kgenAddStmt(ctx, tmp);

        if (useSkewedFetchB(gset)) {
            kgenPrintf(ctx, "const uint skewX = lid %% %u %% %lu;\n",
                       l1Pans, gset->subdims[1].x);
        }
    }

    kgenAddBlankLine(ctx);
}

/*
 * Generate cyclical tile shifting so as to convert the skewed
 * storing to "one-to-one", i. e. the first element in the tile
 * matches to the first element of the respective tile in the
 * output matrix.
 */
static void
genTileCyclicalShift(struct KgenContext *ctx, BlasGenSettings *gset)
{
    const char *tname;
    Kstring k1, k2, *src, *dst, *ktmp;
    unsigned int row, col;
    unsigned int seglen;
    Tile *tileC = &gset->tileCY;

    seglen = tileLineSegmentLen(tileC);
    getVectorTypeName(gset->kextra->dtype, seglen, &tname, NULL);

    kgenAddStmt(ctx, "\n// deliver from skewing in the result\n");
    kgenBeginBranch(ctx, "for (uint i = 0; i < skewX; i++)");
    kgenPrintf(ctx, "%s tmp;\n\n", tname);

    src = &k1;
    dst = &k2;

    // Skewing may be used only in case of transposed C
    for (row = 0; row < tileC->nrRows; row += seglen) {
        sprintfTileElement(dst, tileC, row, tileC->nrCols - 1, seglen);
        kgenPrintf(ctx, "tmp = %s;\n", dst->buf);
        for (col = tileC->nrCols - 1; col > 0; col--) {
            sprintfTileElement(src, tileC, row, col - 1, seglen);
            kgenPrintf(ctx, "%s = %s;\n", dst->buf, src->buf);
            // swap pointer
            ktmp = src;
            src = dst;
            dst = ktmp;
        }
        kgenPrintf(ctx, "%s = tmp;\n", dst->buf);
    }

    kgenEndBranch(ctx, NULL);
    kgenAddBlankLine(ctx);
}

/*
 * Setup coordinates before beginning a trsm stage
 * A caller must ensure the strict stage sequence:
 * BLOCK_UPDATE -> TILE_UPDATE
 */
static void
genSetupCoords(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    enum TrsmStage stage)
{
    char tmp[1024];
    KernelExtraFlags kflags = gset->kextra->flags;
    const SubproblemDim *dims = gset->subdims;
    unsigned int l1Pans = (unsigned int)(dims[0].x / dims[1].x);
    const char *s;

    s = isMatrixUpper(kflags) ? "currM" : "m0";
    sprintf(tmp, "coordA = %s + (lid / %u * %lu);\n",
            s, l1Pans, dims[1].y);
    kgenAddStmt(ctx, tmp);

    switch (stage) {
    case BLOCK_UPDATE:
        if (isMatrixUpper(kflags)) {
            sprintf(tmp, "k0 = currM + %lu;\n", dims[0].y);
        }
        else {
            sprintf(tmp, "k0 = 0;\n");
        }
        break;
    case TILE_UPDATE:
        if (isMatrixUpper(kflags)) {
            sprintf(tmp, "k0 = currM + %lu - m1 * %lu;\n",
                    dims[0].y - dims[1].y, dims[1].y);
        }
        else {
            sprintf(tmp, "k0 = m0 + m1 * %lu;\n", dims[1].y);
        }
        break;
    }

    kgenAddStmt(ctx, tmp);

    sprintf(tmp, "coordB = gid * %lu + (lid %% %u * %lu);\n",
            dims[0].x, l1Pans, dims[1].x);

    kgenAddStmt(ctx, tmp);
    kgenAddBlankLine(ctx);
}

// Generate control block of the loop over K
static void
genInternalLoopCtl(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags,
    size_t stepK,
    size_t boundAlign)
{
    char tmp[1024];

    if (isMatrixUpper(kflags)) {
        if (kflags & KEXTRA_TAILS_M) {
            sprintf(tmp, "for (k0 = currM + %lu; k0 < M / %lu * %lu; "
                               "k0 += %lu)",
                    dim[0].y, boundAlign, boundAlign, stepK);
        }
        else {
            sprintf(tmp, "for (k0 = currM + %lu; k0 < M; k0 += %lu)",
                    dim[0].y, stepK);
        }
    }
    else {
        sprintf(tmp, "for (k0 = 0; k0 < m0; k0 += %lu)",
                stepK);
    }

    kgenBeginBranch(ctx, tmp);
}

static void
initKernelVarNames(KernelVarNames *kvars)
{
    kvars->A = "uA";
    kvars->B = "uB";
    kvars->C = "B";
    kvars->coordA = "coordA";
    kvars->coordB = "coordB";
    kvars->k = "k0";
    kvars->sizeM = "M";
    kvars->sizeN = "N";
    kvars->sizeK = "M";
    kvars->lda = "lda";
    kvars->ldb = "ldb";
    kvars->ldc = "ldb";
    kvars->alpha = "alpha";
    kvars->beta = "beta";
}

static void
setFetchHandler(
    TileMulOpts *mulOpts,
    const BlasGenSettings *gset,
    int handler(struct KgenContext *ctx, MatrixRole mrole, void *priv),
    TilePostFetchPrivate *priv)
{
    int i, nrPrivs;
    const char *regName = NULL;

    if (handler == defaultTilePostFetch) {
        nrPrivs = 1;
    }
    else {
        nrPrivs = 2;
        regName = "b";
    }

    for (i = 0; i < nrPrivs; i++) {
        priv[i].fetchNumA = 0;
        priv[i].wholeA = 1;
        priv[i].funcID = CLBLAS_TRSM;
        priv[i].gset = gset;
        priv[i].regName = regName;
        mulOpts->postFetch = handler;
        mulOpts->postFetchPriv = priv;
    }
}

static void
genCheckShiftTailB(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    int adjustRestore,
    TailStatus *tailStatus)
{
    BlasGenSettings gsetNew;
    CLBLASKernExtra kextraNew;

    memcpy(&gsetNew, gset, sizeof(gsetNew));
    memcpy(&kextraNew, gset->kextra, sizeof(kextraNew));
    // avoid tail shift for the matrix A
    kextraNew.flags &= ~(KEXTRA_TAILS_M | KEXTRA_TAILS_M_LOWER);
    gsetNew.kextra = &kextraNew;

    if (adjustRestore) {
        checkGenRestoreTailCoords(ctx, &gsetNew, *tailStatus);
    }
    else {
        *tailStatus = checkGenAdjustTailCoords(ctx, CLBLAS_TRSM, &gsetNew,
                                               NULL);
    }
}

static void
sprintfHitMatrixCond(
    char *buf,
    MatrixRole mrole,
    const char *prefix,
    const char *suffix)
{
    const char *coordName;
    char bound;

    coordName = (mrole == MATRIX_A) ? "coordA" : "coordB";
    bound = (mrole == MATRIX_A) ? 'M' : 'N';
    if (suffix == NULL) {
        suffix = "";
    }
    sprintf(buf, "%s%s < %c%s", prefix, coordName, bound, suffix);
}

/*
 * 'mulUpd' arguments mean what action is being done: multiplication on
 * an inverted tile or subsequent update
 */
static void
sprintfStage2Condition(
    char *buf,
    const BlasGenSettings *gset,
    int mulUpd)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    char hitCond[1024];
    char *p;
    unsigned int xPans, yPans;


    hitCond[0] = '\0';
    xPans = (unsigned int)(gset->subdims[0].x / gset->subdims[1].x);
    yPans = (unsigned int)(gset->subdims[0].y / gset->subdims[1].y);
    if (kflags & KEXTRA_TAILS_M) {
        sprintfHitMatrixCond(hitCond, MATRIX_A, " && ", NULL);
    }
    p = hitCond + strlen(hitCond);
    if (kflags & KEXTRA_TAILS_N) {
        sprintfHitMatrixCond(p, MATRIX_B, " && ", NULL);
    }

    if (!mulUpd) {
        if (isMatrixUpper(kflags)) {
            sprintf(buf, "if (lid / %u + m1 == %u%s)",
                    xPans, yPans - 1, hitCond);
        }
        else {
            sprintf(buf, "if (lid / %u == m1%s)", xPans, hitCond);
        }
    }
    else {
        if (isMatrixUpper(kflags)) {
            sprintf(buf, "if (lid / %u + m1 < %u%s)",
                    xPans, yPans - 1, hitCond);
        }
        else {
            sprintf(buf, "if (lid / %u > m1%s)", xPans, hitCond);
        }
    }
}

static void
genZeroTileTrash(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    MatrixRole mrole,
    Tile* tile)
{
    char tmp[1024];
    const SubproblemDim *dim = &gset->subdims[1];
    const CLBLASKernExtra *kextra = gset->kextra;
    unsigned int i, j;
    unsigned int step;
    Kstring elem;

    if (mrole == MATRIX_A) {
        kgenAddBlankLine(ctx);
    }
    else {
        kgenBeginBranch(ctx, NULL);
    }

    sprintf(tmp, "const int bound = (coordA + %lu > M) ? (M - coordA) : %lu;\n",
            dim->y, dim->y);
    kgenAddStmt(ctx, tmp);

    step = tileLineSegmentLen(tile);
    step = (tile->trans) ? 1 : step;

    for (j = 0; j < tile->nrRows; ++j) {
        for (i = 0; i < tile->nrCols; i+=step) {
            sprintfTileElement(&elem, tile, j, i, step);
            sprintf(tmp, "%s = (bound <= %u) ? 0 : %s;\n", elem.buf, j, elem.buf);
            kgenAddStmt(ctx, tmp);
        }
    }

    // Set units in the trash diagonal elements for a tile of A
    if (mrole == MATRIX_A) {
        for (i = 0; i < (unsigned int)dim->y; i++) {
            sprintfTileElement(&elem, tile, i, i, 1);
            sprintf(tmp, "%s = (bound <= %d) ? %s : %s;\n",
                    elem.buf, (int)i, strOne(kextra->dtype), elem.buf);
            kgenAddStmt(ctx, tmp);
        }
    }

    if (mrole == MATRIX_A) {
        kgenAddBlankLine(ctx);
    }
    else {
        kgenEndBranch(ctx, NULL);
    }
}

/*
 * NOTE: Before invoking this function 'tileA' must be initialized accordingly
 *       so as it stores a square tile of the matrix A.
 */
static void
genMulOnDiagonalTile(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    TileSet *tileSet,
    const TileMulOpts *mulOpts)
{
    char tmp[1024];
    FetchOpts fetchOpts;
    const SubproblemDim *dim = &gset->subdims[1];
    TilePostFetchPrivate pfPriv[2];
    TileMulOpts optsNew;
    const CLBLASKernExtra *extra = gset->kextra;
    CLBLASKernExtra extraNew;
    KernelExtraFlags kflags = extra->flags;
    Tile t;
    bool isTail;

    memset(&fetchOpts, 0, sizeof(fetchOpts));
    fetchOpts.regName = "b";
    fetchOpts.mrole = MATRIX_A;
    fetchOpts.lineOffset = 0;
    fetchOpts.linesNum = (unsigned int)dim->y;

    // setup options to multiply on the inverted tile
    memcpy(&optsNew, mulOpts, sizeof(TileMulOpts));
    optsNew.flags &= ~TILEMUL_TRB;

    kgenAddStmt(ctx, "// Fetch and invert the square tile located on the "
                     "diagonal\n");

    // The matrix B play the role of A
    t = substituteTile(&gset->tileA, &tileSet->bAsSqA);

    isTail = ((kflags & KEXTRA_TAILS_M) != 0);
    genFetchInputTile(ctx, mulOpts->fctx, gset, &fetchOpts);
    setFetchHandler(&optsNew, gset, genTrxmPostFetchZero, pfPriv);

    /*
     * There is no needs in zeroing tail along K in case of the lower
     * triangular matrix because it is in the "other" triangle which is
     * never accessed
     */
    if (isTail && !isMatrixUpper(kflags)) {
        memcpy(&extraNew, extra, sizeof(extraNew));
        extraNew.flags &= ~KEXTRA_TAILS_K_LOWER;
        gset->kextra = &extraNew;
    }
    genTrxmPostFetchZero(ctx, MATRIX_A, pfPriv);

    /*
     * One must zero the tail part of a fetched square tile
     * in order to avoid influence of the trailing trash on the resulting
     * inverted tile (evaluating proceeds from the bottom towards the top
     *                of the tile)
     */
    if (isTail) {
        genZeroTileTrash(ctx, gset, MATRIX_A, &gset->tileA);
    }

    restoreTile(&gset->tileA, &t);

    if(gset->flags & BGF_EXPLICIT_INLINE) {
        genTileInverting(ctx, gset, tileSet);
    }
    else {
        sprintf(tmp, "invertTile(%s, %s);\n\n",
                tileSet->squareA.baseName, tileSet->bAsSqA.baseName);
        kgenAddStmt(ctx, tmp);
    }

    gset->tileBX = tileSet->bAsC;
    genTileCopy(ctx, &gset->tileBX, &gset->tileCY, TILECOPY_ASSIGN);

    /*
     * For the lower diagonal not integrally decomposed matrix A
     * it's enough to zero the tail part of the result in order to
     * clear trash accumulated over the update loop
     */
    if (isTail && !isMatrixUpper(kflags)) {
        genZeroTileTrash(ctx, gset, MATRIX_B, &gset->tileBX);
    }

    genZeroTile(ctx, &gset->tileCY);

    genMulTiles(ctx, gset, &optsNew);
    kgenAddBlankLine(ctx);

    // restore original extra
    gset->kextra = extra;
}

static void
genUpdateIntermResult(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    bool withMhitCond,
    UpdateResultFlags flags)
{
    char tmp[1024];
    const char *coordY, *coordX;
    char *revAlp, *alp;
    DataType dtype = gset->kextra->dtype;
    KernelExtraFlags kflags = gset->kextra->flags;
    const SubproblemDim *dim = &gset->subdims[1];
    const KernelVarNames *kvarNames = &gset->varNames;
    UpdateResultOp op;
    UpresVarNames uvars;
    const char* ctype;

    memset(&uvars, 0, sizeof(uvars));

    op = (flags & UPRES_WITH_BETA) ? UPRES_SUM : UPRES_SET;

    uvars.startRow = kvarNames->coordA;
    uvars.startCol = kvarNames->coordB;
    uvars.nrRows = "y";
    uvars.nrCols = "x";
    uvars.result = "B";
    uvars.ld = "ldb";

    ctype = dtypeBuiltinType(dtype);
    if (isComplexType(dtype)) {
        if (dtype == TYPE_COMPLEX_FLOAT) {
            revAlp = "div((float2)(-1.f, 0), alpha)";
            alp = "(float2)(1.f, 0)";
        }
        else {
            revAlp = "div((double2)(-1., 0), alpha)";
            alp = "(double2)(1., 0)";
        }
    }
    else {
        revAlp = "-1. / alpha";
        alp = "1.";
    }

    // inline result update
    flags |= UPRES_INLINE;

    coordY = kvarNames->coordA;
    coordX = kvarNames->coordB;

    /*
     * We should be careful here.
     *
     * The non tailed case of updateResult() is rewritted.
     * Now update result for tailed and non tailed cases have a bit
     * different semantics.
     *
     * The first one produces expressions like
     * 'dst = dst * beta + src * alpha'.
     *
     * Here 'dst' and 'src' may be private result stored in registers or
     * result to be updated in the global memory. Let the first one to be
     * designated as tileC and the second one as matC.
     *
     * The non tailed case produces expressions like
     * 'dst = matC * beta + tileC * alpha'.
     *
     * The second variant is more clear and native for the new implementation.
     * But as the difference is not eliminated, both the variants are
     * maintained here.
     */

    if (!(kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N))) {
        kgenBeginBranch(ctx, "");

        sprintf(tmp, "%s %s = %s;\n"
                     "%s alpha = beta;\n",
                ctype, "beta", revAlp, ctype);
        kgenAddStmt(ctx, tmp);

        updateResultGen(ctx,
            gset,
            CLBLAS_TRSM,
            op,
            flags & ~UPRES_WITH_BETA,
            &uvars);

        kgenEndBranch(ctx, NULL);
    }
    else {
        if (withMhitCond) {
            sprintf(tmp, "if ((%s < %s) && (%s < %s))",
                    coordY, kvarNames->sizeM, coordX, kvarNames->sizeN);
            kgenBeginBranch(ctx, tmp);
        }
        else {
            /* for x, y variables scope */
            kgenBeginBranch(ctx, NULL);
        }

        sprintf(tmp, "uint y = min(%luu, %s - (uint)%s);\n"
                     "uint x = min(%luu, %s - (uint)%s);\n",
                dim->y, kvarNames->sizeM, coordY,
                dim->x, kvarNames->sizeN, coordX);
        kgenAddStmt(ctx, tmp);

        sprintf(tmp, "if ((y == %lu) && (x == %lu))",
                dim->y, dim->x);
        kgenBeginBranch(ctx, tmp);

        sprintf(tmp, "%s %s = %s;\n"
                     "%s alpha = beta;\n",
                ctype, "beta", revAlp, ctype);
        kgenAddStmt(ctx, tmp);

        // optimized update
        updateResultGen(ctx,
            gset,
            CLBLAS_TRSM,
            op,
            flags & ~UPRES_WITH_BETA,
            &uvars);

        kgenEndBranch(ctx, NULL);

        flags |= UPRES_GENERIC;
        kgenBeginBranch(ctx, "else ");

        sprintf(tmp, "%s %s = %s;\n"
                     "%s %s = %s;\n",
                ctype, "beta", revAlp,
                ctype, "alpha", alp);
        kgenAddStmt(ctx, tmp);

        // not optimized update
        updateResultGen(ctx,
            gset,
            CLBLAS_TRSM,
            op,
            flags,
            &uvars);

        kgenEndBranch(ctx, NULL);
        kgenEndBranch(ctx, NULL);
    }
}

static void
genPreloadedTileMul(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    TileMulOpts *mulOpts,
    const Tile *parTile,
    const char* copy2LDSFuncName)
{
    char tmp[1024];
    KernelExtraFlags kflags = gset->kextra->flags;
    unsigned int bwidthOld;
    const char *oldNameB;
    const char *ptrName;

    getVectorTypeName(gset->kextra->dtype, parTile->vecLen, NULL, &ptrName);
    kgenPrintf(ctx, "lB.%s = tmpB;\n", ptrName);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    if (!isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B)) {
        sprintf(tmp, "%s(lB, uB, gid * %lu, k0, ldb);\n",
            copy2LDSFuncName, gset->subdims[0].x);
    }
    else {
        sprintf(tmp, "%s(lB, uB, k0, gid * %lu, ldb);\n",
            copy2LDSFuncName, gset->subdims[0].x);
    }
    kgenAddStmt(ctx, tmp);

    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);

    kgenAddStmt(ctx, "lB = lBMain;\n\n");

    mulOpts->memB = CLMEM_LOCAL_MEMORY;
    oldNameB = gset->varNames.B;
    bwidthOld = (unsigned int)gset->subdims[0].bwidth;
    gset->varNames.B = "lB";
    gset->subdims[0].bwidth = (parTile->trans) ? parTile->nrRows :
                                                 parTile->nrCols;

    tileMulGen(ctx, gset, mulOpts);

    gset->varNames.B = oldNameB;
    gset->subdims[0].bwidth = bwidthOld;
    mulOpts->memB = CLMEM_GLOBAL_MEMORY;
}

static void
initTiles(
    BlasGenSettings* gset,
    TileSet* tileSet,
    const struct SubproblemDim *subdims,
    KernelExtraFlags kflags,
    DataType dtype,
    PrivateStorageType storType)
{
    unsigned int rowsA;
    unsigned int rowsB;
    unsigned int rowsC;
    unsigned int colsA;
    unsigned int colsB;
    unsigned int colsC;
    bool transA;
    bool transB;
    unsigned int vecLenA;
    unsigned int vecLenB;
    unsigned int vecLenC;

    rowsA = (unsigned int)subdims[1].y;
    colsA = (unsigned int)szmax(subdims[1].y, subdims[1].bwidth);

    rowsB = (unsigned int)szmax(subdims[1].y, subdims[1].bwidth);
    colsB = (unsigned int)szmax(subdims[1].x, subdims[1].y);

    rowsC = (unsigned int)subdims[1].y;
    colsC = (unsigned int)subdims[1].x;

    transA = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A);
    transB = isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B);

    vecLenA = (unsigned int)((transA) ? subdims[1].y : subdims[1].bwidth);
    vecLenA = umin(vecLenA, MAX_TILE_VECLEN);
    vecLenB = (unsigned int)((transB) ? subdims[1].x : subdims[1].bwidth);
    vecLenB = umin(vecLenB, MAX_TILE_VECLEN);
    vecLenC = (transB) ? vecLenB : vecLenA;

    initTile(&tileSet->rectA, "a", (unsigned int)subdims[1].y,
             (unsigned int)subdims[1].bwidth, vecLenA, dtype,
             storType, transA, false);

    initTile(&tileSet->squareA, "a", (unsigned int)subdims[1].y,
             (unsigned int)subdims[1].y, vecLenA, dtype, storType,
             transA, false);

    initTile(&tileSet->origB, "b", (unsigned int)subdims[1].bwidth,
             (unsigned int)subdims[1].x, vecLenB, dtype, storType,
             !transB, false);

    initTile(&tileSet->bStage2, "b", (unsigned int)subdims[1].y,
             (unsigned int)subdims[1].x, vecLenB, dtype, storType,
             !transB, false);

    initTile(&tileSet->bAsSqA, "b", (unsigned int)subdims[1].y,
             (unsigned int)subdims[1].y, vecLenB, dtype, storType,
             transA, false);

    initTile(&tileSet->bAsC, "b", (unsigned int)subdims[1].y,
             (unsigned int)subdims[1].x, vecLenB, dtype, storType,
             gset->tileCY.trans, false);

    initTile(&gset->tileA, "a", rowsA, colsA,
             vecLenA, dtype, storType, transA, false);

    initTile(&gset->tileBX, "b", rowsB, colsB,
             vecLenB, dtype, storType, !transB, false);

    initTile(&gset->tileCY, "c", rowsC, colsC,
             vecLenC, dtype, storType, !transB, false);

    tileSet->A = gset->tileA;
    tileSet->B = gset->tileBX;
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
    ssize_t ret;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    DataType dtype = kextra->dtype;
    KernelExtraFlags kflags = kextra->flags;
    CLBLASKernExtra extraNew;
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    const char *ptrName;
    UpdateResultFlags upFlags = 0;
    TilePostFetchPrivate pfPriv;
    unsigned int l1Pans;
    bool b;
    Tile parTile;
    TrsmExtraParams *extraParams = (TrsmExtraParams *)kextra->solverPriv;
    int ldsLarge, lds_diagonal;
    bool isInline;
    TileSet tileSet;
    char copy2LDSFuncName[FUNC_NAME_MAXLEN];
    TailStatus tailStatus = 0;
    FetchAddrMode addrMode = 0;
    bool tailM = ((kflags & KEXTRA_TAILS_M) != 0);
    bool tailN = ((kflags & KEXTRA_TAILS_N) != 0);
    size_t alignK;

    if (pgran->wgDim != 1) {
        return -EINVAL;
    }

    l1Pans = (unsigned int)(subdims[0].x / subdims[1].x);

    memset(&gset, 0, sizeof(gset));
    gset.flags = BGF_WHOLE_A | BGF_EXPLICIT_INLINE | BGF_UPTRS;
    memcpy(gset.subdims, subdims, sizeof(SubproblemDim) * 2);
    // there is not need in block structure along K
    gset.subdims[0].bwidth = gset.subdims[1].bwidth;
    subdims = gset.subdims;

    /*
     * Since tiles are changed dynamically, e. g. in the main tilemul
     * loop they are rectangular, but at the second stage both A and B
     * tile storages are used for square tiles. One must adjust physical
     * vectorization accordindly, so as vector length might not be
     * greater than linear size of any tile
     */
    memcpy(&extraNew, kextra, sizeof(extraNew));
    extraNew.vecLenA = umin(kextra->vecLenA, (unsigned int)subdims[1].y);
    extraNew.vecLenB = umin(kextra->vecLenB, (unsigned int)subdims[1].y);

    gset.pgran = pgran;
    gset.kextra = &extraNew;
    initKernelVarNames(&gset.varNames);

    // multiplication options
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;
    mulOpts.core = (kextra->flags & KEXTRA_ENABLE_MAD) ? TILEMUL_MAD :
                                                         TILEMUL_MULADD;
    mulOpts.postFetch = NULL;
    mulOpts.flags = kextraToTilemulFlags(CLBLAS_TRSM, kflags);
    mulOpts.flags |= TILEMUL_EXTERN_RDECL | TILEMUL_NOT_INC_K;
    mulOpts.fctx = createFetchContext();
    if (mulOpts.fctx == NULL) {
        return -ENOMEM;
    }

    disableFetchOptLevels(mulOpts.fctx, FOPTLEV_TMP_COORD_PRECOMPUTING);

    isInline = (gset.flags & BGF_EXPLICIT_INLINE);

    initTiles(&gset, &tileSet, subdims, kflags, dtype,
              PRIV_STORAGE_VARIABLE_SET);

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        destroyFetchContext(mulOpts.fctx);
        return -ENOMEM;
    }

    kgenAddStmt(ctx, "#pragma OPENCL EXTENSION cl_amd_printf : enable\n\n");

    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);
    if (isComplexType(dtype)) {
        genComplexMathOperators(ctx, dtype);
    }
    if(!isInline) {
        genTileInverting(ctx, &gset, &tileSet);
    }

    if ( extraParams->ldsUse != LDS_NO_USE ) {
        SubproblemDim sdims;
        DBlockCopyFlags flags;
        unsigned int vecLen;

        if (!isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_B)) {
            sdims.x = gset.subdims[1].bwidth * extraParams->unrollingFactor;
            sdims.y = gset.subdims[0].x;
        }
        else {
            sdims.x = gset.subdims[0].x;
            sdims.y = gset.subdims[1].bwidth * extraParams->unrollingFactor;
        }

        vecLen = getVecLen(&gset, CLBLAS_TRSM, MATRIX_B);
        flags = (vecLen < 4) ? DBLOCK_COPY_NOT_VECTORIZE : 0;
        copyDataBlockGen(ctx, &sdims, gset.pgran, dtype,
                         DBLOCK_GLOBAL_TO_LOCAL, flags);
        kgenAddBlankLine(ctx);
        kgenGetLastFuncName(copy2LDSFuncName, FUNC_NAME_MAXLEN, ctx);
    }

    declareTrxmKernel(ctx, dtype, pgran, kflags, CLBLAS_TRSM, "Cached", false,
                      true);
    kgenBeginFuncBody(ctx);

    declareLocalVariables(ctx, &gset, &parTile, extraParams);
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    genTrxmBMatrShift(ctx, kflags, false);

    ptrName = dtypeUPtrField(dtype);

    sprintf(tmp, "uB.%s = B;\n\n", ptrName);
    kgenAddStmt(ctx, tmp);

    // external loop
    sprintf(tmp, "for (m0 = 0; m0 < M; m0 += %lu)", subdims[0].y);
    kgenBeginBranch(ctx, tmp);
    genZeroTile(ctx, &gset.tileCY);
    genSetupCoords(ctx, &gset, BLOCK_UPDATE);

    kgenAddStmt(ctx, "// Stage 1. Multiply and update with large blocks\n");

    gset.tileA = tileSet.rectA;
    gset.tileBX = tileSet.origB;

    if (!isMatrixUpper(kflags) && tailM) {
        addrMode |= FETCH_ADDR_A_CYCLICAL;
        setFetchAddrMode(mulOpts.fctx, addrMode);
    }

    ldsLarge = ((extraParams->ldsUse & LDS_USE_LARGE) != 0);
    alignK = subdims[1].bwidth;
    if (ldsLarge) {
        alignK *= extraParams->unrollingFactor;
    }

    if (ldsLarge) {
        const char *oldCoordB;
        FetchAddrMode bamode = addrMode | FETCH_ADDR_K_RELATIVE;
        bool withSkew;

        withSkew = useSkewedFetchB(&gset);
        if (!withSkew) {
            bamode |= FETCH_ADDR_B_RELATIVE;
        }
        else {
            bamode |= FETCH_ADDR_B_CYCLICAL;
        }

        setFetchAddrMode(mulOpts.fctx, bamode);

        if (tailN) {
            /*
             * Conditional branch for those items which hit into
             * matrix B with their matrix coordinates
             */
            sprintf(tmp, "if ((gid + 1) * %lu < N)", subdims[0].x);
            kgenBeginBranch(ctx, tmp);
        }

        if (isMatrixAccessColMaj(CLBLAS_TRSM, kflags, MATRIX_A)) {
            kgenPrintf(ctx, "uA.%s = A + k0 * lda;\n", ptrName);
        }
        else {
            kgenPrintf(ctx, "uA.%s = A + k0;\n", ptrName);
        }

        if (withSkew) {
            unsigned int bwidthOld;

            oldCoordB = gset.varNames.coordB;
            gset.varNames.coordB = "skewX";
            bwidthOld = gset.subdims[0].bwidth;
            gset.subdims[0].bwidth = (parTile.trans) ? parTile.nrRows :
                                                       parTile.nrCols;
            gset.subdims[0].bwidth = bwidthOld;
        }

        genInternalLoopCtl(ctx, subdims, kflags, alignK, alignK);
        genPreloadedTileMul(ctx, &gset, &mulOpts, &parTile, copy2LDSFuncName);
        genInternalLoopEnd(ctx);                             // loop over K

        if (withSkew) {
            gset.varNames.coordB = oldCoordB;
            setFetchAddrMode(mulOpts.fctx, bamode & ~FETCH_ADDR_B_CYCLICAL);
            // deliver from skew in the result before proceed to the next stage
            genTileCyclicalShift(ctx, &gset);
        }

        if (tailN) {
            kgenEndBranch(ctx, NULL);
            kgenBeginBranch(ctx, "else");
        }

        setFetchAddrMode(mulOpts.fctx, addrMode);
    }

    if (!ldsLarge || tailN) {
        genCheckShiftTailB(ctx, &gset, 0, &tailStatus);
        if ((kflags & KEXTRA_TAILS_N_LOWER) && !tailStatus) {
            addrMode |= FETCH_ADDR_B_CYCLICAL;
            setFetchAddrMode(mulOpts.fctx, addrMode);
        }

        if (tailN) {
            sprintfHitMatrixCond(tmp, MATRIX_B, "if (", ")");
            kgenBeginBranch(ctx, tmp);
        }

        genInternalLoopCtl(ctx, subdims, kflags, subdims[1].bwidth, alignK);
        tileMulGen(ctx, &gset, &mulOpts);
        genInternalLoopEnd(ctx);                             // loop over K

        if (tailN) {
            kgenEndBranch(ctx, NULL);
        }

        if (extraParams->ldsUse & LDS_USE_LARGE) {
            kgenEndBranch(ctx, NULL);
        }
    }

    sprintf(tmp, "uA.%s = A;\n\n", ptrName);
    kgenAddStmt(ctx, tmp);

    // processing tails along update dimension
    if (isMatrixUpper(kflags) &&
        ((kflags & KEXTRA_TAILS_K_LOWER) ||
          (ldsLarge && extraParams->unrolledTail))) {

        unsigned int tailChunks;

        tailChunks = (extraParams->ldsUse & LDS_USE_LARGE) ?
            extraParams->unrolledTail : 1;

        if (tailN) {
            char hitCond[1024];

            sprintfHitMatrixCond(hitCond, MATRIX_B, "(", ")");
            sprintf(tmp, "if ((currM + %lu < M) && %s)",
                    subdims[0].y, hitCond);
        }
        else {
            sprintf(tmp, "if (currM + %lu < M)", subdims[0].y);
        }
        kgenBeginBranch(ctx, tmp);

        if (kflags & KEXTRA_TAILS_K_LOWER) {
            setFetchAddrMode(mulOpts.fctx, addrMode | FETCH_ADDR_K_CYCLICAL);
            setFetchHandler(&mulOpts, &gset, defaultTilePostFetch, &pfPriv);
        }
        if (tailChunks > 1) {
            mulOpts.flags &= ~TILEMUL_NOT_INC_K;
            sprintf(tmp, "for (uint k1 = 0; k1 < %u; k1++)", tailChunks);
            kgenBeginBranch(ctx, tmp);
        }

		addrMode |= FETCH_ADDR_B_CYCLICAL;
        setFetchAddrMode(mulOpts.fctx, addrMode);
        tileMulGen(ctx, &gset, &mulOpts);
        if (tailChunks > 1) {
            kgenEndBranch(ctx, NULL);
            mulOpts.flags |= TILEMUL_NOT_INC_K;
        }

        kgenEndBranch(ctx, NULL);
    }

    gset.tileA = tileSet.squareA;

    kgenAddStmt(ctx, "\n/*\n"
                     " * Stage 2. A part of work items multiply got result on "
                     "a respective\n"
                     " * inverted diagonal block, and the remaining ones wait. "
                     "Then they perform\n"
                     " * one step of further intermediate result evaluation as "
                     "multiplying tile by tile.\n"
                     " * It continues until the whole panel of the "
                     "matrix A is processed\n"
                     " */\n");

    // one must deal further with square blocks strictly
    gset.subdims[0].bwidth = gset.subdims[1].bwidth = gset.subdims[1].y;

    sprintf(tmp, "for (m1 = 0; m1 < %lu; m1++)", subdims[0].y / subdims[1].y);
    kgenBeginBranch(ctx, tmp);

    if (extraParams->ldsUse & LDS_USE_DIAGONAL) {
        sprintf(tmp, "const int bid = lid %% %u;\n\n",
                l1Pans);
        kgenAddStmt(ctx, tmp);
    }

    /*
     * Update the intermediate result multiply on the inverted diagonal tile,
     * and write back
     */
    genSetupCoords(ctx, &gset, TILE_UPDATE);

    sprintfStage2Condition(tmp, &gset, 0);
    ret = kgenBeginBranch(ctx, tmp);

    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= tailStatusToUpresFlags(tailStatus);
    upFlags |= UPRES_PRIV_DEST | UPRES_WITH_BETA;
    genUpdateIntermResult(ctx, &gset, false, upFlags);

    kgenAddBlankLine(ctx);

    lds_diagonal = ((extraParams->ldsUse & LDS_USE_DIAGONAL) &&
                    (kflags & (KEXTRA_COLUMN_MAJOR)) == 0 &&
                    !(tailM || tailN) &&
                    !(upFlags & UPRES_NO_VECTORIZATION) &&
                    !isComplexType(kextra->dtype));

    /*
     * it's needed now to adjust addressing mode of A so as to don't
     * exceed the bound of A
     */
    if (tailM) {
        setFetchAddrMode(mulOpts.fctx,
                         addrMode | FETCH_ADDR_A_CYCLICAL |
                         FETCH_ADDR_K_CYCLICAL);
        extraNew.flags |= KEXTRA_TAILS_K_LOWER;
    }

    genMulOnDiagonalTile(ctx, &gset, &tileSet, &mulOpts);
    gset.tileBX = tileSet.bStage2;
    if (tailM) {
        setFetchHandler(&mulOpts, &gset, defaultTilePostFetch, &pfPriv);
    }

    kgenAddStmt(ctx, "// Write back the given result\n");

    upFlags = kextraToUpresFlags(CLBLAS_TRSM, kflags);
    upFlags |= tailStatusToUpresFlags(tailStatus);

    if (lds_diagonal) {
       sprintf(tmp, "tmpB[%%u * %u + bid]", l1Pans);
    }

    genResultUpdateWithFlags(ctx, CLBLAS_TRSM, &gset, upFlags,
                                 NULL, NULL, lds_diagonal ? tmp : NULL);

    kgenEndBranch(ctx, NULL);   // multiply on the inverted tile path
    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    // continue the tile update
    kgenAddBlankLine(ctx);
    sprintfStage2Condition(tmp, &gset, 1);
    kgenBeginBranch(ctx, tmp);
    genCheckShiftTailB(ctx, &gset, 0, &tailStatus);
    if (lds_diagonal) {
        // TODO: add here storing to LDS as well
    }
    else {
		addrMode |= FETCH_ADDR_B_CYCLICAL;
        setFetchAddrMode(mulOpts.fctx, addrMode);
        tileMulGen(ctx, &gset, &mulOpts);
    }
    kgenEndBranch(ctx, NULL);           // tile update path
    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    kgenEndBranch(ctx, NULL);           // second stage loop

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "currM -= %lu;\n", subdims[0].y);
        kgenAddStmt(ctx, tmp);
    }

    kgenEndBranch(ctx, NULL);           // loop over M

    ret = kgenEndFuncBody(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyFetchContext(mulOpts.fctx);
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
    (void)dim;
    (void)dtype;
    (void)ldsSize;
    (void)kernelArgs;

    return true;
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D | SF_TOP_INPUT_SQUARE_BLOCKS);
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
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx], blasArgs->offBX);
    }
}

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    CLBlasKargs *kargs = (CLBlasKargs*)args;
    TrsmExtraParams *extraParams = (TrsmExtraParams *)kextra->solverPriv;
    size_t loadBatch;
    unsigned int wgSize;
    unsigned int workRatio;
    unsigned int ldsUse = LDS_NO_USE;
    KernelExtraFlags kflags = kextra->flags;
    SubproblemDim globDim;
    bool isAmdGPU;

    /*
     * Calculate size of the batch loaded from global to local memory
     * at each iteration of the stage 1. Choose such unrolling factor
     * that allow each work item to load at least 16 bytes that provides
     * efficient global memory access
     */
    loadBatch = subdims[0].x * subdims[1].bwidth * dtypeSize(kargs->dtype);
    wgSize = (unsigned int)((subdims[0].x / subdims[1].itemX) *
                            (subdims[0].y / subdims[1].itemY));
    if (loadBatch < wgSize) {
        workRatio = 1;
    }
    else {
        workRatio = 16 / ((unsigned int)loadBatch / wgSize);
        if (!workRatio) {
            workRatio = 1;
        }
    }

#ifndef NDEBUG
    {
        const char *envImpl = getenv("AMD_CLBLAS_TRSM_LDSUSE");

        if (envImpl != NULL) {
            unsigned int w = atoi(envImpl);
            ldsUse = w % 10;
            w = w / 10;
            workRatio = w > 0 ? w : workRatio;
        }
    }
#endif

    ldsUse = LDS_NO_USE;
    isAmdGPU = ((kflags & KEXTRA_VENDOR_AMD) != 0);
    if ((isAmdGPU && !(kflags & (KEXTRA_TAILS_K_LOWER | KEXTRA_TAILS_M_LOWER)))
        || (!isAmdGPU && !(kflags & KEXTRA_TAILS_M))) {

        ldsUse = LDS_USE_LARGE;
    }

    kargsToProbDims(&globDim, CLBLAS_TRSM, args, false);
    extraParams->ldsUse = ldsUse;
    extraParams->unrollingFactor = workRatio;
    extraParams->unrolledTail = (unsigned int)(((globDim.bwidth %
             (subdims[1].bwidth * workRatio)) + subdims[1].bwidth - 1) /
                                                subdims[1].bwidth);

    fixupTrxmKargs(kargs);
}

static bool
checkCalcDecompDedicated(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check)
{
    bool ret = true;

    DUMMY_ARG_USAGE(subdimsNum);

    if (check == PGRAN_CHECK) {
        unsigned int minSize, maxSize;

        maxSize = (dtype == TYPE_COMPLEX_DOUBLE) ? 4 : 8;
        minSize = (dtype == TYPE_COMPLEX_DOUBLE) ? 1 : 2;
        ret = decompSanityCheck(subdims, minSize, maxSize, 24, dtype, true);
        ret = ret && (subdims[0].bwidth == subdims[1].bwidth);
        ret = ret && (pgran->wgSize[0] == 64);
    }
    else {
        calcPgranDedicated(pgran, subdims, -1, 3);
    }

    return ret;
}

void
initTrsmLdsLessCachedPattern(MemoryPattern *mempat)
{
    mempat->name = "2-staged cached global memory based block trsm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 0;
    mempat->sops = &trsmSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

#if 0

static int
getDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void * pArgs)
{
    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;
    pgran->wgSize[1] = 1;

    subdims[0].x = subdims[0].itemX = 32;
    subdims[0].y = 64;
    subdims[0].itemY = SUBDIM_UNUSED;
    subdims[0].bwidth = subdims[1].bwidth = 4;
    subdims[1].x = subdims[1].itemX = 8;
    subdims[1].y = subdims[1].itemY = 4;
}

#endif
