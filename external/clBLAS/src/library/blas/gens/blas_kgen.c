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
 * common stuff for blas related
 * kernel generators
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <list.h>
#include <clblas_stddef.h>

#include <matrix_props.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "blas_kgen.h"
#include "gen_helper.h"
#include "tile_iter.h"
#include "kerngen.h"

#define IDX_INVAL ((unsigned int)-1)

enum {
    COORD_STRLEN = 64
};

static unsigned int
getTmpVecLen(
    const BlasGenSettings *gset,
    UpdateResultFlags uflags,
    const char **vecName)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    unsigned int vecLen;

    if (isComplexType(kextra->dtype) || (uflags & (UPRES_GENERIC |
                                         UPRES_NO_VECTORIZATION))) {
        vecLen = 1;
    }
    else {
        vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? kextra->vecLenC :
                                                       kextra->vecLen;
        getVectorTypeName(kextra->dtype, vecLen, vecName, NULL);
    }

    return vecLen;
}

/*
 * Try to transform kernel string to integer.
 * Return -1. If this is not a number.
 */
static int
stringToInt(const char *str, unsigned int *num)
{
    char *end;
    unsigned int n;
    int ret = -1;

    n = (unsigned int)strtol(str, &end, 10);
    // believe it is a number if the string has been parsed completely
    if ((end != str) && (*end == '\0')) {
        *num = n;
        ret = 0;
    }

    return ret;
}

void
sprintfVecChunk(
    char *chunk,
    unsigned int vecLen,
    unsigned int clen,
    unsigned int vecOff)
{
    const char *vect = "0123456789abcdef";

    if (clen == vecLen) {
        chunk[0] = '\0';
    }
    else {
        snprintf(chunk, clen + 3, ".s%s", vect + vecOff);
        chunk[clen + 2] = '\0';
    }
}

unsigned int
getVecLen(const BlasGenSettings *gset, BlasFunctionID funcID, MatrixRole mrole)
{
    unsigned int vecLen = 0;
    const CLBLASKernExtra *kextra = gset->kextra;

    DUMMY_ARG_USAGE(funcID);

    if (!(gset->flags & BGF_DISTINCT_VECLEN)) {
        vecLen = umin(kextra->vecLenA, kextra->vecLenB);
        vecLen = umin(vecLen, kextra->vecLenC);
    }
    else {
        switch (mrole) {
        case MATRIX_A:
            vecLen = kextra->vecLenA;
            break;
        case MATRIX_B:
            vecLen = kextra->vecLenB;
            break;
        case MATRIX_C:
            vecLen = kextra->vecLenC;
            break;
        default:
            break;
        }
    }

    return vecLen;
}

void
genScaleLeadingDimensions(struct KgenContext *ctx, const BlasGenSettings *gset)
{
    const KernelVarNames *kvars;
    unsigned int vecLen;
    bool done = false;

    if (!(gset->flags & BGF_LD_IN_VECTORS)) {
        return;
    }

    kvars = &gset->varNames;

    vecLen = getVecLen(gset, CLBLAS_GEMM, MATRIX_A);
    if ((kvars->lda != NULL) && (vecLen > 1)) {
        kgenPrintf(ctx, "%s /= %u;\n", kvars->lda, vecLen);
        done = true;
    }

    vecLen = getVecLen(gset, CLBLAS_GEMM, MATRIX_B);
    if ((kvars->ldb != NULL) && (vecLen > 1) && (kvars->ldb != kvars->lda)) {
        kgenPrintf(ctx, "%s /= %u;\n", kvars->ldb, vecLen);
        done = true;
    }

    vecLen = getVecLen(gset, CLBLAS_GEMM, MATRIX_C);
    if ((kvars->ldc != NULL) && (vecLen > 1) &&
        (kvars->ldc != kvars->lda) && (kvars->ldc != kvars->ldb)) {

        kgenPrintf(ctx, "%s /= %u;\n", kvars->ldc, vecLen);
        done = true;
    }

    if (done) {
        kgenAddBlankLine(ctx);
    }
}

void
getPrivateAreaInfo(
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    MatrixRole mrole,
    PrivateArea *area)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    const SubproblemDim *dim = &gset->subdims[1];

    area->vecLen = getVecLen(gset, funcID, mrole);
    getVectorTypeName(kextra->dtype, area->vecLen, &area->typeName, NULL);
    if (mrole == MATRIX_C) {
        area->size = (unsigned int)(divRoundUp(dim->x, area->vecLen) * dim->y);
    }
    else {
        size_t h = (mrole == MATRIX_A) ? dim->y : dim->x;

        area->size = (unsigned int)(h * dim->bwidth / area->vecLen);
    }
}

void
declarePrivateArea(
    struct KgenContext *ctx,
    const PrivateArea *area,
    const char *baseName,
    PrivateStorageType storType)
{
    char tmp[1024];
    unsigned int i;

    // TODO: separate case for size equal to 1
    if (storType == PRIV_STORAGE_ARRAY) {
        sprintf(tmp, "%s %s[%u];\n", area->typeName, baseName, area->size);
    }
    else {
        char *p;

        sprintf(tmp, "%s %s0", area->typeName, baseName);
        p = tmp + strlen(tmp);
        for (i = 1; i < area->size; i++) {
            sprintf(p, ", %s%u", baseName, i);
            p += strlen(p);
        }
        strcpy(p, ";\n");
    }

    kgenAddStmt(ctx, tmp);
}

int
defaultTilePostFetch(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv)
{
    char tmp[1024], cond[128];
    Kstring src;
    TilePostFetchPrivate *pfPriv = (TilePostFetchPrivate*)priv;
    bool distVect = (pfPriv->gset->flags & BGF_DISTINCT_VECLEN);
    const KernelVarNames *vnames = &pfPriv->gset->varNames;
    const CLBLASKernExtra *kextra = pfPriv->gset->kextra;
    const SubproblemDim *dim = &pfPriv->gset->subdims[1];
    BlasFunctionID funcID = pfPriv->funcID;
    const Tile* tile;
    bool partA;
    unsigned int step;
    unsigned int i, j;
    int ret = 0;
    unsigned int maxJ = 0;
    unsigned int maxI = 0;

    if (!isNeedZeroTileTail(funcID, dim, kextra, mrole, distVect)) {
        return 0;
    }

    if (mrole == MATRIX_A) {
        tile = &pfPriv->gset->tileA;
        maxJ = tile->nrCols;
        maxI = tile->nrRows;
    }
    else {
        tile = &pfPriv->gset->tileBX;
        maxJ = tile->nrRows;
        maxI = tile->nrCols;
    }

    partA = (mrole == MATRIX_A) && tile->trans &&
            !(pfPriv->gset->flags & BGF_WHOLE_A);
    step = tileLineSegmentLen(tile);
    step = (tile->trans ^ (mrole == MATRIX_A)) ? 1 : step;

    for (j = 0; (j < maxJ) && !ret; j++) {
        unsigned int k;

        k = umax(j, (unsigned int)pfPriv->fetchNumA);
        if (k) {
            sprintf(tmp, " + %u", k);
        }
        else {
            tmp[0] = '\0';
        }
        sprintf(cond, "(%s%s < %s)", vnames->k, tmp, vnames->sizeK);

        for (i = 0; (i < maxI) && !ret; i += step) {
            if (mrole != MATRIX_A) {
                sprintfTileElement(&src, tile, j, i, step);
            }
            else {
                sprintfTileElement(&src, tile, i, j, step);
            }
            sprintf(tmp, "%s = %s ? %s : 0;\n", src.buf, cond, src.buf);
            ret = kgenAddStmt(ctx, tmp);
        }
    }

    if (partA) {
        pfPriv->fetchNumA++;
    }

    if ((tile->nrCols * tile->nrRows / tile->vecLen > 1) && !ret) {
        ret = kgenAddBlankLine(ctx);
    }

    return ret;
}

char
dtypeToBlasPrefix(DataType dtype)
{
    char c;

    if (dtype == TYPE_FLOAT) {
        c = 's';
    }
    else {
        c = dtypeToPrefix(dtype);
    }

    return c;
}

TileMulFlags
kextraToTilemulFlags(BlasFunctionID funcID, KernelExtraFlags kflags)
{
    TileMulFlags mf = TILEMUL_NO_FLAGS;

    if (isMatrixAccessColMaj(funcID, kflags, MATRIX_A)) {
        mf |= TILEMUL_TRA;
    }
    if (isMatrixConj(kflags, MATRIX_A)) {
        mf |= TILEMUL_CONJA;
    }
    if (!isMatrixAccessColMaj(funcID, kflags, MATRIX_B)) {
        mf |= TILEMUL_TRB;
    }
    if (isMatrixConj(kflags, MATRIX_B)) {
        mf |= TILEMUL_CONJB;
    }

    return mf;
}

void
getResultGPRsInfo(
    DataType dtype,
    const SubproblemDim *dims,
    unsigned int vecLen,
    unsigned int *nrRegs,
    const char **typeName)
{
    if (isComplexType(dtype)) {
        if (nrRegs) {
            *nrRegs = (unsigned int)(dims->x * dims->y);
        }
        if (typeName != NULL) {
            *typeName = dtypeBuiltinType(dtype);
        }
    }
    else {
        // handle different vecLen values and fetch vector sizes
        if (nrRegs) {
            *nrRegs = (unsigned int)(divRoundUp(dims->x, vecLen) * dims->y);
        }
        if (typeName != NULL) {
            getVectorTypeName(dtype, vecLen, typeName, NULL);
        }
    }
}

static void genVectorCPtr( struct KgenContext *pCtx,
    const BlasGenSettings *pGSet,
    const char* GPtrName,
    const char* VCPtrName )
{
    const char *typeName;
    unsigned int vecLen = 0;

    vecLen = getVecLen( pGSet, 0, MATRIX_C );
    vecLen = vecLen > pGSet->tileCY.vecLen ?
        pGSet->tileCY.vecLen :
        vecLen;

    getVectorTypeName( pGSet->kextra->dtype,
        vecLen,
        &typeName,
        NULL );

    if ( 0 == (pGSet->flags & BGF_LD_IN_VECTORS) ) {

        vecLen = 1;
    }
    // Blas function ID is omitted
    if ( isComplexType( pGSet->kextra->dtype ) ) {
        vecLen *= 2;
    }

    if ( isDoubleBasedType(pGSet->kextra->dtype) ) {

        if ( 1 == vecLen ) {

            kgenPrintf(
                pCtx,
                "__global %s *%s = %s.d;\n",
                typeName,
                VCPtrName,
                GPtrName);
        }
        else {

            kgenPrintf( pCtx,
                "__global %s *%s = %s.d%dv;\n",
                typeName,
                VCPtrName,
                GPtrName,
                vecLen);
        }
    }
    else {

        if ( 1 == vecLen ) {

            kgenPrintf(
                pCtx,
                "__global %s *%s = %s.f;\n",
                typeName,
                VCPtrName,
                GPtrName);
        }
        else {

            kgenPrintf( pCtx,
                "__global %s *%s = %s.f%dv;\n",
                typeName,
                VCPtrName,
                GPtrName,
                vecLen);
        }
    }
}

static void
updateOptimResultGen(
    struct KgenContext *pCtx,
    const BlasGenSettings *pGSet,
    BlasFunctionID funcID,
    UpdateResultOp op,
    UpdateResultFlags flags)
{
    KernelExtraFlags kflags = pGSet->kextra->flags;
    Tile tempCTile;
    Tile fullCTile;
    unsigned int physVecLenC;
    DataType dtype;
    const KernelVarNames *pVNames = NULL;
    PhysTileIterator physIter;
    PhysTileIterator blkIter;
    char cPtrName[] = "pC";
    const char *typeNameC;
    bool phyTrans = 0;
    unsigned int vecLen = 0;
    unsigned int nBlocks = 0;
    unsigned int i = 0;

    Kstring cElem;
    Kstring tempCElem;
    Kstring kstrFirst;
    Kstring kstrSecond;
    Kstring kstrThird;
    Kstring expr;

    //EINVAL
    if ( NULL == pCtx ||
        NULL == pGSet ) {

        return;
    }

    dtype = pGSet->kextra->dtype;
    pVNames = &pGSet->varNames;
    phyTrans = ( (flags & UPRES_COLUMN_MAJOR ) != 0 );

    physVecLenC = getVecLen( pGSet, funcID, MATRIX_C );
    getVectorTypeName( dtype,
        getVecLen( pGSet,0,MATRIX_C ),
        &typeNameC,
        NULL );

    // declare private C pointer
    genVectorCPtr( pCtx, pGSet, "uC", "pC" );

    kgenAddBlankLine( pCtx );

    // calculate the number of blocks, update should be divided on
    nBlocks = pGSet->tileCY.nrCols * pGSet->tileCY.nrRows/(
        pGSet->tileA.nrCols*pGSet->tileA.nrRows +
        pGSet->tileBX.nrCols*pGSet->tileBX.nrRows );

    if( pGSet->tileCY.nrCols * pGSet->tileCY.nrRows%(
        pGSet->tileA.nrCols*pGSet->tileA.nrRows +
        pGSet->tileBX.nrCols*pGSet->tileBX.nrRows ) ){

        nBlocks++;
    }

    nBlocks = roundUpPow2( (int)nBlocks );

    // declare the temporary C tile
    // temporary C tile must have the same transposition as C matrix
    // for read-write optimization it also has the same vectorization
    if ( phyTrans ) {

        if ( nBlocks > pGSet->tileCY.nrCols ) {
            nBlocks = pGSet->tileCY.nrCols;
        }

        initTile( &tempCTile,
            "tempC",
            pGSet->tileCY.nrRows,
            pGSet->tileCY.nrCols/nBlocks,
            pGSet->tileCY.vecLen,
            dtype,
            PRIV_STORAGE_VARIABLE_SET,
            phyTrans,
            true );

        initTile( &fullCTile,
            "fullC",
            pGSet->tileCY.nrRows,
            pGSet->tileCY.nrCols,
            pGSet->tileCY.vecLen,
            dtype,
            PRIV_STORAGE_VARIABLE_SET,
            phyTrans,
            true);
    }
    else {

        if ( nBlocks > pGSet->tileCY.nrRows ) {
            nBlocks = pGSet->tileCY.nrRows;
        }

        initTile( &tempCTile,
            "tempC",
            pGSet->tileCY.nrRows/nBlocks,
            pGSet->tileCY.nrCols,
            pGSet->tileCY.vecLen,
            dtype,
            PRIV_STORAGE_VARIABLE_SET,
            phyTrans,
            true );

        initTile( &fullCTile,
            "fullC",
            pGSet->tileCY.nrRows,
            pGSet->tileCY.nrCols,
            pGSet->tileCY.vecLen,
            dtype,
            PRIV_STORAGE_VARIABLE_SET,
            phyTrans,
            true);
    }

    declareOneTileStorage( pCtx, &tempCTile );

    // splitting update result on several blocks to prevent
    // increasing GPR usage
    for ( i = 0; i < nBlocks; i++ ) {

        kgenAddBlankLine(pCtx);

        // fetch ------------------------------------------------------------------
        vecLen = umin( physVecLenC, pGSet->tileCY.vecLen );
        vecLen = umin( vecLen, tileLineSegmentLen(&tempCTile) );

        iterInit( &blkIter, &tempCTile, vecLen, 0 );
        iterInit( &physIter, &fullCTile, vecLen, 0 );

        iterSeekPhys( &physIter, blkIter.nrLines * i, blkIter.vec );

        if (op == UPRES_SUM) {
            for ( ; 0 == iterIsEnd( &blkIter ); iterIterate( &blkIter ),
                                               iterIterate( &physIter ) ) {

                emptyKstring( &kstrFirst );
                emptyKstring( &kstrSecond );
                emptyKstring( &kstrThird );
                emptyKstring( &cElem );
                emptyKstring( &tempCElem );

                sprintfTileElement( &tempCElem,
                    &tempCTile,
                    blkIter.row,
                    blkIter.col,
                    vecLen);

                ksprintf( &kstrFirst, "%d", physIter.line );
                ksprintf( &kstrSecond, "%s", pVNames->ldc );
                ksprintf( &kstrThird, "%d", blkIter.vec );

                sprintfFastScalarMad( &expr,
                    &kstrFirst,
                    &kstrSecond,
                    vecLen,//physVecLenC,//scale ldc
                    &kstrThird);

                kgenPrintf( pCtx,
                    "%s = %s[%s];\n",
                    tempCElem.buf,
                    cPtrName,
                    expr.buf );

            }
        }

        // beta ---------------------------------------------------------------
        if ( flags & UPRES_WITH_BETA ) {

            if ( isComplexType(dtype) ||
                ( pGSet->tileCY.trans != tempCTile.trans ) ) {
                vecLen = 1;
            }
            //TODO: for real datatype find longest available veclen can be used
            //to generate more compact code
            else {
                vecLen = pGSet->tileCY.vecLen;
            }
            vecLen = umin( vecLen, tileLineSegmentLen(&tempCTile) );

            iterInit( &blkIter, &tempCTile, vecLen, 0 );

            for ( ; 0 == iterIsEnd( &blkIter ); iterIterate( &blkIter ) ) {

                sprintfTileElement( &tempCElem,
                    &tempCTile,
                    blkIter.row,
                    blkIter.col,
                    vecLen);

                if ( isComplexType(dtype) ) {
                    //complex mad
                    ksprintf( &kstrSecond, "%s", pVNames->beta );
                    sprintfComplexMulUpdate( &expr,
                        &tempCElem,
                        &tempCElem,
                        &kstrSecond,
                        NULL,
                        isDoubleBasedType(dtype),
                        0,
                        0,
                        0 );
                    kgenPrintf( pCtx, "%s", expr.buf );
                }
                else {
                    if ((kflags & KEXTRA_ENABLE_MAD) != 0) {
                        kgenPrintf( pCtx,
                            "%s = mad(%s, %s, 0);\n",
                            tempCElem.buf,
                            tempCElem.buf,
                            pVNames->beta);
                    }
                    else {
                        kgenPrintf( pCtx,
                            "%s = %s * %s;\n",
                            tempCElem.buf,
                            tempCElem.buf,
                            pVNames->beta);
                    }
                }
            }
        }

        // alpha---------------------------------------------------------------
        if ( (phyTrans == pGSet->tileCY.trans) && (!isComplexType(dtype)) ) {

            vecLen = pGSet->tileCY.vecLen;
        }
        else {
            vecLen = 1;
        }
        vecLen = umin( vecLen, tileLineSegmentLen(&tempCTile) );

        iterInit( &blkIter, &tempCTile, vecLen, 0 );
        iterInit( &physIter, &fullCTile, vecLen, 0 );

        iterSeekPhys( &physIter, blkIter.nrLines * i, blkIter.vec );

        for ( ; 0 == iterIsEnd( &blkIter ); iterIterate( &blkIter ),
                                            iterIterate( &physIter) ) {

            const Kstring *dst;

            dst = (flags & UPRES_PRIV_DEST) ? &cElem : &tempCElem;

            sprintfTileElement( &tempCElem,
                &tempCTile,
                blkIter.row,
                blkIter.col,
                vecLen);

            sprintfTileElement( &cElem,
                &pGSet->tileCY,
                physIter.row,
                physIter.col,
                vecLen);

            // complex
            if ( isComplexType(dtype) ) {

                ksprintf( &kstrSecond, "%s", pVNames->alpha );

                // upres op: sum or set, if set, third argument
                // of complex mad() is zero
                sprintfComplexMulUpdate( &expr,
                    dst,
                    &cElem,
                    &kstrSecond,
                    (op == UPRES_SUM) ? &tempCElem : NULL,
                    isDoubleBasedType(dtype),
                    0,
                    0,
                    0);
                kgenPrintf( pCtx, "%s", expr.buf );

            }
            // real
            else {

                // upres op: sum or set, if set, third argument
                // of mad() is zero
                if ((kflags & KEXTRA_ENABLE_MAD) != 0) {
                    kgenPrintf( pCtx,
                        "%s = mad(%s, %s, %s);\n",
                        dst,
                        cElem.buf,
                        pVNames->alpha,
                        (op == UPRES_SUM) ? tempCElem.buf : "0" );
                }
                else {
                    kgenPrintf( pCtx,
                        "%s = %s * %s + %s;\n",
                        dst,
                        cElem.buf,
                        pVNames->alpha,
                        (op == UPRES_SUM) ? tempCElem.buf : "0" );
                }
            }
        }

        if (flags & UPRES_PRIV_DEST) {
            return;
        }

        // store---------------------------------------------------------------
        vecLen = umin( physVecLenC, pGSet->tileCY.vecLen );
        vecLen = umin( vecLen, tileLineSegmentLen( &tempCTile ) );

        iterInit( &blkIter, &tempCTile, vecLen, 0 );
        iterInit( &physIter, &fullCTile, vecLen, 0 );

        iterSeekPhys( &physIter, blkIter.nrLines * i, blkIter.vec );

        for ( ; 0 == iterIsEnd( &blkIter ); iterIterate( &blkIter ),
                                            iterIterate( &physIter ) ) {

            emptyKstring( &kstrFirst );
            emptyKstring( &kstrSecond );
            emptyKstring( &kstrThird );
            emptyKstring( &cElem );
            emptyKstring( &tempCElem );

            sprintfTileElement( &tempCElem,
                &tempCTile,
                blkIter.row,
                blkIter.col,
                vecLen);

            ksprintf( &kstrFirst, "%d", physIter.line );
            ksprintf( &kstrSecond, "%s", pVNames->ldc );
            ksprintf( &kstrThird, "%d", blkIter.vec );

            sprintfFastScalarMad( &expr,
                &kstrFirst,
                &kstrSecond,
                vecLen,//physVecLenC,//scale ldc
                &kstrThird);

            kgenPrintf( pCtx,
                "%s[%s] = %s;\n",
                cPtrName,
                expr.buf,
                tempCElem.buf );

        }
    }

}

int
genUpdateResultSingle(
    struct KgenContext *ctx,
    const char *dst,
    const char *src,
    const BlasGenSettings *gset,
    UpdateResultOp op,
    UpdateResultFlags flags)
{
    char tmp[1024];
    char *p;
    const char *opStr;
    UpdateResultFlags m;
    int r;
    bool isComplex = isComplexType(gset->kextra->dtype);

    // copy destination with respective operator and additional operations
    if (flags & UPRES_WITH_BETA) {
        if (isComplex) {
            sprintf(tmp, "%s = %s * betaR + %s.yx * betaI + ",
                    dst, dst, dst);
        }
        else {
            sprintf(tmp, "%s = %s * beta + ", dst, dst);
        }
    }
    else {
        opStr = (op == UPRES_SET) ? "=" : "+=";
        sprintf(tmp, "%s %s ", dst, opStr);
    }

    m = UPRES_WITH_BETA | UPRES_GENERIC;
    if (isComplex && ((flags & m) == m)) {
        strcat(tmp, "\n                    ");
    }
    p = tmp + strlen(tmp);

    // multiply source
    if (flags & UPRES_WITHOUT_ALPHA) {
        sprintf(p, "%s;\n", src);
    }
    else {
        if (isComplex) {
            sprintf(p, "%s * alphaR + %s.yx * alphaI;\n", src, src);
        }
        else {
            sprintf(p, "%s * alpha;\n", src);
        }
    }

    r = kgenAddStmt(ctx, tmp);

    return (r) ? -EOVERFLOW : 0;
}

static void
updateGenericResultGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    size_t pitch,
    UpresVarNames* uvars,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const char *cachedName)
{
    char tmp[1024], dst[128], src[128];
    const char *boundNames[2] = {uvars->nrRows, uvars->nrCols};
    const char *vecType = NULL;
    const char *vFieldVectorized;
    DataType dtype = gset->kextra->dtype;
    unsigned int wvlen;
    unsigned int sizes[2];
    const char*  vfield = dtypeUPtrField(dtype);
    bool tra = ((flags & UPRES_COLUMN_MAJOR) != 0);
    bool row = ((flags & UPRES_TAIL_ROW));
    bool col = ((flags & UPRES_TAIL_COL));
    bool iwc = ((flags & UPRES_INDEXING_WITH_CONSTANTS) != 0) ||
                (gset->tileCY.storType != PRIV_STORAGE_ARRAY);
    int l0;
    int l1;
    bool revert = false;

    Kstring kstr;
    int rowId;
    int colId;

    sizes[0] = (unsigned int)gset->subdims[1].y;
    sizes[1] = (unsigned int)gset->subdims[1].x;

    if (iwc) {
        const char* l0var =  boundNames[tra];
        revert =  (tra && col) || (!tra && row);

        if (revert) {
            sprintf(tmp, "uC.%s += (%s-1) * %s;\n", vfield, l0var, uvars->ld);
        }
        else {
            sprintf(tmp, "\n");
        }
        kgenAddStmt(ctx, tmp);

    }
    wvlen = getTmpVecLen(gset, flags, &vecType);
    if (!iwc) {
        getVectorTypeName(dtype, wvlen, NULL, &vFieldVectorized);
        sprintf(tmp, "res.%s = c;\n", vFieldVectorized);
        kgenAddStmt(ctx, tmp);
    }

    if (flags & (UPRES_TAIL_ROW | UPRES_TAIL_COL)) {
        char offStr[64];
        char *p = offStr;

        offStr[0] = '\0';
        if (flags & UPRES_TAIL_ROW) {
            sprintf(offStr, " + (%u - %s) * %lu",
                    sizes[0], uvars->nrRows, pitch);
            p += strlen(offStr);
        }
        if (flags & UPRES_TAIL_COL) {
            sprintf(p, " + (%u - %s)", sizes[1], uvars->nrCols);
        }
        if (iwc) {
            sprintf(tmp, "res.%s = uC.%s%s;\n", vfield, vfield, offStr);
            sprintf(tmp, "\n");
        }
        else {
            sprintf(tmp, "res.%s = res.%s%s;\n", vfield, vfield, offStr);
        }
        kgenAddStmt(ctx, tmp);

    }
    if (iwc) {
        int l0st = 1; int l0en = sizes[tra];
        int l1st = 1; int l1en = sizes[1-tra];

        const char* l0var =  boundNames[tra];
        const char* l1var = boundNames[1-tra];

        for (l0 = l0en; l0 >= l0st; l0--) {

            sprintf(tmp, "if (%s) ",l0var);
            kgenBeginBranch(ctx, tmp);

            sprintf(tmp, "switch (%s)", l1var);
            kgenBeginBranch(ctx, tmp);

            for (l1 = l1en; l1 >= l1st; l1--) {
                sprintf(tmp, "case %d:\n", l1);
                kgenAddStmt(ctx, tmp);

                if (tra) {
                    rowId = (row)? (l1en-l1): (l1-l1st);
                    colId = (col)? (l0-l0st): (l0en-l0);
                }
                else {
                    ///////////////////////////
                    rowId = (row)? (l0-l0st): (l0en-l0);
                    colId = (col)? (l1en-l1) : (l1-l1st);
                }

                if ((tra && row) || (!tra && col)) {
                     sprintf(dst, "uC.%s[(%s+%d) %% %i]",
                             vfield, l1var, (l1en - l1),  (int)l1en);
                }
                else {
                   sprintf(dst, "uC.%s[%d]", vfield, (l1-l1st));
                }

                sprintfTileElement(&kstr, &gset->tileCY, rowId, colId, wvlen);

                if (flags & UPRES_PRIV_DEST) {
                    genUpdateResultSingle(ctx, kstr.buf, dst, gset, op, flags);
                }
                else {
                    genUpdateResultSingle(ctx, dst, kstr.buf, gset, op, flags);
                }
            }
            kgenEndBranch(ctx, NULL);

            if (revert) {
                sprintf(tmp, "uC.%s -= %s;\n", vfield, uvars->ld);
            }
            else {
                sprintf(tmp, "uC.%s += %s;\n", vfield, uvars->ld);
            }

            kgenAddStmt(ctx, tmp);

            sprintf(tmp, "%s--;\n", l0var);
            kgenAddStmt(ctx, tmp);
            kgenEndBranch(ctx, NULL);
        }

    }
    else {
        sprintf(tmp, "for (i = 0; i < %s; i++)", boundNames[tra]);
        kgenBeginBranch(ctx, tmp);
        sprintf(tmp, "for (j = 0; j < %s; j++)", boundNames[1 - tra]);
        kgenBeginBranch(ctx, tmp);
        sprintf(dst, "uC.%s[i * %s + j]", vfield, uvars->ld);
        if (cachedName) {
            unsigned int i;
            char tmpcachedName[80] = " = ";
            strcat(tmpcachedName, cachedName);
            for (i = 3; i < strlen(tmpcachedName); i++) {
                if (strncmp(tmpcachedName+i, "%u", 2) == 0) {
                    tmpcachedName[i+1] = 's';
                }
            }
            sprintf(tmp, tmpcachedName, "i", "[j]");
            strcat(dst, tmp);
        }
        // result (res) can be transposed independently of the matrix C
        // If the transposition of "C" and "result" is not consistent
        // then change the calculation of the index for "result"
        if (gset->tileCY.trans ^ tra) {
            sprintf(src, "res.%s[j * %lu + i]", vfield, pitch);
        }
        else {
            sprintf(src, "res.%s[i * %lu + j]", vfield, pitch);
        }
        if (flags & UPRES_PRIV_DEST) {
            genUpdateResultSingle(ctx, src, dst, gset, op, flags);
        }
        else {
            genUpdateResultSingle(ctx, dst, src, gset, op, flags);
        }
        kgenEndBranch(ctx, NULL);
        kgenEndBranch(ctx, NULL);
    }
}

//-----------------------------------------------------------------------------

int
updateResultGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const UpresVarNames *uvarNames)
{
    char tmp[1024];
    char *p = tmp;
    const char *typeName;
    const char *vecType = NULL;
    const char *vfield;
    const char *suff1;
    const char *suff2;
    int ret = 0;
    unsigned int sizes[2];
    bool generic, tra;
    unsigned int wvlen;     // length of vectors to copy with
    unsigned int uplen;     // length of vectors to update result with
    size_t pitch;
    char LG;
    DataType dtype = gset->kextra->dtype;
    unsigned int vecLen;
    bool isInlined = (flags & UPRES_INLINE);
    UpresVarNames uvars;

    vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? gset->kextra->vecLenC :
                                                   gset->kextra->vecLen;
    sizes[0] = (unsigned int)gset->subdims[1].y;
    sizes[1] = (unsigned int)gset->subdims[1].x;

    if (isComplexType(dtype)) {
        vecLen = 1;
    }

    if ((flags & UPRES_WITH_BETA) && (op != UPRES_SUM)) {
        return -EINVAL;
    }

    tra = ((flags & UPRES_COLUMN_MAJOR) != 0);
    generic = ((flags & UPRES_GENERIC) != 0);
    typeName = dtypeBuiltinType(dtype);
    vfield = dtypeUPtrField(dtype);
    pitch = roundUp(sizes[1], vecLen);

    // select write vectorization
    wvlen = getTmpVecLen(gset, flags, &vecType);
    uplen = (tra ^ gset->tileCY.trans
             || (flags & UPRES_NO_VECTORIZATION)) ? 1 : vecLen;

    suff1 = (generic) ? "Generic" : "";
    suff2 = (flags & UPRES_PRIV_DEST) ? "Rev" : "";
    LG = (flags & UPRES_USE_LDS) ? 'L' : 'G';

    if (!isInlined) {
        const char *outTypeName;
        const char *memPref = (flags & UPRES_USE_LDS) ? "__local" :
                                                           "__global";

        getResultGPRsInfo(dtype, NULL, vecLen, NULL, &outTypeName);

        // define the function
        sprintf(tmp, "void\n"
                     "updateResult%s%s%c(\n"
                     "    %s %s *C,\n"
                     "    %s *c,\n"
                     "    %s alpha,\n"
                     "    uint startRow,\n"
                     "    uint startCol,\n"
                     "    uint ld",
                     suff1, suff2, LG, memPref, typeName,
                     outTypeName, typeName);

        p += strlen(p);
        if (flags & UPRES_WITH_BETA) {
            sprintf(p, ",\n    %s beta", typeName);
            p += strlen(p);
        }
        if (generic) {
            sprintf(p, ",\n    uint nrRows,\n"
                       "    uint nrCols");
        }

        uvars.result = "C";
        uvars.ld = "ld";
        uvars.startRow = "startRow";
        uvars.startCol = "startCol";
        uvars.nrRows = "nrRows";
        uvars.nrCols = "nrCols";

        strcat(p, ")\n");
        kgenDeclareFunction(ctx, tmp);
        kgenBeginFuncBody(ctx);
    }
    else {
        memcpy(&uvars, uvarNames, sizeof(uvars));
    }

    // declare local variables
    sprintf(tmp, "%cPtr uC;\n", LG);
    kgenAddStmt(ctx, tmp);
    if (generic) {
        kgenAddStmt(ctx, "int i, j;\n"
                         "PPtr res;\n");
    }
    else {
        /*
         * temporary pointer to pass correctly over the
         * destination array since destination rows can be
         * not aligned on a vector bound
         */
        if (sizes[1 - tra] % wvlen != 0) {
            sprintf(tmp, "%cPtr tmpC;\n", LG);
            kgenAddStmt(ctx, tmp);
        }
        if (wvlen > uplen) {
            sprintf(tmp, "%s tmp;\n", vecType);
            kgenAddStmt(ctx, tmp);
        }
    }
    if (isComplexType(dtype) && !(flags & UPRES_WITHOUT_ALPHA)) {
        declareComplexMultParts(ctx, "alpha", typeName);
        if (flags & UPRES_WITH_BETA) {
            declareComplexMultParts(ctx, "beta", typeName);
        }

    }
    kgenAddBlankLine(ctx);

    // LD is scaled
    if ( gset->flags & BGF_LD_IN_VECTORS ) {

        vecLen = getVecLen(gset, 0, MATRIX_C);
    }
    else {

        vecLen = 1;
    }

    if (tra) {

        if ( vecLen > 1 ) {

            sprintf(tmp,
                "uC.%s = %s + (%s * %s + %s)/%d;\n",
                vfield,
                uvars.result,
                uvars.startCol,
                uvars.ld,
                uvars.startRow,
                vecLen);
        }
        else {

            sprintf(tmp,
                "uC.%s = %s + %s * %s + %s;\n",
                vfield,
                uvars.result,
                uvars.startCol,
                uvars.ld,
                uvars.startRow);
        }
    }
    else {

        if ( vecLen > 1 ) {

            sprintf(tmp,
                "uC.%s = %s + (%s * %s + %s)/%d;\n",
                vfield,
                uvars.result,
                uvars.startRow,
                uvars.ld,
                uvars.startCol,
                vecLen);

        }
        else {

            sprintf(tmp,
                "uC.%s = %s + %s * %s + %s;\n",
                vfield,
                uvars.result,
                uvars.startRow,
                uvars.ld,
                uvars.startCol);
        }
    }
    kgenAddStmt(ctx, tmp);

    if ((sizes[1 - tra] % wvlen != 0) && !generic) {
        kgenAddStmt(ctx, "tmpC = uC;\n");
    }
    ret = kgenAddBlankLine(ctx);

    if (generic) {
        updateGenericResultGen(ctx, gset, pitch, &uvars, op, flags,
                               uvarNames ? uvarNames->cachedName : NULL);
    }
    else {
        updateOptimResultGen(ctx,
        gset,
        funcID,
        op,
        flags);
    }

    if (!isInlined) {
        ret = kgenEndFuncBody(ctx);
    }

    return (ret) ? -EOVERFLOW : 0;
}

TailFetch
checkForTailFetches(
    BlasFunctionID funcID,
    const SubproblemDim *dim,
    const CLBLASKernExtra *kextra,
    MatrixRole mrole,
    bool distVect,
    bool lowerTails)
{
    TailFetch ret = FETCH_NO_TAILS;
    size_t x;
    KernelExtraFlags tailFlag;
    unsigned int vecLen;
    KernelExtraFlags tailFlagM, tailFlagN, tailFlagK;

    tailFlagM = lowerTails ? KEXTRA_TAILS_M_LOWER : KEXTRA_TAILS_M;
    tailFlagN = lowerTails ? KEXTRA_TAILS_N_LOWER : KEXTRA_TAILS_N;
    tailFlagK = lowerTails ? KEXTRA_TAILS_K_LOWER : KEXTRA_TAILS_K;

    if (mrole == MATRIX_A) {
        x = dim->y;
        tailFlag = tailFlagM;
        vecLen = (distVect) ? kextra->vecLenA : kextra->vecLen;
    }
    else {
        x = dim->x;
        tailFlag = tailFlagN;
        vecLen = (distVect) ? kextra->vecLenB : kextra->vecLen;
    }

    if (isMatrixAccessColMaj(funcID, kextra->flags, mrole)) {
        if ((kextra->flags & tailFlag) && (x != vecLen)) {
            ret |= FETCH_TAIL_COL;
        }
        if (kextra->flags & tailFlagK) {
            ret |= FETCH_TAIL_ROW;
        }
    }
    else if (kextra->flags & tailFlagK) {
        ret |= FETCH_TAIL_COL;
    }

    return ret;
}

bool
isNeedZeroTileTail(
    BlasFunctionID funcID,
    const SubproblemDim *dim,
    const CLBLASKernExtra *kextra,
    MatrixRole mrole,
    bool distVect)
{
    bool trans;
    TailFetch tf;

    trans = isMatrixAccessColMaj(funcID, kextra->flags, mrole);
    tf = checkForTailFetches(funcID, dim, kextra, mrole, distVect, true);

    return (trans && (tf & FETCH_TAIL_ROW)) ||
           (!trans && (tf & FETCH_TAIL_COL));
}

TailStatus
checkGenAdjustTailCoords(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    int *error)
{
    char tmp[1024];
    const SubproblemDim *dim = &gset->subdims[1];
    const KernelVarNames *varNames = &gset->varNames;
    KernelExtraFlags kflags = gset->kextra->flags;
    TailStatus status = 0;
    int err = 0;
    int n = 0;

    if (!isMatrixAccessColMaj(funcID, kflags, MATRIX_A) &&
        (kflags & KEXTRA_TAILS_M_LOWER)) {

        status |= TAIL_A_RAISED;
        sprintf(tmp, "if (%s + %lu > %s) {\n"
                     "    %s -= %lu - %s %% %lu;\n"
                     "}\n",
                varNames->coordA, dim->y, varNames->sizeM,
                varNames->coordA, dim->y, varNames->sizeM,
                dim->y);
        if (ctx != NULL) {
            err = kgenAddStmt(ctx, tmp);
            n++;
        }
    }

    if (!isMatrixAccessColMaj(funcID, kflags, MATRIX_B) &&
        (kflags & KEXTRA_TAILS_N_LOWER) && !err) {

        status |= TAIL_B_RAISED;
        sprintf(tmp, "if (%s + %lu > %s) {\n"
                     "    %s -= %lu - %s %% %lu;\n"
                     "}\n",
                varNames->coordB, dim->x, varNames->sizeN,
                varNames->coordB, dim->x, varNames->sizeN,
                dim->x);
        if (ctx != NULL) {
            err = kgenAddStmt(ctx, tmp);
            n++;
        }
    }

    if (n && !err) {
        err = kgenAddBlankLine(ctx);
    }

    if (error != NULL) {
        *error = err;
    }

    return status;
}

int
checkGenRestoreTailCoords(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    TailStatus status)
{
    char tmp[1024];
    const SubproblemDim *dim = &gset->subdims[1];
    const KernelVarNames *varNames = &gset->varNames;
    int ret = 0;
    int n = 0;

    if (status & TAIL_A_RAISED) {
        sprintf(tmp, "if ((%s + %lu == %s) && (%s %% %lu)) {\n"
                     "    %s += %lu - %s %% %lu;\n"
                     "}\n",
                varNames->coordA, dim->y, varNames->sizeM,
                varNames->sizeM, dim->y, varNames->coordA,
                dim->y, varNames->sizeM, dim->y);
        ret = kgenAddStmt(ctx, tmp);
        n++;
    }

    if ((status & TAIL_B_RAISED) && !ret) {

        sprintf(tmp, "if ((%s + %lu == %s) && (%s %% %lu)) {\n"
                     "    %s += %lu - %s %% %lu;\n"
                     "}\n",
                varNames->coordB, dim->x, varNames->sizeN,
                varNames->sizeN, dim->x, varNames->coordB,
                dim->x, varNames->sizeN, dim->x);
        kgenAddStmt(ctx, tmp);
        n++;
    }

    if (n) {
        ret = kgenAddBlankLine(ctx);
    }

    return (ret) ? -EOVERFLOW : 0;
}

UpdateResultFlags
tailStatusToUpresFlags(TailStatus status)
{
    UpdateResultFlags flags = 0;

    if (status & TAIL_A_RAISED) {
        flags |= UPRES_TAIL_ROW;
    }
    if (status & TAIL_B_RAISED) {
        flags |= UPRES_TAIL_COL;
    }

    return flags;
}

int
declareComplexMultParts(
    struct KgenContext *ctx,
    const char *baseName,
    const char *typeName)
{
    char tmp[1024];
    int r;

    sprintf(tmp, "%s %sR = (%s)(%s.x);\n"
                 "%s %sI = (%s)(-%s.y, %s.y);\n",
            typeName, baseName, typeName, baseName,
            typeName, baseName, typeName, baseName, baseName);
    r = kgenAddStmt(ctx, tmp);

    return (r) ? -EOVERFLOW : 0;
}

void
sprintfFastScalarMad(
    Kstring *expr,
    const Kstring *first,
    const Kstring *second,
    unsigned int scale,
    const Kstring *third)
{
    unsigned int u1 = 0, u2 = 0, u3 = 0;
    bool isNum1, isNum2, isNum3;
    int shift;
    bool done = false;
    const char *thirdStr;
    const char *suff3;

    // clear up what are these arguments
    if (isKstringEmpty(first)) {
        isNum1 = true;
    }
    else {
        isNum1 = !stringToInt(first->buf, &u1);
    }

    if (isKstringEmpty(second)) {
        isNum2 = true;
    }
    else {
        isNum2 = !stringToInt(second->buf, &u2);
    }

    if (!scale) {
        scale = 1;
    }

    if ((third == NULL) || isKstringEmpty(third)) {
        thirdStr = "0";
        isNum3 = true;
    }
    else {
        thirdStr = third->buf;
        isNum3 = !stringToInt(thirdStr, &u3);
    }
    suff3 = (isNum3) ? "u" : "";

    // singular case at which only the third component can contribute
    if ( (isNum1 && (u1 == 0)) ||
         (isNum2 && (u2 /scale == 0))) {

        kstrcpy(expr, thirdStr);
        return;
    }

    if (isNum1 && isNum2) {
        if (isNum3) {
            ksprintf(expr, "%u", u1 * u2 / scale + u3);
        }
        else {
            ksprintf(expr, "%u + %s", u1 * u2 / scale, thirdStr);
        }
        done = true;
    }
    else if (isNum1) {
        /*
         * If the third argument is not used, then try to build the expression
         * using only shifts if 'scale' and the 'second argument' are both of
         * power of 2. Otherwise use mad24.
         */
        if (isRoundedPow2(u1) && isRoundedPow2(scale)) {
            shift = findHighestSetBit(scale) - findHighestSetBit(u1);
            if (isNum3 && (u3 == 0)) {
                if (shift < 0) {
                    ksprintf(expr, "(%s << %d)", second->buf, -shift);
                }
                else if (shift > 0) {
                    ksprintf(expr, "(%s >> %d)", second->buf, shift);
                }
                else {
                    kstrcpy(expr, second->buf);
                }
            }
            else if (shift > 0) {
                ksprintf(expr, "(%s >> %d) + %s",
                         second->buf, shift, thirdStr);
            }
            else if (shift == 0) {
                ksprintf(expr, "%s + %s", second->buf, thirdStr);
            }
            else {
                ksprintf(expr, "mad24(%uu, %s, %s%s)",
                         1u << -shift, second->buf, thirdStr, suff3);
            }
            done = true;
        }
    }

    if (!done) {
        /*
         * Append unsiged suffixes to avoid cases at which one
         * operand is signed and the other is unsigned. Typically,
         * OpenCL compilers are strict and reject such expressions.
         */
        if (isNum2) {
            if (u2 / scale == 1) {
                if (isNum3 && (u3 == 0)) {
                    kstrcpy(expr, first->buf);
                }
                else {
                    ksprintf(expr, "%s + %s", first->buf, thirdStr);
                }
            }
            else {
                ksprintf(expr, "mad24(%s, %uu, %s%s)",
                         first->buf, u2 / scale, thirdStr, suff3);
            }
        }
        else {
            const char *suff1 = (isNum1) ? "u" : "";
            Kstring tmp;
            const char *p = NULL;

            if (scale == 1) {
                p = second->buf;
            }
            else {
                p = tmp.buf;
                if (isRoundedPow2(scale)) {
                    shift = findHighestSetBit(scale);
                    ksprintf(&tmp, "(%s >> %d)", second->buf, shift);
                }
                else {
                    ksprintf(&tmp, "%s / %d", second->buf, scale);
                }
            }

            ksprintf(expr, "mad24(%s%s, %s, %s%s)",
                     first->buf, suff1, p, thirdStr, suff3);
        }
    }
}
