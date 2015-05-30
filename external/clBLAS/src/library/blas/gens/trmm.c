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
 * Cached global buffers based trmm generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>

#include "init.h"
#include "blas_kgen.h"
#include "blas_subgroup.h"
#include "trxm_common.h"

typedef struct {
    size_t staggered;
} MAY_ALIAS extraData_t;

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

static SolverFlags
solverFlags(void);

static void fixupArgs( void *args,
    SubproblemDim *subdims,
    void *extra );

static int
blockGetPerf( unsigned int kflags,
    const void *args );

static int subgGetPerf( unsigned int kflags,
    const void *args );

static void subgCalcThreads( size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra );

static int trmmGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs);

static int trmmSubgGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs );

static bool subgCheckCalcDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check );

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs);

static bool
blockCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static SolverOps blockSops = {
    generator,
    assignKargs,
    isFitToLDS,
    blockGetPerf,
    NULL,
    NULL,
    NULL,
    solverFlags,
    fixupArgs,
    trmmGetDefaultDecomp,   // getDefaultDecomp
    blockCheckCalcDecomp,
    NULL,
    NULL};

// Solver options for subgroup pattern
static SolverOps subgSops = {
    generator,
    assignKargs,
    NULL,
    subgGetPerf,
    NULL,
    subgCalcThreads,
    NULL,
    solverFlags,
    fixupArgs,
    trmmSubgGetDefaultDecomp,
    subgCheckCalcDecomp,
    NULL,
    NULL};

//-----------------------------------------------------------------------------

static void
initKernelVarNames(KernelVarNames *kvars)
{
    kvars->A = "(Ag)";
    kvars->B = "(Bg)";
    kvars->C = "C";
    kvars->coordA = "coord.y";
    kvars->coordB = "coord.x";
    kvars->k = "coord.z";
    kvars->sizeK = "M";
    kvars->sizeM = "M";
    kvars->sizeN = "N";
    kvars->lda = "lda";
    kvars->ldb = "ldb";
    kvars->ldc = "ldb";
    kvars->alpha = "alpha";
}

//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------

static void
genStartPosK(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    KernelExtraFlags kflags,
    bool subgMode)
{
    char tmp[1024];
    if (isMatrixUpper(kflags)) {
        // K loop - from diagonal till M
        if (subgMode) {
            sprintf(tmp, "uint kBegin = currM;\n");
        }
        else {
            if (!(kflags & KEXTRA_TAILS_M)) {
                sprintf(tmp, "uint kBegin = currM;\n");
            }
            else {
                sprintf(tmp, "uint kBegin = currM / %lu * %lu;\n",
                        dim->bwidth, dim->bwidth);
            }
        }
    }
    else {
        // K loop - from 0 till diagonal
        sprintf(tmp, "uint kBegin = 0;\n");
    }

    kgenAddStmt(ctx, tmp);
}

//-----------------------------------------------------------------------------

static void
resetFetchNumA(TileMulOpts *mulOpts)
{
    TilePostFetchPrivate *pfPriv;
    pfPriv = (TilePostFetchPrivate *) mulOpts->postFetchPriv;

    pfPriv[0].fetchNumA = 0;
    pfPriv[1].fetchNumA = 0;
}

//-----------------------------------------------------------------------------

static int
genSubgLoopsK(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    TileMulOpts *mulOpts,
    SubgVarNames* pSubgVNames,
    size_t staggered)
{
    char tmp[1024];
    KernelExtraFlags kflags = gset->kextra->flags;
    const size_t y0 = gset->subdims[0].y;
    const size_t bw1 = gset->subdims[1].bwidth;
    const size_t bw0 = gset->subdims[0].bwidth;

    // bw, that will be used for diagonal block evaluation
    size_t diagBw1 = getVecLen( gset, CLBLAS_TRMM, MATRIX_A );

    // saving dimensions of tile A, that will be changed for
    // diagonal block
    size_t sDimA = gset->tileA.trans ?
        gset->tileA.nrRows:
        gset->tileA.nrCols;

    size_t sDimB = gset->tileBX.trans ?
        gset->tileBX.nrRows:
        gset->tileBX.nrCols;

    const CLBLASKernExtra* psKExtra = gset->kextra;
    CLBLASKernExtra diagKExtra;
    TilePostFetchPrivate postFPriv;
    int ret = 0;

    kgenPrintf( ctx, "uint k0;\n" );
    kgenPrintf( ctx, "uint kMax;\n" );

    // upper triangle case
    if (isMatrixUpper(kflags)) {

        // diagonal part ------------------------------------------------------

        // adjust tile and kextra settings for
        // processing diagonal block
        gset->subdims[1].bwidth = diagBw1;
        if ( gset->tileA.trans ) {
            gset->tileA.nrRows = diagBw1;
        }
        else {
            gset->tileA.nrCols = diagBw1;
        }
        if ( gset->tileBX.trans ) {
            gset->tileBX.nrRows = diagBw1;
        }
        else {
            gset->tileBX.nrCols = diagBw1;
        }
        memcpy( &diagKExtra,gset->kextra,sizeof(CLBLASKernExtra) );
        diagKExtra.vecLenA = diagBw1 < psKExtra->vecLenA?
            diagBw1:
            psKExtra->vecLenA;
        diagKExtra.vecLenB = diagBw1 < psKExtra->vecLenB?
            diagBw1:
            psKExtra->vecLenB;
        gset->kextra = (const CLBLASKernExtra*)&diagKExtra;

        // Process the triangle block by the 0 item
        // of each subgroup
        kgenPrintf( ctx, "// k-coordinate of the end of diagonal block\n" );
        kgenPrintf( ctx, "// calculated to be aligned to bw1\n");
        kgenPrintf( ctx,
            "kMax = kBegin + %lu + (%lu - %lu%%(kBegin+%lu));\n",
            y0,
            bw1,
            bw1,
            y0);

        sprintf( tmp, "if( %s.x == 0 )", pSubgVNames->itemId );
        kgenBeginBranch( ctx, tmp );

        sprintf( tmp,
            "for( k0=kBegin; (k0<kMax)&&(k0<M); k0+=%lu )",
            diagBw1 );
        kgenBeginBranch( ctx, tmp );

        kgenPrintf( ctx, "%s=k0;\n", gset->varNames.k );
        mulOpts->postFetch = genTrxmPostFetchZero;
        ret = tileMulGen( ctx, gset, mulOpts );
        if( 0 != ret ){
            return ret;
        }

        kgenEndBranch(ctx, NULL);// for()
        kgenEndBranch(ctx, NULL);// if( itemId.x == 0 )

        // Restore tile and kextra settings to the
        // original parameters
        gset->subdims[1].bwidth = bw1;
        if ( gset->tileA.trans ) {
            gset->tileA.nrRows = sDimA;
        }
        else {
            gset->tileA.nrCols = sDimA;
        }
        if ( gset->tileBX.trans ) {
            gset->tileBX.nrRows = sDimB;
        }
        else {
            gset->tileBX.nrCols = sDimB;
        }
        gset->kextra = psKExtra;

        // rectangle part -----------------------------------------------------
        kgenAddBlankLine( ctx );
        kgenPrintf( ctx, "k0 = kMax;\n" );
        if ( kflags & KEXTRA_TAILS_K_LOWER ) {

            kgenPrintf( ctx, "uint alignedK = M-(M%%%lu);\n", bw1 );
        }
        // strided access
        sprintf(tmp,
            "for ( k0 = k0+%s.x*%lu; k0 < %s; k0 += %lu )",
            pSubgVNames->itemId,
            bw1,
            ( kflags & KEXTRA_TAILS_K_LOWER )? "alignedK" : "M",
            bw0);

        kgenBeginBranch(ctx, tmp);
        // TODO: make staggered access operational with lower-K tails
        /*kgenPrintf( ctx,
            "%s = (kBegin+%d) + ( m0*64*(gid%%2) + k0 )%%(M-(kBegin+%d));\n",
            gset->varNames.k,
            diagW,
            diagW); */
        kgenPrintf( ctx, "%s = k0;\n", gset->varNames.k );

        mulOpts->postFetch = NULL;
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        // rectangle tail part ------------------------------------------------

        if ( kflags & KEXTRA_TAILS_K_LOWER ) {

            kgenAddBlankLine( ctx );
            kgenPrintf( ctx,
                "// lower K tail is handled by item 0 of each subgroup\n");

            sprintf(tmp, "if( (%s.x == 0)&&(kMax < M) )", pSubgVNames->itemId);
            kgenBeginBranch( ctx, tmp );

            kgenPrintf( ctx, "%s = alignedK;\n", gset->varNames.k );
            postFPriv.fetchNumA = 0;
            postFPriv.gset = gset;
            mulOpts->postFetch = defaultTilePostFetch;
            mulOpts->postFetchPriv = &postFPriv;

            ret = tileMulGen( ctx, gset, mulOpts );
            if ( ret != 0 ) {
                return ret;
            }
            kgenEndBranch( ctx, NULL );
        }
    }
    // lower triangle case
    else {

        // rectangle part -----------------------------------------------------

        kgenPrintf( ctx, "kMax = currM - currM%%%lu;\n", bw1 );
        // strided access, staggered access
        sprintf( tmp,
            "for( k0 = 0; k0 < kMax; k0 += %lu )",
            bw0 );
        kgenBeginBranch( ctx, tmp );

        kgenPrintf( ctx, "%s=(k0+%s.x*%d+%d*gid)%%kMax;\n",
            gset->varNames.k,
            pSubgVNames->itemId,
            bw1,
            staggered/bw1*bw1 );

        mulOpts->postFetch = NULL;
        // part without diagonal elements post fetch zeroing
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch( ctx, NULL );

        // diagonal part ------------------------------------------------------

        // adjust tile and kextra settings for
        // processing diagonal block
        gset->subdims[1].bwidth = diagBw1;
        if ( gset->tileA.trans ) {
            gset->tileA.nrRows = diagBw1;
        }
        else {
            gset->tileA.nrCols = diagBw1;
        }
        if ( gset->tileBX.trans ) {
            gset->tileBX.nrRows = diagBw1;
        }
        else {
            gset->tileBX.nrCols = diagBw1;
        }
        psKExtra = gset->kextra;
        memcpy( &diagKExtra,gset->kextra,sizeof(CLBLASKernExtra) );
        diagKExtra.vecLenA = diagBw1 < psKExtra->vecLenA?
            diagBw1:
            psKExtra->vecLenA;
        diagKExtra.vecLenB = diagBw1 < psKExtra->vecLenB?
            diagBw1:
            psKExtra->vecLenB;
        gset->kextra = (const CLBLASKernExtra*)&diagKExtra;

        // process the triangle block by the 0 item
        // of each subgroup
        sprintf( tmp, "if( %s.x == 0 )", pSubgVNames->itemId );
        kgenBeginBranch( ctx, tmp );

        sprintf( tmp,
            "for( k0 = kMax; (k0 < currM+%lu)&&(k0 < M); k0 += %lu )",
            y0,
            diagBw1 );
        kgenBeginBranch( ctx, tmp );

        kgenPrintf( ctx, "%s=k0;\n", gset->varNames.k );
        mulOpts->postFetch = genTrxmPostFetchZero;
        resetFetchNumA(mulOpts);
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch( ctx, NULL );// for()
        kgenEndBranch( ctx, NULL );// if( itemId.x == 0 )

        // Restore tile and kextra settings to the
        // original parameters
        gset->subdims[1].bwidth = bw1;
        if ( gset->tileA.trans ) {
            gset->tileA.nrRows = sDimA;
        }
        else {
            gset->tileA.nrCols = sDimA;
        }
        if ( gset->tileBX.trans ) {
            gset->tileBX.nrRows = sDimB;
        }
        else {
            gset->tileBX.nrCols = sDimB;
        }
        gset->kextra = psKExtra;

    }

    return 0;
}

//-----------------------------------------------------------------------------

static int
genLoopsK(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    TileMulOpts *mulOpts,
    char *tmp)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    const size_t y0 = gset->subdims[0].y;
    const size_t bwidth = gset->subdims[1].bwidth;
    int ret;
    bool isRel = false;
    const char *inTypeNameA, *inPtrNameA, *inTypeNameB, *inPtrNameB;

    getVectorTypeName(gset->kextra->dtype, gset->kextra->vecLenA, &inTypeNameA, &inPtrNameA);
    getVectorTypeName(gset->kextra->dtype, gset->kextra->vecLenB, &inTypeNameB, &inPtrNameB);

    sprintf(tmp, "uint k0;\n");
    kgenAddStmt(ctx, tmp);

    if (!(kflags & (KEXTRA_TAILS_M_LOWER | KEXTRA_TAILS_N_LOWER |
                    KEXTRA_TAILS_K_LOWER))) {

        FetchAddrMode addrMode = FETCH_ADDR_A_RELATIVE | FETCH_ADDR_B_RELATIVE |
                                 FETCH_ADDR_K_RELATIVE;

        isRel = true;

        mulOpts->fctx = createFetchContext();
        if (mulOpts->fctx == NULL) {
            return -ENOMEM;
        }
        setFetchAddrMode(mulOpts->fctx, addrMode);

        gset->varNames.A = "pA";
        gset->varNames.B = "pB";
    }
    else {
        gset->flags |= BGF_UPTRS;
        kgenPrintf(ctx, "GPtr Ag, Bg;\n"
                        "\n"
                        "Ag.%s = A;\n"
                        "Bg.%s = B;\n\n",
                   inPtrNameA, inPtrNameB);
    }

    if (isMatrixUpper(kflags)) {
        if (isRel) {
            switch ((((gset->kextra->flags & KEXTRA_TRANS_A) != 0)<<1) |
                    (((gset->kextra->flags & KEXTRA_UPPER_TRIANG) != 0) ^
                     ((gset->kextra->flags & KEXTRA_COLUMN_MAJOR) != 0))
                   ) {
            case 0:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.z, lda, coord.y)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.x, ldb, coord.z)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 1:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.y, lda, coord.z)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.z, ldb, coord.x)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 2:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.z, lda, coord.y)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.z, ldb, coord.x)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 3:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.y, lda, coord.z)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.x, ldb, coord.z)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            }
        }

        sprintf(tmp,
            "for (k0 = kBegin; "
                "(k0 <= (kBegin + %luu))&&(k0 < M); "
                "k0 += %lu)",
            y0,
            bwidth);
        kgenBeginBranch(ctx, tmp);

        kgenPrintf( ctx,
            "coord.z = k0;\n");

        mulOpts->postFetch = genTrxmPostFetchZero;
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        //main triangle part
        sprintf(tmp,
            "for (; k0 <= max(0, (int)M - %lu); k0 += %lu)",
            y0,
            gset->subdims[1].bwidth);

        kgenBeginBranch(ctx, tmp);

        mulOpts->postFetch = NULL;
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        // matrix side part
        // should be calculated by item0 of each subgroup
        sprintf(tmp, "for (; k0 < M; k0 += %lu)", bwidth);
        kgenBeginBranch(ctx, tmp);

        kgenPrintf( ctx,
            "coord.z = k0;\n");

        resetFetchNumA(mulOpts);
        mulOpts->postFetch = genTrxmPostFetchZero;
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

    }
    else {
        // lower
        size_t diagBlocks; //Number of bw *y blocks that fit in y*y square

        if (isRel) {
            switch ((((gset->kextra->flags & KEXTRA_TRANS_A) != 0)<<1) |
                    (((gset->kextra->flags & KEXTRA_UPPER_TRIANG) != 0) ^
                     ((gset->kextra->flags & KEXTRA_COLUMN_MAJOR) != 0))
                   ) {
            case 0:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.y, lda, coord.z)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.z, ldb, coord.x)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 1:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.z, lda, coord.y)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.x, ldb, coord.z)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 2:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.y, lda, coord.z)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.x, ldb, coord.z)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            case 3:
                kgenPrintf(ctx,
                    "__global %s *pA = (__global %s *)&A[mad24(coord.z, lda, coord.y)];\n"
                    "__global %s *pB = (__global %s *)&B[mad24(coord.z, ldb, coord.x)];\n",
                    inTypeNameA, inTypeNameA,inTypeNameB, inTypeNameB);
                break;
            }
        }

        diagBlocks = divRoundUp(y0, bwidth);
        sprintf(tmp, "uint iterK = min(currM + %luu, M);\n", y0);
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "iterK = (iterK + %lu) / %lu;\n", bwidth - 1, bwidth);
        kgenAddStmt(ctx, tmp);

        // main triangle part
        sprintf(tmp, "for (k0 = 0; k0 < max(0, (int)iterK - %lu); k0++)",
                diagBlocks);
        kgenBeginBranch(ctx, tmp);
        mulOpts->postFetch = NULL;
        // part without diagonal elements post fetch zeroing
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        // diagonal part
        sprintf(tmp, "for (; k0 < iterK; k0++)");
        kgenBeginBranch(ctx, tmp);

        kgenPrintf( ctx,
            "coord.z = k0 * %lu;\n",
            bwidth);

        // diagonal blocks part
        mulOpts->postFetch = genTrxmPostFetchZero;
        resetFetchNumA(mulOpts);
        ret = tileMulGen(ctx, gset, mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);
    }

    if (isRel) {
        destroyFetchContext(mulOpts->fctx);
        mulOpts->fctx = NULL;
    }

    return 0;
}

//-----------------------------------------------------------------------------

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    char tmp[4096];
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    KernelExtraFlags kflags = kextra->flags;
    DataType dtype = kextra->dtype;
    bool doubleBased = isDoubleBasedType(dtype);
    size_t staggered = ((extraData_t*)&kextra->solverPriv)->staggered;
    int ret;
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    int tra = isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_A);
    int trb = isMatrixAccessColMaj(CLBLAS_TRMM, kflags, MATRIX_B);
    unsigned int l1Pans;
    TilePostFetchPrivate pfPriv[2];
    UpdateResultFlags upResFlags;
    TailStatus tailStatus;
    bool subgMode = false;
    SubgVarNames subgVNames;

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // mismatching subdims define case with subgroup decomposition
    subgMode = ( subdims[0].bwidth != subdims[1].bwidth );

    memset(&gset, 0, sizeof(gset));
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.flags = BGF_DISTINCT_VECLEN;

    gset.flags |= BGF_WHOLE_A;

    /*FIXME: This used to be a workaround for compilation issues with dtrmm on
     * cpu. Normally BGF_WHOLE_A should be enabled always. But for now,
     * there are wrong results for non-aligned cases on CPU and there is
     * no workaround yet.
    if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N | KEXTRA_TAILS_K)) {
        gset.flags &= ~BGF_WHOLE_A;
    }*/
    gset.kextra = kextra;
    gset.pgran = pgran;
    //avoid [0].bw loop
    //gset.subdims[0].bwidth = gset.subdims[1].bwidth;

    memset(pfPriv, 0, sizeof(pfPriv));
    pfPriv[0].funcID = CLBLAS_TRMM;
    pfPriv[0].gset = &gset;
    if ((gset.flags & BGF_WHOLE_A) != 0) {
        pfPriv[0].wholeA = 1;
    }

    // at first, generate needed declarations
    kgenDeclareUptrs(ctx, doubleBased);

    // For inner callback, because both callbacks use own fetchNumA
    memcpy(&pfPriv[1], &pfPriv[0], sizeof(pfPriv[0]));

    // if both matrices are accessed row-major - using subgroup pattern
    if ( subgMode ) {

        declareTrxmKernel(ctx,
            dtype,
            pgran,
            kflags,
            CLBLAS_TRMM,
            "Subgroup",
            true,
            true);
        gset.flags |= BGF_UPTRS;
    }
    else {

        declareTrxmKernel(ctx,
            dtype,
            pgran,
            kflags,
            CLBLAS_TRMM,
            "Block",
            true,
            true);

    }
    kgenBeginFuncBody(ctx);

    initDefaultTiles(&gset, CLBLAS_TRMM, 0, PRIV_STORAGE_VARIABLE_SET);
    declareTileStorages(ctx, &gset);

    kgenAddStmt(ctx,
                "uint currM, currN;\n"
                "uint4 coord = 0; /* contains coordB, coordA, k */\n");

    kgenDeclareLocalID(ctx, "lid", pgran);
    kgenDeclareGroupID(ctx, "gid", pgran);

    if ( subgMode ) {

        gset.varNames.LDS = "scratch";

        // declaring variables used by subgroup mode
        subgVNames.itemId = "itemId";
        subgVNames.subgCoord = "subgCoord";

        kgenAddBlankLine( ctx );
        kgenAddBlankLine(ctx);

        kgenPrintf(ctx, "int2 %s;\n", subgVNames.itemId );
        kgenPrintf(ctx, "int2 %s;\n", subgVNames.subgCoord);

        // item ID
        kgenPrintf( ctx,
            "%s.x = get_local_id(0)%%%d;\n",
            subgVNames.itemId,
            subdims[0].bwidth/subdims[1].bwidth);

        // subgroup ID
        kgenPrintf( ctx,
            "%s.y = get_local_id(0)/%d;\n",
            subgVNames.itemId,
            subdims[0].bwidth/subdims[1].bwidth);

        // subgroup coordX
        kgenPrintf( ctx,
            "%s.x = %s.y/%d;\n",
            subgVNames.subgCoord,
            subgVNames.itemId,
            subdims[0].y/subdims[1].y );

        // subgroup coordY
        kgenPrintf( ctx,
            "%s.y = %s.y%%%d;\n",
            subgVNames.subgCoord,
            subgVNames.itemId,
            subdims[0].y/subdims[1].y );
    }

    kgenAddBlankLine(ctx);

    sprintf(tmp, "currN = gid * %lu;\n", subdims->x);
    kgenAddStmt(ctx, tmp);
    genInitCurrM(ctx, subdims, kflags);

    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    genTrxmBMatrShift(ctx, kflags, true);

    if ( subgMode ) {
        kgenAddStmt(ctx,
            "GPtr Ag = {A};\n"
            "GPtr Bg = {B};\n");
    }

    l1Pans = (unsigned int)subdims[0].x / (unsigned int)subdims[1].x;

    memset(&mulOpts, 0, sizeof(mulOpts));
    mulOpts.core = ((kflags & KEXTRA_ENABLE_MAD) != 0)
            ? TILEMUL_MAD
            : TILEMUL_MULADD;
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;
    mulOpts.postFetch = NULL;
    mulOpts.postFetchPriv = &pfPriv;
    mulOpts.flags = TILEMUL_NO_FLAGS;
    mulOpts.flags |= TILEMUL_EXTERN_RDECL;

    if ( subgMode ) {

        mulOpts.flags |= TILEMUL_NOT_INC_K;
        mulOpts.flags |= TILEMUL_BW_STRIDE;
    }

    if (kflags & KEXTRA_TAILS_M_LOWER) {
        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_A;
    }
    if (kflags & KEXTRA_TAILS_N_LOWER) {
        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_B;
    }
    if (kflags & KEXTRA_TAILS_K_LOWER) {
        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_K;
        mulOpts.flags |= TILEMUL_WRAP_AROUND_TAIL;
    }

    if (tra) {
        mulOpts.flags |= TILEMUL_TRA;
    }
    if (!trb) {
        mulOpts.flags |= TILEMUL_TRB;
    }
    if (isMatrixConj(kflags, MATRIX_A)) {
        mulOpts.flags |= TILEMUL_CONJA;
    }
    if (isMatrixConj(kflags, MATRIX_B)) {
        mulOpts.flags |= TILEMUL_CONJB;
    }

    initKernelVarNames(&gset.varNames);

    if ( subgMode ) {

        kgenPrintf( ctx,
            "coord.x = currN + %s.x*%d;\n",
            subgVNames.subgCoord,
            subdims[1].x );
    }
    else {

        sprintf(tmp, "coord.x = currN + lid %% %u * %lu;\n", l1Pans, subdims[1].x);
        kgenAddStmt(ctx, tmp);
    }

    // loop over M
    sprintf(tmp, "for (uint m0 = 0; m0 < M; m0 += %lu)", subdims[0].y);
    kgenBeginBranch(ctx, tmp);

    genStartPosK( ctx, subdims, kflags, subgMode );

    sprintf(tmp, "coord.z = kBegin;\n");
    kgenAddStmt(ctx, tmp);

    if ( subgMode ) {

        kgenPrintf(ctx,
            "coord.y = currM + %s.y*%d;\n",
            subgVNames.subgCoord,
            subdims[1].y);
    }
    else {

        sprintf( tmp,
            "coord.y = currM + lid / %u * %lu;\n",
            l1Pans,
            subdims[1].y );
        kgenAddStmt(ctx, tmp);
    }

    genZeroTile(ctx, &gset.tileCY);

    checkGenBeginHitMatrixBlock(ctx, kflags);
    tailStatus = checkGenAdjustTailCoords(ctx, CLBLAS_TRMM, &gset, NULL);

    // loops along 'K'
    if ( subgMode ) {
        ret = genSubgLoopsK( ctx, &gset, &mulOpts, &subgVNames, staggered);
    }
    else {
        ret = genLoopsK( ctx, &gset, &mulOpts, tmp );
    }

    if (ret != 0) {
        printf("%s", buf);
        return ret;
    }

    checkGenEndHitMatrixBlock(ctx, kflags);
    kgenAddBarrier(ctx, CLK_GLOBAL_MEM_FENCE);

    // store results
    // for result update - x coordinate is in elements, not in vectors

    checkGenRestoreTailCoords(ctx, &gset, tailStatus);
    upResFlags = kextraToUpresFlags(CLBLAS_TRMM, kflags);
    upResFlags |= tailStatusToUpresFlags(tailStatus);
    upResFlags |= UPRES_INDEXING_WITH_CONSTANTS;
    upResFlags |= UPRES_TRIANG_WRITE_C;
    upResFlags |= UPRES_EXCEED_PROBLEM_CONDITION;

    if ( subgMode ) {

        mergeUpdateResult( ctx,
            CLBLAS_TRMM,
            &gset,
            &subgVNames,
            upResFlags,
            genResultUpdateWithFlags );
    }
    else {

        //checkGenBeginHitMatrixBlock(ctx, kflags);
        genResultUpdateWithFlags( ctx,
            CLBLAS_TRMM,
            &gset,
            upResFlags,
            NULL,
            NULL,
            NULL );
        //checkGenEndHitMatrixBlock(ctx, kflags);
    }

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "currM += %lu;\n", subdims[0].y);
    }
    else {
        sprintf(tmp, "currM -= %lu;\n", subdims[0].y);
    }
    kgenAddStmt(ctx, tmp);

    kgenEndBranch(ctx, NULL);

    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

//-----------------------------------------------------------------------------

static void
assignKargs(KernelArg *args, const void *params, const void *extra)
{
    const CLBlasKargs *blasArgs = (const CLBlasKargs*)params;
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    int idx;

    (void)extra;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
    initMemobjKarg(&args[3], blasArgs->A, NULL, 0, 0);
    initSizeKarg(&args[4], blasArgs->lda.matrix);
    initMemobjKarg(&args[5], blasArgs->B, NULL, 0, 0);
    initMemobjKarg(&args[6], blasArgs->B, NULL, 0, 0); //C in kernel
    initSizeKarg(&args[7], blasArgs->ldb.matrix);
    idx = 8;
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx], blasArgs->offBX);
    }
}

//-----------------------------------------------------------------------------

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
    /* LDS is not used here so we surely fit to LDS */
    return true;
}

//-----------------------------------------------------------------------------

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D);
}

//-----------------------------------------------------------------------------

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    CLBlasKargs *kargs = (CLBlasKargs*)args;
    extraData_t *extraData = (extraData_t*)&((CLBLASKernExtra*)extra)->solverPriv;

    const size_t nChans = 8; // !!!DEVICE DEPENDED!!!
    const size_t wideChans = 64; // !!!DEVICE DEPENDED!!!
    const size_t sizeType[] = {1,2,2,4};

    size_t sizeBlock = wideChans * nChans / sizeType[kargs->dtype];
    size_t off = kargs->K % sizeBlock;
    if (off == 0) { ///!= or == ???
        extraData->staggered = roundUp(subdims[1].bwidth * sizeType[kargs->dtype]
                                    , wideChans / sizeType[kargs->dtype]);
    }
    else {
        extraData->staggered = 0;
    }
    extraData->staggered = 64 / sizeType[kargs->dtype]; //fixed, not calculated

    fixupTrxmKargs((CLBlasKargs*)args);
}

//-----------------------------------------------------------------------------

static bool
blockCheckCalcDecomp(
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

//-----------------------------------------------------------------------------

void
initTrmmCachedBlockPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based trmm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &blockSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_L2;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

void
initTrmmCachedSubgroupPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based subgroup trmm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &subgSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_L2;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

static int
blockGetPerf( unsigned int kflags,
    const void *args )
{
    DUMMY_ARG_USAGE(args);

    if( !isMatrixAccessColMaj( CLBLAS_TRMM, kflags, MATRIX_A ) &&
        !isMatrixAccessColMaj( CLBLAS_TRMM, kflags, MATRIX_B ) ){

        return PPERF_AVERAGE;
    }

    return PPERF_GOOD;
}

//-----------------------------------------------------------------------------

static int
subgGetPerf( unsigned int kflags,
    const void *args )
{
    DUMMY_ARG_USAGE(args);

    if( !isMatrixAccessColMaj( CLBLAS_TRMM, kflags, MATRIX_A ) &&
        !isMatrixAccessColMaj( CLBLAS_TRMM, kflags, MATRIX_B ) ){

        return PPERF_GOOD;
    }

    return PPERF_NOT_SUPPORTED;
}

//-----------------------------------------------------------------------------

static void
subgCalcThreads( size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra )
{
    CLBLASKernExtra* pKExtra;
    CLBlasKargs    *pArgs;

    //EINVAL
    if ( NULL == subdims ||
        NULL == pgran ||
        NULL == args ||
        NULL == extra ) {

        return;
    }
    pKExtra = (CLBLASKernExtra*)extra;
    pArgs = (CLBlasKargs*)args;

    // if side is right the dimensions outside kernel are swapped
    // A is NxN and B is MxN
    // inside kernel A is still MxM
    if ( pKExtra->flags & KEXTRA_SIDE_RIGHT ) {

        threads[0] = ( (pArgs->M/subdims[0].x) * 64 );
        // B tail group
        if ( pArgs->M%subdims[0].x ) {
            threads[0] += 64;//pgran->wgSize[0];
        }
    }
    else {

        threads[0] = ( (pArgs->N/subdims[0].x) * 64 );
        // B tail group
        if ( pArgs->N%subdims[0].x ) {
            threads[0] += 64;//pgran->wgSize[0];
        }
    }
    threads[1] = 0;

}

//-----------------------------------------------------------------------------

static int trmmGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs)
{
    (void*)subdimsNum;

    if ( NULL == pArgs ) {
        return -EINVAL;
    }

    subdims[1].bwidth = 2;
    subdims[1].x = subdims[1].itemX = 8;
    subdims[1].y = subdims[1].itemY = 8;

    subdims[0].bwidth = 2;
    subdims[0].x = subdims[0].itemX = 32;
    subdims[0].y = 128;
    subdims[0].itemY = -1;

    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;
    pgran->wgSize[1] = 1;

    return 0;
}

static int trmmSubgGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs)
{
    int itemsPerSubg = 4;
    int subgA = 8;
    int subgB = 2;

    int bw1 = 8;
    int x1 = 4;
    int y1 = 4;
    CLBlasKargs *kargs;

    DUMMY_ARG_USAGE(subdimsNum);

    if ( NULL == pArgs ) {
        return -EINVAL;
    }

    kargs = (CLBlasKargs *)pArgs;

    if( isComplexType(kargs->dtype) ){
        bw1 /= 2;
    }
    if( isDoubleBasedType(kargs->dtype) ){
        bw1 /= 2;
    }

    subdims[1].bwidth = bw1;
    subdims[1].x = subdims[1].itemX = x1;
    subdims[1].y = subdims[1].itemY = y1;

    subdims[0].bwidth = bw1 * itemsPerSubg;
    subdims[0].itemX = x1 * subgB;
    subdims[0].x = x1*subgB;

    subdims[0].itemY = y1*subgA;
    subdims[0].y = y1*subgA;

    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;
    pgran->wgSize[1] = 1;

    return 0;
}

//-----------------------------------------------------------------------------
// TODO: reimplement via new validation API
static bool
subgCheckCalcDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check )
{
    unsigned int subgA = 0;
    unsigned int subgB = 0;
    unsigned int regUse = 0;
    unsigned int itemsPerSubg = 0;

    DUMMY_ARG_USAGE(subdimsNum);

    if( 0 == subdims[0].x ||
        0 == subdims[0].y ||
        0 == subdims[0].bwidth ||
        0 == subdims[1].x ||
        0 == subdims[1].y ||
        0 == subdims[1].bwidth ){

        return false;
    }

    subgA = subdims[0].y/subdims[1].y;
    subgB = subdims[0].x/subdims[1].x;
    itemsPerSubg = subdims[0].bwidth/subdims[1].bwidth;

    if( itemsPerSubg < 4 ){
        return false;
    }

    if( subdims[1].y < 4 ||
        subdims[1].x < 4 ||
        subdims[1].bwidth < 4 ){
        return false;
    }

    if( subdims[1].x != subdims[1].itemX ||
        subdims[1].y != subdims[1].itemY ){

        return false;
    }

    // the group block must consist of integer number of subgroup blocks
    if( subdims[0].x % subdims[1].x ||
        subdims[0].y % subdims[1].y ||
        subdims[0].bwidth % subdims[1].bwidth ){

        return false;
    }

    //check fitting of bw to common vector sizes
    if( isComplexType(dtype) ){

        if( 2*subdims[1].bwidth > 16 ){

            return false;
        }
    }

    // check dimensions
    if( subdims[1].bwidth > 16 ||
        subdims[1].x > 16 ||
        subdims[1].y > 16 ){

        return false;
    }

    // estimate register usage, drop
    // inevitably slowed decompositions
    regUse =
        (   subdims[1].bwidth * subdims[1].x +
            subdims[1].bwidth * subdims[1].y +
            subdims[1].x * subdims[1].y ) *
        dtypeSize(dtype);

    regUse /= 16; // 16 bytes per register

    if( regUse >= 64 ){
        return false;
    }

    // passed PGranularity should be checked
    if( PGRAN_CHECK == check ){

        if( pgran->wgDim != 1 ){
            return false;
        }
        if( pgran->wgSize[0] != 64 ){
            return false;
        }

        if( pgran->wgSize[0] != subgA*subgB*itemsPerSubg ){
            return false;
        }
    }
    // PGranularity should be calculated
    else{
        pgran->wgDim = 1;
        pgran->wgSize[0] = subgA * subgB * itemsPerSubg;
    }

    return true;
}
