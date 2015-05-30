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
 * symv generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>

#include "blas_kgen.h"
#include "xxmv_common.h"

static const char *symvDecl =
    "__attribute__((reqd_work_group_size(%lu, %lu, 1)))\n"
    "void __kernel\n"
    "%csymv(\n"
    "    uint N,\n"
    "    const %s alpha,\n"
    "    const __global %s *restrict A,\n"
    "    const __global %s *restrict X,\n"
    "%s"
    "    __global %s *Y,\n"
    "    uint lda,\n"
    "%s"    // offset A, X and Y
    "%s"
    "%s"
    "    const uint startN,\n"
    "    uint actualN)\n";

static CLBLASMpatExtra mpatExtra;

struct symvPrivate {
    TilePostFetchPrivate *pfPriv;
    TileMulOpts *mulOpts;
    Tile tilea;
    bool diag;
    bool coord;
};

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

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs);

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);

static int
symvSubgGetDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void * pArgs);

static bool
subgCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static SolverOps symvSops = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL,
    NULL,
    calcNrThreads,
    NULL,
    solverFlags,
    fixupArgs,
    symvSubgGetDefaultDecomp, //getDefaultDecomposition
    subgCheckCalcDecomp, // get Decomp. List
    NULL,
    NULL
};

static void
declareSymvKernel(
    struct KgenContext *ctx,
    DataType dtype,
    const PGranularity *pgran,
    KernelExtraFlags kflags)
{
    bool incxOne = ((kflags & KEXTRA_INCX_ONE) != 0);
    bool incyOne = ((kflags & KEXTRA_INCY_ONE) != 0);
    bool beta0 = ((kflags & KEXTRA_BETA_ZERO) != 0);
    const char *incxDecl = incxOne ? "" : "    const int incx,\n";
    const char *incyDecl = incyOne ? "" : "    const int incy,\n";
    char betaDecl[128];
    char offDecl[128];
    char tmp[512];
    char fpref;
    const char *typeName;

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);
    if (beta0) {
        betaDecl[0] = '\0';
    }
    else {
        sprintf(betaDecl, "    const %s beta,\n", typeName);
    }

    offDecl[0] = '\0';
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        strcpy(offDecl, "    const uint offA,\n");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        strcat(offDecl, "    const uint offX,\n");
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        strcat(offDecl, "    const uint offY,\n");
    }

    sprintf(tmp, symvDecl, pgran->wgSize[0], pgran->wgSize[1], fpref, typeName,
            typeName, typeName, betaDecl, typeName, offDecl, incxDecl,
            incyDecl);
    kgenDeclareFunction(ctx, tmp);
}

/* avoid " + 0" statements */
static void
genAdd(char *buf, size_t val)
{
    if (val == 0) {
        buf[0] = 0; //zero length string
    }
    else {
        sprintf(buf, " + %lu", val);
    }
}

static int
genPostFetchMirror(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv)
{
    TilePostFetchPrivate *pfPriv = ((struct symvPrivate *)priv)->pfPriv;
    TileMulOpts *mulOpts = ((struct symvPrivate *)priv)->mulOpts;
    Tile *tileb = (Tile *)&pfPriv->gset->tileA;
    Tile *tilea = &((struct symvPrivate *)priv)->tilea;
    bool tra = ((mulOpts->flags & TILEMUL_TRA) != 0);
    char tmp[1024];
    char stmtStr[2][128];
    size_t blockx, blocky;
    unsigned int x, y;
    const struct SubproblemDim *dims = &pfPriv->gset->subdims[1];
    (void)mrole;


    blockx = blocky = 0;
    // zero triangular part of tile a
    // either single row of tile a either the whole tile have been fetched

    if (tra) {
        blocky = dims->bwidth;
        blockx = dims->y;
    }
    else {
        blocky = dims->y;
        blockx = dims->bwidth;
    }

    // loop through block rows
    for(y = 0; y < blocky; y++) {
        // loop through all elements of block row
        for(x = 0; x < blockx; x++) {
            Kstring kstr[3];
            const char *cmp = ">";
            sprintfTileElement(&kstr[0], tileb, x, y, 1);
            sprintfTileElement(&kstr[1], tileb, y, x, 1);
            sprintfTileElement(&kstr[2], tilea, y, x, 1);
            genAdd(stmtStr[0], x);
            genAdd(stmtStr[1], y);
            sprintf(tmp, "%s = k%s %s n%s ? %s : %s;\n",
                    kstr[2].buf, stmtStr[0], cmp, stmtStr[1],
                    kstr[0].buf, kstr[1].buf);
            kgenAddStmt(ctx, tmp);
        }
        pfPriv->fetchNumA++;
    }

    *tileb = *tilea;

    return 0;
}

static int
genPostFetchDiag(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv)
{
    TilePostFetchPrivate *pfPriv = ((struct symvPrivate *)priv)->pfPriv;
    Tile *tile = (Tile *)&pfPriv->gset->tileA;
    bool diag = ((struct symvPrivate *)priv)->diag;
    bool tra = ((struct symvPrivate *)priv)->coord;
    char tmp[1024];
    char stmtStr[2][128];
    const KernelVarNames *vnames = &pfPriv->gset->varNames;
    const char *coord = tra ? vnames->coordA : vnames->k;
    size_t blockx, blocky;
    unsigned int x, y;
    const struct SubproblemDim *dims = &pfPriv->gset->subdims[1];
    (void)mrole;


    blockx = blocky = 0;
    // zero triangular part of tile a
    // either single row of tile a either the whole tile have been fetched

    if (tra) {
        blocky = dims->bwidth;
        blockx = dims->y;
    }
    else {
        blocky = dims->y;
        blockx = dims->bwidth;
    }

    // loop through block rows
    for(y = 0; y < blocky; y++) {
        // loop through all elements of block row
        for(x = 0; x < blockx; x++) {
            Kstring kstr[3];
            const char *cmp = diag ? ">=" : ">";
            if (diag) {
                sprintfTileElement(&kstr[0], tile, x, y, 1);
            }
            else {
                sprintfTileElement(&kstr[0], tile, y, x, 1);
            }
            genAdd(stmtStr[0], x);
            genAdd(stmtStr[1], y);
            sprintf(tmp, "%s = Ktail <= %i || %s%s %s n%s ? 0 : %s;\n",
                    kstr[0].buf, y, coord, stmtStr[0], cmp, stmtStr[1],
                    kstr[0].buf);
            kgenAddStmt(ctx, tmp);
        }
        pfPriv->fetchNumA++;
    }
    return 0;
}

static int
genPostFetchVertDiag(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv)
{
    TilePostFetchPrivate *pfPriv = ((struct symvPrivate *)priv)->pfPriv;
    TileMulOpts *mulOpts = ((struct symvPrivate *)priv)->mulOpts;
    Tile *tile = (Tile *)&pfPriv->gset->tileA;
    bool diag = ((struct symvPrivate *)priv)->diag;
    char tmp[1024], tmp1[128] = "";
    char stmtStr[2][128];
    size_t blockx, blocky;
    unsigned int x, y;
    const struct SubproblemDim *dims = &pfPriv->gset->subdims[1];
    (void)mrole;

    blockx = blocky = 0;
    // zero triangular part of tile a
    // either single row of tile a either the whole tile have been fetched

    if (!diag) {
        blocky = dims->bwidth;
        blockx = dims->y;
    }
    else {
        blocky = dims->y;
        blockx = dims->bwidth;
    }

    // loop through block rows
    for(y = 0; y < blocky; y++) {
        // loop through all elements of block row
        for(x = 0; x < blockx; x++) {
            Kstring kstr[3];
            const char *cmp = diag ? ">=" : ">";
            const char *name = diag ? "k" : "coordA";
            if (diag) {
                sprintfTileElement(&kstr[0], tile, y, x, 1);
            }
            else {
                sprintfTileElement(&kstr[0], tile, x, y, 1);
            }
            genAdd(stmtStr[0], x);
            genAdd(stmtStr[1], y);
            if (mulOpts->flags & TILEMUL_SKEW_B) {
                sprintf(tmp1, "Ktail <= %i || ", y);
            }
            sprintf(tmp, "%s = %s%s%s %s n%s ? 0 : %s;\n",
                    kstr[0].buf, tmp1, name, stmtStr[0], cmp, stmtStr[1],
                    kstr[0].buf);
            kgenAddStmt(ctx, tmp);
        }
        pfPriv->fetchNumA++;
    }
    return 0;
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
    KernelExtraFlags kflags = kextra->flags;
    bool upper = ((kflags & KEXTRA_UPPER_TRIANG) != 0) ^
                  ((kflags & KEXTRA_COLUMN_MAJOR) != 0);
    char tmp[2048];
    const char *typeName;
    DataType dtype = kextra->dtype;
    BlasGenSettings gset, tgset, lset, gset1;
    CLBLASKernExtra kextraTmp;
    TileMulOpts mulOpts, tmulOpts;
    KernelVarNames *vnames = &gset.varNames;
    ssize_t ret;
    size_t vecLen = kextra->vecLen;
    const char *outTypeName;
    bool b;
    TilePostFetchPrivate pfPriv;
    struct symvPrivate priv;
    size_t wgSize;
    bool tailM = (kflags & KEXTRA_TAILS_M) != 0;
    bool tailK = (kflags & KEXTRA_TAILS_K) != 0;
    bool tra = (kflags & KEXTRA_COLUMN_MAJOR) != 0;
    bool rowMaj = !isMatrixAccessColMaj(CLBLAS_SYMV, kflags, MATRIX_A);
    bool isComplex = isComplexType(dtype);
    Tile tileb;
    const char *gid = "get_group_id(0)";
    const char *lid = "get_local_id(0)";
    bool isHoriz = subdims[1].bwidth >= subdims[1].y;
    unsigned int bStep = subdims[0].bwidth / subdims[1].bwidth;
    unsigned int cLocal;
    unsigned int nPlans;

    wgSize = (subdims[0].y / subdims[1].y) *
            (subdims[0].bwidth / subdims[1].bwidth);
    assert(pgran->wgSize[0] == wgSize);
    assert(subdims[0].x == 1);
    assert(subdims[1].x == 1);

    memset(&gset, 0, sizeof(gset));
    memset(&mulOpts, 0, sizeof(mulOpts));
    memset(&pfPriv, 0, sizeof(pfPriv));
    memset(&priv, 0, sizeof(priv));
    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations
    b = isDoubleBasedType(dtype);
    kgenDeclareUptrs(ctx, b);

    typeName = dtypeBuiltinType(dtype);

    declareSymvKernel(ctx, dtype, pgran, kflags);

    ret = kgenBeginFuncBody(ctx);
    /* 1D work space. Matrix is divided among wi, each calculates it's own
     * part of vector y */

    kgenAddStmt(ctx, "#define M actualN\n");
    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.subdims[0].itemX = gset.subdims[0].x = 1;
    gset.subdims[1].itemX = gset.subdims[1].x = 1;
    gset.subdims[0].bwidth = gset.subdims[1].bwidth;
    gset.flags |= BGF_WHOLE_A | BGF_UPTRS;

    gset.kextra = kextra;
    gset.pgran = pgran;

    initDefaultTiles(&gset, CLBLAS_SYMV, 0, PRIV_STORAGE_VARIABLE_SET);
    gset.tileA.vecLen = umin(8u, tra ? gset.tileA.nrCols : gset.tileA.nrRows);

    if (isComplex) {
         gset.tileCY.vecLen = 1;
    }
    declareTileStorages(ctx, &gset);
    genZeroTile(ctx, &gset.tileCY);
    getVectorTypeName(dtype, gset.tileCY.vecLen, &outTypeName, NULL);
    cLocal = wgSize / bStep;
    nPlans = gset.tileCY.nrRows / gset.tileCY.vecLen;

    sprintf(tmp, "__local %s localRes[%u][%u];\n",
                outTypeName, pgran->wgSize[0], nPlans);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint coordA = (%s * %u + %s / %u) * %lu + startN;\n",
                 gid, cLocal, lid, bStep, subdims[1].y);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint n = coordA;\n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint k0 = (%s %% %u) * %lu;\n",
                 lid,  bStep, subdims[1].bwidth);
    kgenAddStmt(ctx, tmp);
    kgenAddStmt(ctx, "actualN += startN;\n");

    kgenAddBlankLine(ctx);

    kgenBeginBranch(ctx,"if (coordA < actualN && k0 < N)");

    genIncPointers(ctx, kflags);
    sprintf(tmp,
            "const GPtr Ag = {(__global %s*)A};\n"
            "const GPtr Xg = {(__global %s*)X};\n",
            typeName, typeName);
    kgenAddStmt(ctx, tmp);

    kgenAddBlankLine(ctx);

    kgenAddStmt(ctx, "uint k = k0;\n");

    if (tailK) {
        sprintf(tmp, "uint Ntail = N %% %lu;\n", subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "uint Ktail = N %% %lu;\n\n", subdims[1].y);
        kgenAddStmt(ctx, tmp);
        kgenBeginBranch(ctx, "if (n + Ktail < N)");
        kgenAddStmt(ctx, "N -= Ntail;\n");
        kgenAddBlankLine(ctx);
    }

    mulOpts.flags |= TILEMUL_OPTIMIZE_COORD_CALC;
    if (tailM) {
        vnames->sizeM = "N";
    }

    vnames->A = "Ag";
    vnames->B = "Xg";
    vnames->coordA = "coordA";
    vnames->coordB = ""; //should not be used for vector
    vnames->k = "k";
    vnames->lda = "lda";
    vnames->sizeK = "N";
    vnames->sizeM = "N";

    mulOpts.flags |= TILEMUL_NOT_FETCH_B | TILEMUL_TRB | TILEMUL_NOT_INC_K;
    if ((kflags & KEXTRA_CONJUGATE_A) != 0) {
        mulOpts.flags |= TILEMUL_CONJA;
    }
    if ((kflags & KEXTRA_ENABLE_MAD) != 0) {
        mulOpts.core = TILEMUL_MAD;
    }
    else {
        mulOpts.core = TILEMUL_MULADD;
    }
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;

    if (rowMaj) {
        mulOpts.flags |= TILEMUL_BW_STRIDE;
    }

    if (upper) {
        kgenAddStmt(ctx, "// k loop over column from the beginning of the column till the diagonal\n");
    }
    else {
        kgenAddStmt(ctx, "// k loop over row from the beginning of the row till the diagonal\n");
    }
    sprintf(tmp, "for (; k < n/%lu*%lu; k += %lu)",
        subdims[1].bwidth, subdims[1].bwidth, bStep*subdims[1].bwidth);
    kgenBeginBranch(ctx, tmp);

    genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
            mulOpts.flags, kflags);

    upper ^= rowMaj;
    tra ^= rowMaj;
    if (upper ^ rowMaj && tra) {
        mulOpts.flags |= TILEMUL_TRA;
    }
    gset.tileA.trans ^= !upper;
    tgset = gset;
    tmulOpts = mulOpts;

    ret = tileMulGen(ctx, &gset, &mulOpts);
    if (ret != 0) {
        return ret;
    }
    kgenEndBranch(ctx, NULL); /* k loop */

    if (tailK)
    {
            kextraTmp = *kextra;
            gset1 = gset;

            kextraTmp.vecLen = 1;
            gset1.kextra = &kextraTmp;

            gset1.subdims[0].bwidth = gset1.subdims[1].bwidth = 1;

            gset1.tileBX.nrRows = 1;
            gset1.tileA.nrCols = 1;
            kextraTmp.vecLenA = 1;
    }


    if (isHoriz)
    {
        lset = gset;
        lset.subdims[0].bwidth = lset.subdims[1].bwidth =
            lset.subdims[1].y = umin(subdims[1].bwidth, subdims[1].y);
        lset.tileA.nrCols = lset.tileA.nrRows =
            lset.tileBX.nrRows = lset.subdims[1].y;

        kgenAddStmt(ctx, "// the diagonal\n");
        kgenBeginBranch(ctx, "if (k <= n)");
        kgenAddStmt(ctx, "uint k1 = k;\n");

        if (subdims[1].bwidth != subdims[1].y) {
            kgenAddStmt(ctx, "// the pred diagonal\n");
            sprintf(tmp, "for (; k < n; k += %lu)", lset.subdims[1].bwidth);
            kgenBeginBranch(ctx, tmp);

            genFetchX(ctx, &lset.tileBX, lset.subdims[1].bwidth, dtype, vnames,
                    mulOpts.flags, kflags);

            ret = tileMulGen(ctx, &lset, &mulOpts);
            if (ret != 0) {
                return ret;
            }
            kgenEndBranch(ctx, NULL); /* k loop */
        }

        initTile(&tileb, "b", lset.subdims[1].bwidth, lset.subdims[1].bwidth,
            lset.subdims[1].bwidth, lset.tileA.dtype, PRIV_STORAGE_VARIABLE_SET,
            lset.tileA.trans, lset.tileA.packed);
        declareOneTileStorage(ctx, &tileb);

        genFetchX(ctx, &lset.tileBX, lset.subdims[1].bwidth, dtype, vnames,
                mulOpts.flags, kflags);

        priv.mulOpts = &mulOpts;
        priv.pfPriv = &pfPriv;
        priv.tilea = lset.tileA;
        priv.diag = false;

        pfPriv.funcID = CLBLAS_SYMV;
        pfPriv.gset = &lset;
        lset.tileA = tileb;
        mulOpts.postFetch = genPostFetchMirror;
        mulOpts.postFetchPriv = &priv;

        ret = tileMulGen(ctx, &lset, &mulOpts);
        if (ret != 0) {
            return ret;
        }

        if (upper ^ rowMaj && tra) {
            mulOpts.flags &= ~TILEMUL_TRA;
        }
        else {
            mulOpts.flags |= TILEMUL_TRA;
        }
        gset.tileA.trans = lset.tileA.trans ^= true;
        mulOpts.postFetch = NULL;
        mulOpts.postFetchPriv = NULL;

        if (subdims[1].bwidth != subdims[1].y) {
            size_t width = umax(subdims[1].bwidth, subdims[1].y);
            kgenAddStmt(ctx, "// the post diagonal\n");
            if (tailK) {
                kgenBeginBranch(ctx, "if(k < N)");
            }
            sprintf(tmp, "for (k += %lu; k < n/%lu*%lu+%lu; k += %lu)",
                    lset.subdims[1].bwidth,
                    width, width, width,
                    lset.subdims[1].bwidth);
            kgenBeginBranch(ctx, tmp);

            genFetchX(ctx, &lset.tileBX, lset.subdims[1].bwidth, dtype, vnames,
                    mulOpts.flags, kflags);

            ret = tileMulGen(ctx, &lset, &mulOpts);
            if (ret != 0) {
                return ret;
            }
            kgenEndBranch(ctx, NULL); /* k loop */

            if (tailK) {
                kgenEndBranch(ctx, NULL);
                kgenBeginBranch(ctx, "else");
                /* Handle tail along vector X */

                kgenAddStmt(ctx, "N += Ntail;\n");

                mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_A;
#if 1
                sprintf(tmp, "for (k += %lu; k < actualN; k++)",
                    lset.subdims[1].bwidth);
                kgenBeginBranch(ctx, tmp);

                gset1.tileA.trans = gset.tileA.trans;

                genFetchX(ctx, &gset1.tileBX, gset1.kextra->vecLen, dtype, vnames,
                          mulOpts.flags, kflags);
                ret = tileMulGen(ctx, &gset1, &mulOpts);
                if (ret != 0) {
                    return ret;
                }
                kgenEndBranch(ctx, NULL); /* k loop for tails along vector X */
#else
                mulOpts.flags |= TILEMUL_SKEW_B | TILEMUL_NOT_INC_K;
                genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
                          mulOpts.flags, kflags);
                ret = tileMulGen(ctx, &gset, &mulOpts);
                if (ret != 0) {
                    return ret;
                }
#endif

                mulOpts.flags &= ~TILEMUL_GLOBAL_CYCLIC_A;
                kgenEndBranch(ctx, NULL);
            }
        }

        sprintf(tmp, "k = k1 + %lu;\n", bStep*subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        kgenEndBranch(ctx, NULL);
    }
    else
    {

        kgenAddStmt(ctx, "// the diagonal\n");
        sprintf(tmp, "if (k <= (n  + (get_local_id(0)%%%lu)*%lu))",
            subdims[1].y/subdims[1].bwidth, subdims[1].bwidth);
        kgenBeginBranch(ctx, tmp);

        genFetchX(ctx, &gset.tileBX, gset.subdims[1].bwidth, dtype, vnames,
                    mulOpts.flags, kflags);

        kgenBeginBranch(ctx, NULL);

        priv.mulOpts = &mulOpts;
        priv.pfPriv = &pfPriv;
        priv.diag = true;

        pfPriv.funcID = CLBLAS_SYMV;
        pfPriv.gset = &gset;
        mulOpts.postFetch = genPostFetchVertDiag;
        mulOpts.postFetchPriv = &priv;

        ret = tileMulGen(ctx, &gset, &mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        if (upper ^ rowMaj && tra) {
            mulOpts.flags &= ~TILEMUL_TRA;
        }
        else {
            mulOpts.flags |= TILEMUL_TRA;
        }
        gset.tileA.trans ^= true;
        lset = gset;

        sprintf(tmp, "n += (get_local_id(0)%%%lu)*%lu;\n",
            subdims[1].y/subdims[1].bwidth, subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        kgenBeginBranch(ctx, NULL);

        priv.diag = false;
        ret = tileMulGen(ctx, &gset, &mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);

        mulOpts.postFetch = NULL;
        mulOpts.postFetchPriv = NULL;

        sprintf(tmp, "k += %lu;\n", bStep*subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        kgenEndBranch(ctx, NULL); /* if */
    }

    if (upper) {
        kgenAddStmt(ctx, "// k loop over row from the diagonal till the right\n");
    }
    else {
        kgenAddStmt(ctx, "// k loop over column from the diagonal till the bottom\n");
    }
    sprintf(tmp, "for (; k < N; k += %lu)", bStep*subdims[1].bwidth);
    kgenBeginBranch(ctx, tmp);

    genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
            mulOpts.flags, kflags);

    ret = tileMulGen(ctx, &gset, &mulOpts);
    if (ret != 0) {
        return ret;
    }
    kgenEndBranch(ctx, NULL); /* k loop */

    if (tailK) {
        /* Handle tail along vector X */
        kgenAddStmt(ctx, "N += Ntail;\n");

        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_A;
#if 1
        sprintf(tmp, "for (; k < N; k++)");
        kgenBeginBranch(ctx, tmp);

        gset1.tileA.trans = gset.tileA.trans;

        genFetchX(ctx, &gset1.tileBX, gset1.kextra->vecLen, dtype, vnames,
                  mulOpts.flags, kflags);
        ret = tileMulGen(ctx, &gset1, &mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL); /* k loop for tails along vector X */
#else
        mulOpts.flags |= TILEMUL_SKEW_B | TILEMUL_NOT_INC_K;
        genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
                  mulOpts.flags, kflags);
        ret = tileMulGen(ctx, &gset, &mulOpts);
        if (ret != 0) {
            return ret;
        }
#endif

        kgenEndBranch(ctx, NULL);

        kgenBeginBranch(ctx, "else");

        sprintf(tmp, "for (; k < N; k += %lu)", bStep*subdims[1].bwidth);
        kgenBeginBranch(ctx, tmp);

        tmulOpts.flags |= TILEMUL_SKEW_B | TILEMUL_GLOBAL_CYCLIC_A;
        genFetchX(ctx, &tgset.tileBX, tgset.kextra->vecLen, dtype, vnames,
                tmulOpts.flags, kflags);

        priv.mulOpts = &tmulOpts;
        priv.pfPriv = &pfPriv;
        pfPriv.gset = &tgset;
        priv.diag = false;

        pfPriv.funcID = CLBLAS_SYMV;
        tmulOpts.postFetch = genPostFetchDiag;
        tmulOpts.postFetchPriv = &priv;

        ret = tileMulGen(ctx, &tgset, &tmulOpts);
        if (ret != 0) {
            return ret;
        }

        if (isHoriz) {
            sprintf(tmp, "if (k + %lu > N) break;\n", subdims[1].bwidth);
        }
        else {
            sprintf(tmp, "if (k + %lu > N + (get_local_id(0)%%%lu)*%lu) break;\n",
                subdims[1].y, subdims[1].y/subdims[1].bwidth, subdims[1].bwidth);
        }
        kgenAddStmt(ctx, tmp);

        kgenEndBranch(ctx, NULL); /* k loop */

        kgenBeginBranch(ctx, "if (k < N)");
        if (isHoriz) {
            kgenAddStmt(ctx, "k = n;\n");
        }
        else {
            sprintf(tmp, "n += (get_local_id(0)%%%lu)*%lu;\n",
                subdims[1].y/subdims[1].bwidth, subdims[1].bwidth);
            kgenAddStmt(ctx, tmp);
        }

        genFetchX(ctx, &lset.tileBX, lset.kextra->vecLen, dtype, vnames,
                tmulOpts.flags, kflags);

        priv.mulOpts = &tmulOpts;
        priv.pfPriv = &pfPriv;
        priv.diag = true;

        pfPriv.funcID = CLBLAS_SYMV;
        pfPriv.gset = &lset;
        tmulOpts.postFetch = genPostFetchDiag;
        tmulOpts.postFetchPriv = &priv;

        if (!isHoriz) {
            if (upper ^ rowMaj && tra) {
                tmulOpts.flags &= ~TILEMUL_TRA;
            }
            else {
                tmulOpts.flags |= TILEMUL_TRA;
            }
            kgenAddStmt(ctx, "Ktail = N - n;\n");
            priv.coord = true;
        }
        else {
            priv.coord = false;
        }
        tmulOpts.flags |= TILEMUL_SKEW_B | TILEMUL_GLOBAL_CYCLIC_A | TILEMUL_GLOBAL_CYCLIC_K;


        ret = tileMulGen(ctx, &lset, &tmulOpts);
        if (ret != 0) {
            return ret;
        }

        kgenEndBranch(ctx, NULL);

        kgenEndBranch(ctx, NULL);
    }


    if (!isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
        mulOpts.flags &= ~TILEMUL_BW_STRIDE;
    }

    kgenEndBranch(ctx,NULL);

    genStoreLocalResult(ctx, &gset.tileCY, lid);

    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);

    sprintf(tmp, "if ((%s %% %u) == 0 && coordA < actualN && k0 < N)", lid, bStep);
    kgenBeginBranch(ctx, tmp);

    genAddLocalResult(ctx, &gset.tileCY, lid, bStep, 1);

    /* write back the results */
    /* y := alpha*A*x + beta*y */
    sprintf(tmp,"(%s - startN)", vnames->coordA);
    setResultPos(ctx, kflags, tmp);

    updateResultVectorTiled(ctx, kflags, vecLen, &gset.tileCY);

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
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    cl_int inc;
    int i;

    initSizeKarg(&args[0], blasArgs->K);
    assignScalarKarg(&args[1], &(blasArgs->alpha), blasArgs->dtype);
    INIT_KARG(&args[2], blasArgs->A);
    INIT_KARG(&args[3], blasArgs->B);
    i = 4;
    if (!(kflags & KEXTRA_BETA_ZERO)) {
        assignScalarKarg(&args[i++], &(blasArgs->beta), blasArgs->dtype);
    }
    initMemobjKarg(&args[i++], blasArgs->C, NULL, 0, 0);
    initSizeKarg(&args[i++], blasArgs->lda.matrix);
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offBX);
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offCY);
    }
    if (!(kflags & KEXTRA_INCX_ONE)) {
        inc = blasArgs->ldb.vector;
        INIT_KARG(&args[i], inc);
        i++;
    }
    if (!(kflags & KEXTRA_INCY_ONE)) {
        inc = blasArgs->ldc.vector;
        INIT_KARG(&args[i], inc);
        i++;
    }

    initSizeKarg(&args[i++], blasArgs->offsetN);
    initSizeKarg(&args[i++], blasArgs->N); //Actual N
}

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    CLBlasKargs *kargs = (CLBlasKargs*)args;

    (void)extra;
    (void)subdims;

    if (kargs->offsetN) {
        if (kargs->ldc.vector < 0) {
            // K store the original height of the matrix A
            kargs->offCY += (kargs->K - kargs->offsetN) *
                            abs(kargs->ldc.vector);
        }
        else {
            kargs->offCY += kargs->offsetN * kargs->ldc.vector;
        }
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
    (void)kernelArgs;

    /*
     * One needs y1 * wgSize size of local memory in elements,
     * but y1 is not calculated yet. The expression below produces
     * reliable a larger value. It is larger in dims[1].bwidth times.
     */
    size = dim[0].y * dim[0].bwidth * dtypeSize(dtype);

    return (size <= ldsSize);
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
    unsigned int subgr = subdims[0].bwidth / subdims[1].bwidth;
    (void)extra;

    //each work item handles y1 lines
    threads[0] = divRoundUp(kargs->N, subdims[1].y) * subgr;
    threads[0] = roundUp(threads[0], pgran->wgSize[0]);
    threads[1] = 0;
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D);
}

static int
symvSubgGetDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void * pArgs)
{
    (void)subdimsNum;
    DUMMY_ARG_USAGE(pArgs);

    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;
    pgran->wgSize[1] = 1;

    subdims[1].bwidth = 4;
    subdims[1].itemX = subdims[1].x = 1;
    subdims[1].itemY = subdims[1].y = 4;

    subdims[0].bwidth = 8 * subdims[1].bwidth;
    subdims[0].itemX = subdims[0].x = 1;
    subdims[0].itemY = subdims[0].y = 8 * subdims[1].y;

    return 0;
}

static bool
subgCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check)
{
    unsigned int divider1 = dtypeSize(dtype)/sizeof(cl_float);

    //EINVAL
    if( (subdimsNum<2)||
        (NULL==pgran)||
        (NULL==subdims) ){

        return false;
    }

    if( 0 == subdims[0].x ||
        0 == subdims[0].y ||
        0 == subdims[0].bwidth ||
        0 == subdims[1].x ||
        0 == subdims[1].y ||
        0 == subdims[1].bwidth ){

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

        if( 2*subdims[1].bwidth > 32 ){

            return false;
        }
    }

    // check dimensions
    if( subdims[1].bwidth > 16 / divider1 ||
        subdims[1].x > 1 ||
        subdims[1].y > 16 / divider1 ){

        return false;
    }

    if( subdims[0].bwidth > 128 ||
        subdims[0].x > 1 ||
        subdims[0].y > 128 ){

        return false;
    }

    if (64 != (subdims[0].y / subdims[1].y) *
        (subdims[0].bwidth / subdims[1].bwidth)) {
        return false;
    }

    if (subdims[0].y > subdims[0].bwidth &&
        subdims[0].y / subdims[0].bwidth < (subdims[0].bwidth / subdims[1].bwidth)) {
        return false;
    }

    // passed PGranularity should be checked
    if( PGRAN_CHECK == check ){
        if( pgran->wgSize[0] * pgran->wgSize[1] != 64 ){
            return false;
        }
    }
    // PGranularity should be calculated
    else{
        pgran->wgDim = 1;
        pgran->wgSize[1] = 1;
        pgran->wgSize[0] = 64;
        //subdims[0].bwidth = (pgran->wgSize[0] * subdims[1].bwidth) /
        //    (subdims[0].y / subdims[1].y);
    }
    /*Debug out for Tune*/

    return true;
}

void
initSymvPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block symv";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &symvSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}
