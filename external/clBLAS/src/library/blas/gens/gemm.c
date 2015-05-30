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
 * Cached global buffers based gemm generator
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>

#include "blas_kgen.h"
#include "blas_subgroup.h"
#include "gen_helper.h"

typedef struct {
    size_t staggered;
} MAY_ALIAS extraData_t;

static CLBLASMpatExtra mpatExtra;

static ssize_t
blockGen(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static ssize_t
subgGen(
    char *pBuf,
    size_t buflen,
    const struct SubproblemDim *pSubDims,
    const struct PGranularity *pPGran,
    void *pExtra );

static void
assignBlockKargs(
    KernelArg *args,
    const void *params,
    const void *extra);

static bool
blockCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static int
blockGetPerf(
    unsigned int kflags,
    const void *args);

static void
assignSubgKargs(
    KernelArg *args,
    const void *params,
    const void *extra);

static SolverFlags
solverFlags(void);

static DecompositionAxis
innerDecompositionAxis(const void *args);

static int
gemmSubgGetDefaultDecomp(
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

static void
subgCalcGlobalThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra
);

static int
subgGetPerf(
    unsigned int kflags,
    const void *args);

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static SolverOps blockSOps = {
    blockGen,
    assignBlockKargs,
    NULL,
    blockGetPerf,
    innerDecompositionAxis,
    NULL,
    NULL,
    solverFlags,
    NULL,// fixup kargs
    NULL, //blockGetDefaultDecomp,
    blockCheckCalcDecomp,
    NULL,
    NULL
};

static SolverOps subgSOps = {
    subgGen,
    assignSubgKargs,
    NULL,
    subgGetPerf,
    innerDecompositionAxis,
    subgCalcGlobalThreads,
    NULL,
    solverFlags,
    fixupArgs,// fixup kargs
    gemmSubgGetDefaultDecomp,
    subgCheckCalcDecomp,
    NULL,
    NULL
};

//*****************************************************************************
//-----------------------------------------------------------------------------

static void
genSetupItemPtr(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    MatrixRole mrole)
{
    char tmp[1024];
    unsigned int vecLen;
    char ldv[64];
    int shift;
    char ptrLit;
    char shiftMul[128];
    size_t tileWidth;
    int widx;
    KernelExtraFlags kflags = gset->kextra->flags;

    /*
     * The matrix was made B inner if every thread should accesses their
     * elements with a large stride but accesses elements of the matrix A
     * sequentially to provide more coalesced memory accesses.
     * Otherwise, the matrix A was made inner.
     */
    widx = (!isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_A) &&
            isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_B)) ? 1 : 0;

    vecLen = getVecLen(gset, CLBLAS_GEMM, mrole);
    shift = findHighestSetBit(vecLen);
    if (mrole == MATRIX_A) {
        tileWidth = gset->subdims[1].y;
        ptrLit = 'A';
        if ((shift > 0) && !(gset->flags & BGF_LD_IN_VECTORS)) {
            sprintf(ldv, "(lda >> %d)", shift);
        }
        else {
            strcpy(ldv, "lda");
        }
    }
    else {
        tileWidth = gset->subdims[1].x;
        ptrLit = 'B';
        if ((shift > 0) && !(gset->flags & BGF_LD_IN_VECTORS)) {
            sprintf(ldv, "(ldb >> %d)", shift);
        }
        else {
            strcpy(ldv, "ldb");
        }
        widx = 1 - widx;
    }

    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, mrole)) {
        if (tileWidth / vecLen > 1) {
            sprintf(shiftMul, " * %lu", tileWidth / vecLen);
        }
        else {
            shiftMul[0] = '\0';
        }
        // Alternative calculate global thead id to eliminate Channel Conflicts.
        if (mrole == MATRIX_B) {
            int bankSize = 2048;
            int dataSize = 0;
            int grShift;

            DataType dtype = gset->kextra->dtype;
            switch (dtype) {
            case TYPE_FLOAT:           dataSize = 4; break;
            case TYPE_COMPLEX_DOUBLE:  dataSize = 16; break;
            default:                   dataSize = 8; break;
            }

            grShift = bankSize/ dataSize;

            sprintf(tmp,
                "get_group_id_%d = (get_group_id(0) + get_group_id(1))"
                    "%% get_num_groups(%d);\n", widx, widx);
            kgenAddStmt(ctx, tmp);

            sprintf(tmp,
                "get_global_id_%d = get_group_id_%d * get_local_size(%d) "
                    "+ get_local_id(%d);\n",widx, widx, widx, widx);
            kgenAddStmt(ctx, tmp);


            sprintf(tmp,
                "kif = (N %% %d != 0);\n"
                "get_global_id_%d = (kif*(uint)get_global_id(%d)) + "
                    "((1-kif)*get_global_id_%d);\n",grShift, widx, widx, widx);
            kgenAddStmt(ctx, tmp);

            sprintf(tmp,
                "%c += get_global_id_%d%s;",
                    ptrLit, widx, shiftMul);
        }
        else {
            sprintf(tmp, "%c += (uint)get_global_id(%d)%s;\n",
                    ptrLit, widx, shiftMul);
        }

    }
    else {
        sprintf(tmp, "%c += %luu * (uint)get_global_id(%d) * %s;\n",
                ptrLit, tileWidth, widx, ldv);
    }
    kgenAddStmt(ctx, tmp);
}

static void
genShiftPointers(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    KernelExtraFlags kflags,
    bool vectorizedPtrs)
{
    char tmp[1024];
    unsigned int flags[3] = {KEXTRA_A_OFF_NOT_ZERO, KEXTRA_BX_OFF_NOT_ZERO,
                             KEXTRA_CY_OFF_NOT_ZERO};
    char ptrNames[3] = {'A', 'B', 'C'};
    const char *offNames[3] = {"offA", "offB", "offC"};
    MatrixRole mroles[3] = {MATRIX_A, MATRIX_B, MATRIX_C};
    int i;

    for (i = 0; i < 3; i++) {
        if (kflags & flags[i]) {
            unsigned int vecLen;

            vecLen = getVecLen(gset, CLBLAS_GEMM, mroles[i]);

            if( vectorizedPtrs && (vecLen > 1) ) {
                sprintf(tmp, "%c += %s / %u;\n",
                        ptrNames[i], offNames[i], vecLen);
            }
            else {
                sprintf(tmp, "%c += %s;\n", ptrNames[i], offNames[i]);
            }
            kgenAddStmt(ctx, tmp);
        }
    }
}

//-----------------------------------------------------------------------------

static void
sprintfOffABC(
    char *str,
    KernelExtraFlags kflags)
{
    str[0] = '\0';
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        str += sprintf(str, ",\n    const uint offA");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        str += sprintf(str, ",\n    const uint offB");
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        str += sprintf(str, ",\n    const uint offC");
    }
}

static void
declareKernel(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const char *nameSuffix)
{
    char tmp[4096];
    char offABC[1024];
    char fpref;
    char *tnameA, *tnameB;
    const char *tnameC;
    const char *rawType;
    DataType dtype = gset->kextra->dtype;
    unsigned int vecLen;
    const PGranularity *pgran = gset->pgran;

    fpref = dtypeToBlasPrefix(dtype);
    rawType = dtypeBuiltinType(dtype);
    vecLen = getVecLen(gset, CLBLAS_GEMM, MATRIX_A);
    getVectorTypeName(dtype, vecLen, (const char **)&tnameA, NULL);
    vecLen = getVecLen(gset, CLBLAS_GEMM, MATRIX_B);
    getVectorTypeName(dtype, vecLen, (const char **)&tnameB, NULL);

    // FIXME - take into account flag BGF_LD_IN_VECTORS
    //sprintf( tnameC, "%s", rawType );
    getVectorTypeName( dtype,
        getVecLen( gset, 0, MATRIX_C ),
        &tnameC,
        NULL );

    sprintfOffABC(offABC, gset->kextra->flags);

    sprintf(tmp, "__attribute__((reqd_work_group_size(%u, %u, 1)))\n"
                 "void __kernel\n"
                 "%cgemm%s(\n"
                 "    uint M,\n"
                 "    uint N,\n"
                 "    uint K,\n"
                 "    const %s alpha,\n"
                 "    const %s beta,\n"
                 "    const __global %s *restrict A,\n"
                 "    const __global %s *restrict B,\n"
                 "    __global %s *C,\n"
                 "    uint lda,\n"
                 "    uint ldb,\n"
                 "    uint ldc%s)\n",
            pgran->wgSize[0], pgran->wgSize[1], fpref, nameSuffix,
            rawType, rawType, tnameA, tnameB, tnameC, offABC);

    kgenDeclareFunction(ctx, tmp);
}

//-----------------------------------------------------------------------------

static void
genHitMatrixCheck(
    struct KgenContext *ctx,
    KernelExtraFlags kflags)
{
    /* tails of upper level blocks */
    bool tailsM = kflags & KEXTRA_TAILS_M;
    bool tailsN = kflags & KEXTRA_TAILS_N;

    if (tailsM) {
        if (tailsN) {
            kgenAddStmt(ctx, "if ((coord.y >= M) || (coord.x >= N)) {\n");
        }
        else {
            kgenAddStmt(ctx, "if (coord.y >= M) {\n");
        }
    }
    else {
        if (tailsN) {
            kgenAddStmt(ctx, "if (coord.x >= N) {\n");
        }
    }

    if (tailsM || tailsN) {
        kgenAddStmt(ctx, "    return;\n}\n\n");
    }
}

//-----------------------------------------------------------------------------

static ssize_t
blockGen(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    struct KgenContext *ctx;
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    KernelExtraFlags kflags = kextra->flags;
    bool isRelA, isRelB;
    bool tailsK = ((kflags & KEXTRA_TAILS_K_LOWER) != 0);
    DataType dtype = kextra->dtype;
    char tmp[2048];
    bool doubleBased = isDoubleBasedType(dtype);
    BlasGenSettings gset;
    KernelVarNames *vnames = &gset.varNames;
    TileMulOpts mulOpts;
    ssize_t ret;
    char globalIdB[64];
    const char *alignedK;
    FetchAddrMode addrMode, addrMask = 0;
    FetchOpts fopts;
    TilePostFetchPrivate pfPriv;
    TailStatus tailStatus;
    UpdateResultFlags upFlags;
    unsigned int i;
    unsigned int vecLen;
    int isColMajA;
    int isColMajB;

    memset(&gset, 0, sizeof(gset));
    memset(&mulOpts, 0, sizeof(mulOpts));
    memset(&pfPriv, 0, sizeof(pfPriv));
    memset(&fopts, 0, sizeof(fopts));

    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.flags = BGF_DISTINCT_VECLEN | BGF_LD_IN_VECTORS;

    // FIXME: throw the explicit constant away
    switch (dtype) {
    case TYPE_FLOAT:
//        i = 12;
        i = 16;
        break;
    case TYPE_COMPLEX_DOUBLE:
        i = 6;
        break;
    default:
        i = 8;
        break;
    }

    if (subdims[1].y + subdims[1].x <= i) {
        gset.flags |= BGF_WHOLE_A;
    }
    gset.kextra = kextra;
    gset.pgran = pgran;
    //avoid [0].bw loop
    gset.subdims[0].bwidth = gset.subdims[1].bwidth;

    mulOpts.core = ((kflags & KEXTRA_ENABLE_MAD) &&
                    (dtype != TYPE_COMPLEX_FLOAT)) ? TILEMUL_MAD
                                                   : TILEMUL_MULADD;
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;
    mulOpts.fctx = createFetchContext();
    if (mulOpts.fctx == NULL) {
        return -ENOMEM;
    }

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        destroyFetchContext(mulOpts.fctx);
        return -ENOMEM;
    }

    isColMajA = isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_A);
    isColMajB = isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_B);

    alignedK = (tailsK) ? "Kbase" : "K";

    // setup kernel variables
    vnames->A = "A";
    vnames->B = "B";
    vnames->C = "C";
    vnames->coordA = "coord.y";
    vnames->coordB = "coord.x";
    vnames->k = "coord.z";
    vnames->sizeK = alignedK;
    vnames->sizeM = "M";
    vnames->sizeN = "N";
    vnames->lda = "lda";
    vnames->ldb = "ldb";
    vnames->ldc = "ldc";
    vnames->alpha = "alpha";
    vnames->beta = "beta";

    // at first, generate needed declarations
    ret = kgenDeclareUptrs(ctx, doubleBased);

    declareKernel(ctx, &gset, "Block");
    ret = kgenBeginFuncBody(ctx);

    if (tailsK) {
        sprintf(tmp, "const uint Ktail = K %% %lu;\n"
                     "const uint Kbase = K - Ktail;\n",
                subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        alignedK = "Kbase";
    }
    else {
        alignedK = "K";
    }

    initDefaultTiles(&gset, CLBLAS_GEMM, 0, PRIV_STORAGE_VARIABLE_SET);
    declareTileStorages(ctx, &gset);
    kgenAddStmt(ctx, "uint4 coord = 0u; /* contains coordB, coordA, k */\n");
    kgenAddBlankLine(ctx);

    vecLen = getVecLen(&gset, CLBLAS_GEMM, MATRIX_A);
    if (vecLen > 1) {
        kgenPrintf(ctx, "lda /= %u;\n", vecLen);
    }
    vecLen = getVecLen(&gset, CLBLAS_GEMM, MATRIX_B);
    if (vecLen > 1) {
        kgenPrintf(ctx, "ldb /= %u;\n", vecLen);
    }

    /*
     * The matrix was made B inner if every thread should accesses their
     * elements with a large stride but accesses elements of the matrix A
     * sequentially to provide more coalesced memory accesses.
     * Otherwise, the matrix A was made inner.
     */
    i = (!isColMajA && isColMajB) ? 1 : 0;

    tailStatus = checkGenAdjustTailCoords(NULL, CLBLAS_GEMM, &gset, NULL);

    if (tailStatus & TAIL_A_RAISED) {
        addrMask |= FETCH_ADDR_A_RELATIVE;
    }
    if (tailStatus & TAIL_B_RAISED) {
        addrMask |= FETCH_ADDR_B_RELATIVE;
    }

    enableFetchOptLevels(mulOpts.fctx, FOPTLEV_MERGE_FETCHES);
    addrMode = setDefaultFetchAddrMode(mulOpts.fctx, &gset, addrMask,
                                       tailStatus, false);
    isRelA = ((addrMode & FETCH_ADDR_A_RELATIVE) != 0);
    isRelB = ((addrMode & FETCH_ADDR_B_RELATIVE) != 0);

    // Alternative calculate global thead id to eliminate Channel conflicts
    if (isRelB &&
        isMatrixAccessColMaj(CLBLAS_GEMM, gset.kextra->flags, MATRIX_B)) {

        sprintf(globalIdB, "get_global_id_%d", 1-i);
        sprintf(tmp,
                "uint kif;\n"
                "uint get_group_id_%d;\n"
                "uint get_global_id_%d;\n",1-i, 1-i);
        kgenAddStmt(ctx, tmp);
    }
    else {
        sprintf(globalIdB, "(uint)get_global_id(%d)", 1-i);
    }

    if (!(isColMajA || isColMajB)) {
        size_t tsize;

        tsize = dtypeSize(dtype);
        sprintf(tmp, "coord.z = (get_local_id(0) %% 2 * %lu) %% %s;\n",
                sizeof(cl_float8) / tsize, alignedK);
        kgenAddStmt(ctx, tmp);

        /*
         * Adjust fetch addressing mode. It is used staggered access. That
         * means there is a starting offset along K and hence addressing
         * in this dimension should be cycled.
         */
        addrMode &= ~FETCH_ADDR_K_RELATIVE;
        addrMode |= FETCH_ADDR_K_CYCLICAL;
        setFetchAddrMode(mulOpts.fctx, addrMode & ~addrMask);
    }

    if (isRelA) {
        genSetupItemPtr(ctx, &gset, MATRIX_A);
    }
    if (isRelB) {
        genSetupItemPtr(ctx, &gset, MATRIX_B);
    }

    /*
     * Setup coordinates and check if they don't exceed matrix
     */

    sprintf(tmp, "\n"
                 "coord.y = %luu * (uint)get_global_id(%d);\n"
                 "coord.x = %luu * (uint)%s;\n",
            subdims[1].y, i, subdims[1].x, globalIdB);
    kgenAddStmt(ctx, tmp);

    genHitMatrixCheck(ctx, kflags);
    genShiftPointers(ctx, &gset, kflags, true);
    genZeroTile(ctx, &gset.tileCY);

    tailStatus = checkGenAdjustTailCoords(ctx, CLBLAS_GEMM, &gset, NULL);

    mulOpts.core = ((kflags & KEXTRA_ENABLE_MAD) != 0)
            ? TILEMUL_MAD
            : TILEMUL_MULADD;

    mulOpts.flags |= TILEMUL_EXTERN_RDECL;
    mulOpts.flags |= kextraToTilemulFlags(CLBLAS_GEMM, kflags);

    sprintf(tmp, "for (uint k1 = 0; k1 < %s; k1 += %lu)",
            alignedK, subdims[1].bwidth);

    prepareFetchLoop(ctx, mulOpts.fctx, &gset, CLMEM_GLOBAL_MEMORY,
                     CLMEM_GLOBAL_MEMORY);

    kgenBeginBranch(ctx, tmp);
    ret = tileMulGen(ctx, &gset, &mulOpts);
    if (ret != 0) {
        goto out;
    }
    kgenEndBranch(ctx, NULL); // 0..K loop
    kgenAddBlankLine(ctx);

    //Optionally handle tails along K
    if (tailsK) {
        setDefaultFetchAddrMode(mulOpts.fctx, &gset, addrMask,
                                tailStatus, true);

        vnames->sizeK = "K";
        pfPriv.fetchNumA = 0;
        pfPriv.wholeA = 0;
        pfPriv.funcID = CLBLAS_GEMM;
        pfPriv.gset = &gset;
        mulOpts.postFetch = defaultTilePostFetch;
        mulOpts.postFetchPriv = &pfPriv;

        if (!(isColMajA || isColMajB)) {
            kgenAddStmt(ctx, "coord.z = Kbase;\n");
        }

        sprintf(tmp, "for (uint k1 = 0u; k1 < Ktail; k1 += %luu)",
                subdims[1].bwidth);
        kgenBeginBranch(ctx, tmp);
        ret = tileMulGen(ctx, &gset, &mulOpts);
        if (ret != 0) {
            goto out;
        }
        kgenEndBranch(ctx, NULL); // 0..Ktail loop
        kgenAddBlankLine(ctx);
    }

    gset.kextra = kextra;
    checkGenRestoreTailCoords(ctx, &gset, tailStatus);

    upFlags = kextraToUpresFlags(CLBLAS_GEMM, kflags);
    upFlags |= tailStatusToUpresFlags(tailStatus);
    upFlags |= UPRES_INDEXING_WITH_CONSTANTS;
    genResultUpdateWithFlags(ctx, CLBLAS_GEMM, &gset, upFlags,
                             NULL, NULL, NULL);

    kgenEndFuncBody(ctx);
    ret = kgenAddBlankLine(ctx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

out:
    destroyFetchContext(mulOpts.fctx);
    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

//-----------------------------------------------------------------------------

/*    the generator for subgroup access pattern
    (used when A and B matrices are accessed row-major)*/
static ssize_t
subgGen(
    char *pBuf,
    size_t buflen,
    const struct SubproblemDim *pSubDims,
    const struct PGranularity *pPGran,
    void *pExtra )
{
    struct KgenContext *pCtx;
    CLBLASKernExtra *pKExtra = (CLBLASKernExtra*)pExtra;
    KernelExtraFlags kflags = pKExtra->flags;
    DataType dtype = pKExtra->dtype;
    size_t staggered = ((extraData_t*)&pKExtra->solverPriv)->staggered;
    char tmp[2048];
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    ssize_t ret;
    FetchOpts fopts;
    TilePostFetchPrivate pfPriv;
    UpdateResultFlags upResFlags = 0;
    TailStatus tailStatus;
    FetchAddrMode addrMode;
    Kstring exprK;
    SubgVarNames subVNames;

    KernelVarNames *vnames = NULL;
    const char *alignedK;

    unsigned int vecLenA;

    bool isDoubleBased = isDoubleBasedType(dtype);

    bool tailsLowerK = ( (kflags & KEXTRA_TAILS_K_LOWER) != 0 );
    bool tailsM = ( (kflags & KEXTRA_TAILS_M) != 0 );
    bool tailsN = ( (kflags & KEXTRA_TAILS_N) != 0 );
    bool tailsLowerM = ( (kflags & KEXTRA_TAILS_M_LOWER) != 0 );
    bool tailsLowerN = ( (kflags & KEXTRA_TAILS_N_LOWER) != 0 );

    unsigned int subgroupsA = 0;
    unsigned int subgroupsB = 0;

    memset(&gset, 0, sizeof(gset));
    memset(&mulOpts, 0, sizeof(mulOpts));
    memset(&pfPriv, 0, sizeof(pfPriv));
    memset(&fopts, 0, sizeof(fopts));

    memcpy( gset.subdims, pSubDims, sizeof(gset.subdims) );
    gset.pgran  = pPGran;
    gset.flags  = BGF_DISTINCT_VECLEN | BGF_WHOLE_A | BGF_LD_IN_VECTORS;
    gset.kextra = pKExtra;

    vnames = &gset.varNames;
    // setting the basic names for kernel variables
    vnames->A = "A";
    vnames->B = "B";
    vnames->C = "C";
    vnames->LDS = "scratch";
    vnames->sizeM = "M";
    vnames->sizeN = "N";
    vnames->lda = "lda";
    vnames->ldb = "ldb";
    vnames->ldc = "ldc";

    vnames->alpha = "alpha";
    vnames->beta = "beta";

    vnames->vectCoordA = "vca";
    vnames->vectCoordB = "vcb";
    vnames->k = exprK.buf;

    subgroupsA = (unsigned int)(gset.subdims[0].y/gset.subdims[1].y);
    subgroupsB = (unsigned int)(gset.subdims[0].x/gset.subdims[1].x);

    initDefaultTiles(&gset, CLBLAS_GEMM, 0, PRIV_STORAGE_VARIABLE_SET);

    vecLenA = gset.tileA.vecLen;

    // channel offset based coordinate
    ksprintf(&exprK, "( (uint)(get_group_id(0))*%lu + k )", staggered/vecLenA*vecLenA);

    // starting code generation--------------------------------------------------
    pCtx = createKgenContext(pBuf, buflen, true);
    if ( pCtx == NULL) {
        return -ENOMEM;
    }

    //define required macros
    /* B_BLK_H should be one of common vector sizes,
    as matrix C is accessed by vectors of this length*/
    sprintf(tmp,"#define A_BLK_H %lu\n",gset.subdims[1].y);
    kgenAddStmt(pCtx,tmp);
    sprintf(tmp,"#define B_BLK_H %lu\n",gset.subdims[1].x);
    kgenAddStmt(pCtx,tmp);
    sprintf(tmp,"#define SUBG_ITEMS %d\n",pPGran->wgSize[0]);
    kgenAddStmt(pCtx,tmp);

    sprintf(tmp,"#define SUBG_A %d\n",subgroupsA);
    kgenAddStmt(pCtx,tmp);
    sprintf(tmp,"#define SUBG_B %d\n",subgroupsB);
    kgenAddStmt(pCtx,tmp);

    kgenAddBlankLine(pCtx);

    kgenAddStmt(pCtx,tmp);
    sprintf(
        tmp,
        "#define K_VLEN_A %u\n"
        "#define K_VLEN_B %u\n",
        getVecLen(&gset, CLBLAS_GEMM, MATRIX_A),
        getVecLen(&gset, CLBLAS_GEMM, MATRIX_B));

    kgenAddStmt(pCtx,tmp);
    kgenAddBlankLine(pCtx);

    // Declare pointer unions
    kgenDeclareUptrs(pCtx, isDoubleBased);
    kgenAddBlankLine(pCtx);

    // declaring kernel function
    declareKernel( pCtx, &gset, "Subgroup" );
    ret = kgenBeginFuncBody( pCtx );
    // kernel generation steps:

    // register variables declarations-----------------------------------------

    // K tail
    // if postfetch should be engaged, generate tail code for
    // whole subgroup, otherwise tail is handled by main cycle.
    if( tailsLowerK ){
        sprintf(tmp,
            "uint Ktail = K %% %lu;\n"
            "uint Kbase = K - Ktail;\n",
            pSubDims[0].bwidth);

        kgenAddStmt(pCtx, tmp);
        alignedK = "Kbase";
    }
    else {
        alignedK = "K";
    }
    vnames->sizeK = alignedK;

    declareTileStorages(pCtx, &gset);

    // scaling leading dims
    // If lower-K tails need to be handled, vectorized access is disabled
    // scaling is performed by factor 1
    sprintf(tmp, "%s /= K_VLEN_A;\n", vnames->lda);
    kgenAddStmt(pCtx, tmp);
    sprintf(tmp, "%s /= K_VLEN_B;\n", vnames->ldb);
    kgenAddStmt(pCtx, tmp);

    //declare variables for subgroup mode
    subVNames.itemId = "itemId";

    kgenAddBlankLine( pCtx );

    kgenPrintf( pCtx, "int2 %s;\n", subVNames.itemId );

    // item id
    kgenPrintf( pCtx,
        "%s.x = get_local_id(0);\n",
        subVNames.itemId );

    // subgroup id
    kgenPrintf( pCtx,
        "%s.y = get_local_id(1);\n",
        subVNames.itemId );

    kgenAddBlankLine( pCtx );

    // coordinate variables
    vnames->coordA = "coordY";
    vnames->coordB = "coordX";

    // generate offsets
    genShiftPointers( pCtx, &gset, kflags, true );

    // FIXME add new subgroup variables support
    sprintf(tmp, "int %s = "
                    "A_BLK_H*( "
                        "get_group_id(1)*SUBG_A + "
                        "get_local_id(1)/SUBG_B );\n",
            vnames->coordA);
    kgenAddStmt(pCtx, tmp);

    sprintf(tmp, "int %s = "
                    "B_BLK_H*( "
                        "get_group_id(0)*SUBG_B + "
                        "get_local_id(1)%%SUBG_B );\n",
            vnames->coordB);

    kgenAddStmt(pCtx, tmp);
    kgenAddBlankLine(pCtx);

    // Block M N tails. Drop excess blocks ------------------------------------
    kgenAddStmt(pCtx,"uint skipTileMul = 0;\n");
    //M
    if( tailsM ){

        kgenAddStmt(pCtx,"//M block tail\n");

        sprintf(tmp,
            "if( %s >= %s )",
            vnames->coordA,
            vnames->sizeM);

        kgenBeginBranch( pCtx,tmp );
        kgenAddStmt(pCtx,"skipTileMul = 1;\n");
        kgenEndBranch(pCtx,NULL);

    }

    //N
    if( tailsN ){

        kgenAddStmt(pCtx,"//N block tail\n");

        sprintf(tmp,
            "if( %s >= %s )",
            vnames->coordB,
            vnames->sizeN);

        kgenBeginBranch( pCtx,tmp );
        kgenAddStmt(pCtx,"skipTileMul = 1;\n");
        kgenEndBranch(pCtx,NULL);

    }
    kgenAddBlankLine(pCtx);

    //"Lower" tails
    if( tailsLowerM || tailsLowerN ){
        kgenAddStmt(pCtx, "//Raising \"Lower\" M N tails\n");
    }
    tailStatus = checkGenAdjustTailCoords(pCtx, CLBLAS_GEMM, &gset, NULL);

    // A, B pointers-----------------------------------------------------------

    sprintf(tmp,
            "A += %s*%s;\n",
            vnames->lda,
            vnames->coordA);

    kgenAddStmt(pCtx, tmp);

    sprintf(tmp,
        "B += %s*%s;\n",
        vnames->ldb,
        vnames->coordB);

    kgenAddStmt(pCtx, tmp);

    // calculated in vectors, C access is aligned to.
    // if row of C-block is splitted into smaller vectors -
    // multiply offset by number of these vectors

    kgenAddBlankLine(pCtx);

    genZeroTile( pCtx, &gset.tileCY );

    kgenAddBlankLine(pCtx);
    kgenAddBlankLine(pCtx);

    mulOpts.fctx = createFetchContext();
    if (mulOpts.fctx == NULL) {
        destroyKgenContext(pCtx);
        return -ENOMEM;
    }

    enableFetchOptLevels(mulOpts.fctx,
                         FOPTLEV_CAN_SHARE_TMP_AB);

    addrMode = setDefaultFetchAddrMode(mulOpts.fctx,
                                       &gset,
                                       FETCH_ADDR_K_RELATIVE,
                                       tailStatus,
                                       false);

    addrMode |= FETCH_ADDR_A_RELATIVE |
                FETCH_ADDR_B_RELATIVE |
                FETCH_ADDR_K_CYCLICAL;

    setFetchAddrMode(mulOpts.fctx, addrMode);
    prepareFetchLoop(pCtx,
                     mulOpts.fctx,
                     &gset,
                     CLMEM_GLOBAL_MEMORY,
                     CLMEM_GLOBAL_MEMORY);

    if( tailsM || tailsN ){
        kgenBeginBranch(pCtx,"if( !skipTileMul )");
    }

    sprintf(tmp,
            "for(int k = %u*get_local_id(0); k < %s; k += %u*SUBG_ITEMS)",
            vecLenA,
            alignedK,
            vecLenA);
    kgenBeginBranch( pCtx, tmp );

    // tiles multiplier--------------------------------------------------------

    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;

    mulOpts.core    = ((kflags & KEXTRA_ENABLE_MAD) != 0) ? TILEMUL_MAD :
                                                            TILEMUL_MULADD;

    mulOpts.flags = kextraToTilemulFlags( CLBLAS_GEMM, kflags );
    mulOpts.flags |= TILEMUL_EXTERN_RDECL;
    mulOpts.flags |= TILEMUL_NOT_INC_K;
    mulOpts.flags |= TILEMUL_BW_STRIDE;
    /* both matrices are accessed row - major */
    mulOpts.flags |= TILEMUL_TRB;

    ret = tileMulGen( pCtx, &gset, &mulOpts );
    if (ret != 0) {
        goto out;
    }

    kgenEndBranch(pCtx, NULL);
    kgenAddBlankLine(pCtx);

    // K - Tail
    if ( tailsLowerK ) {
        setFetchAddrMode(mulOpts.fctx, addrMode | FETCH_ADDR_TAILK_PADD);

        vnames->sizeK    = "K";
        vnames->k        = "k";

        kgenPrintf(pCtx,
                   "uint %s = %s + get_local_id(0)*%u;\n",
                   vnames->k,
                   alignedK,
                   vecLenA);

        pfPriv.fetchNumA = 0;
        pfPriv.wholeA = 0;
        pfPriv.funcID = CLBLAS_GEMM;
        pfPriv.gset = &gset;
        mulOpts.postFetch = defaultTilePostFetch;
        mulOpts.postFetchPriv = &pfPriv;

        kgenBeginBranch(pCtx, NULL);
        ret = tileMulGen(pCtx, &gset, &mulOpts);
        if (ret != 0) {
            goto out;
        }
        kgenEndBranch(pCtx, NULL);
    }

    if( tailsM || tailsN ){
        kgenEndBranch(pCtx, NULL);          // skip tilemul condition
    }
    kgenAddBlankLine(pCtx);

    upResFlags = kextraToUpresFlags(CLBLAS_GEMM, kflags) |
                 tailStatusToUpresFlags(tailStatus);
    // restore coordinates, if tail was raised
    checkGenRestoreTailCoords(pCtx, &gset, tailStatus);
    // merge and update result
    mergeUpdateResult( pCtx,
        CLBLAS_GEMM,
        &gset,
        &subVNames,
        upResFlags |
        UPRES_EXCEED_PROBLEM_CONDITION |
        UPRES_INDEXING_WITH_CONSTANTS,
        (UpresProcPtr)genResultUpdateWithFlags );
    kgenEndFuncBody(pCtx);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(pCtx) + 1;
    }

out:
    destroyFetchContext(mulOpts.fctx);
    destroyKgenContext(pCtx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

//-----------------------------------------------------------------------------

static void
assignBlockKargs(KernelArg *args, const void *params, const void *extra)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    int idx;
    (void)extra;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    initSizeKarg(&args[2], blasArgs->K);
    assignScalarKarg(&args[3], &(blasArgs->alpha), blasArgs->dtype);
    assignScalarKarg(&args[4], &(blasArgs->beta), blasArgs->dtype);
    INIT_KARG(&args[5], blasArgs->A);
    INIT_KARG(&args[6], blasArgs->B);
    INIT_KARG(&args[7], blasArgs->C);
    initSizeKarg(&args[8], blasArgs->lda.matrix);
    initSizeKarg(&args[9], blasArgs->ldb.matrix);
    initSizeKarg(&args[10], blasArgs->ldc.matrix);
    idx = 11;
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offBX);
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offCY);
    }
}

static bool
blockCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check)
{
    bool ret = true;
	bool ret_multiple = false;
	int i;

    DUMMY_ARG_USAGE(subdimsNum);

    if (check == PGRAN_CHECK) {
        unsigned int minSize, maxSize;

        maxSize = (dtype == TYPE_COMPLEX_DOUBLE) ? 4 : 8;
        minSize = (dtype == TYPE_COMPLEX_DOUBLE) ? 1 : 2;
        ret = decompSanityCheck(subdims, minSize, maxSize, 24, dtype, true);
        ret = ret && (subdims[0].bwidth == subdims[1].bwidth);
		for(i = 0; i < ( (pgran->maxWorkGroupSize) / (pgran->wfSize) ); i++)
		{
			// returns true if wgSize[0] * wgSize[1] is multiples of the 64 but not bigger than maxWorkGroupSize
			ret_multiple = ret_multiple || ( pgran->wgSize[0] * pgran->wgSize[1] == pgran->wfSize * (i + 1) );
		}
		ret = ret && ret_multiple;
    }
    else {
        calcPgranDedicated(pgran, subdims, 1, 3);
    }

    return ret;
}

//-----------------------------------------------------------------------------

static void
assignSubgKargs(KernelArg *args, const void *params, const void *extra)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    KernelExtraFlags kflags = ((const CLBLASKernExtra*)extra)->flags;
    int idx = 0;
    (void)extra;

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    initSizeKarg(&args[2], blasArgs->K);
    assignScalarKarg(&args[3], &(blasArgs->alpha), blasArgs->dtype);
    assignScalarKarg(&args[4], &(blasArgs->beta), blasArgs->dtype);
    INIT_KARG(&args[5], blasArgs->A);
    INIT_KARG(&args[6], blasArgs->B);
    INIT_KARG(&args[7], blasArgs->C);
    initSizeKarg(&args[8], blasArgs->lda.matrix);
    initSizeKarg(&args[9], blasArgs->ldb.matrix);
    initSizeKarg(&args[10], blasArgs->ldc.matrix);
    idx = 11;
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offBX);
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        initSizeKarg(&args[idx++], blasArgs->offCY);
    }

    return;
}

//-----------------------------------------------------------------------------

static DecompositionAxis
innerDecompositionAxis(const void *args)
{
    const CLBlasKargs *kargs = args;
    int tra, trb;

    tra = (kargs->order == clblasColumnMajor) ^
           (kargs->transA != clblasNoTrans);
    trb = (kargs->order == clblasRowMajor) ^
           (kargs->transB != clblasNoTrans);

    /*
     * Make the matrix B inner if every thread should access their elements
     * with a large stride but accesses elements of the matrix A sequentially
     * to provide more coalesced memory accesses.
     */
    return (!tra && trb) ? DECOMP_AXIS_X : DECOMP_AXIS_Y;
}

//-----------------------------------------------------------------------------

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_2D);
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
    if (off == 0) {
        extraData->staggered = roundUp(subdims[1].bwidth * sizeType[kargs->dtype]
                                    , wideChans / sizeType[kargs->dtype]);
    }
    else {
        extraData->staggered = 0;
    }
}

//-----------------------------------------------------------------------------

void
InitGEMMCachedBlockPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block gemm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &blockSOps;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

static int
blockGetPerf(
    unsigned int kflags,
    const void *args)
{
    (void)args;

    if( !isMatrixAccessColMaj( CLBLAS_GEMM, kflags, MATRIX_A ) &&
        !isMatrixAccessColMaj( CLBLAS_GEMM, kflags, MATRIX_B ) ){

        return PPERF_AVERAGE;
    }

    return PPERF_GOOD;
}

//-----------------------------------------------------------------------------

void
InitGEMMCachedSubgroupPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based subgroup gemm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &subgSOps;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

static int
gemmSubgGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs )
{
    DUMMY_ARG_USAGE(subdimsNum);
    pgran->wgDim = 2;
    return subgGetDefaultDecomp( pgran, subdims, pArgs );
}

//-----------------------------------------------------------------------------

static bool
subgCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check)
{
    unsigned int subgroupsA = 0;
    unsigned int subgroupsB = 0;
    unsigned int itemsPerSubg = 0;
    unsigned int regUse = 0;

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

    if( !(isDoubleBasedType(dtype) && isComplexType(dtype) ) ){

        if ( subdims[1].x < 2 || subdims[1].y < 2 || subdims[1].bwidth < 2 ) {

            return false;
        }
    }

    // check dimensions
    if( subdims[1].bwidth > 8 ||
        subdims[1].x > 8 ||
        subdims[1].y > 8 ){

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

    if( regUse >= 50 ){
        return false;
    }

    // validate the subgroup decomposition
    itemsPerSubg = subdims[0].bwidth/subdims[1].bwidth;

    subgroupsA = subdims[0].y/subdims[1].y;
    subgroupsB = subdims[0].x/subdims[1].x;

    // passed PGranularity should be checked
    if( PGRAN_CHECK == check ){

        if( pgran->wgSize[0] != itemsPerSubg ||
            pgran->wgSize[1] != subgroupsA*subgroupsB ){

            return false;
        }

        //filter subgroup numbers with poor performance
        //(less than 2 items in subgroup)
        if( pgran->wgSize[0] < 2 ){
            return false;
        }

        // drop groups consisting of number of items other than 64
        if( pgran->wgSize[0] * pgran->wgSize[1] != 64 ){
            return false;
        }
    }
    // PGranularity should be calculated
    else{
        pgran->wgSize[0] = itemsPerSubg;
        pgran->wgSize[1] = subgroupsA*subgroupsB;
    }

    pgran->wgDim = 2;

    /*Debug out for Tune*/

    return true;
}

//-----------------------------------------------------------------------------

static void
subgCalcGlobalThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra
)
{
    CLBlasKargs    *pArgs;

    //EINVAL
    if( NULL == subdims ||
        NULL == pgran ||
        NULL == args ||
        NULL == extra)
    {
        return;
    }
    pArgs = (CLBlasKargs*)args;

    threads[0] = (pArgs->N/subdims[0].x)*pgran->wgSize[0];
    threads[1] = (pArgs->M/subdims[0].y)*pgran->wgSize[1];

    // N tail group
    if( pArgs->N%subdims[0].x ){
        threads[0] += pgran->wgSize[0];
    }
    // M tail group
    if( pArgs->M%subdims[0].y ){
        threads[1] += pgran->wgSize[1];
    }
}

//-----------------------------------------------------------------------------
static int
subgGetPerf(
    unsigned int kflags,
    const void *args)
{
    DUMMY_ARG_USAGE(args);

    if( !isMatrixAccessColMaj( CLBLAS_GEMM, kflags, MATRIX_A ) &&
        !isMatrixAccessColMaj( CLBLAS_GEMM, kflags, MATRIX_B ) ){

        return PPERF_GOOD;
    }

    return PPERF_NOT_SUPPORTED;
}
