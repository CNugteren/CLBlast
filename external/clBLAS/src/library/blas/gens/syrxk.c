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


/**
 *  SYRk and SYR2K kernel generator
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <clBLAS.h>
#include <clblas_stddef.h>
#include <blas_mempat.h>
#include <solution_seq.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "init.h"
#include "blas_kgen.h"
#include "gen_helper.h"
#include "blas_subgroup.h"
#include "tile_iter.h"

/*
 * Priority within a statement batch of different kind
 * of statements consisting update around the diagonal.
 */
enum {
    CALC_COORDS_STMT_PRIO,
    FETCH_STMT_PRIO,
    MAD_STMT_PRIO,
    STORE_STMT_PRIO
};

enum {
    MAX_DIAG_UPRES_STORAGE_SIZE = 95,
    MAX_FETCH_CLAUSE_SIZE = 8
};

typedef struct {
    size_t staggered;
} extraData_t;

struct SetupPtrAttrs {
    MatrixRole mrole;
    const char *basePtr;
    const char *ldName;
    const char *offName;
    KernelExtraFlags offMask;
};

typedef struct SyrxkExtraPriv {
    unsigned int maxVlenC;
} MAY_ALIAS SyrxkExtraPriv;

static CLBLASMpatExtra mpatExtra;

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra,
   BlasFunctionID funcID);

static void
assignKargs(
    KernelArg *args,
    const CLBlasKargs *blasArgs,
    KernelExtraFlags kflags,
    BlasFunctionID funcID);

static void
syrkAssignKargs(KernelArg *args, const void *params, const void *extra);

static void
syr2kAssignKargs(KernelArg *args, const void *params, const void *extra);

static SolverFlags
solverFlags(void);

static void fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static bool
checkCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static void
syrkCalcThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);

static ssize_t
syrkGenerator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    return generator(buf, buflen, subdims, pgran, extra, CLBLAS_SYRK);
}

static ssize_t
syr2kGenerator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    return generator(buf, buflen, subdims, pgran, extra, CLBLAS_SYR2K);
}

static bool
subgCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static int
syrkSubgGetPerf(
    unsigned int kflags,
    const void *args);

static int
syrkSubgGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs );

static int
syrkBlockGetPerf(
    unsigned int kflags,
    const void *args);

#if 0
static int
syrkBlockGetDefaultDecomp(
        PGranularity *pgran,
        SubproblemDim *subdims,
        unsigned int subdimsNum);
#endif

// ----------------------------------------------------------------------------

static SolverOps syrkSolverOps = {
    syrkGenerator,
    syrkAssignKargs,
    NULL,
    syrkBlockGetPerf,
    NULL,
    syrkCalcThreads,
    NULL,
    solverFlags,
    fixupArgs,
    NULL,//getDefaultDecomp
	checkCalcDecomp,
	NULL,
	NULL
};

static SolverOps syr2kSolverOps = {
    syr2kGenerator,
    syr2kAssignKargs,
    NULL,
    syrkBlockGetPerf,
    NULL,
    syrkCalcThreads,
    NULL,
    solverFlags,
    fixupArgs,
    NULL,//getDefaultDecomp
   	checkCalcDecomp,
   	NULL,
   	NULL
};

static SolverOps syrkSubgSops = {
    syrkGenerator,
    syrkAssignKargs,
    NULL,
    syrkSubgGetPerf,
    NULL,
    syrkCalcThreads,
    NULL,
    solverFlags,
    fixupArgs,
    syrkSubgGetDefaultDecomp,
    subgCheckCalcDecomp,
	NULL,
	NULL
};

static SolverOps syr2kSubgSops = {
    syr2kGenerator,
    syr2kAssignKargs,
    NULL,
    syrkSubgGetPerf,
    NULL,
    syrkCalcThreads,
    NULL,
    solverFlags,
    fixupArgs,
    syrkSubgGetDefaultDecomp,
    subgCheckCalcDecomp,
   	NULL,
   	NULL
};

//-----------------------------------------------------------------------------

static void
genPanelBlocksStmt(
    struct KgenContext *ctx,
    const char *varName,
    int roundDir,
    const SubproblemDim *dim,
    const char *start,
    const char *end)
{
    char tmp[1024];
    char *p;

    p = tmp + sprintf(tmp, "%s = (%s", varName, end);
    if (start[0] != '\0') {
        p += sprintf(p, " - %s", start);
    }

    if (roundDir) {
        p += sprintf(p, " + %lu", dim->y - 1);
    }
    sprintf(p, ") / %lu;\n", dim->y);

    kgenAddStmt(ctx, tmp);
}

//-----------------------------------------------------------------------------

static void
genSetupPointers(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    FetchAddrMode addrMode,
    int rank)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    char dstPtr[64];
    const char *coordName;
    struct SetupPtrAttrs attrs[3] = {
        {MATRIX_A, "A", "lda", "offA", KEXTRA_A_OFF_NOT_ZERO},
        {MATRIX_B, "B", "ldb", "offB", KEXTRA_BX_OFF_NOT_ZERO},
        {MATRIX_C, "C", "ldc", "offC", KEXTRA_CY_OFF_NOT_ZERO}
    };
    int idx = 0;
    int i;
    Kstring k1, k2, k3;
    Kstring madExpr;
    unsigned int scale;
    unsigned int vecLen;
    FetchAddrMode relFlag;

    /*
     * Pointers are serviced in the following order:
     * B for tilemul, A for tilemul, C
     */
    for (i = 0; i < 3; i++) {
        // The output pointer should be shifted once in case of 2-rank update
        if ((i == 2) && rank) {
            break;
        }

        emptyKstring(&k1);
        emptyKstring(&k2);
        emptyKstring(&k3);
        scale = 0;

        // select start coordinate
        relFlag = (i) ? FETCH_ADDR_A_RELATIVE : FETCH_ADDR_B_RELATIVE;
        if (addrMode & relFlag) {
            coordName = (i) ? "coord.y" : "coord.x";
            kstrcpy(&k2, coordName);
        }

        // fill destination pointer to assign
        if (i == 2) {
            strcpy(dstPtr, "C");
        }
        else {
            const char *p;

            p = (i) ? gset->varNames.A : gset->varNames.B;
            strcpy(dstPtr, p);
        }

        // select index in the attribute array
        switch (i) {
        case 0:
            idx = (funcID == CLBLAS_SYRK) ? 0 : (1 - rank);
            break;
        case 1:
            idx = (funcID == CLBLAS_SYRK) ? 0 : rank;
            break;
        case 2:
            idx = 2;
            break;
        }

        vecLen = getVecLen(gset, funcID, attrs[idx].mrole);

        // construct expression
        if (attrs[idx].mrole != MATRIX_C) {
            if (isMatrixAccessColMaj(funcID, gset->kextra->flags,
                                     attrs[idx].mrole)) {

                kstrcpy(&k1, "1");
                scale = vecLen;
            }
            else {
                kstrcpy(&k1, attrs[idx].ldName);
            }
        }

        if (kextra->flags & attrs[idx].offMask) {
            if ((attrs[idx].mrole == MATRIX_C) || (vecLen == 1)) {
                kstrcpy(&k3, attrs[idx].offName);
            }
            else {
                int shift = findHighestSetBit(vecLen);

                ksprintf(&k3, "(%s >> %d)", attrs[idx].offName, shift);
            }
        }
        sprintfFastScalarMad(&madExpr, &k1, &k2, scale, &k3);

        // check if it is not "0" or empty string
        if (strlen(madExpr.buf) <= 1) {
            if (attrs[idx].mrole != MATRIX_C) {
                kgenPrintf(ctx, "%s = %s;\n", dstPtr, attrs[idx].basePtr);
            }
        }
        else {
            kgenPrintf(ctx, "%s = %s + %s;\n",
                       dstPtr, attrs[idx].basePtr, madExpr.buf);
        }
    }
}

//-----------------------------------------------------------------------------

static void
declareKernel(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    const char* nameSuffix )
{
    char tmp[1024], betaStr[64], bstr[64], strOffABC[256];
    DataType dtype = gset->kextra->dtype;
    KernelExtraFlags kflags = gset->kextra->flags;
    const PGranularity *pgran = gset->pgran;
    const char *tnameOrig, *tnameA;
    unsigned int vecLen;
    char fpref;
    const char *rank;

    tnameOrig = dtypeBuiltinType(dtype);
    vecLen = getVecLen(gset, funcID, MATRIX_A);
    getVectorTypeName(dtype, vecLen, &tnameA, NULL);
    fpref = dtypeToBlasPrefix(dtype);

    if (kflags & KEXTRA_BETA_ZERO) {
        betaStr[0] = '\0';
    }
    else {
        sprintf(betaStr, "    const %s beta,\n", tnameOrig);
    }

    if (funcID == CLBLAS_SYR2K) {
        const char *tnameB;

        rank = "2";
        vecLen = getVecLen(gset, funcID, MATRIX_B);
        getVectorTypeName(dtype, vecLen, &tnameB, NULL);
        sprintf(bstr, "    const __global %s *restrict B,\n"
                      "    uint ldb,\n",
                tnameB);
    }
    else {
        rank = "";
        bstr[0] = '\0';
    }

    strOffABC[0] = '\0';
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        strcpy(strOffABC, ",\n    uint offA");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        strcat(strOffABC, ",\n    uint offB");
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        strcat(strOffABC, ",\n    uint offC");
    }

    sprintf(tmp, "__attribute__((reqd_work_group_size(%u, 1, 1)))\n"
                 "void __kernel\n"
                 "%csyr%sk%s(\n"
                 "    uint N,\n"
                 "    const uint K,\n"
                 "    const %s alpha,\n"
                 "    const __global %s *restrict A,\n"
                 "    uint lda,\n"
                 "%s"   // B and ldb
                 "%s"   // beta
                 "    __global %s *C,\n"
                 "    uint ldc,\n"
                 "    const uint startN,\n"
                 "    const uint origN%s)\n",
            pgran->wgSize[0], fpref, rank, nameSuffix, tnameOrig, tnameA, bstr,
            betaStr, tnameOrig, strOffABC);

    kgenDeclareFunction(ctx, tmp);
}

//-----------------------------------------------------------------------------

static void
genHead(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    BlasFunctionID funcID,
    SubgVarNames *pSubgVNames,
    bool subgMode)
{
    char tmp[1024], tmp1[128];
    char start[128], end[128];
    char *p;
    const char *vecTypeA;
    unsigned int vlenA, vlenB;
    unsigned int l1Pans;
    const SubproblemDim *dim = gset->subdims;
    const CLBLASKernExtra *kextra = gset->kextra;
    KernelExtraFlags kflags = kextra->flags;
    KernelExtraFlags diagFlags = KEXTRA_SYRK_SEPARATE_DIAGONAL |
                                 KEXTRA_SYRK_EVALUATE_DIAGONAL;
    bool isDiagSep= ((kflags & KEXTRA_SYRK_SEPARATE_DIAGONAL) != 0);
    bool isEvalOnlyDiag = ((kflags & diagFlags) == diagFlags);

    l1Pans = (unsigned int)(dim[0].y / dim[1].y);

    vlenA = getVecLen(gset, funcID, MATRIX_A);
    vlenB = getVecLen(gset, funcID, MATRIX_B);
    getVectorTypeName(kextra->dtype, vlenA, &vecTypeA, NULL);

    // the variable stores N, passed as argument.
    // this variable is used for C matrix hit check
    kgenPrintf( ctx, "uint argN = N;\n" );

    if ( subgMode ) {

        gset->varNames.LDS = "scratch";

        // declaring variables used by subgroup mode
        pSubgVNames->itemId = "itemId";
        pSubgVNames->subgCoord = "subgCoord";

        kgenAddBlankLine( ctx );
        kgenAddBlankLine(ctx);

        kgenPrintf(ctx, "int skipTilemul = 0;\n" );
        kgenPrintf(ctx, "int2 %s;\n", pSubgVNames->itemId );
        kgenPrintf(ctx, "int2 %s;\n", pSubgVNames->subgCoord);

        // item ID
        kgenPrintf( ctx,
            "%s.x = get_local_id(0)%%%d;\n",
            pSubgVNames->itemId,
            dim[0].bwidth/dim[1].bwidth);

        // subgroup ID
        kgenPrintf( ctx,
            "%s.y = get_local_id(0)/%d;\n",
            pSubgVNames->itemId,
            dim[0].bwidth/dim[1].bwidth);

        // subgroup coordX
        kgenPrintf( ctx,
            "%s.x = %s.y/%d;\n",
            pSubgVNames->subgCoord,
            pSubgVNames->itemId,
            dim[0].y/dim[1].y );

        // subgroup coordY
        kgenPrintf( ctx,
            "%s.y = %s.y%%%d;\n",
            pSubgVNames->subgCoord,
            pSubgVNames->itemId,
            dim[0].y/dim[1].y );

    }

    if (funcID == CLBLAS_SYRK) {
        sprintf(tmp, "__global %s *B;\n", vecTypeA);
        kgenAddStmt(ctx, tmp);
    }

    if (kflags & KEXTRA_SYRK_2K_RANK) {
        const char *vecTypeB;

        getVectorTypeName(kextra->dtype, vlenB, &vecTypeB, NULL);
        sprintf(tmp, "__global %s *wiA;\n"
                     "__global %s *wiB;\n", vecTypeA, vecTypeB);
        kgenAddStmt(ctx, tmp);
    }

    kgenAddStmt(ctx, "uint4 coord = 0;\n"   /* contains coordB, coordA, k */
                     "uint k0 = 0;\n\n");

    // extra variables needed for the upper triangular case
    if ( kflags & KEXTRA_UPPER_TRIANG ) {
        if (kflags & KEXTRA_TAILS_N) {
            kgenAddStmt(ctx, "uint step;\n");
        }
        kgenAddStmt(ctx, "uint w;\n");
    }

    kgenAddStmt(ctx, "const int lid = get_local_id(0);\n"
                     "uint block = get_group_id(0);\n\n");

    /*
     * Increase/decrease the outer block coordinate while the inner block number
     * exceeds the number of blocks. Inner block number is counted from the
     * diagonal up to the matrix edge. A is always the inner matrix. It is from
     * the largest panel. The resulting block number determines starting
     * coordinates.
     *
     * In the case of separate evaluating of the area around the diagonal it's
     * critically important that at least on step would be aligned.
     * Otherwise, solution areas will overlap that will lead to a wrong result.
     */
    if ( kflags & KEXTRA_UPPER_TRIANG ) {
        char step[128], tmp2[128], *stepCalc = NULL;
        int roundDir;

        if ((kflags & KEXTRA_TAILS_N)) {
            sprintf(tmp2, "step = (coord.x %% %lu) ? (coord.x %% %lu) : %lu;\n",
                    dim[0].x, dim[0].x, dim[0].x);
            stepCalc = tmp2;
            sprintf(step, "step");
        }
        else {
            tmp2[0] = '\0';
            sprintf(step, "%lu", dim[0].x);
        }

        if (!isEvalOnlyDiag) {
            start[0] = '\0';
        }
        else {
            sprintf(start, "(coord.x - %s) / %lu * %lu",
                    step, dim[0].y, dim[0].y);
        }

        if (!isDiagSep || isEvalOnlyDiag) {
            strcpy(end, "coord.x");
            roundDir = 1;            // round up
        }
        else {
            sprintf(end, "(coord.x - %s) / %lu * %lu",
                    step, dim[0].y, dim[0].y);
            roundDir = 0;            // round down
        }

        if (!isEvalOnlyDiag) {
            kgenAddStmt(ctx, "coord.x = origN;\n");
            kgenAddStmt(ctx, stepCalc);
            sprintf(tmp, "w = (origN - startN - N + %lu) / %lu * %lu;\n"
                         "k0 = (N + %lu) / %lu;\n"
                         "if (block <= k0 * (w / %lu)) {\n"
                         "    coord.x -= (block / k0) * %lu;\n"
                         "    block %%= k0;\n"
                         "}\n",
                    dim[0].x - 1, dim[0].x, dim[0].x, dim[0].y - 1,
                    dim[0].y, dim[0].x, dim[0].x);
            kgenAddStmt(ctx, tmp);
            kgenBeginBranch(ctx, "else");
            sprintf(tmp, "coord.x = N;\n"
                         "block -= k0 * (w / %lu);\n",
                    dim[0].x);
            kgenAddStmt(ctx, tmp);
            kgenAddStmt(ctx, stepCalc);
        }
        else {
            kgenAddStmt(ctx, "coord.x = N;\n");
            kgenAddStmt(ctx, stepCalc);
        }

        if (isDiagSep) {
            genPanelBlocksStmt(ctx, "k0", roundDir, dim, start, end);
        }

        kgenBeginBranch(ctx, "while (block >= k0)");
        kgenAddStmt(ctx, "block -= k0;\n");
        sprintf(tmp, "coord.x -= %s;\n", step);
        kgenAddStmt(ctx, tmp);
        kgenAddStmt(ctx, stepCalc);
        genPanelBlocksStmt(ctx, "k0", roundDir, dim, start, end);
        kgenEndBranch(ctx, NULL);
        kgenAddStmt(ctx, "coord.x += startN;\n");

        if (!isEvalOnlyDiag) {
            kgenEndBranch(ctx, NULL);
        }

        if (isEvalOnlyDiag) {
            sprintf(tmp1, "%s", start);
            p = tmp1;
        }
        else {
            p = (char*)"startN";
        }

        if ( subgMode ) {

            kgenPrintf( ctx,
                "coord.y = %s + block * %lu + %s.y * %lu;\n",
                p,
                dim[0].y,
                pSubgVNames->subgCoord,
                dim[1].y );

            kgenPrintf( ctx,
                "coord.x = coord.x - %s + %s.x * %lu;\n",
                step,
                pSubgVNames->subgCoord,
                dim[1].x);

            kgenBeginBranch( ctx,
                "if (coord.y >= startN + argN || coord.x >= origN)");
            kgenPrintf( ctx, "skipTilemul = 1;\n" );
            kgenEndBranch( ctx, NULL );

            sprintf( tmp, "if (coord.y >= coord.x + %lu)", dim[1].x );
            kgenBeginBranch( ctx, tmp );
            kgenPrintf( ctx, "skipTilemul = 1;\n" );
            kgenEndBranch( ctx, NULL );

        }
        else {

            sprintf(tmp, "coord.y = %s + block * %lu + lid %% %u * %lu;\n"
                         "coord.x = coord.x - %s + lid / %u * %lu;\n"
                         "\n"
                         "if (coord.y >= startN + N || coord.x >= origN) {\n"
                         "    return;\n"
                         "}\n\n"
                         // Check if the tile is fully out of diagonal
                         "if (coord.y >= coord.x + %lu) {\n"
                         "    return;\n"
                         "}\n\n",
                    p, dim[0].y, l1Pans, dim[1].y,
                    step, l1Pans, dim[1].x,
                    dim[1].x);
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        int vecAlign = 1;

        if (!isDiagSep || isEvalOnlyDiag) {
            strcpy(start, "coord.x");
        }
        else {
            sprintf(start, "(coord.x + %lu) / %lu * %lu",
                    dim[0].x + dim[0].y - 1, dim[0].y, dim[0].y);
        }

        if (isEvalOnlyDiag) {
            sprintf(end, "(coord.x + %lu) / %lu * %lu",
                    dim[0].x + dim[0].y - 1, dim[0].y, dim[0].y);
        }
        else {
            vecAlign = umax(vlenA, vlenB);
            if (isMatrixAccessColMaj(funcID, kflags, MATRIX_A) &&
                (vecAlign > 1)) {

                sprintf(end, "(N + %u) / %u * %u",
                        vecAlign - 1, vecAlign, vecAlign);
            }
            else {
                strcpy(end, "N");
            }
        }

        if (!isEvalOnlyDiag) {
            sprintf(tmp, "k0 = (N + %lu) / %lu;\n"
                         "if (block < k0 * (startN / %lu)) {\n"
                         "    coord.x = (block / k0) * %lu;\n"
                         "    block %%= k0;\n"
                         "}\n",
                    dim[0].y - 1, dim[0].y, dim[0].x, dim[0].x);
            kgenAddStmt(ctx, tmp);
            kgenBeginBranch(ctx, "else");
            sprintf(tmp, "block -= k0 * (startN / %lu);\n", dim[0].x);
            kgenAddStmt(ctx, tmp);
        }

        if (isDiagSep) {
            genPanelBlocksStmt(ctx, "k0", 1, dim, start, end);
        }

        kgenBeginBranch(ctx, "while (block >= k0)");
        sprintf(tmp, "block -= k0;\n"
                     "coord.x += %lu;\n",
                dim[0].x);
        kgenAddStmt(ctx, tmp);
        genPanelBlocksStmt(ctx, "k0", 1, dim, start, end);
        kgenEndBranch(ctx, NULL);
        kgenAddStmt(ctx, "coord.x += startN;\n");

        if (!isEvalOnlyDiag) {
            kgenEndBranch(ctx, NULL);
        }

        if (!isDiagSep && (kflags & KEXTRA_TAILS_M)) {
            sprintf(tmp, "coord.y = (%s >= startN + N %% %lu) ? "
                                "(N - (block + 1) * %lu) : "
                                "(N - N %% %lu - block * %lu);\n",
                    start, dim[0].y, dim[0].y, dim[0].y, dim[0].y);
        }
        else if ((isDiagSep && !isEvalOnlyDiag) && (kflags & KEXTRA_TAILS_M)) {
            sprintf(tmp, "coord.y = (N - N %% %lu - block * %lu);\n",
                    dim[0].y, dim[0].y);
        }
        else {
            sprintf(tmp, "coord.y = %s - (block + 1) * %lu;\n", end, dim[0].y);
        }
        kgenAddStmt(ctx, tmp);

        if (isMatrixAccessColMaj(funcID, kflags, MATRIX_A) && (vecAlign > 1)) {
            sprintf(tmp, "coord.y = (coord.y + %u) / %u * %u;\n",
                    vecAlign - 1, vecAlign, vecAlign);
            kgenAddStmt(ctx, tmp);
        }

        if ( subgMode ) {

            kgenPrintf( ctx,
                "coord.y += startN + %s.y * %lu;\n",
                pSubgVNames->subgCoord,
                dim[1].y );

            kgenPrintf( ctx,
                "coord.x += %s.x * %lu;\n",
                pSubgVNames->subgCoord,
                dim[1].x );

            kgenBeginBranch( ctx,
                "if (coord.y >= startN + argN || coord.x >= startN + argN)" );
            kgenPrintf( ctx, "skipTilemul = 1;\n" );
            kgenEndBranch( ctx, NULL );

            sprintf( tmp, "if (coord.x >= coord.y + %lu)", dim[1].y );
            kgenBeginBranch( ctx, tmp );
            kgenPrintf( ctx, "skipTilemul = 1;\n" );
            kgenEndBranch( ctx, NULL );
        }
        else {

            sprintf(tmp, "coord.y += startN + lid %% %u * %lu;\n",
                    l1Pans, dim[1].y);
            kgenAddStmt(ctx, tmp);

            sprintf(tmp, "coord.x += lid / %u * %lu;\n"
                         "if (coord.y >= startN + N || coord.x >= startN + N) {\n"
                         "    return;\n"
                         "}\n"
                         // check if the tile is fully out of the diagonal
                         "if (coord.x >= coord.y + %lu) {\n"
                         "    return;\n"
                         "}\n\n",
                    l1Pans, dim[1].x, dim[1].y);
            kgenAddStmt(ctx, tmp);

        }
    }

    kgenAddBlankLine(ctx);
}

//-----------------------------------------------------------------------------

static void
declareComplexMults(
    struct KgenContext *ctx,
    DataType dtype,
    UpdateResultFlags uflags)
{
    const char *tname;

    if (isComplexType(dtype)) {
        tname = dtypeBuiltinType(dtype);
        declareComplexMultParts(ctx, "alpha", tname);
        if (uflags & UPRES_WITH_BETA) {
            declareComplexMultParts(ctx, "beta", tname);
        }
    }
}

//-----------------------------------------------------------------------------

static void
genUpdateSingleOptimized(
    struct StatementBatch *batch,
    const BlasGenSettings *gset,
    const Kstring *tempC,
    const Kstring *result,
    const Kstring *complexOpTmp)
{
    const char *alphaName;
    const char *betaName;
    bool useMad;
    const CLBLASKernExtra *kextra = gset->kextra;

    alphaName = gset->varNames.alpha;
    betaName = (kextra->flags & KEXTRA_BETA_ZERO) ?
                    NULL : gset->varNames.beta;
    useMad = (kextra->flags & KEXTRA_ENABLE_MAD) != 0;

    if (isComplexType(kextra->dtype)) {
        TileMulCore core;
        Kstring expr;
        Kstring alphaStr;
        const Kstring *k3;
        bool isDouble;

        isDouble = isDoubleBasedType(kextra->dtype);
        core = (useMad) ? TILEMUL_MAD : TILEMUL_MULADD;
        kstrcpy(&alphaStr, alphaName);
        k3 = ((betaName != NULL) && (core == TILEMUL_MAD) && complexOpTmp) ?
                        complexOpTmp : tempC;

        if (betaName != NULL) {
            Kstring betaStr;

            kstrcpy(&betaStr, betaName);
            sprintfComplexMulUpdate(&expr, k3, tempC, &betaStr, NULL,
                                    isDouble, false, false, core);
            kgenAddStmtToBatch(batch, MAD_STMT_PRIO, expr.buf);
			
			sprintfComplexMulUpdate(&expr, tempC, result, &alphaStr, k3,
									isDouble, false, false, core);
			kgenAddStmtToBatch(batch, MAD_STMT_PRIO, expr.buf);
        }
		else
		{
			//fix correctness bug for c/z syr2k when beta = (0,0)
			sprintfComplexMulUpdate_syr2k_beta0(&expr, tempC, result, &alphaStr, NULL,
									isDouble, false, false, core);
			kgenAddStmtToBatch(batch, MAD_STMT_PRIO, expr.buf);
		}

    }
    else {
        if (betaName != NULL) {
            if (useMad) {
                kgenBatchPrintf(batch, MAD_STMT_PRIO,
                                "%s = mad(%s, %s, 0);\n"
                                "%s = mad(%s, %s, %s);\n",
                                tempC->buf, tempC->buf, betaName,
                                tempC->buf, result->buf, alphaName, tempC->buf);
            }
            else {
                kgenBatchPrintf(batch, MAD_STMT_PRIO,
                                "%s = %s * %s + %s * %s;\n",
                                tempC->buf, result->buf, alphaName,
                                tempC->buf, betaName);
            }
        }
        else {
            if (useMad) {
                kgenBatchPrintf(batch, MAD_STMT_PRIO,
                                "%s = mad(%s, %s, 0);\n",
                                tempC->buf, result->buf, alphaName);
            }
            else {
                kgenBatchPrintf(batch, MAD_STMT_PRIO,
                                "%s = %s * %s;\n",
                                tempC->buf, result->buf, alphaName);
            }
        }
    }
}

//-----------------------------------------------------------------------------

// Init temporary file for diagonal result update
static void
initTmpResTile(Tile *tile, const BlasGenSettings *gset, bool forceNoTrans)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    bool cmaj = ((kflags & KEXTRA_COLUMN_MAJOR) != 0) && !forceNoTrans;
    const Tile *tc = &gset->tileCY;

    memcpy(tile, tc, sizeof(Tile));

    if (!(kflags & KEXTRA_BETA_ZERO)) {
        unsigned int maxTmpSize;
        unsigned int pitch;

        maxTmpSize = tileStorageSize(&gset->tileA) +
                     tileStorageSize(&gset->tileBX);

        tile->baseName = "tempC";
        tile->vecLen = getVecLen(gset, CLBLAS_SYRK, MATRIX_C);
        tile->trans = cmaj;
        pitch = (cmaj) ? tile->nrRows : tile->nrCols;
        tile->vecLen = (unsigned int)roundDownPow2(pitch);
        tile->vecLen = umin(tile->vecLen, MAX_TILE_VECLEN);

        /*
         * restrict number of rows or columns of the new tile according
         * to the maximum tile size evaluated above
         */
        if (cmaj) {
            pitch = (unsigned int)roundUp(tile->nrRows, tile->vecLen);
            tile->nrCols = umin(maxTmpSize / pitch, tile->nrCols);
            tile->nrCols = (unsigned int)roundDownPow2(tile->nrCols);
        }
        else {
            pitch = (unsigned int)roundUp(tile->nrCols, tile->vecLen);
            tile->nrRows = umin(maxTmpSize / pitch, tile->nrRows);
            tile->nrRows = (unsigned int)roundDownPow2(tile->nrRows);
        }
    }
}

//-----------------------------------------------------------------------------

// Declare and setup pointer to the start of updated outpu tile
const char
*declareSetupOutputPtr(struct KgenContext *ctx, const BlasGenSettings *gset)
{
    const KernelVarNames *kvars = &gset->varNames;
    const char *coords[2] = {kvars->coordA, kvars->coordB};
    const char *tname;
    int cmaj;

    tname = dtypeBuiltinType(gset->kextra->dtype);
    cmaj = ((gset->kextra->flags & KEXTRA_COLUMN_MAJOR) != 0);

    kgenPrintf(ctx, "__global %s *dst = %s + %s * %s + %s;\n\n",
               tname, kvars->C, coords[cmaj], kvars->ldc, coords[1 - cmaj]);

    return "dst";
}

//-----------------------------------------------------------------------------

/*
 * Check if an additional temporary variable is need for updating complex
 * result. It is needed if using "mad" buit-in OpenCL functions because
 * a single operation is evaluated with 2 statements. Without that the result
 * part evaluated with the first statement is used as an input argument
 * in the second one that leads to wrong evaluation.  Declare and put its
 * name to the passed string if it's really needed or just empty the
 * string otherwise
 */
static void
checkDeclareUpcomTmp(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    Kstring *kstr)
{
    DataType dtype = gset->kextra->dtype;
    const char *tname;

    if (isComplexType(dtype) &&
        (gset->kextra->flags & KEXTRA_ENABLE_MAD)) {

        tname = dtypeBuiltinType(dtype);
        kgenPrintf(ctx, "%s sctmp;\n", tname);
        kstrcpy(kstr, "sctmp");
    }
    else {
        emptyKstring(kstr);
    }
}

//-----------------------------------------------------------------------------

// Declare set of variables differing with trailing index
static void
declareDiagUpresIndexedVars(
    struct KgenContext *ctx,
    const char *type,
    const char *baseName,
    unsigned int nrVars)
{
    Kstring kstr;
    unsigned int i;

    ksprintf(&kstr, "%s %s0", type, baseName);
    for (i = 1; i < nrVars; i++) {
           kstrcatf(&kstr, ", %s%u", baseName, i);
    }
    kstrcatf(&kstr, ";\n");
    kgenAddStmt(ctx, kstr.buf);
}

//-----------------------------------------------------------------------------

/*
 * Add blank line for each diagonal update statement priority
 * to make the code more readable
 */
static void
addDiagUpdateBlanks(struct StatementBatch *batch)
{
    kgenAddStmtToBatch(batch, FETCH_STMT_PRIO, "\n");
    kgenAddStmtToBatch(batch, MAD_STMT_PRIO, "\n");
    kgenAddStmtToBatch(batch, STORE_STMT_PRIO, "\n");
}

//-----------------------------------------------------------------------------

/*
 * The function update result around the diagonal in case of
 * 'y' and 'x' subdimensions equal at the tile level, and not
 * having tails along those subdimensions.
 */
static int
genUpdateIsoscelesDiagTile(
    struct KgenContext *ctx,
    const BlasGenSettings *gset)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    DataType dtype = gset->kextra->dtype;
    struct StatementBatch *batch;
    PhysTileIterator iter;
    unsigned int vlen;
    const Tile *tileC = &gset->tileCY;
    Tile tileTempC;
    bool isPhysUpper;
    bool isHit;
    bool withBeta;
    bool cmaj;
    unsigned int nrStored;
    unsigned int skipCnt = 0;
    const char *glbType;
    const char *dstPtr;
    Kstring tempElem, resElem;
    Kstring k1, k2, ldcName;
    Kstring comTmp;
    const Kstring *ptmp;
    Kstring offExpr;
    unsigned int tempRows, tempCols;
    unsigned int madLen;

    batch = createStmtBatch();
    if (batch == NULL) {
        return -ENOMEM;
    }

    cmaj = (kflags & KEXTRA_COLUMN_MAJOR) != 0;
    isPhysUpper = ((kflags & KEXTRA_UPPER_TRIANG) != 0) ^ cmaj;
    withBeta = !(kflags & KEXTRA_BETA_ZERO);

    iterInit(&iter, tileC, 1, 0);
    vlen = getVecLen(gset, CLBLAS_SYRK, MATRIX_C);
    kstrcpy(&ldcName, gset->varNames.ldc);

    initTmpResTile(&tileTempC, gset, false);
    tempRows = tileTempC.nrRows;
    tempCols = tileTempC.nrCols;

    // declare and initialize needed variables
    dstPtr = declareSetupOutputPtr(ctx, gset);
    checkDeclareUpcomTmp(ctx, gset, &comTmp);
    ptmp = (isKstringEmpty(&comTmp)) ? NULL : &comTmp;
    if (tileTempC.baseName != tileC->baseName) {
        declareOneTileStorage(ctx, &tileTempC);
        kgenAddBlankLine(ctx);
    }

    while (!iterIsEnd(&iter)) {
        if (!(iter.row % tempRows ||
              iter.col % tempCols)) {

            addDiagUpdateBlanks(batch);
            flushStmtBatch(ctx, batch);
        }

        isHit = (isPhysUpper) ? (iter.vec >= iter.line) :
                                (iter.line >= iter.vec);

        skipCnt = (skipCnt) ? (skipCnt - 1) : 0;
        if (!isHit) {
            iterIterate(&iter);
            continue;
        }

        if (skipCnt) {
            nrStored = 0;
        }
        else if (isPhysUpper) {
            if (iter.vec && !isRoundedPow2(iter.vec)) {
                size_t s = iter.vec;

                s = szmin(roundUpPow2(s) - s, s - roundDownPow2(s));
                nrStored = (unsigned int)s;
            }
            else {
                nrStored = (iter.vec) ? umin(iter.vec, iter.nrVecs - iter.vec) :
                                        (unsigned int)iter.nrVecs;
            }
        }
        else {
            nrStored = (unsigned int)roundDownPow2(iter.line - iter.vec + 1);
        }

        nrStored = umin(nrStored, vlen);
        skipCnt = umax(skipCnt, nrStored);

        if (nrStored) {
            getVectorTypeName(dtype, nrStored, &glbType, NULL);
            ksprintf(&k1, "%u", iter.line);
            ksprintf(&k2, "%u", iter.vec);
            sprintfFastScalarMad(&offExpr, &k1, &ldcName, 0, &k2);

            if (withBeta) {
                sprintfTileElement(&tempElem, &tileTempC, iter.row % tempRows,
                                   iter.col % tempCols, nrStored);

                kgenBatchPrintf(batch, FETCH_STMT_PRIO,
                                "%s = *(__global %s*)(&%s[%s]);\n",
                                tempElem.buf, glbType, dstPtr, offExpr.buf);
            }
        }

        madLen = (isComplexType(dtype) || (tileC->trans != cmaj)) ?
                        1 : nrStored;
        if (madLen) {
            sprintfTileElement(&tempElem, &tileTempC, iter.row % tempRows,
                               iter.col % tempCols, madLen);
            sprintfTileElement(&resElem, tileC, iter.row, iter.col, madLen);
            genUpdateSingleOptimized(batch, gset, &tempElem, &resElem, ptmp);
        }

        if (nrStored) {
            sprintfTileElement(&tempElem, &tileTempC, iter.row % tempRows,
                               iter.col % tempCols, nrStored);
            kgenBatchPrintf(batch, STORE_STMT_PRIO,
                            "*(__global %s*)(&%s[%s]) = %s;\n",
                            glbType, dstPtr, offExpr.buf, tempElem.buf);
        }

        iterIterate(&iter);
    }

    addDiagUpdateBlanks(batch);
    flushStmtBatch(ctx, batch);
    destroyStmtBatch(batch);

    return 0;
}

//-----------------------------------------------------------------------------

/*
 * Update diagonal tile of arbitrary shape in case of not having tails
 * along 'x' and 'y' subdimensions at the tile level.
 */
static int
genUpdateGenericDiagTile(
    struct KgenContext *ctx,
    const BlasGenSettings *gset)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    DataType dtype = gset->kextra->dtype;
    const char *typeName;
    struct StatementBatch *batch;
    PhysTileIterator iter;
    TileIterFlags tifl;
    BlasGenSettings gsetNew;
    const Tile *tileC = &gset->tileCY;
    Tile tileTempC;
    bool withBeta;
    bool isUpper;
    const char *dstPtr;
    const char *s;
    Kstring tempElem, resElem;
    Kstring comTmp;
    const Kstring *ptmp;
    Kstring kstr, alphaStr, betaStr;
    unsigned int nrRows, nrCols;
    unsigned int tempRows;
    // type of the vectorized coordinates
    Kstring vctype;
    Kstring constOffs, constShifts, constMasks;
    unsigned int i, j, nops,size;
    unsigned int maxFetches = 0;
    const char *yname, *xname;
    const char *ldcName;
	char hexadec[2];

    batch = createStmtBatch();
    if (batch == NULL) {
        return -ENOMEM;
    }

    typeName = dtypeBuiltinType(dtype);

    nrRows = tileC->nrRows;
    nrCols = tileC->nrCols;
    withBeta = !(kflags & KEXTRA_BETA_ZERO);
    isUpper = ((kflags & KEXTRA_UPPER_TRIANG) != 0);

    yname = gset->varNames.coordA;
    xname = gset->varNames.coordB;
    ldcName = gset->varNames.ldc;

    memcpy(&gsetNew, gset, sizeof(BlasGenSettings));

    /*
     * Fetches are done by single element. Non transposed shape
     * is forced to facilitate further size restriction and tile
     * manipulation
     */
    memcpy(&tileTempC, tileC, sizeof(Tile));
    tileTempC.trans = false;

    tifl = (isUpper) ? TILE_ITER_BACKWARD_ROWS :
                       TILE_ITER_BACKWARD_COLS;
    iterInit(&iter, &tileTempC, 1, tifl);
	nops = 0;
	while (!iterIsEnd(&iter)) {
		nops++;
		size = nops / nrCols;
		iterIterate(&iter);
	}

	iterInit(&iter, &tileTempC, 1, tifl);

    initTmpResTile(&tileTempC, gset, true);

    if (nrCols == 1) {
        kstrcpy(&vctype, "uint");
    }
    else {
        ksprintf(&vctype, "uint%u", nrCols);
    }

    /*
     * fill constant offsets, shifts and masks within each line
     * for vectorized coorinates
     */
    ksprintf(&constOffs, "(%s)(", vctype.buf);
    ksprintf(&constShifts, "(%s)(", vctype.buf);
    ksprintf(&constMasks, "(%s)(", vctype.buf);
    for (i = 0; i < nrCols; i++) {
        s = (i == nrCols - 1) ? "" : ", ";

        j = (isUpper) ? (nrCols - i - 1) : i;
        kstrcatf(&constOffs, "%uu%s", j, s);
        kstrcatf(&constShifts, "%uu%s", i, s);
        kstrcatf(&constMasks, "%#x%s", 1 << i, s);
    }
    kstrcatf(&constOffs, ")");
    kstrcatf(&constShifts, ")");
    kstrcatf(&constMasks, ")");


    // declare and initialize needed variables

    dstPtr = declareSetupOutputPtr(ctx, gset);
    checkDeclareUpcomTmp(ctx, gset, &comTmp);
    ptmp = (isKstringEmpty(&comTmp)) ? NULL : &comTmp;

    if (tileTempC.baseName != tileC->baseName) {
        /*
         * Make additional temporary tile size restrition because of the
         * following factors:
         *
         * No more than 16 fetches can be combined into single clause.
         * So, there is no need to maintain larger temporary tile as well
         * as more vector coordinates to reduce number of consumed registers.
         * However, actually, the compiler pains even 16 fetches merged into
         * single clause and allocate huge number of registers.
         */
        if (tileStorageSize(&tileTempC) > MAX_FETCH_CLAUSE_SIZE) {
            tileTempC.nrRows = (unsigned int)roundDownPow2(
                                        MAX_FETCH_CLAUSE_SIZE / nrCols);
            if (!tileTempC.nrRows) {
                tileTempC.nrRows = 1;
            }
        }
    }

    tempRows = tileTempC.nrRows;
    maxFetches = MAX_FETCH_CLAUSE_SIZE / nrCols * nrCols;
    maxFetches = umin(maxFetches, tempRows * nrCols);
    i = tileStorageSize(&tileTempC);
    maxFetches = umin(maxFetches, i);

    // declare vectorized coordinates
    declareDiagUpresIndexedVars(ctx, vctype.buf, "cc", size);

    /*
     * real y coordinate, offset mask and
     * substituted beta and alpha (one value per temporary line)
     */
    kgenAddStmt(ctx, "unsigned int ry;\n"
                     "unsigned int mask;\n"
                     "int hit;\n");
    if (withBeta) {
        declareDiagUpresIndexedVars(ctx, typeName, "alphaNew", size);
        declareDiagUpresIndexedVars(ctx, typeName, "betaNew", size);
    }

    // declare tile
    if (tileTempC.baseName != gset->tileCY.baseName) {
        declareOneTileStorage(ctx, &tileTempC);
        kgenAddBlankLine(ctx);
    }

    // set start mask value
    if (isUpper) {
        kgenPrintf(ctx, "if (%s + %u <= %s) {\n"
                        "    mask = ~0;\n"
                        "}\n"
                        "else {\n"
                        "    mask = (%s + %u < %s + %u) "
                        "    ? ~((1 << (%s + %u - %s)) - 1) : 0;\n"
                        "}\n\n",
                   yname, nrRows - 1, xname,
                   yname, nrRows - 1, xname, nrCols - 1,
                   yname, nrRows, xname);
    }
    else {
        kgenPrintf(ctx, "if (%s + %u <= %s) {\n"
                        "    mask = ~0;\n"
                        "}\n"
                        "else {\n"
                        "    mask = (%s > %s) ? ((1 << (%s - %s)) - 1) : 0;\n"
                        "}\n\n",
                   xname, nrCols - 1, yname,
                   yname, xname, yname, xname);
    }

    // let's go
    nops = 0;
    while (!iterIsEnd(&iter)) {
        if (nops == maxFetches) {
            addDiagUpdateBlanks(batch);
            flushStmtBatch(ctx, batch);
            nops = 0;
        }

        // index for all temporary coordinates
        i = nops / nrCols;

        // prepare vectorized coordinates for the next line
        if (nops % tileTempC.nrCols == 0) {
            if (isUpper) {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,
                                "hit = (%s + %u <= %s + %u);\n",
                                yname, iter.row, xname, nrCols - 1);
            }
            else {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,
                                "hit = (%s + %u >= %s);\n",
                                yname, iter.row, xname);
            }

            if (withBeta) {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,
                                "betaNew%u = (hit) ? %s : %s;\n"
                                "alphaNew%u = (hit) ? %s : (%s)0;\n",
                                i, gset->varNames.beta, strOne(dtype),
                                i, gset->varNames.alpha, typeName);
            }

            if (isUpper) {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,

                                "ry = select(0, %u, hit);\n"
                                "mask = select(mask, mask >> 1 | %#x, hit);\n"
                                "cc%u = ((%s)mask &\n"
                                "       %s) >>\n"
                                "      %s;\n"
                                "cc%u = %u - mad24(cc%u, %s, 0u);\n",

                                iter.row,
                                (1 << (nrCols - 1)),
                                i, vctype.buf, constMasks.buf, constShifts.buf,
                                i, nrCols - 1, i, constOffs.buf);
            }
            else {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,

                                "ry = select(%u, %u, hit);\n"
                                "mask = select(mask, mask << 1 | 1, hit);\n"
                                "cc%u = ((%s)mask &\n"
                                "       %s) >>\n"
                                "      %s;\n"
                                "cc%u = mad24(cc%u, %s, 0u);\n",

                                nrRows - 1, iter.row,
                                i, vctype.buf, constMasks.buf, constShifts.buf,
                                i, i, constOffs.buf);
            }

            if (kflags & KEXTRA_COLUMN_MAJOR) {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,
                                "cc%u = mad24(cc%u, (%s)%s, (%s)ry);\n\n",
                                i, i, vctype.buf, ldcName, vctype.buf);
            }
            else {
                kgenBatchPrintf(batch, CALC_COORDS_STMT_PRIO,
                                "cc%u = mad24((%s)ry, (%s)%s, cc%u);\n\n",
                                i, vctype.buf, vctype.buf, ldcName, i);
            }
        }

        // prepare for the immediate update
        sprintfTileElement(&tempElem, &tileTempC,
                           iter.row % tempRows, iter.col, 1);
        sprintfTileElement(&resElem, tileC, iter.row, iter.col, 1);
        if (nrCols == 1) {
            ksprintf(&kstr, "cc%u", i);
        }
        else {
			snprintf(hexadec, sizeof(char)*2, "%x", iter.col);
			//itoa(iter.col, hexadec, 16);
            ksprintf(&kstr, "cc%u.s%s", i, hexadec);
        }

        // prepare multipliers and fetch
        if (withBeta) {
            ksprintf(&alphaStr, "alphaNew%u", i);
            ksprintf(&betaStr, "betaNew%u", i);
            gsetNew.varNames.alpha = alphaStr.buf;
            gsetNew.varNames.beta = betaStr.buf;

            kgenBatchPrintf(batch, FETCH_STMT_PRIO, "%s = %s[%s];\n",
                            tempElem.buf, dstPtr, kstr.buf);
        }

        genUpdateSingleOptimized(batch, &gsetNew, &tempElem, &resElem, ptmp);


        // store
        kgenBatchPrintf(batch, STORE_STMT_PRIO, "%s[%s] = %s;\n",
                        dstPtr, kstr.buf, tempElem.buf);

        nops++;
        iterIterate(&iter);
    }

    addDiagUpdateBlanks(batch);
    flushStmtBatch(ctx, batch);
    destroyStmtBatch(batch);

    return 0;
}

//-----------------------------------------------------------------------------

static int
genUpdateTailedDiagTile(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    UpdateResultFlags uflags)
{
    char tmp[1024];
    char s1[1024], s2[256];
    char src[32], dst[32];
    char *p;
    const char *vfield;
    size_t pitch;
    struct KgenContext *ctx1;
    const CLBLASKernExtra *kextra = gset->kextra;
    DataType dtype = kextra->dtype;
    KernelExtraFlags kflags = kextra->flags;
    const SubproblemDim *dims = gset->subdims;
    UpdateResultOp op;
    /*
     * solution tile coordinate without consideration of
     * row/column order
     */
    const char *trow, *tcol, *s3, *s4;

    vfield = dtypeUPtrField(dtype);
    pitch = roundUp(gset->tileCY.nrCols, gset->tileCY.vecLen);

    tcol = gset->varNames.coordB;
    trow = gset->varNames.coordA;

    s3 = (kflags & KEXTRA_COLUMN_MAJOR) ? tcol : trow;
    s4 = (kflags & KEXTRA_COLUMN_MAJOR) ? trow : tcol;

    // declare and initialize variables
    sprintf(s1, "uint m = min(%luu, N - %s);\n"
                "uint n = min(%luu, N - %s);\n",
            dims[1].y, trow, dims[1].x, tcol);

    p = s1 + strlen(s1);
    sprintf(p, "uint i, j, j0;\n"
                "PPtr res;\n"
                "GPtr uC;\n"
                "\n"
                "res.%s = c;\n"
                "uC.%s = C + %s * ldc + %s;\n",
            vfield, vfield, s3, s4);

    if (uflags & (UPRES_TAIL_ROW | UPRES_TAIL_COL)) {
        char offStr[64];
        char *p = offStr;

        offStr[0] = '\0';
        if (uflags & UPRES_TAIL_ROW) {
            sprintf(offStr, " + (%lu - m) * %lu", dims[1].y, pitch);
            p += strlen(offStr);
        }
        if (uflags & UPRES_TAIL_COL) {
            sprintf(p, " + (%lu - n)", dims[1].x);
        }

        p = s1 + strlen(s1);
        sprintf(p, "res.%s = res.%s%s;\n", vfield, vfield, offStr);
    }

    kgenAddBlankLine(ctx);

    ctx1 = createKgenContext(s2, sizeof(s2), true);
    if (ctx1 == NULL) {
        return -ENOMEM;
    }

    kgenSyncFormatting(ctx1, ctx, 1);

    // update logic
    sprintf(src, "res.%s[i * %lu + j]", vfield, pitch);
    if (uflags & UPRES_COLUMN_MAJOR) {
        sprintf(dst, "uC.%s[j * ldc + i]", vfield);
    }
    else {
        sprintf(dst, "uC.%s[i * ldc + j]", vfield);
    }
    op = (kflags & KEXTRA_BETA_ZERO) ? UPRES_SET : UPRES_SUM;
    genUpdateResultSingle(ctx1, dst, src, gset, op, uflags);

    if ( kflags & KEXTRA_UPPER_TRIANG ) {
        declareComplexMults(ctx, dtype, uflags);

        sprintf(tmp, "%s"   // variables
                     /*
                      * setup number of rows to update
                      * and start column to update from
                      */
                     "j = min(%s + %lu, %s + %lu) - %s;\n"
                     "m = min(m, j);\n"
                     "j0 = (%s < %s) ? (%s - %s) : 0;\n"
                     "\n"
                     "for (i = 0; i < m; i++) {\n"
                     "    for (j = j0; j < n; j++) {\n"
                     "%s" // update logic
                     "    }\n"
                          /*
                           * increment row, increment start column
                           * if the diagonal is reached
                           */
                     "    %s++;\n"
                     "    j0 = (%s >= %s) ? j0 : (j0 + 1);\n"
                     "}\n",
                s1, trow, dims[1].y, tcol, dims[1].x, trow,
                tcol, trow, trow, tcol, s2, trow, tcol, trow);
    }
    else {
        declareComplexMults(ctx, dtype, uflags);

        sprintf(tmp, "uint i0;\n"
                     "%s"       // variables
                     "i0 = (%s < %s) ? (%s - %s) : 0;\n"
                     "j = min(%s + %lu, %s + %lu) - %s;\n"
                     "n = min(j, n);\n"
                     "j0 = (%s < %s) ? (%s - %s + 1) : 1;\n"
                     "\n"
                     "for (i = i0; i < m; i++) {\n"
                     "    for (j = 0; j < j0; j++) {\n"
                     "%s"       // update logic
                     "    }\n"
                     "    j0 = min(j0 + 1, n);\n"
                     "}\n",
                s1, trow, tcol, tcol, trow, trow, dims[1].y, tcol,
                dims[1].x, tcol, tcol, trow, trow, tcol, s2);
    }

    destroyKgenContext(ctx1);

    return kgenAddStmt(ctx, tmp);
}

//-----------------------------------------------------------------------------

static int
genUpdateResult(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    BlasGenSettings *gset,
    UpdateResultFlags upresFlags,
    const char * d1, // dummy parameters for compatibility with callback ptr
    const char * d2,
    const char * d3)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    KernelExtraFlags diagFlags = KEXTRA_SYRK_SEPARATE_DIAGONAL |
                                 KEXTRA_SYRK_EVALUATE_DIAGONAL;
    int ret;
    char tmp[1024];

    DUMMY_ARGS_USAGE_3(d1, d2, d3);

    if ( gset->kextra->flags & KEXTRA_UPPER_TRIANG ) {

        sprintf( tmp,
            "if ( !( (coord.y >= startN + argN) || "
                "(coord.x >= origN) || "
                "(coord.y >= coord.x + %lu) ) )",
             gset->subdims[1].x );

        kgenBeginBranch( ctx, tmp );
    }
    else {

        sprintf( tmp,
            "if ( !( (coord.y >= startN + argN) || "
                  "(coord.x >= startN + argN) || "
                  "(coord.x >= coord.y + %lu) ) )",
            gset->subdims[1].y );

        kgenBeginBranch( ctx, tmp );

    }

    // update diagonal if the chosen mode implies its processing
    if ((kflags & diagFlags) != KEXTRA_SYRK_SEPARATE_DIAGONAL) {
        const char *tcol = gset->varNames.coordB;
        const char *trow = gset->varNames.coordA;
        bool areTails;

        areTails = ((kflags & (KEXTRA_TAILS_M_LOWER |
                               KEXTRA_TAILS_N_LOWER)) != 0);

        if (areTails || (gset->subdims[1].y == gset->subdims[1].x)) {
            if ( kflags & KEXTRA_UPPER_TRIANG ) {
                sprintf(tmp, "if (%s + %lu > %s)",
                        trow, gset->subdims[1].y, tcol);
            }
            else {
                sprintf(tmp, "if (%s + %lu > %s)",
                        tcol, gset->subdims[1].x, trow);
            }

            kgenBeginBranch(ctx, tmp);
            if (!areTails) {
                ret = genUpdateIsoscelesDiagTile(ctx, gset);
            }
            else {
                ret = genUpdateTailedDiagTile(ctx, gset, upresFlags);
            }
        }
        else {
            unsigned int xb, yb;

            xb = (unsigned int)gset->subdims[0].x;
            yb = (unsigned int)gset->subdims[0].y;

            if ( kflags & KEXTRA_UPPER_TRIANG ) {
                sprintf(tmp, "if (%s / %u * %u + %u > %s / %u * %u)",
                        trow, yb, yb, yb - 1, tcol, xb, xb);
            }
            else {
                sprintf(tmp, "if (%s / %u * %u + %u > %s / %u * %u)",
                        tcol, xb, xb, xb - 1, trow, yb, yb);
            }

            kgenBeginBranch(ctx, tmp);
            ret = genUpdateGenericDiagTile(ctx, gset);
        }

        if (ret) {
            return ret;
        }

        kgenEndBranch(ctx, NULL);
        // the function above put a respective code into a conditional path
        kgenBeginBranch(ctx, "else");
    }
    ret = genResultUpdateWithFlags( ctx,
        funcID,
        gset,
        upresFlags,
        NULL,
        NULL,
        NULL );

    if ((kflags & diagFlags) != KEXTRA_SYRK_SEPARATE_DIAGONAL) {
        ret = kgenEndBranch(ctx, NULL);
    }

    kgenEndBranch( ctx, NULL );

    return ret;
}

//-----------------------------------------------------------------------------

static void
initGenSettings(
    BlasGenSettings *gset,
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const CLBLASKernExtra *kextra,
    BlasFunctionID funcID)
{
    KernelVarNames *vnames = &gset->varNames;
    unsigned int vecLen;

    memset(gset, 0, sizeof(BlasGenSettings));

    memcpy(gset->subdims, subdims, sizeof(gset->subdims));
    gset->flags = BGF_LD_IN_VECTORS;
    if ((funcID == CLBLAS_SYR2K) && !(kextra->flags & KEXTRA_SYRK_2K_RANK)) {
        gset->flags |= BGF_DISTINCT_VECLEN;
    }

    gset->pgran = pgran;
    gset->kextra = kextra;

    // !!! WORKAROUND; some cases fails with fetched fully tile of A
    vecLen = getVecLen(gset, funcID, MATRIX_A);
    if (vecLen != 1) {
        gset->flags |= BGF_WHOLE_A;
    }
    ///////////////////////////////////////////////////////////////////////

    if ((funcID == CLBLAS_SYR2K) && kextra->flags & KEXTRA_SYRK_2K_RANK) {
        vnames->A = "wiA";
        vnames->B = "wiB";
    }
    else {
        vnames->A = "A";
        vnames->B = "B";
    }

    vnames->C = "C";
    vnames->lda = "lda";
    vnames->ldb = (funcID == CLBLAS_SYR2K) ? "ldb" : vnames->lda;
    vnames->alpha = "alpha";
    if (!(kextra->flags & KEXTRA_BETA_ZERO)) {
        vnames->beta = "beta";
    }

    vnames->coordA = "coord.y";
    vnames->coordB = "coord.x";
    vnames->k = "coord.z";

    vnames->sizeM = "N";
    vnames->sizeN = "N";
    vnames->sizeK = "K";
    vnames->skewA = NULL;
    vnames->skewB = NULL;
    vnames->skewK = NULL;
}

//-----------------------------------------------------------------------------

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra,
   BlasFunctionID funcID)
{
    ssize_t ret;
    struct KgenContext *ctx;
    char tmp[1024];
    CLBLASKernExtra kextraNew;
    TileCreationFlags tcflags;
    DataType dtype;
    KernelExtraFlags kflags;
    UpdateResultFlags uflags;
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    KernelVarNames *vnames = &gset.varNames;
    int i, numRanks;
    TilePostFetchPrivate pfPriv;
    TailStatus tailStatus = 0;
    FetchAddrMode addrMode;
    SyrxkExtraPriv *priv;
    bool subgMode = 0;
    SubgVarNames subgVNames;
    bool areTailsMN;

    memcpy(&kextraNew, extra, sizeof(kextraNew));

    subgMode = ( subdims[0].bwidth != subdims[1].bwidth );

    // fixup tail flags in respect with the selected separate diagonal mode
    kflags = kextraNew.flags;
    if (kflags & KEXTRA_SYRK_SEPARATE_DIAGONAL) {
        bool isUpper = ((kflags & KEXTRA_UPPER_TRIANG) != 0);

        if ((kflags & (KEXTRA_SYRK_SEPARATE_DIAGONAL |
                       KEXTRA_SYRK_EVALUATE_DIAGONAL)) ==
                           KEXTRA_SYRK_SEPARATE_DIAGONAL) {
            if (isUpper) {
                kflags &= ~(KEXTRA_TAILS_M | KEXTRA_TAILS_M_LOWER);
            }
            else {
                kflags &= ~(KEXTRA_TAILS_N | KEXTRA_TAILS_N_LOWER);
            }
        }

        kextraNew.flags = kflags;
    }
    dtype = kextraNew.dtype;

    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    kgenDeclareUptrs(ctx, isDoubleBasedType(dtype));

    initGenSettings(&gset, subdims, pgran, &kextraNew, funcID);
    /*
     * fixup vectorization for C if some restrictions for it has been set
     * during the generic solve stage
     */
    priv = (SyrxkExtraPriv*)&kextraNew.solverPriv;
    if (priv->maxVlenC) {
        kextraNew.vecLenC = umin(kextraNew.vecLenC, priv->maxVlenC);
        if (!(gset.flags & BGF_DISTINCT_VECLEN)) {
            kextraNew.vecLen = umin(kextraNew.vecLenC, kextraNew.vecLen);
        }
    }

    mulOpts.memA = mulOpts.memB = CLMEM_GLOBAL_MEMORY;
    mulOpts.core = (kflags & KEXTRA_ENABLE_MAD) ? TILEMUL_MAD : TILEMUL_MULADD;
    mulOpts.postFetch = NULL;
    mulOpts.flags = TILEMUL_NO_FLAGS;
    if (isMatrixAccessColMaj(funcID, kflags, MATRIX_A)) {
        mulOpts.flags |= TILEMUL_TRA;
    }
    else {
        mulOpts.flags |= TILEMUL_TRB;
    }

    mulOpts.fctx = createFetchContext();
    if (mulOpts.fctx == NULL) {
        destroyKgenContext(ctx);
        return -ENOMEM;
    }

    if (kflags & KEXTRA_TAILS_K_LOWER) {
        // setup post fetch callback
        memset(&pfPriv, 0, sizeof(pfPriv));
        pfPriv.wholeA = 1;
        pfPriv.funcID = funcID;
        pfPriv.gset = &gset;
        mulOpts.postFetch = defaultTilePostFetch;
        mulOpts.postFetchPriv = &pfPriv;
    }

    if( subgMode ) {
        declareKernel( ctx, &gset, funcID, "Subg" );
    }
    else {
        declareKernel( ctx, &gset, funcID, "Block" );
    }

    kgenBeginFuncBody(ctx);

    areTailsMN = (kflags & (KEXTRA_TAILS_M_LOWER | KEXTRA_TAILS_N_LOWER)) != 0;
    tcflags = areTailsMN ? TILE_C_FORCE_NOTRANS : 0;
    initDefaultTiles(&gset, funcID, tcflags, PRIV_STORAGE_VARIABLE_SET);

    /*
     * FIXME: since now it is used PPtr for updating diagonal
     *        in case of tails variables cannot be used
     */
    if (areTailsMN) {
        gset.tileCY.storType = PRIV_STORAGE_ARRAY;
    }
    declareTileStorages(ctx, &gset);

    genHead( ctx, &gset, funcID, &subgVNames, subgMode );
    genZeroTile(ctx, &gset.tileCY);
    /* For adjusting coordinates, skews and updating result */
    kgenAddStmt(ctx,
            "// Set N to initial argument of blas function, not divided one\n"
            "N = origN;\n");

    if ( kflags & KEXTRA_UPPER_TRIANG ) {
        tailStatus = checkGenAdjustTailCoords(ctx, funcID, &gset, NULL);
        kgenAddBlankLine(ctx);
    }

    // generate multiplication logic
    numRanks = (kflags & KEXTRA_SYRK_2K_RANK) ? 2 : 1;
    addrMode = setDefaultFetchAddrMode(mulOpts.fctx, &gset, 0, tailStatus,
                                      (kflags & KEXTRA_TAILS_K_LOWER) != 0);

    genScaleLeadingDimensions(ctx, &gset);
    // ldc should not be scaled, so it is initialized after that
    gset.varNames.ldc = "ldc";

    // Begin loop over the small panel

    for (i = 0; i < numRanks; i++) {
        if (i) {
            kgenAddStmt(ctx, "// begin the second rank update\n");

            /*
             * For the second rank, reset coordinates and swap leading
             * dimensions
             */
            if (!(addrMode & FETCH_ADDR_K_RELATIVE)) {
                kgenAddStmt(ctx, "coord.z = 0;\n");
            }
            vnames->lda = "ldb";
            vnames->ldb = "lda";
        }
        genSetupPointers(ctx, &gset, funcID, addrMode, i);

        if (i) {
            kgenBeginBranch(ctx, NULL);
        }
        prepareFetchLoop(ctx, mulOpts.fctx, &gset, CLMEM_GLOBAL_MEMORY,
                         CLMEM_GLOBAL_MEMORY);

        if ( subgMode ) {

            mulOpts.flags |= TILEMUL_BW_STRIDE;
            mulOpts.flags |= TILEMUL_NOT_INC_K;
            mulOpts.postFetch = NULL;
            setFetchAddrMode(mulOpts.fctx, (addrMode&~FETCH_ADDR_K_RELATIVE));

            sprintf( tmp, "if( skipTilemul == 0 )");
            kgenBeginBranch( ctx, tmp );

            if ( kflags & KEXTRA_TAILS_K_LOWER ) {

                kgenPrintf( ctx, "uint kBase = K - (K%%%lu);\n", subdims[0].bwidth );
                sprintf( tmp,
                    "for ( k0 = %s.x * %lu; k0 < kBase; k0 += %lu )",
                    subgVNames.itemId,
                    subdims[1].bwidth,
                    subdims[0].bwidth );
            }
            else {

                sprintf( tmp,
                    "for ( k0 = %s.x * %lu; k0 < K; k0 += %lu )",
                    subgVNames.itemId,
                    subdims[1].bwidth,
                    subdims[0].bwidth );
            }

            // main loop branch
            kgenBeginBranch( ctx, tmp );
            gset.varNames.k = "k0";
        }
        else {

            sprintf(tmp, "for (k0 = 0; k0 < K; k0 += %lu)", subdims[1].bwidth);
            kgenBeginBranch(ctx, tmp);
        }

        pfPriv.fetchNumA = 0;
        tileMulGen(ctx, &gset, &mulOpts);
        // main loop branch
        kgenEndBranch(ctx, NULL);

        if ( subgMode ) {

            // lowerK tails for subgroup mode
            if( kflags & KEXTRA_TAILS_K_LOWER ) {

                setFetchAddrMode(mulOpts.fctx, addrMode | FETCH_ADDR_TAILK_PADD);
                mulOpts.postFetch = defaultTilePostFetch;
                mulOpts.flags |= TILEMUL_EXTERN_RDECL;

                kgenPrintf( ctx,
                    "%s = kBase + %s.x*%lu;\n",
                    vnames->k,
                    subgVNames.itemId,
                    subdims[1].bwidth );

                tileMulGen( ctx, &gset, &mulOpts );
            }

            // skipTilemul branch
            kgenEndBranch( ctx, NULL );

        }

        if (i) {
            kgenEndBranch(ctx, NULL);
        }

        kgenAddBlankLine(ctx);
    }

    if ( kflags & KEXTRA_UPPER_TRIANG ) {
        checkGenRestoreTailCoords(ctx, &gset, tailStatus);
    }
    kgenAddBlankLine(ctx);
    gset.flags &= ~BGF_LD_IN_VECTORS;

    uflags = kextraToUpresFlags(funcID, kflags);
    uflags |= tailStatusToUpresFlags(tailStatus);

    if ( subgMode ) {

        mergeUpdateResult( ctx,
            funcID,
            &gset,
            &subgVNames,
            //uflags | UPRES_EXCEED_PROBLEM_CONDITION,
            uflags,
            (UpresProcPtr)genUpdateResult );
    }
    else {
        genUpdateResult( ctx,
            funcID,
            &gset,
            uflags,
            NULL,
            NULL,
            NULL );
    }

    ret = kgenEndFuncBody(ctx);
    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }

    destroyFetchContext(mulOpts.fctx);
    destroyKgenContext(ctx);

    return (ret < 0) ? -EOVERFLOW : ret;
}

//-----------------------------------------------------------------------------

static void
assignKargs(
    KernelArg *args,
    const CLBlasKargs *blasArgs,
    KernelExtraFlags kflags,
    BlasFunctionID funcID)
{
    int i = 5;

    // height of the diagonal part
    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->K);
    assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
    initMemobjKarg(&args[3], blasArgs->A, NULL, 0, 0);
    initSizeKarg(&args[4], blasArgs->lda.matrix);
    if (funcID == CLBLAS_SYR2K) {
        initMemobjKarg(&args[i++], blasArgs->B, NULL, 0, 0);
        initSizeKarg(&args[i++], blasArgs->ldb.matrix);
    }

    if (!(kflags & KEXTRA_BETA_ZERO)) {
        assignScalarKarg(&args[i++], &(blasArgs->beta), blasArgs->dtype);
    }

    initMemobjKarg(&args[i++], blasArgs->C, NULL, 0, 0);
    initSizeKarg(&args[i++], blasArgs->ldc.matrix);
    initSizeKarg(&args[i++], blasArgs->offsetM);
    /* Original N */
    initSizeKarg(&args[i++], blasArgs->N);
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offA);
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offBX);
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        initSizeKarg(&args[i++], blasArgs->offCY);
    }
}

//-----------------------------------------------------------------------------

static void
syrkAssignKargs(KernelArg *args, const void *params, const void *extra)
{
    (void)extra;

    assignKargs(args, (const CLBlasKargs*)params,
                ((const CLBLASKernExtra*)extra)->flags, CLBLAS_SYRK);
}

//-----------------------------------------------------------------------------

static void
syr2kAssignKargs(KernelArg *args, const void *params, const void *extra)
{
    (void)extra;

    assignKargs(args, (const CLBlasKargs*)params,
                ((const CLBLASKernExtra*)extra)->flags, CLBLAS_SYR2K);
}

//-----------------------------------------------------------------------------

static void
syrkCalcThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    const CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    CLBlasKargs *blasArgs = (CLBlasKargs*)args;
    size_t nrGroups = 0;
    size_t x, procX, startN, N, origN, step;
    bool isU = (blasArgs->uplo == clblasUpper);
    KernelExtraFlags kflags = ((CLBLASKernExtra*)extra)->flags;
    KernelExtraFlags diagFlags = KEXTRA_SYRK_SEPARATE_DIAGONAL |
                                 KEXTRA_SYRK_EVALUATE_DIAGONAL;
    bool isDiagSep = ((kflags & KEXTRA_SYRK_SEPARATE_DIAGONAL) != 0);
    bool isEvalOnlyDiag = ((kflags & diagFlags) == diagFlags);
    size_t start, end;
    int roundDir = 1;
    size_t vecAlign = 1;

    /*
     * Traverse the output matrix with panels from
     * the largest one
     */
    N = blasArgs->M;                // width of the diagonal part
    startN = blasArgs->offsetM;     // vertical offset of the diagonal part
    origN = blasArgs->N;
    x = (isU) ? N : 0;
    step = subdims[0].x;

    /*
     * NOTE:
     *
     * In the case of separate evaluating of the area around the diagonal it's
     * critically important that at least on step would be aligned.
     * Otherwise, solution areas will overlap that will lead to a wrong result.
     */

    if (isU && (isDiagSep && !isEvalOnlyDiag)) {
        roundDir = 0;
    }
    else {
        roundDir = 1;
    }

    if (!isU && (!isDiagSep || isEvalOnlyDiag)) {
        vecAlign = isMatrixAccessColMaj(CLBLAS_SYRK, kflags, MATRIX_A) ?
                        (size_t)umax(kextra->vecLenA, kextra->vecLenB) : 1;
    }

    for (procX = 0; procX < N; procX += step) {
        if (isU) {
            step = (isU && (x % subdims[0].x)) ? (x % subdims[0].x) :
                                                  subdims[0].x;
            start = (!isEvalOnlyDiag) ? 0 : roundDown(x - step, subdims[0].y);
            end = (!isDiagSep || isEvalOnlyDiag) ? x :
                                        roundDown(x - step, subdims[0].y);
            x -= step;
        }
        else {
            start = (!isDiagSep || isEvalOnlyDiag) ? x :
                                    roundUp(x + step, subdims[0].y);
            end = (isEvalOnlyDiag) ? roundUp(x + step, subdims[0].y) : N;

            end = roundUp(end, vecAlign);
            x += step;
            if (start >= end) {
                continue;
            }
        }

        if (roundDir) {
            nrGroups += divRoundUp(end - start, subdims[0].y);
        }
        else {
            nrGroups += (end - start) / subdims[0].y;
        }
    }

    /* rectangular part of trapezium */
    if (!isEvalOnlyDiag) {
        if (isU) {
            nrGroups += divRoundUp(N, subdims[0].y) *
                divRoundUp(origN - N - startN, subdims[0].x);
        }
        else {
            nrGroups += (startN / subdims[0].x) * divRoundUp(N, subdims[0].y);
        }
    }

    if (nrGroups == 0) { // in case we got N==0
        nrGroups = 1;
    }
    threads[0] = nrGroups * pgran->wgSize[0];
    threads[1] = 1;
}

//-----------------------------------------------------------------------------

static SolverFlags
solverFlags(void)
{
    return SF_WSPACE_1D;
}

//-----------------------------------------------------------------------------

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    const CLBlasKargs *blasArgs = (const CLBlasKargs*)args;
    size_t moddim;

    extraData_t *extraData = (extraData_t*)&((CLBLASKernExtra*)extra)->solverPriv;

    const size_t nChans = 8; // !!!DEVICE DEPENDED!!!
    const size_t wideChans = 64; // !!!DEVICE DEPENDED!!!
    const size_t sizeType[] = {1,2,2,4};

    size_t sizeBlock = wideChans * nChans / sizeType[blasArgs->dtype];
    size_t off = blasArgs->K % sizeBlock;
    if (off == 0) {
        extraData->staggered = roundUp(subdims[1].bwidth * sizeType[blasArgs->dtype]
                                    , wideChans / sizeType[blasArgs->dtype]);
    }
    else {
        extraData->staggered = 0;
    }
    extraData->staggered = 64 / sizeType[blasArgs->dtype]; //fixed, not calculated

    /*
     * Save maxium possible vectorization for C in case of column-major order
     * and lower triangular matrix C. It is needed because the 'y' problem
     * dimensions expands in backward direction and aligned access to memory
     * can occur.
     */
    moddim = (unsigned int)(blasArgs->N % subdims[1].y);
    if (isMatrixAccessColMaj(CLBLAS_SYRK, kextra->flags, MATRIX_C) &&
        (blasArgs->uplo == clblasLower) && moddim) {

        SyrxkExtraPriv *priv = (SyrxkExtraPriv*)kextra->solverPriv;
        size_t tsize;

        tsize = dtypeSize(kextra->dtype);
        priv->maxVlenC = appropriateVecLen(blasArgs->N, tsize, subdims[1].y, 3);
    }
}

//-----------------------------------------------------------------------------

static bool
checkCalcDecomp(
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
initSyr2kBlockPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block syr2k";
    mempat->nrLevels = 2;
    mempat->sops = &syr2kSolverOps;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

void
initSyrkBlockPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block syrk";
    mempat->nrLevels = 2;
    mempat->sops = &syrkSolverOps;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

void
initSyrkSubgPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based subgroup syrk";
    mempat->nrLevels = 2;
    mempat->sops = &syrkSubgSops;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mempat->extra = &mpatExtra;
}

//-----------------------------------------------------------------------------

void
initSyr2kSubgPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based subgroup syr2k";
    mempat->nrLevels = 2;
    mempat->sops = &syr2kSubgSops;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mempat->extra = &mpatExtra;
}

// ----------------------------------------------------------------------------

static int
syrkSubgGetPerf(
        unsigned int kflags,
        const void *args)
{
    DUMMY_ARG_USAGE(args);

    if ( !isMatrixAccessColMaj( CLBLAS_SYRK, kflags, MATRIX_A ) &&
         !isMatrixAccessColMaj( CLBLAS_SYRK, kflags, MATRIX_B ) ) {

        return PPERF_GOOD;
    }

    return PPERF_NOT_SUPPORTED;
}

//-----------------------------------------------------------------------------

static int
syrkBlockGetPerf(
        unsigned int kflags,
        const void *args)
{
    DUMMY_ARG_USAGE(args);

    if ( !isMatrixAccessColMaj( CLBLAS_SYRK, kflags, MATRIX_A ) &&
         !isMatrixAccessColMaj( CLBLAS_SYRK, kflags, MATRIX_B ) ) {

        return PPERF_AVERAGE;
    }

    return PPERF_GOOD;
}

//-----------------------------------------------------------------------------

static int
syrkSubgGetDefaultDecomp( PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs )
{
    DUMMY_ARG_USAGE(subdimsNum);
    pgran->wgDim = 1;
    return subgGetDefaultDecomp( pgran, subdims, pArgs );
}

//-----------------------------------------------------------------------------

#if 0

// for debug
static int
syrkBlockGetDefaultDecomp(
        PGranularity *pgran,
        SubproblemDim *subdims,
        unsigned int subdimsNum)
{
    // !!! DEBUG
#if 1
    subdims[0].itemX = subdims[0].x = 64;
    subdims[0].itemY = subdims[0].y = 32;
    subdims[0].bwidth = subdims[1].bwidth = 2;
    subdims[1].itemX = subdims[1].x = 8;
    subdims[1].itemY = subdims[1].y = 4;
#else
    subdims[0].itemX = subdims[0].x = 32;
    subdims[0].itemY = subdims[0].y = 32;
    subdims[0].bwidth = subdims[1].bwidth = 4;
    subdims[1].itemX = subdims[1].x = 4;
    subdims[1].itemY = subdims[1].y = 4;
#endif
    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;

    return 0;
    //////////////////////////////////////////////////

    if( (subdimsNum<2)||
        (NULL==pgran)||
        (NULL==subdims) ){

        return EINVAL;
    }

    pgran->wgDim = 1;
    pgran->wgSize[0] = 64;

    subdims[1].bwidth = 4;
    subdims[1].itemX = subdims[1].x = 4;
    subdims[1].itemY = subdims[1].y = 4;

    //subdims[0].bwidth = subdims[1].bwidth * itemsPerSubg;
    subdims[0].bwidth = subdims[1].bwidth;
    subdims[0].itemX = subdims[0].x = subdims[1].x * 8;
    subdims[0].itemY = subdims[0].y = subdims[1].y * 8;

    return 0;

}

#endif

//-----------------------------------------------------------------------------

static bool
subgCheckCalcDecomp(
        PGranularity *pgran,
        SubproblemDim *subdims,
        unsigned int subdimsNum,
        DataType dtype,
        int check)
{
    size_t subgA = 0;
    size_t subgB = 0;
    size_t regUse = 0;
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
