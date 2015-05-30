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
 * gemv generator
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

typedef struct {
    size_t staggered;
} MAY_ALIAS extraData_t;

static const char *gemvDecl =
    "__attribute__((reqd_work_group_size(%lu, %lu, 1)))\n"
    "void __kernel\n"
    "%cgemv(\n"
    "    uint %c,\n"
    "    uint %c,\n"
    "    const %s alpha,\n"
    "    const __global %s *restrict A,\n"
    "    const __global %s *restrict X,\n"
    "%s"
    "    __global %s *Y,\n"
    "    uint lda"
    "%s"    // offset A, X and Y
    "%s"
    "%s)\n";

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

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static SolverFlags
solverFlags(void);

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

static bool
subgCheckCalcDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    DataType dtype,
    int check);

static int
subgGetDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void * pArgs);

static SolverOps gemvSops = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL,
    NULL,
    calcNrThreads,
    NULL,
    solverFlags,
    fixupArgs,
    subgGetDefaultDecomp,//getDefaultDecomposition
    subgCheckCalcDecomp, //get Decomp. list
    NULL,
    NULL
};

static void
declareGemvKernel(
    struct KgenContext *ctx,
    DataType dtype,
    const PGranularity *pgran,
    KernelExtraFlags kflags)
{
    char sizeNames[2] = {'M', 'N'};
    bool incxOne = ((kflags & KEXTRA_INCX_ONE) != 0);
    bool incyOne = ((kflags & KEXTRA_INCY_ONE) != 0);
    bool beta0 = ((kflags & KEXTRA_BETA_ZERO) != 0);
    const char *incxDecl = incxOne ? "" : ",\n    const int incx";
    const char *incyDecl = incyOne ? "" : ",\n    const int incy";
    char offDecl[128];
    char betaDecl[128];
    char tmp[512];
    char fpref;
    bool tra = ((kflags & KEXTRA_TRANS_A) != 0);
    const char *typeName;

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);

    offDecl[0] = '\0';
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        strcpy(offDecl, ",\n    const uint offA");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        strcat(offDecl, ",\n    const uint offX");
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        strcat(offDecl, ",\n    const uint offY");
    }

    if (beta0) {
        betaDecl[0] = '\0';
    }
    else {
        sprintf(betaDecl, "    const %s beta,\n", typeName);
    }
    sprintf(tmp, gemvDecl, pgran->wgSize[0], pgran->wgSize[1], fpref,
            sizeNames[tra], sizeNames[1 - tra],
            typeName, typeName, typeName, betaDecl, typeName,
            offDecl, incxDecl, incyDecl);

    kgenDeclareFunction(ctx, tmp);
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

    nrPrivs = 1;
    for (i = 0; i < nrPrivs; i++) {
        priv[i].fetchNumA = 0;
        priv[i].wholeA = 1;
        priv[i].funcID = CLBLAS_GEMV;
        priv[i].gset = gset;
        priv[i].regName = regName;
        mulOpts->postFetch = handler;
        mulOpts->postFetchPriv = priv;
    }
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
    size_t staggered = ((extraData_t*)&kextra->solverPriv)->staggered;
    //yes, KEXTRA_TAILS_K because it is set if N % bw != 0
    bool tailN = ((kflags & KEXTRA_TAILS_K) != 0);
    bool tailM = ((kflags & KEXTRA_TAILS_M) != 0);
    char tmp[4096];
    DataType dtype = kextra->dtype;
    bool doubleBased = isDoubleBasedType(dtype);
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    KernelVarNames *vnames = &gset.varNames;
    ssize_t ret;
    TilePostFetchPrivate pfPriv;
    unsigned int vecLen = kextra->vecLen;
    const char *outTypeName;
    const char *gid = "get_group_id(0)";
    const char *lid = "get_local_id(0)";
    const char *typeName;
    size_t wgSize;
    //unsigned int nStep = 32;
    unsigned int bStep = subdims[0].bwidth / subdims[1].bwidth; //8;
    unsigned int cLocal;
    bool isComplex = isComplexType(dtype);
    unsigned int nPlans;

    typeName = dtypeBuiltinType(dtype);
    memset(&gset, 0, sizeof(gset));
    memset(&mulOpts, 0, sizeof(mulOpts));
    ctx = createKgenContext(buf, buflen, true);
    if (ctx == NULL) {
        return -ENOMEM;
    }

    // at first, generate needed declarations
    kgenDeclareUptrs(ctx, doubleBased);

    // now, generate the kernel
    declareGemvKernel(ctx, dtype, pgran, kflags);
    ret = kgenBeginFuncBody(ctx);
    kgenAddStmt(ctx, "// M always denotes length of Y "
                     "and N denotes length of X in the kernel\n");
    /* 1D work space. Matrix is divided among wi, each calculates it's own
     * part of vector y */

    wgSize = (subdims[0].y / subdims[1].y) *
            (subdims[0].bwidth / subdims[1].bwidth);
    assert(pgran->wgSize[0] == wgSize);
    assert(subdims[0].x == 1);
    assert(subdims[1].x == 1);
    cLocal = wgSize/bStep;

    memcpy(gset.subdims, subdims, sizeof(gset.subdims));
    gset.subdims[0].itemX = gset.subdims[0].x = 1;
    gset.subdims[1].itemX = gset.subdims[1].x = 1;
    gset.subdims[0].bwidth = gset.subdims[1].bwidth;

    gset.pgran = pgran;
    gset.kextra = kextra;
    gset.flags = BGF_UPTRS;

    initDefaultTiles(&gset, CLBLAS_GEMV, 0, PRIV_STORAGE_VARIABLE_SET);
    if (isComplex) {
         gset.tileCY.vecLen = 1;
    }
    declareTileStorages(ctx, &gset);
    genZeroTile(ctx, &gset.tileCY);
    getVectorTypeName(dtype, gset.tileCY.vecLen, &outTypeName, NULL);
    nPlans = gset.tileCY.nrRows / gset.tileCY.vecLen;

    sprintf(tmp, "__local %s localRes[%u][%u];\n",
                outTypeName, pgran->wgSize[0], nPlans);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint coordA = (%s * %u + %s %% %u) * %lu;\n",
                 gid, bStep, lid, bStep, subdims[1].y);
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint k0 = (%s / %u) * %lu;\n",
                 lid,  bStep, subdims[1].bwidth);
    kgenAddStmt(ctx, tmp);

    kgenAddBlankLine(ctx);

    kgenBeginBranch(ctx,"if (coordA < M && k0 < N)");

    genIncPointers(ctx, kflags);
    sprintf(tmp,
            "const GPtr Ag = {(__global %s*)A};\n"
            "const GPtr Xg = {(__global %s*)X};\n",
            typeName, typeName);
    kgenAddStmt(ctx, tmp);

    kgenAddBlankLine(ctx);

    if (tailN) {
        sprintf(tmp, "uint Ntail = N %% %lu;\n", subdims[1].bwidth);
        kgenAddStmt(ctx, tmp);
        kgenAddStmt(ctx, "N -= Ntail;\n");
        kgenAddBlankLine(ctx);
    }

    mulOpts.flags |= TILEMUL_OPTIMIZE_COORD_CALC;
    if (tailM) {
        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_A;
    }

    vnames->A = "Ag";
    vnames->B = "Xg";
    vnames->coordA = "coordA";
    vnames->coordB = ""; //should not be used for vector
    vnames->k = "k";
    vnames->lda = "lda";
    vnames->sizeK = "N";
    vnames->sizeM = "M";

    mulOpts.flags |= TILEMUL_NOT_FETCH_B | TILEMUL_TRB | TILEMUL_C_COLUMN_MAJOR | TILEMUL_NOT_INC_K;
    if ((kflags & KEXTRA_CONJUGATE_A) != 0) {
        mulOpts.flags |= TILEMUL_CONJA;
    }
    if (isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
        mulOpts.flags |= TILEMUL_TRA;
    }
    if ((kflags & KEXTRA_ENABLE_MAD) != 0) {
        mulOpts.core = TILEMUL_MAD;
    }
    else {
        mulOpts.core = TILEMUL_MULADD;
    }
    mulOpts.memA = CLMEM_GLOBAL_MEMORY;
    mulOpts.memB = CLMEM_GLOBAL_MEMORY;

    if (!isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
        gset.subdims[0].bwidth = pgran->wgSize[0] * subdims[1].bwidth;
        mulOpts.flags |= TILEMUL_BW_STRIDE;
    }

    sprintf(tmp, "uint k = k0;\nfor (; k < N; k += %lu)", cLocal*subdims[1].bwidth);
    kgenBeginBranch(ctx, tmp);

    if (staggered) {
        vnames->k = "k1";
        sprintf(tmp, "const uint k1 = (k + get_group_id(0)*%lu)%%N;\n",staggered);
        kgenAddStmt(ctx, tmp);
    }

    genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
            mulOpts.flags, kflags);

    ret = tileMulGen(ctx, &gset, &mulOpts);
    if (ret != 0) {
        return ret;
    }
    vnames->k = "k";
    kgenEndBranch(ctx, NULL); /* k loop */

    if (tailN) {
        /* Handle tail along vector X */
        kgenAddStmt(ctx, "N += Ntail;\n");
        kgenBeginBranch(ctx, "if (k < N)");

        mulOpts.flags |= TILEMUL_SKEW_B;
        genFetchX(ctx, &gset.tileBX, gset.kextra->vecLen, dtype, vnames,
                  mulOpts.flags, kflags);
        mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_K|TILEMUL_WRAP_AROUND_TAIL;
        setFetchHandler(&mulOpts, &gset, defaultTilePostFetch, &pfPriv);
        ret = tileMulGen(ctx, &gset, &mulOpts);
        if (ret != 0) {
            return ret;
        }
        kgenEndBranch(ctx, NULL);
    }

    if (!isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
        gset.subdims[0].bwidth = subdims[1].bwidth;
        mulOpts.flags &= ~TILEMUL_BW_STRIDE;
    }

    kgenEndBranch(ctx,NULL);

    genStoreLocalResult(ctx, &gset.tileCY, lid);

    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);
    kgenAddBlankLine(ctx);

    sprintf(tmp, "if (%s < %u && coordA < M && k0 < N)", lid, bStep);
    kgenBeginBranch(ctx, tmp);

    genAddLocalResult(ctx, &gset.tileCY, lid, cLocal, bStep);

    /* write back the results */
    /* y := alpha*A*x + beta*y */
    setResultPos(ctx, kflags, vnames->coordA);

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

    initSizeKarg(&args[0], blasArgs->M);
    initSizeKarg(&args[1], blasArgs->N);
    assignScalarKarg(&args[2], &(blasArgs->alpha), blasArgs->dtype);
    INIT_KARG(&args[3], blasArgs->A);
    INIT_KARG(&args[4], blasArgs->B);
    i = 5;
    if (!(kflags & KEXTRA_BETA_ZERO)) {
        assignScalarKarg(&args[i++], &(blasArgs->beta), blasArgs->dtype);
    }
    INIT_KARG(&args[i], blasArgs->C);
    i++;
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
}

static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    CLBlasKargs *kargs = (CLBlasKargs*)args;
    KernelExtraFlags kflags = ((CLBLASKernExtra*)extra)->flags;

    const size_t nChans = 8; // !!!DEVICE DEPENDED!!!
    const size_t wideChans = 64; // !!!DEVICE DEPENDED!!!
    const size_t sizeType[] = {1,2,2,4};

    size_t sizeBlock = wideChans * nChans / sizeType[kargs->dtype];
    size_t off = kargs->K % sizeBlock;
    extraData_t *extraData = (extraData_t*)&((CLBLASKernExtra*)extra)->solverPriv;
    if (off == 0 && !isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
        /*
         * FIXME: staggered access is not enabled now since for some reason
         *        it leads to slowdown at small sizes
         */
        extraData->staggered = 0; // wideChans / sizeType[kargs->dtype];
    }
    else {
        extraData->staggered = 0;
    }

    (void)subdims;

    off = (kargs->offsetM) ? kargs->offsetM : kargs->offsetN;
    if (off) {
        if (isMatrixAccessColMaj(CLBLAS_GEMV, kflags, MATRIX_A)) {
            kargs->offA += off;
        }
        else {
            kargs->offA += off * kargs->lda.matrix;
        }
        if (kargs->ldc.vector < 0) {
            // K store the original height of the matrix A
            kargs->offCY += (kargs->K - off) * abs(kargs->ldc.vector);
        }
        else {
            kargs->offCY += off * kargs->ldc.vector;
        }
    }

    kargs->offsetM = kargs->offsetN = 0;

}

static int
subgGetDefaultDecomp(
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
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    (void)kernelArgs;

    if (1) {
        cl_ulong size;

	    /*
         * One needs y1 * wgSize size of local memory in elements, but
         * y1 is not calculated yet. The expression below produces
         * reliable a larger value. It is larger in dims[1].bwidth times.
         */
        size = dim[0].y * dim[0].bwidth * dtypeSize(dtype);

        return (size <= ldsSize);
    }
    return true;
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    size_t yLen;     /* Length of "Y" vector */
    const CLBlasKargs *kargs = args;
    unsigned int subgr = pgran->wgSize[0] / (subdims[0].bwidth / subdims[1].bwidth);

    (void)subdims;
    (void)extra;

    yLen = kargs->transA == clblasNoTrans ? kargs->M : kargs->N;

    if (yLen == 0) {
        yLen = 1;
        //launch one group to avoid CL_INVALID_WORK_GROUP_SIZE error
    }

    //each work item handles y1 lines
    threads[0] = divRoundUp(yLen, subdims[1].y) * subgr;
    threads[0] = roundUp(threads[0], pgran->wgSize[0]);
    threads[1] = 0;
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D);
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
    unsigned int divider0 = 2-!isComplexType(dtype);
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

    if( subdims[0].bwidth > 256 / divider0 ||
        subdims[0].x > 1 ||
        subdims[0].y > 256 / divider0 ){

        return false;
    }

    if (64 != (subdims[0].y / subdims[1].y) *
        (subdims[0].bwidth / subdims[1].bwidth)) {
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

//-----------------------------------------------------------------------------

void
initGemvPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block gemv";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &gemvSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;
}
