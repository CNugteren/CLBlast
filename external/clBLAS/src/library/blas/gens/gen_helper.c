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


#include <stdio.h>
#include "gen_helper.h"
#include "clblas_stddef.h"

#define IDX_INVAL ((unsigned int)-1)

typedef struct CopyPattern {
    SubproblemDim dim;
    const PGranularity *pgran;
    DataType dtype;
    DBlockCopyDirection dir;
    DBlockCopyFlags flags;
    bool generic;
    bool zeroing;
} CopyPattern;

static __inline void
dimSwapXY(SubproblemDim *dim)
{
    size_t tmp = dim->x;

    dim->x = dim->y;
    dim->y = tmp;
}

/*
 * Initialize a dimension structure with the
 * respective values if it's needed or mark them
 * as unused
 */
static void
checkInitSubdim(
    SubproblemDim *dim,
    unsigned int flags,
    unsigned int checkedFlag,
    size_t x,
    size_t y)
{
    if (flags & checkedFlag) {
        dim->x = x;
        dim->y = y;
    }
    else {
        dim->x = SUBDIM_UNUSED;
        dim->y = SUBDIM_UNUSED;
    }
}

/*
 * check if such dimension instance
 * does already exist in the array
 */
static int
lookupDim(
    const SubproblemDim *dim,
    unsigned int idx)
{
    unsigned int i;

    for (i = 0; i < idx; i++) {
        if (dim[i].x == dim[idx].x &&
            dim[i].y == dim[idx].y) {
            break;
        }
    }

    return (i == idx) ? IDX_INVAL : i;
}

static int
cpyGenCallback(struct KgenContext *ctx, const void *pattern)
{
    const CopyPattern *pat = (CopyPattern*)pattern;
    const void *dim = (pat->generic) ? NULL : &pat->dim;

    return copyDataBlockGen(ctx, dim, pat->pgran, pat->dtype,
                            pat->dir, pat->flags);
}

static void
initCopyPattern(
    CopyPattern *pattern,
    const SubproblemDim *blasDim,
    KernelExtraFlags flags,
    MatrixRole mrole,
    BlasFunctionID funcID)
{
    SubproblemDim *dim = &pattern->dim;
    unsigned int vecFlag = 0;

    pattern->flags = 0;

    if (blasDim == NULL) {
        pattern->generic = true;
        dim->x = 0;
        dim->y = 0;
    }
    else {
        pattern->generic = false;

        switch (mrole) {
        case MATRIX_A:
            dim->x = blasDim->bwidth;
            dim->y = blasDim->y;
            break;
        case MATRIX_B:
            dim->x = blasDim->bwidth;
            dim->y = blasDim->x;
            break;
        case MATRIX_C:
            dim->x = blasDim->x;
            dim->y = blasDim->y;
            break;
        default:
            break;
        }
    }

    switch (mrole) {
    case MATRIX_A:
        vecFlag = KEXTRA_NO_COPY_VEC_A;
        break;
    case MATRIX_B:
        vecFlag = KEXTRA_NO_COPY_VEC_B;
        break;
    case MATRIX_C:
        if ((funcID == CLBLAS_TRMM) || (funcID == CLBLAS_TRSM)) {
            vecFlag = KEXTRA_NO_COPY_VEC_B;
        } else {
            vecFlag = KEXTRA_NO_COPY_VEC_C;
        }
        break;
    default:
        break;
    }

    if (flags & vecFlag) {
        pattern->flags |= DBLOCK_COPY_NOT_VECTORIZE;
    }

    if (isMatrixAccessColMaj(funcID, flags, mrole)) {
        if ((pattern->dir == DBLOCK_GLOBAL_TO_LOCAL) &&
            !pattern->generic) {
            dimSwapXY(dim);
        }
        pattern->flags |= DBLOCK_COPY_TRANSPOSE;
    }
    if (isMatrixConj(flags, mrole)) {
        pattern->flags |= DBLOCK_COPY_CONJUGATE;
    }
}

int
generateBufCopyFuncs(
    CopyBufFuncs *funcNames,
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    BufCopyHelperFlags flags)
{
    CopyPattern pattern;
    struct KgenGuard *guard;
    int ret = 0;
    MatrixRole mrole;
    bool needed[MATRIX_ROLES_NUMBER];
    KernelExtraFlags kgenFlags = gset->kextra->flags;
    DataType dtype = gset->kextra->dtype;
    const SubproblemDim *blasDim = gset->subdims;
    const PGranularity *pgran = gset->pgran;
    bool outputTails = (kgenFlags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N));

    guard = createKgenGuard(ctx, cpyGenCallback, sizeof(CopyPattern));
    if (guard == NULL) {
        return -ENOMEM;
    }

    memset(&pattern, 0, sizeof(pattern));

    pattern.dir = DBLOCK_GLOBAL_TO_LOCAL;
    pattern.dtype = dtype;
    pattern.pgran = pgran;

    needed[MATRIX_A] = (flags & BCHF_MATRIX_A);
    needed[MATRIX_B] = (flags & BCHF_MATRIX_B);
    needed[MATRIX_C] = (flags & BCHF_READ_OUTPUT);

    for (mrole = MATRIX_A; mrole <= MATRIX_C; mrole++) {
        if (!needed[mrole]) {
            continue;
        }

        initCopyPattern(&pattern, blasDim, kgenFlags, mrole, funcID);
        findGenerateFunction(guard, &pattern, funcNames->read[mrole],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    if (flags & BCHF_WRITE_OUTPUT) {
        if (flags & BCHF_IMAGE_WRITE) {
            pattern.dir = DBLOCK_LOCAL_TO_IMAGE;
            initCopyPattern(&pattern, NULL, kgenFlags, MATRIX_A, funcID);
            pattern.flags &= ~DBLOCK_COPY_TRANSPOSE;
        }
        else {
            pattern.dir = DBLOCK_LOCAL_TO_GLOBAL;
            initCopyPattern(&pattern, blasDim, kgenFlags, MATRIX_C, funcID);
        }
        ret = findGenerateFunction(guard, &pattern, funcNames->write,
                                   FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    if (ret) {
        destroyKgenGuard(guard);

        return ret;
    }

    // reevaluate needed flags
    needed[MATRIX_A] = needed[MATRIX_A] &&
        (kgenFlags & (KEXTRA_TAILS_M | KEXTRA_TAILS_K));
    needed[MATRIX_B] = needed[MATRIX_B] &&
        (kgenFlags & (KEXTRA_TAILS_N | KEXTRA_TAILS_K));
    needed[MATRIX_C] = needed[MATRIX_C] && outputTails;

    pattern.dir = DBLOCK_GLOBAL_TO_LOCAL;
    for (mrole = MATRIX_A; mrole <= MATRIX_C; mrole++) {
        if (!needed[mrole]) {
            continue;
        }

        initCopyPattern(&pattern, NULL, kgenFlags, mrole, funcID);
        findGenerateFunction(guard, &pattern,
                             funcNames->readGeneric[mrole],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    if ((flags & BCHF_WRITE_OUTPUT) && outputTails) {
        if (flags & BCHF_IMAGE_WRITE) {
            pattern.dir = DBLOCK_LOCAL_TO_IMAGE;
            initCopyPattern(&pattern, NULL, kgenFlags, MATRIX_A, funcID);
            pattern.flags &= ~DBLOCK_COPY_TRANSPOSE;
        }
        else {
            pattern.dir = DBLOCK_LOCAL_TO_GLOBAL;
            initCopyPattern(&pattern,NULL, kgenFlags, MATRIX_C, funcID);
        }
        ret = findGenerateFunction(guard, &pattern, funcNames->writeGeneric,
                                   FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    destroyKgenGuard(guard);

    return ret;
}

int
generateZeroingFuncs(
    ZeroFuncs *funcNames,
    struct KgenContext *ctx,
    const SubproblemDim *blasDim,
    const PGranularity *pgran,
    DataType dtype,
    ZeroGenHelperFlags flags)
{
    int ret = 0;
    SubproblemDim dim[MATRIX_ROLES_NUMBER];
    size_t tsize, nvecs;
    unsigned int i, j;

    tsize = dtypeSize(dtype);
    nvecs = fl4RowWidth(blasDim->bwidth, tsize);

    checkInitSubdim(&dim[MATRIX_A], flags, ZF_MATRIX_A, nvecs * blasDim->y, 1);
    checkInitSubdim(&dim[MATRIX_B], flags, ZF_MATRIX_B, nvecs * blasDim->x, 1);
    nvecs = fl4RowWidth(blasDim->x, tsize);
    checkInitSubdim(&dim[MATRIX_C], flags, ZF_MATRIX_C, nvecs * blasDim->y, 1);

    for (i = 0; (i < MATRIX_ROLES_NUMBER) && !ret; i++) {
        if (dim[i].x == SUBDIM_UNUSED) {
            continue;
        }

        // check whether the function is already generated
        j = lookupDim(dim, i);
        if (j != IDX_INVAL) {
            strcpy(funcNames->names[i], funcNames->names[j]);
        }
        else {
            ret = f4zeroBlockGen(ctx, &dim[i], pgran, "__local");
            if (!ret) {
                kgenGetLastFuncName(funcNames->names[i], FUNC_NAME_MAXLEN,
                                    ctx);
            }
            kgenAddBlankLine(ctx);
        }
    }

    return ret;
}

UpdateResultFlags
kextraToUpresFlags(BlasFunctionID funcID, KernelExtraFlags kflags)
{
    UpdateResultFlags uf = 0;

    if (funcHasBeta(funcID) && !(kflags & KEXTRA_BETA_ZERO)) {
        uf |= UPRES_WITH_BETA;
    }
    if (isMatrixAccessColMaj(funcID, kflags, MATRIX_C)) {
        uf |= UPRES_COLUMN_MAJOR;
    }
    if (kflags & KEXTRA_NO_COPY_VEC_C) {
        uf |= UPRES_NO_VECTORIZATION;
    }

    return uf;
}

int
generateResultUpdate(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName)
{
    UpdateResultFlags flags;

    flags = kextraToUpresFlags(funcID, gset->kextra->flags);

    return genResultUpdateWithFlags(ctx, funcID, gset, flags,
                                    optFuncName, genericFuncName, NULL);
}

int
genResultUpdateWithFlags(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    UpdateResultFlags flags,
    const char *optFuncName,
    const char *genericFuncName,
    const char *cachedName)
{
    KernelExtraFlags kflags = gset->kextra->flags;
    UpdateResultOp op;
    char tmp[1024];
    int ret = 0;
    const char *coordY, *coordX;
    UpresVarNames uvars;
    const KernelVarNames *kvarNames = &gset->varNames;
    const SubproblemDim *dim = &gset->subdims[1];
    bool areTails, useCondition;

    memset(&uvars, 0, sizeof(uvars));

    coordX = kvarNames->coordB;
    coordY = kvarNames->coordA;

    if (funcHasTriangMatrix(funcID)) {
        if (flags & UPRES_TRIANG_WRITE_C) {
            uvars.result = "C";
        }
        else {
            uvars.result = "B";
        }
        uvars.ld = "ldb";
    }
    else {
        uvars.result = "C";
        uvars.ld = "ldc";
    }

    uvars.cachedName = cachedName;

    /* For now, kernels that do not use UPRES_EXCEED_PROBLEM_CONDITION
     * must return in case problem exceeds more precise lower level conditions
     * (KEXTRA_TAILS_M_LOWER, KEXTRA_TAILS_N_LOWER) before updating result
    */
    areTails = (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N));
    useCondition = areTails && ((flags & UPRES_EXCEED_PROBLEM_CONDITION) != 0);
    if (useCondition) {
        bool tailM = (kflags & KEXTRA_TAILS_M) != 0;
        bool tailN = (kflags & KEXTRA_TAILS_N) != 0;

        if (tailM) {
            if (tailN) {
                sprintf(tmp, "if ((%s < %s) && (%s < %s))",
                        coordY, kvarNames->sizeM, coordX, kvarNames->sizeN);
            }
            else {
                sprintf(tmp, "if (%s < %s)", coordY, kvarNames->sizeM);
            }
        }
        else {
            // here tailN is true
            sprintf(tmp, "if (%s < %s)", coordX, kvarNames->sizeN);
        }
        kgenBeginBranch(ctx, tmp);
    }
    else {
        kgenAddBlankLine(ctx);
    }

    if (optFuncName) {
        const char *betaStr;
        betaStr = (flags & UPRES_WITH_BETA) ? ", beta" : "";

        // update with functions invoking
        if (!(kflags & (KEXTRA_TAILS_M_LOWER | KEXTRA_TAILS_N_LOWER))) {
            sprintf(tmp, "%s(%s, c, alpha, %s, %s, %s%s);\n",
                    optFuncName, uvars.result, coordY, coordX,
                    uvars.ld, betaStr);
        }
        else {
            sprintf(tmp, "uint y = min(%luu, %s - (uint)%s);\n"
                         "uint x = min(%luu, %s - (uint)%s);\n"

                         "if ((y == %lu) && (x == %lu)) {\n"
                         "    %s(%s, c, alpha, %s, %s, %s%s);\n"
                         "}\n"
                         "else {\n"
                         "    %s(%s, c, alpha, %s, %s, %s%s, y, x);\n"
                         "}\n",
                     dim->y, kvarNames->sizeM, coordY,
                     dim->x, kvarNames->sizeN, coordX,
                     dim->y, dim->x,
                     optFuncName, uvars.result, coordY, coordX, uvars.ld,
                     betaStr,
                     genericFuncName, uvars.result, coordY, coordX, uvars.ld,
                     betaStr);
        }

        kgenAddStmt(ctx, tmp);
    }
    else {
        // inline result update
        flags |= UPRES_INLINE;

        op = (flags & UPRES_WITH_BETA) ? UPRES_SUM : UPRES_SET;

        uvars.startRow = coordY;
        uvars.startCol = coordX;
        uvars.nrRows = "y";
        uvars.nrCols = "x";

        if (!(kflags & (KEXTRA_TAILS_M_LOWER | KEXTRA_TAILS_N_LOWER))) {
            ret = updateResultGen(ctx,
                gset,
                funcID,
                op,
                flags,
                &uvars);
        }
        else {
            sprintf(tmp, "uint y = min(%luu, %s - (uint)%s);\n"
                         "uint x = min(%luu, %s - (uint)%s);\n",
                    dim->y, kvarNames->sizeM, coordY,
                    dim->x, kvarNames->sizeN, coordX);
            kgenAddStmt(ctx, tmp);

            sprintf(tmp, "if ((y == %lu) && (x == %lu))",
                    dim->y, dim->x);
            kgenBeginBranch(ctx, tmp);

            // optimized update
            updateResultGen(ctx,
                gset,
                funcID,
                op,
                flags,
                &uvars);

            kgenEndBranch(ctx, NULL);

            kgenBeginBranch(ctx, "else ");

            // not optimized update
            flags |= UPRES_GENERIC;
            updateResultGen(ctx,
                gset,
                funcID,
                op,
                flags,
                &uvars);

            ret = kgenEndBranch(ctx, NULL);
        }
    }

    if (useCondition) {
        ret = kgenEndBranch(ctx, NULL);
    }

    return (ret) ? -EOVERFLOW : 0;
}

//-----------------------------------------------------------------------------

void checkGenBeginHitMatrixBlock(
    struct KgenContext *ctx,
    KernelExtraFlags kflags)
{
    bool tailsM = (kflags & KEXTRA_TAILS_M) != 0;
    bool tailsN = (kflags & KEXTRA_TAILS_N) != 0;

    if (tailsM) {
        if (tailsN) {
            kgenBeginBranch(ctx, "if ((coord.x < N) && (coord.y < M))");
        }
        else {
            kgenBeginBranch(ctx, "if (coord.y < M)");
        }
    }
    else {
        if (tailsN) {
            kgenBeginBranch(ctx, "if (coord.x < N)");
        }
    }
}

//-----------------------------------------------------------------------------

void checkGenEndHitMatrixBlock(
    struct KgenContext *ctx,
    KernelExtraFlags kflags)
{
    if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N)) {
        kgenEndBranch(ctx, NULL);
    }
}