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
#include "gen_helper_legacy.h"
#include "blas_kgen_legacy.h"
#include "../gen_helper.h"

typedef struct CopyPattern {
    SubproblemDim dim;
    const PGranularity *pgran;
    DataType dtype;
    DBlockCopyDirection dir;
    DBlockCopyFlags flags;
    bool generic;
    bool zeroing;
} CopyPattern;

static int
cpyImgGenCallback(struct KgenContext *ctx, const void *pattern)
{
    const CopyPattern *pat = (CopyPattern*)pattern;
    const void *dim = (pat->generic) ? NULL : &pat->dim;
    if(pat->zeroing) {
        return f4zeroBlockGen(ctx, dim, pat->pgran, "__local");
    }
    else {
        return copyDataBlockGen(ctx, dim, pat->pgran, pat->dtype, pat->dir,
                                pat->flags);
    }
}

int
generateImageCopyFuncs(
    CopyImgFuncs *copyFuncs,
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset)
{
    const SubproblemDim *dims = gset->subdims;
    KernelExtraFlags kflags = gset->kextra->flags;
    DataType dtype = gset->kextra->dtype;
    const PGranularity *pgran = gset->pgran;
    CopyPattern pattern;
    // mandatory flags for global to local copying
    DBlockCopyFlags glcpFlags[2] = {0, 0};
    struct KgenGuard *guard;
    unsigned int tsize;
    int ret = 0;
    bool isTra, areTails, isConjA;
    bool customize;

    if (kflags & KEXTRA_NO_COPY_VEC_A) {
        glcpFlags[0] = DBLOCK_COPY_NOT_VECTORIZE;
    }
    if (kflags & KEXTRA_NO_COPY_VEC_B) {
        glcpFlags[1] = DBLOCK_COPY_NOT_VECTORIZE;
    }

    tsize = dtypeSize(dtype);
    isTra = isMatrixAccessColMaj(funcID, kflags, MATRIX_A);
    isConjA = isMatrixConj(kflags, MATRIX_A);
    areTails = (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N));
    customize = (funcID == CLBLAS_TRMM);

    guard = createKgenGuard(ctx, cpyImgGenCallback, sizeof(CopyPattern));
    if (guard == NULL) {
        return -ENOMEM;
    }

    memset(&pattern, 0, sizeof(pattern));

    pattern.zeroing = false;
    pattern.dim = dims[0];
    pattern.dir = DBLOCK_GLOBAL_TO_IMAGE;
    pattern.dtype = dtype;
    pattern.flags = 0;
    pattern.generic = false;
    pattern.pgran = pgran;

    if (!(customize && (isTra || isConjA))) {
        pattern.dim.x = dims[0].bwidth;
        pattern.dim.y = dims[0].y;
        findGenerateFunction(guard, &pattern, copyFuncs->globalToImage[MATRIX_A],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    pattern.dim.x = dims[0].bwidth;
    pattern.dim.y = dims[0].x;
    findGenerateFunction(guard, &pattern, copyFuncs->globalToImage[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    pattern.dim.x = dims[0].bwidth;
    pattern.dim.y = dims[1].y;
    pattern.dir = DBLOCK_LOCAL_TO_IMAGE;
    findGenerateFunction(guard, &pattern, copyFuncs->localToImage[MATRIX_A],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    pattern.dim.x = dims[0].bwidth;
    pattern.dim.y = dims[1].x;
    pattern.dir = DBLOCK_LOCAL_TO_IMAGE;
    findGenerateFunction(guard, &pattern, copyFuncs->localToImage[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    // Global to local optimized
    pattern.dir = DBLOCK_GLOBAL_TO_LOCAL;
    if (customize || isComplexType(dtype)) {
        pattern.flags = (!customize || isConjA) ? DBLOCK_COPY_CONJUGATE : 0;
        pattern.flags |= glcpFlags[0];
        pattern.dim.x = dims[0].bwidth;
        pattern.dim.y = dims[1].y;
        findGenerateFunction(guard, &pattern, copyFuncs->globalToLocal[MATRIX_A],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    if ((funcID == CLBLAS_GEMM) && isComplexType(dtype)) {
        pattern.flags = DBLOCK_COPY_CONJUGATE | glcpFlags[1];
        pattern.dim.x = dims[0].bwidth;
        pattern.dim.y = dims[1].x;
        findGenerateFunction(guard, &pattern, copyFuncs->globalToLocal[MATRIX_B],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    // Global to local generic
    pattern.dim = dims[0];
    pattern.dir = DBLOCK_GLOBAL_TO_LOCAL;
    pattern.generic = true;
    if (!customize || areTails) {
        pattern.flags = (isConjA) ? DBLOCK_COPY_CONJUGATE : 0;
        pattern.flags |= glcpFlags[0];
        findGenerateFunction(guard, &pattern,
                             copyFuncs->globalToLocalGeneric[MATRIX_A],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    pattern.flags = (kflags & KEXTRA_CONJUGATE_B) ? DBLOCK_COPY_CONJUGATE : 0;
    pattern.flags |= glcpFlags[1];
    findGenerateFunction(guard, &pattern,
                         copyFuncs->globalToLocalGeneric[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    // Global to local transposed functions
    pattern.dir = DBLOCK_GLOBAL_TO_LOCAL;
    pattern.flags = (kflags & KEXTRA_NO_COPY_VEC_A) ?
                    DBLOCK_COPY_NOT_VECTORIZE : 0;
    pattern.flags |= glcpFlags[0];
    if (!customize || isTra) {
        pattern.generic = false;
        if (isConjA) {
            pattern.flags |= DBLOCK_COPY_TRANSPOSE | DBLOCK_COPY_CONJUGATE;
        }
        else {
            pattern.flags |= DBLOCK_COPY_TRANSPOSE;
        }
        pattern.dim.x = dims[1].y;
        pattern.dim.y = dims[0].bwidth;

        findGenerateFunction(guard, &pattern,
                             copyFuncs->globalToLocalTransposed[MATRIX_A],
                             FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    if (!customize || (isTra && areTails)) {
        pattern.generic = true;
        pattern.dim.x = 0;
        pattern.dim.y = 0;
        findGenerateFunction(guard, &pattern,
                         copyFuncs->globalToLocalTransposedGeneric[MATRIX_A],
                         FUNC_NAME_MAXLEN);
        kgenAddBlankLine(ctx);
    }

    pattern.generic = false;
    pattern.dim.x = dims[1].x;
    pattern.dim.y = dims[0].bwidth;
    if (kflags & KEXTRA_CONJUGATE_B) {
        pattern.flags = DBLOCK_COPY_TRANSPOSE | DBLOCK_COPY_CONJUGATE;
    }
    else {
        pattern.flags = DBLOCK_COPY_TRANSPOSE;
    }
    pattern.flags |= glcpFlags[1];
    findGenerateFunction(guard, &pattern,
                         copyFuncs->globalToLocalTransposed[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    pattern.generic = true;
    pattern.dim.x = 0;
    pattern.dim.y = 0;
    findGenerateFunction(guard, &pattern,
                         copyFuncs->globalToLocalTransposedGeneric[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    // generate two local zeroing functions for matrix A and matrix B blocks
    pattern.zeroing = true;
    pattern.dim = dims[0];
    pattern.generic = false;
    pattern.flags = 0;
    pattern.dim.y = 1;
    pattern.dim.x = fl4RowWidth(dims[0].bwidth, tsize) * dims[1].y;

    findGenerateFunction(guard, &pattern,
                         copyFuncs->zeroBlock[MATRIX_A],
                         FUNC_NAME_MAXLEN);
    kgenAddBlankLine(ctx);

    pattern.dim.x = fl4RowWidth(dims[0].bwidth, tsize) * dims[1].x;
    findGenerateFunction(guard, &pattern,
                         copyFuncs->zeroBlock[MATRIX_B],
                         FUNC_NAME_MAXLEN);
    ret = kgenAddBlankLine(ctx);

    destroyKgenGuard(guard);
    return ret;
}

int
generateResultUpdateOld(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName)
{
    UpdateResultFlags flags;

    flags = kextraToUpresFlags(funcID, gset->kextra->flags);

    return genResultUpdateWithFlagsOld(ctx, funcID, gset, flags,
                                       optFuncName, genericFuncName, NULL);
}

int
genResultUpdateWithFlagsOld(
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
            ret = updateResultGenOld(ctx, gset, op, flags, &uvars);
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
            updateResultGenOld(ctx, gset, op, flags, &uvars);
            kgenEndBranch(ctx, NULL);

            flags |= UPRES_GENERIC;
            kgenBeginBranch(ctx, "else ");
            // not optimized update
            updateResultGenOld(ctx, gset, op, flags, &uvars);
            ret = kgenEndBranch(ctx, NULL);
        }
    }

    if (useCondition) {
        ret = kgenEndBranch(ctx, NULL);
    }

    return (ret) ? -EOVERFLOW : 0;
}

int
genUpresFuncsWithFlags(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    UpdateResultFlags flags,
    char optFuncName[FUNC_NAME_MAXLEN],
    char genericFuncName[FUNC_NAME_MAXLEN])
{
    KernelExtraFlags kflags = gset->kextra->flags;
    UpdateResultOp op;
    int ret;

    op = (flags & UPRES_WITH_BETA) ? UPRES_SUM : UPRES_SET;

    updateResultGenOld(ctx, gset, op, flags, NULL);
    ret = kgenAddBlankLine(ctx);
    if (ret) {
        return -EOVERFLOW;
    }

    kgenGetLastFuncName(optFuncName, FUNC_NAME_MAXLEN, ctx);

    if (kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N)) {
        flags |= UPRES_GENERIC;
        updateResultGenOld(ctx, gset, op, flags, NULL);
        kgenAddBlankLine(ctx);
        kgenGetLastFuncName(genericFuncName, FUNC_NAME_MAXLEN, ctx);
    }

    return (ret) ? -EOVERFLOW : 0;
}

int
generateUpresFuncs(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    char optFuncName[FUNC_NAME_MAXLEN],
    char genericFuncName[FUNC_NAME_MAXLEN])
{
    UpdateResultFlags flags;

    flags = kextraToUpresFlags(funcID, gset->kextra->flags);

    return genUpresFuncsWithFlags(ctx, gset, flags,
                                  optFuncName, genericFuncName);
}
