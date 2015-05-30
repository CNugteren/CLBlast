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

#include "../blas_kgen.h"
#include "trsm_kgen_legacy.h"

void
genUpdateIntermTrsmResult(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName,
    bool withMhitCond)
{
    char tmp[1024];
    const char *coordY, *coordX;
    char *revAlp, *alp;
    DataType dtype = gset->kextra->dtype;
    KernelExtraFlags kflags = gset->kextra->flags;
    const SubproblemDim *dim = &gset->subdims[1];
    const KernelVarNames *kvarNames = &gset->varNames;

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

    coordY = kvarNames->coordA;
    coordX = kvarNames->coordB;

    if (!(kflags & (KEXTRA_TAILS_M | KEXTRA_TAILS_N))) {
        sprintf(tmp, "%s(B, c, %s, %s, %s, ldb, %s);\n",
                optFuncName, alp, coordY, coordX, revAlp);
        kgenAddStmt(ctx, tmp);
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
                     "uint x = min(%luu, %s - (uint)%s);\n"
                     "if ((y == %luu) && (x == %luu)) {\n"
                     "    %s(B, c, %s, %s, %s, ldb, %s);\n"
                     "}\n"
                     "else {\n"
                     "    %s(B, c, %s, %s, %s, ldb, %s, y, x);\n"
                     "}\n",
                dim->y, kvarNames->sizeM, coordY,
                dim->x, kvarNames->sizeN, coordX,
                dim->y, dim->x,
                optFuncName, alp, coordY, coordX, revAlp,
                genericFuncName, alp, coordY, coordX, revAlp);

        kgenAddStmt(ctx, tmp);

        kgenEndBranch(ctx, NULL);
    }
}

void
genHeapTrsmResultToLDS(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const char *funcName,
    const char *dstName)
{
    char tmp[1024];
    char *alp;
    unsigned int l1Pans;
    DataType dtype = gset->kextra->dtype;
    const SubproblemDim *dims = gset->subdims;

    if(isComplexType(dtype)) {
        if (dtype == TYPE_COMPLEX_FLOAT) {
            alp = "(float2)(1.f, 0)";
        }
        else {
            alp = "(double2)(1., 0)";
        }
    }
    else {
        alp = "1.";
    }

    l1Pans = (unsigned int)dims[0].x / (unsigned int)dims[1].x;
    sprintf(tmp, "%s(%s, c, %s, (lid / %u * %lu), (lid %% %u * %lu), %lu);\n",
            funcName, dstName, alp, l1Pans, dims[1].y, l1Pans, dims[1].x,
            dims[0].bwidth);
    kgenAddStmt(ctx, tmp);
}

void
genInvertingBlockFunc(
    struct KgenContext *ctx,
    size_t pitch,
    DataType dtype,
    KernelExtraFlags kflags)
{
    char tmp[1024];
    const char *ctype;
    ctype = dtypeBuiltinType(dtype);

    sprintf(tmp, "void\ninvert(__local %s *src, __local %s *dst, int lid, "
                              "int lastRow)\n", ctype, ctype);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);
    kgenAddStmt(ctx, "int i, k;\n");

    if (isComplexType(dtype)) {
        sprintf(tmp, "dst[lid * %lu + lid].x = 1.f;\n", pitch);
    }
    else {
        sprintf(tmp, "dst[lid * %lu + lid] = 1.f;\n", pitch);
    }
    kgenAddStmt(ctx, tmp);

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "for (i = lastRow - 1; i >= 0; i--)");
    }
    else {
        sprintf(tmp, "for (i = 0; i < lastRow; i++)");
    }
    kgenBeginBranch(ctx, tmp);

    if (isComplexType(dtype)) {
        sprintf(tmp, "dst[i * %lu + lid] = div(dst[i * %lu + lid], "
                     "src[i * %lu + i]);\n", pitch, pitch, pitch);
    }
    else {
        sprintf(tmp, "dst[i * %lu + lid] = dst[i * %lu + lid] / "
                     "src[i * %lu + i];\n", pitch, pitch, pitch);
    }
    kgenAddStmt(ctx, tmp);

    if (isMatrixUpper(kflags)) {
        sprintf(tmp, "for (k = 0; k < i; k++)");
    }
    else {
        sprintf(tmp, "for (k = i + 1; k < %lu; k++)", pitch);
    }
    kgenBeginBranch(ctx, tmp);
    if (isComplexType(dtype)) {
        sprintf(tmp, "dst[k * %lu + lid] = dst[k * %lu + lid] - "
                     "mul(src[k * %lu + i], dst[i * %lu + lid]);\n",
                pitch, pitch, pitch, pitch);
    }
    else {
        sprintf(tmp, "dst[k * %lu + lid] = dst[k * %lu + lid] - "
                      "dst[i * %lu + lid] * src[k * %lu + i];\n",
                pitch, pitch, pitch, pitch);
    }
    kgenAddStmt(ctx, tmp);
    kgenEndBranch(ctx, NULL);
    kgenEndBranch(ctx, NULL);
    kgenEndFuncBody(ctx);
}

