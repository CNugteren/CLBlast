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
#include <clblas_stddef.h>
#include "xxmv_common.h"

static void
genMul(char *buf, size_t val, const char* type, const char* sum, const char* mul)
{
    if (mul == NULL) {
        if (sum == NULL) {
            sprintf(buf, "%lu", val);
        }
        else {
            if (val == 0) {
                sprintf(buf, "%s", sum); //zero length string
            }
            else {
                sprintf(buf, "%s + %lu", sum, val);
            }
        }
    }
    else {
        if (sum == NULL) {
            if (val == 0) {
                sprintf(buf, "0"); //zero length string
            }
            else
            if (val == 1) {
                sprintf(buf, "%s",
                    mul); //zero length string
            }
            else {
                sprintf(buf, "mad24((%s)%lu, (%s)%s, (%s)0)",
                    type, val, type, mul, type);
                //sprintf(buf, "%lu * %s", val, mul);
            }
        }
        else {
            if (val == 0) {
                sprintf(buf, "mad24((%s)%s, (%s)%s, (%s)0)",
                    type, sum, type, mul, type); //zero length string
                //sprintf(buf, "%s * %s", sum, mul);
            }
            else {
                sprintf(buf, "mad24((%s)%s + %lu, (%s)%s, (%s)0)",
                    type, sum, val, type, mul, type);
                //sprintf(buf, "(%s + %lu) * %s", sum, val, mul);
            }
        }
    }
}


void
genFetchX(
    struct KgenContext *ctx,
    Tile *tile,
    unsigned int vecLen,
    DataType dtype,
    const KernelVarNames *varNames,
    TileMulFlags tflags,
    KernelExtraFlags kflags)
{
    Kstring kstr[1];
    Tile memtile;
    char tmp[1024], strMul[128];
    unsigned int n;
    const char *ptrName;
    bool tailN = (tflags & TILEMUL_SKEW_B) != 0;
    bool incxOne = ((kflags & KEXTRA_INCX_ONE) != 0);
    bool elemFetch = ((kflags & KEXTRA_NO_COPY_VEC_B) != 0);
    unsigned int nfetch = !tailN && incxOne && !elemFetch ? vecLen : 1;

    (void)dtype;
    initTile(&memtile, NULL, tile->nrRows, tile->nrCols, nfetch,
             tile->dtype, tile->storType,  tile->trans, tile->packed);
    getVectorTypeName(tile->dtype, vecLen, NULL, &ptrName);

    if (!tailN && incxOne && !elemFetch) {
        sprintf(tmp, "const uint xk = %s / %u;\n", varNames->k, vecLen);
        kgenAddStmt(ctx, tmp);
        for (n = 0; forEachTile(kstr, n, 0, 2, tile, &memtile); n++) {
            sprintf(tmp,"%s = %s.%s[xk + %u];\n",
                        kstr[0].buf, varNames->B, ptrName, n);
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        for (n = 0; forEachTile(kstr, n, 0, 2, tile, &memtile); n++) {
            genMul(strMul, n, "int", "k", incxOne ? NULL : "incx");
            if (tailN) {
                sprintf(tmp,"%s = X[k + %u < %s ? %s : 0];\n",
                 kstr[0].buf, n, varNames->sizeK, strMul);
            }
            else {
                sprintf(tmp,"%s = X[%s];\n",kstr[0].buf, strMul);
            }
            kgenAddStmt(ctx, tmp);
        }
    }

    if (tailN) {
        for (n = 0; forEachTile(kstr, n, 0, 2, tile, &memtile); n++) {
            sprintf(tmp,"%s = k + %u < %s ? %s : 0;\n",
                        kstr[0].buf, n, varNames->sizeK, kstr[0].buf);
            kgenAddStmt(ctx, tmp);
        }
    }
}

void
setResultPos(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    const char *axVar)
{
    bool incyOne = ((kflags & KEXTRA_INCY_ONE) != 0);

    char tmp[2048];

    if (incyOne) {
        sprintf(tmp, "Y += %s;\n", axVar);
    }
    else {
        sprintf(tmp, "Y += incy * (int)%s;\n", axVar);
    }
    kgenAddStmt(ctx, tmp);
}

void
updateResultVectorTiled(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    unsigned int vecLen,
    Tile *tile)
{
    bool beta0 = ((kflags & KEXTRA_BETA_ZERO) != 0);
    bool incyOne = ((kflags & KEXTRA_INCY_ONE) != 0);
    bool tailM = ((kflags & KEXTRA_TAILS_M) != 0);
    bool isComplex = isComplexType(tile->dtype);
    unsigned int n, i;
    const char *outTypeName, *outPtrName;
    Tile result, memtile;

    char tmp[2048],strMul[256];
    Kstring kstr[2];

    if (isComplex) {
        vecLen = 1;
    }
    initTile(&result, "r", tile->nrRows, tile->nrCols, tile->nrRows,
                    tile->dtype, tile->storType, true, tile->packed);
    declareOneTileStorage(ctx, &result);

    memtile = result;
    memtile.baseName = NULL;
    memtile.vecLen = !tailM && incyOne ? vecLen : 1;
    getVectorTypeName(memtile.dtype, memtile.vecLen, &outTypeName, &outPtrName);

    sprintf(tmp,"GPtr uC;\n"
                "uC.f = Y;\n");
    kgenAddStmt(ctx, tmp);

    if (!tailM && incyOne) {
        for (n = 0; forEachTile(kstr, n, 0, 2, &result, &memtile); n++) {
            sprintf(tmp,"%s = uC.%s[%u];\n",
                        kstr[0].buf, outPtrName, n);
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        for (n = 0; forEachTile(kstr, n, 0, 2, &result, &memtile); n++) {
            genMul(strMul, n, "int", NULL, incyOne ? NULL : "incy");
            if (tailM) {
                sprintf(tmp,"%s = Y[coordA + %u >= M ? 0 : %s];\n",
                        kstr[0].buf, n, strMul);
            }
            else {
                sprintf(tmp,"%s = Y[%s];\n",
                        kstr[0].buf, strMul);
            }
            kgenAddStmt(ctx, tmp);
        }
    }

    if (isComplex) {
        const char *complVec =
                    isDoubleBasedType(tile->dtype) ? "double2" : "float2";
        Tile onetile = result;
        onetile.baseName = NULL;
        onetile.vecLen = 1;
        for (n = 0; forEachTile(kstr, n, 0, 3, &result, tile, &onetile); n++) {
            if (beta0) {
                sprintf(tmp,
                       "%s = %s * alpha.x + %s.yx * (%s)(-alpha.y, alpha.y);\n",
                       kstr[0].buf, kstr[1].buf, kstr[1].buf, complVec);
            }
            else {
                sprintf(tmp,
                        "%s = %s * beta.x + %s.yx * (%s)(-beta.y, beta.y) + "
                        "%s * alpha.x + %s.yx * (%s)(-alpha.y, alpha.y);\n",
                        kstr[0].buf, kstr[0].buf, kstr[0].buf, complVec,
                        kstr[1].buf, kstr[1].buf, complVec);
            }
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        for (n = 0; forEachTile(kstr, n, 0, 2, &result, tile); n++) {
            if (beta0) {
                sprintf(tmp, "%s = alpha * %s;\n", kstr[0].buf, kstr[1].buf);
            }
            else {
                sprintf(tmp, "%s = beta * %s + alpha * %s;\n",
                             kstr[0].buf, kstr[0].buf, kstr[1].buf);
            }
            kgenAddStmt(ctx, tmp);
        }
    }

    if (!tailM && incyOne) {
        for (i = 0; forEachTile(kstr, i, 0, 2, &result, &memtile); i++) {
            sprintf(tmp,"uC.%s[%u] = %s;\n",
                        outPtrName, i, kstr[0].buf);
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        if (!tailM) {
            for (i = 0; forEachTile(kstr, i, 0, 2, &result, &memtile); i++) {
                sprintf(tmp,"*Y = %s;\n", kstr[0].buf);
                //sprintf(tmp,"Y[%u * incy] = %s;\n", i, kstr.buf);
                kgenAddStmt(ctx, tmp);
                kgenAddStmt(ctx, "Y += incy;\n");
            }
        }
        else {
            for (n = forEachTile(NULL, 0, 0, 2, &result, &memtile);
                     n != 0; n--) {
                i = n - 1;
                forEachTile(kstr, i, 0, 2, &result, &memtile);
                genMul(strMul, i, "int", NULL, incyOne ? NULL : "incy");
                sprintf(tmp,"Y[coordA + %u >= M ? 0 : %s] = %s;\n",
                        i, strMul, kstr[0].buf);
                kgenAddStmt(ctx, tmp);
            }
        }
    }
}

void
genIncPointers(
    struct KgenContext *ctx,
    KernelExtraFlags kflags)
{
    bool incxOne = ((kflags & KEXTRA_INCX_ONE) != 0);
    bool incyOne = ((kflags & KEXTRA_INCY_ONE) != 0);

    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "A += offA;\n");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "X += offX;\n");
    }
    if (kflags & KEXTRA_CY_OFF_NOT_ZERO) {
        kgenAddStmt(ctx, "Y += offY;\n");
    }

    if (!incxOne) {
        kgenAddStmt(ctx, "X += incx > 0 ? 0 : (N - 1) * abs(incx);\n");
    }
    if (!incyOne) {
        kgenAddStmt(ctx, "Y += incy > 0 ? 0 : (M - 1) * abs(incy);\n");
    }
}

void
genStoreLocalResult(
    struct KgenContext *ctx,
    Tile *tile,
    const char *lid)
{
    Kstring kstr;
    char tmp[1024];
    unsigned int i;

    for (i = 0; forEachTile(&kstr, i, 0, 1, tile); i++) {
        sprintf(tmp, "localRes[%s][%u] = %s;\n", lid, i, kstr.buf);
        kgenAddStmt(ctx, tmp);
    }
}

void
genAddLocalResult(
    struct KgenContext *ctx,
    Tile *tile,
    const char *lid,
    unsigned int cLocal,
    unsigned int bStep)
{
    Kstring kstr;
    char tmp[1024];
    unsigned int i;

    sprintf(tmp, "for (uint i = 1; i < %u; i++)", cLocal);
    kgenBeginBranch(ctx, tmp);
    for (i = 0; forEachTile(&kstr, i, 0, 1, tile); i++) {
        sprintf(tmp, "%s += localRes[%s + i*%u][%u];\n",
                     kstr.buf, lid, bStep, i);
        kgenAddStmt(ctx, tmp);
    }
    kgenEndBranch(ctx, NULL);
}

void
genMergeResults(
    struct KgenContext *ctx,
    Tile *result,
    Tile *source)
{
    unsigned int i;
    Kstring kstr[2];
    char tmp[2048];

    for (i = 0; forEachTile(kstr, i, 0, 2, result, source); i++) {
        sprintf(tmp, "%s += %s;\n", kstr[0].buf, kstr[1].buf);
        kgenAddStmt(ctx, tmp);
    }
}

