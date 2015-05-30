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

#include <matrix_props.h>
#include <matrix_dims.h>

#include "trxm_common_legacy.h"

void
declareLdsBasedTrxmVariables(
    struct KgenContext *ctx,
    DataType dtype,
    const SubproblemDim *dims,
    const PGranularity *pgran,
    bool useLocalC)
{
    char tmp[1024];
    size_t pitchAB, pitchC;
    const char *inTypeName, *outTypeName;
    unsigned int nrRegs;
    unsigned int vecLen;

    inTypeName = dtypeBuiltinType(dtype);
    pitchAB = matrBlockPitch(dims, MATRIX_A, dtype, clblasLeft);
    pitchC = matrBlockPitch(dims, MATRIX_C, dtype, clblasLeft);
    vecLen = sizeof(cl_float4) / dtypeSize(dtype);

    sprintf(tmp, "__local %s tempA[%lu];\n"
                 "__local %s tempB[%lu];\n"
                 "uint m0, k0;\n"
                 "uint currM, currN;\n"
                 "uint2 coordA, coordB;\n"
                 "uint x, y;\n",
            inTypeName, pitchAB * dims->y, inTypeName,
            pitchAB * dims->x);
    kgenAddStmt(ctx, tmp);

    getResultGPRsInfo(dtype, &dims[1], vecLen, &nrRegs, &outTypeName);
    if (useLocalC) {
        sprintf(tmp, "__local %s tempC[%lu];\n", inTypeName,
                pitchC * dims->y);
    }
    else {
        sprintf(tmp, "%s c[%u];\n", outTypeName, nrRegs);
    }

    kgenAddStmt(ctx, tmp);
    kgenDeclareLocalID(ctx, "lid", pgran);
    kgenDeclareGroupID(ctx, "gid", pgran);
    kgenAddBlankLine(ctx);
}

void
genPrepareTrxmBlockA(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags,
    const char *nameM)
{
    char tmp[1024];
    size_t pitch;
    const char *coordName[2] = {"currM", "k0"};
    const char *sizeName[2] = {"y", "x"};
    int tra;

    pitch = matrBlockPitch(dim, MATRIX_A, dtype, clblasLeft);
    tra = isMatrixAccessColMaj(CLBLAS_TRMM, flags, MATRIX_A);

    /*
     * If the (sub)problem is integrally divisible,
     * skip any checks, and just read with optimal blocks,
     * otherwise check for tails and then read with a
     * fast function in the case of optimal blocks, and with
     * the slow one in the case of tails respectively
     */

    if (!(flags & KEXTRA_TAILS_M)) {
        sprintf(tmp, "%s((LPtr)tempA, (GPtr)A, %s, %s, lda);\n",
                copyFuncs->read[MATRIX_A], coordName[tra], coordName[1 - tra]);
    }
    else {
        sprintf(tmp,
                "y = (currM + %lu <= M) ? %lu : M - currM;\n"
                "x = (k0 + %lu <= %s) ? %lu : %s - k0;\n"
                "if ((y == %lu) && (x == %lu)) {\n"
                     // fast read
                "    %s((LPtr)tempA, (GPtr)A, %s, %s, lda);\n"
                "}\n"
                "else {\n"
                "    %s((__local float4*)tempA);\n"           // zeroing
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                     // slow read
                "    %s((LPtr)tempA, (GPtr)A, %s, %s, %s, %s, %lu, lda);\n"
                "}\n\n",
                dim->y, dim->y, dim->bwidth, nameM, dim->bwidth, nameM, dim->y,
                dim->bwidth, copyFuncs->read[MATRIX_A], coordName[tra],
                coordName[1 - tra], zeroFuncs->names[MATRIX_A],
                copyFuncs->readGeneric[MATRIX_A], coordName[tra],
                coordName[1 - tra], sizeName[tra], sizeName[1 - tra],
                pitch);
    }

    kgenAddStmt(ctx, tmp);
}

void
genPrepareTrxmBlockB(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    const CopyBufFuncs *copyFuncs,
    const ZeroFuncs *zeroFuncs,
    KernelExtraFlags flags)
{
    char tmp[1024];
    size_t pitch;
    const char *coordName[2] = {"currN", "k0"};
    const char *sizeName[2] = {"y", "x"};
    int trb;

    trb = isMatrixAccessColMaj(CLBLAS_TRMM, flags, MATRIX_B);
    pitch = matrBlockPitch(dim, MATRIX_B, dtype, clblasLeft);

    if (!(flags & (KEXTRA_TAILS_N | KEXTRA_TAILS_K))) {
        sprintf(tmp, "%s((LPtr)tempB, (GPtr)B, %s, %s, ldb);\n",
                copyFuncs->read[MATRIX_B], coordName[trb],
                coordName[1 - trb]);
    }
    else {
        sprintf(tmp,
                "y = (currN + %lu <= N) ? %lu : N - currN;\n"
                "x = (k0 + %lu <= M) ? %lu : M - k0;\n"
                "if ((y == %lu) && (x == %lu)) {\n"
                     // fast read
                "    %s((LPtr)tempB, (GPtr)B, %s, %s, ldb);\n"
                "}\n"
                "else {\n"
                "    %s((__local float4*)tempB);\n"           // zeroing
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"    // barrier if it's needed
                             // slow read
                "    %s((LPtr)tempB, (GPtr)B, %s, %s, %s, %s, %lu, ldb);\n"
                "}\n\n",
                dim->x, dim->x, dim->bwidth, dim->bwidth, dim->x, dim->bwidth,
                copyFuncs->read[MATRIX_B], coordName[trb], coordName[1 - trb],
                zeroFuncs->names[MATRIX_B],
                copyFuncs->readGeneric[MATRIX_B], coordName[trb],
                coordName[1 - trb], sizeName[trb], sizeName[1 - trb], pitch);
    }

    kgenAddStmt(ctx, tmp);
}

void
genTriangMatrBlock(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    DataType dtype,
    KernelExtraFlags kflags)
{
    char tmp[1024], tmp1[512];
    const char *one;
    size_t pitch;

    pitch = matrBlockPitch(dim, MATRIX_A, dtype, clblasLeft);
    one = strOne(dtype);

    strcpy(tmp1, "");
    // staring diagonal coordinates
    kgenAddStmt(ctx, "y = (k0 < currM) ? 0 : (k0 - currM);\n"
                     "x = (k0 < currM) ? (currM - k0) : 0;\n\n");

    if (isMatrixUpper(kflags)) {
        /*
         * resulting block is upper diagonal, zeroing everything
         * below the diagonal and set "1" on the diagonal for the
         * unit diagonal matrix
         */
        if (kflags & KEXTRA_UNIT_DIAGONAL) {
            sprintf(tmp1, "\n"
                          "    if (x < %lu) {\n"
                          "        tempA[lid * %lu + x] = %s;\n"
                          "    }\n",
                    dim->bwidth, pitch, one);
        }

        sprintf(tmp, "if (lid >= y && lid < %lu) {\n"
                     "    uint i;\n"
                     "\n"
                     "    x = x + lid - y;\n"
                     "    x = (x > %lu) ? %lu : x;\n"
                     "\n"
                     "    for (i = 0; i < x; i++) {\n"
                     "        tempA[lid * %lu + i] = 0;\n"
                     "    }\n"
                     "%s"
                     "}\n",
                dim->y, dim->bwidth, dim->bwidth, pitch, tmp1);
    }
    else {
        /*
         * resulting block is lower diagonal, zeroing everything
         * above the diagonal and set "1" on the diagonal for the
         * unit diagonal matrix
         */
        if (kflags & KEXTRA_UNIT_DIAGONAL) {
            sprintf(tmp1, "\n"
                          "    if (y < %lu) {\n"
                          "        tempA[y * %lu + lid] = %s;\n"
                          "    }\n",
                    dim->y, pitch, one);
        }

        sprintf(tmp, "if (lid >= x && lid < %lu) {\n"
                     "    uint i;\n"
                     "\n"
                     "    y = y + lid - x;\n"
                     "    y = (y > %lu) ? %lu : y;\n"
                     "\n"
                     "    for (i = 0; i < y; i++) {\n"
                     "        tempA[i * %lu + lid] = 0;\n"
                     "    }\n"
                     "%s"
                     "}\n",
                dim->bwidth, dim->y, dim->y, pitch, tmp1);
    }

    kgenAddStmt(ctx, tmp);
    kgenAddBarrier(ctx, CLK_LOCAL_MEM_FENCE);

    kgenAddBlankLine(ctx);
}
