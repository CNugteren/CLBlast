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
 * TODO: throw away this generator and replace it with tileMulGen() in all
 *       kernel generators
 */

#include <stdio.h>
#include <string.h>

#include <defbool.h>
#include <clblas_stddef.h>
#include <sys/types.h>
#include <kerngen.h>
#include <matrix_props.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "../blas_kgen.h"
#include "blas_kgen_legacy.h"

#define MAX_LENGTH 4096
#define BITS_INT (sizeof(int) * 8)

typedef enum VectMulType {
    VECT_MULT_REAL,
    VECT_MULT_COMPLEX_REAL,
    VECT_MULT_IMAG_FLOAT,
    VECT_MULT_IMAG_DOUBLE
} VectMulType;

static __inline bool
isPower2(size_t a)
{
    return (a && ((a & (a - 1)) == 0));
}

/*
 * get vector chunk size to copy
 * taking into account its alignment
 */
static unsigned int
vecChunkSize(size_t offset, size_t vecLen)
{
    size_t chunk;

    for (chunk = vecLen; (chunk > 1) && (offset % chunk); chunk /= 2) { }

    return (unsigned int)chunk;
}

static void
getCyclicAddrData(
    BlkMulFlags flags,
    const char **op,
    size_t *value,
    size_t bound)
{
    if (isPower2(bound) && !(flags & BLKMUL_AVOID_AND)) {
        *op = "&";
        *value = bound - 1;
    }
    else {
        *op = "%";
        *value = bound;
    }
}

static void
sprintfInputOffset(
    char *buf,
    MatrixRole mrole,
    int row,
    int col,
    size_t vecPitch,
    size_t bheight,
    const BlkMulOpts *opts,
    BlkmulArgNames *argNames,
    bool singleStepK)
{
    const char *vfield;
    const char *coordName;
    const char *op;
    size_t bound;
    char colOff[64], rowOff[64];
    CLMemType mtype;
    BlkMulFlags flags = opts->flags;

    vfield = (mrole == MATRIX_A) ? "y" : "x";
    mtype = (mrole == MATRIX_A) ? opts->aMobj : opts->bMobj;
    if ((mrole == MATRIX_B) && (flags & BLKMUL_TRANSPOSED_B)) {
        flags &= ~BLKMUL_SKEW_ROW;
    }

    if (flags & BLKMUL_SKEW_ROW) {
        getCyclicAddrData(flags, &op, &bound, bheight);
        sprintf(rowOff, "((%s.%s + %d) %s %lu)",
                argNames->skewRow, vfield, row, op, bound);
    }
    else {
        sprintf(rowOff, "%d", row);
    }

    if (flags & BLKMUL_SKEW_COLUMN) {
        getCyclicAddrData(flags, &op, &bound, vecPitch);
        if (flags & BLKMUL_INLINE) {
            if (singleStepK) {
                sprintf(colOff, "%d", col);
            }
            else {
                sprintf(colOff, "(%s + %s + %d) %% %s",
                        argNames->skewCol, argNames->k, col,
                        argNames->vectBoundK);
            }
        }
        else {
            if (singleStepK) {
                sprintf(colOff, "%s", argNames->skewCol);
            }
            else {
                sprintf(colOff, "((skewCol + k + %d) %s %lu)",
                        col, op, bound);
            }
        }
    }
    else {
        sprintf(colOff, "%d", col);
    }

    if (mtype == CLMEM_IMAGE) {
        coordName = (mrole == MATRIX_A) ? argNames->coordA : argNames->coordB;
        if (flags & BLKMUL_IMAGE_PACKED) {
            sprintf(buf, "(int2)(%s.x + mad24(%s, %lu, %s), %s.y)",
                    coordName, rowOff, vecPitch, colOff, coordName);
        }
        else {
            sprintf(buf, "(int2)(%s.x + %s, %s.y + %s)",
                    coordName, colOff, coordName, rowOff);
        }
    }
    else {
        if (flags & BLKMUL_SKEW_ROW) {
            sprintf(buf, "mad24(%s, %lu, %s)", rowOff, vecPitch, colOff);
        }
        else {
            sprintf(buf, "%lu + %s", row * vecPitch, colOff);
        }
    }
}

static void
genRealDot(
    struct KgenContext *ctx,
    size_t m,
    size_t n,
    size_t nrCols,
    size_t lenK,
    unsigned int vecLen)
{
    size_t k;
    char tmp[MAX_LENGTH], prefix[MAX_LENGTH];
    const char *vect = "xyzw";
    size_t regPitch = nrCols;
    size_t off;

    if (regPitch % vecLen) {
        regPitch += vecLen - regPitch % vecLen;
    }

    off = m * regPitch + n;
    sprintf(prefix, "c[%lu].%c += ", off / vecLen, vect[off % vecLen]);

    for (k = 0; k < lenK / vecLen; k++) {
        off = n * lenK / vecLen + k;
        sprintf(tmp, "%sdot(a[%lu], b[%lu]);\n", prefix, k, off);
        kgenAddStmt(ctx, tmp);
    }
}

/*
 * sprintf vector multiplication expression
 */
static void
genVecMul(
    struct KgenContext *ctx,
    size_t currCol,
    size_t lenK,
    VectMulType type)
{
    size_t k;
    char tmp[MAX_LENGTH];
    const char *suff[] = {"", "", ".yxwz", ".yx"};

    sprintf(tmp, "sum = a[%d] * b[%lu]%s", 0, currCol * lenK, suff[type]);
    for (k = 1; k < lenK; k++) {
        sprintf(tmp, "%s + a[%lu] * b[%lu]%s", tmp, k,
                currCol * lenK + k, suff[type]);
    }
    strcat(tmp, ";\n");
    kgenAddStmt(ctx, tmp);
}

/*
 * sprintf vector multiplication expression using mad()'s
 */
static void
genMadMul(
    struct KgenContext *ctx,
    size_t currCol,
    size_t lenK,
    VectMulType type)
{
    size_t k;
    char tmp[MAX_LENGTH];
    const char *suff[] = {"", "", ".yxwz", ".yx"};

    sprintf(tmp, "sum = a[%d] * b[%lu]%s;\n", 0, currCol * lenK,
            suff[type]);
    for (k = 1; k < lenK; k++) {
        sprintf(tmp, "%ssum = mad(a[%lu], b[%lu]%s, sum);\n", tmp, k,
                currCol * lenK + k, suff[type]);
    }
    kgenAddStmt(ctx, tmp);
}


/*
 * sprint expression for all the vector components
 * accumulation
 */
static void
genVecSum(
    struct KgenContext *ctx,
    DataType dataType,
    size_t currRow,
    size_t currCol,
    size_t nrCols,
    unsigned int vecLen,
    VectMulType mulType)
{
    const char *vect = "xyzw";
    unsigned long vecOff, regOff;
    char c;
    unsigned int k;
    size_t pitch = nrCols;
    char tmp1[MAX_LENGTH], tmp2[MAX_LENGTH];
    unsigned int sumLen;

    // get offset taking into account alignment
    if ((pitch % vecLen) && !isComplexType(dataType)) {
        pitch += vecLen - pitch % vecLen;
    }

    regOff = (unsigned int)(currRow * pitch + currCol);
    if (isComplexType(dataType)) {
        vecOff = (mulType == VECT_MULT_COMPLEX_REAL) ? 0 : 1;
        sumLen = vecLen * 2;
    }
    else {
        vecOff = regOff % vecLen;
        regOff /= vecLen;
        sumLen = vecLen;
    }

    sprintf(tmp1, " sum.x");
    for (k = 1; k < sumLen; k++) {
        c = ((mulType == VECT_MULT_COMPLEX_REAL) && (k & 1)) ? '-' : '+';
        sprintf(tmp1, "%s %c sum.%c", tmp1, c, vect[k]);
    }

    sprintf(tmp2, "c[%lu].%c += %s;\n", regOff, vect[vecOff], tmp1);
    kgenAddStmt(ctx, tmp2);
}

/*
 * vector multiplication expression using mad() operations
 */
static void
genMad(
    struct KgenContext *ctx,
    DataType dataType,
    size_t currRow,
    size_t currCol,
    size_t nrCols,
    size_t lenK,
    unsigned int vecLen,
    bool vectorized)
{
    const char *vect = {"xyzw"};
    unsigned long vecOff, regOff;
    unsigned int k;
    size_t pitch = nrCols;
    char tmp[MAX_LENGTH];
    unsigned int sumLen;
    int bIndex;

    // get offset taking into account alignment
    if ((pitch % vecLen) && !isComplexType(dataType)) {
        pitch += vecLen - pitch % vecLen;
    }
    regOff = (unsigned int)(currRow * pitch + currCol);
    vecOff = (unsigned int)(regOff % vecLen);

    if (isComplexType(dataType)) {
        sumLen = vecLen * 2;
        for (k = 0; k < lenK; k++) {
            int aIndex = k;
            bIndex = (int)(currCol * lenK + k);

            sprintf(tmp, "c[%lu] = mad(a[%d].xy, (float2)(b[%d].x), c[%lu]);\n",
                    regOff, aIndex, bIndex, regOff);
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, "c[%lu] = mad(a[%d].yx, (float2)(-b[%d].y, b[%d].y), c[%lu]);\n",
                    regOff, aIndex, bIndex, bIndex, regOff);
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, "c[%lu] = mad(a[%d].zw, (float2)(b[%d].z), c[%lu]);\n",
                    regOff, aIndex, bIndex, regOff);
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, "c[%lu] = mad(a[%d].wz, (float2)(-b[%d].w, b[%d].w), c[%lu]);\n",
                    regOff, aIndex, bIndex, bIndex, regOff);
            kgenAddStmt(ctx, tmp);
        }
    }
    else {
        // Real case
        if (vectorized) {
            const char *tname = (isDoubleBasedType(dataType)) ? "double" : "float";

            regOff = (unsigned int)(currRow * nrCols / vecLen + currCol);
            for (k = 0; k < lenK * vecLen; k++) {
                bIndex = (int)(currCol * lenK * vecLen + k);
                sprintf(tmp, "c[%lu] = mad((%s%u)a[%u].%c, b[%d], c[%lu]);\n",
                        regOff, tname, vecLen, k / vecLen, vect[k % vecLen],
                        bIndex, regOff);
                kgenAddStmt(ctx, tmp);
            }
        }
        else {
            int dimNum;

            regOff /= vecLen;
            sumLen = vecLen;
            if (isDoubleBasedType(dataType)) {
                dimNum = 2;
            }
            else {
                dimNum = 4;
            }

            for (k = 0; k < sumLen*lenK; k++) {
                sprintf(tmp, "c[%lu].%c = mad(a[%u].%c, b[%lu].%c, "
                                                    "c[%lu].%c);\n",
                        regOff, vect[vecOff], k / sumLen, vect[k % dimNum],
                        currCol * lenK + (k / sumLen), vect[k % dimNum],
                        regOff, vect[vecOff]);
                kgenAddStmt(ctx, tmp);
            }
            kgenAddBlankLine(ctx);
        }
    }
}

static void
getUpdateSkewCoords(
    struct KgenContext *ctx,
    const BlkMulOpts *opts,
    size_t subK,
    size_t pitchA,
    size_t pitchB,
    unsigned int vecLen,
    const char *ptrNameIn)
{
   char tmp[1024];
   bool trb = ((opts->flags & BLKMUL_TRANSPOSED_B) != 0);

   if (!(opts->flags & BLKMUL_SKEW_COLUMN)) {
        kgenAddBlankLine(ctx);
        if (opts->aMobj == CLMEM_IMAGE) {
            sprintf(tmp, "coordA.x += %lu;\n", subK / vecLen);
        }
        else {
            sprintf(tmp, "A.%s += %lu;\n", ptrNameIn, subK / vecLen);
        }
        kgenAddStmt(ctx, tmp);

        if (!trb) {
            subK /= vecLen;
        }
        if (opts->bMobj == CLMEM_IMAGE) {
            const char *vfield = (trb) ? "y" : "x";

            sprintf(tmp, "coordB.%s += %lu;\n", vfield, subK);
        }
        else {
            size_t u = (trb) ? (subK * pitchB / vecLen) : subK;

            sprintf(tmp, "B.%s += %lu;\n", ptrNameIn, u);
        }
        kgenAddStmt(ctx, tmp);
    }
    else if (subK == vecLen) {
        if (isPower2(pitchA / vecLen)) {
            sprintf(tmp, "\nskewCol = (skewCol + 1) & %lu;\n",
                    pitchA / vecLen - 1);
        }
        else {
            sprintf(tmp, "\nskewCol = (skewCol + 1) %% %lu;\n",
                    pitchA / vecLen);
        }
        kgenAddStmt(ctx, tmp);
    }
}

// MUST BE LATER DEPRECATED
static void
genScaleAccResults(
    struct KgenContext *ctx,
    DataType dtype,
    size_t m,
    size_t n,
    size_t outPitch,
    unsigned int vecLen,
    bool transpose)
{
    char s[MAX_LENGTH];
    const char *vect = "xyzw";
    char vecChunk[6];
    size_t inOff = 0, outOff, vecOff;
    size_t regPitch = n;
    size_t i, j, k;
    bool isDouble;
    const char *ptrNames[2][4] = {
        {"f", "f2v", "", "f4v"},
        {"d", "d2v", "", ""}};

    if ((regPitch % vecLen) && !isComplexType(dtype)) {
        regPitch += vecLen - regPitch % vecLen;
    }

    isDouble = isDoubleBasedType(dtype);
    for (i = 0; i < m; i++) {
        j = 0;
        inOff = i * regPitch;

        do {
            /*
             * get power of 2 size vector element to copy
             * in the case without transposing and copy
             * just with single element in the case with
             * transposing
             */

            if (transpose) {
                k = 1;
                outOff = (j * outPitch + i);
            }
            else {
                if (isComplexType(dtype)) {
                    k = 1;
                }
                else {
                    k = vecChunkSize(j, vecLen);
                    k = szmin(k, n - j);
                }
                outOff = (i * outPitch + j);
            }

            if (isComplexType(dtype)) {
                sprintf(s, "tempC.%s[%lu] += "
                           "c[%lu] * alphaR + c[%lu].yx * alphaI;\n",
                        ptrNames[isDouble][1], outOff, inOff, inOff);
            }
            else {
                if (k == vecLen) {
                    strcpy(vecChunk, "");
                }
                else {
                    vecOff = inOff % vecLen;
                    strcpy(vecChunk, ".");
                    strncat(vecChunk, &vect[vecOff], k);
                }

                sprintf(s, "tempC.%s[%lu] += c[%lu]%s * alpha;\n",
                        ptrNames[isDouble][k - 1], outOff / k,
                        inOff / vecLen, vecChunk);
            }

            kgenAddStmt(ctx, s);

            j += k;
            inOff += k;
        } while (j < n);
    }
}

static void
declareBlkMul(
    struct KgenContext *ctx,
    DataType dtype,
    size_t m,
    size_t n,
    const BlkMulOpts *opts,
    BlkmulArgNames *argNames)
{
    char s[MAX_LENGTH];
    const char *s1;
    char c;
    const char *typeName;
    bool isPriv = (opts->flags & BLKMUL_OUTPUT_PRIVATE);

    c = dtypeToBlasPrefix(dtype);
    typeName = dtypeBuiltinType(dtype);
    s1 = (opts->flags & BLKMUL_TRANSPOSE) ? "Transp" : "";

    // fill argument names
    argNames->coordA = "coordA";
    argNames->coordB = "coordB";
    argNames->skewRow = "skewRow";
    argNames->skewCol = "skewCol";

    sprintf(s, "void\n"
               "%cgemmBlock%s_%lu_%lu(\n",
            c, s1, m, n);

    if (!isPriv) {
        sprintf(s, "%s    %s alpha,\n", s, typeName);
    }
    if (opts->aMobj == CLMEM_IMAGE) {
        sprintf(s, "%s    __read_only image2d_t A,\n"
                   "    int2 coordA,\n", s);
    }
    else {
        sprintf(s, "%s    LPtr A,\n", s);
    }
    if (opts->bMobj == CLMEM_IMAGE) {
        sprintf(s, "%s    __read_only image2d_t B,\n"
                   "    int2 coordB,\n", s);
    }
    else {
        sprintf(s, "%s    LPtr B,\n", s);
    }

    if (opts->flags & BLKMUL_OUTPUT_PRIVATE) {
        if (isDoubleBasedType(dtype)) {
            typeName = "double2";
        }
        else {
            typeName = (dtype == TYPE_COMPLEX_FLOAT) ? "float2" : "float4";
        }
        sprintf(s, "%s    %s *c", s, typeName);
    }
    else {
        sprintf(s, "%s    LPtr tempC", s);

    }

    if (opts->flags & BLKMUL_SKEW_ROW) {
        sprintf(s, "%s,\n    int2 skewRow", s);
    }
    if (opts->flags & BLKMUL_SKEW_COLUMN) {
        sprintf(s, "%s,\n    int skewCol", s);
    }
    strcat(s, ")\n");

    kgenDeclareFunction(ctx, (const char*)s);
}

int
blkMulGen(
    struct KgenContext *ctx,
    const SubproblemDim subdims[2],
    DataType dtype,
    const BlkMulOpts *opts)
{
    char s[MAX_LENGTH], s1[MAX_LENGTH];
    const char *tNameIn, *tNameOut, *ptrNameIn;
    size_t vecLen, vlenJ, vlenK;
    size_t i, j, k;
    size_t m, n, subK;
    unsigned int nrRegs;
    int ret = 0;
    bool isReal, isDouble;
    bool isImageA, isImageB;
    size_t off;
    size_t pitchA, pitchB, pitchC;
    unsigned int tsize = dtypeSize(dtype);
    bool transpose = (opts->flags & BLKMUL_TRANSPOSE);
    bool trb = ((opts->flags & BLKMUL_TRANSPOSED_B) != 0);
    bool isPriv = (opts->flags & BLKMUL_OUTPUT_PRIVATE);
    bool isInlined = (opts->flags & BLKMUL_INLINE);
    BlkmulCore core = opts->core;
    BlkmulArgNames argNames;
    // code to fetch from images for double and float based types
    const char *imageFetch[2] = {
        "%c[%lu] = as_float4(read_imageui(%s, sampler, %s));\n",
        "%c[%lu] = as_double2(read_imageui(%s, sampler, %s));\n"};

    if (trb && (opts->flags & BLKMUL_SKEW_COLUMN)) {
        return -EINVAL;
    }

    memcpy(&argNames, &opts->argNames, sizeof(BlkmulArgNames));
    strcpy(s, "");

    isImageA = (opts->aMobj == CLMEM_IMAGE);
    isImageB = (opts->bMobj == CLMEM_IMAGE);

    m = subdims[1].y;
    n = subdims[1].x;
    subK = subdims[1].bwidth;
    tsize = dtypeSize(dtype);

    // matrix block pitches
    pitchA = matrBlockPitch(subdims, MATRIX_A, dtype, clblasLeft);
    k = (trb) ? subdims[0].x : subdims[0].bwidth;
    pitchB = fl4RowWidth(k, tsize) * sizeof(cl_float4) / tsize;
    pitchC = matrBlockPitch(subdims, MATRIX_C, dtype, clblasLeft);

    isReal = !isComplexType(dtype);
    isDouble = isDoubleBasedType(dtype);

    vecLen = FLOAT4_VECLEN * sizeof(cl_float) / tsize;
    if (isDouble) {
        tNameIn = "double2";
        ptrNameIn = "d2v";
    }
    else {
        tNameIn = "float4";
        ptrNameIn = "f4v";
    }

    getResultGPRsInfo(dtype, &subdims[1], (unsigned int)vecLen, &nrRegs, &tNameOut);

    if (!isInlined) {
        declareBlkMul(ctx, dtype, m, n, opts, &argNames);
        kgenBeginFuncBody(ctx);
    }

    //variables declaration
    if (isImageA || isImageB) {
        kgenAddStmt(ctx, "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE "
            "| CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n");
    }
    if (!isInlined) {
        strcpy(s, "uint k;\n");
    }
    sprintf(s, "%s%s a[%lu], b[%lu];\n",s , tNameIn, subK / vecLen,
			n * subK / vecLen);

    if (!isPriv) {
        // declare registers for result
        sprintf(s, "%s%s c[%u];\n", s, tNameOut, nrRegs);
    }

    // 'dot' function can't be used for complex types
    if (isComplexType(dtype) && (core == BLKMUL_DOT)) {
        core = BLKMUL_SEPARATE_MULADD;
    }

    if ((core == BLKMUL_SEPARATE_MULADD) || isComplexType(dtype)) {
        sprintf(s,"%s%s sum;\n", s, tNameIn);
    }

    kgenAddStmt(ctx, s);

    if (!isPriv && !isReal) {
        declareComplexMultParts(ctx, "alpha", tNameOut);
    }
    kgenAddBlankLine(ctx);

    // zeroing temporary multiplication data stored to registers
    if (!isPriv) {
        sprintf(s, "for (k = 0; k < %u; k++) {\n"
                   "   c[k] = 0;\n"
                   "}\n\n", nrRegs);
        kgenAddStmt(ctx, s);
    }

    //main loop start
    if (!isInlined) {
        // initial skew correction
        if ((opts->flags & BLKMUL_SKEW_COLUMN) && (subK == vecLen)) {
            if (isPower2(pitchA / vecLen) &&
                    !(opts->flags & BLKMUL_AVOID_AND)) {
                sprintf(s, "skewCol = skewCol & %lu;\n", pitchA / vecLen - 1);
            }
            else {
                sprintf(s, "\nskewCol = skewCol %% %lu;\n", pitchA / vecLen);
            }
            kgenAddStmt(ctx, s);
        }
        sprintf(s, "\nfor (k = 0; k < %lu; k += %lu)",
                subdims[0].bwidth / vecLen, subK / vecLen);
        ret = kgenBeginBranch(ctx, s);
    }

    if (trb) {
        vlenJ = vecLen;
        vlenK = 1;
    }
    else {
        vlenJ = 1;
        vlenK = vecLen;
    }

    for (j = 0; j < n / vlenJ; j++) {
        // fetch elements of matrix B
        for (k = 0; k < subK / vlenK; k++) {
            size_t coords[2] = {k, j};
            if (trb) {
                off = j * subK + k;
            }
            else {
                off = j * subK / vecLen + k;
            }
            sprintfInputOffset(s1, MATRIX_B, (int)coords[1 - trb],
                              (int)coords[trb], pitchB / vecLen,
                               subdims[1].x, opts, &argNames, (subK == vecLen));
            if (isImageB) {
                sprintf(s, imageFetch[isDouble], 'b', off, "B", s1);
            }
            else {
                sprintf(s, "b[%lu] = B.%s[%s];\n", off, ptrNameIn, s1);
            }
            ret = kgenAddStmt(ctx, s);
        }
    }

    for (i = 0; i < m; i++) {
        kgenAddBlankLine(ctx);
        // fetch elements of matrix A from single row
        for (k = 0; k < subK / vecLen; k++) {
            sprintfInputOffset(s1, MATRIX_A, (int)i,
                               (int)k, pitchA / vecLen, subdims[1].y, opts,
                               &argNames, (subK == vecLen));
            if (isImageA) {
                sprintf(s, imageFetch[isDouble], 'a', k, "A", s1);
            }
            else {
                sprintf(s,"a[%lu] = A.%s[%s];\n", k, ptrNameIn, s1);
            }
            ret = kgenAddStmt(ctx, s);
        }

        // multiply matrix A row on matrix B block
        for (j = 0; j < n / vlenJ; j++) {
            if (isReal) { //real case
                switch (core) {
                case BLKMUL_DOT:
                    genRealDot(ctx, i, j, n, subK, (unsigned int)vecLen);
                    break;
                case BLKMUL_MAD:
                    genMad(ctx, dtype, i, j, n, subK / vecLen,
                           (unsigned int)vecLen, trb);
                    break;
                case BLKMUL_SEPARATE_MULADD:
                    genVecMul(ctx, j, subK / vecLen, VECT_MULT_REAL);
                    genVecSum(ctx, dtype, i, j, n, (unsigned int)vecLen,
                            VECT_MULT_REAL);
                    break;
                }
            }
            else { //complex case
                VectMulType mulType = (dtype == TYPE_COMPLEX_FLOAT) ?
                        VECT_MULT_IMAG_FLOAT : VECT_MULT_IMAG_DOUBLE;

                if (core == BLKMUL_MAD) {
                    //real part
                    genMadMul(ctx, j, subK / vecLen, VECT_MULT_COMPLEX_REAL);
                    genVecSum(ctx, dtype, i, j, n, (unsigned int)vecLen,
                            VECT_MULT_COMPLEX_REAL);

                    //imaginary part
                    genMadMul(ctx, j, subK / vecLen, mulType);
                    genVecSum(ctx, dtype, i, j, n, (unsigned int)vecLen, mulType);
                }
                else {
                    //real part
                    genVecMul(ctx, j, subK / vecLen, VECT_MULT_COMPLEX_REAL);
                    genVecSum(ctx, dtype, i, j, n, (unsigned int)vecLen,
                              VECT_MULT_COMPLEX_REAL);

                    //imaginary part
                    genVecMul(ctx, j, subK / vecLen, mulType);
                    genVecSum(ctx, dtype, i, j, n, (unsigned int)vecLen, mulType);
                }
            }
        }
    }

    // update coordinates/skews and end the loop
    if (!isInlined) {
        getUpdateSkewCoords(ctx, opts, subK, pitchA, pitchB,
                            (unsigned int)vecLen, ptrNameIn);
        kgenEndBranch(ctx, NULL);
    }

    if (!isPriv) {
        kgenAddBlankLine(ctx);
        genScaleAccResults(ctx, dtype, m, n, pitchC, (unsigned int)vecLen, transpose);
    }

    if (!isInlined) {
        ret = kgenEndFuncBody(ctx);
    }

    return ret ? -EOVERFLOW : 0;
}
