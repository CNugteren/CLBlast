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
#include <string.h>
#include <assert.h>

#include <defbool.h>
#include <clblas_stddef.h>
#include <sys/types.h>
#include <kerngen.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "blas_kgen.h"


#define MAX_LENGTH 4096
#define BITS_INT (sizeof(int) * 8)

typedef enum VectMulType {
    VECT_MULT_REAL,
    VECT_MULT_COMPLEX_REAL,
    VECT_MULT_COMPLEX_IMAG
} VectMulType;

static const char *vectComponents = "0123456789abcdef";

static void
getVecLens(
    const BlasGenSettings *gset,
    unsigned int *vlenA,
    unsigned int *vlenB,
    unsigned int *vlenC)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    bool distVect = ((gset->flags & BGF_DISTINCT_VECLEN) != 0);

    if (vlenA != NULL) {
        *vlenA = (distVect) ? kextra->vecLenA : kextra->vecLen;
    }
    if (vlenB != NULL) {
        *vlenB = (distVect) ? kextra->vecLenB : kextra->vecLen;
    }
    if (vlenC != NULL) {
        *vlenC = (distVect) ? kextra->vecLenC : kextra->vecLen;
    }
}

static TileMulCore
checkReplaceCore(
    const BlasGenSettings *gset,
    TileMulCore core,
    bool tra,
    bool trb)
{
    const SubproblemDim *subdims = gset->subdims;
    DataType dtype = gset->kextra->dtype;
    unsigned int vlenC;

    // 'dot' function can't be used for complex types
    if (isComplexType(dtype) && (core == TILEMUL_DOT)) {
        core = TILEMUL_MULADD;
    }

    // 'dot' is supported only for one case of vectors fetch
    // where A is fetched by rows and B - by columns
    if (core == TILEMUL_DOT && !(!tra && trb)) {
        core = TILEMUL_MULADD;
    }

    // dot is not supported for vector unaligned bwidth
    getVecLens(gset, NULL, NULL, &vlenC);
    if (core == TILEMUL_DOT && (subdims[1].bwidth % vlenC != 0)) {
        core = TILEMUL_MULADD;
    }

    return core;
}

static int
checkTriggerPostFetch(
    struct KgenContext *ctx,
    const TileMulOpts *mulOpts,
    MatrixRole mrole)
{
    int ret = 0;

    if (mulOpts->postFetch) {
        ret = mulOpts->postFetch(ctx, mrole, mulOpts->postFetchPriv);
        kgenAddBlankLine(ctx);
    }

    return ret;
}

/*
 * In an expression of a complex elements swap real and imaginary parts
 */
static void
swapComplexComponents(Kstring *expr, unsigned int vecLen)
{
    char *p;
    unsigned int i;
    char tmp;

    /*
     * If the string doesn't contain a suffix of vector components, then
     * construct it from scratch in the swapped form right away, otherwise
     * swap all even and odd components
     */
    p = strchr(expr->buf, '.');
    if (p == NULL) {
        strcat(expr->buf, ".s");
        p = expr->buf + strlen(expr->buf);
        for (i = 0; i < vecLen; i++) {
            *p++ = vectComponents[2 * i + 1];
            *p++ = vectComponents[2 * i];
    }
        *p = '\0';
    }
                else {
        p = expr->buf;
        i = (unsigned int)strlen(p) - 1;
        for (; vecLen != 0; i -= 2, vecLen--) {
            tmp = p[i];
            p[i] = p[i - 1];
            p[i - 1] = tmp;
                }
            }
        }

static void
takeComplexApart(Kstring *re, Kstring *im, const Kstring *src)
{
    char *p;
    int i;

    p = strchr(src->buf, '.');
    if (p == NULL) {
        strcpy(re->buf, src->buf);
        strcat(re->buf, ".s0");
        strcpy(im->buf, src->buf);
        strcat(im->buf, ".s1");
                }
                else {
        i = (int)strlen(src->buf) - 1;

        strcpy(re->buf, src->buf);
        strcpy(im->buf, src->buf);
        re->buf[i] = '\0';
        im->buf[i - 1] = im->buf[i];
        im->buf[i] = '\0';
                }
            }

/*
 * Select physical row in tile A depending on current row in tile C
 * and storing mode of A: whole or not, transposed or not
 */
static __inline unsigned int
selectRowA(const Tile *a, unsigned int m, bool wholeA)
{
    return (a->trans || wholeA) ? m : 0;
        }

/*
 * Select physical column in tile A depending on current column in tile C
 * and storing mode of A: whole or not, transposed or not
 */
static __inline unsigned int
selectColA(const Tile *a, unsigned int k, bool wholeA)
{
    return (!a->trans || wholeA) ? k : 0;
    }

/*
 * Common line segment length of 2 tiles being arguments in tile multiplication
 */
static unsigned int
commonTileSegmentLen(const Tile *tile1, const Tile *tile2)
{
    unsigned int u1, u2;

    u1 = tileLineSegmentLen(tile1);
    u2 = tileLineSegmentLen(tile2);

    return umin(u1, u2);
            }

static void
genPointerUpdate(
    struct KgenContext *ctx,
    const char *ptrName,
    const char *ldName,
    size_t bwidth,
    size_t bheight,
    unsigned int vecLen,
    DataType dtype,
    BlasGenFlags gflags,
    bool rowMaj,
    bool isLocal)
{
    const char *uptr;
    Kstring tmp;
    const char *p;

    if (gflags & BGF_UPTRS) {
        getVectorTypeName(dtype, vecLen, NULL, &uptr);
        ksprintf(&tmp, "%s.%s", ptrName, uptr);
        p = tmp.buf;
    }
    else {
        p = ptrName;
    }

    if (rowMaj) {
        kgenPrintf(ctx, "%s += %lu;\n", p, bwidth / vecLen);
    }
    else if (isLocal) {
        kgenPrintf(ctx, "%s += %lu;\n",
                   p, bwidth * (bheight / vecLen));
    }
    else {
        Kstring ld;
        Kstring bwStr, madExpr;
        unsigned int scale;

        kstrcpy(&ld, ldName);
        ksprintf(&bwStr, "%lu", bwidth);
        scale = (gflags & BGF_LD_IN_VECTORS) ? 0 : vecLen;
        sprintfFastScalarMad(&madExpr, &bwStr, &ld, scale, NULL);
        kgenPrintf(ctx, "%s += %s;\n", p, madExpr.buf);
    }
}

static void
genRealMulUpdate(
    struct KgenContext *ctx,
    const Kstring *elA,
    const Kstring *elB,
    const Kstring *elC,
    bool transC,
    TileMulCore core)
{
    char tmp[MAX_LENGTH];
    const char *src1, *src2;

    /*
     * Select order of source operands because type of 'mad' result is
     * determined by the first operand
     */
    src1 = (transC) ? elA->buf : elB->buf;
    src2 = (transC) ? elB->buf : elA->buf;

        if (core == TILEMUL_MAD) {
        sprintf(tmp, "%s = mad(%s, %s, %s);\n",
                elC->buf, src1, src2, elC->buf);
        }
        else {
        sprintf(tmp, "%s += %s * %s;\n", elC->buf, src1, src2);
        }

    kgenAddStmt(ctx, tmp);
}

// Generate complete vector-vector product
static void
genVecMul(
    struct KgenContext *ctx,
    unsigned int m,
    unsigned int n,
    const Tile *a,
    const Tile *b,
    const Tile *c,
    bool conjA,
    bool conjB,
    TileMulCore core,
    bool wholeA)
{
    unsigned int k;
    char tmp[MAX_LENGTH];
    Kstring elA, elB, elC;
    unsigned int vlen = 0;
    bool isComplex;
    bool isDouble;

    isDouble = isDoubleBasedType(c->dtype);
    isComplex = isComplexType(c->dtype);
    if ((core == TILEMUL_DOT) && !isComplex) {
        vlen = commonTileSegmentLen(a, b);
    }
    else {
        vlen = 1;
    }

    sprintfTileElement(&elC, c, m, n, 1);
    if (!wholeA) {
        m = 0;
        }

    for (k = 0; k < a->nrCols; k += vlen) {
        sprintfTileElement(&elA, a, m, k, vlen);
        sprintfTileElement(&elB, b, k, n, vlen);

        /*
         * Using 'dot' is not valid for complex, and replaced with '*' operator
         * for unvectorized real data
         */
        if ((core == TILEMUL_DOT) && (vlen > 1)) {
            sprintf(tmp, "%s += dot(%s, %s);\n",
                    elC.buf, elA.buf, elB.buf);
        }
        else if (isComplex) {
            Kstring expr;

            sprintfComplexMulUpdate(&expr, &elC, &elA, &elB, &elC, isDouble,
                                    conjA, conjB, core);
            kgenAddStmt(ctx, expr.buf);
        }
        else {
            genRealMulUpdate(ctx, &elA, &elB, &elC, c->trans, core);
        }
    }
}

/*
 * Generate complete vector-vector product using separate multiple-add
 * operations and explicit vectorization
 */
static void
genVectorizedVecMulAdd(
    struct KgenContext *ctx,
    unsigned int m,
    unsigned int n,
    const Tile *a,
    const Tile *b,
    const Tile *c,
    bool conjA,
    bool conjB,
    VectMulType type,
    bool wholeA)
{
    unsigned int k;
    unsigned int sumLen;
    char tmp[MAX_LENGTH], tmp2[MAX_LENGTH];
    char *str = tmp;
    const char *s;
    char op;
    Kstring elA, elB, elC;
    unsigned int vlen;
    // signs for even and odd components
    int signs[2] = {0, 0};

    vlen = commonTileSegmentLen(a, b);
    if (!wholeA) {
        m = 0;
    }

    if (type == VECT_MULT_REAL) {
        sprintfTileElement(&elC, c, m, n, 1);
        sumLen = vlen;
    }
    else {
        TileElementHalf half = (type == VECT_MULT_COMPLEX_REAL) ?
            TE_HALF_LOW : TE_HALF_HIGH;

        sprintfTileElementHalf(&elC, c, m, n, half);
        sumLen = vlen * 2;
        if (type == VECT_MULT_COMPLEX_REAL) {
            if ((conjA && conjB) || (!conjA && !conjB)) {
                signs[1] = 1;
    }
        }
        else if (!(conjA && conjB)) {
            /*
             * When both the matrix are conjugated, the sum is substracted
             * from the temporary result
             */
            signs[0] = (int)conjB;
            signs[1] = (int)conjA;
        }
    }

    // initial expression
    sprintfTileElement(&elA, a, m, 0, vlen);
    sprintfTileElement(&elB, b, 0, n, vlen);
    if (type == VECT_MULT_COMPLEX_IMAG) {
        swapComplexComponents(&elB, vlen);
    }
    str += sprintf(str, "sum = %s * %s", elA.buf, elB.buf);

    // add expressions for remaining elements
    for (k = vlen; k < a->nrCols; k += vlen) {
        sprintfTileElement(&elA, a, m, k, vlen);
        sprintfTileElement(&elB, b, k, n, vlen);
        if (type == VECT_MULT_COMPLEX_IMAG) {
            swapComplexComponents(&elB, vlen);
    }
        str += sprintf(str, " + %s * %s", elA.buf, elB.buf);
    }

    strcat(tmp, ";\n");
    kgenAddStmt(ctx, tmp);

    // sum components of the temporary results
    str = tmp2;
    s = (signs[0]) ? "-" : "";
    str += sprintf(tmp2, "%ssum.s0", s);
        for (k = 1; k < sumLen; k++) {
        op = signs[k & 1] ? '-' : '+';
        str += sprintf(str, " %c sum.s%c", op, vectComponents[k]);
        }

    if ((type == VECT_MULT_COMPLEX_IMAG) && conjA & conjB) {
        op = '-';
    }
    else {
        op = '+';
    }

    sprintf(tmp, "%s %c= %s;\n", elC.buf, op, tmp2);

    kgenAddStmt(ctx, tmp);
}

/*
 * Generate one stage of vector-vector product. Iterating over M and N having
 * fixed coordinate over K.
 */
static void
genStagedVecMul(
    struct KgenContext *ctx,
    unsigned int lineA,
    unsigned int k,
    const Tile *a,
    const Tile *b,
    const Tile *c,
    bool conjA,
    bool conjB,
    TileMulCore core,
    bool wholeA)
{
    Kstring elA, elB, elC;
    unsigned int stepM, endM, stepN, vlenC;
    unsigned int i, j;
    unsigned int m, ma, ka;
    bool isDouble;
    bool isComplex;

    if (a->trans) {
        m = 0;
        endM = a->nrRows;
    }
    else {
        m = lineA;
        endM = m + 1;
    }

    isDouble = isDoubleBasedType(c->dtype);
    isComplex = isComplexType(c->dtype);

    if (( (c->trans == a->trans) || (c->trans == b->trans) ) &&
        !isComplex) {

        if (c->trans) {
            stepM = vlenC = commonTileSegmentLen(a, c);
            stepN = 1;
    }
    else {
            stepM = 1;
            stepN = vlenC = commonTileSegmentLen(b, c);
    }
    }
    else {
        stepM = stepN = 1;
        vlenC = 1;
    }

    ka = selectColA(a, k, wholeA);

    for (i = m; i < endM; i += stepM) {
        ma = selectRowA(a, i, wholeA);
        sprintfTileElement(&elA, a, ma, ka, stepM);

        for (j = 0; j < b->nrCols; j += stepN) {
            sprintfTileElement(&elB, b, k, j, stepN);
            sprintfTileElement(&elC, c, i, j, vlenC);

            if (isComplex) {
                Kstring expr;

                sprintfComplexMulUpdate(&expr, &elC, &elA, &elB, &elC,
                                        isDouble, conjA, conjB, core);
                kgenAddStmt(ctx, expr.buf);
            }
            else {
                genRealMulUpdate(ctx, &elA, &elB, &elC, c->trans, core);
            }
        }
    }
}

/* check input values like x, y, bw to be fetch vector aligned and so on */
static int
checkInput(const BlasGenSettings *gset, const TileMulOpts *mulOpts)
{
    //bool localA = (mulOpts->memA == CLMEM_LOCAL_MEMORY);
    //bool localB = (mulOpts->memB == CLMEM_LOCAL_MEMORY);
    TileMulFlags mflags = mulOpts->flags;
    //bool cyclicGlobal = ((mflags & TILEMUL_GLOBAL_CYCLIC) != 0);
    bool isReal = ! isComplexType(gset->kextra->dtype);
    bool conjA = ((mflags & TILEMUL_CONJA) != 0);
    bool conjB = ((mflags & TILEMUL_CONJB) != 0);

    // This condition is not validate the case
    // when the matrix B is in the local memory
    // and the matrix A in the global memory.
    //

    //if ((localA ||localB) && cyclicGlobal) {
    //    return -EINVAL;
    //}

    if (isReal && (conjA || conjB)) {
        /* 'Conjugated' flag can be used for complex types only */
        return -EINVAL;
    }

    return 0;
}

static void
genMulLineOnTile(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileMulOpts *mulOpts,
    unsigned int lineOffset,
    bool wholeA)
{
    TileMulFlags mflags = mulOpts->flags;
    const Tile *a = &gset->tileA;
    const Tile *b = &gset->tileBX;
    const Tile *c = &gset->tileCY;
    bool isReal;
    bool conjA, conjB;
    const SubproblemDim *subdims = gset->subdims;
    TileMulCore core;
    DataType dtype = gset->kextra->dtype;
    unsigned int j, n;

    n = (unsigned int)subdims[1].x;
    core = checkReplaceCore(gset, mulOpts->core, a->trans, b->trans);

    isReal = !isComplexType(dtype);
    conjA = ((mflags & TILEMUL_CONJA) != 0);
    conjB = ((mflags & TILEMUL_CONJB) != 0);

    if (a->trans || !b->trans) {
        unsigned int startK, endK;

        startK = (a->trans)? lineOffset : 0;
        endK = (a->trans)? lineOffset + 1 : (unsigned int)subdims[1].bwidth;
        for (j = startK; j < endK; j++) {
            genStagedVecMul(ctx, lineOffset, j, a, b, c, conjA,
                            conjB, core, wholeA);
                    }
                }
    else {
        bool vectorize = false;

        if (commonTileSegmentLen(a, b) > 1) {
            vectorize = ((mflags & TILEMUL_FORCE_VECTORIZATION) != 0);
                }
        for (j = 0; j < n; j++) {
            /* full dot product of row of A by column of B */
            if ((core == TILEMUL_MULADD) && vectorize) {
                if (isReal) {
                    genVectorizedVecMulAdd(ctx, lineOffset, j, a, b, c,
                                           false, false, VECT_MULT_REAL,
                                           wholeA);
            }
    else {
                    genVectorizedVecMulAdd(ctx, lineOffset, j, a, b, c,
                                           conjA, conjB, VECT_MULT_COMPLEX_REAL,
                                     wholeA);
                    genVectorizedVecMulAdd(ctx, lineOffset, j, a, b, c, conjA,
                                           conjB, VECT_MULT_COMPLEX_IMAG,
                                           wholeA);
            }
        }
            else {
                genVecMul(ctx, lineOffset, j, a, b, c, conjA, conjB,
                          core, wholeA);
            }
        }
            }
        }

void
sprintfComplexMulUpdate(
    Kstring *expr,
    const Kstring *dst,
    const Kstring *a,
    const Kstring *b,
    const Kstring *c,
    bool isDouble,
    bool conjA,
    bool conjB,
    TileMulCore core)
{
    Kstring swSrc1;      // swapped element of the first source
    // real and imaginary part of the second source
    Kstring reSrc2, imSrc2;
    const Kstring *src11, *src12, *src21, *src22;
    const char *sign1 = "", *sign2 = "", *sign3 = "";
    const char *baseType;

    baseType = (isDouble) ? "double2" : "float2";

    /*
     * Prepare components for multiplying. We should get the following
     * vectorized operations:
     *
     * c = b * a1 + bsw * (-a2, a2)       if both 'a' and 'b' are not conjugated
     * c = b * a1 + bsw * (a2, -a2)       if 'b' is conjugated and 'a' is not
     * c = a * b1 + asw * (-b2, b2)       if 'a' is conjugated and 'b' is not
     * c = asw * (-b2) + a * (b1, -b1)    if both 'a' and 'b' are conjugated
     *
     * Where (a1, a2) and (b1, b2) are complex components of 'a' and 'b',
     * and asw and bsw - swapped elements of 'a' and 'b' respectively.
     */

    src11 = (conjB) ? a : b;
    src21 = (conjB) ? b : a;

    kstrcpy(&swSrc1, src11->buf);
    swapComplexComponents(&swSrc1, 1);
    takeComplexApart(&reSrc2, &imSrc2, src21);

    if (conjA && conjB) {
        src12 = src11;
        src11 = &swSrc1;
        src21 = &imSrc2;
        src22 = &reSrc2;
        sign1 = sign3 = "-";
    }
    else {
        src12 = &swSrc1;
        src21 = &reSrc2;
        src22 = &imSrc2;
        if (conjA || conjB) {
            sign3 = "-";
        }
        else {
            sign2 = "-";
        }
    }

    if (core == TILEMUL_MAD) {
        const char *strC = (c == NULL) ? "0" : c->buf;

        ksprintf(expr, "%s = mad(%s, %s%s, %s);\n"
                       "%s = mad(%s, (%s)(%s%s, %s%s), %s);\n",
                 dst->buf, src11->buf, sign1, src21->buf, strC,
                 dst->buf, src12->buf, baseType, sign2, src22->buf,
                 sign3, src22->buf, dst->buf);
    }
    else {
        const char *op = (dst == c) ? "+=" : "=";

        ksprintf(expr, "%s %s %s * %s%s + %s * (%s)(%s%s, %s%s)",
                 dst->buf, op, src11->buf, sign1,
                 src21->buf, src12->buf, baseType, sign2, src22->buf,
                 sign3, src22->buf);
        if (!((c == NULL) || (c == dst))) {
            kstrcatf(expr, " + %s", c->buf);
        }
        kstrcatf(expr, "%s", ";\n");
    }
}

void
sprintfComplexMulUpdate_syr2k_beta0(
    Kstring *expr,
    const Kstring *dst,
    const Kstring *a,
    const Kstring *b,
    const Kstring *c,
    bool isDouble,
    bool conjA,
    bool conjB,
    TileMulCore core)
{
    Kstring swSrc1;      // swapped element of the first source
    // real and imaginary part of the second source
    Kstring reSrc2, imSrc2;
    const Kstring *src11, *src12, *src21, *src22;
    const char *sign1 = "", *sign2 = "", *sign3 = "";
    const char *baseType;

    baseType = (isDouble) ? "double2" : "float2";

    /*
     * Prepare components for multiplying. We should get the following
     * vectorized operations:
     *
     * c = b * a1 + bsw * (-a2, a2)       if both 'a' and 'b' are not conjugated
     * c = b * a1 + bsw * (a2, -a2)       if 'b' is conjugated and 'a' is not
     * c = a * b1 + asw * (-b2, b2)       if 'a' is conjugated and 'b' is not
     * c = asw * (-b2) + a * (b1, -b1)    if both 'a' and 'b' are conjugated
     *
     * Where (a1, a2) and (b1, b2) are complex components of 'a' and 'b',
     * and asw and bsw - swapped elements of 'a' and 'b' respectively.
     */

    src11 = (conjB) ? a : b;
    src21 = (conjB) ? b : a;

    kstrcpy(&swSrc1, src11->buf);
    swapComplexComponents(&swSrc1, 1);
    takeComplexApart(&reSrc2, &imSrc2, src21);

    if (conjA && conjB) {
        src12 = src11;
        src11 = &swSrc1;
        src21 = &imSrc2;
        src22 = &reSrc2;
        sign1 = sign3 = "-";
    }
    else {
        src12 = &swSrc1;
        src21 = &reSrc2;
        src22 = &imSrc2;
        if (conjA || conjB) {
            sign3 = "-";
        }
        else {
            sign2 = "-";
        }
    }

    if (core == TILEMUL_MAD) {
        const char *strC = (c == NULL) ? "0" : c->buf;

        ksprintf(expr, "%s = mad(%s, %s%s, %s);\n"
                       "%s = mad(%s, (%s)(%s%s, %s%s), %s);\n",
                 "sctmp", src11->buf, sign1, src21->buf, strC,
                 dst->buf, src12->buf, baseType, sign2, src22->buf,
                 sign3, src22->buf, "sctmp");
    }
    else {
        const char *op = (dst == c) ? "+=" : "=";

        ksprintf(expr, "%s %s %s * %s%s + %s * (%s)(%s%s, %s%s)",
                 dst->buf, op, src11->buf, sign1,
                 src21->buf, src12->buf, baseType, sign2, src22->buf,
                 sign3, src22->buf);
        if (!((c == NULL) || (c == dst))) {
            kstrcatf(expr, " + %s", c->buf);
        }
        kstrcatf(expr, "%s", ";\n");
    }
}


int
genMulTiles(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileMulOpts *mulOpts)
{
    char s[32];
    const CLBLASKernExtra *kextra = gset->kextra;
    const char *tNameIn;
    unsigned int i;
    unsigned int iend;
    bool tra = ((mulOpts->flags & TILEMUL_TRA) != 0);
    bool trb = ((mulOpts->flags & TILEMUL_TRB) != 0);
    TileMulCore core;
    int ret;

    ret = checkInput(gset, mulOpts);
    if (ret) {
        return ret;
    }

    getVectorTypeName(kextra->dtype, kextra->vecLen, &tNameIn, NULL);
    core = checkReplaceCore(gset, mulOpts->core, tra, trb);

    if (((core == TILEMUL_MULADD || isComplexType(kextra->dtype)) &&
          !tra && trb)) {
        sprintf(s,"%s sum;\n", tNameIn);
        kgenAddStmt(ctx, s);
    }

    iend = (unsigned int)((mulOpts->flags & TILEMUL_TRA) ?
                            gset->subdims[1].bwidth : gset->subdims[1].y);
    for (i = 0; i < iend; i++) {
        genMulLineOnTile(ctx, gset, mulOpts, i, true);
    }

    // just to get state
    ret = kgenAddStmt(ctx, NULL);

    return (ret) ? -EOVERFLOW : 0;
}

int
tileMulGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileMulOpts *mulOpts)
{
    char s[MAX_LENGTH];
    unsigned int vlenA, vlenB;
    unsigned int i, iend; //counters
    // size_t m, n, subK;
    int ret = 0;
    TileMulFlags mflags = mulOpts->flags;
    bool tra = ((mflags & TILEMUL_TRA) != 0);
    bool trb = ((mflags & TILEMUL_TRB) != 0);
    bool localA = (mulOpts->memA == CLMEM_LOCAL_MEMORY);
    bool localB = (mulOpts->memB == CLMEM_LOCAL_MEMORY);
    bool internalFetchB = ((mflags & TILEMUL_NOT_FETCH_B) == 0);
    bool bwStride = ((mflags & TILEMUL_BW_STRIDE) != 0);
    bool incK = ((mflags & TILEMUL_NOT_INC_K) == 0);
    const SubproblemDim *subdims = gset->subdims;
    size_t bwidth = bwStride ? subdims[0].bwidth : subdims[1].bwidth;
    TileMulCore core = mulOpts->core;
    DataType dtype = gset->kextra->dtype;
    const KernelVarNames *varNames = &gset->varNames;
    FetchOpts fetchOpts;
    struct FetchContext *fctx = mulOpts->fctx;
    FetchAddrMode addrMode;
    FetchOptLevel foptlev;
    struct StatementBatch *batch = NULL;
    const Tile *tile;

    memset(&fetchOpts, 0, sizeof(fetchOpts));
    fetchOpts.memA = mulOpts->memA;
    fetchOpts.memB = mulOpts->memB;

    kgenAddStmt(ctx, "/* -- Tiles multiplier -- */\n");

    getVecLens(gset, &vlenA, &vlenB, NULL);

    /* check generator input values */
    ret = checkInput(gset, mulOpts);
    if (ret) {
        return ret;
    }

    if (!bwStride && (subdims[0].bwidth != subdims[1].bwidth)) {
        sprintf(s, "for (int k1 = 0; k1 < %lu; k1 += %lu)",
                subdims[0].bwidth, subdims[1].bwidth);
        kgenBeginBranch(ctx, s);
    }

    core = checkReplaceCore(gset, core, tra, trb);
    if (((core == TILEMUL_MULADD || isComplexType(dtype)) &&
            !tra && trb)) {

        unsigned int n;
        const char *tname;

        n = commonTileSegmentLen(&gset->tileA, &gset->tileBX);
        getVectorTypeName(gset->tileA.dtype, n, &tname, NULL);

        sprintf(s,"%s sum;\n", tname);
        kgenAddStmt(ctx, s);
    }

    // FIXME: remove this kludge for backward compatibility
    if (fctx == NULL) {
        fctx = createFetchContext();
        if (fctx == NULL) {
            return -ENOMEM;
        }
        fetchOpts.mulOpts = mulOpts;
    }
    //////////////////////////////////////////////////////

    foptlev = getFetchOptLevels(fctx);

    if ((gset->flags & BGF_WHOLE_A) && internalFetchB &&
        (foptlev & FOPTLEV_MERGE_FETCHES)) {

        batch = createStmtBatch();
        if (batch == NULL) {
            ret = -ENOMEM;
            goto out;
        }
    }

    /*
     * First, disable sharing internal variables of the fetch code for
     * the first call so as the fetch generator could declares it for the
     * first matrix. And then re-enable it when invoking the fetch for
     * the other matrix if it has been actually enabled.
     */

    disableFetchOptLevels(fctx, FOPTLEV_CAN_SHARE_TMP_AB);

    /*
     * fetch elements of the matrix B, by rows or by columns depending on
     * the transposing flag
     */
    if (internalFetchB) {
        tile = &gset->tileBX;
        fetchOpts.mrole = MATRIX_B;
        fetchOpts.linesNum = trb ? tile->nrCols : tile->nrRows;
        if (batch == NULL) {
            ret = genFetchInputTile(ctx, fctx, gset, &fetchOpts);
            if (!ret) {
                ret = checkTriggerPostFetch(ctx, mulOpts, MATRIX_B);
            }
        }
        else {
            genFetchInputTileBatch(batch, fctx, gset, &fetchOpts);
        }
    }

    fetchOpts.mrole = MATRIX_A;

    if (foptlev & FOPTLEV_CAN_SHARE_TMP_AB) {
        enableFetchOptLevels(fctx, FOPTLEV_CAN_SHARE_TMP_AB);
    }

    if (ret) {
        goto out;
    }

    if (gset->flags & BGF_WHOLE_A) {
        tile = &gset->tileA;
        iend = (tra) ? tile->nrCols : tile->nrRows;
        fetchOpts.linesNum = iend;
        if (batch == NULL) {
            ret = genFetchInputTile(ctx, fctx, gset, &fetchOpts);
        }
        else {
            genFetchInputTileBatch(batch, fctx, gset, &fetchOpts);
            ret = flushStmtBatch(ctx, batch);
            if (!ret) {
                ret = checkTriggerPostFetch(ctx, mulOpts, MATRIX_B);
            }
        }

        if (!ret) {
            ret = checkTriggerPostFetch(ctx, mulOpts, MATRIX_A);
        }
        if (ret) {
            goto out;

        }

        // main multiplying loop
        for (i = 0; i < iend; i++) {
            if (i) {
                kgenAddBlankLine(ctx);
            }
            genMulLineOnTile(ctx, gset, mulOpts, i, true);
        }
    }
    else {
        iend = (unsigned int)((tra) ? subdims[1].bwidth : subdims[1].y);
        fetchOpts.linesNum = 1;

        // main multiplying loop
        for (i = 0; i < iend; i++) {
            if (i) {
                kgenAddBlankLine(ctx);
                revalidateFetchContext(fctx, MATRIX_A);
            }
            // fetch elements of matrix A from single row
            fetchOpts.lineOffset = i;
            genFetchInputTile(ctx, fctx, gset, &fetchOpts);
            ret = checkTriggerPostFetch(ctx, mulOpts, MATRIX_A);
            if (ret) {
                goto out;
            }
            genMulLineOnTile(ctx, gset, mulOpts, i, false);
        }
    }

    /*
     * increment K-related coordinates or pointers depending on addressing
     * mode
     */
    addrMode = getFetchAddrMode(fctx);
    if (addrMode & FETCH_ADDR_K_RELATIVE) {
        kgenAddBlankLine(ctx);
        genPointerUpdate(ctx, varNames->A, varNames->lda, bwidth,
                         subdims[0].y, vlenA, dtype, gset->flags,
                         !tra, localA);

        genPointerUpdate(ctx, varNames->B, varNames->ldb, bwidth,
                         subdims[0].x, vlenB, dtype, gset->flags,
                         trb, localB);
    }
    else {
        if (incK && (varNames->k != NULL) && !(localA && localB)) {
            sprintf(s, "\n%s += %lu;\n", varNames->k, bwidth);
            kgenAddStmt(ctx, s);
        }
    }

    if (!bwStride && (subdims[0].bwidth != subdims[1].bwidth)) {
        kgenEndBranch(ctx, NULL); // k1 loop
    }
    ret = kgenAddStmt(ctx, "/* ---------------------- */\n");
    ret = (ret) ? -EOVERFLOW : 0;

out:
    if (batch != NULL) {
        destroyStmtBatch(batch);
    }
    if (fctx != mulOpts->fctx) {
        destroyFetchContext(fctx);
    }

    return ret;
}
