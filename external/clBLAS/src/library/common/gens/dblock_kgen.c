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


#include <string.h>
#include <stdio.h>

#include <dis_warning.h>
#include <dblock_kgen.h>

/*
 * TODO:
 * 1) barriers in the case when work group size is greater than the
 *    wavefront size
 * 2) 2D dimensional work group size
 * 3) Try version with array indexing
 * 4) Option to avoid unaligned access to vector data (?)
 */

// work performed by work items
typedef struct ItemWork {
    // number of rows to be processed by single work item
    size_t nrRows;
    // number of columns to be processed by single work item
    size_t nrCols;
    // number of items processing the same row
    unsigned int itemsPerRow;
    // total number of items performing the work
    unsigned int nrItems;
    // reduced number of rows at the block tail
    size_t blockTail;
    // work size to be done with the row tail non packed in float4
    size_t tail;
} ItemWork;

/*
 * Private data for loop unrolling
 *
 * NOTE: lmemLD is not used if both
 *       'locLDName' is initialized
 */
typedef struct GenPriv {
    DBlockCopyDirection dir;
    bool transp;
    bool packed;
    bool conjugate;
    bool notVectorize;
    // local memory block leading dimension
    size_t lmemLD;
    // local memory leading dimension variable name
    const char *locLDName;
    // global memory leading dimension variable name
    const char *globLDName;
    DataType dtype;
    unsigned int nfloats;
    unsigned int typeSize;
    const SubproblemDim *dim;
    const ItemWork *work;
    const char *srcName;
    const char *dstName;
    // variables names used while copying to images
    const char *imgXName;
    const char *imgYName;
    size_t cnt;
    // The block size used for copying.
    // The default is 4.
    unsigned int vecLen;

} GenPriv;


/*
 *  'ld' in the list of arguments is matrix leading dimension
 *
 *  Common name forming rule:
 *  (type prefix)(generic part)['Transp']['Conj']['Nvec'](src mem][dst mem][block height][block width]
 */
const char *copyMemDBlockDecl =
    "void\n"
    "%ccopyDBlock%s%s%s%c%c%lu%lu(\n"
    "    %cPtr dst,\n"
    "    %cPtr src,\n"
    "    uint startRow,\n"
    "    uint startCol,\n"
    "    uint ld)\n";

const char *copyMemGImgDBlockDecl =
    "void\n"
    "%ccopyDBlock%sGI%lux%lu(\n"
    "    __write_only image2d_t dst,\n"
    "    int startX,\n"
    "    int startY,\n"
    "    GPtr src,\n"
    "    uint startRow,\n"
    "    uint startCol,\n"
    "    uint ld)\n";

const char *copyMemLImgDBlockDecl =
    "void\n"
    "%ccopyDBlock%sLI%lux%lu(\n"
    "    __write_only image2d_t dst,\n"
    "    int startX,\n"
    "    int startY,\n"
    "    LPtr src)\n";

/*
 * declaration for function performing slow data block copying
 */
const char *copyMemDBlockSlowDecl =
    "void\n"
    "%ccopyDBlock%s%s%s%c%c(\n"
    "    %cPtr dst,\n"
    "    %cPtr src,\n"
    "    uint startRow,\n"
    "    uint startCol,\n"
    "    uint nrRows,\n"
    "    uint nrCols,\n"
    "    uint dstLD,\n"
    "    uint srcLD)\n";

/*
 * declaration for function performing slow data to image block copying
 */
const char *copyMemGImgDBlockSlowDecl =
    "void\n"
    "%ccopyDBlock%sGI(\n"
    "    __write_only image2d_t dst,\n"
    "    int startX,\n"
    "    int startY,\n"
    "    GPtr src,\n"
    "    uint startRow,\n"
    "    uint startCol,\n"
    "    uint nrRows,\n"
    "    uint nrCols,\n"
    "    uint srcLD)\n";

const char *copyMemLImgDBlockSlowDecl =
    "void\n"
    "%ccopyDBlock%sLI(\n"
    "    __write_only image2d_t dst,\n"
    "    int startX,\n"
    "    int startY,\n"
    "    LPtr src,\n"
    "    uint nrRows,\n"
    "    uint nrCols,\n"
    "    uint srcLD)\n";

/*
 * local variables for slow copying between the global and
 * the local memory
 */

const char *copyMemSlowLvars =
    "uint i, j, n;\n"
    /*
     * end counters for copying with vector blocks and just vectors
     * depending in copying type and direction
     */
    "uint jb, jv;\n"
    // end counter for coying with single data with size lesser than float4
    "%s"
    // temporaty float4 variable for the transposing version
    "%s"
    "%cPtr dst1;\n"
    "%cPtr src1;\n\n";

/*
 * One version use passing over the rows, and the second one use
 * passing over the columns. The Second variant is used for transposed
 * copying from the local to the global memory.
 */
const char *copyMemDBlockSlowStart[2] = {
    "if (nrRows %% lsize) {\n"
    "    n = nrRows / lsize + 1;\n"
    "}\n"
    "else {\n"
    "    n = nrRows / lsize;\n"
    "}\n"
    "\n"
    "jb = nrCols / %u;\n"
    "jv = (nrCols - jb * %u) / %u;\n"
    // set counter end for copying with data which size is lesser than float4
    "%s"
    // set pointers to initial position
    "%s"
    "%s"
    "n = (n * lid >= nrRows) ? 0 : n;\n"
    "n = (n * lid + n > nrRows) ? (n - 1) : n;\n"
    "\n",

    "if (nrCols %% lsize) {\n"
    "    n = nrCols / lsize + 1;\n"
    "}\n"
    "else {\n"
    "    n = nrCols / lsize;\n"
    "}\n"
    "\n"
    // set counters for vector copying
    "jb = nrRows / %u;\n"
    "jv = (nrRows - jb * %u) / %u;\n"
    // set counter end for copying with data which size is lesser than float4
    "%s"
    // set pointers to initial position
    "%s"
    "%s"
    "n = (n * lid >= nrCols) ? 0 : n;\n"
    "n = (n * lid + n > nrCols) ? (n - 1) : n;\n"
    "\n"
};

/*
 * declaration for function zeroing float4 aligned
 * block of data
 */
const char *f4zeroDecl =
    "void\n"
    "%cf4zero%lu(%s float4 *data)\n";

const char *fzeroSlowDecl = "void\n"
                            "%cf4zero(%s float4 *buf, size_t cnt)\n";

const char *copyMemImgDBlockSlow =
    "for (i = 0; i < n; i++) {\n"
    "    int x1 = x;\n"
    "    int y1 = y;\n"
    "    %cPtr src1 = src;\n"
    "\n"
    "    for (j = 0; j < jb; j++) {\n"
    "        write_imageui(dst, (int2)(x1++, y1), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x1++, y1), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x1++, y1), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x1++, y1), as_uint4(*src1.f4v++));\n"
    "    }\n"
    "    for (j = 0; j < jv; j++) {\n"
    "        write_imageui(dst, (int2)(x1++, y1), as_uint4(*src1.f4v++));\n"
    "    }\n"
    "\n"
    "    y++;\n"
    "    src.%s += srcLD;\n"
    "}\n";


const char *copyMemImgDBlockPackedSlow =
    "for (i = 0; i < n; i++) {\n"
    "    %cPtr src1 = src;\n"
    "    x = startX + ((index + i) %% nLines) * nrCols / %lu;\n"
    "    y = startY + (index + i) / nLines;\n"
    "\n"
    "    for (j = 0; j < jb; j++) {\n"
    "        write_imageui(dst, (int2)(x++, y), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x++, y), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x++, y), as_uint4(*src1.f4v++));\n"
    "        write_imageui(dst, (int2)(x++, y), as_uint4(*src1.f4v++));\n"
    "    }\n"
    "    for (j = 0; j < jv; j++) {\n"
    "        write_imageui(dst, (int2)(x++, y), as_uint4(*src1.f4v++));\n"
    "    }\n"
    "\n"
    "    src.%s += srcLD;\n"
    "}\n";

const char *setLoopBoundStmt =
        "if (lid > %u) {\n"
        "   nrows = 0;\n"
        "}\n"
        "else {\n"
        "   nrows = (lid == %u) ? %u : %u;\n"
        "}\n";

const char *privatePtrs =
        "%cPtr src1;\n"
        "%cPtr dst1;\n";

// loop bound variable name
const char *lboundVarName = "nrows";
// local id variable
const char *lidVarName = "lid";


/*
 * Partial initialization of the generator private information
 */
static void
initGenPriv(
    GenPriv *priv,
    DataType dtype,
    unsigned int typeSize,
    const SubproblemDim *dim,
    DBlockCopyDirection dir,
    const ItemWork *work,
    const PGranularity *pgran)
{
    unsigned int gsize;

    priv->dtype = dtype;
    priv->typeSize = typeSize;
    priv->nfloats = typeSize / sizeof(float);
    priv->dim = dim;
    priv->dir = dir;
    priv->work = work;
    priv->cnt = 0;
    priv->vecLen = FLOAT4_VECLEN;
    if (dir == DBLOCK_GLOBAL_TO_LOCAL || dir == DBLOCK_LOCAL_TO_GLOBAL) {
        gsize = pgran->wgSize[0] * pgran->wgSize[1];
        priv->vecLen = (unsigned int)(dim->x * dim->y * priv->nfloats / gsize);

        if (priv->vecLen < 1) {
            priv->vecLen = 1;
        } else if (priv->vecLen > 4) {
            priv->vecLen = FLOAT4_VECLEN;
        }
    }

}

/*
 * get info about work to be done by the work group
 *
 * Resulting work data chunk for each item is float4 aligned.
 * Remaining data chunk presented as tail for which code is
 * generated just after the loop part getting deal with float4
 * aligned chunks.
 */
static void
getItemWork(ItemWork *work, const SubproblemDim *dim,
            const PGranularity *pgran, size_t nfloats,
            unsigned int vecLen)
{
    size_t n;
    size_t gsize;

    memset(work, 0, sizeof(ItemWork));
    gsize = pgran->wgSize[0] * pgran->wgSize[1];

    if (dim->y < gsize) {
        // one work item processes a part of a row (or none at all)
        work->itemsPerRow = (unsigned int)(gsize / dim->y);
        work->nrCols = dim->x / work->itemsPerRow;
        work->nrRows = 1;
        if (work->itemsPerRow * dim->y < gsize) {
            work->nrItems = (unsigned int)(work->itemsPerRow * dim->y);
        }
    }
    else {
        // one work item processes typically several rows (or none at all)
        work->itemsPerRow = 1;
        work->nrCols = dim->x;
        work->nrRows = dim->y / gsize;
        if (dim->y % gsize) {
            work->nrRows++;
            work->nrItems = (unsigned int)(dim->y / work->nrRows);
            // remaining number of rows
            n = dim->y - work->nrItems * work->nrRows;
            if (n) {
                work->blockTail = n;
                // total number of work items needed for the transfer
                work->nrItems++;
            }
        }
    }
    work->nrCols -= (work->nrCols * nfloats % vecLen) / nfloats;
    work->tail = dim->x - work->nrCols * work->itemsPerRow;
}

/*
 * Prepare generator outer loop
 */
static void
prepareLoop(struct KgenContext *ctx, ItemWork *work, LoopCtl *loopCtl)
{
    char tmp[1024];

    kgenAddStmt(ctx, "size_t n;\n");
    loopCtl->ocName = "n";

    if (work->nrItems) {
        sprintf(tmp, "size_t %s;\n\n", lboundVarName);
        kgenAddStmt(ctx, tmp);

        /*
         * set number of rows to be processed by the work item;
         * in the case it is not a constant
         */
        if (work->blockTail) {
            sprintf(tmp, setLoopBoundStmt, work->nrItems - 1, work->nrItems - 1,
                    work->blockTail, work->nrRows);
            kgenAddStmt(ctx, tmp);
        }
        else {
            sprintf(tmp, "nrows = (%s >= %u) ? 0 : %lu;\n", lidVarName,
                    work->nrItems, work->nrRows);
            kgenAddStmt(ctx, tmp);
        }

        loopCtl->outBound.name = lboundVarName;
    }
    else {
        loopCtl->outBound.val = (unsigned long)work->nrRows;
        loopCtl->obConst = true;
    }
}

static int
getVecLen(struct KgenContext *ctx, void *priv)
{
    GenPriv *gpriv = (GenPriv*)priv;
    (void) ctx;
    return gpriv->vecLen;
}

/*
 * common function for loop tail generating
 */
static void
addTailCode(
    struct KgenContext *ctx,
    GenPriv *gpriv,
    LoopUnrollGen genSingleVec,
    LoopUnrollGen genSingle)
{
    char tmp[1024];
    const ItemWork *work = gpriv->work;
    LoopCtl loopCtl;
    LoopUnrollers unrollers;

    memset(&loopCtl, 0, sizeof(loopCtl));
    memset(&unrollers, 0, sizeof(unrollers));

    loopCtl.inBound = (unsigned long)work->tail;

    if (work->itemsPerRow > 1) {
        if (work->nrItems) {
            sprintf(tmp, "if ((%s %% %u == %u) && (%s < %u))",
                    lidVarName, work->itemsPerRow, work->itemsPerRow - 1,
                    lidVarName, work->nrItems);
        }
        else {
            sprintf(tmp, "if (%s %% %u == %u)",
                    lidVarName, work->itemsPerRow, work->itemsPerRow - 1);
        }
        kgenBeginBranch(ctx, tmp);
    }

    unrollers.genSingleVec = genSingleVec;
    unrollers.genSingle = genSingle;
    unrollers.getVecLen = getVecLen;

    kgenLoopUnroll(ctx, &loopCtl, gpriv->dtype, &unrollers, gpriv);

    if (work->itemsPerRow > 1) {
        kgenEndBranch(ctx, NULL);
    }
}

static int
copyMemPreUnroll(struct KgenContext *ctx, void *priv)
{
    DUMMY_ARG_USAGE(priv);

    kgenAddStmt(ctx, "src1 = src;\n");

    return kgenAddStmt(ctx, "dst1 = dst;\n\n");
}

static int
copyImgPreUnroll(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    GenPriv *gpriv = (GenPriv*)priv;
    if (gpriv->packed) {
        sprintf(tmp, "%s = startX + (index * %lu) %% pLine / %u;\n"
                "%s = startY + (index * %lu) / pLine;\n" "%s = src;\n\n",
                gpriv->imgXName, gpriv->dim->x, FLOAT4_VECLEN / gpriv->nfloats,
                gpriv->imgYName, gpriv->dim->x, gpriv->srcName);
    }
    else {
        sprintf(tmp, "%s = x;\n" "%s = y;\n" "%s = src;\n\n", gpriv->imgXName,
                gpriv->imgYName, gpriv->srcName);
    }
    return kgenAddStmt(ctx, tmp);
}

static int
copyImgVec(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    GenPriv *gpriv = (GenPriv*)priv;

    dtypeUPtrField(gpriv->dtype);
    sprintf(tmp, "write_imageui(%s, (int2)(%s++,%s), as_uint4(*%s.f4v++));\n",
            gpriv->dstName, gpriv->imgXName, gpriv->imgYName, gpriv->srcName);

    return kgenAddStmt(ctx, tmp);
}

static int
copyImgSingle(struct KgenContext *ctx, void *priv)
{
    GenPriv *gpriv = (GenPriv*)priv;
    if (gpriv->dtype == TYPE_COMPLEX_DOUBLE) {
        return copyImgVec(ctx, priv);
    }
    else {
        return -EINVAL;
    }
}

static int
copyMemVec(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    char vec[64];
    GenPriv *gpriv = (GenPriv*)priv;

    if (gpriv->vecLen == 1)
        sprintf(vec,"f");
    else
        sprintf(vec,"f%dv", gpriv->vecLen);

    if (gpriv->conjugate) {
        sprintf(tmp, "tmp = *%s.%s++;\n", gpriv->srcName, vec);
        kgenAddStmt(ctx, tmp);
        if (gpriv->dtype == TYPE_COMPLEX_FLOAT) {
            kgenAddStmt(ctx, "tmp.y = -tmp.y;\n"
                             "tmp.w = -tmp.w;\n");
        }
        else {
            kgenAddStmt(ctx, "tmp.y = -tmp.y;\n");
        }
        sprintf(tmp, "*%s.%s++ = tmp;\n",
                gpriv->dstName, vec);
    }
    else {
        sprintf(tmp, "*%s.%s++ = *%s.%s++;\n", gpriv->dstName, vec,
                gpriv->srcName, vec);
    }

    return kgenAddStmt(ctx, tmp);
}

static int
copyMemSingle(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    GenPriv *gpriv = (GenPriv*)priv;
    const char *vfield;

    vfield = dtypeUPtrField(gpriv->dtype);

    if (gpriv->conjugate) {
        sprintf(tmp, "*%s.%s = *%s.%s++;\n",
                gpriv->dstName, vfield, gpriv->srcName, vfield);
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "(*%s.%s).y = -(*%s.%s).y;\n",
                gpriv->dstName, vfield, gpriv->dstName, vfield);
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "%s.%s++;\n", gpriv->dstName, vfield);
    }
    else {
        sprintf(tmp, "*%s.%s++ = *%s.%s++;\n",
                gpriv->dstName, vfield, gpriv->srcName, vfield);
    }

    return kgenAddStmt(ctx, tmp);
}

static int
copyMemVecTransp(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    size_t i;
    GenPriv *gpriv = (GenPriv*)priv;
    unsigned int n = gpriv->nfloats;
    const char *tmpSuff[2][4] = {
            {"x", "y", "z", "w"},
            {"xy", "zw", NULL, NULL}};
    const char *dstSuff[4] = {"f", "f2v", NULL, "f4v"};
    const char *vfield;
    const char *s;

    vfield = dtypeUPtrField(gpriv->dtype);
    kgenAddBlankLine(ctx);

    if (gpriv->dir == DBLOCK_GLOBAL_TO_LOCAL) {
        sprintf(tmp, "tmp = *%s.f4v++;\n", gpriv->srcName);
        kgenAddStmt(ctx, tmp);

        if (gpriv->conjugate) {
            /*
             * Only complex float element can be conjugated here,
             * those of double complex type are processed with no vectrized
             * function
             */
            kgenAddStmt(ctx, "tmp.y = -tmp.y;\n"
                             "tmp.w = -tmp.w;\n");
        }

        for (i = 0; i < FLOAT4_VECLEN / n; i++) {
            if (gpriv->locLDName) {
                sprintf(tmp, "%s.%s[%s * %lu] = tmp.%s;\n",
                        gpriv->dstName, dstSuff[n - 1],
                        gpriv->locLDName, i, tmpSuff[n - 1][i]);
            }
            else {
                sprintf(tmp, "%s.%s[%lu] = tmp.%s;\n", gpriv->dstName,
                        dstSuff[n - 1], gpriv->lmemLD * i, tmpSuff[n - 1][i]);
            }
            kgenAddStmt(ctx, tmp);
        }
        s = gpriv->dstName;
    }
    else {
        for (i = 0; i < FLOAT4_VECLEN / n; i++) {
            if (gpriv->locLDName) {
                sprintf(tmp, "tmp.%s = %s.%s[%s * %lu];\n", tmpSuff[n - 1][i],
                        gpriv->srcName, dstSuff[n - 1], gpriv->locLDName, i);
            }
            else {
                sprintf(tmp, "tmp.%s = %s.%s[%lu];\n", tmpSuff[n - 1][i],
                        gpriv->srcName, dstSuff[n - 1], gpriv->lmemLD * i);
            }
            kgenAddStmt(ctx, tmp);
        }

        sprintf(tmp, "*%s.f4v++ = tmp;\n", gpriv->dstName);
        kgenAddStmt(ctx, tmp);

        s = gpriv->srcName;
    }

    if (gpriv->locLDName) {
        sprintf(tmp, "%s.%s += %s * %lu;\n", s, vfield, gpriv->locLDName, i);
    }
    else {
        sprintf(tmp, "%s.%s += %lu;\n", s, vfield, gpriv->lmemLD * i);
    }

    return kgenAddStmt(ctx, tmp);
}

static int
copyMemSingleTransp(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    GenPriv *gpriv = (GenPriv*)priv;
    const char *vfield;

    vfield = dtypeUPtrField(gpriv->dtype);
    kgenAddBlankLine(ctx);

    if (gpriv->dir == DBLOCK_GLOBAL_TO_LOCAL) {
        if (gpriv->locLDName) {
            sprintf(tmp, "*%s.%s = *%s.%s++;\n",
                    gpriv->dstName, vfield,
                    gpriv->srcName, vfield);
            kgenAddStmt(ctx, tmp);

            if (gpriv->conjugate) {
                sprintf(tmp, "(*%s.%s).y = -(*%s.%s).y;\n",
                        gpriv->dstName, vfield, gpriv->dstName,
                        vfield);
                kgenAddStmt(ctx, tmp);
            }
            sprintf(tmp, "%s.%s += %s;\n",
                    gpriv->dstName, vfield, gpriv->locLDName);
        }
        else {
            sprintf(tmp, "%s.%s[%lu] = *%s.%s++;\n",
                    gpriv->dstName, vfield,
                    gpriv->lmemLD * gpriv->cnt, gpriv->srcName,
                    vfield);
            if (gpriv->conjugate) {
                kgenAddStmt(ctx, tmp);
                sprintf(tmp, "%s.%s[%lu].y = -%s.%s[%lu].y;\n",
                        gpriv->dstName, vfield, gpriv->lmemLD * gpriv->cnt,
                        gpriv->dstName, vfield, gpriv->lmemLD * gpriv->cnt);
            }
        }
    }
    else {
        if (gpriv->locLDName) {
            sprintf(tmp, "*%s.%s++ = *%s.%s;\n"
                         "%s.%s += %s;\n",
                    gpriv->dstName, vfield,
                    gpriv->srcName, vfield,
                    gpriv->srcName, vfield, gpriv->locLDName);
        }
        else {
            sprintf(tmp, "*%s.%s++ = %s.%s[%lu];\n",
                    gpriv->dstName, vfield, gpriv->srcName, vfield,
                    gpriv->lmemLD * gpriv->cnt);
        }
    }
    gpriv->cnt++;

    return kgenAddStmt(ctx, tmp);
}

/*
 *  transfer row tail elements being not packing in float4 vector
 *  and zeroing row tail
 */
static void
addCopyTailCode(struct KgenContext *ctx, GenPriv *gpriv)
{
    LoopUnrollGen singleVec;
    LoopUnrollGen single;
    bool image;

    image = (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE ||
            gpriv->dir == DBLOCK_LOCAL_TO_IMAGE);

    if (image) {
        singleVec = copyImgVec;
        single = copyImgSingle;
    }
    else {
        if (gpriv->transp) {
            singleVec = copyMemVecTransp;
            single = copyMemSingleTransp;
        }
        else {
            singleVec = copyMemVec;
            single = copyMemSingle;
        }
    }

    if (gpriv->notVectorize) {
        singleVec = NULL;
    }
    addTailCode(ctx, gpriv, singleVec, single);
}

static int
copyMemPostUnroll(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    const char *s[2] = {"src", "dst"};
    GenPriv *gpriv = (GenPriv*)priv;
    int gdir;
    const char *vfield;

    gdir = (gpriv->dir == DBLOCK_GLOBAL_TO_LOCAL) ? 0 : 1;

    if (gpriv->work && gpriv->work->tail) {
        addCopyTailCode(ctx, gpriv);
    }

    if (!gpriv->transp) {
        kgenAddBlankLine(ctx);
    }

    // modify pointers
    vfield = dtypeUPtrField(gpriv->dtype);
    sprintf(tmp, "%s.%s += %s;\n", s[gdir], vfield, gpriv->globLDName);
    kgenAddStmt(ctx, tmp);

    if (gpriv->transp) {
        sprintf(tmp, "%s.%s++;\n", s[1 - gdir], vfield);
    }
    else {
        if (gpriv->locLDName) {
            sprintf(tmp, "%s.%s += %s;\n", s[1 - gdir],
                    vfield, gpriv->locLDName);
        }
        else {
            sprintf(tmp, "%s.%s += %lu;\n", s[1 - gdir],
                    vfield, gpriv->lmemLD);
        }
    }

    return kgenAddStmt(ctx, tmp);
}

static int
copyImgPostUnroll(struct KgenContext *ctx, void *priv)
{
    char tmp[1024];
    GenPriv *gpriv = (GenPriv*)priv;
    const char *vfield = dtypeUPtrField(gpriv->dtype);

    if (gpriv->work && gpriv->work->tail) {
        addCopyTailCode(ctx, gpriv);
    }

    kgenAddBlankLine(ctx);

    if (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) {
        sprintf(tmp, "src.%s += %s;\n", vfield, gpriv->globLDName);
    }
    else if (gpriv->dir == DBLOCK_LOCAL_TO_IMAGE) {
        sprintf(tmp, "src.%s += %lu;\n", vfield, gpriv->lmemLD);
    }
    kgenAddStmt(ctx, tmp);
    if(gpriv->packed) {
        sprintf(tmp, "index++;\n");
    } else {
        sprintf(tmp, "y++;\n");
    }
    return kgenAddStmt(ctx, tmp);
}

// unrolling generator for the f4zero function
static int
f4zeroSingle(struct KgenContext *ctx, void *priv)
{
    DUMMY_ARG_USAGE(priv);

    return kgenAddStmt(ctx, "*data++ = 0;\n");
}

/*
 * Add statement setting initial local pointer for the work item
 *
 * @ld: lead dimension for the local block in float words;
 *       if it's zero, the "ld" argument of a generated function is
 *       used instead
 */
static void
addSettingPtrCode(
    struct KgenContext *ctx,
    const char *ptrName,
    size_t ld,
    bool transpose,
    const PGranularity *pgran,
    GenPriv *gpriv)
{
    char tmp[4096];
    const char *vfield;
    const SubproblemDim *dim = gpriv->dim;
    const ItemWork *work = gpriv->work;
    size_t gsize;

    vfield = dtypeUPtrField(gpriv->dtype);
    gsize = pgran->wgSize[0] * pgran->wgSize[1];

    if (ld) {
        // offset between two rows and two elements in each row
        size_t roff, eoff;

        if (transpose) {
            roff = 1;
            eoff = ld;
        }
        else {
            roff = ld;
            eoff = 1;
        }

        if (dim->y < gsize) {
            sprintf(tmp, "%s.%s += (%s / %u) * %lu + (%s %% %u * %lu) * %lu;\n",
                    ptrName, vfield, lidVarName, work->itemsPerRow,
                    roff, lidVarName, work->itemsPerRow, work->nrCols, eoff);
        }
        else {
            sprintf(tmp, "%s.%s += %s * %lu * %lu;\n",
                    ptrName, vfield, lidVarName, work->nrRows, roff);
        }
    }
    else {
        if (dim->y < gsize) {
            sprintf(tmp, "%s.%s += (startRow + %s / %u) * %s + "
                                   "startCol + %s %% %u * %lu;\n",
                    ptrName, vfield, lidVarName, work->itemsPerRow,
                    gpriv->globLDName, lidVarName, work->itemsPerRow, work->nrCols);
        }
        else {
            sprintf(tmp, "%s.%s += (startRow + %s * %lu) * %s + startCol;\n",
                    ptrName, vfield, lidVarName, work->nrRows, gpriv->globLDName);
        }
    }

    kgenAddStmt(ctx, tmp);
    kgenAddBlankLine(ctx);
}

/*
 * Add statement setting initial coordinates pointer for image
 *
 */
static void
addSettingImageXYCode(
    struct KgenContext *ctx,
    const char *xName,
    const char *yName,
    const PGranularity *pgran,
    GenPriv *gpriv)
{
    char tmp[4096];
    const ItemWork *work = gpriv->work;
    size_t gsize = pgran->wgSize[0] * pgran->wgSize[1];

    if (gpriv->packed) {
        sprintf(tmp, "pLine = ((get_image_width(dst) - startX) * %d / %lu) * %lu;\n",
                FLOAT4_VECLEN / gpriv->nfloats, gpriv->dim->x, gpriv->lmemLD);
        kgenAddStmt(ctx, tmp);
        if (gpriv->dim->y < gsize) {
            sprintf(tmp, "index = %s / %u;\n", lidVarName,
                    work->itemsPerRow);
        }
        else {
            sprintf(tmp, "index = %s * %lu;\n", lidVarName,
                    work->nrRows);
        }
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "x = startX + (index * %lu) %% pLine / %u;\n", gpriv->dim->x,
                FLOAT4_VECLEN / gpriv->nfloats);
        kgenAddStmt(ctx, tmp);
        if (gpriv->dim->y < gsize) {
            sprintf(tmp, "x += (%s %% %u) * (%lu / %u / %u);\n", lidVarName,
                    work->itemsPerRow, gpriv->dim->x,
                    (FLOAT4_VECLEN / gpriv->nfloats), work->itemsPerRow);
            kgenAddStmt(ctx, tmp);
        }
        sprintf(tmp, "y = startY + (index * %lu) / pLine;\n", gpriv->dim->x);
        kgenAddStmt(ctx, tmp);
    }
    else {
        if (gpriv->dim->y < gsize) {
            sprintf(tmp, "%s = startX + %s %% %u * %lu / %d;\n",
                    xName, lidVarName, work->itemsPerRow, work->nrCols,
                    FLOAT4_VECLEN/gpriv->nfloats);
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, "%s = startY + %s / %u;\n", yName, lidVarName,
                    work->itemsPerRow);
            kgenAddStmt(ctx, tmp);
        }
        else {
            sprintf(tmp, "%s = startX;\n", xName);
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, "%s = startY + %s * %lu;\n", yName, lidVarName,
                    gpriv->work->nrRows);
            kgenAddStmt(ctx, tmp);
        }
    }

    kgenAddBlankLine(ctx);
}

// generator working with subproblems of any dimension
static int
copyDBlockGenericGen(
    struct KgenContext *ctx,
    const PGranularity *pgran,
    GenPriv *gpriv)
{
    char fpref;
    const char varPref[2] = {'G', 'L'};
    char tmp[1024];
    bool image;
    const char *s[3];
    int gdir;
    unsigned int i, n, gsize;
    const char *vfield;
    DataType dtype = gpriv->dtype;

    fpref = dtypeToPrefix(dtype);
    if (!fpref || (fpref == 'i')) {
        return -EINVAL;
    }

    image = (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE ||
             gpriv->dir == DBLOCK_LOCAL_TO_IMAGE);
    s[0] = (gpriv->transp) ? "Transp" : "";
    vfield = dtypeUPtrField(dtype);
    n = FLOAT4_VECLEN / gpriv->nfloats;
    gsize = pgran->wgSize[0] * pgran->wgSize[1];

    if (image) {
        char srcStr[1024];
        s[1] = (gpriv->packed) ? "Pack" : "";
        if (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) {
            sprintf(srcStr, "src.%s += (startRow + lid * n) *"
                    " srcLD + startCol;\n", vfield);
            sprintf(tmp, copyMemGImgDBlockSlowDecl, fpref, s[1]);
        }
        else {
            sprintf(srcStr, "src.%s += srcLD * lid * n;\n", vfield);
            sprintf(tmp, copyMemLImgDBlockSlowDecl, fpref, s[1]);
        }
        kgenDeclareFunction(ctx, tmp);
        kgenBeginFuncBody(ctx);
        sprintf(tmp, "int x, y;\n"
                     "uint i, j, n, jb, jv;\n"
                     "int lsize = %u;\n", gsize);
        kgenAddStmt(ctx, tmp);
        kgenDeclareLocalID(ctx, "lid", pgran);
        if (gpriv->packed) {
            char nLinesStr[1024];
            sprintf(nLinesStr,
                    "nLines = (get_image_width(dst) - startX) * %d / nrCols;\n"
                    "index = lid * n;\n", FLOAT4_VECLEN / gpriv->nfloats);
            sprintf(tmp, "int nLines, index;\n");
            kgenAddStmt(ctx, tmp);
            sprintf(tmp, copyMemDBlockSlowStart[0], 4 * n, 4 * n, n,"",
                    nLinesStr, srcStr);
        }
        else {
            sprintf(tmp, copyMemDBlockSlowStart[0], 4 * n, 4 * n, n, "",
                    "x = startX;\n" "y = startY + lid * n;\n", srcStr);
        }
        kgenAddStmt(ctx, tmp);

        gdir = (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) ? 0 : 1;
        if (gpriv->packed) {
            sprintf(tmp, copyMemImgDBlockPackedSlow, varPref[gdir],
                    FLOAT4_VECLEN / gpriv->nfloats, vfield);
        }
        else {
            sprintf(tmp, copyMemImgDBlockSlow, varPref[gdir], vfield);
        }
        kgenAddStmt(ctx, tmp);
    }
    else {
        LoopCtl loopCtl;
        LoopUnrollers unrollers;
        char buf[3][256];

        memset(&loopCtl, 0, sizeof(loopCtl));
        memset(&unrollers, 0, sizeof(unrollers));

        s[1] = (gpriv->conjugate) ? "Conj" : "";
        s[2] = (gpriv->notVectorize) ? "Nvec" : "";
        gdir = (gpriv->dir == DBLOCK_GLOBAL_TO_LOCAL) ? 0 : 1;
        sprintf(tmp, copyMemDBlockSlowDecl,
                fpref, s[0], s[1], s[2], varPref[gdir], varPref[1 - gdir],
                varPref[1 - gdir], varPref[gdir]);
        kgenDeclareFunction(ctx, tmp);
        kgenBeginFuncBody(ctx);
        kgenDeclareLocalID(ctx, "lid", pgran);
        sprintf(tmp, "int lsize = %u;\n", gsize);
        kgenAddStmt(ctx, tmp);

        if (dtype == TYPE_COMPLEX_DOUBLE) {
            s[0] = "";
            s[1] = "";
        }
        else {
            s[0] = "uint js;\n";
            s[1] = (gpriv->transp || gpriv->conjugate) ? "float4 tmp;\n" : "";
        }

        // pass over rows or columns?
        i = (gpriv->transp && gdir) ? 1 : 0;

        if (dtype == TYPE_COMPLEX_DOUBLE) {
            buf[0][0] = '\0';
        }
        else {
            const char *boundName;

            // set counter bound to copy tail part, each work less than float4
            boundName = (i) ? "nrRows" : "nrCols";

            /*
             * FIXME: the kludge is introduced due to strange
             * runtime segfault at block transferring for another
             * data types. Verify it later. Now, for non float types
             * keep only simple loop.
             */
            if (i && (dtype != TYPE_FLOAT)) {
                gpriv->notVectorize = true;
            }

            if (gpriv->notVectorize) {
                sprintf(buf[0], "jb = 0;\n"
                                "jv = 0;\n"
                                "js = %s;\n",
                        boundName);
            }
            else {
                sprintf(buf[0], "js = %s - jb * %u - jv * %u;\n",
                        boundName, 4 * n, n);
            }
        }

        // set initial pointers
        if (!gdir) {
            sprintf(buf[1], "src.%s += (startRow + lid * n) * srcLD + "
                                       "startCol;\n", vfield);
            if (gpriv->transp) {
                sprintf(buf[2], "dst.%s += lid * n;\n", vfield);
            }
            else {
                sprintf(buf[2], "dst.%s += dstLD * lid * n;\n", vfield);
            }
        }
        else {
            if (gpriv->transp) {
                sprintf(buf[1], "src.%s += lid * n;\n", vfield);
            }
            else {
                sprintf(buf[1], "src.%s += srcLD * lid * n;\n", vfield);
            }
            sprintf(buf[2], "dst.%s += (startRow + lid * n) * dstLD + "
                                       "startCol;\n", vfield);
        }

        sprintf(tmp, copyMemSlowLvars, s[0], s[1],
                varPref[1 - gdir], varPref[gdir]);
        kgenAddStmt(ctx, tmp);

        sprintf(tmp, copyMemDBlockSlowStart[i],
                4 * n, 4 * n, n, buf[0], buf[1], buf[2]);
        kgenAddStmt(ctx, tmp);

        // prepare to loop unrolling
        gpriv->srcName = "src1";
        gpriv->dstName = "dst1";
        if (gdir) {
            gpriv->locLDName = "srcLD";
            gpriv->globLDName = "dstLD";
        }
        else {
            gpriv->locLDName = "dstLD";
            gpriv->globLDName = "srcLD";
        }

        loopCtl.ocName = "j";

        if (gpriv->transp) {
            unrollers.genSingle = copyMemSingleTransp;
            if (dtype != TYPE_COMPLEX_DOUBLE) {
                unrollers.genSingleVec = copyMemVecTransp;
            }
        }
        else {
            unrollers.genSingle = copyMemSingle;
            if (dtype != TYPE_COMPLEX_DOUBLE) {
                unrollers.genSingleVec = copyMemVec;
            }
        }

        // external loop
        kgenBeginBranch(ctx, "for (i = 0; i < n; i++)");
        copyMemPreUnroll(ctx, gpriv);

        // finally, unroll all loops
        unrollers.getVecLen = getVecLen;

        // copying with 4 float4 words
        if (!gpriv->notVectorize) {
            loopCtl.outBound.name = "jb";
            loopCtl.inBound = 4 * n;
            kgenLoopUnroll(ctx, &loopCtl, dtype, &unrollers, gpriv);

            // copying with float4 words
            loopCtl.outBound.name = "jv";
            loopCtl.inBound = n;
            kgenLoopUnroll(ctx, &loopCtl, dtype, &unrollers, gpriv);
        }

        // copying the remaining tail
        if (dtype != TYPE_COMPLEX_DOUBLE) {
            unrollers.genSingleVec = NULL;
            loopCtl.outBound.name = "js";
            loopCtl.inBound = 1;
            kgenLoopUnroll(ctx, &loopCtl, dtype, &unrollers, gpriv);
        }

        copyMemPostUnroll(ctx, gpriv);
        kgenEndBranch(ctx, NULL);
    }

    return kgenEndFuncBody(ctx);
}

// generator optimizing to a subproblem size
static int
copyDBlockOptimGen(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const PGranularity *pgran,
    GenPriv *gpriv)
{
    char fpref;
    const char varPref[2] = {'G', 'L'};
    char tmp[1024];
    // lead dimension for right and transposed local block in float words
    ItemWork work;
    LoopCtl loopCtl;
    LoopUnrollers unrollers;
    const char *s, *s1, *s2;
    bool image;
    SubproblemDim newDim;
    // copying direction within the memory or image related function group
    int gdir = 0;
    int r;

    fpref = dtypeToPrefix(gpriv->dtype);
    if (!fpref || (fpref == 'i')) {
        return -EINVAL;
    }

    image = (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE ||
             gpriv->dir == DBLOCK_LOCAL_TO_IMAGE);

    memset(&unrollers, 0, sizeof(unrollers));
    memset(&loopCtl, 0, sizeof(loopCtl));
    memset(&newDim, 0, sizeof(newDim));

    gpriv->dim = &newDim;
    gpriv->work = (const ItemWork*)&work;
    gpriv->globLDName = "ld";
    s = (gpriv->transp) ? "Transp" : "";
    s1 = (gpriv->conjugate) ? "Conj" : "";
    s2 = (gpriv->notVectorize) ? "Nvec" : "";

    if ((gpriv->dir == DBLOCK_LOCAL_TO_GLOBAL) && gpriv->transp) {
        // pass over columns of the block stored in the local memory
        newDim.x = dim->y;
        newDim.y = dim->x;
    }
    else {
        // pass over rows
        newDim.x = dim->x;
        newDim.y = dim->y;
    }

    getItemWork(&work, &newDim, pgran, gpriv->nfloats, gpriv->vecLen);

    if (image) {
        s = (gpriv->packed) ? "Pack" : "";
        if (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) {
            sprintf(tmp, copyMemGImgDBlockDecl, fpref, s, dim->y, dim->x);
        }
        else {
            sprintf(tmp, copyMemLImgDBlockDecl, fpref, s, dim->y, dim->x);
        }

    }
    else {
        gdir = (gpriv->dir == DBLOCK_GLOBAL_TO_LOCAL) ? 0 : 1;
        sprintf(tmp, copyMemDBlockDecl, fpref, s, s1, s2, varPref[gdir],
                varPref[1 - gdir], dim->y, dim->x, varPref[1 - gdir],
                varPref[gdir]);
    }

    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    kgenDeclareLocalID(ctx, lidVarName, pgran);

    if (image) {
        // data for loop unrolling
        if (work.nrRows > 1) {
            gpriv->srcName = "src1";
            gpriv->dstName = "dst";
            gpriv->imgXName="x1";
            gpriv->imgYName="y1";
            if(gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) {
                kgenAddStmt(ctx, "GPtr src1;\n");
            }
            else if(gpriv->dir == DBLOCK_LOCAL_TO_IMAGE) {
                kgenAddStmt(ctx, "LPtr src1;\n");
            }
            kgenAddStmt(ctx, "int x1, y1;\n");

            unrollers.preUnroll = copyImgPreUnroll;
            unrollers.postUnroll = copyImgPostUnroll;
        }
        else {
            gpriv->srcName = "src";
            // dst has image2d_t type here
            gpriv->dstName = "dst";
            gpriv->imgXName="x";
            gpriv->imgYName="y";
        }
    }
    else {
        if ((gpriv->nfloats != FLOAT4_VECLEN) &&
            (gpriv->transp || gpriv->conjugate)) {

            /*
             * temporary variable to transpose or conjugate non double
             * complex elements
             */
            kgenAddStmt(ctx, "float4 tmp;\n");
        }

        if (work.nrRows > 1) {
            sprintf(tmp, privatePtrs, varPref[gdir], varPref[1 - gdir]);
            kgenAddStmt(ctx, tmp);

            // data for loop unrolling
            unrollers.preUnroll = copyMemPreUnroll;
            unrollers.postUnroll = copyMemPostUnroll;
            gpriv->srcName = "src1";
            gpriv->dstName = "dst1";
        }
        else {
            gpriv->srcName = "src";
            gpriv->dstName = "dst";
        }
    }

    if ((work.nrRows > 1) || work.nrItems) {
        prepareLoop(ctx, &work, &loopCtl);
    }
    kgenAddBlankLine(ctx);
    loopCtl.inBound = (unsigned long)work.nrCols;

    // now, prepare all needed for loop unrolling

    if (image) {
        kgenAddStmt(ctx, "int x, y;\n");
        if (gpriv->packed) {
            kgenAddStmt(ctx, "int pLine, index;\n");
        }
        gpriv->lmemLD = fl4RowWidth(dim->x, gpriv->typeSize) *
                           FLOAT4_VECLEN / gpriv->nfloats;
        // set up starting x and y in image
        addSettingImageXYCode(ctx, "x", "y", pgran, gpriv);

        if (gpriv->dir == DBLOCK_GLOBAL_TO_IMAGE) {
            // set initial global pointer
            addSettingPtrCode(ctx, "src", 0, false, pgran, gpriv);
        }
        else if (gpriv->dir == DBLOCK_LOCAL_TO_IMAGE) {
            // set initial local pointer
            addSettingPtrCode(ctx, "src", gpriv->lmemLD, gpriv->transp,
                              pgran, gpriv);
        }

        unrollers.genSingleVec = copyImgVec;
        unrollers.genSingle = copyImgSingle;
    }
    else {
        // set initial global pointer
        s = (gdir) ? "dst" : "src";
        addSettingPtrCode(ctx, s, 0, false, pgran, gpriv);

        s = (gdir) ? "src" : "dst";

        if (!gdir && gpriv->transp) {
            gpriv->lmemLD = fl4RowWidth(dim->y, gpriv->typeSize) *
                           FLOAT4_VECLEN / gpriv->nfloats;
        }
        else {
            gpriv->lmemLD = fl4RowWidth(dim->x, gpriv->typeSize) *
                           FLOAT4_VECLEN / gpriv->nfloats;
        }

        if (gpriv->transp) {
            unrollers.genSingleVec = (gpriv->notVectorize) ? NULL :
                                                             copyMemVecTransp;
            unrollers.genSingle = copyMemSingleTransp;
        }
        else {
            unrollers.genSingleVec = (gpriv->notVectorize) ? NULL : copyMemVec;
            unrollers.genSingle = copyMemSingle;
        }

        addSettingPtrCode(ctx, s, gpriv->lmemLD, gpriv->transp,
                          pgran, gpriv);
    }
    unrollers.getVecLen = getVecLen;

    // unroll for float4 aligned data chunk
    kgenLoopUnroll(ctx, &loopCtl, gpriv->dtype, &unrollers, gpriv);

    /*
     * Unroll for remaining data tail.
     * Block tail reading/writing is done separately
     * when many work items process single row
     * because the compiler don't like any conditional
     * branches in loops
     */
    if ((unrollers.postUnroll == NULL) && work.tail) {
        addCopyTailCode(ctx, gpriv);
    }

    r = kgenEndFuncBody(ctx);

    return r ? -EOVERFLOW : 0;
}

int
copyDataBlockGen(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const PGranularity *pgran,
    DataType dtype,
    DBlockCopyDirection dir,
    DBlockCopyFlags flags)
{
    int r;
    GenPriv gpriv;
    unsigned int tsize;

    tsize = dtypeSize(dtype);

    if (dir == DBLOCK_LOCAL_TO_IMAGE ||
        dir == DBLOCK_GLOBAL_TO_IMAGE) {
        size_t rowSize;

        if (dim != NULL) {
            rowSize = tsize * dim->x;
            if (rowSize % sizeof(cl_float4) != 0) {
                // only float4 aligned rows are supported
                return -EINVAL;
            }
        }
        if (flags & DBLOCK_COPY_TRANSPOSE) {
            return -EINVAL;
        }
    }

    memset(&gpriv, 0, sizeof(gpriv));
    gpriv.transp = (flags & DBLOCK_COPY_TRANSPOSE);
    gpriv.packed = (flags & DBLOCK_COPY_PACKED_IMAGE);
    if (dtype != TYPE_COMPLEX_DOUBLE) {
        gpriv.notVectorize = (flags & DBLOCK_COPY_NOT_VECTORIZE);
    }
    if ((flags & DBLOCK_COPY_CONJUGATE) && isComplexType(dtype)) {
        gpriv.conjugate = true;
    }
    initGenPriv(&gpriv, dtype, tsize, dim ,dir, NULL, pgran);

    if (dim) {
        r = copyDBlockOptimGen(ctx, dim, pgran, &gpriv);
    }
    else {
        r = copyDBlockGenericGen(ctx, pgran, &gpriv);
    }
    return r;
}

int
f4zeroBlockGen(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const PGranularity *pgran,
    const char *memPrefix)
{
    char tmp[1024];
    ItemWork work;
    LoopCtl loopCtl;
    GenPriv priv;
    char pref;
    LoopUnrollers unrollers;

    if (!strcmp(memPrefix, "__local")) {
        pref = 'l';
    }
    else if (!strcmp(memPrefix, "__global")) {
        pref = 'g';
    }
    else {
        return -EINVAL;
    }

    if (dim->y != 1) {
        return -EINVAL;
    }

    memset(&loopCtl, 0, sizeof(loopCtl));
    memset(&unrollers, 0, sizeof(unrollers));
    memset(&priv, 0, sizeof(GenPriv));
    initGenPriv(&priv, TYPE_COMPLEX_DOUBLE, FLOAT4_VECLEN * sizeof(cl_float),
                dim, 0, (const ItemWork*)&work, pgran);
    getItemWork(&work, dim, pgran, priv.nfloats, priv.vecLen);

    sprintf(tmp, f4zeroDecl, pref, dim->x, memPrefix);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    // declare local ID variable and set data offset
    kgenDeclareLocalID(ctx, lidVarName, pgran);
    sprintf(tmp, "\ndata += %s * %lu;\n\n",
            lidVarName, work.nrCols);
    kgenAddStmt(ctx, tmp);

    unrollers.genSingle = f4zeroSingle;
    loopCtl.inBound = (unsigned int)work.nrCols;
    unrollers.getVecLen = getVecLen;

    kgenLoopUnroll(ctx, &loopCtl, TYPE_COMPLEX_DOUBLE, &unrollers, &priv);
    if (work.tail) {
        addTailCode(ctx, &priv, NULL, f4zeroSingle);
    }

    return kgenEndFuncBody(ctx);
}
