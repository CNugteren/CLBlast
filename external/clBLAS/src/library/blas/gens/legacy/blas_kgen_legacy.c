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
 * common stuff for blas related
 * kernel generators, legacy part
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <list.h>
#include <clblas_stddef.h>

#include <matrix_props.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "blas_kgen_legacy.h"

void
declareBlasEnums(struct KgenContext *ctx)
{
    kgenAddStmt(ctx,
        "typedef enum clblasOrderEnum {\n"
        "   clblasRowMajor,\n"
        "   clblasColumnMajor\n"
        "} clblasOrder;\n"
        "\n"
        "typedef enum clblasTransposeEnum {\n"
        "   clblasNoTrans,\n"
        "   clblasTrans,\n"
        "   clblasConjTrans\n"
        "} clblasTranspose;\n"
        "\n"
        "typedef enum clblasUploEnum {\n"
        "   clblasUpper,\n"
        "   clblasLower\n"
        "} clblasUplo;\n"
        "\n"
        "typedef enum clblasDiagEnum {\n"
        "   clblasUnit,\n"
        "   clblasNonUnit\n"
        "} clblasDiag;\n"
        "\n"
        "typedef enum clblasSideEnum {\n"
        "   clblasLeft,\n"
        "   clblasRight\n"
        "} clblasSide;\n\n");
}

static unsigned int
getTmpVecLen(
    const BlasGenSettings *gset,
    UpdateResultFlags uflags,
    const char **vecName)
{
    const CLBLASKernExtra *kextra = gset->kextra;
    unsigned int vecLen;

    if (isComplexType(kextra->dtype) || (uflags & (UPRES_GENERIC |
                                         UPRES_NO_VECTORIZATION))) {
        vecLen = 1;
    }
    else {
        vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? kextra->vecLenC :
                                                       kextra->vecLen;
        getVectorTypeName(kextra->dtype, vecLen, vecName, NULL);
    }

    return vecLen;
}

static void
updateOptimResultGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    unsigned int wvlen,
    unsigned int pitch,
    unsigned int regOff,
    const char *ldName,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const char *cachedName)
{
    char tmp[1024];
    int tra, isDouble;
    bool useReg = true;
    char *regRole;
    char dst[80], src[80];
    char vchunkTmp[64], vchunkReg[64];
    unsigned int sizes[2];
    unsigned int i, j, k;
    unsigned int off;
    const char *vfield;
    DataType dtype = gset->kextra->dtype;
    bool isPrivDest = ((flags & UPRES_PRIV_DEST) != 0);
    unsigned int vecLen;     // vector length of the result's register block
    // vector length to update with at immediate operations
    unsigned int uplen;
    // vector length of the temporary storage location
    unsigned int tmpVecLen;
    const char *ptrName;

    sizes[0] = (unsigned int)gset->subdims[1].y;
    sizes[1] = (unsigned int)gset->subdims[1].x;

    j = 0;
    tra = ((flags & UPRES_COLUMN_MAJOR) != 0);
    isDouble = isDoubleBasedType(dtype);
    vfield = dtypeUPtrField(dtype);
    vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? gset->kextra->vecLenC :
                                                   gset->kextra->vecLen;
    tmpVecLen = getTmpVecLen(gset, flags, NULL);
    getVectorTypeName(dtype, wvlen, NULL, &ptrName);
    if (isComplexType(dtype)) {
        vecLen = 1;
    }
    uplen = (tra || (flags & UPRES_NO_VECTORIZATION)) ? 1 : vecLen;

    /*
     * Pass recursively over the major dimension with power of 2 vectors.
     * If the used type size is less then the current vector size,
     * use assembling/disassembling into/from a temporary vector. This is
     * for trying to increase effectiveness of operations with the global
     * memory due to vectorization.
     */
    if (wvlen > sizes[1 - tra]) {
        wvlen /= 2;
        updateOptimResultGen(ctx, gset, wvlen, pitch, regOff, ldName,
                             op, flags, cachedName);
        return;
    }

    if (wvlen == 1) {
        kgenAddStmt(ctx, "// Copy with single words\n");
    }
    else {
        const char *s = (isDouble) ? "double" : "float";

        sprintf(tmp, "// Copy with %s%d vectors\n", s, wvlen);
        kgenAddStmt(ctx, tmp);
    }

    for (i = 0; i < sizes[tra]; i++) {
        unsigned int roff;

        if (tra) {
            roff = regOff + i;
        }
        else {
            roff = regOff + i * pitch;
        }

        for (j = 0; j < sizes[1 - tra] / wvlen; j++) {
            if (wvlen > uplen) {
                if (isPrivDest) {
                    sprintfVecChunk(vchunkTmp, tmpVecLen, wvlen, 0);
                    sprintf(tmp, "tmp%s = uC.%s[%u];\n",
                            vchunkTmp, ptrName, j);
                    kgenAddStmt(ctx, tmp);
                }
                else {
                    // assemble vector
                    for (k = 0; k < wvlen; k += uplen) {
                        off = (tra) ? (roff + k * pitch) : (roff + k);
                        sprintfVecChunk(vchunkTmp, tmpVecLen, uplen, k);
                        sprintfVecChunk(vchunkReg, vecLen, uplen, off % vecLen);
                        sprintf(tmp, "tmp%s = c[%u]%s;\n",
                                vchunkTmp, off / vecLen, vchunkReg);
                        kgenAddStmt(ctx, tmp);
                    }
                }
            }

            if (isPrivDest && (wvlen > uplen)) {
                // disassemble temporary vector and do immediate result update
                for (k = 0; k < wvlen; k += uplen) {
                    off = (tra) ? (roff + k * pitch) : (roff + k);
                    sprintfVecChunk(vchunkTmp, tmpVecLen, uplen, k);
                    sprintfVecChunk(vchunkReg, vecLen, uplen, off % vecLen);
                    sprintf(src, "tmp%s", vchunkTmp);
                    sprintf(dst, "c[%u]%s", off / vecLen, vchunkReg);
                    genUpdateResultSingle(ctx, dst, src, gset, op, flags);
                }
            }
            else {
                if (wvlen > uplen) {
                    sprintfVecChunk(vchunkTmp, tmpVecLen, wvlen, 0);
                    sprintf(src, "tmp%s", vchunkTmp);
                    useReg = false;
                }

                if (!isPrivDest) {
                    sprintf(dst, "uC.%s[%u]", ptrName, j);
                    if (cachedName) {
                        char *p = dst + strlen(dst);
                        strcat(p, " = ");
                        p = dst + strlen(dst);
                        sprintf(p, cachedName, i, j);
                    }
                    regRole = src;
                }
                else {
                    useReg = true;
                    regRole = dst;
                    sprintf(src, "uC.%s[%u]", ptrName, j);
                }

                if (useReg) {
                    sprintfVecChunk(vchunkReg, vecLen, uplen, roff % vecLen);
                    sprintf(regRole, "c[%u]%s", roff / vecLen, vchunkReg);
                }

                genUpdateResultSingle(ctx, dst, src, gset, op, flags);
            }

            // update register offset
            if (tra) {
                roff += wvlen * pitch;
            }
            else {
                roff += wvlen;
            }
        }

        // move the destination pointer to the next line
        if ((i != sizes[tra] - 1)) {
            sprintf(tmp, "uC.%s += %s;\n", vfield, ldName);
            kgenAddStmt(ctx, tmp);
            if (tra) {
                kgenAddBlankLine(ctx);
            }
        }
    }

    if (j * wvlen != sizes[1 - tra]) {
        // increment pointers
        if (tra) {
            regOff += j * wvlen * pitch;
        }
        else {
            regOff += j * wvlen;
        }

        sprintf(tmp, "\n"
                     "uC.%s = tmpC.%s + %u;\n"
                     "tmpC = uC;\n",
                vfield, vfield, j * wvlen);
        kgenAddStmt(ctx, tmp);

        // go down
        sizes[1 - tra] -= j * wvlen;
        wvlen /= 2;
        updateOptimResultGen(ctx, gset, wvlen, pitch, regOff, ldName,
                             op, flags, cachedName);
    }
}

static void
updateGenericResultGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    size_t pitch,
    UpresVarNames* uvars,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const char *cachedName)
{
    char tmp[1024], dst[128], src[128];
    const char *boundNames[2] = {uvars->nrRows, uvars->nrCols};
    const char *vecType = NULL;
    const char *vFieldVectorized;
    DataType dtype = gset->kextra->dtype;
    unsigned int wvlen;
    unsigned int sizes[2];
    const char*  vfield = dtypeUPtrField(dtype);
    bool tra = ((flags & UPRES_COLUMN_MAJOR) != 0);
    bool row = ((flags & UPRES_TAIL_ROW));
    bool col = ((flags & UPRES_TAIL_COL));
    bool iwc = ((flags & UPRES_INDEXING_WITH_CONSTANTS) != 0);
    int l0;
    int l1;
    unsigned int vecLen;     // vector length of the result's register block
    // vector length to update with at immediate operations
    unsigned int uplen;
    // vector length of the temporary storage location
    char vchunkReg[64];
    bool revert = false;

    vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? gset->kextra->vecLenC :
                                                   gset->kextra->vecLen;
    if (isComplexType(dtype)) {
        vecLen = 1;
    }
    uplen = (tra || (flags & UPRES_NO_VECTORIZATION)) ? 1 : vecLen;
    uplen = 1;


    sizes[0] = (unsigned int)gset->subdims[1].y;
    sizes[1] = (unsigned int)gset->subdims[1].x;

    if (iwc) {
        const char* l0var =  boundNames[tra];
        revert =  (tra && col) || (!tra && row);

        if (revert) {
            sprintf(tmp, "uC.%s += (%s-1) * %s;\n", vfield, l0var, uvars->ld);
        }
        else {
            sprintf(tmp, "\n");
        }
        kgenAddStmt(ctx, tmp);

    }
    wvlen = getTmpVecLen(gset, flags, &vecType);
    getVectorTypeName(dtype, wvlen, NULL, &vFieldVectorized);
    sprintf(tmp, "res.%s = c;\n", vFieldVectorized);
    kgenAddStmt(ctx, tmp);

    if (flags & (UPRES_TAIL_ROW | UPRES_TAIL_COL)) {
        char offStr[64];
        char *p = offStr;

        offStr[0] = '\0';
        if (flags & UPRES_TAIL_ROW) {
            sprintf(offStr, " + (%u - %s) * %lu",
                    sizes[0], uvars->nrRows, pitch);
            p += strlen(offStr);
        }
        if (flags & UPRES_TAIL_COL) {
            sprintf(p, " + (%u - %s)", sizes[1], uvars->nrCols);
        }
        if (iwc) {
            sprintf(tmp, "res.%s = uC.%s%s;\n", vfield, vfield, offStr);
            sprintf(tmp, "\n");
        }
        else {
            sprintf(tmp, "res.%s = res.%s%s;\n", vfield, vfield, offStr);
        }
        kgenAddStmt(ctx, tmp);

    }
    if (iwc) {
        int l0st = 1; int l0en = sizes[tra];
        int l1st = 1; int l1en = sizes[1-tra];

        const char* l0var =  boundNames[tra];
        const char* l1var = boundNames[1-tra];

        for (l0 = l0en; l0 >= l0st; l0--) {

            sprintf(tmp, "if (%s) ",l0var);
            kgenBeginBranch(ctx, tmp);

            sprintf(tmp, "switch (%s)", l1var);
            kgenBeginBranch(ctx, tmp);

            for (l1 = l1en; l1 >= l1st; l1--) {
                int resId;

                sprintf(tmp, "case %d:\n", l1);
                kgenAddStmt(ctx, tmp);

                if (tra) {
                    resId = (row)
                             ? (l1en-l1)*(int)pitch
                             : (l1-l1st)*(int)pitch;

                    resId += (col)? (l0-l0st): (l0en-l0);
                }
                else {
                    ///////////////////////////
                    resId = (row)
                            ? (l0-l0st)*(int)pitch
                            : (l0en-l0)*(int)pitch;
                    resId += (col)? (l1en-l1) : (l1-l1st);
                }

                if ((tra && row) || (!tra && col)) {
                     sprintf(dst, "uC.%s[(%s+%d) %% %i]",
                             vfield, l1var, (l1en - l1),  (int)l1en);
                }
                else {
                   sprintf(dst, "uC.%s[%d]", vfield, (l1-l1st));
                }
                sprintfVecChunk(vchunkReg, vecLen, uplen, resId % vecLen);
                sprintf(src, "c[%u]%s", resId / vecLen, vchunkReg);

                if (flags & UPRES_PRIV_DEST) {
                    genUpdateResultSingle(ctx, src, dst, gset, op, flags);
                }
                else {
                    genUpdateResultSingle(ctx, dst, src, gset, op, flags);
                }
            }
            kgenEndBranch(ctx, NULL);

            if (revert) {
                sprintf(tmp, "uC.%s -= %s;\n", vfield, uvars->ld);
            }
            else {
                sprintf(tmp, "uC.%s += %s;\n", vfield, uvars->ld);
            }

            kgenAddStmt(ctx, tmp);

            sprintf(tmp, "%s--;\n", l0var);
            kgenAddStmt(ctx, tmp);
            kgenEndBranch(ctx, NULL);
        }

    }
    else {

        sprintf(tmp, "for (i = 0; i < %s; i++)", boundNames[tra]);
        kgenBeginBranch(ctx, tmp);
        sprintf(tmp, "for (j = 0; j < %s; j++)", boundNames[1 - tra]);
        kgenBeginBranch(ctx, tmp);
        sprintf(dst, "uC.%s[i * %s + j]", vfield, uvars->ld);
        if (cachedName) {
            unsigned int i;
            char tmpcachedName[80] = " = ";
            strcat(tmpcachedName, cachedName);
            for (i = 3; i < strlen(tmpcachedName); i++) {
                if (strncmp(tmpcachedName+i, "%u", 2) == 0) {
                    tmpcachedName[i+1] = 's';
                }
            }
            sprintf(tmp, tmpcachedName, "i", "[j]");
            strcat(dst, tmp);
        }
        if (tra) {
            sprintf(src, "res.%s[j * %lu + i]", vfield, pitch);
        }
        else {
            sprintf(src, "res.%s[i * %lu + j]", vfield, pitch);
        }
        if (flags & UPRES_PRIV_DEST) {
            genUpdateResultSingle(ctx, src, dst, gset, op, flags);
        }
        else {
            genUpdateResultSingle(ctx, dst, src, gset, op, flags);
        }
        kgenEndBranch(ctx, NULL);
        kgenEndBranch(ctx, NULL);
    }
}

int
updateResultGenOld(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const UpresVarNames *uvarNames)
{
    char tmp[1024];
    char *p = tmp;
    const char *typeName;
    const char *vecType = NULL;
    const char *vfield;
    const char *suff1;
    const char *suff2;
    int ret = 0;
    unsigned int sizes[2];
    bool generic, tra;
    unsigned int wvlen;     // length of vectors to copy with
    unsigned int uplen;     // length of vectors to update result with
    size_t pitch;
    char LG;
    DataType dtype = gset->kextra->dtype;
    unsigned int vecLen;
    bool isInlined = (flags & UPRES_INLINE);
    UpresVarNames uvars;

    vecLen = (gset->flags & BGF_DISTINCT_VECLEN) ? gset->kextra->vecLenC :
                                                   gset->kextra->vecLen;
    sizes[0] = (unsigned int)gset->subdims[1].y;
    sizes[1] = (unsigned int)gset->subdims[1].x;

    if (isComplexType(dtype)) {
        vecLen = 1;
    }

    if ((flags & UPRES_WITH_BETA) && (op != UPRES_SUM)) {
        return -EINVAL;
    }

    tra = ((flags & UPRES_COLUMN_MAJOR) != 0);
    generic = ((flags & UPRES_GENERIC) != 0);
    typeName = dtypeBuiltinType(dtype);
    vfield = dtypeUPtrField(dtype);
    pitch = roundUp(sizes[1], vecLen);

    // select write vectorization
    wvlen = getTmpVecLen(gset, flags, &vecType);
    uplen = (tra || (flags & UPRES_NO_VECTORIZATION)) ? 1 : vecLen;

    suff1 = (generic) ? "Generic" : "";
    suff2 = (flags & UPRES_PRIV_DEST) ? "Rev" : "";
    LG = (flags & UPRES_USE_LDS) ? 'L' : 'G';

    if (!isInlined) {
        const char *outTypeName;
        const char *memPref = (flags & UPRES_USE_LDS) ? "__local" :
                                                           "__global";

        getResultGPRsInfo(dtype, NULL, vecLen, NULL, &outTypeName);

        // define the function
        sprintf(tmp, "void\n"
                     "updateResult%s%s%c(\n"
                     "    %s %s *C,\n"
                     "    %s *c,\n"
                     "    %s alpha,\n"
                     "    uint startRow,\n"
                     "    uint startCol,\n"
                     "    uint ld",
                     suff1, suff2, LG, memPref, typeName,
                     outTypeName, typeName);

        p += strlen(p);
        if (flags & UPRES_WITH_BETA) {
            sprintf(p, ",\n    %s beta", typeName);
            p += strlen(p);
        }
        if (generic) {
            sprintf(p, ",\n    uint nrRows,\n"
                       "    uint nrCols");
        }

        uvars.result = "C";
        uvars.ld = "ld";
        uvars.startRow = "startRow";
        uvars.startCol = "startCol";
        uvars.nrRows = "nrRows";
        uvars.nrCols = "nrCols";

        strcat(p, ")\n");
        kgenDeclareFunction(ctx, tmp);
        kgenBeginFuncBody(ctx);
    }
    else {
        memcpy(&uvars, uvarNames, sizeof(uvars));
    }

    // declare local variables
    sprintf(tmp, "%cPtr uC;\n", LG);
    kgenAddStmt(ctx, tmp);
    if (generic) {
        kgenAddStmt(ctx, "int i, j;\n"
                         "PPtr res;\n");
    }
    else {
        /*
         * temporary pointer to pass correctly over the
         * destination array since destination rows can be
         * not aligned on a vector bound
         */
        if (sizes[1 - tra] % wvlen != 0) {
            sprintf(tmp, "%cPtr tmpC;\n", LG);
            kgenAddStmt(ctx, tmp);
        }
        if (wvlen > uplen) {
            sprintf(tmp, "%s tmp;\n", vecType);
            kgenAddStmt(ctx, tmp);
        }
    }
    if (isComplexType(dtype) && !(flags & UPRES_WITHOUT_ALPHA)) {
        declareComplexMultParts(ctx, "alpha", typeName);
        if (flags & UPRES_WITH_BETA) {
            declareComplexMultParts(ctx, "beta", typeName);
        }

    }
    kgenAddBlankLine(ctx);

    if (tra) {
        sprintf(tmp, "uC.%s = %s + %s * %s + %s;\n",
                vfield, uvars.result, uvars.startCol, uvars.ld,
                uvars.startRow);
    }
    else {
        sprintf(tmp, "uC.%s = %s + %s * %s + %s;\n",
                vfield, uvars.result, uvars.startRow, uvars.ld,
                uvars.startCol);
    }
    kgenAddStmt(ctx, tmp);

    if ((sizes[1 - tra] % wvlen != 0) && !generic) {
        kgenAddStmt(ctx, "tmpC = uC;\n");
    }
    ret = kgenAddBlankLine(ctx);

    if (generic) {
        updateGenericResultGen(ctx, gset, pitch, &uvars, op, flags,
                               uvarNames ? uvarNames->cachedName : NULL);
    }
    else {
        updateOptimResultGen(ctx, gset, wvlen, (unsigned int)pitch, 0, uvars.ld,
                           op, flags, uvarNames ? uvarNames->cachedName : NULL);
    }

    if (!isInlined) {
        ret = kgenEndFuncBody(ctx);
    }

    return (ret) ? -EOVERFLOW : 0;
}
