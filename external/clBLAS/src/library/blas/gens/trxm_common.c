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

#include "trxm_common.h"

void
declareTrxmKernel(
    struct KgenContext *ctx,
    DataType dtype,
    const PGranularity *pgran,
    KernelExtraFlags kflags,
    BlasFunctionID funcID,
    const char *nameSuffix,
    bool declareC,
    bool restrictPointers)
{
    char tmp[1024];
    char strC[1024];
    char fpref, fsuff;
    const char *typeName;
    // swap coordinate names for the right side
    char coordNames[2] = {'M', 'N'};
    int side = ((kflags & KEXTRA_SIDE_RIGHT) != 0);
    char offStr[1024];
    int len = 0;
    const char *qualA[2], *qualB[2];      // type qualifiers

    typeName = dtypeBuiltinType(dtype);
    fpref = dtypeToBlasPrefix(dtype);
    fsuff = (funcID == CLBLAS_TRMM) ? 'm' : 's';
    if (nameSuffix == NULL) {
        nameSuffix = "";
    }
    strC[0] = '\0';
    if (declareC) {
        sprintf(strC, "    __global %s *C,\n", typeName);
    }

    offStr[0] = '\0';
    if (kflags & KEXTRA_STARTM_NOT_ZERO) {
        len = sprintf(offStr, ",\n    uint offset%c", coordNames[side]);
    }
    if (kflags & KEXTRA_STARTN_NOT_ZERO) {
        len += sprintf(offStr + len, ",\n    uint offset%c",
                       coordNames[1 - side]);
    }
    if (kflags & KEXTRA_A_OFF_NOT_ZERO) {
        strcat(offStr, ",\n    uint offA");
    }
    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        strcat(offStr, ",\n    uint offB");
    }

    if (restrictPointers) {
        qualA[0] = "const ";
        qualA[1] = "restrict ";
    }
    else {
        qualA[0] = qualA[1] = "";
    }

    if (restrictPointers && declareC) {
        qualB[0] = "const ";
        qualB[1] = "restrict ";
    }
    else {
        qualB[0] = qualB[1] = "";
    }

    sprintf(tmp, "__attribute__((reqd_work_group_size(%u, 1, 1)))\n"
                 "void __kernel\n"
                 "%ctr%cm%s(\n"
                 "    uint %c,\n"
                 "    uint %c,\n"
                 "    %s alpha,\n"
                 "    %s__global %s *%sA,\n"
                 "    uint lda,\n"
                 "    %s__global %s *%sB,\n"
                 "%s"
                 "    uint ldb%s)\n",
            pgran->wgSize[0], fpref, fsuff, nameSuffix, coordNames[side],
            coordNames[1 - side], typeName, qualA[0], typeName, qualA[1],
            qualB[0], typeName, qualB[1], strC, offStr);

    kgenDeclareFunction(ctx, tmp);
}


void
genTrxmBMatrShift(
    struct KgenContext *ctx,
    KernelExtraFlags kflags,
    bool useC)
{
    char tmp[1024], addstr[1024];
    int len = 0;
    const char *opstr;
    char coordNames[2] = {'M', 'N'};
    int side = (int)((kflags & KEXTRA_SIDE_RIGHT) != 0);
    bool cmaj = ((kflags & KEXTRA_COLUMN_MAJOR) != 0);

    if (kflags & KEXTRA_BX_OFF_NOT_ZERO) {
        len = sprintf(addstr, "offB");
    }
    if (kflags & KEXTRA_STARTM_NOT_ZERO) {
        opstr = (len) ? " + " : "";

        if (cmaj) {
            len += sprintf(addstr + len, "%soffset%c",
                           opstr, coordNames[side]);
        }
        else {
            len += sprintf(addstr + len, "%soffset%c * ldb",
                           opstr, coordNames[side]);
        }
    }
    if (kflags & KEXTRA_STARTN_NOT_ZERO) {
        opstr = (len) ? " + " : "";

        if (cmaj) {
            len += sprintf(addstr + len, "%soffset%c * ldb",
                           opstr, coordNames[1 - side]);
        }
        else {
            len += sprintf(addstr + len, "%soffset%c",
                           opstr, coordNames[1 - side]);
        }
    }

    if (len) {
        sprintf(tmp, "B += %s;\n", addstr);
        kgenAddStmt(ctx, tmp);
        if (useC) {
            sprintf(tmp, "C += %s;\n", addstr);
            kgenAddStmt(ctx, tmp);
        }
        kgenAddBlankLine(ctx);
    }
}

void
fixupTrxmKargs(CLBlasKargs *kargs)
{
    size_t offA = (kargs->side == clblasRight) ? kargs->offsetN :
                                                    kargs->offsetM;
    kargs->offA += offA * kargs->lda.matrix + offA;
    if (kargs->order == clblasColumnMajor) {
        kargs->offBX += kargs->offsetN * kargs->ldb.matrix + kargs->offsetM;
    }
    else {
        kargs->offBX += kargs->offsetM * kargs->ldb.matrix + kargs->offsetN;
    }

    kargs->offsetM = kargs->offsetN = 0;
}

/* avoid " + 0" statements */
static void
genAdd(char *buf, size_t val)
{
    if (val == 0) {
        buf[0] = 0; //zero length string
    }
    else {
        sprintf(buf, " + %lu", val);
    }
}

int
genTrxmPostFetchZero(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv)
{
    TilePostFetchPrivate *pfPriv = (TilePostFetchPrivate*)priv;
    char tmp[1024];
    char stmtStr[512];

    const CLBLASKernExtra *kextra = pfPriv->gset->kextra;
    KernelExtraFlags kflags = kextra->flags;
    const KernelVarNames *vnames = &pfPriv->gset->varNames;
    char yCoordVar[64], xCoordVar[64];
    size_t blockx, blocky;
    unsigned int x, y;
    const struct SubproblemDim *dims = &pfPriv->gset->subdims[1];
    DataType dtype = pfPriv->gset->kextra->dtype;
    bool b;
    bool tra;
    Kstring kstr;
    const Tile* pTile = &pfPriv->gset->tileA;

    // For both A and B tiles, zero tail along K
    b = ((pfPriv->gset->flags & BGF_DISTINCT_VECLEN));
    if (checkForTailFetches(pfPriv->funcID, dims, kextra,
                            mrole, b, true) != FETCH_NO_TAILS) {
        defaultTilePostFetch(ctx, mrole, &pfPriv[1]);
    }

    if (mrole == MATRIX_B) {
        /* This is not triangular matrix, just go away from here */
        return 0;
    }

    blockx = blocky = 0;
    // zero triangular part of tile a
    // either single row of tile a either the whole tile have been fetched

    tra = isMatrixAccessColMaj(pfPriv->funcID, kflags, mrole);
    if (tra) {
        blocky = pfPriv->wholeA ? dims->bwidth : 1;
        blockx = dims->y;
        sprintf(xCoordVar, "%s", vnames->coordA);
        sprintf(yCoordVar, "%s", vnames->k);
    }
    else {
        blocky = pfPriv->wholeA ? dims->y : 1;
        blockx = dims->bwidth;
        sprintf(xCoordVar, "%s", vnames->k);
        sprintf(yCoordVar, "%s", vnames->coordA);
    }

    kgenAddStmt(ctx, "// post fetch A\n");
    kgenBeginBranch(ctx, NULL);

    genAdd(stmtStr, (size_t)pfPriv->fetchNumA);
    sprintf(tmp, "uint zy = %s%s;\n", yCoordVar, stmtStr);
    kgenAddStmt(ctx, tmp);

    // loop through block rows (there is only one row in A block)
    for(y = 0; y < blocky; y++) {
        // loop through all elements of block row
        for(x = 0; x < blockx; x++) {
            unsigned int row, col;
            char cmp = '<';

            row = (unsigned int)(tra ? x : y);
            col = (unsigned int)(tra ? y : x);

            if (((kflags & KEXTRA_UPPER_TRIANG) != 0) ^
                    ((kflags & KEXTRA_COLUMN_MAJOR) != 0)) {
                cmp = '>';
            }

            genAdd(stmtStr, x);
            sprintfTileElement(&kstr, pTile, row, col, 1);

            sprintf(tmp, "%s = zy %c %s%s ? 0 : %s;\n",
                    kstr.buf,
                    cmp, xCoordVar, stmtStr,
                    kstr.buf);

            kgenAddStmt(ctx, tmp);
            if (kflags & KEXTRA_UNIT_DIAGONAL) {
                const char *one = strOne(dtype);

                sprintf(tmp, "%s = zy == %s%s ? "
                        "%s : %s;\n",
                        kstr.buf, xCoordVar, stmtStr,
                        one, kstr.buf);
                kgenAddStmt(ctx, tmp);
            }
        }
        if (y != blocky - 1) {
            kgenAddStmt(ctx, "zy++;\n");
        }
        pfPriv->fetchNumA++;
    }

    return kgenEndBranch(ctx, NULL);
}
