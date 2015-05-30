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


#include <sys/types.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <defbool.h>
#include <clblas_stddef.h>

#include "blas_kgen.h"

// assign tile's base name to 'name' if it is assigned to zero pointer
static __inline void
selectTileBaseName(Tile *tile, const char *name)
{
    if (tile->baseName == NULL) {
        tile->baseName = name;
    }
}

static void
selectDefaultTileVecLen(
    Tile *tile,
    TileCreationFlags tflags,
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    MatrixRole mrole)
{
    if (tflags & TILE_WITH_FETCH_VECLEN) {
        tile->vecLen = getVecLen(gset, funcID, mrole);
    }
    else {
        size_t w;

        w = (tile->trans) ? tile->nrRows : tile->nrCols;
        if (tile->packed) {
            size_t wpad, height;

            wpad = roundUpPow2(w);
            height = (tile->trans) ? tile->nrCols : tile->nrRows;
            tile->vecLen = (unsigned int)szmin(height * wpad, MAX_TILE_VECLEN);
        }
        else {
            tile->vecLen = (unsigned int)roundUpPow2(w);
            tile->vecLen = (unsigned int)szmin(tile->vecLen, MAX_TILE_VECLEN);
        }
    }
}

// physical tile pitch, can be less than one vector in case of packed mode
static unsigned int
tilePitch(const Tile *tile)
{
    unsigned int pitch;

    if (!tile->trans) {
        if (tile->packed) {
            pitch = (unsigned int)roundUpPow2(tile->nrCols);
        }
        else {
            pitch = (unsigned int)roundUp(tile->nrCols, tile->vecLen);
        }
    }
    else {
        if (tile->packed) {
            pitch = (unsigned int)roundUpPow2(tile->nrRows);
        }
        else {
            pitch = (unsigned int)roundUp(tile->nrRows, tile->vecLen);
        }
    }

    return pitch;
}

void
initTile(
    Tile *tile,
    const char *baseName,
    unsigned int nrRows,
    unsigned int nrCols,
    unsigned int vecLen,
    DataType dtype,
    PrivateStorageType storType,
    bool trans,
    bool packed)
{
    assert(baseName == NULL || strlen(baseName) <= MAX_TILE_BASE_NAMELEN);

    tile->baseName = baseName;
    tile->nrRows = nrRows;
    tile->nrCols = nrCols;
    tile->vecLen = umin(MAX_TILE_VECLEN, vecLen);
    tile->dtype = dtype;
    tile->storType = storType;
    tile->trans = trans;
    tile->packed = packed;
}

void
initDefaultTiles(
    BlasGenSettings *gset,
    BlasFunctionID funcID,
    TileCreationFlags flags,
    PrivateStorageType storType)
{
    const SubproblemDim *dim = &gset->subdims[1];
    KernelExtraFlags kflags = gset->kextra->flags;
    DataType dtype = gset->kextra->dtype;
    Tile *tile;
    const char *name;
    int level;
    bool packed;

    level = funcBlasLevel(funcID);
    packed = ((flags & TILE_PACKED) != 0);

    tile = &gset->tileA;
    selectTileBaseName(tile, "a");
    initTile(tile, tile->baseName, (unsigned int)dim->y,
             (unsigned int)dim->bwidth, 1, dtype, storType, false, packed);

    tile->trans = isMatrixAccessColMaj(funcID, kflags, MATRIX_A);
    if (!(gset->flags & BGF_WHOLE_A)) {
        if (tile->trans) {
            tile->nrCols = 1;
        }
        else {
            tile->nrRows = 1;
        }
    }
    selectDefaultTileVecLen(tile, flags, gset, funcID, MATRIX_A);

    tile = &gset->tileBX;
    name = (level == 2) ? "x" : "b";
    selectTileBaseName(tile, name);
    initTile(tile, tile->baseName, (unsigned int)dim->bwidth,
             (unsigned int)dim->x, 1, dtype, storType, false, packed);

    /*
     * NOTE: Tiles for the level 2 functions are forced to be transposed
     *       in order to allow user to fetch elements belonging to different
     *       rows which is very useful in case of unit increment between
     *       elements because provides faster access to the global memory.
     */
    if (level == 2) {
        tile->trans = true;
    }
    else {
        tile->trans = !isMatrixAccessColMaj(funcID, kflags, MATRIX_B);
    }
    selectDefaultTileVecLen(tile, flags, gset, funcID, MATRIX_B);

    tile = &gset->tileCY;
    name = (level == 2) ? "y" : "c";
    selectTileBaseName(tile, name);

    initTile(tile, tile->baseName, (unsigned int)dim->y,
             (unsigned int)dim->x, 1, dtype, storType, false,
             packed);

    if (level == 2) {
        tile->trans = true;
    }
    else if (!(flags & TILE_C_FORCE_NOTRANS)) {
        tile->trans = isMatrixAccessColMaj(funcID, kflags, MATRIX_C);
    }
    selectDefaultTileVecLen(tile, flags, gset, funcID, MATRIX_C);

    // FIXME: remove the restriction
    /*if (isComplexType(tile->dtype)) {
        tile->vecLen = 1;
    }*/
}

unsigned int
tileVectorsNum(const Tile *tile)
{
    size_t pitch, height;

    pitch = tilePitch(tile);
    height = (tile->trans) ? tile->nrCols : tile->nrRows;

    return (unsigned int)divRoundUp(height * pitch, tile->vecLen);
}

unsigned int
tileStorageSize(const Tile *tile)
{
    unsigned int u;

    u = tileVectorsNum(tile) * tile->vecLen;

    return u;
}

unsigned int
tileLineSegmentLen(const Tile *tile)
{
    unsigned int pitch;
    unsigned int len;

    pitch = tilePitch(tile);
    len = umin(pitch, tile->vecLen);
    if (tile->trans) {
        len = umin(len, tile->nrRows);
    }
    else {
        len = umin(len, tile->nrCols);
    }

    return len;
}

int
declareOneTileStorage(struct KgenContext *ctx, const Tile *tile)
{
    char tmp[1024];
    const char *tname;
    int r;
    size_t size;

    getVectorTypeName(tile->dtype, tile->vecLen, &tname, NULL);
    size = tileVectorsNum(tile);
    if (tile->storType == PRIV_STORAGE_ARRAY) {
        sprintf(tmp, "%s %s[%lu];\n", tname, tile->baseName, size);
    }
    else {
        size_t i;
        char *p;

        sprintf(tmp, "%s %s0", tname, tile->baseName);
        p = tmp + strlen(tmp);
        for (i = 1; i < size; i++) {
            sprintf(p, ", %s%lu", tile->baseName, i);
            p += strlen(p);
        }
        strcpy(p, ";\n");
    }

    r = kgenAddStmt(ctx, tmp);

    return (r) ? -EOVERFLOW : 0;
}

int
declareTileStorages(struct KgenContext *ctx, const BlasGenSettings *gset)
{
    int ret;

    ret = declareOneTileStorage(ctx, &gset->tileA);
    if (!ret) {
        ret = declareOneTileStorage(ctx, &gset->tileBX);
    }
    if (!ret) {
        declareOneTileStorage(ctx, &gset->tileCY);
    }

    return ret;
}

void
sprintfTileElement(
    Kstring *str,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    unsigned int len)
{
    unsigned int pitch;
    unsigned int elemLen;
    unsigned int off;
    unsigned int vecLen = tile->vecLen;
    char vchunk[24];

    if (len == 0) {
        len = vecLen;
    }

    pitch = tilePitch(tile);
    elemLen = isComplexType(tile->dtype) ? 2 : 1;
    if (!tile->trans) {
        assert((row < tile->nrRows) && (col + len <= tile->nrCols));
        off = (row * pitch + col) * elemLen;
    }
    else {
        assert((row + len <= tile->nrRows) && (col < tile->nrCols));
        off = (col * pitch + row) * elemLen;
    }

    vecLen *= elemLen;
    sprintfVecChunk(vchunk, vecLen, len * elemLen, off % vecLen);

    if (tile->storType == PRIV_STORAGE_ARRAY) {
        sprintf(str->buf, "%s[%u]%s", tile->baseName, off / vecLen, vchunk);
    }
    else {
        sprintf(str->buf, "%s%u%s", tile->baseName, off / vecLen, vchunk);
    }
}

void
sprintfTileElementHalf(
    Kstring *str,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    TileElementHalf half)
{
    int len;

    assert(isComplexType(tile->dtype));

    // sprintf the full element and the drop an unneded half
    sprintfTileElement(str, tile, row, col, 1);
    len = (int)strlen(str->buf);
    if (half == TE_HALF_HIGH) {
        str->buf[len - 2] = str->buf[len - 1];
    }
    str->buf[len - 1] = '\0';
}

int
forEachTile(Kstring *kstr, unsigned int row, unsigned int col,
            unsigned int num, Tile *first, ...)
{
   unsigned int minVecLen = first->vecLen;
   unsigned int valRow = first->nrRows;
   unsigned int valCol = first->nrCols;
   va_list argptr;
   unsigned int i;

   va_start(argptr, first);
   for (i = 1; i < num; i++) {
       Tile * cur = va_arg( argptr, Tile * );
       minVecLen = umin(minVecLen, cur->vecLen);
   }
   va_end(argptr);

   if (first->trans) {
       valRow /= minVecLen;
   }
   else {
       valCol /= minVecLen;
   }

   if (row >= valRow || col >= valCol /*|| row < 0 || col < 0*/) { //would be signed
       return 0;
   }
   if (kstr) {
       va_start(argptr, first);
       for (i = 0; i < num; i++) {
           Tile * cur = i ? va_arg( argptr, Tile * ) : first;
           if (cur->baseName) {
               unsigned int vRow = (cur->trans ? row * minVecLen : row);
               unsigned int vCol = (cur->trans ? col : col * minVecLen);
               sprintfTileElement(&kstr[i], cur, vRow, vCol, minVecLen);
           }
       }
       va_end(argptr);
   }
   return first->trans ? valRow : valCol;
}

void
genSetZeroInTile(
    struct KgenContext *ctx,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    unsigned int len)
{
    char tmp[1024];
    Kstring elem;

    sprintfTileElement(&elem, tile, row, col, len);
    sprintf(tmp, "%s = 0;\n", elem.buf);
    kgenAddStmt(ctx, tmp);
}

void
genSetUnitInTile(
    struct KgenContext *ctx,
    const Tile *tile,
    unsigned int row,
    unsigned int col)
{
    char tmp[1024];
    Kstring elem;
    const char *s;

    sprintfTileElement(&elem, tile, row, col, 1);
    s = strOne(tile->dtype);
    sprintf(tmp, "%s = %s;\n", elem.buf, s);
    kgenAddStmt(ctx, tmp);
}

void
genZeroTile(struct KgenContext *ctx, const Tile *tile)
{
    char tmp[1024];
    Kstring elem;
    unsigned int incRows, incCols;
    unsigned int i, j, v;

    v = tileLineSegmentLen(tile);
    if (!tile->trans) {
        incRows = 1;
        incCols = v;
    }
    else {
        incRows = v;
        incCols = 1;
    }

    for (i = 0; i < tile->nrRows; i += incRows) {
        for (j = 0; j < tile->nrCols; j += incCols) {
            sprintfTileElement(&elem, tile, i, j, v);
            sprintf(tmp, "%s = 0;\n", elem.buf);
            kgenAddStmt(ctx, tmp);
        }
    }

    kgenAddBlankLine(ctx);
}

void
genTileCopy(
    struct KgenContext *ctx,
    const Tile *dst,
    const Tile *src,
    TileCopyOps op)
{
    char tmp[1024];
    Kstring el1, el2;
    unsigned int nrRows, nrCols;
    unsigned int incRows, incCols;
    unsigned int vlen;
    unsigned int i, j;

    nrRows = umin(dst->nrRows, src->nrRows);
    nrCols = umin(dst->nrCols, src->nrCols);
    if (dst->trans != src->trans) {
        vlen = 1;
        incRows = incCols = 1;
    }
    else {
        vlen = umin(dst->vecLen, src->vecLen);
        if (!dst->trans) {
            incRows = 1;
            incCols = umin(dst->nrCols, src->nrCols);
            incCols = umin(incCols, vlen);
        }
        else {
            incRows = umin(dst->nrRows, src->nrRows);
            incRows = umin(incRows, vlen);
            incCols = 1;
        }
    }

    for (i = 0; i < nrRows; i += incRows) {
        for (j = 0; j < nrCols; j += incCols) {
            sprintfTileElement(&el1, dst, i, j, vlen);
            sprintfTileElement(&el2, src, i, j, vlen);
            switch( op )
            {
                case TILECOPY_ASSIGN:
                    sprintf(tmp, "%s = %s;\n", el1.buf, el2.buf);
                    break;

                case TILECOPY_ADD_ASSIGN:
                    sprintf(tmp, "%s += %s;\n", el1.buf, el2.buf);
                    break;

                case TILECOPY_SUB_ASSIGN:
                    sprintf(tmp, "%s -= %s;\n", el1.buf, el2.buf);
                    break;

                case TILECOPY_MUL_ASSIGN:
                    sprintf(tmp, "%s *= %s;\n", el1.buf, el2.buf);
                    break;

                case TILECOPY_DIV_ASSIGN:
                    sprintf(tmp, "%s /= %s;\n", el1.buf, el2.buf);
                    break;

                case TILECOPY_MOD_ASSIGN:
                    sprintf(tmp, "%s %%= %s;\n", el1.buf, el2.buf);
                    break;

                default:
                    break;
            }
            kgenAddStmt(ctx, tmp);
        }
    }

    kgenAddBlankLine(ctx);
}
