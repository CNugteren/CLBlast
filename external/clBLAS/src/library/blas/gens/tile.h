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
 * API to manupulate with matrix tiles
 */

#ifndef TILE_H_
#define TILE_H_

#include <kerngen.h>
#include <blas_funcs.h>

#define tileLineElemNum forEachTile

struct BlasGenSettings;

enum {
    MAX_TILE_BASE_NAMELEN = sizeof(Kstring) - 25,
    /*
     * It may be 16 vector components at maximum. Adding the length of the
     * subscript and selector operator, 2 digit index, and the end-line symbol,
     * to the maximum base name length we get the maximum tile element string
     * length
     */
    MAX_TILE_ELEMENT_STRLEN = sizeof(Kstring) - 1,
    MAX_TILE_VECLEN = 8
};

/**
 * @internal
 * @brief Flags showing tile storing specifics
 * @ignroup TILES
 */
typedef enum TileCreationFlags {
    /** Tile C should be forced to non-transposed form */
    TILE_C_FORCE_NOTRANS = 0x01,
    /** tile vector length is equal to the length of fetched vectors */
    TILE_WITH_FETCH_VECLEN = 0x02,
    /**
     * If depending of transposing vector length is greater than
     * number of rows or columns, store several rows or columns respectively
     * in each vector
     */
    TILE_PACKED = 0x04
} TileCreationFlags;

/**
 * @internal
 * @brief Type of storage in the private memory
 * @ingroup TILES
 */
typedef enum PrivateStorageType {
    /** Tile is stored in array */
    PRIV_STORAGE_ARRAY,
    /** Tile is stored in a set of variables */
    PRIV_STORAGE_VARIABLE_SET
} PrivateStorageType;

typedef enum TileCopyOps {
    TILECOPY_ASSIGN,
    TILECOPY_ADD_ASSIGN,
    TILECOPY_SUB_ASSIGN,
    TILECOPY_MUL_ASSIGN,
    TILECOPY_DIV_ASSIGN,
    TILECOPY_MOD_ASSIGN
} TileCopyOps;

/**
 * @internal
 * @brief Tile element half types
 * @ingroup TILES
 */
typedef enum TileElementHalf {
    TE_HALF_LOW,
    TE_HALF_HIGH
} TileElementHalf;

/**
 * @internal
 * @brief Matrix tile stored in a private area
 * @ingroup TILES
 */
typedef struct Tile {
    const char *baseName;
    unsigned int nrRows;
    unsigned int nrCols;
    unsigned int vecLen;
    DataType dtype;
    PrivateStorageType storType;
    /** Flag of storing tile in the transposed form */
    bool trans;
    /*
     * Depending on the transposing several rows or columns can be fit
     * into single vector. It makes sense only when number of rows or column
     * respectively is less than vector length
     */
    bool packed;
} Tile;

/**
 * @internal
 * @brief Initialize tile
 *
 * @param[out] tile      Tile description structure to fill
 * @param[in] baseName   Tile base name
 * @param[in] nrRows     Number of rows in the tile
 * @param[in] nrCols     Number of columns in the tile
 * @param[in] vecLen     Length of one native OpenCL element being a part of
 *                       the tile
 * @param[in] dtype      Data type
 * @param[in] storType   Tile storate type
 * @param[in] trans      Shows if tile is stored in the transposed form
 *                       or direct
 * @param[in] packed     Tile is stored in packed form. Has not effect if
 *                       a single line can be fit into the single vector.
 *
 * If \b vecLen param is above MAX_TILE_VECLEN then will be truncated into
 * MAX_TILE_VECLEN.
 *
 * @ingroup TILES
 */
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
    bool packed);

/**
 * @internal
 * @brief Initialize matrix tile from generator settings
 *
 * @param[out] gset      Generator settings which tile should be initialized in
 * @param[in] funcID     BLAS function ID
 * @param[in] flags      Tile creation flags
 * @param[in] storType   Storage type
 *
 * If \b baseName field of a tile structure in the generator settings is zero,
 * it is initialized with the default value: "a" for the matrix A, "b" for
 * the matrix B, "x" for the vector X, "c" for the matrix C, and "y" for the
 * vector Y.
 *
 * As X and Y are column-vectors from the math point of view, tiles for them
 * are always packed irrespectively the TileCreationFlags::TILE_PACKED flag
 * is specified or not.
 *
 *
 * Transposition of C tile matches transposition of C matrix by default, until
 * the TILE_C_FORCE_NOTRANS flag is not set. If the flag is set, tile is
 * forced to be initialized as non-transposed and veclen must be verified.
 *
 */
void
initDefaultTiles(
    struct BlasGenSettings *gset,
    BlasFunctionID funcID,
    TileCreationFlags flags,
    PrivateStorageType storType);

/**
 * @internal
 * @brief Get entire number of vectors in the tile
 *
 * @param[in] tile          Tile to get number of vectors of
 */
unsigned int
tileVectorsNum(const Tile *tile);

/**
 * @internal
 * @brief Size of entire tile storage in elements
 *
 * @param[in] tile          Tile to get size of
 */
unsigned int
tileStorageSize(const Tile *tile);

/**
 * @brief Get length of tile line segment
 *
 * @param[in] Tile       Source tile
 *
 * Under that segment it is assumed such a part of line which doesn't cross over
 * vector bound and row/column bound depending on the tile is transposed or not.
 * In the other words, this is a piece of data which provides maximum possible
 * vectorization don't breaking correctness.
 */
unsigned int
tileLineSegmentLen(const Tile *tile);

/**
 * @internal
 * @brief Declare variables needed to store a tile
 *
 * @param[out] ctx        Generator context
 * @param[in] gset        Generator settings containing desctiptors of
 *                        tiles to declare storages for
 *
 * If a tile is fit into a single variable of the native type matching
 * to the tile's vector length, it is declared a single variable with the name
 * matching the \b baseName field being a part of the @ref Tile structure.
 * If not, the following rules are applied. If the tile is needed to be stored
 * in a private array, variable name matches the base name and array size
 * is sufficient to fit such a tile. If the tile is needed to be stored
 * in a set of variables which names are arranged as the base name followed
 * with an integer index starting from zero and incremented by one for each
 * subsequent variable.
 *
 * @return 0 on success, and -EOVERFLOW if the source buffer is overflowed
 *
 * @ingroup TILES
 */
int
declareTileStorages(struct KgenContext *ctx, const struct BlasGenSettings *gset);

/**
 * @internal
 * @brief Declare variable needed to store one tile
 *
 * @param[out] ctx        Generator context
 * @param[in] tile        Tile settings containing desctiptors of
 *                        a tile to declare storages for
 *
 * If a tile is fit into a single variable of the native type matching
 * to the tile's vector length, it is declared a single variable with the name
 * matching the \b baseName field being a part of the @ref Tile structure.
 * If not, the following rules are applied. If the tile is needed to be stored
 * in a private array, variable name matches the base name and array size
 * is sufficient to fit such a tile. If the tile is needed to be stored
 * in a set of variables which names are arranged as the base name followed
 * with an integer index starting from zero and incremented by one for each
 * subsequent variable.
 *
 * @return 0 on success, and -EOVERFLOW if the source buffer is overflowed
 *
 * @ingroup TILES
 */
int
declareOneTileStorage(struct KgenContext *ctx, const Tile *tile);

/**
 * @internal
 * @brief Sprintf element composed of one or several data elements
 *        stored in the tile
 *
 * @param[out] str          Kernel string object to store tile element
 *                          expression
 * @param[in] tile          Tile description structure
 * @param[in] row           Row of the starting element
 * @param[in] col           Element column
 * @param[in] len           Number of tile elements needed to be captured by
 *                          the expression
 *
 * \b row should be less than number of rows and \b col should be less than
 * number of columns in the tile. Traversal of a tile line is not allowed.
 * That means \b col plus \b len should be not greater than number of columns
 * if the tile is stored in direct form, and \b row plus \b len should be not
 * greater than number of rows if the tile is stored in transposed form.
 * If it is not hold true in debug mode, an assertion is triggered.
 * In the release may produce a wrong code which can be even not compilable.
 *
 * @ingroup TILES
 */
void
sprintfTileElement(
    Kstring *str,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    unsigned int len);

/**
 * @internal
 * @brief Sprintf half of a single complex data element stored in the tile
 *
 * @param[out] str          Kernel string object to store tile element
 *                          expression
 * @param[in] tile          Tile description structure
 * @param[in] row           Row of the starting element
 * @param[in] col           Element column
 * @param[in] half          Half type
 *
 * The restrictions for \b row and \b col are the same as for
 * sprintfTileElement(). This function is applicable only for tiles containing
 * complex data and must not be used in case of real data.
 *
 * @ingroup TILES
 */
void
sprintfTileElementHalf(
    Kstring *str,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    TileElementHalf half);

/**
 * @internal
 * @brief Sprintf element composed of one or several data elements
 *        stored in each of the tiles
 *
 * @param[out] kstrs        Kernel string objects array to store element
 *                          expression for each tile
 * @param[in] row           Vectorizable element row
 * @param[in] col           Vectorizable element column
 * @param[in] num           Number of tile description structure
 * @param[in] first         First tile description structure
 *
 * Decides how many vectored access in for each line of each tile will be and
 * does sprintfTileElement() for each of tiles. This function can have got any
 * value of \b row \b and \b col \b. \b kstrs \b and \b tile->baseName \b can
 * have NULL, then no sprintfTileElement() will be executed.
 *
 * @return 0 if no sprintf tiles, or number of vectors in one line
 *
 * @ingroup TILES
 */
int
forEachTile(Kstring *kstrs,
            unsigned int row,
            unsigned int col,
            unsigned int num,
            Tile *first,
            ...);

/**
 * @internal
 * @brief Generate assigning a tile element with zero
 *
 * @param[out] ctx      Generator context
 * @param[in] tile      Tile description structure
 * @param[in] row       Row of the starting element
 * @param[in] col       Element column
 * @param[in] len       Number of elements needed to be assigned with zero
 *
 * See decription of sprintfTileElement() for more details about restrictions
 * on \b row, \b col and \b len.
 *
 * @ingroup TILES
 */
void
genSetZeroInTile(
    struct KgenContext *ctx,
    const Tile *tile,
    unsigned int row,
    unsigned int col,
    unsigned int len);

/**
 * @internal
 * @brief Generate assigning a tile element with unit
 *
 * @internal
 * @brief Generate assigning a tile element with zero
 *
 * @param[out] ctx      Generator context
 * @param[in] tile      Tile description structure
 * @param[in] row       Row of the starting element
 * @param[in] col       Element column
 *
 * \b row should be less than number of rows and \b col should be less than
 * number of columns in the tile. If it is not hold true in debug mode,
 * an assertion is triggered. In the release may produce a wrong code which
 * can be even not compilable.
 *
 * @ingroup TILES
 */
void
genSetUnitInTile(
    struct KgenContext *ctx,
    const Tile *tile,
    unsigned int row,
    unsigned int col);

/**
 * @internal
 * @brief Generate zeroing an entire tile
 *
 * @param[out] ctx      Generator context
 * @param[in] tile      Tile description structure
 *
 * @ingroup TILES
 */
void
genZeroTile(struct KgenContext *ctx, const Tile *tile);

/**
 * @internal
 * @brief Generate copying between 2 tiles
 *
 * @param[out] ctx      Generator context
 * @param[in] dst       Destination tile
 * @param[in] src       Source tile
 *
 * @ingroup TILES
 */
void
genTileCopy(
    struct KgenContext *ctx,
    const Tile *dst,
    const Tile *src,
    TileCopyOps op);

#endif /* TILE_H_ */
