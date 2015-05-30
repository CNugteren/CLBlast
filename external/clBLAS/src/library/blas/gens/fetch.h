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


#ifndef FETCH_H_
#define FETCH_H_

/**
 * @internal
 * @defgroup FETCH_GEN Generating fetches from memory
 * @ingroup BLAS_GENERATORS
 */

/*@{*/

/**
 * @internal
 * @brief Context for the fetch generator
 */
struct FetchContext;

struct BlasGenSettings;
//enum TailStatus;

// FIXME: Deprecated. Throw later
struct TileMulOpts;

/**
 * @internal
 * @brief Optimization levels for the fetch generator with witch the caller
 *        can control some aspects of the code generation.
 *
 * !!NOTE: At expanding this list, the levels must be placed in ascending
 *         order of their importance.
 *
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef enum FetchOptLevel {
    /** Expand the fetch loop in the way providing a prefetch effect */
    FOPTLEV_PREFETCH = 0x01,
    /**
     * Can share temporary coordinates for A and B. Usable in case when
     * A and fetches are fired sequentially and hence in some cases can
     * share the same temporary coordinates. Must be set only if fetch
     * has been already fired for one of the tiles. Otherwise result is
     * undefined.
     */
    FOPTLEV_CAN_SHARE_TMP_AB = 0x02,
    /**
     * Reorder generated statements so as fethes would be groupped
     * all together
     */
    FOPTLEV_MERGE_FETCHES = 0x04,
    /** Enable using of temporary precomputed coordinates */
    FOPTLEV_TMP_COORD_PRECOMPUTING = 0x08,
    /** Enable using of persistent precomputed coordinates */
    FOPTLEV_PERS_COORD_PRECOMPUTING = 0x10
} FetchOptLevel;

/**
 * @internal
 * @brief Addressing modes for the fetch generator
 */
typedef enum FetchAddrMode {
    /**
     * Normal mode. Fetching is performed only with full vectors.
     * Physical coordinates in memory are absolute for the matrices and
     * evaluated only based on the logical coordinates along rows of the
     * matrix \b A, columns of the matrix \b B and coordinate along K
     */
    FETCH_ADDR_NORMAL = 0,
    /**
     * Pointer for the matrix A is set at start of the tile panel.
     * All resulting coordinates will be relative against this base.
     * KernelVarNames::CoordA the generator settings structure is not used
     */
    FETCH_ADDR_A_RELATIVE = 0x01,
    /**
     * Pointer for the matrix B is set at start of the tile panel.
     * All resulting coordinates will be relative against this base.
     * KernelVarNames::CoordB the generator settings structure is not used
     */
    FETCH_ADDR_B_RELATIVE = 0x02,
    /**
     * Pointers for A and B match the current coordinate along dimension K and
     * thus set at the beginning of the tile. All resulting coordinates will be
     * relative against the current value of the pointers.
     * KernelVarNames::CoordA, KernelVarNames::coordB and KernelVarNames
     * accessible via the generator settings structure are not used
     */
    FETCH_ADDR_K_RELATIVE = 0x04,
    /**
     * Cyclical addressing along rows of \b A. That means substracting
     * number of rows from the coordinate in case of exceeding it.
     */
    FETCH_ADDR_A_CYCLICAL = 0x08,
    /** Cyclical addressing along columns of B */
    FETCH_ADDR_B_CYCLICAL = 0x10,
    /** Cyclical addressing along K dimension */
    FETCH_ADDR_K_CYCLICAL = 0x20,
    /**
     * Perform padding of the trailing part along dimension K.
     * That allows perform a vectorized fetch of tail including a piece being
     * outside the size along K. It affects only if K expands along the leading
     * dimension
     */
    FETCH_ADDR_TAILK_PADD = 0x40,
    /*
     * Expand loop with stride equal to witdth of the top level block
     */
    FETCH_ADDR_BW_STRIDE = 0x80
} FetchAddrMode;


// FIXME: Deprecated and should be thrown away later
union FetchTmpVarName {
    const char *idx;
    const char *uptr;
};

/**
 * @internal
 * @brief Specific settings for the fetching generator
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef struct FetchOpts {
    MatrixRole mrole;
    CLMemType memA;             /**< type of memory matrix A is located on */
    CLMemType memB;             /**< type of memory matrix B is located on */
    unsigned int lineOffset;
    unsigned int linesNum;
    const char *regName;        // TODO: the field is deprecated. Remove it

    /*
     * FIXME: one more klugde for backward compatibility; get addressing
     *        mode from the options of tilemul
     */
    const struct TileMulOpts *mulOpts;

    // TODO: All the following fields are deprecated. Remove it
    union FetchTmpVarName tmpYvar;
    union FetchTmpVarName tmpXvar;
    const char *alvM;     /**< vecLen-aligned M in vectors */
    const char *alvN;     /**< vecLen-aligned N in vectors */
    const char *alvKA;    /**< vecLen-aligned K in vectors of A */
    const char *avlKB;    /**< vecLen-aligned K in vectors of B */
    const char *ax;       /**< matrix A x coordinate, in vectors */
    const char *ay;       /**< matrix A y coordinate */
    const char *bx;       /**< matrix B x coordinate, in vectors */
    const char *by;       /**< matrix B y coordinate */
    const char *ldav;     /**< matrix A leading dimension, in vectors */
    const char *ldbv;     /**< matrix B leading dimension, in vectors */
    const char *skewArow; /**< matrix A rows skew */
    const char *skewAcol; /**< matrix A columns skew, in vectors */
    const char *skewBrow; /**< matrix A rows skew */
    const char *skewBcol; /**< matrix A columns skew, in vectors */
} FetchOpts;


/**
 * @internal
 * @brief Create context for the fetch generator
 *
 * After creation there are enabled optimization levels relating
 * to precomputing with storing to temporary coordinates.
 * Addressing mode is set to ::FETCH_ADDR_NORMAL
 *
 * @return pointer to a new context object on success, NULL otherwise
 */
struct FetchContext
*createFetchContext(void);

/**
 * @internal
 * @brief Destroy fetch generator context
 *
 * @param[out] fctx            Fetch generator context to destroy
 */
void
destroyFetchContext(struct FetchContext *fctx);

/**
 * @internal
 * @brief Get current fetch optimization levels
 *
 * @param[in] fctx              Fetch context
 */
FetchOptLevel
getFetchOptLevels(struct FetchContext *fctx);

/**
 * @internal
 * @brief Enable needed code optimization levels the fetch generator
 *
 * @param[out] ctx              Generator context
 * @param[in] opts              Fetch Options
 */
void
enableFetchOptLevels(struct FetchContext *fctx, FetchOptLevel levels);

/**
 * @internal
 * @brief Disable unneeded code optimization levels for the fetch generator
 *
 * @param[out] ctx              Generator context
 * @param[in] opts              Fetch Options
 */
void
disableFetchOptLevels(struct FetchContext *fctx, FetchOptLevel levels);

/**
 * @internal
 * @brief Get current addressing mode used by the fetch generator
 *
 * @param[in] fctx              Fetch context
 */
FetchAddrMode
getFetchAddrMode(const struct FetchContext *fctx);

/**
 * @internal
 * @brief Set addressing mode for the fetch generator
 *
 * @param[out] fctx             Fetch context
 * @param[in]  mode             Addressing mode to set
 */
void
setFetchAddrMode(struct FetchContext *fctx, FetchAddrMode mode);

/**
 * @internal
 * @brief Set default fetch addressing mode based on the problem specifics
 *
 * @param[out] fctx             Fetch context
 * @param[in]  gset             Generator settings
 * @param[in]  mask             Addressing mode mask
 * @param[in]  tailStatus       Tails handling status
 * @param[in]  processTailK     Flag showing if the tail part along the
 *                              dimension K is picked up or not.
 *
 * Primarily, the function checks if there are tails along rows of A,
 * columns of B, dimension K and if some tails are raised or not.
 * Based on this info and also taking into account fetch vector length,
 * it set appropriate addressing mode to don't exceed matrix bounds during
 * the fetch operations. If there are not "small" tails for rows of A and
 * columns of B is selects relative addressing for them. If there are not
 * "small" tails along K, it selects relative addressing for this dimension
 * as well.
 *
 * The addressing mode mask passed via the \b mask parameter is used to
 * not set addressing modes not suitable for callers. Resulting addressing
 * mode which is set is presented as bitwise AND of  a default value selected
 * by the function and bitwise negated value of the mask
 *
 * \b tailStatus is a bit mask of values consisting the #TailStatus enumeration.
 *
 * @return Addressing mode the function set during the last call.
 */
FetchAddrMode
setDefaultFetchAddrMode(
    struct FetchContext *fctx,
    const struct BlasGenSettings *gset,
    FetchAddrMode mask,
    int tailStatus,
    bool processTailK);

/**
 * @internal
 * @brief Prepare the fetch generator to generate efficient fetches
 *        within the K loop
 *
 * @param[out] genCtx           Generator context
 * @param[out] fetchCtx         Fetch context
 * @param[in] gset              Generator settings
 * @param[in] memA              Type of memory the matrix A is stored in
 * @param[in] memB              Type of memory the matrix B is stored in
 *
 * Basically, the function lets to declare all needed for work of the fetch
 * generator. If a user lots upon efficient fetching within the tilemul loop,
 * he should call the function before generating that loop.
 * If it is not invoked, the fetch generator produces a code in some default
 * way which may be far from efficient. The stuff prepared with the function is
 * valid only for one fetch call. If the user needs to use the same once again,
 * it may use revalidateFetchContext().
 */
int
prepareFetchLoop(
    struct KgenContext *genCtx,
    struct FetchContext *fetchCtx,
    const struct BlasGenSettings *gset,
    CLMemType memA,
    CLMemType memB);

/**
 * @internal
 * @brief Revalidate fetch context
 *
 * @param[out] fctx             Fetch context
 * @param[in]  mrole            Matrix to revalidate the context for
 *
 * Enable the fetch generator to use the stuff produces with the last call
 * of prepareFetch() once again.
 */
void
revalidateFetchContext(struct FetchContext *fctx, MatrixRole mrole);

/**
 * @internal
 * @brief Tile fetching generator
 *
 * @param[out] genCtx         Generator context
 * @param[in]  fetchCtx       FetchContext
 * @param[in]  gset           Generator settings
 * @param[in]  fetchOpts      Fetch-specific generator options
 *
 * This function generates code which fetches tile a or b from global or local
 * memory into private memory.\n
 * Generated code fetches tiles by vectors using coordinate values in vectors
 * from @ref FetchOpts.
 * Complex types and conjugated tiles are supported. Global cycling is supported
 * for global memory fetching - this mean that if tile overlaps matrix
 * the tail of tile will be fetched from the beginning instead of accessing
 * memory outside the matrix.\n
 * Second level of subdimensions is used for tile sizes.\n
 * Tile can be fetched from global memory or from local memory.
 * If tile is fetched from local memory then leading dimensions for local
 * memory area are taken from first level subdimensions.\n
 * Post-fetch callback generator function can be called after fetching tile
 * for zeroing tails or setting diagonal elements to one. This function is
 * provided by caller in @ref TileMulOpts.postFetch.\n
 * After the function completes its work it invalidates the fetch context, and
 * all the stuff that has been prepared before, will not be used in the next
 * fetch transaction.
 *
 * @return 0 on success
 * @return -EOVERFLOW on source buffer overflowing
 */
int
genFetchInputTile(
    struct KgenContext *genCtx,
    struct FetchContext *fetchCtx,
    const struct BlasGenSettings *gset,
    const FetchOpts *fetchOpts);

/**
 * @internal
 * @brief Fetch input tile
 *
 * @param[out] batch                    Statement batch
 * @param[in]  gset                     Generator settings
 * @param[in]  fetchOpts                Fetch Options
 *
 * The function has the same effect and semantics as the previous one,
 * but put the code to the intermediate statement batch rather than a target
 * generator context.
 */
void
genFetchInputTileBatch(
    struct StatementBatch *batch,
    struct FetchContext *fctx,
    const struct BlasGenSettings *gset,
    const FetchOpts *fetchOpts);

/*@}*/

#endif /* FETCH_H_ */
