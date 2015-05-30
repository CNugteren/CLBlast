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
 * Something specific for BLAS generators
 *
 * NOTE:
 *      1) All the blas kernel generators should
 *         perceive fields of the SubproblemDim
 *         structure as following:
 *         'y' - rows of matrix A, i. e. M dimension
 *               of matrix C
 *         'x' - columns of matrix B and C
 *         'bwidth' - block width in K dimension
 *
 *      2) At generating copying functions and their calls one should
 *         keep in mind, all the matrix blocks are copied in
 *         the local memory such that sequentially accessed elements
 *         are located in memory sequentially. In this context
 *         transposing is perceived as transposing at copying
 *         to/from the local memory, not matrix storage way in
 *         the array passed to kernel.
 */

#ifndef BLAS_KGEN_H_
#define BLAS_KGEN_H_

#include <clBLAS.h>

#include <cltypes.h>
#include <kerngen.h>
#include <mempat.h>
#include <dblock_kgen.h>

#include <blas_funcs.h>
#include <matrix_props.h>

#include "tile.h"
#include "fetch.h"


#define BLAS_KGEN_FORMAT 1

#define genInternalLoopEnd(ctx) kgenEndBranch(ctx, NULL)

enum {
    MAX_OPENCL_VECTOR_LENGTH = 16
};

typedef enum TailFetch {
    FETCH_NO_TAILS = 0,
    FETCH_TAIL_ROW = 0x01,
    FETCH_TAIL_COL = 0x02
} TailFetch;

/**
 * @internal
 * @brief Blas generator flags
 * @ingroup GEN_SETTINGS
 */
typedef enum BlasGenFlags {
    BGF_EXPLICIT_INLINE = 0x01,
    BGF_DISTINCT_VECLEN = 0x02,
    // TODO: replace with a flags with inverse semantics
    BGF_WHOLE_A = 0x04,
    /** Leading dimension are in vectors rather than in elements */
    BGF_LD_IN_VECTORS = 0x08,
    /**
     * Objects in the global memory are accessed through the unified pointers.
     * This feature is deprecated and should be not used in new generators.
     * It is left for backward compatibility
     */
    BGF_UPTRS = 0x10
} BlasGenFlags;

/**
 * @internal
 * @brief Flags showing how problem tails are handled
 * @ingroup TAILS_HANDLING
 */
typedef enum TailStatus {
    /** Tail of the matrix A is raised */
    TAIL_A_RAISED = 0x01,
    /** Tail of the matrix B is raised */
    TAIL_B_RAISED = 0x02
} TailStatus;

/**
 * @internal
 * @brief Tiles multiplier flags
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef enum TileMulFlags {
    TILEMUL_NO_FLAGS = 0,              /**< No flags */
    TILEMUL_TRA = 0x01,                /**< Transposed matrix A */
    TILEMUL_TRB = 0x02,                /**< Transposed matrix B */
    TILEMUL_CONJA = 0x04,              /**< Conjugated elements of A */
    TILEMUL_CONJB = 0x08,              /**< Conjugated elements of B */
    TILEMUL_C_COLUMN_MAJOR = 0x10,     /**< Column major block for matrix C */
    TILEMUL_NOT_FETCH_B = 0x20,        /**< Do not fetch matrix B block */
    TILEMUL_EXTERN_RDECL = 0x40,       /**< External register tiles declaration,
                                          the generator must not declare them
                                          itself */

    /**
     * Deprecated. Use the repsective mode being a part of FetchAddr mode.
     * He is left just for backward compatibility to don't break the working
     * code and will be removed soon
     */
    TILEMUL_WRAP_AROUND_TAIL = 0x80,   /**< Sizes used for column skew are
                                            rounded to next vecLen bound */
    /** Use global cyclic along subproblem A coordinate.
     * Deprecated. Don't use it */
    TILEMUL_GLOBAL_CYCLIC_A = 0x100,
    /** Use global cyclic along subproblem B coordinate.
     * Deprecated don't use it */
    TILEMUL_GLOBAL_CYCLIC_B = 0x200,
    /* Deprecated. Don't use it */
    TILEMUL_GLOBAL_CYCLIC_K = 0x400,   /**< Use global cyclic along K */
    /** Use skew along subproblem A coordinate */
    TILEMUL_SKEW_A = 0x800,
    /** Use skew along subproblem B coordinate. Deprecated */
    TILEMUL_SKEW_B = 0x1000,
    /* Deprecated */
    TILEMUL_SKEW_K = 0x2000,           /**< Use skew along K */
    /** Use size of whole matrix for cyclic addressing. Deprecated */
    TILEMUL_GLOBAL_CYCLIC = TILEMUL_GLOBAL_CYCLIC_A |
                            TILEMUL_GLOBAL_CYCLIC_B |
                            TILEMUL_GLOBAL_CYCLIC_K,
    // Deprecated
    TILEMUL_SKEWS = TILEMUL_SKEW_A | TILEMUL_SKEW_B | TILEMUL_SKEW_K,
    /** Optimize coordinates calculations by storing coordinates values */
    // Deprecated
    TILEMUL_OPTIMIZE_COORD_CALC = 0x4000,
    /** Use bwidth0 stride */
    TILEMUL_BW_STRIDE = 0x8000,
    /** Optimize coordinates calculations by using vectors
     *  and pointer increments */
    // Deprecated
    TILEMUL_OPTIMIZE_VEC_COORDS = 0x10000,
    /** Do not increment K*/
    TILEMUL_NOT_INC_K = 0x20000,
    /**
     * Use variants with explicit vectorization. Useful on platforms with
     * true SIMD.
     */
    TILEMUL_FORCE_VECTORIZATION = 0x40000
} TileMulFlags;


/**
 * @internal
 * @brief Tiles multiplier core
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef enum TileMulCore {
    /** Use multiplication and addition operations */
    TILEMUL_MULADD,
    /** Use the 'dot' function where possible */
    TILEMUL_DOT,
    /** Use the 'mad' function */
    TILEMUL_MAD
} TileMulCore;

/**
 * @internal
 * @brief Update result operations
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef enum UpdateResultOp {
    /** Just set the values stored in a target buffer */
    UPRES_SET,
    /** Summarize values stored in a target buffer with the temporary result */
    UPRES_SUM
} UpdateResultOp;

/**
 * @internal
 * @brief Update result generator flags
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef enum UpdateResultFlags {
    /** Resulting matrix is stored in the column major form */
    UPRES_COLUMN_MAJOR = 0x01,
    /** Generic version, non optimal sizes */
    UPRES_GENERIC = 0x02,
    /** Multiply result on beta */
    UPRES_WITH_BETA = 0x04,
    /** do not multiply on the alpha scalar */
    UPRES_WITHOUT_ALPHA = 0x08,
    /**
     * Destination is private memory;
     * if not set destination is in the global one
     */
    UPRES_PRIV_DEST = 0x10,
    /** Use the local memory instead the global memory */
    UPRES_USE_LDS = 0x20,
    /** Generate the inline version */
    UPRES_INLINE = 0x40,
    /** Disable vectorization at memory access */
    UPRES_NO_VECTORIZATION = 0x80,
    /** For the generic version useful data reside at the tile rows' tail */
    UPRES_TAIL_ROW = 0x100,
    /** For the generic version useful data reside at the tile columns' tail */
    UPRES_TAIL_COL = 0x200,
    /** Generate condition whether coordinates don't exceed problem bounds */
    UPRES_EXCEED_PROBLEM_CONDITION = 0x400,
    /****/
    UPRES_INDEXING_WITH_CONSTANTS = 0x800,
    /** Write result to C instead of B for functions with triangular matrix */
    UPRES_TRIANG_WRITE_C = 0x1000
} UpdateResultFlags;

typedef struct PrivateArea {
    const char *typeName;
    unsigned int vecLen;
    unsigned int size;
} PrivateArea;

/**
 * @internal
 * @defgroup GEN_SETTINGS Generator settings
 * @ingroup BLAS_GENERATORS
 */
/*@{*/

/**
 * @internal
 * @brief Kernel variable and argument names
 */
typedef struct KernelVarNames {
    const char *A;          /**< Matrix A variable name */
    const char *B;          /**< Matrix B variable name */
    const char *C;
    const char *LDS;		/**< LDS pointer name */
    const char *coordA;     /**< Variable for subproblem A coordinate */
    const char *coordB;     /**< Variable for subproblem B coordinate */
    const char *k;          /**< Variable for incrementable K offset value*/
    const char *skewA;      /**< Variable for skews along A */
    const char *skewB;      /**< Variable for skews along B */
    const char *skewK;      /**< Variable for skews along K */
    const char *sizeM;      /**< Matrix A size M */
    const char *sizeN;      /**< Matrix B size N */
    const char *sizeK;      /**< Matrixes size K */
    const char *lda;        /**< Leading dimension of matrix A */
    const char *ldb;        /**< Leading dimension of matrix B */
    const char *ldc;        /**< Leading dimension of matrix C, in vectors */
    const char *vectCoordA; /**< Vector containing indexes of tile a elements
                                 in matrix A */
    const char *vectCoordB; /**< Vector containing indexes of tile b elements
                                 in matrix B*/
    const char *startM;
    const char *startN;
    const char *startK;
    const char *alpha;
    const char *beta;
} KernelVarNames;

/**
 * @internal
 * @brief Blas generator settings
 *
 * This structure is designed to be used with most of subgenerators
 * and generator helpers. It is assumed to be initialized once at the
 * generator beginning and modified as few as possible over the rest of
 * the process.
 */
typedef struct BlasGenSettings {
    /**
     * Subproblem dimensions:
     *
     * work group dimensions are at index 0
     * work item dimensions are at index 1
     */
    SubproblemDim subdims[2];
    const PGranularity *pgran;      /**< Data parallelism granularity */
    const CLBLASKernExtra *kextra;  /**< Kernel extra */
    BlasGenFlags flags;             /**< Global generator flags */
    KernelVarNames varNames;        /**< Kernel variables and argument names */
    Tile tileA;
    Tile tileBX;
    Tile tileCY;
} BlasGenSettings;

/*@}*/

/**
 * @internal
 * @brief Variable names for the inline version of a function updating result
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef struct UpresVarNames {
    const char *result;     /**< Name of an output matrix */
    /** Leading dimension of a matrix stored in the global memory */
    const char *ld;
    const char *startRow;   /**< Start row to update from */
    const char *startCol;   /**< Start column to update from */
    const char *nrRows;     /**< Number of rows */
    const char *nrCols;     /**< Number of columns */
    const char *cachedName; /**< Name of lds chached values */
} UpresVarNames;

/**
 * @internal
 * @brief Options for matrix tiles multiplication generator
 * @ingroup BLAS_MAJOR_SUBGENS
 */
typedef struct TileMulOpts {
    CLMemType memA;             /**< type of memory matrix A is located on */
    CLMemType memB;             /**< type of memory matrix B is located on */
    TileMulFlags flags;         /**< Flags on objects and computing specifics */
    TileMulCore core;           /**< Multiply and add core */
    int (*postFetch)(
        struct KgenContext *ctx,
        MatrixRole mrole,
        void *arg);             /**< Tile post fetch callback */
    void *postFetchPriv;        /**< Postfetch callback's private date */
    struct FetchContext *fctx;
} TileMulOpts;

typedef struct ZeroFuncs {
    char names[MATRIX_ROLES_NUMBER][FUNC_NAME_MAXLEN];
} ZeroFuncs;

/**
 * @internal
 * @brief Private data for fetch postprocessing callback
 * @ingroup TAILS_HANDLING
 */
typedef struct TilePostFetchPrivate {
    BlasFunctionID funcID;
    const BlasGenSettings *gset;
    const char *regName;
    int fetchNumA;
    int wholeA;
} TilePostFetchPrivate;

void
getPrivateAreaInfo(
    const BlasGenSettings *gset,
    BlasFunctionID funcID,
    MatrixRole mrole,
    PrivateArea *area);

void
declarePrivateArea(
    struct KgenContext *ctx,
    const PrivateArea *area,
    const char *baseName,
    PrivateStorageType storType);

/*
 * Declare separately the real and imaginary part of
 * a complex multiplier.
 *
 * @ctx: generator context
 * @baseName: variable's base name matching to an existing variable
 *            with not sepated parts
 * @typeName: variable type name
 *
 * Rule naming
 *      real part:      <baseName>R
 *      imaginary part: <baseName>I
 *
 * On success returns 0, and -EOVERFLOW at source buffer
 * overflowing
 */
int
declareComplexMultParts(
    struct KgenContext *ctx,
    const char *baseName,
    const char *typeName);

/**
 * @internal
 * @defgroup CHECK_DECOMP_CACL_GRAN  Checking decomposition and calculate
 *                                   parallelism granularity
 * @ingroup BLAS_GENERATORS
 */

/*@{*/

/**
 * @brief Sanity check for decomposition
 *
 * @param[in] subdims           Subproblem dimensions. 2 levels.
 * @param[in] minSize           Minimum size for any of the dimension
 *                              components
 * @param[in] maxSize           Maxium size which can't be exceeded by
 *                              any of the dimension components at the tile
 *                              layer
 * @param[in] maxRegs           Maximum registers it's allowed to use
 * @param[in] dtype             BLAS data type
 * @param[in] wholeA            Is matrix A stored in registers entirely or
 *                              partially
 *
 * The function rejects only decompositions that are completely invalid or lead
 * to consumption of too many registers or just have component values at the
 * tile layer that are out of the range [\b MinSize, \b MaxSize].
 * Completely invalid decompositions are those which don't allow to divide
 * problem integrally among work items, e. g. zeroed components are wrong,
 * the step components (x, y, bwidth) of the 0-th level not integrally
 * divisible on respective size components (itemX, itemY, bwidth) of the 1-st
 * level are wrong as well. The decomposition is also wrong if the size
 * components are not integrally divisible on the step components and not equal
 * to #SUBDIM_UNUSED.
 *
 * @return true if the decomposition is valid, or false otherwise
 */
bool
decompSanityCheck(
    const SubproblemDim *subdims,
    unsigned int minSize,
    unsigned int maxSize,
    unsigned int maxRegs,
    DataType dtype,
    bool wholeA);

/**
 * @brief Calculate granularity in case when a work item is responsible
 *        for its own part of solution not overlapping with those of other
 *        items
 *
 * @param[out] pgran            Location to store calculated granularity
 * @pararm[in] subdims          Subproblem dimensions
 * @param[in] xdim              Dimension in the OpenCL work space X component
 *                              of decomposition is mapped on
 * @param[in] level             Function BLAS level. Reserved for future use.
 *
 * If value of \b xdim is -1, then the function assumes that OpenCL work
 * space is single dimensional, and puts the product of granularity against
 * X and Y component to 0-th element of \b wgSize field. If its value is
 * 0 or 1, the function assumes that OpenCL work space is 2D and puts
 * granularity against X component to \b xdim element of \b wgSize field
 * of the granularity decriptor. Granularity against Y component is put to
 * 1 - \b xdim element. Other values are invalid and forces abort in debug
 * build. The function initializes the \b wgDim field properly.
 *
 * NOTE: Now, only this function is supported only for level 3 and
 *       must not be called for level 2
 */
void
calcPgranDedicated(
    PGranularity *pgran,
    const SubproblemDim *subdims,
    int xdim,
    int level);

/**
 * @brief Calculate granularity in case when several items evaluate the same
 *        part of solution together
 *
 * @param[out] pgran            Location to store calculated granularity
 * @pararm[in] subdims          Subproblem dimensions
 * @param[in] xdim              Dimension in the OpenCL work space X component
 *                              of decomposition is mapped on
 * @param[in] ydim              Dimension in the OpenCL work space Y component
 *                              of decomposition is mapped on
 * @param[in] level             Function BLAS level. Reserved for future use
 *
 * If \b xdim and \b ydim values are equal, then the function puts the product
 * of granularity against X and Y component to \b xdim element of \b wgSize
 * field. If not, it puts separated granularity for X and Y in \b xdim and
 * \b ydim element respectively. Both the values must be non negative and less
 * than 3 (since OpenCL workspace cannot have more than 3 dimensions).
 * If some of these parameters is zero, then the other one must be zero as well.
 * If some of these parameters is 2, then the other one must be 1. These
 * restrictions are caused by needs in reflecting \b bwidth in granularity
 * in case of multidimensional decomposition. For 2D and 3D decompositions
 * granularity for bwidth is calculated as well, and it is always mapped
 * onto 0-th workspace dimension. If some of these parameters are wrong,
 * it forces abort in debug build. The function sets the \b wgDim field
 * to maximum of xdim and ydim plus 1.
 *
 * NOTE: Now, only this function is supported only for level 3 and
 *       must not be called for level 2
 */
void
calcPgranCooperative(
    PGranularity *pgran,
    const SubproblemDim *subdims,
    int xdim,
    int ydim,
    int level);

/*@}*/

/**
 * @internal
 * @defgroup COMMON_MATH_OPERATIONS Constructing useful math expression
 * @ingroup BLAS_GENERATORS
 */
/*@{*/

/**
 * @brief Sprintf a complex MAD operation
 *
 * Operations:
 *     - \f$ dst \leftarrow a * b + c \f$
 *     - \f$ dst \leftarrow conj(a) * b + c \f$
 *     - \f$ dst \leftarrow a * conj(b) + c \f$
 *     - \f$ dst \leftarrow conj(a) * conj(b) + c \f$
 *
 *  @param[out] expr            String object to hold the target expression
 *  @param[in] dst              Destination argument
 *  @param[in] a                The first multiplier
 *  @param[in] b                The second multiplier
 *  @param[in] c                Added argument
 *  @param[in] isDouble         If set, the arguments have double precision
 *  @param[in] isConjA          If set, the argument A should be conjugated
 *  @param[in] isConjB          If set, the argument B should be conjugated
 *  @param[in] TileMulCore      Multiplying core
 *
 *  The \b c argument can be NULL. In this case it is ignored, and the function
 *  produces pure multiplication
 */
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
    TileMulCore core);

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
    TileMulCore core);

/**
 * @brief Sprintf expression of fast scalar mad
 *
 * @param[out] expr         Output expression
 * @param[in]  first        First multiplier
 * @param[in]  second       Second multiplier
 * @param[in]  scale        Scale of the second argument, i. e. it's divider.
 *                          Ignored if zero.
 * @param[in]  third        Added argument. Ignored if NULL.
 *
 * It can use mad24. So, expected result should not exceed 2^24
 */
void
sprintfFastScalarMad(
    Kstring *expr,
    const Kstring *first,
    const Kstring *second,
    unsigned int scale,
    const Kstring *third);

/*@}*/

/**
 * @internal
 * @defgroup BLAS_GEN_MISC_FUNCTIONS Miscellaneous functions
 * @ingroup BLAS_GENERATORS
 */

/*@{*/

/**
 * @brief Default function prefix for the data type
 *
 * @param[in] dtype     One of the data types supported by the library
 */
char
dtypeToBlasPrefix(DataType dtype);

/**
 * @brief Convert kernel extra flags to tilemul flags
 *
 * @param[in] funcID        BLAS function ID
 * @param[in] kflags        Kernel flags
 */
TileMulFlags
kextraToTilemulFlags(BlasFunctionID funcID, KernelExtraFlags kflags);

/**
 * @brief Get vector length elements should be fetched from (stored to)
 *        the global memory
 *
 * @param[in] gset          Generator settings
 * @param[in] funcID        BLAS function ID (deprecated)
 * @param[in] mrole         Role of the matrix to get vectorization for
 */
unsigned int
getVecLen(const BlasGenSettings *gset, BlasFunctionID funcID, MatrixRole mrole);

/**
 * @brief Sprintf chunk (set of components) of an OpenCL vector type
 *
 * @param[out] chunk        Buffer to sprintf to
 * @param[in] vecLen        Entire vector length
 * @param[in] clen          Length of the chunk
 * @param[in] vecOff        Starting component offset
 */
void
sprintfVecChunk(
    char *chunk,
    unsigned int vecLen,
    unsigned int clen,
    unsigned int vecOff);

/**
 * @brief Generate code containing scaling of leading dimensions on
 *        vector size
 *
 * @param[out] ctx          Generator context
 * @param[in] gset          Generator settings
 *
 * The function first checks whether the scaling is actually needed.
 * If vector size is 1. If some of the kernel variables for 'lda', 'ldb'
 * or 'ldc' is NULL, the function skips code generation for the dimension.
 * Calling this function has no effect if the @ref BGF_LD_IN_VECTORS generator
 * flag is not set. If some of the leading dimensions are not unique, only
 * one of the instances is scaled. Originality of the dimensions is detected
 * by values of the respective pointers being a part of @ref KernelVarNames.
 * For example, 'lda' and 'ldb' pointers are the same, only 'lda' is scaled.
 */
void
genScaleLeadingDimensions(struct KgenContext *ctx, const BlasGenSettings *gset);

/*@}*/

/**
 * @internal
 * @brief Generate default post processing logic after tile fetch
 *
 * @param[out] ctx      Generator context
 * @param[in] mrole     Matrix role
 * @priv[out]           Handler's private data
 *
 * @ingroup TAILS_HANDLING
 */
int
defaultTilePostFetch(
    struct KgenContext *ctx,
    MatrixRole mrole,
    void *priv);

void
getResultGPRsInfo(
    DataType dtype,
    const SubproblemDim *dims,
    unsigned int vecLen,
    unsigned int *nrRegs,
    const char **typeName);

/**
 * @internal
 * @defgroup BLAS_MAJOR_SUBGENS Major subgenerators
 * @ingroup BLAS_GENERATORS
 */
/*@{*/

/**
 * @internal
 * @brief Tiles fetching and multiplication inlined code generator
 *
 * @param[out] ctx          Generator context
 * @param[in] gset          Generator settings
 * @param[in] mulOpts       TileMul-specific generator settings
 *
 * This function generates code which fetches tiles a and b from global or local
 * memory into private memory, multiply them storing result into tile c in
 * private memory and increment coordinate k. Caller is responsible for loop
 * along K.\n
 * All combinations of tiles a and b orientations are supported. Generated
 * code fetches tiles by vectors which size can be different for tiles a and b.
 * Complex types and conjugated tiles are supported. Global cycling is supported
 * for global memory fetching - this mean that if tile overlaps matrix
 * the tail of tile will be fetched from the beginning instead of accessing
 * memory outside the matrix.\n
 * Second level of subdimensions is used for tiles sizes.\n
 * Generated code will fetch tiles a, b, multiply them and add result to tile c
 * in private memory, then increment k. By default, k is incremented by
 * second level bwidth but it is incremented by first level bwidth if
 * @ref TILEMUL_BW_STRIDE flag is set. It is used if whole work group goes
 * along K loop.\n
 * Each tile can be fetched from global memory or from local memory.
 * If tile is fetched from local memory then leading dimensions for local
 * memory area are taken from first level subdimensions.\n
 * Post-fetch callback generator function can be called after fetching tiles
 * for zeroing tails or setting diagonal elements to one. This function is
 * provided by caller.\n
 * If second level bwidth is not equal to first level bwidth, and
 * @ref TILEMUL_BW_STRIDE flag is not set then TileMul generates
 * loop from zero to first level bwidth with second level bwidth step. The
 * most common case is second level bwidth equal to first level bwidth where
 * single iteration of multiplication is generated.\n
 *
 * If the caller assume for efficient fetching from the global memory and the
 * tilemul logic is generated within a loop, prepareFetchCycle() should be
 * called before generation of the loop.
 *
 * @return 0 on success
 * @return -EOVERFLOW on source buffer overflowing
 * @return -EINVAL if input arguments are invalid
 */
int
tileMulGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileMulOpts *mulOpts);

/**
 * @internal
 * @brief Tiles pure multiplication code generator
 *
 * @param[out] ctx          Generator context
 * @param[in] gset          Generator settings
 * @param[in] mulOpts       TileMul-specific generator settings
 *
 * This function multiply two tiles, a and b, storing result in tile c. No
 * additional operations are made. It just performs tiles multiplication without
 * fetching, post-fetch processing and incrementing coordinates which can be
 * made by caller.
 *
 * @return 0 on success
 * @return -EOVERFLOW on source buffer overflowing
 */
int
genMulTiles(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const TileMulOpts *mulOpts);

/**
 * @internal
 * @brief Update result generator
 *
 * @param[out] ctx          Generator context
 * @param[in] gset          Generator settings
 * @param[in] op            Update operation
 * @param[in] flags         Update result flags
 * @argNames
 *
 * It generates a function applying an operation to the temporary result
 * stored in the private memory and updating the target result.
 *\n
 * The code can be generated as well in the form of callable function
 * as in the inlined form.
 *\n
 * List of taken argument differs depending on specified flags. In general,
 * these functions are defined as: \n
 * @code
 * void
 * funcName(
 *     <input type> C,
 *     <output type> *c,
 *     <input type> alpha,
 *     size_t startRow,
 *     size_t startCol,
 *     size_t ld
 *     [,<input type> beta]
 *     [,size_t nrRows]
 *     [,size_t nrCols])
 * @endcode
 *
 * @return 0 on success, -EOVERFLOW at source buffer overflowing.
 */
int
updateResultGen(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    BlasFunctionID funcId,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const UpresVarNames *uvarNames);

/**
 * @internal
 * @brief Produce a code updating a single result element
 *
 * @param[out] ctx      Generator context
 * @param[in] dst       Destination element expression
 * @param[in] src       Source element expression
 * @param[in] gset      Generator settings
 * @param[in] op        Update operation
 * @param[in] flags     Flags showing specifics of the code needed to be
 *                      generated
 *
 * @return 0 on success, -EOVERFLOW if the source buffer is exceeded.
 */
int
genUpdateResultSingle(
    struct KgenContext *ctx,
    const char *dst,
    const char *src,
    const BlasGenSettings *gset,
    UpdateResultOp op,
    UpdateResultFlags flags);

/*@}*/

TailFetch
checkForTailFetches(
    BlasFunctionID funcID,
    const SubproblemDim *dim,
    const CLBLASKernExtra *kextra,
    MatrixRole mrole,
    bool distVect,
    bool lowerTails);

bool
isNeedZeroTileTail(
    BlasFunctionID funcID,
    const SubproblemDim *dim,
    const CLBLASKernExtra *kextra,
    MatrixRole mrole,
    bool distVect);

/**
 * @internal
 * @brief Generate tail coordinates adjustment if needed
 *
 * @param[out] ctx              Generator context
 * @param[in] funcID            BLAS function ID
 * @param[in] gset              Generator settings
 * @param[out] *error           Location to store error.
 *                              Ignored if NULL.
 *
 * Adjust coordinates if work is distributed over matrix rows so as
 * a tile would not exceed the matrix bound. Cyclic addressing is not
 * applicable for that since skew over rows can be used for performance goals.
 *
 * If it's needed, issues an expression like
 *
 * if (coord.y + dy > M) {
 *     coord.y -= dy - M % dy;
 * }
 *
 * Return status showing if the tails have been actually adjusted or not.
 * If \b ctx is NULL the function doesn't try to generate a code, but just
 * return actual tail handling status
 *
 * @ingroup TAILS_HANDLING
 */
TailStatus
checkGenAdjustTailCoords(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    int *error);

/**
 * @internal
 * @brief Generate restoring original coordinates if needed
 *
 * @param[out] ctx              Generator context
 * @param[in] gset              Generator settings
 * @param[in] status            Tails handling status
 *
 * Coordinates restoring is needed to have ability to write back result to
 * a correct location.
 *
 * If it's needed, issues an expression like
 *
 * if (coord.y + dy == M) {
 *     coord.y += dy - M % dy;
 * }
 *
 * @ingroup TAILS_HANDLING
 */
int
checkGenRestoreTailCoords(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    TailStatus status);

/**
 * @internal
 * @brief Convert tail handling status to the respective flags
 *        of the update result generator
 *
 * @param[in] status            Status of the handling to convert to
 *                              the update result flags
 *
 * @ingroup TAILS_HANDLING
 */
UpdateResultFlags
tailStatusToUpresFlags(TailStatus status);



#endif /* BLAS_KGEN_H_ */
