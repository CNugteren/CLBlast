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


#ifndef BLAS_KGEN_LEGACY_H_
#define BLAS_KGEN_LEGACY_H_

#include "../blas_kgen.h"

/**
 * @internal
 * @brief Block multiplier flags
 * @ingroup BLAS_MAJOR_GENS
 */
typedef enum BlkmulFlags {
    BLKMUL_NO_FLAGS,            /**< No flags */
    BLKMUL_TRANSPOSE = 0x01,    /**< Transpose result */
    BLKMUL_IMAGE_PACKED = 0x02, /**< Data in image are packed */
    /**
     * Accumulate multiplication results to a
     * private location provided by caller
     */
    BLKMUL_OUTPUT_PRIVATE = 0x04,
    BLKMUL_SKEW_ROW = 0x08,     /**< Use skew over block rows */
    BLKMUL_SKEW_COLUMN = 0x10,  /**< Use skew over block columns */
    BLKMUL_INLINE = 0x20,       /**< Generate an inline version */
    BLKMUL_TRANSPOSED_B = 0x40, /**< Block B is transposed */
    /** Don't use "&" operation in cyclic address evaluation, use always "%" */
    BLKMUL_AVOID_AND = 0x80
} BlkMulFlags;

/**
 * @internal
 * @brief Block multiplier core
 * @ingroup BLAS_MAJOR_GENS
 */
typedef enum BlkmulCore {
    /** Use separate multiplication and summation implemented by hand */
    BLKMUL_SEPARATE_MULADD,
    /** Use the 'dot' function */
    BLKMUL_DOT,
    /** Use the 'mad' function */
    BLKMUL_MAD
} BlkmulCore;

/**
 * @internal
 * @brief Argument names for the inline version of the block
 *        multiplier
 * @ingroup BLAS_MAJOR_GENS
 */
typedef struct BlkmulArgNames {
    const char *coordA;     /**< Matrix A start coordinates */
    const char *coordB;     /**< Matrix B start coordinates */
    const char *skewRow;    /**< Skew over rows */
    const char *skewCol;    /**< Skew over columns */
    const char *k;          /**< Counter name in the loop over K */
    const char *vectBoundK; /**< Bound in the loop over K */
} BlkmulArgNames;

/**
 * @internal
 * @brief Options for matrix block multiplication
 *        generator
 * @ingroup BLAS_MAJOR_GENS
 */
typedef struct BlkMulOpts {
    /** OpenCL memory object type storing matrix (whole or its blocks) A */
    CLMemType aMobj;
    /** OpenCL memory object type storing matrix (whole or its blocks) A */
    CLMemType bMobj;
    BlkMulFlags flags;      /**< Specific flags */
    BlkmulCore core;        /**< Multiply and add core */
    /** List of argument names for the inline version */
    BlkmulArgNames argNames;
} BlkMulOpts;

void
declareBlasEnums(struct KgenContext *ctx);

/**
 * @internal
 * @brief Matrix block multiplication generator
 *
 * @param[out] ctx          Generator context
 * @param[in] subdims       Subproblem dimensions; the first level reflects
 *                          dimensions of the large blocks processed with the
 *                          whole work group, and the second level
 *                          reflects sizes of immediately multiplied small
 *                          blocks within the single work item
 * @param[in] dtype         Data type the multiplying function will be
 *                          generated for
 * @param[in] opts          Block multiplication options
 *
 * Generated functions have the following definitions: \n
 *\n
 * For the buffer based version:
 * @code
 * void
 * funcName(
 *     <type> alpha,
 *     LPtr A,
 *     LPtr B,
 *     LPtr C,
 *     [,int2 skewRow]
 *     [,int skewCol]);
 * @endcode
 *
 * Function naming rule:
 * (type prefix)gemmBlock[Transp]_<width>_<height>
 *\n
 * It's assumed A, B and C point to start of data to be
 * processed during this step.
 *\n
 * For the image based version: \n
 * @code
 * void
 * funcName(
 *     <type> alpha,
 *     __read_only image2d_t A,
 *     int2 coordA,
 *     __read_only image2d_t B,
 *     int2 coordB,
 *     LPtr C,
 *     [,int2 skewRow],
 *     [,int skewCol]);
 * @endcode
 *
 * Where coordA and coordB mean start image coordinates to fetch data from.
 *\n
 * For the image based version a mixed variant is possible when
 * either A or B blocks are passed through the local memory.
 *\n
 * The 'skewRow' and 'skewCol' are optional arguments if the
 * 'BLKMUL_SKEW_ROW' and "BLKMUL_SKEW_COLUMN" flag is specified
 * respectively. 'y' field of the row skew is for the block A, and the
 * 'x' one is for the block B.
 *\n
 * Output result can be put directly into a private location provided by the
 * caller instead of the local one. It is achieved with 'BLKMUL_OUTPUT_PRIVATE'
 * flag using.
 *\n
 * Pointer to this location should have the following types depending on the type
 * of processed data: \n
 * - float4 - for float
 * - float2 - for complex float
 * - double2 - for double and complex double
 *\n\n
 * Alpha is not taken in this case.
 *\n
 * The multiplier can be generated as well in the form of the dedicated
 * function as in the inline form inserted to a kernel. \n In case of inline
 * version the block multiplier becomes in fact the tile multiplier. In this
 * case the caller should provide iteration over K.
 *
 * @return 0 on success, -EOVERFLOW on source buffer overflowing
 */

/**
 * @internal
 * @defgroup BLAS_MAJOR_GENS BLAS specific generators
 * @ingroup MAJOR_GENS
 */
/*@{*/
int
blkMulGen(
    struct KgenContext *ctx,
    const SubproblemDim subdims[2],
    DataType dtype,
    const BlkMulOpts *opts);

int
updateResultGenOld(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    UpdateResultOp op,
    UpdateResultFlags flags,
    const UpresVarNames *uvarNames);

/*@}*/

#endif /* BLAS_KGEN_LEGACY_H_ */
