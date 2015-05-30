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
 * Common generators for functions manipulating
 * with data blocks placed in the global, local,
 * or private memory.
 */

/*
 * TODO: add the unroll option to 'rwMatrBlockGen'
 *       and 'smulMatrBlockGen'
 */

#ifndef DBLOCK_KGEN_H_
#define DBLOCK_KGEN_H_

#include <cltypes.h>
#include <kerngen.h>

/**
 * @internal
 * @defgroup MAJOR_GENS Major common used generators
 */
/*@{*/

/**
 * @internal
 * @brief Data block copying directions
 */
typedef enum DBlockCopyDirection {
    /** Copy from the global to the local memory */
    DBLOCK_GLOBAL_TO_LOCAL,
    /** Copy from the local to the global memory */
    DBLOCK_LOCAL_TO_GLOBAL,
    /** Copy from the global memory to an image */
    DBLOCK_GLOBAL_TO_IMAGE,
    /** Copy from the local memory to an image */
    DBLOCK_LOCAL_TO_IMAGE
} DBlockCopyDirection;

/**
 * @internal
 * @brief Data block copying flags
 */
typedef enum DBlockCopyFlags {
    DBLOCK_COPY_TRANSPOSE = 0x1,        /**< Transpose 2D block */
    /** pack several rows into single image row */
    DBLOCK_COPY_PACKED_IMAGE = 0x2,
    DBLOCK_COPY_CONJUGATE = 0x4,        /**< Conjugate complex elements */
    DBLOCK_COPY_NOT_VECTORIZE = 0x8     /**< Disable vectorized copying */
} DBlockCopyFlags;

/**
 * @internal
 * @brief Generator to copy data blocks between different kinds
 *        of memory
 *
 * @param[out] ctx              Generator context
 * @param[in] dim               Subproblem dimension to generate a function for
 * @param[in] pgran             Data parallelism granularity
 * @param[in] dtype             Data type
 * @param[in] dir               Copying direction
 * @param[in] flags             Copying flags; when an image is used as destination
 *                              block transposing is prohibited
 *
 * If 'dim' is set to NULL a generic version working with subproblem
 * of any dimension is generated. In the case specific work group
 * sizes are ignored, only work group dimension is used.
 *
 * 'x' field of the passed SuproblemDim structure should contain
 *     the block width
 * 'y' should contain the block height
 *
 * Copied blocks can be as well one as two dimensional. For any one
 * dimensional block 'y' field of the dimension structure should be
 * set to 1. If a block is two dimensional, and the local memory is \n
 * the source or destination memory, the block's rows must be aligned
 * to float4 boundary.
 *
 * Rows of the matrix block must be aligned to float4 boundary. \n
 *
 * Generated functions have the following definitions: \n
 *\n
 * Buffer-buffer copying function for optimal blocks: \n
 * @code
 * void
 * funcName(
 *     <Unified pointer type> dst,
 *     <Unified pointer type> src,
 *     size_t startRow,
 *     size_t startCol,
 *     size_t ld)
 * @endcode
 *
 * The unified pointer types can be GPtr if the global memory is used or LPtr
 * is the local memory is used respectively
 * (See the "Data types in kernels" section). Function naming rule is follow: \n
 * (type prefix)copyDBlock['Transp']['Conj']['Nvec'](src mem][dst mem]
 * [block height][block width] \n
 * The 'Nvec' suffix is added if vectorized copying is prohibited.\n
 *\n
 * Buffer-buffer copying function, generic version: \n
 * @code
 * void
 * funcName(
 *     <Unified pointer type> dst,
 *     <Unified pointer type> src,
 *     size_t startRow,
 *     size_t startCol,
 *     size_t nrRows,
 *     size_t nrCols,
 *     size_t dstLD,
 *     size_t srcLD)
 * @endcode
 *
 * Here "dstLD" is destination leading dimension, "srcLD" - source leading
 * dimension. \n
 * Naming rule is the same as for the function above except block sizes. \n
 *\n
 * Function copying optimal blocks from the global memory to an image: \n
 * @code
 * void
 * funcName(
 *     __write_only image2d_t dst,
 *     size_t startX,
 *     size_t startY,
 *     GPtr src,
 *     size_t startRow,
 *     size_t startCol,
 *     size_t ld)
 * @endcode
 * 'start' and 'startY' arguments is start X and Y coordinate in the image to
 * write from. The generic version has the analogous definition, and takes two
 * additional arguments 'nrRows' and 'nrCols' of the size_t type following just
 * fter the 'startCol' argument. \n
 *\n
 * Function copying optimal blocks from the local memory to an image: \n
 * @code
 * void
 * funcName(
 *     __write_only image2d_t dst,
 *     size_t startX,
 *     size_t startY,
 *     LPtr src)
 * @endcode
 * The generic version takes two additional arguments 'nrRows' and 'nrCols' of the
 * size_t type following just after the 'src' argument.
 *
 * @return 0 on success; on error returns negated error code:
 *
 *      - -EINVAL: unsupported data type is passed, or
 *               'DBLOCK_COPY_TRANSPOSE' is set when
 *               an image is used as destination
 *      - -ENOTSUP: unsupported copying direction is passed
 *      - -EOVEFFLOW: code buffer overflowed
 */
int
copyDataBlockGen(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const PGranularity *pgran,
    DataType dtype,
    DBlockCopyDirection dir,
    DBlockCopyFlags flags);

/*@}*/

/*
 * Zero data block in the local or global memory
 *
 * @ctx: generator context
 * @dim: Subproblem dimension to generate the function for
 * @pgran: data parallelism granularity
 * @memPrefix: type of memory to generate the function for
 *
 * The 'memPrefix' field of the passed BlasKernExtra structure
 * should contain the type of memory the buffer is stored in.
 * It cane take one of the "__local", or the "__global" value.
 *
 * 'x' field of the passed SuproblemDim structure should contain
 * the block width in float4 words. In the case the function takes only
 * a buffer pointer. If the field is set to 'SUBDIM_UNUSED'
 * the function is generated without any loop unrollings. In the case
 * the function takes buffer length as the second argument.
 *
 * If 'unroll' is set, the 'bwidth' field of the structure should
 * contain the maximum width of a block zeroed with loop unrolling.
 * If 'unroll' is set but the 'bwidth' is set to 'SUBDIM_UNUSED',
 * the generator don't apply any restriction to loop unrolling.
 * The parameter is ignored if the 'x' field of the 'dim' is set to
 * 'SUBDIM_UNUSED'.
 *
 * On success returns 0, on error returns negated error code:
 *
 *      -EINVAL: wrong memory prefix is passed
 *      -EOVEFFLOW: code buffer overflowed
 */
int
f4zeroBlockGen(
    struct KgenContext *ctx,
    const SubproblemDim *dim,
    const PGranularity *pgran,
    const char *memPrefix);

#endif /* DBLOCK_KGEN_H_ */
