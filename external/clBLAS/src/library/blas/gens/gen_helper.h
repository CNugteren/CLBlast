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


#ifndef GEN_HELPER_H_
#define GEN_HELPER_H_

#include <kerngen.h>
#include <dblock_kgen.h>
#include <matrix_props.h>

#include "blas_kgen.h"

typedef enum BufCopyHelperFlags {
    // buffer copy functions are needed for matrix A blocks
    BCHF_MATRIX_A = 0x01,
    // buffer copy functions are needed for matrix B blocks
    BCHF_MATRIX_B = 0x02,
    /*
     * read block of output matrix
     * (either B or C)
     */
    BCHF_READ_OUTPUT = 0x04,
    // write block of output matrix
    BCHF_WRITE_OUTPUT = 0x08,
    // not unroll loops in transposing versions of customized generators
    BCHF_NOT_UNROLL_TRANSPOSE = 0x10,
    // output to image
    BCHF_IMAGE_WRITE = 0x20
} BufCopyHelperFlags;

typedef enum ZeroGenHelperFlags {
    ZF_MATRIX_A = 0x01,
    ZF_MATRIX_B = 0x02,
    ZF_MATRIX_C = 0x04
} ZeroGenHelperFlags;

/*
 * Name of functions copying matrix blocks between the global
 * and the local memory. Contains customized and generic transposing
 * or not transposing variants for reading and writing back depending on
 * generator flags, for all the matrices.
 *
 * A function name contained in a 'read*' field matches to a function
 * copying data from the global memory to the local, and this one
 * contained in a 'write*' field matches to a function copying in
 * inverse direction.
 */
typedef struct CopyBufFuncs {
    char read[MATRIX_ROLES_NUMBER][FUNC_NAME_MAXLEN];
    char write[FUNC_NAME_MAXLEN];
    char readGeneric[MATRIX_ROLES_NUMBER][FUNC_NAME_MAXLEN];
    char writeGeneric[FUNC_NAME_MAXLEN];
} CopyBufFuncs;

/*
 * Generate all needed functions copying matrix
 * blocks between the global and the local memory
 *
 * @funcs: function names structure
 * @ctx: generator context
 * @funcID: function ID
 * @gset: generator settings
 * @flags: helper flags
 *
 * The 'flags' field of the 'gset' structure must store flags from
 * the 'BufCopyHelperFlags' enumeration
 *
 * Name of functions dealing with blocks of the output matrix
 * are always stored to 'MATRIX_C' name fields.
 *
 * On success returns 0. If generation fails due
 * to buffer overflowing, returns -1.
 */
int
generateBufCopyFuncs(
    CopyBufFuncs *funcNames,
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    BufCopyHelperFlags flags);

/*
 * Have the same semantics as the previous helper,
 * but generate functions for zeroing local buffers.
 */
int
generateZeroingFuncs(
    ZeroFuncs *funcNames,
    struct KgenContext *ctx,
    const SubproblemDim *blasDim,
    const PGranularity *pgran,
    DataType dtype,
    ZeroGenHelperFlags flags);

UpdateResultFlags
kextraToUpresFlags(BlasFunctionID, KernelExtraFlags kflags);

int
generateResultUpdate(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName);

int
genResultUpdateWithFlags(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    UpdateResultFlags flags,
    const char *optFuncName,
    const char *genericFuncName,
    const char *cachedName);

void checkGenBeginHitMatrixBlock(
    struct KgenContext *ctx,
    KernelExtraFlags kflags);

void checkGenEndHitMatrixBlock(
    struct KgenContext *ctx,
    KernelExtraFlags kflags);

#endif /* GEN_HELPER_H_ */
