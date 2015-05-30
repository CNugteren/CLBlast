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


#ifndef KERNEL_EXTRA_H_
#define KERNEL_EXTRA_H_

#include <cltypes.h>

enum {
    MAX_SOLVER_PRIVATE_SIZE = 16
};

//
// Moving BUILD_OPTS_MAXLEN here. Originally in clblas-internal.h
//      Including "clblas-internal.h"
enum {
    MEMPAT_PER_BLASFN = 8,
    BUILD_OPTS_MAXLEN = 256
};

/**
 * @internal
 * @brief BLAS kernel type identifiers
 *
 * @ingroup BLAS_SOLVERIF_SPEC
 */
typedef enum CLBlasKernelType {
    CLBLAS_COMPUTING_KERNEL,        /**< Main computing kernel */
    CLBLAS_PREP_A_KERNEL,           /**< Kernel preparing matrix A */
    CLBLAS_PREP_B_KERNEL,           /**< Kernel preparing matrix B */
    MAX_CLBLAS_KERNELS_PER_STEP
} CLBlasKernelType;

/**
 * @internal
 * @defgroup BLAS_SOLVERIF_SPEC BLAS specifics
 * @ingroup SOLVERIF
 */

/*@{*/

/**
 * @brief BLAS kernel flags
 *
 * These flags uniquely determine problem options kernels are generated for
 */
typedef enum KernelExtraFlags {
    /** Matches to a problem without any options */
    KEXTRA_NO_FLAGS = 0,
    KEXTRA_TRANS_A = 0x01,      /**< Matrix A should be transposed */
    /** matrix A should be took in the conjugate form */
    KEXTRA_CONJUGATE_A = 0x02,
    KEXTRA_TRANS_B = 0x04,      /**< matrix B should be transposed */
    /** Matrix B should be taken in the conjugate form */
    KEXTRA_CONJUGATE_B = 0x08,
    KEXTRA_COLUMN_MAJOR = 0x10, /**< Order is column major */
    /**
     * Matrix A is upper triangular, it is lower triangular
     * if this flag is not set
     */
    KEXTRA_UPPER_TRIANG = 0x20,
    /**
     * Matrix A is placed on the right, it is placed
     * on the left if this flag is not set
     */
    KEXTRA_SIDE_RIGHT = 0x40,
    /**
     * Unit diagonal matrix
     */
    KEXTRA_UNIT_DIAGONAL = 0x80,
    /** kernel should process tails of upper level blocks in M dimension */
    KEXTRA_TAILS_M = 0x100,
    /** kernel should process tails of upper level blocks in N dimension */
    KEXTRA_TAILS_N = 0x200,
    /** kernel should process tails of upper level blocks in K dimension */
    KEXTRA_TAILS_K = 0x400,
    /** Beta multiplier is zero */
    KEXTRA_BETA_ZERO = 0x800,
    /** Disable vectorization at block copying for matrix A */
    KEXTRA_NO_COPY_VEC_A = 0x1000,
    /** Disable vectorization at block copying for matrix B */
    KEXTRA_NO_COPY_VEC_B = 0x2000,
    /** Disable vectorization at block copying for matrix C */
    KEXTRA_NO_COPY_VEC_C = 0x4000,
    // SYRXK specific flags
    /** Diagonal solution blocks are evaluated in a separate kernel */
    KEXTRA_SYRK_SEPARATE_DIAGONAL = 0x8000,
    /** Evaluate diagonal solution blocks for a SYRXK function */
    KEXTRA_SYRK_EVALUATE_DIAGONAL = 0x10000,
    /** 2k rank update */
    KEXTRA_SYRK_2K_RANK = 0x20000,
    // BLAS2 specific flags
    /** Incx increment is one */
    KEXTRA_INCX_ONE = 0x40000,
    /** Incy increment is one */
    KEXTRA_INCY_ONE = 0x80000,
    // Generator specific flags
    /** MAD function can be used */
    // FIXME: throw this kludge away
    KEXTRA_ENABLE_MAD = 0x100000,
    // FIXME: It's a kludge, pass further DeviceIndent structure to generators
    KEXTRA_VENDOR_AMD = 0x200000,
    /* Flags showing not zero starting offsets for kernels */
    KEXTRA_STARTM_NOT_ZERO = 0x400000,
    KEXTRA_STARTN_NOT_ZERO = 0x800000,
    //KEXTRA_STARTK_NOT_ZERO = 0x2000000,
    /** Matrix A offset in a memory object is not zero */
    KEXTRA_A_OFF_NOT_ZERO = 0x1000000,
    /** Matrix B or vector X offset in a memory object is not zero */
    KEXTRA_BX_OFF_NOT_ZERO = 0x2000000,
    /** Matrix C or vector Y offset in a memory object is not zero */
    KEXTRA_CY_OFF_NOT_ZERO = 0x4000000,
    /** kernel should process tails of lower level blocks in M dimension */
    KEXTRA_TAILS_M_LOWER = 0x8000000,
    /** kernel should process tails of lower level blocks in N dimension */
    KEXTRA_TAILS_N_LOWER = 0x10000000,
    /** kernel should process tails of lower level blocks in K dimension */
    KEXTRA_TAILS_K_LOWER = 0x20000000
} KernelExtraFlags;

/**
 * @internal
 * @brief extra information CLBLAS kernel generator
 * @ingroup BLAS_SOLVERIF_SPEC
 */
typedef struct CLBLASKernExtra {
    DataType dtype;             /**< Data type */
    KernelExtraFlags flags;     /**< Kernel flags identifying a problem */
    CLBlasKernelType kernType;  /**< Kernel type */
    // Fixme: Deprecate it; now it is just for backward compatibility
    unsigned int vecLen;        /**< vector length to evaluate with */
    /** vector length for matrix A elements to evaluate with */
    unsigned int vecLenA;
    /** vector length for matrix B elements to evaluate with */
    unsigned int vecLenB;
    /*
     * FIXME: remove this kludge; vectorization for the result should be
     *        autodetected
     */
    unsigned int vecLenC;
    char solverPriv[MAX_SOLVER_PRIVATE_SIZE];
    char buildOptions[BUILD_OPTS_MAXLEN]; // Build Flags used for the kernel call
} CLBLASKernExtra;

/*
 * function to compare blas kernels extra information
 */
int
clblasKernelExtraCmp(const void *extra, const void *extraKey);

/*@}*/

#endif /* KERNEL_EXTRA_H_ */
