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


#include "storage_data.h"

void
initGemm(BlasFunctionInfo* bFunc)
{
    bFunc->name = "GEMM";
    bFunc->envImplementation = "AMD_CLBLAS_GEMM_IMPLEMENTATION";
    bFunc->numPatterns = initGemmMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 2;
    bFunc->maskForTuningsKernel =
            KEXTRA_TRANS_A
            | KEXTRA_TRANS_B
            | KEXTRA_COLUMN_MAJOR
            ;

    bFunc->maskForUniqueKernels =
            KEXTRA_TRANS_A
            | KEXTRA_CONJUGATE_A
            | KEXTRA_TRANS_B
            | KEXTRA_CONJUGATE_B
            | KEXTRA_COLUMN_MAJOR
            | KEXTRA_BETA_ZERO
            ;
}

void
initTrmm(BlasFunctionInfo* bFunc)
{
    bFunc->name = "TRMM";
    bFunc->envImplementation = "AMD_CLBLAS_TRMM_IMPLEMENTATION";
    bFunc->numPatterns = initTrmmMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_TRANS_A
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_SIDE_RIGHT
        | KEXTRA_COLUMN_MAJOR
        ;
    bFunc->maskForUniqueKernels =
        KEXTRA_TRANS_A
        | KEXTRA_CONJUGATE_A
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_SIDE_RIGHT
        | KEXTRA_UNIT_DIAGONAL
        | KEXTRA_COLUMN_MAJOR
        ;
}
void
initTrsm(BlasFunctionInfo* bFunc)
{
    bFunc->name = "TRSM";
    bFunc->envImplementation = "AMD_CLBLAS_TRSM_IMPLEMENTATION";
    bFunc->numPatterns = initTrsmMemPatterns(bFunc->pattern);
    // FIXME Correct, when adding a new pattern will not lead to corrupt it.

    // don't create a partition for new TRSM pattern
    if (bFunc->numPatterns == 3) {
        bFunc->numPatterns = 2;
    }
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_TRANS_A
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_SIDE_RIGHT
        | KEXTRA_COLUMN_MAJOR
        ;
    bFunc->maskForUniqueKernels =
        KEXTRA_TRANS_A
        | KEXTRA_CONJUGATE_A
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_SIDE_RIGHT
        | KEXTRA_UNIT_DIAGONAL
        | KEXTRA_COLUMN_MAJOR
        ;
}

void
initGemv(BlasFunctionInfo* bFunc)
{
    bFunc->name = "GEMV";
    bFunc->envImplementation = NULL;
    bFunc->numPatterns = initGemvMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_TRANS_A
        | KEXTRA_COLUMN_MAJOR
        | KEXTRA_UPPER_TRIANG
        ;
    bFunc->maskForUniqueKernels =
        KEXTRA_TRANS_A
        | KEXTRA_COLUMN_MAJOR
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_BETA_ZERO
        | KEXTRA_INCX_ONE
        | KEXTRA_INCY_ONE
        ;
}

void
initSymv(BlasFunctionInfo* bFunc)
{
    bFunc->name = "SYMV";
    bFunc->envImplementation = NULL;
    bFunc->numPatterns = initSymvMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_COLUMN_MAJOR
        | KEXTRA_UPPER_TRIANG
        ;
    bFunc->maskForUniqueKernels =
        KEXTRA_COLUMN_MAJOR
        | KEXTRA_UPPER_TRIANG
        | KEXTRA_BETA_ZERO
        | KEXTRA_INCX_ONE
        | KEXTRA_INCY_ONE
        ;
}

void
initSyr2k(BlasFunctionInfo* bFunc)
{
    bFunc->name = "SYR2K";
    bFunc->envImplementation = NULL;
    bFunc->numPatterns = initSyr2kMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_TRANS_A
        //| KEXTRA_CONJUGATE_A
        //| KEXTRA_TRANS_B
        //| KEXTRA_CONJUGATE_B
        | KEXTRA_COLUMN_MAJOR
        //| KEXTRA_UPPER_TRIANG
        //|KEXTRA_SIDE_RIGHT
        //| KEXTRA_TAILS_M
        //| KEXTRA_TAILS_N
        //| KEXTRA_TAILS_K
        //| KEXTRA_BETA_ZERO
        //| KEXTRA_NO_COPY_VEC_A = 0x1000,
        //| KEXTRA_NO_COPY_VEC_B = 0x2000,
        //| KEXTRA_NO_COPY_VEC_C = 0x4000,
        ;
    bFunc->maskForUniqueKernels = bFunc->maskForTuningsKernel;
}

void
initSyrk(BlasFunctionInfo* bFunc)
{
    bFunc->name = "SYRK";
    bFunc->envImplementation = NULL;
    bFunc->numPatterns = initSyrkMemPatterns(bFunc->pattern);
    bFunc->defaultPattern = bFunc->numPatterns - 1;
    bFunc->maskForTuningsKernel =
        KEXTRA_TRANS_A
        //| KEXTRA_CONJUGATE_A
        //| KEXTRA_TRANS_B
        //| KEXTRA_CONJUGATE_B
        | KEXTRA_COLUMN_MAJOR
        //| KEXTRA_UPPER_TRIANG
        //|KEXTRA_SIDE_RIGHT
        //| KEXTRA_TAILS_M
        //| KEXTRA_TAILS_N
        //| KEXTRA_TAILS_K
        //| KEXTRA_BETA_ZERO
        //| KEXTRA_NO_COPY_VEC_A = 0x1000,
        //| KEXTRA_NO_COPY_VEC_B = 0x2000,
        //| KEXTRA_NO_COPY_VEC_C = 0x4000,
        ;
    bFunc->maskForUniqueKernels = bFunc->maskForTuningsKernel;}

void
initBlasFuncionData(BlasFunctionInfo* fInfo)
{
//    unsigned int func;

    memset(fInfo, 0, BLAS_FUNCTIONS_NUMBER * sizeof(BlasFunctionInfo));

    fInfo[CLBLAS_GEMM].initFunctionInfo = initGemm;
    fInfo[CLBLAS_TRMM].initFunctionInfo = initTrmm;
    fInfo[CLBLAS_TRSM].initFunctionInfo = initTrsm;
    fInfo[CLBLAS_GEMV].initFunctionInfo = initGemv;
    fInfo[CLBLAS_SYMV].initFunctionInfo = initSymv;
    fInfo[CLBLAS_SYR2K].initFunctionInfo = initSyr2k;
    fInfo[CLBLAS_SYRK].initFunctionInfo = initSyrk;

}
