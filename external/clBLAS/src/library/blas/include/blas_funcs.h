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
 * Blas function identifiers and properties
 */

#ifndef BLASFUNCS_H_
#define BLASFUNCS_H_

#include <defbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BlasFunctionID {
    CLBLAS_GEMV,
    CLBLAS_SYMV,
    CLBLAS_GEMM,
    CLBLAS_TRMM,
    CLBLAS_TRSM,
    CLBLAS_SYRK,
    CLBLAS_SYR2K,
	CLBLAS_TRMV,
	CLBLAS_HEMV,
	CLBLAS_TRSV,
	CLBLAS_TRSV_GEMV,	// Need a Kludge as current "gemv" don't support complex types
	CLBLAS_SYMM,
    CLBLAS_SYMM_DIAGONAL,
    CLBLAS_HEMM_DIAGONAL,
	CLBLAS_GEMM2,
	CLBLAS_GEMM_TAIL,
	CLBLAS_SYR,
	CLBLAS_SYR2,
	CLBLAS_GER,
	CLBLAS_HER,
	CLBLAS_HER2,
    CLBLAS_HEMM,
    CLBLAS_HERK,
    CLBLAS_TPMV,
    CLBLAS_SPMV,
    CLBLAS_HPMV,
    CLBLAS_TPSV,
    CLBLAS_SPR,
    CLBLAS_SPR2,
    CLBLAS_HPR,
    CLBLAS_HPR2,
    CLBLAS_GBMV,
    CLBLAS_TBMV,
    CLBLAS_SBMV,
    CLBLAS_HBMV,
    CLBLAS_TBSV,
    CLBLAS_SWAP,
    CLBLAS_SCAL,
    CLBLAS_COPY,
    CLBLAS_AXPY,
    CLBLAS_DOT,
    CLBLAS_REDUCTION_EPILOGUE,
    CLBLAS_ROTG,
    CLBLAS_ROTMG,
    CLBLAS_ROT,
    CLBLAS_ROTM,
    CLBLAS_iAMAX,
    CLBLAS_NRM2,
    CLBLAS_ASUM,
    CLBLAS_TRANSPOSE,

    /* ! Must be the last */
    BLAS_FUNCTIONS_NUMBER
} BlasFunctionID;

int funcBlasLevel(BlasFunctionID funcID);
bool funcHasBeta(BlasFunctionID funcID);
bool funcHasTriangMatrix(BlasFunctionID funcID);

#ifdef __cplusplus
}
#endif

#endif /* BLASFUNCS_H_ */
