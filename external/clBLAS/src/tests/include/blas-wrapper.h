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


#ifndef BLAS_WRAPPER_H_
#define BLAS_WRAPPER_H_

#include <clBLAS.h>

namespace clMath {

class blas {
public:

    // GEMV wrappers
    static void
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        float alpha,
        const float *A,
        size_t lda,
        const float *X,
        int incx,
        float beta,
        float *Y,
        int incy);

    static void
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        double alpha,
        const double *A,
        size_t lda,
        const double *X,
        int incx,
        double beta,
        double *Y,
        int incy);

    static void
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        const FloatComplex *X,
        int incx,
        FloatComplex beta,
        FloatComplex *Y,
        int incy);

    static void
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        const DoubleComplex *X,
        int incx,
        DoubleComplex beta,
        DoubleComplex *Y,
        int incy);

    // SYMV wrappers
    static void
    symv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        const float *A,
        size_t lda,
        const float *X,
        int incx,
        float beta,
        float *Y,
        int incy);

    static void
    symv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        const double *A,
        size_t lda,
        const double *X,
        int incx,
        double beta,
        double *Y,
        int incy);

    // GEMM wrappers
    static void
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const float *A,
        size_t lda,
        const float *B,
        size_t ldb,
        float beta,
        float *C,
        size_t ldc);

    static void
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        double alpha,
        const double *A,
        size_t lda,
        const double *B,
        size_t ldb,
        double beta,
        double *C,
        size_t ldc);

    static void
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        const FloatComplex *B,
        size_t ldb,
        FloatComplex beta,
        FloatComplex *C,
        size_t ldc);

    static void
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        const DoubleComplex *B,
        size_t ldb,
        DoubleComplex beta,
        DoubleComplex *C,
        size_t ldc);

    // TRMM wrappers
    static void
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        float alpha,
        const float *A,
        size_t lda,
        float *B,
        size_t ldb);

    static void
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        double alpha,
        const double *A,
        size_t lda,
        double *B,
        size_t ldb);

    static void
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        FloatComplex *B,
        size_t ldb);

    static void
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        DoubleComplex *B,
        size_t ldb);

    // TRSM wrappers
    static void
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        float alpha,
        const float *A,
        size_t lda,
        float *B,
        size_t ldb);

    static void
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        double alpha,
        const double *A,
        size_t lda,
        double *B,
        size_t ldb);

    static void
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        FloatComplex *B,
        size_t ldb);

    static void
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        DoubleComplex *B,
        size_t ldb);

    // SYR2K wrappers
    static void
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        float alpha,
        const float *A,
        size_t lda,
        const float *B,
        size_t ldb,
        float beta,
        float *C,
        size_t ldc);

    static void
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        double alpha,
        const double *A,
        size_t lda,
        const double *B,
        size_t ldb,
        double beta,
        double *C,
        size_t ldc);

    static void
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        const FloatComplex *B,
        size_t ldb,
        FloatComplex beta,
        FloatComplex *C,
        size_t ldc);

    static void
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        const DoubleComplex *B,
        size_t ldb,
        DoubleComplex beta,
        DoubleComplex *C,
        size_t ldc);

    // SYRK wrappers
    static void
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        float alpha,
        const float *A,
        size_t lda,
        float beta,
        float *C,
        size_t ldc);

    static void
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        double alpha,
        const double *A,
        size_t lda,
        double beta,
        double *C,
        size_t ldc);

    static void
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t lda,
        FloatComplex beta,
        FloatComplex *C,
        size_t ldc);

    static void
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t lda,
        DoubleComplex beta,
        DoubleComplex *C,
        size_t ldc);

	static void
    trmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
	    clblasDiag diag,
        size_t N,
        float *A,
	    size_t offa,
        size_t lda,
        float *X,
	    size_t offx,
        int incx);

	static void
    trmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        double *A,
        size_t offa,
        size_t lda,
        double *X,
        size_t offx,
        int incx);

	static void
    trmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        FloatComplex *A,
        size_t offa,
        size_t lda,
        FloatComplex *X,
        size_t offx,
        int incx);

	static void
    trmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        DoubleComplex *A,
	    size_t offa,
        size_t lda,
        DoubleComplex *X,
	    size_t offx,
        int incx);

    //TPMV
   static void
    tpmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        float *AP,
        size_t offa,
        float *X,
        size_t offx,
        int incx);

    static void
    tpmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        double *AP,
        size_t offa,
        double *X,
        size_t offx,
        int incx);

    static void
    tpmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        FloatComplex *AP,
        size_t offa,
        FloatComplex *X,
        size_t offx,
        int incx);

    static void
    tpmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        DoubleComplex *AP,
        size_t offa,
        DoubleComplex *X,
        size_t offx,
        int incx);



	 static void
    trsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        float *A,
	    size_t offa,
        size_t lda,
        float *X,
	    size_t offx,
        int incx);

        static void
    trsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        double *A,
	    size_t offa,
        size_t lda,
        double *X,
	    size_t offx,
        int incx);

        static void
    trsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        FloatComplex *A,
	    size_t offa,
        size_t lda,
        FloatComplex *X,
	    size_t offx,
        int incx);

        static void
     trsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        DoubleComplex *A,
	    size_t offa,
        size_t lda,
        DoubleComplex *X,
	    size_t offx,
        int incx);

static void
    tpsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        float *A,
        size_t offa,
        float *X,
        size_t offx,
        int incx);

static void
    tpsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        double *A,
        size_t offa,
        double *X,
        size_t offx,
        int incx);

static void
    tpsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        FloatComplex *A,
        size_t offa,
        FloatComplex *X,
        size_t offx,
        int incx);

static void
    tpsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        DoubleComplex *A,
        size_t offa,
        DoubleComplex *X,
        size_t offx,
        int incx);

static void
	symm(
		clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        float alpha,
        float* A,
	    size_t offa,
        size_t lda,
        float* B,
	    size_t offb,
        size_t ldb,
        float beta,
        float* C,
	    size_t offc,
        size_t ldc);

static void
    symm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        double alpha,
        double* A,
	    size_t offa,
        size_t lda,
        double* B,
	    size_t offb,
        size_t ldb,
        double beta,
        double* C,
	    size_t offc,
        size_t ldc);

static void
    symm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        FloatComplex alpha,
        FloatComplex* A,
	    size_t offa,
        size_t lda,
        FloatComplex* B,
	    size_t offb,
        size_t ldb,
        FloatComplex  beta,
        FloatComplex* C,
	    size_t offc,
        size_t ldc);

static void
    symm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* A,
	    size_t offa,
        size_t lda,
        DoubleComplex* B,
	    size_t offb,
        size_t ldb,
        DoubleComplex beta,
        DoubleComplex* C,
	    size_t offc,
        size_t ldc);


static void
   ger(
        clblasOrder order,
        size_t M,
        size_t N,
        float alpha,
        float* x,
        size_t offx,
        int incx,
        float* y,
        size_t offy,
        int incy,
        float* A ,
        size_t offa,
        size_t lda);

static void
    ger(
        clblasOrder order,
        size_t M,
        size_t N,
        double alpha,
        double* x,
        size_t offx,
        int incx,
        double* y,
        size_t offy,
        int incy,
        double* A,
        size_t offa,
        size_t lda);

static void
   ger(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex alpha,
         FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* y,
        size_t offy,
        int incy,
        FloatComplex* A ,
        size_t offa,
        size_t lda);

static void
    ger(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* y,
        size_t offy,
        int incy,
        DoubleComplex* A,
        size_t offa,
        size_t lda);

static void
   gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* y,
        size_t offy,
        int incy,
        FloatComplex* A ,
        size_t offa,
        size_t lda);

static void
    gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* y,
        size_t offy,
        int incy,
        DoubleComplex* A,
        size_t offa,
        size_t lda);


//HER wrappers

static void
   her(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* A ,
        size_t offa,
        size_t lda);

static void
    her(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* A,
        size_t offa,
        size_t lda);

// SYR wrappers
static void
    syr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* A,
        size_t offa,
        size_t lda);

	static void
    syr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double Alpha,
        double* X,
        size_t offx,
        int incx,
        double* A,
        size_t offa,
        size_t lda);

//SPR

static void
    spr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* AP,
        size_t offa);

	static void
    spr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double Alpha,
        double* X,
        size_t offx,
        int incx,
        double* AP,
        size_t offa);


	static void
    syr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* Y,
        size_t offy,
        int incy,
        float* A,
        size_t offa,
        size_t lda);

	static void
    syr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double Alpha,
        double* X,
        size_t offx,
        int incx,
        double* Y,
        size_t offy,
        int incy,
		double* A,
        size_t offa,
        size_t lda);

//HER2
 static void
    her2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        FloatComplex* X,
        size_t offx,
        int incx,
        FloatComplex* Y,
        size_t offy,
        int incy,
        FloatComplex* A,
        size_t offa,
        size_t lda);

    static void
    her2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* X,
        size_t offx,
        int incx,
        DoubleComplex* Y,
        size_t offy,
        int incy,
        DoubleComplex* A,
        size_t offa,
        size_t lda);


static void
    hemv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        FloatComplex* A,
        size_t offa,
        size_t lda,
        FloatComplex* X,
        size_t offx,
        int incx,
        FloatComplex beta,
        FloatComplex* Y,
        size_t offy,
        int incy);

static void
    hemv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* A,
        size_t offa,
        size_t lda,
        DoubleComplex* X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        DoubleComplex* Y,
        size_t offy,
        int incy);

//HEMM
static void
    hemm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        FloatComplex alpha,
        FloatComplex* A,
        size_t offa,
        size_t lda,
        FloatComplex* B,
        size_t offb,
        size_t ldb,
        FloatComplex  beta,
        FloatComplex* C,
        size_t offc,
        size_t ldc);

static void
    hemm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* A,
        size_t offa,
        size_t lda,
        DoubleComplex* B,
        size_t offb,
        size_t ldb,
        DoubleComplex beta,
        DoubleComplex* C,
        size_t offc,
        size_t ldc);

// HERK wrappers
static void
    herk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        float alpha,
        const FloatComplex *A,
        size_t lda,
        float beta,
        FloatComplex *C,
        size_t ldc);

static void
    herk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        double alpha,
        const DoubleComplex *A,
        size_t lda,
        double beta,
        DoubleComplex *C,
        size_t ldc);

// SPMV wrappers
    static void
    spmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        const float *A,
        size_t offa,
        const float *X,
        size_t offx,
        int incx,
        float beta,
        float *Y,
        size_t offy,
        int incy);

    static void
    spmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        const double *A,
        size_t offa,
        const double *X,
        size_t offx,
        int incx,
        double beta,
        double *Y,
        size_t offy,
        int incy);

static void
    hpmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        FloatComplex* A,
        size_t offa,
        FloatComplex* X,
        size_t offx,
        int incx,
        FloatComplex beta,
        FloatComplex* Y,
        size_t offy,
        int incy);

static void
    hpmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* A,
        size_t offa,
        DoubleComplex* X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        DoubleComplex* Y,
        size_t offy,
        int incy);

//HPR wrappers
static void
   hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* AP ,
        size_t offa);

static void
    hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* AP,
        size_t offa);

static void
    spr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* Y,
        size_t offy,
        int incy,
        float* AP,
        size_t offa);

	static void
    spr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double Alpha,
        double* X,
        size_t offx,
        int incx,
        double* Y,
        size_t offy,
        int incy,
		double* AP,
        size_t offa);

 static void
    hpr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        FloatComplex* X,
        size_t offx,
        int incx,
        FloatComplex* Y,
        size_t offy,
        int incy,
        FloatComplex* AP,
        size_t offa);

    static void
    hpr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex* X,
        size_t offx,
        int incx,
        DoubleComplex* Y,
        size_t offy,
        int incy,
        DoubleComplex* AP,
        size_t offa);

    // GBMV wrappers
    static void
    gbmv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        float alpha,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx,
        float beta,
        float *Y,
        size_t offy,
        int incy);

    static void
    gbmv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        double alpha,
        double *A,
        size_t offa,
        size_t lda,
        double *X,
        size_t offx,
        int incx,
        double beta,
        double *Y,
        size_t offy,
        int incy);

    static void
    gbmv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        FloatComplex alpha,
        FloatComplex *A,
        size_t offa,
        size_t lda,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex beta,
        FloatComplex *Y,
        size_t offy,
        int incy);

	static void
    gbmv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        DoubleComplex alpha,
        DoubleComplex *A,
        size_t offa,
        size_t lda,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//TBMV

static void
    tbmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx);

static void
    tbmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        double *A,
        size_t offa,
        size_t lda,
        double *X,
        size_t offx,
        int incx);

static void
    tbmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        FloatComplex *A,
        size_t offa,
        size_t lda,
        FloatComplex *X,
        size_t offx,
        int incx);

static void
    tbmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        DoubleComplex *A,
        size_t offa,
        size_t lda,
        DoubleComplex *X,
        size_t offx,
        int incx);

//SBMV

static void
    sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        float alpha,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx,
        float beta,
        float *Y,
        size_t offy,
        int incy);

static void
    sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        double alpha,
        double *A,
        size_t offa,
        size_t lda,
        double *X,
        size_t offx,
        int incx,
        double beta,
        double *Y,
        size_t offy,
        int incy);

//HBMV
static void
    hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        FloatComplex alpha,
        FloatComplex *A,
        size_t offa,
        size_t lda,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex beta,
        FloatComplex *Y,
        size_t offy,
        int incy);

static void
    hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        DoubleComplex *A,
        size_t offa,
        size_t lda,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//TBSV

static void
    tbsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx);

static void
    tbsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        double *A,
        size_t offa,
        size_t lda,
        double *X,
        size_t offx,
        int incx);

static void
    tbsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        FloatComplex *A,
        size_t offa,
        size_t lda,
        FloatComplex *X,
        size_t offx,
        int incx);

static void
    tbsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        size_t K,
        DoubleComplex *A,
        size_t offa,
        size_t lda,
        DoubleComplex *X,
        size_t offx,
        int incx);

static void
    her2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        FloatComplex alpha,
        const FloatComplex *A,
        size_t offa,
        size_t lda,
        const FloatComplex *B,
        size_t offb,
        size_t ldb,
        float beta,
        FloatComplex *C,
        size_t offc,
        size_t ldc);

static void
    her2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        const DoubleComplex *A,
        size_t offa,
        size_t lda,
        const DoubleComplex *B,
        size_t offb,
        size_t ldb,
        double beta,
        DoubleComplex *C,
        size_t offc,
        size_t ldc);

//copy

static void
    copy(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy);

static void
    copy(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy);

static void
    copy(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy);
static void
    copy(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//DOT

static float
    dot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy);

static double
    dot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy);

static FloatComplex
    dot(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy);

static DoubleComplex
    dot(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//ASUM

static float
    asum(
        size_t N,
        float *X,
        size_t offx,
        int incx);

static double
    asum(
        size_t N,
        double *X,
        size_t offx,
        int incx);

static float
    asum(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

static double
    asum(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);


static FloatComplex
    dotc(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy);

static DoubleComplex
    dotc(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy);


 // SWAP wrappers
    static void
    swap(
        size_t N,
        float *X,
        size_t offa,
		int incx,
        float *Y,
		size_t offb,
        int incy);

	static void
    swap(
        size_t N,
        double *X,
        size_t offa,
		int incx,
        double *Y,
		size_t offb,
        int incy);

	static void
    swap(
        size_t N,
        FloatComplex *X,
        size_t offa,
		int incx,
        FloatComplex *Y,
		size_t offb,
        int incy);

 	static void
    swap(
        size_t N,
        DoubleComplex *X,
        size_t offa,
		int incx,
        DoubleComplex *Y,
		size_t offb,
        int incy);

// Scal
static void scal(
        bool is_css_zds,
        size_t N,
        float alpha,
        float *X,
        size_t offx,
        int incx);

static void scal(
        bool is_css_zds,
        size_t N,
        double alpha,
        double *X,
        size_t offx,
        int incx);

static void scal(
        bool is_css_zds,
        size_t N,
        FloatComplex alpha,
        FloatComplex *X,
        size_t offx,
        int incx);

static void scal(
        bool is_css_zds,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex *X,
        size_t offx,
        int incx);

//axpy calls
static void
	axpy(
		size_t N,
        float alpha,
		const float * X,
		size_t offBX,
		int incx,
		float *Y,
		size_t offCY,
		int incy);

 static void
	axpy(
		size_t N,
        double alpha,
		const double *X,
		size_t offBX,
		int incx,
		double *Y,
		size_t offCY,
		int incy);

 static void
	axpy(
		size_t N,
        FloatComplex alpha,
		const FloatComplex *X,
		size_t offBX,
		int incx,
		FloatComplex *Y,
		size_t offCY,
		int incy);

 static void
	axpy(
		size_t N,
        DoubleComplex alpha,
		const DoubleComplex *X,
		size_t offBX,
		int incx,
		DoubleComplex *Y,
		size_t offCY,
		int incy);

static void rotmg(
        float* D1,
        size_t offD1,
        float* D2,
        size_t offD2,
        float* X1,
        size_t offX1,
        const float* Y1,
        size_t offY1,
        float* PARAM,
        size_t offParam);

static void rotmg(
        double* D1,
        size_t offD1,
        double* D2,
        size_t offD2,
        double* X1,
        size_t offX1,
        const double* Y1,
        size_t offY1,
        double* PARAM,
        size_t offParam);

static void rotm(
        size_t N,
        float* X,
        size_t offx,
        int incx,
        float* Y,
        size_t offy,
        int incy,
        float* PARAM,
        size_t offParam);

static void rotm(
        size_t N,
        double* X,
        size_t offx,
        int incx,
        double* Y,
        size_t offy,
        int incy,
        double* PARAM,
        size_t offParam);

static void rotg(
        float* SA,
        size_t offSA,
        float* SB,
        size_t offSB,
        float* C,
        size_t offC,
        float* S,
        size_t offS);

static void rotg(
        double* SA,
        size_t offSA,
        double* SB,
        size_t offSB,
        double* C,
        size_t offC,
        double* S,
        size_t offS);

static void rotg(
        FloatComplex* SA,
        size_t offSA,
        FloatComplex* SB,
        size_t offSB,
        float* C,
        size_t offC,
        FloatComplex* S,
        size_t offS);

static void rotg(
        DoubleComplex* SA,
        size_t offSA,
        DoubleComplex* SB,
        size_t offSB,
        double* C,
        size_t offC,
        DoubleComplex* S,
        size_t offS);

static void rot(
        size_t N,
        float* X,
        size_t offx,
        int incx,
        float* Y,
        size_t offy,
        int incy,
        float C,
        float S);

static void rot(
        size_t N,
        double* X,
        size_t offx,
        int incx,
        double* Y,
        size_t offy,
        int incy,
        double C,
        double S);

static void rot(
        size_t N,
        FloatComplex* X,
        size_t offx,
        int incx,
        FloatComplex* Y,
        size_t offy,
        int incy,
        FloatComplex C,
        FloatComplex S);

static void rot(
        size_t N,
        DoubleComplex* X,
        size_t offx,
        int incx,
        DoubleComplex* Y,
        size_t offy,
        int incy,
        DoubleComplex C,
        DoubleComplex S);

static int
    iamax(
        size_t N,
        float *X,
        size_t offx,
        int incx);

static int
    iamax(
        size_t N,
        double *X,
        size_t offx,
        int incx);

static int
    iamax(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

static int
    iamax(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);

static float
    nrm2(
        size_t N,
        float *X,
        size_t offx,
        int incx);

static double
    nrm2(
        size_t N,
        double *X,
        size_t offx,
        int incx);

static float
    nrm2(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

static double
    nrm2(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);


};// class blas

}   // namespace clMath;

#endif  // BLAS_WRAPPER_H_
