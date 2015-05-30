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


#ifndef BLAS_INTERNAL_H_
#define BLAS_INTERNAL_H_

#ifdef __cplusplus
extern "C" {
#endif


/* BLAS-2 functions */

void
blasSgemv(
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

void
blasDgemv(
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

void
blasCgemv(
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

void
blasZgemv(
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

void
blasSsymv(
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

void
blasDsymv(
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

/* BLAS-3 functions */

void
blasSgemm(
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

void
blasDgemm(
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

void
blasCgemm(
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

void
blasZgemm(
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

void
blasStrmm(
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

void
blasDtrmm(
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

void
blasCtrmm(
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

void
blasZtrmm(
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

void
blasStrsm(
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

void
blasDtrsm(
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

void
blasCtrsm(
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

void
blasZtrsm(
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

void
blasSsyr2k(
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

void
blasDsyr2k(
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

void
blasCsyr2k(
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

void
blasZsyr2k(
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

void
blasSsyrk(
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

void
blasDsyrk(
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

void
blasCsyrk(
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

void
blasZsyrk(
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

void
blasStrmv(
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

void
blasDtrmv(
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

void
blasCtrmv(
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

void
blasZtrmv(
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


void
blasStpmv(
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

void
blasDtpmv(
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

void
blasCtpmv(
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

void
blasZtpmv(
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


void
blasStrsv(
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

void
blasDtrsv(
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

void
blasCtrsv(
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

void
blasZtrsv(
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

void
blasStpsv(
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

void
blasDtpsv(
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

void
blasCtpsv(
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

void
blasZtpsv(
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

void
    blasSsymm(
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

void
    blasDsymm(
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

void
    blasCsymm(
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

void
    blasZsymm(
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

void
    blasSger(
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
        float* A,
        size_t offa,
        size_t lda);

void
    blasDger(
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

void
    blasCgeru(
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
        FloatComplex* A,
        size_t offa,
        size_t lda);

void
    blasZgeru(
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

void
    blasCgerc(
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
        FloatComplex* A,
        size_t offa,
        size_t lda);

void
    blasZgerc(
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


void
    blasCher(
        clblasOrder order,
 	    clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* A,
        size_t offa,
        size_t lda);

void
    blasZher(
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


void
	 blasDsyr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        double* X,
        size_t offx,
        int incx,
        double* A,
        size_t offa,
        size_t lda);
void
        blasSsyr(
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

void
     blasDspr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        double* X,
        size_t offx,
        int incx,
        double* AP,
        size_t offa);

void
    blasSspr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* AP,
        size_t offa);


void
	blasSsyr2(
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

void
	 blasDsyr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
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
void
    blasCher2(
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

void
     blasZher2(
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



void
    blasChemv(
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

void
    blasZhemv(
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
void
    blasChemm(
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

void
    blasZhemm(
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


void
	blasCherk(
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

void
	blasZherk(
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


void
blasSspmv(
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

void
blasDspmv(
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


void
    blasChpmv(
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

void
    blasZhpmv(
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

void
    blasChpr(
        clblasOrder order,
 	    clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* AP,
        size_t offa);

void
    blasZhpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* AP,
        size_t offa);

void
	blasSspr2(
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

void
	 blasDspr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        double* X,
        size_t offx,
        int incx,
        double* Y,
        size_t offy,
        int incy,
		double* AP,
        size_t offa);


void
    blasChpr2(
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

void
     blasZhpr2(
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

void
    blasSgbmv(
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

void
    blasDgbmv(
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

void
    blasCgbmv(
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

void
    blasZgbmv(
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

void
    blasStbmv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        size_t K,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx);

void
    blasDtbmv(
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
void
    blasCtbmv(
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

void
    blasZtbmv(
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

void
    blasSsbmv(
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

void
    blasDsbmv(
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


void
    blasChbmv(
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

void
    blasZhbmv(
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

void
    blasStbsv(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t N,
        size_t K,
        float *A,
        size_t offa,
        size_t lda,
        float *X,
        size_t offx,
        int incx);

void
    blasDtbsv(
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
void
    blasCtbsv(
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

void
    blasZtbsv(
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

void
	blasCher2k(
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

void
	blasZher2k(
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

/* BLAS-1 functions */

//swap
void
blasSswap(
	size_t N,
	float *X,
	size_t offBX,
	int incx,
	float *Y,
	size_t offCY,
	int incy);

void
blasDswap(
	size_t N,
	double *X,
	size_t offBX,
	int incx,
	double *Y,
	size_t offCY,
	int incy);

void
blasCswap(
	size_t N,
	FloatComplex *X,
	size_t offBX,
	int incx,
	FloatComplex *Y,
	size_t offCY,
	int incy);

void
blasZswap(
	size_t N,
	DoubleComplex *X,
	size_t offBX,
	int incx,
	DoubleComplex *Y,
	size_t offCY,
	int incy);



//Scal
void
	blasSscal(
	    size_t N,
        float alpha,
        float *X,
        size_t offx,
        int incx);

void
	blasDscal(
	    size_t N,
        double alpha,
        double *X,
        size_t offx,
        int incx);

void
	blasCscal(
	    size_t N,
        FloatComplex alpha,
        FloatComplex *X,
        size_t offx,
        int incx);


void
	blasZscal(
	    size_t N,
        DoubleComplex alpha,
        DoubleComplex *X,
        size_t offx,
        int incx);

void
	blasCsscal(
	    size_t N,
        float alpha,
        FloatComplex *X,
        size_t offx,
        int incx);

void
	blasZdscal(
	    size_t N,
        double alpha,
        DoubleComplex *X,
        size_t offx,
        int incx);

//COPY

void
blasScopy(
    size_t N,
    float *X,
    size_t offx,
    int incx,
    float *Y,
    size_t offy,
    int incy);

void
blasDcopy(
    size_t N,
    double *X,
    size_t offx,
    int incx,
    double *Y,
    size_t offy,
    int incy);

void
blasCcopy(
    size_t N,
    FloatComplex *X,
    size_t offx,
    int incx,
    FloatComplex *Y,
    size_t offy,
    int incy);

void
blasZcopy(
    size_t N,
    DoubleComplex *X,
    size_t offx,
    int incx,
    DoubleComplex *Y,
    size_t offy,
    int incy);


// DOT
float
blasSdot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy);

double
blasDdot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy);

FloatComplex
blasCdotu(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy);

DoubleComplex
blasZdotu(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//ASUM

float
blasSasum(
        size_t N,
        float *X,
        size_t offx,
        int incx);

double
blasDasum(
        size_t N,
        double *X,
        size_t offx,
        int incx);

float
blasScasum(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

double
blasDzasum(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);

//DOTC
FloatComplex
blasCdotc(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy);

DoubleComplex
blasZdotc(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy);

//axpy
void
blasSaxpy(
        size_t N,
        float alpha,
        const float *X,
        size_t offBX,
        int incx,
        float *Y,
        size_t offCY,
        int incy);

void
blasDaxpy(
        size_t N,
        double alpha,
        const double *X,
        size_t offBX,
        int incx,
        double *Y,
        size_t offCY,
        int incy);
void
blasCaxpy(
        size_t N,
        FloatComplex alpha,
        const FloatComplex *X,
        size_t offBX,
        int incx,
        FloatComplex *Y,
        size_t offCY,
        int incy);
void
blasZaxpy(
        size_t N,
        DoubleComplex alpha,
        const DoubleComplex *X,
        size_t offBX,
        int incx,
        DoubleComplex *Y,
        size_t offCY,
        int incy);

//ROTG
void
blasSrotg(
        float* SA,
        size_t offSA,
        float* SB,
        size_t offSB,
        float* C,
        size_t offC,
        float* S,
        size_t offS);

void
blasDrotg(
        double* SA,
        size_t offSA,
        double* SB,
        size_t offSB,
        double* C,
        size_t offC,
        double* S,
        size_t offS);

void
blasCrotg(
        FloatComplex* SA,
        size_t offSA,
        FloatComplex* SB,
        size_t offSB,
        float* C,
        size_t offC,
        FloatComplex* S,
        size_t offS);

void
blasZrotg(
        DoubleComplex* SA,
        size_t offSA,
        DoubleComplex* SB,
        size_t offSB,
        double* C,
        size_t offC,
        DoubleComplex* S,
        size_t offS);
void
blasSrotmg(
        float *D1,
        size_t offD1,
        float *D2,
        size_t offD2,
        float *X1,
        size_t offX1,
        const float *Y1,
        size_t offY1,
        float *PARAM,
        size_t offParam);

void
blasDrotmg(
        double *D1,
        size_t offD1,
        double *D2,
        size_t offD2,
        double *X1,
        size_t offX1,
        const double *Y1,
        size_t offY1,
        double *PARAM,
        size_t offParam);

void
blasSrotm(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy,
        float *PARAM,
        size_t offParam);

void
blasDrotm(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy,
        double *PARAM,
        size_t offParam);

void
blasSrot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy,
        float C,
        float S);

void
blasDrot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy,
        double C,
        double S);

void
blasCsrot(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy,
        float C,
        float S);

void
blasZdrot(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy,
        double C,
        double S);

int
blasiSamax(
        size_t N,
        float *X,
        size_t offx,
        int incx);

int
blasiDamax(
        size_t N,
        double *X,
        size_t offx,
        int incx);

int
blasiCamax(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

int
blasiZamax(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);

float
blasSnrm2(
        size_t N,
        float *X,
        size_t offx,
        int incx);

double
blasDnrm2(
        size_t N,
        double *X,
        size_t offx,
        int incx);

float
blasScnrm2(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx);

double
blasDznrm2(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx);

#ifdef __cplusplus
}
   /* extern "C" { */
#endif

#endif  /* BLAS_INTERNAL_H_ */
