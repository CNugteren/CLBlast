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


#include <clBLAS.h>

#include <blas-internal.h>
#include <blas-wrapper.h>

void
::clMath::blas::gemv(
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
    int incy)
{
    blasSgemv(order, transA, M, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::gemv(
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
    int incy)
{
    blasDgemv(order, transA, M, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::gemv(
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
    int incy)
{
    blasCgemv(order, transA, M, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::gemv(
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
    int incy)
{
    blasZgemv(order, transA, M, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::symv(
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
    int incy)
{
    blasSsymv(order, uplo, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::symv(
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
    int incy)
{
    blasDsymv(order, uplo, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void
::clMath::blas::gemm(
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
    size_t ldc)
{
    blasSgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::gemm(
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
    size_t ldc)
{
    blasDgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::gemm(
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
    size_t ldc)
{
    blasCgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::gemm(
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
    size_t ldc)
{
    blasZgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::trmm(
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
    size_t ldb)
{
    blasStrmm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trmm(
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
    size_t ldb)
{
    blasDtrmm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trmm(
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
    size_t ldb)
{
    blasCtrmm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trmm(
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
    size_t ldb)
{
    blasZtrmm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trsm(
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
    size_t ldb)
{
    blasStrsm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trsm(
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
    size_t ldb)
{
    blasDtrsm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trsm(
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
    size_t ldb)
{
    blasCtrsm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::trsm(
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
    size_t ldb)
{
    blasZtrsm(order, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

void
::clMath::blas::syr2k(
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
    size_t ldc)
{
    blasSsyr2k(order, uplo, transA, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::syr2k(
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
    size_t ldc)
{
    blasDsyr2k(order, uplo, transA, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::syr2k(
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
    size_t ldc)
{
    blasCsyr2k(order, uplo, transA, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::syr2k(
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
    size_t ldc)
{
    blasZsyr2k(order, uplo, transA, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
::clMath::blas::syrk(
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
    size_t ldc)
{
    blasSsyrk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}

void
::clMath::blas::syrk(
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
    size_t ldc)
{
    blasDsyrk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}

void
::clMath::blas::syrk(
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
    size_t ldc)
{
    blasCsyrk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}

void
::clMath::blas::syrk(
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
    size_t ldc)
{
    blasZsyrk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}

void
::clMath::blas::trmv(
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
    int incx)
{
	blasStrmv( order, uplo, transA, diag, N, A, offa, lda, X, offx, incx );
}

void
::clMath::blas::trmv(
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
    int incx)
{
    blasDtrmv( order, uplo, transA, diag, N, A, offa, lda, X, offx, incx );
}

void
::clMath::blas::trmv(
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
    int incx)
{
    blasCtrmv( order, uplo, transA, diag, N, A, offa, lda, X, offx, incx );
}

void
::clMath::blas::trmv(
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
    int incx)
{
    blasZtrmv( order, uplo, transA, diag, N, A, offa, lda, X, offx, incx );

}

//TPMV
void
::clMath::blas::tpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    float *AP,
    size_t offa,
    float *X,
    size_t offx,
    int incx)
{
    blasStpmv( order, uplo, transA, diag, N, AP, offa, X, offx, incx );
}

void
::clMath::blas::tpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    double *AP,
    size_t offa,
    double *X,
    size_t offx,
    int incx)
{
    blasDtpmv( order, uplo, transA, diag, N, AP, offa, X, offx, incx );
}

void
::clMath::blas::tpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    FloatComplex *AP,
    size_t offa,
    FloatComplex *X,
    size_t offx,
    int incx)
{
    blasCtpmv( order, uplo, transA, diag, N, AP, offa, X, offx, incx );
}

void
::clMath::blas::tpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    DoubleComplex *AP,
    size_t offa,
    DoubleComplex *X,
    size_t offx,
    int incx)
{
    blasZtpmv( order, uplo, transA, diag, N, AP, offa, X, offx, incx );

}


void
::clMath::blas::trsv(
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
    int incx)
{
        blasStrsv( order, uplo, transA, diag, N, A,offa, lda, X,offx, incx );
}

void
::clMath::blas::trsv(
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
    int incx)
{
    blasDtrsv( order, uplo, transA, diag, N, A,offa,  lda, X,offx, incx );
}

void
::clMath::blas::trsv(
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
    int incx)
{
    blasCtrsv( order, uplo, transA, diag, N, A,offa, lda, X,offx,  incx );
}

void
::clMath::blas::trsv(
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
    int incx)
{
    blasZtrsv( order, uplo, transA, diag, N, A,offa, lda, X,offx,  incx );

}

void
::clMath::blas::tpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    float *A,
    size_t offa,
    float *X,
    size_t offx,
    int incx)
{
        blasStpsv( order, uplo, transA, diag, N, A, offa, X, offx, incx );
}

void
::clMath::blas::tpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    double *A,
    size_t offa,
    double *X,
    size_t offx,
    int incx)
{
    blasDtpsv( order, uplo, transA, diag, N, A, offa, X, offx, incx );
}

void
::clMath::blas::tpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    FloatComplex *A,
    size_t offa,
    FloatComplex *X,
    size_t offx,
    int incx)
{
    blasCtpsv( order, uplo, transA, diag, N, A, offa, X, offx, incx );
}

void
::clMath::blas::tpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    DoubleComplex *A,
    size_t offa,
    DoubleComplex *X,
    size_t offx,
    int incx)
{
    blasZtpsv( order, uplo, transA, diag, N, A, offa, X, offx, incx );

}

void
::clMath::blas::symm(
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
        size_t ldc)
{
	blasSsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::symm(
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
        size_t ldc)
{
    blasDsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::symm(
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
        FloatComplex beta,
        FloatComplex* C,
		size_t offc,
        size_t ldc)
{
    blasCsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::symm(
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
        size_t ldc)
{
    blasZsymm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::ger(
        clblasOrder order,
        size_t M,
        size_t N,
        float  alpha,
        float *x,
        size_t offx,
	int  incx,
	float *y,
        size_t offy,
	int incy,
	float *A,
	size_t offa,
        size_t lda)
{
    blasSger( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}


void
::clMath::blas::ger(
        clblasOrder order,
        size_t M,
        size_t N,
        double  alpha,
        double *x,
        size_t offx,
        int  incx,
        double *y,
        size_t offy,
        int incy,
        double *A,
        size_t offa,
        size_t lda)
{
    blasDger( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}


void
::clMath::blas::ger(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex  alpha,
      FloatComplex *x,
        size_t offx,
        int  incx,
        FloatComplex *y,
        size_t offy,
        int incy,
        FloatComplex *A,
        size_t offa,
        size_t lda)
{
    blasCgeru( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}


void
::clMath::blas::ger(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex  alpha,
        DoubleComplex *x,
        size_t offx,
        int  incx,
        DoubleComplex *y,
        size_t offy,
        int incy,
        DoubleComplex *A,
        size_t offa,
        size_t lda)
{
    blasZgeru( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}

void
::clMath::blas::gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex  alpha,
        FloatComplex *x,
        size_t offx,
        int  incx,
        FloatComplex *y,
        size_t offy,
        int incy,
        FloatComplex *A,
        size_t offa,
        size_t lda)
{
    blasCgerc( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}


void
::clMath::blas::gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex  alpha,
         DoubleComplex *x,
        size_t offx,
        int  incx,
        DoubleComplex *y,
        size_t offy,
        int incy,
        DoubleComplex *A,
        size_t offa,
        size_t lda)
{
    blasZgerc( order, M, N, alpha, x, offx, incx, y, offy, incy , A, offa, lda );
}



void
::clMath::blas::syr(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* A,
        size_t offa,
        size_t lda)
{
	blasSsyr(order, uplo, N, alpha, X, offx, incx, A, offa, lda);
}

void
::clMath::blas::syr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        double* X,
        size_t offx,
        int incx,
        double* A,
        size_t offa,
        size_t lda)
{
    blasDsyr(order, uplo, N, alpha, X, offx, incx, A, offa, lda);
}

//SPR
void
::clMath::blas::spr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        float* X,
        size_t offx,
        int incx,
        float* AP,
        size_t offa)
{
    blasSspr(order, uplo, N, alpha, X, offx, incx, AP, offa);
}

void
::clMath::blas::spr(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        double* X,
        size_t offx,
        int incx,
        double* AP,
        size_t offa)
{
    blasDspr(order, uplo, N, alpha, X, offx, incx, AP, offa);
}

void
::clMath::blas::her(
        clblasOrder order,
	clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex *x,
        size_t offx,
        int  incx,
        FloatComplex *A,
        size_t offa,
        size_t lda)
{
    blasCher( order, uplo, N, alpha, x, offx, incx, A, offa, lda );
}


void
::clMath::blas::her(
        clblasOrder order,
	clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex *x,
        size_t offx,
        int  incx,
        DoubleComplex *A,
        size_t offa,
        size_t lda)
{
    blasZher( order, uplo,  N, alpha, x, offx, incx, A, offa, lda );
}


void
::clMath::blas::syr2(
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
        size_t lda)
{
	blasSsyr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda);
}

void
::clMath::blas::syr2(
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
        size_t lda)
{
    blasDsyr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda);
}

//HER2
void
::clMath::blas::her2(
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
        size_t lda)
{
    blasCher2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda);
}

void
::clMath::blas::her2(
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
        size_t lda)
{
    blasZher2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda);
}


void
::clMath::blas::hemv(
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
        int incy)
{
    blasChemv(order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy);
}

void
::clMath::blas::hemv(
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
        int incy)
{
    blasZhemv(order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy);
}

//HEMM
void
::clMath::blas::hemm(
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
        FloatComplex beta,
        FloatComplex* C,
        size_t offc,
        size_t ldc)
{
    blasChemm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::hemm(
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
        size_t ldc)
{
    blasZhemm( order, side, uplo, M, N, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc );
}

void
::clMath::blas::herk(
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
    size_t ldc)
{
    blasCherk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}

void
::clMath::blas::herk(
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
    size_t ldc)
{
    blasZherk(order, uplo, transA, N, K, alpha, A, lda, beta, C, ldc);
}


void
::clMath::blas::spmv(
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
    int incy)
{
    blasSspmv(order, uplo, N, alpha, A, offa, X, offx, incx, beta, Y, offy, incy);
}

void
::clMath::blas::spmv(
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
    int incy)
{
    blasDspmv(order, uplo, N, alpha, A, offa, X, offx, incx, beta, Y, offy, incy);
}

void
::clMath::blas::hpmv(
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
        int incy)
{
    blasChpmv(order, uplo, N, alpha, A, offa, X, offx, incx, beta, Y, offy, incy);
}

void
::clMath::blas::hpmv(
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
        int incy)
{
    blasZhpmv(order, uplo, N, alpha, A, offa, X, offx, incx, beta, Y, offy, incy);
}

void
::clMath::blas::hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex *x,
        size_t offx,
        int  incx,
        FloatComplex *AP,
        size_t offa)
{
    blasChpr( order, uplo, N, alpha, x, offx, incx, AP, offa);
}


void
::clMath::blas::hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex *x,
        size_t offx,
        int  incx,
        DoubleComplex *AP,
        size_t offa)
{
    blasZhpr( order, uplo,  N, alpha, x, offx, incx, AP, offa );
}

void
::clMath::blas::spr2(
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
        size_t offa)
{
	blasSspr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa);
}

void
::clMath::blas::spr2(
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
        size_t offa)
{
    blasDspr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa);
}

void
::clMath::blas::hpr2(
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
        size_t offa)
{
    blasChpr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa);
}

void
::clMath::blas::hpr2(
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
        size_t offa)
{
    blasZhpr2(order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa);
}

void
clMath::blas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
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
        int incy)
        {
            return blasSgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy );
        }

void
clMath::blas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
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
        int incy)
        {
            return blasDgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy);
        }

void
clMath::blas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
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
        int incy)
        {
            return blasCgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy);
        }

void
clMath::blas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
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
        int incy)
        {
            return blasZgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy );
        }
//TBMV

void
clMath::blas::tbmv(
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
        int incx)
        {
            return blasStbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbmv(
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
        int incx)
        {
            return blasDtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbmv(
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
        int incx)
        {
            return blasCtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbmv(
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
        int incx)
        {
            return blasZtbmv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

//SBMV

void
clMath::blas::sbmv(
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
        int incy)
        {
            return blasSsbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy );
        }

void
clMath::blas::sbmv(
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
        int incy)
        {
            return blasDsbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy );
        }

//HBMV

void
clMath::blas::hbmv(
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
        int incy)
        {
            return blasChbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy );
        }

void
clMath::blas::hbmv(
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
        int incy)
        {
            return blasZhbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy );
        }

//TBSV

void
clMath::blas::tbsv(
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
        int incx)
        {
            return blasStbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbsv(
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
        int incx)
        {
            return blasDtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbsv(
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
        int incx)
        {
            return blasCtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
clMath::blas::tbsv(
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
        int incx)
        {
            return blasZtbsv( order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx );
        }

void
::clMath::blas::her2k(
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
    size_t ldc)
{
    blasCher2k(order, uplo, transA, N, K, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc);
}

void
::clMath::blas::her2k(
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
    size_t ldc)
{
    blasZher2k(order, uplo, transA, N, K, alpha, A, offa, lda, B, offb, ldb, beta, C, offc, ldc);
}

//copy
void
::clMath::blas::copy(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy)
        {

            return blasScopy( N, X, offx, incx, Y, offy, incy );
        }

void
::clMath::blas::copy(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy)
        {

            return blasDcopy( N, X, offx, incx, Y, offy, incy );
        }

void
::clMath::blas::copy(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {

            return blasCcopy( N, X, offx, incx, Y, offy, incy );
        }

void
::clMath::blas::copy(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {

            return blasZcopy( N, X, offx, incx, Y, offy, incy );
        }


//swap

void
clMath::blas::swap(
        size_t N,
        float *X,
        size_t offBX,
        int incx,
        float *Y,
        size_t offCY,
        int incy)
        {

            return blasSswap( N, X, offBX, incx, Y, offCY, incy );
        }

void
clMath::blas::swap(
        size_t N,
        double *X,
        size_t offBX,
        int incx,
        double *Y,
        size_t offCY,
        int incy)
        {

            return blasDswap( N, X, offBX, incx, Y, offCY, incy );
        }

void
clMath::blas::swap(
        size_t N,
        FloatComplex *X,
        size_t offBX,
        int incx,
        FloatComplex *Y,
        size_t offCY,
        int incy)
        {

            return blasCswap( N, X, offBX, incx, Y, offCY, incy );
        }

void
clMath::blas::swap(
        size_t N,
        DoubleComplex *X,
        size_t offBX,
        int incx,
        DoubleComplex *Y,
        size_t offCY,
        int incy)
        {

            return blasZswap( N, X, offBX, incx, Y, offCY, incy );
        }

void
::clMath::blas::scal(
        bool is_css_zds,
        size_t N,
        float alpha,
        float *X,
        size_t offx,
        int incx)
{
        is_css_zds = is_css_zds;
        return blasSscal(N, alpha, X, offx, incx);
}

void
::clMath::blas::scal(
        bool is_css_zds,
        size_t N,
        double alpha,
        double *X,
        size_t offx,
        int incx)
{
        is_css_zds = is_css_zds;    // Remove warning
        return blasDscal(N, alpha, X, offx, incx);
}

void
::clMath::blas::scal(
        bool is_css_zds,
        size_t N,
        FloatComplex alpha,
        FloatComplex *X,
        size_t offx,
        int incx)
{
    if(is_css_zds) {
        return blasCsscal(N, CREAL(alpha), X, offx, incx);
    } else {
        return blasCscal(N, alpha, X, offx, incx);
    }
}

void
::clMath::blas::scal(
        bool is_css_zds,
        size_t N,
        DoubleComplex alpha,
        DoubleComplex *X,
        size_t offx,
        int incx)
{
    if(is_css_zds) {
        return blasZdscal(N, CREAL(alpha), X, offx, incx);
    } else {
        return blasZscal(N, alpha, X, offx, incx);
    }
}

//DOT
float
clMath::blas::dot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy)
        {

            return blasSdot( N, X, offx, incx, Y, offy, incy );
        }

double
clMath::blas::dot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy)
        {

            return blasDdot( N, X, offx, incx, Y, offy, incy );
        }

FloatComplex
clMath::blas::dot(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {

            return blasCdotu( N, X, offx, incx, Y, offy, incy );
        }

DoubleComplex
clMath::blas::dot(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {

            return blasZdotu( N, X, offx, incx, Y, offy, incy );
        }

//ASUM

float
clMath::blas::asum(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {

            return blasSasum( N, X, offx, incx);
        }

double
clMath::blas::asum(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {

            return blasDasum( N, X, offx, incx);
        }

float
clMath::blas::asum(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {

            return blasScasum( N, X, offx, incx);
        }

double
clMath::blas::asum(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {

            return blasDzasum( N, X, offx, incx);
        }

//DOTC
FloatComplex
clMath::blas::dotc(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {

            return blasCdotc( N, X, offx, incx, Y, offy, incy );
        }

DoubleComplex
clMath::blas::dotc(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {

            return blasZdotc( N, X, offx, incx, Y, offy, incy );
        }




//axpy calls
void
	clMath::blas::axpy(
		size_t N,
        float alpha,
		const float * X,
		size_t offBX,
		int incx,
		float *Y,
		size_t offCY,
		int incy)
{
    return blasSaxpy(N, alpha, X, offBX, incx, Y, offCY, incy);
}

void
	clMath::blas::axpy(
		size_t N,
        double alpha,
		const double *X,
		size_t offBX,
		int incx,
		double *Y,
		size_t offCY,
		int incy)
{
    return blasDaxpy(N, alpha, X, offBX, incx, Y, offCY, incy);
}

void
	clMath::blas::axpy(
		size_t N,
        FloatComplex alpha,
		const FloatComplex *X,
		size_t offBX,
		int incx,
		FloatComplex *Y,
		size_t offCY,
		int incy)
{
    return blasCaxpy(N, alpha, X, offBX, incx, Y, offCY, incy);
}

void
	clMath::blas::axpy(
		size_t N,
        DoubleComplex alpha,
		const DoubleComplex *X,
		size_t offBX,
		int incx,
		DoubleComplex *Y,
		size_t offCY,
		int incy)
{
    return blasZaxpy(N, alpha, X, offBX, incx, Y, offCY, incy);
}

void
clMath::blas::rotg(
        float* SA,
        size_t offSA,
        float* SB,
        size_t offSB,
        float* C,
        size_t offC,
        float* S,
        size_t offS)
        {
            return blasSrotg(SA, offSA, SB, offSB, C, offC, S, offS);
        }

void
clMath::blas::rotg(
        double* SA,
        size_t offSA,
        double* SB,
        size_t offSB,
        double* C,
        size_t offC,
        double* S,
        size_t offS)
        {
            return blasDrotg(SA, offSA, SB, offSB, C, offC, S, offS);
        }

void
clMath::blas::rotg(
        FloatComplex* SA,
        size_t offSA,
        FloatComplex* SB,
        size_t offSB,
        float* C,
        size_t offC,
        FloatComplex* S,
        size_t offS)
        {
            return blasCrotg(SA, offSA, SB, offSB, C, offC, S, offS);
        }

void
clMath::blas::rotg(
        DoubleComplex* SA,
        size_t offSA,
        DoubleComplex* SB,
        size_t offSB,
        double* C,
        size_t offC,
        DoubleComplex* S,
        size_t offS)
        {
            return blasZrotg(SA, offSA, SB, offSB, C, offC, S, offS);
        }

void
clMath::blas::rotmg(
        float *D1,
        size_t offD1,
        float *D2,
        size_t offD2,
        float *X1,
        size_t offX1,
        const float *Y1,
        size_t offY1,
        float *PARAM,
        size_t offParam)
        {
            return blasSrotmg(D1, offD1, D2, offD2, X1, offX1, Y1, offY1,
                        PARAM, offParam);
        }

void
clMath::blas::rotmg(
        double *D1,
        size_t offD1,
        double *D2,
        size_t offD2,
        double *X1,
        size_t offX1,
        const double *Y1,
        size_t offY1,
        double *PARAM,
        size_t offParam)
        {
            return blasDrotmg(D1, offD1, D2, offD2, X1, offX1, Y1, offY1,
                        PARAM, offParam);
        }

void
clMath::blas::rotm(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy,
        float *PARAM,
        size_t offParam)
        {
            return blasSrotm(N, X, offx, incx, Y, offy, incy,
                     PARAM, offParam);
        }

void
clMath::blas::rotm(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy,
        double *PARAM,
        size_t offParam)
        {
            return blasDrotm(N, X, offx, incx, Y, offy, incy,
                     PARAM, offParam);
        }
//rot
void
clMath::blas::rot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy,
		float C,
		float S)
        {
            return blasSrot(N, X, offx, incx, Y, offy, incy,
                     C, S);
        }

void
clMath::blas::rot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy,
		double C,
		double S)
        {
            return blasDrot(N, X, offx, incx, Y, offy, incy,
                     C, S);
        }

void
clMath::blas::rot(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy,
        FloatComplex C,
        FloatComplex S)
        {
            return blasCsrot(N, X, offx, incx, Y, offy, incy,
                     CREAL(C), CREAL(S));
        }

void
clMath::blas::rot(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy,
        DoubleComplex C,
        DoubleComplex S)
        {
            return blasZdrot(N, X, offx, incx, Y, offy, incy,
                     CREAL(C), CREAL(S));
        }

int
clMath::blas::iamax(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {
            return blasiSamax( N, X, offx, incx );
        }

int
clMath::blas::iamax(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {

            return blasiDamax( N, X, offx, incx );
        }

int
clMath::blas::iamax(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {

            return blasiCamax( N, X, offx, incx );
        }

int
clMath::blas::iamax(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {

            return blasiZamax( N, X, offx, incx );
        }


float
clMath::blas::nrm2(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {

            return blasSnrm2( N, X, offx, incx );
        }

double
clMath::blas::nrm2(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {

            return blasDnrm2( N, X, offx, incx );
        }

float
clMath::blas::nrm2(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {

            return blasScnrm2( N, X, offx, incx );
        }

double
clMath::blas::nrm2(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {

            return blasDznrm2( N, X, offx, incx );
        }
