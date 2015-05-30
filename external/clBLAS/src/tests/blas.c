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


#include <stdlib.h>         /* abort() */
#include <stdio.h>          /* fprintf(), stderr */

#include <clBLAS.h>
#include <blas-internal.h>
#include <common.h>

#if defined CORR_TEST_WITH_ACML
#include <acml.h>
#else
#include <blas-cblas.h>
#endif

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
    int incy)
{
    char fTransA;
    int fM, fN;
    int fLDA;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;

    sgemv(fTransA, fM, fN,
        alpha, (float*)A, fLDA, (float*)X, incx, beta, Y, incy);
}

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
    int incy)
{
    char fTransA;
    int fM, fN;
    int fLDA;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;

    dgemv(fTransA, fM, fN,
        alpha, (double*)A, fLDA, (double*)X, incx, beta, Y, incy);
}

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
    int incy)
{
    char fTransA;
    int fM, fN;
    int fLDA;
    complex *fA, *fX, *fY;
    complex fAlpha, fBeta;
#if 0
    size_t sizeA, sizeX, sizeY;
    size_t i;

    sizeA = lda * N; //column major

    if (transA == clblasNoTrans) {
        sizeX = (N - 1) * abs(incx) + 1;
        sizeY = (M - 1) * abs(incy) + 1;
    } else {
        sizeX = (M - 1) * abs(incx) + 1;
        sizeY = (N - 1) * abs(incy) + 1;
    }
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_complex(CREAL(beta), CIMAG(beta));

#if 0
    fA = (complex*)calloc(sizeA, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fX = (complex*)calloc(sizeX, sizeof(complex));
    if (fX == NULL) {
        free(fA);
        return;
    }
    fY = (complex*)calloc(sizeY, sizeof(complex));
    if (fY == NULL) {
        free(fX);
        free(fA);
        return;
    }

    for (i = 0; i < sizeA; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < sizeX; i++) {
        fX[i] = compose_complex(CREAL(X[i]), CIMAG(X[i]));
    }
    for (i = 0; i < sizeY; i++) {
        fY[i] = compose_complex(CREAL(Y[i]), CIMAG(Y[i]));
    }
#else
    fA = (complex*)A;
    fX = (complex*)X;
    fY = (complex*)Y;
#endif
    cgemv(fTransA, fM, fN,
        &fAlpha, fA, fLDA, fX, incx, &fBeta, fY, incy);
#if 0
    for (i = 0; i < sizeY; i++) {
        Y[i] = floatComplex(complex_real(fY[i]), complex_imag(fY[i]));
    }
    free(fY);
    free(fX);
    free(fA);
#endif
}

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
    int incy)
{
    char fTransA;
    int fM, fN;
    int fLDA;
    doublecomplex *fA, *fX, *fY;
    doublecomplex fAlpha, fBeta;
#if 0
    size_t sizeA, sizeX, sizeY;
    size_t i;

    sizeA = lda * N; //column major

    if (transA == clblasNoTrans) {
        sizeX = (N - 1) * abs(incx) + 1;
        sizeY = (M - 1) * abs(incy) + 1;
    } else {
        sizeX = (M - 1) * abs(incx) + 1;
        sizeY = (N - 1) * abs(incy) + 1;
    }
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (doublecomplex*)calloc(sizeA, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fX = (doublecomplex*)calloc(sizeX, sizeof(doublecomplex));
    if (fX == NULL) {
        free(fA);
        return;
    }
    fY = (doublecomplex*)calloc(sizeY, sizeof(doublecomplex));
    if (fY == NULL) {
        free(fX);
        free(fA);
        return;
    }

    for (i = 0; i < sizeA; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < sizeX; i++) {
        fX[i] = compose_doublecomplex(CREAL(X[i]), CIMAG(X[i]));
    }
    for (i = 0; i < sizeY; i++) {
        fY[i] = compose_doublecomplex(CREAL(Y[i]), CIMAG(Y[i]));
    }
#else
    fA = (doublecomplex*)A;
    fX = (doublecomplex*)X;
    fY = (doublecomplex*)Y;
#endif
    zgemv(fTransA, fM, fN,
        &fAlpha, fA, fLDA, fX, incx, &fBeta, fY, incy);
#if 0
    for (i = 0; i < sizeY; i++) {
        Y[i] = doubleComplex(
            doublecomplex_real(fY[i]),
            doublecomplex_imag(fY[i]));
    }
    free(fY);
    free(fX);
    free(fA);
#endif
}

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
    int incy)
{
    char fUplo;
    int fN;
    int fLDA;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fN = (int)N;
    fLDA = (int)lda;

    ssymv(fUplo, fN, alpha, (float*)A, fLDA, (float*)X, incx, beta, Y, incy);
}

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
    int incy)
{
    char fUplo;
    int fN;
    int fLDA;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fN = (int)N;
    fLDA = (int)lda;

    dsymv(fUplo, fN, alpha, (double*)A, fLDA, (double*)X, incx, beta, Y, incy);
}

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
    size_t ldc)
{
    char fTransA, fTransB;
    int fM, fN, fK;
    int fLDA, fLDB, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fTransB = encodeTranspose(transB);
    fM = (int)M;
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    sgemm(fTransA, fTransB, fM, fN, fK,
        alpha, (float*)A, fLDA, (float*)B, fLDB, beta, C, fLDC);
}

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
    size_t ldc)
{
    char fTransA, fTransB;
    int fM, fN, fK;
    int fLDA, fLDB, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fTransB = encodeTranspose(transB);
    fM = (int)M;
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    dgemm(fTransA, fTransB, fM, fN, fK,
        alpha, (double*)A, fLDA, (double*)B, fLDB, beta, C, fLDC);
}

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
    size_t ldc)
{
    char fTransA, fTransB;
    int fM, fN, fK;
    int fLDA, fLDB, fLDC;
    complex *fA, *fB, *fC;
    complex fAlpha, fBeta;
#if 0
    size_t ma, ka, nb, kb, mc, nc;
    size_t i;

    if (transA == clblasNoTrans) {
        ma = lda;
        ka = K;
    } else {
        ka = lda;
        ma = M;
    }
    if (transB == clblasNoTrans) {
        kb = ldb;
        nb = N;
    } else {
        nb = ldb;
        kb = K;
    }
    mc = ldc;
    nc = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fTransB = encodeTranspose(transB);
    fM = (int)M;
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_complex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (complex*)calloc(ma * ka, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fB = (complex*)calloc(kb * nb, sizeof(complex));
    if (fB == NULL) {
        free(fA);
        return;
    }
    fC = (complex*)calloc(mc * nc, sizeof(complex));
    if (fC == NULL) {
        free(fB);
        free(fA);
        return;
    }

    for (i = 0; i < ma * ka; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < kb * nb; i++) {
        fB[i] = compose_complex(CREAL(B[i]), CIMAG(B[i]));
    }
    for (i = 0; i < mc * nc; i++) {
        fC[i] = compose_complex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (complex*)A;
    fB = (complex*)B;
    fC = (complex*)C;
#endif
    cgemm(fTransA, fTransB, fM, fN, fK,
        &fAlpha, fA, fLDA, fB, fLDB, &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < mc * nc; i++) {
        C[i] = floatComplex(complex_real(fC[i]), complex_imag(fC[i]));
    }
    free(fC);
    free(fB);
    free(fA);
#endif
}

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
    size_t ldc)
{
    char fTransA, fTransB;
    int fM, fN, fK;
    int fLDA, fLDB, fLDC;
    doublecomplex *fA, *fB, *fC;
    doublecomplex fAlpha, fBeta;
#if 0
    size_t ma, ka, nb, kb, mc, nc;
    size_t i;

    if (transA == clblasNoTrans) {
        ma = lda;
        ka = K;
    } else {
        ka = lda;
        ma = M;
    }
    if (transB == clblasNoTrans) {
        kb = ldb;
        nb = N;
    } else {
        nb = ldb;
        kb = K;
    }
    mc = ldc;
    nc = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fTransA = encodeTranspose(transA);
    fTransB = encodeTranspose(transB);
    fM = (int)M;
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (doublecomplex*)calloc(ma * ka, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fB = (doublecomplex*)calloc(kb * nb, sizeof(doublecomplex));
    if (fB == NULL) {
        free(fA);
        return;
    }
    fC = (doublecomplex*)calloc(mc * nc, sizeof(doublecomplex));
    if (fC == NULL) {
        free(fB);
        free(fA);
        return;
    }

    for (i = 0; i < ma * ka; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < kb * nb; i++) {
        fB[i] = compose_doublecomplex(CREAL(B[i]), CIMAG(B[i]));
    }
    for (i = 0; i < mc * nc; i++) {
        fC[i] = compose_doublecomplex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (doublecomplex*)A;
    fB = (doublecomplex*)B;
    fC = (doublecomplex*)C;
#endif
    zgemm(fTransA, fTransB, fM, fN, fK,
        &fAlpha, fA, fLDA, fB, fLDB, &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < mc * nc; i++) {
        C[i] = doubleComplex(doublecomplex_real(fC[i]), doublecomplex_imag(fC[i]));
    }
    free(fC);
    free(fB);
    free(fA);
#endif
}

void blasStrmm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    strmm(fSide, fUplo, fTransA, fDiag, fM, fN,
        alpha, (float*)A, fLDA, B, fLDB);
}

void blasDtrmm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    dtrmm(fSide, fUplo, fTransA, fDiag, fM, fN,
        alpha, (double*)A, fLDA, B, fLDB);
}

void blasCtrmm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;
    complex *fA, *fB;
    complex fAlpha;
#if 0
    size_t ma, na, mb, nb;
    size_t i;

    ma = lda;
    if (side == clblasLeft) {
        na = M;
    }
    else {
        na = N;
    }
    mb = ldb;
    nb = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
#if 0
    fA = (complex*)calloc(ma * na, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fB = (complex*)calloc(mb * nb, sizeof(complex));
    if (fB == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < ma * na; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < mb * nb; i++) {
        fB[i] = compose_complex(CREAL(B[i]), CIMAG(B[i]));
    }
#else
    fA = (complex*)A;
    fB = (complex*)B;
#endif
    ctrmm(fSide, fUplo, fTransA, fDiag, fM, fN,
        &fAlpha, fA, fLDA, fB, fLDB);
#if 0
    for (i = 0; i < mb * nb; i++) {
        B[i] = floatComplex(complex_real(fB[i]), complex_imag(fB[i]));
    }
    free(fB);
    free(fA);
#endif
}

void blasZtrmm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;
    doublecomplex *fA, *fB;
    doublecomplex fAlpha;
#if 0
    size_t ma, na, mb, nb;
    size_t i;

    ma = lda;
    if (side == clblasLeft) {
        na = M;
    }
    else {
        na = N;
    }
    mb = ldb;
    nb = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
#if 0
    fA = (doublecomplex*)calloc(ma * na, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fB = (doublecomplex*)calloc(mb * nb, sizeof(doublecomplex));
    if (fB == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < ma * na; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < mb * nb; i++) {
        fB[i] = compose_doublecomplex(CREAL(B[i]), CIMAG(B[i]));
    }
#else
    fA = (doublecomplex*)A;
    fB = (doublecomplex*)B;
#endif
    ztrmm(fSide, fUplo, fTransA, fDiag, fM, fN,
        &fAlpha, fA, fLDA, fB, fLDB);
#if 0
    for (i = 0; i < mb * nb; i++) {
        B[i] = doubleComplex(doublecomplex_real(fB[i]), doublecomplex_imag(fB[i]));
    }
    free(fB);
    free(fA);
#endif
}

void blasStrsm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    strsm(fSide, fUplo, fTransA, fDiag, fM, fN,
        alpha, (float*)A, fLDA, B, fLDB);
}

void blasDtrsm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    dtrsm(fSide, fUplo, fTransA, fDiag, fM, fN,
        alpha, (double*)A, fLDA, B, fLDB);
}

void blasCtrsm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;
    complex *fA, *fB;
    complex fAlpha;
#if 0
    size_t ma, na, mb, nb;
    size_t i;

    ma = lda;
    if (side == clblasLeft) {
        na = M;
    }
    else {
        na = N;
    }
    mb = ldb;
    nb = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
#if 0
    fA = (complex*)calloc(ma * na, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fB = (complex*)calloc(mb * nb, sizeof(complex));
    if (fB == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < ma * na; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < mb * nb; i++) {
        fB[i] = compose_complex(CREAL(B[i]), CIMAG(B[i]));
    }
#else
    fA = (complex*)A;
    fB = (complex*)B;
#endif
    ctrsm(fSide, fUplo, fTransA, fDiag, fM, fN,
        &fAlpha, fA, fLDA, fB, fLDB);
#if 0
    for (i = 0; i < mb * nb; i++) {
        B[i] = floatComplex(complex_real(fB[i]), complex_imag(fB[i]));
    }
    free(fB);
    free(fA);
#endif
}

void blasZtrsm(
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
    char fSide, fUplo, fTransA, fDiag;
    int fM, fN;
    int fLDA, fLDB;
    doublecomplex *fA, *fB;
    doublecomplex fAlpha;
#if 0
    size_t ma, na, mb, nb;
    size_t i;

    ma = lda;
    if (side == clblasLeft) {
        na = M;
    }
    else {
        na = N;
    }
    mb = ldb;
    nb = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fSide = encodeSide(side);
    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fDiag = encodeDiag(diag);
    fM = (int)M;
    fN = (int)N;
    fLDA = (int)lda;
    fLDB = (int)ldb;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
#if 0
    fA = (doublecomplex*)calloc(ma * na, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fB = (doublecomplex*)calloc(mb * nb, sizeof(doublecomplex));
    if (fB == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < ma * na; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < mb * nb; i++) {
        fB[i] = compose_doublecomplex(CREAL(B[i]), CIMAG(B[i]));
    }
#else
    fA = (doublecomplex*)A;
    fB = (doublecomplex*)B;
#endif
    ztrsm(fSide, fUplo, fTransA, fDiag, fM, fN,
        &fAlpha, fA, fLDA, fB, fLDB);
#if 0
    for (i = 0; i < mb * nb; i++) {
        B[i] = doubleComplex(doublecomplex_real(fB[i]), doublecomplex_imag(fB[i]));
    }
    free(fB);
    free(fA);
#endif
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDB, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    ssyr2k(fUplo, fTransA, fN, fK, alpha, (float*)A, fLDA, (float*)B, fLDB,
        beta, C, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDB, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    dsyr2k(fUplo, fTransA, fN, fK, alpha, (double*)A, fLDA, (double*)B, fLDB,
        beta, C, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDB, fLDC;
    complex *fA, *fB, *fC;
    complex fAlpha, fBeta;
#if 0
    size_t na, ka, nb, kb, rowsC, columnsC;
    size_t i;

    if (transA == clblasNoTrans) {
        na = lda;
        ka = K;
        nb = ldb;
        kb = K;
    }
    else {
        ka = lda;
        na = N;
        kb = ldb;
        nb = N;
    }
    rowsC = ldc;
    columnsC = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_complex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (complex*)calloc(na * ka, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fB = (complex*)calloc(nb * kb, sizeof(complex));
    if (fB == NULL) {
        free(fA);
        return;
    }
    fC = (complex*)calloc(rowsC * columnsC, sizeof(complex));
    if (fC == NULL) {
        free(fB);
        free(fA);
        return;
    }

    for (i = 0; i < na * ka; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < nb * kb; i++) {
        fB[i] = compose_complex(CREAL(B[i]), CIMAG(B[i]));
    }
    for (i = 0; i < rowsC * columnsC; i++) {
        fC[i] = compose_complex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (complex*)A;
    fB = (complex*)B;
    fC = (complex*)C;
#endif
    csyr2k(fUplo, fTransA, fN, fK, &fAlpha, fA, fLDA,
        fB, fLDB, &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < rowsC * columnsC; i++) {
        C[i] = floatComplex(complex_real(fC[i]), complex_imag(fC[i]));
    }
    free(fC);
    free(fB);
    free(fA);
#endif
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDB, fLDC;
    doublecomplex *fA, *fB, *fC;
    doublecomplex fAlpha, fBeta;
#if 0
    size_t na, ka, nb, kb, rowsC, columnsC;
    size_t i;


    if (transA == clblasNoTrans) {
        na = lda;
        ka = K;
        nb = ldb;
        kb = K;
    }
    else {
        ka = lda;
        na = N;
        kb = ldb;
        nb = N;
    }
    rowsC = ldc;
    columnsC = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (doublecomplex*)calloc(na * ka, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fB = (doublecomplex*)calloc(nb * kb, sizeof(doublecomplex));
    if (fB == NULL) {
        free(fA);
        return;
    }
    fC = (doublecomplex*)calloc(rowsC * columnsC, sizeof(doublecomplex));
    if (fC == NULL) {
        free(fB);
        free(fA);
        return;
    }

    for (i = 0; i < na * ka; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < nb * kb; i++) {
        fB[i] = compose_doublecomplex(CREAL(B[i]), CIMAG(B[i]));
    }
    for (i = 0; i < rowsC * columnsC; i++) {
        fC[i] = compose_doublecomplex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (doublecomplex*)A;
    fB = (doublecomplex*)B;
    fC = (doublecomplex*)C;
#endif
    zsyr2k(fUplo, fTransA, fN, fK, &fAlpha, fA, fLDA,
        fB, fLDB, &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < rowsC * columnsC; i++) {
        C[i] = doubleComplex(doublecomplex_real(fC[i]), doublecomplex_imag(fC[i]));
    }
    free(fC);
    free(fB);
    free(fA);
#endif
}


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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    ssyrk(fUplo, fTransA, fN, fK, alpha, (float*)A, fLDA,
        beta, C, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    dsyrk(fUplo, fTransA, fN, fK, alpha, (double*)A, fLDA,
        beta, C, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;
    complex *fA, *fC;
    complex fAlpha, fBeta;
#if 0
    size_t i;
    size_t na, ka, rowsC, columnsC;
    if (transA == clblasNoTrans) {
        na = lda;
        ka = K;
    }
    else {
        ka = lda;
        na = N;
    }
    rowsC = ldc;
    columnsC = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_complex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (complex*)calloc(na * ka, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fC = (complex*)calloc(rowsC * columnsC, sizeof(complex));
    if (fC == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < na * ka; i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < rowsC * columnsC; i++) {
        fC[i] = compose_complex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (complex*)A;
    fC = (complex*)C;
#endif
    csyrk(fUplo, fTransA, fN, fK, &fAlpha, fA, fLDA,
        &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < rowsC * columnsC; i++) {
        C[i] = floatComplex(complex_real(fC[i]), complex_imag(fC[i]));
    }
    free(fC);
    free(fA);
#endif
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;
    doublecomplex *fA, *fC;
    doublecomplex fAlpha, fBeta;
#if 0
    size_t na, ka, rowsC, columnsC;
    size_t i;

    if (transA == clblasNoTrans) {
        na = lda;
        ka = K;
    }
    else {
        ka = lda;
        na = N;
    }
    rowsC = ldc;
    columnsC = N;
#endif

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
    fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));
#if 0
    fA = (doublecomplex*)calloc(na * ka, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fC = (doublecomplex*)calloc(rowsC * columnsC, sizeof(doublecomplex));
    if (fC == NULL) {
        free(fA);
        return;
    }

    for (i = 0; i < na * ka; i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < rowsC * columnsC; i++) {
        fC[i] = compose_doublecomplex(CREAL(C[i]), CIMAG(C[i]));
    }
#else
    fA = (doublecomplex*)A;
    fC = (doublecomplex*)C;
#endif
    zsyrk(fUplo, fTransA, fN, fK, &fAlpha, fA, fLDA,
        &fBeta, fC, fLDC);
#if 0
    for (i = 0; i < rowsC * columnsC; i++) {
        C[i] = doubleComplex(doublecomplex_real(fC[i]), doublecomplex_imag(fC[i]));
    }
    free(fC);
    free(fA);
#endif
}

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
        int incx)
{
	char fUplo, fDiag, fTrans;
	int fN, fLda;

	fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

	if (order != clblasColumnMajor)
    {
		fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
	}

	fN = (int)N;
	fLda = (int)lda;

	strmv( fUplo, fTrans, fDiag, fN, A+offa, fLda, X+offx, incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

	if (order != clblasColumnMajor)
    {
    	fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
	}


    fN = (int)N;
    fLda = (int)lda;

    dtrmv( fUplo, fTrans, fDiag, fN, A+offa , fLda, X+offx,  incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;
	complex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

	if (order != clblasColumnMajor)
    {
    	fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
	}


    fN = (int)N;
    fLda = (int)lda;
	fA = (complex*) A + offa;
	fX = (complex*) X + offx;

    ctrmv( fUplo, fTrans, fDiag, fN, fA, fLda, fX, incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;
    doublecomplex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

	if (order != clblasColumnMajor)
    {
		fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }


    fN = (int)N;
    fLda = (int)lda;

    fA = (doublecomplex*)A + offa;
    fX = (doublecomplex*)X + offx;
    ztrmv( fUplo, fTrans, fDiag, fN, fA, fLda, fX, incx );
}

//TPMV

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fN = (int)N;

    stpmv( fUplo, fTrans, fDiag, fN, AP+offa, X+offx, incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }


    fN = (int)N;

    dtpmv( fUplo, fTrans, fDiag, fN, AP+offa , X+offx,  incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;
    complex *fAP, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }


    fN = (int)N;
    fAP = (complex*) AP + offa;
    fX = (complex*) X + offx;

    ctpmv( fUplo, fTrans, fDiag, fN, fAP, fX, incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;
    doublecomplex *fAP, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
      abort();
    }


    fN = (int)N;

    fAP = (doublecomplex*)AP + offa;
    fX = (doublecomplex*)X + offx;
    ztpmv( fUplo, fTrans, fDiag, fN, fAP, fX, incx );
}


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
        int incx)
{
        char fUplo, fDiag, fTrans;
        int fN, fLda;

        fUplo = encodeUplo(uplo);
        fTrans = encodeTranspose(transA);
        fDiag = encodeDiag(diag);


	if (order != clblasColumnMajor)
    	{
			fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    	}


        fN = (int)N;
        fLda = (int)lda;

        strsv( fUplo, fTrans, fDiag, fN, (A+offa), fLda, (X+offx), incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor) {
		fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fN = (int)N;
    fLda = (int)lda;

    dtrsv( fUplo, fTrans, fDiag, fN, (A+offa), fLda, (X+offx),  incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;
        complex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
		fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }




    fN = (int)N;
    fLda = (int)lda;

#if 0
    fA = (complex*)calloc(N * lda, sizeof(complex));
    if (fA == NULL) {
        return;
    }
    fX = (complex*)calloc(1 + ((N-1)* abs(incx)), sizeof(complex));
    if (fX == NULL) {
        free(fA);
        return;
    }


    for (i = 0; i < (N * lda); i++) {
        fA[i] = compose_complex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < (1 +((N-1)* abs(incx))); i++) {
        fX[i] = compose_complex(CREAL(X[i]), CIMAG(X[i]));
    }
#else
    fA = (complex*)A;
    fX = (complex*)X;
#endif
    ctrsv(fUplo, fTrans,fDiag, fN,fA+offa, fLda,
         fX+offx, incx);
#if 0
    for (i = 0; i < (1 +((N-1)* abs(incx))); i++) {
        X[i] = floatComplex(complex_real(fX[i]), complex_imag(fX[i]));
    }
    free(fX);
    free(fA);
#endif

}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN, fLda;
    doublecomplex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor) {
		fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fN = (int)N;
    fLda = (int)lda;
#if 0
    fA = (doublecomplex*)calloc(N * lda, sizeof(doublecomplex));
    if (fA == NULL) {
        return;
    }
    fX = (doublecomplex*)calloc((1 + ((N-1) * abs(incx))), sizeof(doublecomplex));
    if (fX == NULL) {
        free(fX);
        return;
    }

    for (i = 0; i < (N * lda); i++) {
        fA[i] = compose_doublecomplex(CREAL(A[i]), CIMAG(A[i]));
    }
    for (i = 0; i < (1 + ((N-1) * abs(incx))); i++) {
        fX[i] = compose_doublecomplex(CREAL(X[i]), CIMAG(X[i]));
    }
#else
    fA = (doublecomplex*)A;
    fX = (doublecomplex*)X;
#endif
    ztrsv( fUplo, fTrans, fDiag, fN, fA + offa, fLda, fX + offx, incx );
#if 0
    for (i = 0; i < ((1 + ((N-1) * abs(incx))); i++) {
        X[i] = doubleComplex(doublecomplex_real(fX[i]), doublecomplex_imag(fX[i]));
    }
    free(fX);
    free(fA);
#endif

}

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
        int incx)
{
        char fUplo, fDiag, fTrans;
        int fN;

        fUplo = encodeUplo(uplo);
        fTrans = encodeTranspose(transA);
        fDiag = encodeDiag(diag);


        if (order != clblasColumnMajor)
        {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
        }

        fN = (int)N;
        stpsv( fUplo, fTrans, fDiag, fN, (A+offa), (X+offx), incx );
}

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
        int incx)
{
        char fUplo, fDiag, fTrans;
        int fN;

        fUplo = encodeUplo(uplo);
        fTrans = encodeTranspose(transA);
        fDiag = encodeDiag(diag);


        if (order != clblasColumnMajor)
        {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
        }

        fN = (int)N;
        dtpsv( fUplo, fTrans, fDiag, fN, (A+offa), (X+offx), incx );
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;
    complex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }
    fN = (int)N;

    fA = (complex*)A;
    fX = (complex*)X;

    ctpsv(fUplo, fTrans,fDiag, fN,fA+offa, fX+offx, incx);
}

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
        int incx)
{
    char fUplo, fDiag, fTrans;
    int fN;
    doublecomplex *fA, *fX;

    fUplo = encodeUplo(uplo);
    fTrans = encodeTranspose(transA);
    fDiag = encodeDiag(diag);

    if (order != clblasColumnMajor)
    {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }
    fN = (int)N;

    fA = (doublecomplex*)A;
    fX = (doublecomplex*)X;

    ztpsv(fUplo, fTrans,fDiag, fN,fA+offa, fX+offx, incx);
}


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
        size_t ldc)
		{

			char fSide, fUplo;
			int fM, fN, fLda, fLdb, fLdc;

			fSide = encodeSide( side );
			fUplo = encodeUplo( uplo );

			fM = (int) M;
			fN = (int) N;
			fLda= (int) lda;
			fLdb = (int) ldb;
			fLdc = (int) ldc;

			if (order != clblasColumnMajor) {

			fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
	        abort();

			}

			ssymm( fSide, fUplo, fM, fN, alpha, (A+offa), fLda, (B+offb), fLdb, beta, (C+offc), fLdc );

		}

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
        size_t ldc)
        {

            char fSide, fUplo;
            int fM, fN, fLda, fLdb, fLdc;

            fSide = encodeSide( side );
            fUplo = encodeUplo( uplo );

            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fLdb = (int) ldb;
            fLdc = (int) ldc;

			if (order != clblasColumnMajor) {

            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
	        abort();

			}

            dsymm( fSide, fUplo, fM, fN, alpha, (A+offa), fLda, (B+offb), fLdb, beta, (C+offc), fLdc );

        }

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
        FloatComplex beta,
        FloatComplex* C,
        size_t offc,
        size_t ldc)
        {

            char fSide, fUplo;
            int fM, fN, fLda, fLdb, fLdc;
			complex *fA, *fB, *fC, fAlpha, fBeta;

            fSide = encodeSide( side );
            fUplo = encodeUplo( uplo );

            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fLdb = (int) ldb;
            fLdc = (int) ldc;
			fA = (complex*) A;
			fB = (complex*) B;
			fC = (complex*) C;
			fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
			fBeta = compose_complex(CREAL(beta), CIMAG(beta));

			if (order != clblasColumnMajor) {

            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
	        abort();

			}

            csymm( fSide, fUplo, fM, fN, &fAlpha, (fA+offa), fLda, (fB+offb), fLdb, &fBeta, (fC+offc), fLdc );

        }

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
        size_t ldc)
        {

            char fSide, fUplo;
            int fM, fN, fLda, fLdb, fLdc;
			doublecomplex *fA, *fB, *fC, fAlpha, fBeta;

            fSide = encodeSide( side );
            fUplo = encodeUplo( uplo );

            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fLdb = (int) ldb;
            fLdc = (int) ldc;
			fA =(doublecomplex*) A;
			fB =(doublecomplex*) B;
			fC =(doublecomplex*) C;

			fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            fBeta  = compose_doublecomplex(CREAL(beta), CIMAG(beta));

			if (order != clblasColumnMajor) {

            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
	        abort();

			}

            zsymm( fSide, fUplo, fM, fN, &fAlpha, (fA+offa), fLda, (fB+offb), fLdb, &fBeta, (fC+offc), fLdc );

        }


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
        size_t lda)
                {

                        int fM, fN, fLda;

                        fM = (int) M;
                        fN = (int) N;
                        fLda= (int) lda;

                        if (order != clblasColumnMajor) {
                        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
                        }
                        sger( fM, fN, alpha, (x+offx), incx, (y+offy), incy, (A+offa), fLda );

                }


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
        size_t lda)
        {

            int fM, fN, fLda;
            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;

                        if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
                        }
            dger( fM, fN, alpha, (x+offx), incx, (y+offy), incy, (A+offa), fLda );

        }


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
        size_t lda)
        {

            int fM, fN, fLda;
            complex *fA, *fx, *fy, fAlpha;
            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fA = (complex*) A;
 	    fx = (complex*) x;
	    fy = (complex*) y;
            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            cgeru( fM, fN, &fAlpha, (fx+offx), incx, (fy+offy), incy, (fA+offa), fLda );

        }


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
        size_t lda)
        {

            int fM, fN, fLda;
            doublecomplex *fA, *fx, *fy, fAlpha;
            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fA =(doublecomplex*) A;
            fx =(doublecomplex*) x;
            fy =(doublecomplex*) y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
 }

            zgeru( fM, fN, &fAlpha, (fx+offx), incx, (fy+offy), incy, (fA+offa), fLda );
        }


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
        size_t lda)
        {

            int fM, fN, fLda;
            complex *fA, *fx, *fy, fAlpha;
            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fA = (complex*) A;
            fx = (complex*) x;
            fy = (complex*) y;
            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            cgerc( fM, fN, &fAlpha, (fx+offx), incx, (fy+offy), incy, (fA+offa), fLda );

        }

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
        size_t lda)
        {

            int fM, fN, fLda;
            doublecomplex *fA, *fx, *fy, fAlpha;
            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fA =(doublecomplex*) A;
            fx =(doublecomplex*) x;
            fy =(doublecomplex*) y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
 }
            zgerc( fM, fN, &fAlpha, (fx+offx), incx, (fy+offy), incy, (fA+offa), fLda );

        }

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
		size_t lda)
		{
			char fUplo;
            int fN, fLda, fIncx;
			float *fA, fAlpha, *fX;

            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fLda  = (int) lda;
			fIncx = (int) incx;
			fA = (float*) A;
			fX = (float*) X;
			fAlpha = alpha;

			if (order != clblasColumnMajor)
			{
    	        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
		        abort();
			}
			ssyr(fUplo, fN, fAlpha, (fX + offx), fIncx, (fA + offa), fLda);
		}


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
        size_t lda)
        {
            char fUplo;
            int fN, fLda, fIncx;
            double *fA, fAlpha, *fX;

            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fLda  = (int) lda;
            fIncx = (int) incx;
            fA = (double*) A;
            fX = (double*) X;
            fAlpha = alpha;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
	    dsyr(fUplo, fN, fAlpha, (fX + offx), fIncx, (fA + offa), fLda);


}

//SPR

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
        size_t offa)
        {
            char fUplo;
            int fN, fIncx;
            float *fAP, fAlpha, *fX;

            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fIncx = (int) incx;
            fAP = (float*) AP;
            fX = (float*) X;

            fAlpha = alpha;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            sspr(fUplo, fN, fAlpha, (fX + offx), fIncx, (fAP + offa));
        }


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
        size_t offa)
        {
            char fUplo;
            int fN, fIncx;
            double *fAP, fAlpha, *fX;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fIncx = (int) incx;
            fAP = (double*) AP;
            fX = (double*) X;
            fAlpha = alpha;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
        dspr(fUplo, fN, fAlpha, (fX + offx), fIncx, (fAP + offa));


}


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
        size_t lda)
        {
	    	char fUplo;
            int fN, fLda;
            complex *fA, *fx ;
			fUplo = encodeUplo( uplo );
            fN = (int) N;
            fLda= (int) lda;
            fA = (complex*) A;
            fx = (complex*) x;

            if (order != clblasColumnMajor) {

            	fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
			}
            cher( fUplo, fN, alpha, (fx+offx), incx, (fA+offa), fLda );

        }


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
        size_t lda)
        {
            char fUplo;
            int  fN, fLda;
            doublecomplex *fA, *fx;
			fUplo = encodeUplo( uplo );
            fN = (int) N;
            fLda= (int) lda;
            fA =(doublecomplex*) A;
            fx =(doublecomplex*) x;

            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
 }

            zher( fUplo, fN, alpha, (fx+offx), incx, (fA+offa), fLda );
        }

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
		size_t lda)
		{
			char fUplo;
            int fN, fLda, fIncx, fIncy;
			float *fA, fAlpha, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fLda  = (int) lda;
			fIncx = (int) incx;
			fIncy = (int) incy;

			fA = (float*) A;
			fX = (float*) X;
			fY = (float*) Y;

			fAlpha = alpha;

			if (order != clblasColumnMajor)
			{
    	        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
		        abort();
			}

			ssyr2(fUplo, fN, fAlpha, (fX + offx), fIncx, (fY + offy), fIncy, (fA + offa), fLda);
		}


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
        size_t lda)
        {
            char fUplo;
            int fN, fLda, fIncx, fIncy;
            double *fA, fAlpha, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fLda  = (int) lda;
            fIncx = (int) incx;
			fIncy = (int) incy;

            fA = (double*) A;
            fX = (double*) X;
			fY = (double*) Y;

            fAlpha = alpha;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dsyr2(fUplo, fN, fAlpha, (fX + offx), fIncx, (fY + offy), fIncy, (fA + offa), fLda);
        }

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
        size_t lda)
        {
            char fUplo;
            int fN, fLda;
            complex *fA, fAlpha, *fX, *fY;
            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fLda  = (int) lda;
            fA = (complex*) A;
            fX = (complex*) X;
            fY = (complex*) Y;

	    fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            cher2(fUplo, fN, &fAlpha, (fX + offx), incx, (fY + offy), incy, (fA + offa), fLda);
        }

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
        size_t lda)
        {
            char fUplo;
            int fN, fLda ;
            doublecomplex *fA, fAlpha, *fX, *fY;
            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fLda  = (int) lda;
            fA = (doublecomplex*) A;
            fX = (doublecomplex*) X;
            fY = (doublecomplex*) Y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            zher2(fUplo, fN, &fAlpha, (fX + offx), incx, (fY + offy), incy, (fA + offa), fLda);
        }



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
        int incy)
        {
            char fUplo;
            int fN, fLda, fIncx, fIncy;
            complex *fA, fAlpha, fBeta, *fX, *fY;

		    fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fLda  = (int) lda;
            fIncx = (int) incx;
            fIncy = (int) incy;

		    fA = (complex*) A;
            fX = (complex*) X;
            fY = (complex*) Y;

            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_complex(CREAL(beta), CIMAG(beta));
            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            chemv(fUplo, fN, &fAlpha, (fA + offa), fLda, (fX + offx), fIncx, &fBeta, (fY + offy), fIncy);
        }

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
        int incy)
        {
            char fUplo;
            int fN, fLda, fIncx, fIncy;
            doublecomplex *fA, fAlpha, fBeta, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fLda  = (int) lda;
            fIncx = (int) incx;
            fIncy = (int) incy;

            fA = (doublecomplex*) A;
            fX = (doublecomplex*) X;
            fY = (doublecomplex*) Y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            zhemv(fUplo, fN, &fAlpha, (fA + offa), fLda, (fX + offx), fIncx, &fBeta, (fY + offy), fIncy);
        }

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
        FloatComplex beta,
        FloatComplex* C,
        size_t offc,
        size_t ldc)
        {

            char fSide, fUplo;
            int fM, fN, fLda, fLdb, fLdc;
            complex *fA, *fB, *fC, fAlpha, fBeta;

            fSide = encodeSide( side );
            fUplo = encodeUplo( uplo );

            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fLdb = (int) ldb;
            fLdc = (int) ldc;
            fA = (complex*) A;
            fB = (complex*) B;
            fC = (complex*) C;
            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_complex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor) {

            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
						    }

            chemm( fSide, fUplo, fM, fN, &fAlpha, (fA+offa), fLda, (fB+offb), fLdb, &fBeta, (fC+offc), fLdc );

        }

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
        size_t ldc)
        {

            char fSide, fUplo;
            int fM, fN, fLda, fLdb, fLdc;
            doublecomplex *fA, *fB, *fC, fAlpha, fBeta;

            fSide = encodeSide( side );
            fUplo = encodeUplo( uplo );

            fM = (int) M;
            fN = (int) N;
            fLda= (int) lda;
            fLdb = (int) ldb;
            fLdc = (int) ldc;
            fA =(doublecomplex*) A;
            fB =(doublecomplex*) B;
            fC =(doublecomplex*) C;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
			fBeta  = compose_doublecomplex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor) {

            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();

                        }

            zhemm( fSide, fUplo, fM, fN, &fAlpha, (fA+offa), fLda, (fB+offb), fLdb, &fBeta, (fC+offc), fLdc );

        }


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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;
    complex *fA, *fC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    fA = (complex*)A;
    fC = (complex*)C;

	cherk(fUplo, fTransA, fN, fK, alpha, fA, fLDA, beta, fC, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC;
    doublecomplex *fA, *fC;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDC = (int)ldc;

    fA = (doublecomplex*)A;
    fC = (doublecomplex*)C;

	zherk(fUplo, fTransA, fN, fK, alpha, fA, fLDA, beta, fC, fLDC);
}


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
    int incy)
{
    char fUplo;
    int fN;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fN = (int)N;

    sspmv(fUplo, fN, alpha, (float*)(A+offa), (float*)(X+offx), incx, beta, (Y+offy), incy);
}

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
    int incy)
{
    char fUplo;
    int fN;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fN = (int)N;

    dspmv(fUplo, fN, alpha, (double*)(A+offa),(double*)(X+offx), incx, beta, (Y+offy), incy);
}

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
        int incy)
        {
            char fUplo;
            int fN, fIncx, fIncy;
            complex *fA, fAlpha, fBeta, *fX, *fY;

		    fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fIncx = (int) incx;
            fIncy = (int) incy;

		    fA = (complex*) A;
            fX = (complex*) X;
            fY = (complex*) Y;

            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_complex(CREAL(beta), CIMAG(beta));
            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            chpmv(fUplo, fN, &fAlpha, (fA + offa), (fX + offx), fIncx, &fBeta, (fY + offy), fIncy);
        }

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
        int incy)
        {
            char fUplo;
            int fN, fIncx, fIncy;
            doublecomplex *fA, fAlpha, fBeta, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fIncx = (int) incx;
            fIncy = (int) incy;

            fA = (doublecomplex*) A;
            fX = (doublecomplex*) X;
            fY = (doublecomplex*) Y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            zhpmv(fUplo, fN, &fAlpha, (fA + offa), (fX + offx), fIncx, &fBeta, (fY + offy), fIncy);
        }

void
    blasChpr(
        clblasOrder order,
		clblasUplo uplo,
        size_t N,
        float alpha,
        FloatComplex* x,
        size_t offx,
        int incx,
        FloatComplex* A,
        size_t offa)
        {
	    	char fUplo;
            int fN;
            complex *fA, *fx ;
			fUplo = encodeUplo( uplo );
            fN = (int) N;
            fA = (complex*) A;
            fx = (complex*) x;

            if (order != clblasColumnMajor) {

            	fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
			}
            chpr( fUplo, fN, alpha, (fx+offx), incx, (fA+offa));

        }


void
    blasZhpr(
        clblasOrder order,
		clblasUplo uplo,
        size_t N,
        double alpha,
        DoubleComplex* x,
        size_t offx,
        int incx,
        DoubleComplex* A,
        size_t offa)
        {
            char fUplo;
            int  fN;
            doublecomplex *fA, *fx;
			fUplo = encodeUplo( uplo );
            fN = (int) N;
            fA =(doublecomplex*) A;
            fx =(doublecomplex*) x;

            if (order != clblasColumnMajor) {
            fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
            abort();
            }

            zhpr( fUplo, fN, alpha, (fx+offx), incx, (fA+offa) );
        }

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
		float* A,
		size_t offa)
		{
			char fUplo;
            int fN, fIncx, fIncy;
			float *fA, fAlpha, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
			fIncx = (int) incx;
			fIncy = (int) incy;

			fA = (float*) A;
			fX = (float*) X;
			fY = (float*) Y;

			fAlpha = alpha;

			if (order != clblasColumnMajor)
			{
    	        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
		        abort();
			}

			sspr2(fUplo, fN, fAlpha, (fX + offx), fIncx, (fY + offy), fIncy, (fA + offa));
		}


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
		double* A,
        size_t offa)
        {
            char fUplo;
            int fN, fIncx, fIncy;
            double *fA, fAlpha, *fX, *fY;

            fUplo = encodeUplo( uplo );

            fN    = (int) N;
            fIncx = (int) incx;
			fIncy = (int) incy;

            fA = (double*) A;
            fX = (double*) X;
			fY = (double*) Y;

            fAlpha = alpha;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dspr2(fUplo, fN, fAlpha, (fX + offx), fIncx, (fY + offy), fIncy, (fA + offa));
        }

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
        FloatComplex* A,
        size_t offa)
        {
            char fUplo;
            int fN;
            complex *fA, fAlpha, *fX, *fY;
            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fA = (complex*) A;
            fX = (complex*) X;
            fY = (complex*) Y;

	        fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            chpr2(fUplo, fN, &fAlpha, (fX + offx), incx, (fY + offy), incy, (fA + offa));
        }

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
        DoubleComplex* A,
        size_t offa)
        {
            char fUplo;
            int fN ;
            doublecomplex *fA, fAlpha, *fX, *fY;
            fUplo = encodeUplo( uplo );
            fN    = (int) N;
            fA = (doublecomplex*) A;
            fX = (doublecomplex*) X;
            fY = (doublecomplex*) Y;

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            zhpr2(fUplo, fN, &fAlpha, (fX + offx), incx, (fY + offy), incy, (fA + offa));
        }

void
blasSgbmv(
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
            char fTrans;
            int fN, fM, fKL, fKU, fLda;

            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fM = (int) M;
            fKL = (int) KL;
            fKU = (int) KU;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            sgbmv(fTrans, fM, fN, fKL, fKU, alpha, (A+offa), fLda, (X+offx), incx, beta, (Y+offy), incy);
        }

void
blasDgbmv(
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
            char fTrans;
            int fN, fM, fKL, fKU, fLda;

            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fM = (int) M;
            fKL = (int) KL;
            fKU = (int) KU;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dgbmv(fTrans, fM, fN, fKL, fKU, alpha, (A+offa), fLda, (X+offx), incx, beta, (Y+offy), incy);
        }

void
blasCgbmv(
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
            char fTrans;
            int fN, fM, fKL, fKU, fLda;
            complex *fA, *fX, *fY, fAlpha, fBeta;

            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fM = (int) M;
            fKL = (int) KL;
            fKU = (int) KU;
            fLda = (int) lda;
            fA = (complex*) (A + offa);
            fX = (complex*) (X + offx);
            fY = (complex*) (Y + offy);
            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_complex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            cgbmv(fTrans, fM, fN, fKL, fKU, &fAlpha, fA, fLda, fX, incx, &fBeta, fY, incy);
        }

void
blasZgbmv(
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
            char fTrans;
            int fN, fM, fKL, fKU, fLda;
            doublecomplex *fA, *fX, *fY, fAlpha, fBeta;

            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fM = (int) M;
            fKL = (int) KL;
            fKU = (int) KU;
            fLda = (int) lda;
            fA = (doublecomplex*) (A + offa);
            fX = (doublecomplex*) (X + offx);
            fY = (doublecomplex*) (Y + offy);
            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            zgbmv(fTrans, fM, fN, fKL, fKU, &fAlpha, fA, fLda, fX, incx, &fBeta, fY, incy);
        }


//TBMV

void
blasStbmv(
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
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;

            fTrans = encodeTranspose(trans);
            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            stbmv(fUplo, fTrans, fDiag, fN, fK, (A+offa), fLda, (X+offx), incx );
        }

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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;

            fTrans = encodeTranspose(trans);
            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dtbmv(fUplo, fTrans, fDiag, fN, fK, (A+offa), fLda, (X+offx), incx );
        }
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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;
            complex *fA, *fX;

            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (complex*) (A + offa);
            fX = (complex*) (X + offx);

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            ctbmv(fUplo, fTrans, fDiag, fN, fK, fA, fLda, fX, incx );
        }

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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;
            doublecomplex *fA, *fX;

            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (doublecomplex*) (A + offa);
            fX = (doublecomplex*) (X + offx);

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            ztbmv(fUplo, fTrans, fDiag, fN, fK, fA, fLda, fX, incx );
        }


//SBMV

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
        int incy)
        {
            char  fUplo;
            int fN, fK, fLda;

            fUplo = encodeUplo(uplo);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            ssbmv( fUplo, fN, fK, alpha, (A+offa), fLda, (X+offx), incx, beta, (Y+offy), incy );
        }

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
        int incy)
        {
            char fUplo;
            int fN, fK, fLda;

            fUplo = encodeUplo(uplo);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dsbmv(fUplo, fN, fK, alpha, (A+offa), fLda, (X+offx), incx, beta, (Y+offy), incy );
        }

//HBMV

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
        int incy)
        {
            char fUplo;
            int fN, fK, fLda;
            complex *fA, *fX, *fY, fAlpha, fBeta;

            fUplo = encodeUplo(uplo);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (complex*) (A + offa);
            fX = (complex*) (X + offx);
            fY = (complex*) (Y + offy);

            fAlpha = compose_complex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_complex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            chbmv( fUplo, fN, fK, &fAlpha, fA, fLda, fX, incx, &fBeta, fY, incy );
        }

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
        int incy)
        {
            char fUplo;
            int fN, fK, fLda;
            doublecomplex *fA, *fX, *fY, fAlpha, fBeta;

            fUplo = encodeUplo(uplo);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (doublecomplex*) (A + offa);
            fX = (doublecomplex*) (X + offx);
            fY = (doublecomplex*) (Y + offy);

            fAlpha = compose_doublecomplex(CREAL(alpha), CIMAG(alpha));
            fBeta = compose_doublecomplex(CREAL(beta), CIMAG(beta));

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            zhbmv(fUplo, fN, fK, &fAlpha, fA, fLda, fX, incx, &fBeta, fY, incy );
        }


//TBSV

void
blasStbsv(
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
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;

            fTrans = encodeTranspose(trans);
            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            stbsv(fUplo, fTrans, fDiag, fN, fK, (A+offa), fLda, (X+offx), incx );
        }

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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;

            fTrans = encodeTranspose(trans);
            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            dtbsv(fUplo, fTrans, fDiag, fN, fK, (A+offa), fLda, (X+offx), incx );
        }


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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;
             complex *fA, *fX;

            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (complex*) (A + offa);
            fX = (complex*) (X + offx);

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }

            ctbsv(fUplo, fTrans, fDiag, fN, fK, fA, fLda, fX, incx );
        }

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
        int incx)
        {
            char fTrans, fUplo, fDiag;
            int fN, fK, fLda;
            doublecomplex *fA, *fX;

            fUplo = encodeUplo(uplo);
            fDiag = encodeDiag(diag);
            fTrans = encodeTranspose(trans);
            fN = (int) N;
            fK = (int) K;
            fLda = (int) lda;
            fA = (doublecomplex*) (A + offa);
            fX = (doublecomplex*) (X + offx);

            if (order != clblasColumnMajor)
            {
                fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
                abort();
            }
            ztbsv(fUplo, fTrans, fDiag, fN, fK, fA, fLda, fX, incx );
        }

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC, fLDB;
    complex *fA, *fC, *fB, *fAlpha;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fA = (complex*)(A+offa);
    fB = (complex*)(B+offb);
    fC = (complex*)(C+offc);
    fAlpha = (complex*)(&alpha);

	cher2k(fUplo, fTransA, fN, fK, fAlpha, fA, fLDA, fB, fLDB, beta, fC, fLDC);
}

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
    size_t ldc)
{
    char fUplo, fTransA;
    int fN, fK;
    int fLDA, fLDC, fLDB;
    doublecomplex *fA, *fC, *fB, *fAlpha;

    if (order != clblasColumnMajor) {
        fprintf(stderr, "LAPACK routines require clblasColumnMajor order\n");
        abort();
    }

    fUplo = encodeUplo(uplo);
    fTransA = encodeTranspose(transA);
    fN = (int)N;
    fK = (int)K;
    fLDA = (int)lda;
    fLDB = (int)ldb;
    fLDC = (int)ldc;

    fA = (doublecomplex*)(A+offa);
    fB = (doublecomplex*)(B+offb);
    fC = (doublecomplex*)(C+offc);
    fAlpha = (doublecomplex*)(&alpha);

	zher2k(fUplo, fTransA, fN, fK, fAlpha, fA, fLDA, fB, fLDB, beta, fC, fLDC);
}


//COPY


void
blasScopy(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy)
        {
            int fN;
            fN = (int) N;

            scopy(fN, (X+offx), incx, (Y+offy), incy);
        }



void
blasDcopy(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy)
        {

            int fN;
            fN = (int) N;

            dcopy( fN, (X+offx), incx, (Y+offy), incy );
        }


void
blasCcopy(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {

            int fN;
            complex *fY, *fX;

            fN = (int) N;
            fY = (complex*) (Y + offy);
            fX = (complex*) (X + offx);

            ccopy( fN, fX, incx, fY, incy );
        }


void
blasZcopy(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {
            int fN;
            doublecomplex *fY, *fX;

            fN = (int) N;
            fY = (doublecomplex*) (Y + offy);
            fX = (doublecomplex*) (X + offx);

            zcopy(fN, fX, incx , fY, incy);
        }


//SWAP

void
blasSswap(
        size_t N,
        float *X,
        size_t offBX,
        int incx,
        float *Y,
        size_t offCY,
        int incy)
        {
            int fN;
            fN = (int) N;

            sswap(fN, (X+offBX), incx, (Y+offCY), incy);
        }



void
blasDswap(
        size_t N,
        double *X,
        size_t offBX,
        int incx,
        double *Y,
        size_t offCY,
        int incy)
        {

            int fN;
            fN = (int) N;

            dswap( fN, (X+offBX), incx, (Y+offCY), incy );
        }


void
blasCswap(
        size_t N,
        FloatComplex *X,
        size_t offBX,
        int incx,
        FloatComplex *Y,
        size_t offCY,
        int incy)
        {

            int fN;
            complex *fY, *fX;

            fN = (int) N;
            fY = (complex*) (Y + offCY);
            fX = (complex*) (X + offBX);

            cswap( fN, fX, incx, fY, incy );
        }


void
blasZswap(
        size_t N,
        DoubleComplex *X,
        size_t offBX,
        int incx,
        DoubleComplex *Y,
        size_t offCY,
        int incy)
        {
            int fN;
            doublecomplex *fY, *fX;

            fN = (int) N;
            fY = (doublecomplex*) (Y + offCY);
            fX = (doublecomplex*) (X + offBX);

            zswap(fN, fX, incx , fY, incy);
        }


void
	blasSscal(
	    size_t N,
        float alpha,
        float *X,
        size_t offx,
        int incx)
{
    sscal((int)N, alpha, (X+offx), incx);
}

void
	blasDscal(
	    size_t N,
        double alpha,
        double *X,
        size_t offx,
        int incx)
{
    dscal((int)N, alpha, (X+offx), incx);
}

void
	blasCscal(
	    size_t N,
        FloatComplex alpha,
        FloatComplex *X,
        size_t offx,
        int incx)
{
    cscal((int)N, (complex*)(&alpha), (complex*)(X+offx), incx);
}

void
	blasZscal(
	    size_t N,
        DoubleComplex alpha,
        DoubleComplex *X,
        size_t offx,
        int incx)
{
    zscal((int)N, (doublecomplex*)(&alpha), (doublecomplex*)(X+offx), incx);
}

void
	blasCsscal(
	    size_t N,
        float alpha,
        FloatComplex *X,
        size_t offx,
        int incx)
{
    csscal((int)N, alpha, (complex*)(X+offx), incx);
}

void
	blasZdscal(
	    size_t N,
        double alpha,
        DoubleComplex *X,
        size_t offx,
        int incx)
{
    zdscal((int)N, alpha, (doublecomplex*)(X+offx), incx);
}

//DOT
float
blasSdot(
        size_t N,
        float *X,
        size_t offx,
        int incx,
        float *Y,
        size_t offy,
        int incy)
        {
            return sdot((int)N, (X+offx), incx, (Y+offy), incy);
        }

double
blasDdot(
        size_t N,
        double *X,
        size_t offx,
        int incx,
        double *Y,
        size_t offy,
        int incy)
        {
            return ddot( (int)N, (X+offx), incx, (Y+offy), incy );
        }

FloatComplex
blasCdotu(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {
            complex ans = cdotu((int)N, (complex*)(X+offx), incx, (complex*)(Y+offy), incy);
            FloatComplex ret;
            CREAL(ret) = ans.real;
            CIMAG(ret) = ans.imag;
            return ret;
        }

DoubleComplex
blasZdotu(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {
            doublecomplex answer = zdotu( (int)N, (doublecomplex*)(X+offx), incx, (doublecomplex*)(Y+offy), incy );
            DoubleComplex ret2;
            CREAL(ret2) = answer.real;
            CIMAG(ret2) = answer.imag;
            return ret2;
        }

//ASUM
float
blasSasum(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {
            return sasum((int)N, (X+offx), incx);
        }

double
blasDasum(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {
            return dasum( (int)N, (X+offx), incx);
        }

float
blasScasum(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {
            return scasum((int)N, (complex*)(X+offx), incx);
        }

double
blasDzasum(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {
            return dzasum( (int)N, (doublecomplex*)(X+offx), incx);
        }

//DOTC
FloatComplex
blasCdotc(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx,
        FloatComplex *Y,
        size_t offy,
        int incy)
        {
            complex ans = cdotc((int)N, (complex*)(X+offx), incx, (complex*)(Y+offy), incy);
            FloatComplex ret;
            CREAL(ret) = ans.real;
            CIMAG(ret) = ans.imag;
            return ret;
        }

DoubleComplex
blasZdotc(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx,
        DoubleComplex *Y,
        size_t offy,
        int incy)
        {
            doublecomplex answer = zdotc( (int)N, (doublecomplex*)(X+offx), incx, (doublecomplex*)(Y+offy), incy );
            DoubleComplex ret2;
            CREAL(ret2) = answer.real;
            CIMAG(ret2) = answer.imag;
            return ret2;
        }


void
blasSaxpy(
        size_t N,
        float alpha,
        const float *X,
        size_t offBX,
        int incx,
        float *Y,
        size_t offCY,
        int incy)
{
    saxpy((int)N, alpha, (float*)(X+offBX), incx, (Y+offCY), incy);
}

void
blasDaxpy(
        size_t N,
        double alpha,
        const double *X,
        size_t offBX,
        int incx,
        double *Y,
        size_t offCY,
        int incy)
{
    daxpy((int)N, alpha, (double*)(X+offBX), incx, (Y+offCY), incy);
}

void
blasCaxpy(
        size_t N,
        FloatComplex alpha,
        const FloatComplex *X,
        size_t offBX,
        int incx,
        FloatComplex *Y,
        size_t offCY,
        int incy)
{
    caxpy((int)N, (complex*)(&alpha),(complex*)(X+offBX), incx, (complex*)(Y+offCY), incy);
}

void
blasZaxpy(
        size_t N,
        DoubleComplex alpha,
        const DoubleComplex *X,
        size_t offBX,
        int incx,
        DoubleComplex *Y,
        size_t offCY,
        int incy)
{
    zaxpy((int)N, (doublecomplex*)(&alpha), (doublecomplex*)(X+offBX), incx, (doublecomplex*)(Y+offCY), incy);
}


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
        size_t offS)
        {
            srotg((SA+offSA), (SB+offSB), (C+offC), (S+offS));
        }

void
blasDrotg(
        double* SA,
        size_t offSA,
        double* SB,
        size_t offSB,
        double* C,
        size_t offC,
        double* S,
        size_t offS)
        {
            drotg((SA+offSA), (SB+offSB), (C+offC), (S+offS));
        }

void
blasCrotg(
        FloatComplex* SA,
        size_t offSA,
        FloatComplex* SB,
        size_t offSB,
        float* C,
        size_t offC,
        FloatComplex* S,
        size_t offS)
        {
            crotg((complex*)(SA+offSA), (complex*)(SB+offSB), (C+offC), (complex*)(S+offS));
        }

void
blasZrotg(
        DoubleComplex* SA,
        size_t offSA,
        DoubleComplex* SB,
        size_t offSB,
        double* C,
        size_t offC,
        DoubleComplex* S,
        size_t offS)
        {
            zrotg((doublecomplex*)(SA+offSA), (doublecomplex*)(SB+offSB), (C+offC), (doublecomplex*)(S+offS));
        }

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
        size_t offParam)
        {
            // C and fortran interface are different for rotmg..  FIXME
            #if defined CORR_TEST_WITH_ACML
                srotmg(D1[offD1], D2[offD2], X1[offX1], Y1[offY1],
                        (PARAM+offParam));
            #else
                srotmg((D1+offD1), (D2+offD2), (X1+offX1), (Y1+offY1),
                        (PARAM+offParam));
            #endif
        }

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
        size_t offParam)
        {
            // C and fortran interface are different for rotmg..  FIXME
            #if defined CORR_TEST_WITH_ACML
                drotmg(D1[offD1], D2[offD2], X1[offX1], Y1[offY1],
                        (PARAM+offParam));
            #else
                drotmg((D1+offD1), (D2+offD2), (X1+offX1), (Y1+offY1),
                        (PARAM+offParam));
            #endif
        }

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
        size_t offParam)
        {
            srotm(N, (X+offx), incx, (Y+offy), incy, (PARAM+offParam));
        }

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
        size_t offParam)
        {
            drotm(N, (X+offx), incx, (Y+offy), incy, (PARAM+offParam));
        }
//ROT

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
        float S)
        {
            srot(N, (X+offx), incx, (Y+offy), incy, C, S);
        }

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
        double S)
        {
            drot(N, (X+offx), incx, (Y+offy), incy, C, S);
        }

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
        float S)
        {
            csrot(N, (complex*)(X+offx), incx, (complex*)(Y+offy), incy, C, S);
        }

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
        double S)
        {
            zdrot(N, (doublecomplex*)(X+offx), incx, (doublecomplex*)(Y+offy), incy, C, S);
        }

int
blasiSamax(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {
            return isamax((int)N, (X+offx), incx);
        }

int
blasiDamax(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {
            return idamax( (int)N, (X+offx), incx);
        }

int
blasiCamax(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {
            return icamax((int)N, (complex*)(X+offx), incx);
        }

int
blasiZamax(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {
            return izamax( (int)N, (doublecomplex*)(X+offx), incx);
        }

float
blasSnrm2(
        size_t N,
        float *X,
        size_t offx,
        int incx)
        {
            return snrm2((int)N, (X+offx), incx);
        }

double
blasDnrm2(
        size_t N,
        double *X,
        size_t offx,
        int incx)
        {
            return dnrm2( (int)N, (X+offx), incx);
        }

float
blasScnrm2(
        size_t N,
        FloatComplex *X,
        size_t offx,
        int incx)
        {
            return scnrm2((int)N, (complex*)(X+offx), incx);
        }

double
blasDznrm2(
        size_t N,
        DoubleComplex *X,
        size_t offx,
        int incx)
        {
            return dznrm2( (int)N, (doublecomplex*)(X+offx), incx);
        }
