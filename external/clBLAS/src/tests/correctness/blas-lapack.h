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


#ifndef BLAS_LAPACK_H_
#define BLAS_LAPACK_H_

#ifdef __cplusplus
extern "C" {
#endif

/* BLAS-2 functions */

void sgemv_(
    const char *transA,
    const int *M,
    const int *N,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *X,
    const int *incx,
    const float *beta,
    float *Y,
    const int *incy);

void dgemv_(
    const char *transA,
    const int *M,
    const int *N,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *X,
    const int *incx,
    const double *beta,
    double *Y,
    const int *incy);

void cgemv_(
    const char *transA,
    const int *M,
    const int *N,
    const complex *alpha,
    const complex *A,
    const int *lda,
    const complex *X,
    const int *incx,
    const complex *beta,
    complex *Y,
    const int *incy);

void zgemv_(
    const char *transA,
    const int *M,
    const int *N,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    const doublecomplex *X,
    const int *incx,
    const doublecomplex *beta,
    doublecomplex *Y,
    const int *incy);

void ssymv_(
    const char *uplo,
    const int *N,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *X,
    const int *incx,
    const float *beta,
    float *Y,
    int *incy);

void dsymv_(
    const char *uplo,
    const int *N,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *X,
    const int *incx,
    const double *beta,
    double *Y,
    const int *incy);

/* BLAS-3 functions */

void sgemm_(
    const char *transA,
    const char *transB,
    const int *M,
    const int *N,
    const int *K,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc);

void dgemm_(
    const char *transA,
    const char *transB,
    const int *M,
    const int *N,
    const int *K,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *B,
    const int *ldb,
    const double *beta,
    double *C,
    const int *ldc);

void cgemm_(
    const char *transA,
    const char *transB,
    const int *M,
    const int *N,
    const int *K,
    const complex *alpha,
    const complex *A,
    const int *lda,
    const complex *B,
    const int *ldb,
    const complex *beta,
    complex *C,
    const int *ldc);

void zgemm_(
    const char *transA,
    const char *transB,
    const int *M,
    const int *N,
    const int *K,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    const doublecomplex *B,
    const int *ldb,
    const doublecomplex *beta,
    doublecomplex *C,
    const int *ldc);

void strmm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const float *alpha,
    const float *A,
    const int *lda,
    float *B,
    const int *ldb);

void dtrmm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const double *alpha,
    const double *A,
    const int *lda,
    double *B,
    const int *ldb);

void ctrmm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const complex *alpha,
    const complex *A,
    const int *lda,
    complex *B,
    const int *ldb);

void ztrmm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    doublecomplex *B,
    const int *ldb);

void strsm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const float *aplha,
    const float *A,
    const int *lda,
    float *B,
    const int *ldb);

void dtrsm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const double *alpha,
    const double *A,
    const int *lda,
    double *B,
    const int *ldb);

void ctrsm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const complex *alpha,
    const complex *A,
    const int *lda,
    complex *B,
    const int *ldb);

void ztrsm_(
    const char *side,
    const char *uplo,
    const char *transA,
    const char *diag,
    const int *M,
    const int *N,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    doublecomplex *B,
    const int *ldb);

void ssyr2k_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc);

void dsyr2k_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *B,
    const int *ldb,
    const double *beta,
    double *C,
    const int *ldc);

void csyr2k_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const complex *alpha,
    const complex *A,
    const int *lda,
    const complex *B,
    const int *ldb,
    const complex *beta,
    complex *C,
    const int *ldc);

void zsyr2k_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    const doublecomplex *B,
    const int *ldb,
    const doublecomplex *beta,
    doublecomplex *C,
    const int *ldc);

void ssyrk_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *beta,
    float *C,
    const int *ldc);

void dsyrk_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *beta,
    double *C,
    const int *ldc);

void csyrk_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const complex *alpha,
    const complex *A,
    const int *lda,
    const complex *beta,
    complex *C,
    const int *ldc);

void zsyrk_(
    const char *uplo,
    const char *transA,
    const int *N,
    const int *K,
    const doublecomplex *alpha,
    const doublecomplex *A,
    const int *lda,
    const doublecomplex *beta,
    doublecomplex *C,
    const int *ldc);

void strmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const float *a,
    const int *lda,
    float *x,
    const int *incx);

void dtrmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const double *a,
    const int *lda,
    double *x,
    const int *incx);

void ctrmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const complex *a,
    const int *lda,
    complex *x,
    const int *incx);

void ztrmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const doublecomplex *a,
    const int *lda,
    doublecomplex *x,
    const int *incx);

void strsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const float *a,
    const int *lda,
    float *x,
    const int *incx);

void dtrsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const double *a,
    const int *lda,
    double *x,
    const int *incx);

void ctrsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const complex *a,
    const int *lda,
    complex *x,
    const int *incx);

void ztrsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const doublecomplex *a,
    const int *lda,
    doublecomplex *x,
    const int *incx);

void ssymm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const float *alpha,
    const float *a,
    const int *lda,
    const float *b,
    const int *ldb,
    const float *beta,
    float *c,
    const int *ldc);

void dsymm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const double *alpha,
    const double *a,
    const int *lda,
    const double *b,
    const int *ldb,
    const double *beta,
    double *c,
    const int *ldc);

void csymm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const complex *alpha,
    const complex *a,
    const int *lda,
    const complex *b,
    const int *ldb,
    const complex *beta,
    complex *c,
    const int *ldc);

void zsymm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *lda,
    const doublecomplex *b,
    const int *ldb,
    const doublecomplex *beta,
    doublecomplex *c,
    const int *ldc);

void sger_(
    const int *m,
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    const float *y,
    const int *incy,
    float *a,
    const int *lda);

void dger_(
    const int *m,
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    const double *y,
    const int *incy,
    double *a,
    const int *lda);

void cgeru_(
    const int *m,
    const int *n,
    const complex *alpha,
    const complex *x,
    const int *incx,
    const complex *y,
    const int *incy,
    complex *a,
    const int *lda);

void zgeru_(
    const int *m,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *y,
    const int *incy,
    doublecomplex *a,
    const int *lda);

void cgerc_(
    const int *m,
    const int *n,
    const complex *alpha,
    const complex *x,
    const int *incx,
    const complex *y,
    const int *incy,
    complex *a,
    const int *lda);

void zgerc_(
    const int *m,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *y,
    const int *incy,
    doublecomplex *a,
    const int *lda);

void ssyr_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    float *a,
    const int *lda);

void dsyr_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    double *a,
    const int *lda);

void ssyr2_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    const float *y,
    const int *incy,
    float *a,
    const int *lda);

void dsyr2_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    const double *y,
    const int *incy,
    double *a,
    const int *lda);

void cher_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const complex *x,
    const int *incx,
    complex *a,
    const int *lda);

void zher_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const doublecomplex *x,
    const int *incx,
    doublecomplex *a,
    const int *lda);

void cher2_(
    const char *uplo,
    const int *n,
    const complex *alpha,
    const complex *x,
    const int *incx,
    const complex *y,
    const int *incy,
    complex *a,
    const int *lda);

void zher2_(
    const char *uplo,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *y,
    const int *incy,
    doublecomplex *a,
    const int *lda);

void chemv_(
    const char *uplo,
    const int *n,
    const complex *alpha,
    const complex *a,
    const int *lda,
    const complex *x,
    const int *incx,
    const complex *beta,
    complex *y,
    const int *incy);

void zhemv_(
    const char *uplo,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *lda,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *beta,
    doublecomplex *y,
    const int *incy);

void stpmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const float *ap,
    float *x,
    const int *incx);

void dtpmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const double *ap,
    double *x,
    const int *incx);

void ctpmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const complex *ap,
    complex *x,
    const int *incx);

void ztpmv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const doublecomplex *ap,
    doublecomplex *x,
    const int *incx);

void stpsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const float *ap,
    float *x,
    const int *incx);

void dtpsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const double *ap,
    double *x,
    const int *incx);

void ctpsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const complex *ap,
    complex *x,
    const int *incx);

void ztpsv_(
    const char *uplo,
    const char *transa,
    const char *diag,
    const int *n,
    const doublecomplex *ap,
    doublecomplex *x,
    const int *incx);

void sspr_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    float *ap);

void dspr_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    double *ap);

void
sspmv_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const float *ap,
    const float *x,
    const int *incx,
    const float *beta,
    float *y,
    const int *incy);

void
dspmv_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const double *ap,
    const double *x,
    const int *incx,
    const double *beta,
    double *y,
    const int *incy);

void
chpmv_(
    const char *uplo,
    const int *n,
    const complex *alpha,
    const complex *ap,
    const complex *x,
    const int *incx,
    const complex *beta,
    complex *y,
    const int *incy);

void
zhpmv_(
    const char *uplo,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *ap,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *beta,
    doublecomplex *y,
    const int *incy);

void chpr_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const complex *x,
    const int *incx,
    complex *ap);

void zhpr_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const doublecomplex *x,
    const int *incx,
    doublecomplex *ap);

void sspr2_(
    const char *uplo,
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    const float *y,
    const int *incy,
    float *a );

void dspr2_(
    const char *uplo,
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    const double *y,
    const int *incy,
    double *a );

void chpr2_(
    const char *uplo,
    const int *n,
    const complex *alpha,
    const complex *x,
    const int *incx,
    const complex *y,
    const int *incy,
    complex *a );

void zhpr2_(
    const char *uplo,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *y,
    const int *incy,
    doublecomplex *a );

void sgbmv_(
    const char *trans,
    const int *m,
    const int *n,
    const int *kl,
    const int *ku,
    const float *alpha,
    const float *a,
    const int *inca,
    const float *x,
    const int *incx,
    const float *beta,
    float *y,
    const int *incy );

void dgbmv_(
    const char *trans,
    const int *m,
    const int *n,
    const int *kl,
    const int *ku,
    const double *alpha,
    const double *a,
    const int *inca,
    const double *x,
    const int *incx,
    const double *beta,
    double *y,
    const int *incy );

void cgbmv_(
    const char *trans,
    const int *m,
    const int *n,
    const int *kl,
    const int *ku,
    const complex *alpha,
    const complex *a,
    const int *inca,
    const complex *x,
    const int *incx,
    const complex *beta,
    complex *y,
    const int *incy );

void zgbmv_(
    const char *trans,
    const int *m,
    const int *n,
    const int *kl,
    const int *ku,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *inca,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *beta,
    doublecomplex *y,
    const int *incy );

void stbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const float *a,
    const int *lda,
    float *x,
    const int *incx );

void dtbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const double *a,
    const int *lda,
    double *x,
    const int *incx );

void ctbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const complex *a,
    const int *lda,
    complex *x,
    const int *incx );

void ztbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const doublecomplex *a,
    const int *lda,
    doublecomplex *x,
    const int *incx );

void ssbmv_(
    const char *uplo,
    const int *n,
    const int *k,
    const float *alpha,
    const float *a,
    const int *lda,
    const float *x,
    const int *incx,
    const float *beta,
    float *y,
    const int *incy );

void dsbmv_(
    const char *uplo,
    const int *n,
    const int *k,
    const double *alpha,
    const double *a,
    const int *lda,
    const double *x,
    const int *incx,
    const double *beta,
    double *y,
    const int *incy );

void chbmv_(
    const char *uplo,
    const int *n,
    const int *k,
    const complex *alpha,
    const complex *a,
    const int *lda,
    const complex *x,
    const int *incx,
    const complex *beta,
    complex *y,
    const int *incy );

void zhbmv_(
    const char *uplo,
    const int *n,
    const int *k,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *lda,
    const doublecomplex *x,
    const int *incx,
    const doublecomplex *beta,
    doublecomplex *y,
    const int *incy );

void stbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const float *a,
    const int *lda,
    float *x,
    const int *incx );

void dtbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const double *a,
    const int *lda,
    double *x,
    const int *incx );

void ctbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const complex *a,
    const int *lda,
    complex *x,
    const int *incx );

void ztbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const int *k,
    const doublecomplex *a,
    const int *lda,
    doublecomplex *x,
    const int *incx );

void chemm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const complex *alpha,
    const complex *a,
    const int *lda,
    const complex *b,
    const int *ldb,
    const complex *beta,
    complex *c,
    const int *ldc);

void zhemm_(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *lda,
    const doublecomplex *b,
    const int *ldb,
    const doublecomplex *beta,
    doublecomplex *c,
    const int *ldc);

void cherk_(
    const char *uplo,
    const char *transa,
    const int *n,
    const int *k,
    const float *alpha,
    const complex *a,
    const int *lda,
    const float *beta,
    complex *c,
    const int *ldc);

void zherk_(
    const char *uplo,
    const char *transa,
    const int *n,
    const int *k,
    const double *alpha,
    const doublecomplex *a,
    const int *lda,
    const double *beta,
    doublecomplex *c,
    const int *ldc);

void cher2k_(
    const char *uplo,
    const char *transa,
    const int *n,
    const int *k,
    const complex *alpha,
    const complex *a,
    const int *lda,
    const complex *b,
    const int *ldb,
    const float *beta,
    complex *c,
    const int *ldc);

void zher2k_(
    const char *uplo,
    const char *transa,
    const int *n,
    const int *k,
    const doublecomplex *alpha,
    const doublecomplex *a,
    const int *lda,
    const doublecomplex *b,
    const int *ldb,
    const double *beta,
    doublecomplex *c,
    const int *ldc);

void sscal_(int *n, float *alpha, float *x, int *incx);
void dscal_(int *n, double *alpha, double *x, int *incx);
void cscal_(int *n, complex *alpha, complex *x, int *incx);
void zscal_(int *n, doublecomplex *alpha, doublecomplex *x, int *incx);

void csscal_(int *n, float *alpha, complex *x, int *incx);
void zdscal_(int *n, double *alpha, doublecomplex *x, int *incx);

void scopy_(int *n, float *x, int *incx, float* y, int *incy);
void dcopy_(int *n, double *x, int *incx, double* y, int *incy);
void ccopy_(int *n, complex *x, int *incx, complex *y, int *incy);
void zcopy_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

float sdot_(int *n, float *x, int *incx, float* y, int *incy);
double ddot_(int *n, double *x, int *incx, double* y, int *incy);

#if defined( _WIN32 ) || defined( _WIN64 ) || defined( __APPLE__)
    complex cdotu_(int *n, complex *x, int *incx, complex* y, int *incy);
    doublecomplex zdotu_(int *n, doublecomplex *x, int *incx, doublecomplex* y, int *incy);
    complex cdotc_(int *n, complex *x, int *incx, complex* y, int *incy);
    doublecomplex zdotc_(int *n, doublecomplex *x, int *incx, doublecomplex* y, int *incy);
#else
    void cdotusub_(int *n, complex *x, int *incx, complex* y, int *incy, complex *ans);
    void zdotusub_(int *n, doublecomplex *x, int *incx, doublecomplex* y, int *incy, doublecomplex *ans);
    void cdotcsub_(int *n, complex *x, int *incx, complex* y, int *incy, complex *ans);
    void zdotcsub_(int *n, doublecomplex *x, int *incx, doublecomplex* y, int *incy, doublecomplex *ans);
#endif

void sswap_(int *n, float *x, int *incx, float* y, int *incy);
void dswap_(int *n, double *x, int *incx, double* y, int *incy);
void cswap_(int *n, complex *x, int *incx, complex *y, int *incy);
void zswap_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

void saxpy_(int *n, float *alpha, float *x, int *incx, float* y, int *incy);
void daxpy_(int *n, double *alpha, double *x, int *incx, double* y, int *incy);
void caxpy_(int *n, complex *alpha, complex *x, int *incx, complex *y, int *incy);
void zaxpy_(int *n, doublecomplex *alpha, doublecomplex *x, int *incx, doublecomplex *y, int *incy);


void srotg_(float *A, float *B, float *C, float *S);
void drotg_(double *A, double *B, double *C, double *S);
void crotg_(complex *A, complex *B, float *C, complex *S);
void zrotg_(doublecomplex *A, doublecomplex *B, double *C, doublecomplex *S);

void srotmg_(float *D1, float *D2, float *X1, float *Y1, float *PARAM);
void drotmg_(double *D1, double *D2, double *X1, double *Y1, double *PARAM);

void srot_(int *n, float *x, int *incx, float *y, int *incy, float *c, float *s);
void drot_(int *n, double *x, int *incx, double *y, int *incy, double *c, double *s);
void csrot_(int *n, complex *x, int *incx, complex *y, int *incy, float *c, float *s);
void zdrot_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy, double *c, double *s);

void srotm_(int* N, float *X, int* incx, float *Y, int* incy, float* PARAM);
void drotm_(int* N, double *X, int* incx, double *Y, int* incy, double* PARAM);

float sasum_(int *n, float *x, int *incx);
double dasum_(int *n, double *x, int *incx);
float scasum_(int *n, complex *x, int *incx);
double dzasum_(int *n, doublecomplex *x, int *incx);

int isamax_(int *n, float *x, int *incx);
int idamax_(int *n, double *x, int *incx);
int icamax_(int *n, complex *x, int *incx);
int izamax_(int *n, doublecomplex *x, int *incx);

float snrm2_(int *n, float *x, int *incx);
double dnrm2_(int *n, double *x, int *incx);
float scnrm2_(int *n, complex *x, int *incx);
double dznrm2_(int *n, doublecomplex *x, int *incx);

#ifdef __cplusplus
}
#endif

#endif  /* BLAS_LAPACK_H */
