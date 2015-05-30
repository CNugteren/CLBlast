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


#ifndef BLAS_CBLAS_H_
#define BLAS_CBLAS_H_

/* Under Windows math.h defines "complex" to mean "_complex". */
#include <math.h>
#undef complex

#ifdef __cplusplus
extern "C" {
#endif

/* A complex datatype for use by the C interfaces to ACML routines */
#ifndef _ACML_COMPLEX
#define _ACML_COMPLEX
typedef struct
{
    float real, imag;
} complex;
typedef struct
{
    double real, imag;
} doublecomplex;
#endif /* !defined(_ACML_COMPLEX) */

/* Basic complex arithmetic routines for C */
complex compose_complex(float x, float y);
float complex_real(complex z);
float complex_imag(complex z);

doublecomplex compose_doublecomplex(double x, double y);
double doublecomplex_real(doublecomplex z);
double doublecomplex_imag(doublecomplex z);

/* BLAS-2 functions */
void sgemv(char transa, int m, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy);
void dgemv(char transa, int m, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy);
void cgemv(char transa, int m, int n, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy);
void zgemv(char transa, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy);

void ssymv(char uplo, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy);
void dsymv(char uplo, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy);

void strmv(char uplo, char transa, char diag, int n, float *a, int lda, float *x, int incx);
void dtrmv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx);
void ctrmv(char uplo, char transa, char diag, int n, complex *a, int lda, complex *x, int incx);
void ztrmv(char uplo, char transa, char diag, int n, doublecomplex *a, int lda, doublecomplex *x, int incx);

void strsv(char uplo, char transa, char diag, int n, float *a, int lda, float *x, int incx);
void dtrsv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx);
void ctrsv(char uplo, char transa, char diag, int n, complex *a, int lda, complex *x, int incx);
void ztrsv(char uplo, char transa, char diag, int n, doublecomplex *a, int lda, doublecomplex *x, int incx);

void sger(int m, int n, float alpha, float *x, int incx, float *y, int incy, float *a, int lda);
void dger(int m, int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda);

void cgeru(int m, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda);
void zgeru(int m, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda);

void cgerc(int m, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda);
void zgerc(int m, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda);

void ssyr(char uplo, int n, float alpha, float *x, int incx, float *a, int lda);
void dsyr(char uplo, int n, double alpha, double *x, int incx, double *a, int lda);
void ssyr2(char uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *a, int lda);
void dsyr2(char uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda);

void cher(char uplo, int n, float alpha, complex *x, int incx, complex *a, int lda);
void zher(char uplo, int n, double alpha, doublecomplex *x, int incx, doublecomplex *a, int lda);
void cher2(char uplo, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda);
void zher2(char uplo, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda);

void chemv(char uplo, int n, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy);
void zhemv(char uplo, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy);

void stpmv(char uplo, char transa, char diag, int n, float *ap, float *x, int incx);
void dtpmv(char uplo, char transa, char diag, int n, double *ap, double *x, int incx);
void ctpmv(char uplo, char transa, char diag, int n, complex *ap, complex *x, int incx);
void ztpmv(char uplo, char transa, char diag, int n, doublecomplex *ap, doublecomplex *x, int incx);

void stpsv(char uplo, char transa, char diag, int n, float *ap, float *x, int incx);
void dtpsv(char uplo, char transa, char diag, int n, double *ap, double *x, int incx);
void ctpsv(char uplo, char transa, char diag, int n, complex *ap, complex *x, int incx);
void ztpsv(char uplo, char transa, char diag, int n, doublecomplex *ap, doublecomplex *x, int incx);

void sspr(char uplo, int n, float alpha, float *x, int incx, float *ap );
void dspr(char uplo, int n, double alpha, double *x, int incx, double *ap );

void sspmv(char uplo, int n, float alpha, float *ap, float *x, int incx, float beta, float *y, int incy);
void dspmv(char uplo, int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy);
void chpmv(char uplo, int n, complex *alpha, complex *ap, complex *x, int incx, complex *beta, complex *y, int incy);
void zhpmv(char uplo, int n, doublecomplex *alpha, doublecomplex *ap, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy);

void chpr(char uplo, int n, float alpha, complex *x, int incx, complex *ap );
void zhpr(char uplo, int n, double alpha, doublecomplex *x, int incx, doublecomplex *ap );

void sspr2(char uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *a );
void dspr2(char uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *a );
void chpr2(char uplo, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a );
void zhpr2(char uplo, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a );

void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, float *a, int inca, float *x, int incx, float beta, float *y, int incy );
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, double *a, int inca, double *x, int incx, double beta, double *y, int incy );
void cgbmv(char trans, int m, int n, int kl, int ku, complex *alpha, complex *a, int inca, complex *x, int incx, complex *beta, complex *y, int incy );
void zgbmv(char trans, int m, int n, int kl, int ku, doublecomplex *alpha, doublecomplex *a, int inca, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy );

void stbmv(char uplo, char trans, char diag, int n, int k, float *a, int lda, float *x, int incx );
void dtbmv(char uplo, char trans, char diag, int n, int k, double *a, int lda, double *x, int incx );
void ctbmv(char uplo, char trans, char diag, int n, int k, complex *a, int lda, complex *x, int incx );
void ztbmv(char uplo, char trans, char diag, int n, int k, doublecomplex *a, int lda, doublecomplex *x, int incx );

void ssbmv(char uplo, int n, int k, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy );
void dsbmv(char uplo, int n, int k, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy );
void chbmv(char uplo, int n, int k, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy );
void zhbmv(char uplo, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy );

void stbsv(char uplo, char trans, char diag, int n, int k, float *a, int lda, float *x, int incx );
void dtbsv(char uplo, char trans, char diag, int n, int k, double *a, int lda, double *x, int incx );
void ctbsv(char uplo, char trans, char diag, int n, int k, complex *a, int lda, complex *x, int incx );
void ztbsv(char uplo, char trans, char diag, int n, int k, doublecomplex *a, int lda, doublecomplex *x, int incx );

/* BLAS-3 functions */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc);
void dgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void cgemm(char transa, char transb, int m, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc);
void zgemm(char transa, char transb, int m, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc);

void strmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb);
void dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb);
void ctrmm(char side, char uplo, char transa, char diag, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb);
void ztrmm(char side, char uplo, char transa, char diag, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb);

void strsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb);
void dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb);
void ctrsm(char side, char uplo, char transa, char diag, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb);
void ztrsm(char side, char uplo, char transa, char diag, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb);

void ssyr2k(char uplo, char transa, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc);
void dsyr2k(char uplo, char transa, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void csyr2k(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc);
void zsyr2k(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc);

void ssyrk(char uplo, char transa, int n, int k, float alpha, float *a, int lda, float beta, float *c, int ldc);
void dsyrk(char uplo, char transa, int n, int k, double alpha, double *a, int lda, double beta, double *c, int ldc);
void csyrk(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *beta, complex *c, int ldc);
void zsyrk(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *beta, doublecomplex *c, int ldc);

void ssymm(char side, char uplo, int m, int n, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc);
void dsymm(char side, char uplo, int m, int n, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void csymm(char side, char uplo, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc);
void zsymm(char side, char uplo, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc);

void chemm(char side, char uplo, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc);
void zhemm(char side, char uplo, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc);

void cherk(char uplo, char transa, int n, int k, float alpha, complex *a, int lda, float beta, complex *c, int ldc);
void zherk(char uplo, char transa, int n, int k, double alpha, doublecomplex *a, int lda, double beta, doublecomplex *c, int ldc);

void cher2k(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, float beta, complex *c, int ldc);
void zher2k(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, double beta, doublecomplex *c, int ldc);

void sscal( int n, float alpha, float *x, int incx);
void dscal( int n, double alpha, double *x, int incx);
void cscal( int n, complex* alpha, complex *x, int incx);
void zscal( int n, doublecomplex* alpha, doublecomplex *x, int incx);

void csscal( int n, float alpha, complex *x, int incx);
void zdscal( int n, double alpha, doublecomplex *x, int incx);

void sswap( int n, float *x, int incx, float *y, int incy);
void dswap( int n, double *x, int incx, double *y, int incy);
void cswap( int n, complex *x, int incx, complex *y, int incy);
void zswap( int n, doublecomplex *x, int incx, doublecomplex *y, int incy);

void scopy( int n, float *x, int incx, float *y, int incy);
void dcopy( int n, double *x, int incx, double *y, int incy);
void ccopy( int n, complex *x, int incx, complex *y, int incy);
void zcopy( int n, doublecomplex *x, int incx, doublecomplex *y, int incy);

float sdot( int n, float *x, int incx, float *y, int incy);
double ddot( int n, double *x, int incx, double *y, int incy);
complex cdotu( int n, complex *x, int incx, complex *y, int incy);
doublecomplex zdotu( int n, doublecomplex *x, int incx, doublecomplex *y, int incy);
complex cdotc( int n, complex *x, int incx, complex *y, int incy);
doublecomplex zdotc( int n, doublecomplex *x, int incx, doublecomplex *y, int incy);

void saxpy( int n, float alpha, float *x, int incx, float *y, int incy);
void daxpy( int n, double aplha, double *x, int incx, double *y, int incy);
void caxpy( int n, complex *alpha, complex *x, int incx, complex *y, int incy);
void zaxpy( int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy);

void srotg(float *A, float *B, float *C, float *S);
void drotg(double *A, double *B, double *C, double *S);
void crotg(complex *A, complex *B, float *C, complex *S);
void zrotg(doublecomplex *A, doublecomplex *B, double *C, doublecomplex *S);

void srotmg(float *D1, float *D2, float *X1, const float *Y1, float *PARAM);
void drotmg(double *D1, double *D2, double *X1, const double *Y1, double *PARAM);

void srotm(int N, float *X, int incx, float *Y, int incy, float* PARAM);
void drotm(int N, double *X, int incx, double *Y, int incy, double* PARAM);

void srot(int N, float *X, int incx, float *Y, int incy, float C, float S);
void drot(int N, double *X, int incx, double *Y, int incy, double C, double S);
void csrot(int N, complex *X, int incx, complex *Y, int incy, float C, float S);
void zdrot(int N, doublecomplex *X, int incx, doublecomplex *Y, int incy, double C, double S);

float sasum(int n, float *x, int incx);
double dasum(int n, double *x, int incx);
float scasum(int n, complex *x, int incx);
double dzasum(int n, doublecomplex *x, int incx);

float snrm2( int n, float *x, int incx);
double dnrm2( int n, double *x, int incx);
float scnrm2( int n, complex *x, int incx);
double dznrm2( int n, doublecomplex *x, int incx);

int isamax(int n, float *x, int incx);
int idamax(int n, double *x, int incx);
int icamax(int n, complex *x, int incx);
int izamax(int n, doublecomplex *x, int incx);

#ifdef __cplusplus
}
#endif

#endif  /* BLAS_CBLAS_H_ */
