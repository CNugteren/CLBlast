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
 * cblas to lapack's blas interface adapter
 */

#include <blas-cblas.h>

#if !defined CORR_TEST_WITH_ACML

#include "blas-lapack.h"
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

void
sgemv(char transa, int m, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy)
{
    sgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void
dgemv(char transa, int m, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy)
{
    dgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void
cgemv(char transa, int m, int n, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy)
{
    cgemv_(&transa, &m, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
}

void
zgemv(char transa, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy)
{
    zgemv_(&transa, &m, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
}

void
ssymv(char uplo, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy)
{
    ssymv_(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void
dsymv(char uplo, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy)
{
    dsymv_(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void
sgemm(char transa, char transb, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc)
{
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
dgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
cgemm(char transa, char transb, int m, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc)
{
    cgemm_(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
zgemm(char transa, char transb, int m, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc)
{
    zgemm_(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
strmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb)
{
    strmm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void
dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb)
{
    dtrmm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void
ctrmm(char side, char uplo, char transa, char diag, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb)
{
    ctrmm_(&side, &uplo, &transa, &diag, &m, &n, alpha, a, &lda, b, &ldb);
}

void
ztrmm(char side, char uplo, char transa, char diag, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb)
{
    ztrmm_(&side, &uplo, &transa, &diag, &m, &n, alpha, a, &lda, b, &ldb);
}

void
strsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb)
{
    strsm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void
dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb)
{
    dtrsm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void
ctrsm(char side, char uplo, char transa, char diag, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb)
{
    ctrsm_(&side, &uplo, &transa, &diag, &m, &n, alpha, a, &lda, b, &ldb);
}

void
ztrsm(char side, char uplo, char transa, char diag, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb)
{
    ztrsm_(&side, &uplo, &transa, &diag, &m, &n, alpha, a, &lda, b, &ldb);
}

void
ssyr2k(char uplo, char transa, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc)
{
    ssyr2k_(&uplo, &transa, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
dsyr2k(char uplo, char transa, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    dsyr2k_(&uplo, &transa, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
csyr2k(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc)
{
    csyr2k_(&uplo, &transa, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
zsyr2k(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc)
{
    zsyr2k_(&uplo, &transa, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
ssyrk(char uplo, char transa, int n, int k, float alpha, float *a, int lda, float beta, float *c, int ldc)
{
    ssyrk_(&uplo, &transa, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void
dsyrk(char uplo, char transa, int n, int k, double alpha, double *a, int lda, double beta, double *c, int ldc)
{
    dsyrk_(&uplo, &transa, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void
csyrk(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *beta, complex *c, int ldc)
{
    csyrk_(&uplo, &transa, &n, &k, alpha, a, &lda, beta, c, &ldc);
}

void
zsyrk(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *beta, doublecomplex *c, int ldc)
{
    zsyrk_(&uplo, &transa, &n, &k, alpha, a, &lda, beta, c, &ldc);
}

void
strmv(char uplo, char transa, char diag, int n, float *a, int lda, float *x, int incx)
{
   strmv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
dtrmv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx)
{
   dtrmv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
ctrmv(char uplo, char transa, char diag, int n, complex *a, int lda, complex *x, int incx)
{
   ctrmv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
ztrmv(char uplo, char transa, char diag, int n, doublecomplex *a, int lda, doublecomplex *x, int incx)
{
   ztrmv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
strsv(char uplo, char transa, char diag, int n, float *a, int lda, float *x, int incx)
{
   strsv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
dtrsv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx)
{
   dtrsv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
ctrsv(char uplo, char transa, char diag, int n, complex *a, int lda, complex *x, int incx)
{
   ctrsv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
ztrsv(char uplo, char transa, char diag, int n, doublecomplex *a, int lda, doublecomplex *x, int incx)
{
   ztrsv_( &uplo, &transa, &diag, &n, a, &lda, x, &incx);
}

void
ssymm(char side, char uplo, int m, int n, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc)
{
   ssymm_( &side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
dsymm(char side, char uplo, int m, int n, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc)
{
   dsymm_( &side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
csymm(char side, char uplo, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc)
{
   csymm_( &side, &uplo, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
zsymm(char side, char uplo, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc)
{
   zsymm_( &side, &uplo, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
sger(int m, int n, float alpha, float *x, int incx, float *y, int incy, float *a, int lda)
{
   sger_( &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void
dger(int m, int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda)
{
   dger_( &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void
cgeru(int m, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda)
{
   cgeru_( &m, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
zgeru(int m, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda)
{
   zgeru_( &m, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
cgerc(int m, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda)
{
   cgerc_( &m, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
zgerc(int m, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda)
{
   zgerc_( &m, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
ssyr(char uplo, int n, float alpha, float *x, int incx, float *a, int lda)
{
   ssyr_( &uplo, &n, &alpha, x, &incx, a, &lda);
}

void
dsyr(char uplo, int n, double alpha, double *x, int incx, double *a, int lda)
{
   dsyr_( &uplo, &n, &alpha, x, &incx, a, &lda);
}

void
ssyr2(char uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *a, int lda)
{
   ssyr2_( &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void
dsyr2(char uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda)
{
   dsyr2_( &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void
cher(char uplo, int n, float alpha, complex *x, int incx, complex *a, int lda)
{
   cher_( &uplo, &n, &alpha, x, &incx, a, &lda);
}

void
zher(char uplo, int n, double alpha, doublecomplex *x, int incx, doublecomplex *a, int lda)
{
   zher_( &uplo, &n, &alpha, x, &incx, a, &lda);
}

void
cher2(char uplo, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a, int lda)
{
   cher2_( &uplo, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
zher2(char uplo, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a, int lda)
{
   zher2_( &uplo, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void
chemv(char uplo, int n, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy)
{
   chemv_( &uplo, &n, alpha, a, &lda, x, &incx, beta, y, &incy );
}

void
zhemv(char uplo, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy)
{
   zhemv_( &uplo, &n, alpha, a, &lda, x, &incx, beta, y, &incy );
}

void
stpmv(char uplo, char transa, char diag, int n, float *ap, float *x, int incx)
{
   stpmv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
dtpmv(char uplo, char transa, char diag, int n, double *ap, double *x, int incx)
{
   dtpmv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
ctpmv(char uplo, char transa, char diag, int n, complex *ap, complex *x, int incx)
{
   ctpmv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
ztpmv(char uplo, char transa, char diag, int n, doublecomplex *ap, doublecomplex *x, int incx)
{
   ztpmv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
stpsv(char uplo, char transa, char diag, int n, float *ap, float *x, int incx)
{
   stpsv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
dtpsv(char uplo, char transa, char diag, int n, double *ap, double *x, int incx)
{
   dtpsv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
ctpsv(char uplo, char transa, char diag, int n, complex *ap, complex *x, int incx)
{
   ctpsv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
ztpsv(char uplo, char transa, char diag, int n, doublecomplex *ap, doublecomplex *x, int incx)
{
   ztpsv_( &uplo, &transa, &diag, &n, ap, x, &incx);
}

void
sspr(char uplo, int n, float alpha, float *x, int incx, float *ap )
{
   sspr_( &uplo, &n, &alpha, x, &incx, ap );
}

void
dspr(char uplo, int n, double alpha, double *x, int incx, double *ap )
{
   dspr_( &uplo, &n, &alpha, x, &incx, ap );
}

void
sspmv(char uplo, int n, float alpha, float *ap, float *x, int incx, float beta, float *y, int incy)
{
   sspmv_( &uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy );
}

void
dspmv(char uplo, int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy)
{
   dspmv_( &uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy );
}

void
chpmv(char uplo, int n, complex *alpha, complex *ap, complex *x, int incx, complex *beta, complex *y, int incy)
{
   chpmv_( &uplo, &n, alpha, ap, x, &incx, beta, y, &incy );
}

void
zhpmv(char uplo, int n, doublecomplex *alpha, doublecomplex *ap, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy)
{
   zhpmv_( &uplo, &n, alpha, ap, x, &incx, beta, y, &incy );
}

void
chpr(char uplo, int n, float alpha, complex *x, int incx, complex *ap )
{
   chpr_( &uplo, &n, &alpha, x, &incx, ap );
}

void
zhpr(char uplo, int n, double alpha, doublecomplex *x, int incx, doublecomplex *ap )
{
   zhpr_( &uplo, &n, &alpha, x, &incx, ap );
}

void
sspr2(char uplo, int n, float alpha, float *x, int incx, float *y, int incy, float *a )
{
   sspr2_( &uplo, &n, &alpha, x, &incx, y, &incy, a );
}
void
dspr2(char uplo, int n, double alpha, double *x, int incx, double *y, int incy, double *a )
{
   dspr2_( &uplo, &n, &alpha, x, &incx, y, &incy, a );
}
void
chpr2(char uplo, int n, complex *alpha, complex *x, int incx, complex *y, int incy, complex *a )
{
   chpr2_( &uplo, &n, alpha, x, &incx, y, &incy, a );
}
void
zhpr2(char uplo, int n, doublecomplex *alpha, doublecomplex *x, int incx, doublecomplex *y, int incy, doublecomplex *a )
{
   zhpr2_( &uplo, &n, alpha, x, &incx, y, &incy, a );
}

void
sgbmv(char trans, int m, int n, int kl, int ku, float alpha, float *a, int inca, float *x, int incx, float beta, float *y, int incy )
{
   sgbmv_( &trans, &m, &n, &kl, &ku, &alpha, a, &inca, x, &incx, &beta, y, &incy );
}
void
dgbmv(char trans, int m, int n, int kl, int ku, double alpha, double *a, int inca, double *x, int incx, double beta, double *y, int incy )
{
   dgbmv_( &trans, &m, &n, &kl, &ku, &alpha, a, &inca, x, &incx, &beta, y, &incy );
}
void
cgbmv(char trans, int m, int n, int kl, int ku, complex *alpha, complex *a, int inca, complex *x, int incx, complex *beta, complex *y, int incy )
{
   cgbmv_( &trans, &m, &n, &kl, &ku, alpha, a, &inca, x, &incx, beta, y, &incy );
}
void
zgbmv(char trans, int m, int n, int kl, int ku, doublecomplex *alpha, doublecomplex *a, int inca, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy )
{
   zgbmv_( &trans, &m, &n, &kl, &ku, alpha, a, &inca, x, &incx, beta, y, &incy );
}

void
stbmv(char uplo, char trans, char diag, int n, int k, float *a, int lda, float *x, int incx )
{
   stbmv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
dtbmv(char uplo, char trans, char diag, int n, int k, double *a, int lda, double *x, int incx )
{
   dtbmv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
ctbmv(char uplo, char trans, char diag, int n, int k, complex *a, int lda, complex *x, int incx )
{
   ctbmv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
ztbmv(char uplo, char trans, char diag, int n, int k, doublecomplex *a, int lda, doublecomplex *x, int incx )
{
   ztbmv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
ssbmv(char uplo, int n, int k, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy )
{
   ssbmv_( &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy );
}

void
dsbmv(char uplo, int n, int k, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy )
{
   dsbmv_( &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy );
}

void
chbmv(char uplo, int n, int k, complex *alpha, complex *a, int lda, complex *x, int incx, complex *beta, complex *y, int incy )
{
   chbmv_( &uplo, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy );
}

void
zhbmv(char uplo, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *x, int incx, doublecomplex *beta, doublecomplex *y, int incy )
{
   zhbmv_( &uplo, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy );
}

void
stbsv(char uplo, char trans, char diag, int n, int k, float *a, int lda, float *x, int incx )
{
   stbsv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
dtbsv(char uplo, char trans, char diag, int n, int k, double *a, int lda, double *x, int incx )
{
   dtbsv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
ctbsv(char uplo, char trans, char diag, int n, int k, complex *a, int lda, complex *x, int incx )
{
   ctbsv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
ztbsv(char uplo, char trans, char diag, int n, int k, doublecomplex *a, int lda, doublecomplex *x, int incx )
{
   ztbsv_( &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx );
}

void
chemm(char side, char uplo, int m, int n, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc)
{
   chemm_( &side, &uplo, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
zhemm(char side, char uplo, int m, int n, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc)
{
   zhemm_( &side, &uplo, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

void
cherk(char uplo, char transa, int n, int k, float alpha, complex *a, int lda, float beta, complex *c, int ldc)
{
   cherk_( &uplo, &transa, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void
zherk(char uplo, char transa, int n, int k, double alpha, doublecomplex *a, int lda, double beta, doublecomplex *c, int ldc)
{
   zherk_( &uplo, &transa, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void
cher2k(char uplo, char transa, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, float beta, complex *c, int ldc)
{
   cher2k_( &uplo, &transa, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void
zher2k(char uplo, char transa, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, double beta, doublecomplex *c, int ldc)
{
   zher2k_( &uplo, &transa, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void sscal( int n, float alpha, float *x, int incx)
{
    sscal_(&n, &alpha, x, &incx);
}

void dscal( int n, double alpha, double *x, int incx)
{
    dscal_(&n, &alpha, x, &incx);
}

void cscal( int n, complex* alpha, complex *x, int incx)
{
    cscal_(&n, alpha, x, &incx);
}

void zscal( int n, doublecomplex* alpha, doublecomplex *x, int incx)
{
    zscal_(&n, alpha, x, &incx);
}

void csscal( int n, float alpha, complex *x, int incx)
{
    csscal_(&n, &alpha, x, &incx);
}

void zdscal( int n, double alpha, doublecomplex *x, int incx)
{
    zdscal_(&n, &alpha, x, &incx);
}

float sdot( int n, float *x, int incx,  float *y, int incy)
{
#ifdef __APPLE__
    return cblas_sdot(n, x, incx, y, incy);
#else
    return sdot_(&n, x, &incx, y, &incy);
#endif
}

double ddot( int n, double *x, int incx,  double *y, int incy)
{
#ifdef __APPLE__
    return cblas_ddot(n, x, incx, y, incy);
#else
    return ddot_(&n, x, &incx, y, &incy);
#endif
}

complex cdotu( int n, complex *x, int incx, complex *y, int incy)
{
    complex ans;

#if defined( _WIN32 ) || defined( _WIN64 )
        ans = cdotu_(&n, x, &incx, y, &incy);
    #elif defined( __APPLE__)
        cblas_cdotu_sub(n, x, incx, y, incy, &ans);
    #else
        cdotusub_(&n, x, &incx, y, &incy, &ans);
    #endif

    return ans;
}

doublecomplex zdotu( int n, doublecomplex *x, int incx,  doublecomplex *y, int incy)
{
    doublecomplex ans;

    #if defined( _WIN32 ) || defined( _WIN64 )
        ans = zdotu_(&n, x, &incx, y, &incy);
    #elif defined(__APPLE__)
        cblas_zdotu_sub(n, x, incx, y, incy, &ans);
    #else
        zdotusub_(&n, x, &incx, y, &incy, &ans);
    #endif

    return ans;
}

complex cdotc( int n, complex *x, int incx, complex *y, int incy)
{
    complex ans;

    #if defined( _WIN32 ) || defined( _WIN64 )
        ans = cdotc_(&n, x, &incx, y, &incy);
    #elif defined(__APPLE__)
        cblas_cdotc_sub(n, x, incx, y, incy, &ans);
    #else
        cdotcsub_(&n, x, &incx, y, &incy, &ans);
    #endif

    return ans;
}

doublecomplex zdotc( int n, doublecomplex *x, int incx,  doublecomplex *y, int incy)
{
    doublecomplex ans;

    #if defined( _WIN32 ) || defined( _WIN64 )
        ans = zdotc_(&n, x, &incx, y, &incy);
    #elif defined(__APPLE__)
        cblas_zdotc_sub(n, x, incx, y, incy, &ans);
    #else
        zdotcsub_(&n, x, &incx, y, &incy, &ans);
    #endif

    return ans;
}

void scopy( int n, float *x, int incx,  float *y, int incy)
{
    scopy_(&n, x, &incx, y, &incy);
}

void dcopy( int n, double *x, int incx,  double *y, int incy)
{
    dcopy_(&n, x, &incx, y, &incy);
}

void ccopy( int n, complex *x, int incx,  complex *y, int incy)
{
    ccopy_(&n, x, &incx, y, &incy);
}

void zcopy( int n, doublecomplex *x, int incx,  doublecomplex *y, int incy)
{
    zcopy_(&n, x, &incx, y, &incy);
}

void sswap( int n, float *x, int incx,  float *y, int incy)
{
    sswap_(&n, x, &incx, y, &incy);
}

void dswap( int n, double *x, int incx,  double *y, int incy)
{
    dswap_(&n, x, &incx, y, &incy);
}

void cswap( int n, complex *x, int incx,  complex *y, int incy)
{
    cswap_(&n, x, &incx, y, &incy);
}

void zswap( int n, doublecomplex *x, int incx,  doublecomplex *y, int incy)
{
    zswap_(&n, x, &incx, y, &incy);
}

void saxpy( int n, float alpha, float *x, int incx,  float *y, int incy)
{
    saxpy_(&n, &alpha, x, &incx, y, &incy);
}

void daxpy( int n, double alpha, double *x, int incx,  double *y, int incy)
{
    daxpy_(&n, &alpha, x, &incx, y, &incy);
}

void caxpy( int n, complex *alpha, complex *x, int incx,  complex *y, int incy)
{
    caxpy_(&n, alpha, x, &incx, y, &incy);
}

void zaxpy( int n, doublecomplex *alpha, doublecomplex *x, int incx,  doublecomplex *y, int incy)
{
    zaxpy_(&n, alpha, x, &incx, y, &incy);
}

void srotg(float *A, float *B, float *C, float *S)
{
    srotg_(A, B, C, S);
}

void drotg(double *A, double *B, double *C, double *S)
{
    drotg_(A, B, C, S);
}

void crotg(complex *A, complex *B, float *C, complex *S)
{
    crotg_(A, B, C, S);
}

void zrotg(doublecomplex *A, doublecomplex *B, double *C, doublecomplex *S)
{
    zrotg_(A, B, C, S);
}

void srotmg(float *D1, float *D2, float *X1, const float *Y1, float *PARAM)
{
    srotmg_(D1, D2, X1, (float*)Y1, PARAM);
}

void drotmg(double *D1, double *D2, double *X1, const double *Y1, double *PARAM)
{
    drotmg_(D1, D2, X1, (double*)Y1, PARAM);
}

void srot(int N, float *x, int incx, float *y, int incy, float c, float s)
{
    srot_(&N, x, &incx, y, &incy, &c, &s);
}

void drot(int N, double *x, int incx, double *y, int incy, double c, double s)
{
    drot_(&N, x, &incx, y, &incy, &c, &s);
}

void csrot(int N, complex *x, int incx, complex *y, int incy, float c, float s)
{
    csrot_(&N, x, &incx, y, &incy, &c, &s);
}

void zdrot(int N, doublecomplex *cx, int incx, doublecomplex *cy, int incy, double c, double s)
{
    zdrot_(&N, cx, &incx, cy, &incy, &c, &s);
}

void srotm(int N, float *X, int incx, float *Y, int incy, float* PARAM)
{
    srotm_(&N, X, &incx, Y, &incy, PARAM);
}

void drotm(int N, double *X, int incx, double *Y, int incy, double* PARAM)
{
    drotm_(&N, X, &incx, Y, &incy, PARAM);
}

int isamax( int n, float *x, int incx)
{
    return isamax_(&n, x, &incx);
}

int idamax( int n, double *x, int incx)
{
    return idamax_(&n, x, &incx);
}

int icamax( int n, complex *x, int incx)
{
    return icamax_(&n, x, &incx);
}

int izamax( int n, doublecomplex *x, int incx)
{
    return izamax_(&n, x, &incx);
}

float snrm2( int n, float *x, int incx)
{
#ifdef __APPLE__
    //On OSX passing negative values for incx can lead to a
    //a crash, so we catch it here (cf. Github issue #37).
    if (n < 1 || incx < 1) {
        return 0;
    }
    return cblas_snrm2(n, x, incx);
#else
    return snrm2_(&n, x, &incx);
#endif
}

double dnrm2( int n, double *x, int incx)
{
#ifdef __APPLE__
    //On OSX passing negative values for incx can lead to a
    //a crash, so we catch it here (cf. Github issue #37).
    if (n < 1 || incx < 1) {
        return 0;
    }
    return cblas_dnrm2(n, x, incx);
#else
    return dnrm2_(&n, x, &incx);
#endif
}

float scnrm2( int n, complex *x, int incx)
{
#ifdef __APPLE__
    //On OSX passing negative values for incx can lead to a
    //a crash, so we catch it here (cf. Github issue #37).
    if (n < 1 || incx < 1) {
        return 0;
    }
    return cblas_scnrm2(n, x, incx);
#else
    return scnrm2_(&n, x, &incx);
#endif
}

double dznrm2( int n, doublecomplex *x, int incx)
{
#ifdef __APPLE__
    //On OSX passing negative values for incx can lead to a
    //a crash, so we catch it here (cf. Github issue #37).
    if (n < 1 || incx < 1) {
        return 0;
    }
    return cblas_dznrm2(n, x, incx);
#else
    return dznrm2_(&n, x, &incx);
#endif
}

float sasum( int n, float *x, int incx)
{
#ifdef __APPLE__
    return cblas_sasum(n, x, incx);
#else
    return sasum_(&n, x, &incx);
#endif
}

double dasum( int n, double *x, int incx)
{
#ifdef __APPLE__
    return cblas_dasum(n, x, incx);
#else
    return dasum_(&n, x, &incx);
#endif
}

float scasum( int n, complex *x, int incx)
{
#ifdef __APPLE__
    return cblas_scasum(n, x, incx);
#else
    return scasum_(&n, x, &incx);
#endif
}

double dzasum( int n, doublecomplex *x, int incx)
{
#ifdef __APPLE__
    return cblas_dzasum(n, x, incx);
#else
    return dzasum_(&n, x, &incx);
#endif
}

#endif
