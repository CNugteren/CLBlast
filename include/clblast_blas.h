
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Netlib CBLAS interface to the CLBlast BLAS routines, performing all buffer
// copies automatically and running on the default OpenCL platform and device. For full control over
// performance, it is advised to use the regular clblast.h or clblast_c.h headers instead.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_BLAS_H_
#define CLBLAST_CLBLAST_BLAS_H_

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#ifdef _WIN32
  #ifdef COMPILING_DLL
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

// The C interface
#ifdef __cplusplus
extern "C" {
#endif

// =================================================================================================

// Matrix layout and transpose types
typedef enum Layout_ { kRowMajor = 101, kColMajor = 102 } Layout;
typedef enum Transpose_ { kNo = 111, kYes = 112, kConjugate = 113 } Transpose;
typedef enum Triangle_ { kUpper = 121, kLower = 122 } Triangle;
typedef enum Diagonal_ { kNonUnit = 131, kUnit = 132 } Diagonal;
typedef enum Side_ { kLeft = 141, kRight = 142 } Side;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
void PUBLIC_API cblas_srotg(float* sa,
                            float* sb,
                            float* sc,
                            float* ss);
void PUBLIC_API cblas_drotg(double* sa,
                            double* sb,
                            double* sc,
                            double* ss);

// Generate modified givens plane rotation: SROTMG/DROTMG
void PUBLIC_API cblas_srotmg(float* sd1,
                             float* sd2,
                             float* sx1,
                             const float* sy1,
                             float* sparam);
void PUBLIC_API cblas_drotmg(double* sd1,
                             double* sd2,
                             double* sx1,
                             const double* sy1,
                             double* sparam);

// Apply givens plane rotation: SROT/DROT
void PUBLIC_API cblas_srot(const int n,
                           float* x, const int x_inc,
                           float* y, const int y_inc,
                           const float cos,
                           const float sin);
void PUBLIC_API cblas_drot(const int n,
                           double* x, const int x_inc,
                           double* y, const int y_inc,
                           const double cos,
                           const double sin);

// Apply modified givens plane rotation: SROTM/DROTM
void PUBLIC_API cblas_srotm(const int n,
                            float* x, const int x_inc,
                            float* y, const int y_inc,
                            float* sparam);
void PUBLIC_API cblas_drotm(const int n,
                            double* x, const int x_inc,
                            double* y, const int y_inc,
                            double* sparam);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
void PUBLIC_API cblas_sswap(const int n,
                            float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dswap(const int n,
                            double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cswap(const int n,
                            float2* x, const int x_inc,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zswap(const int n,
                            double2* x, const int x_inc,
                            double2* y, const int y_inc);
void PUBLIC_API cblas_hswap(const int n,
                            half* x, const int x_inc,
                            half* y, const int y_inc);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
void PUBLIC_API cblas_sscal(const int n,
                            const float alpha,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dscal(const int n,
                            const double alpha,
                            double* x, const int x_inc);
void PUBLIC_API cblas_cscal(const int n,
                            const void* alpha,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_zscal(const int n,
                            const void* alpha,
                            double2* x, const int x_inc);
void PUBLIC_API cblas_hscal(const int n,
                            const void* alpha,
                            half* x, const int x_inc);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
void PUBLIC_API cblas_scopy(const int n,
                            const float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dcopy(const int n,
                            const double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_ccopy(const int n,
                            const float2* x, const int x_inc,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zcopy(const int n,
                            const double2* x, const int x_inc,
                            double2* y, const int y_inc);
void PUBLIC_API cblas_hcopy(const int n,
                            const half* x, const int x_inc,
                            half* y, const int y_inc);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
void PUBLIC_API cblas_saxpy(const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_daxpy(const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_caxpy(const int n,
                            const void* alpha,
                            const float2* x, const int x_inc,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zaxpy(const int n,
                            const void* alpha,
                            const double2* x, const int x_inc,
                            double2* y, const int y_inc);
void PUBLIC_API cblas_haxpy(const int n,
                            const void* alpha,
                            const half* x, const int x_inc,
                            half* y, const int y_inc);

// Dot product of two vectors: SDOT/DDOT/HDOT
void PUBLIC_API cblas_sdot(const int n,
                           float* dot,
                           const float* x, const int x_inc,
                           const float* y, const int y_inc);
void PUBLIC_API cblas_ddot(const int n,
                           double* dot,
                           const double* x, const int x_inc,
                           const double* y, const int y_inc);
void PUBLIC_API cblas_hdot(const int n,
                           half* dot,
                           const half* x, const int x_inc,
                           const half* y, const int y_inc);

// Dot product of two complex vectors: CDOTU/ZDOTU
void PUBLIC_API cblas_cdotu(const int n,
                            float2* dot,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc);
void PUBLIC_API cblas_zdotu(const int n,
                            double2* dot,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
void PUBLIC_API cblas_cdotc(const int n,
                            float2* dot,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc);
void PUBLIC_API cblas_zdotc(const int n,
                            double2* dot,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
void PUBLIC_API cblas_snrm2(const int n,
                            float* nrm2,
                            const float* x, const int x_inc);
void PUBLIC_API cblas_dnrm2(const int n,
                            double* nrm2,
                            const double* x, const int x_inc);
void PUBLIC_API cblas_scnrm2(const int n,
                            float2* nrm2,
                            const float2* x, const int x_inc);
void PUBLIC_API cblas_dznrm2(const int n,
                            double2* nrm2,
                            const double2* x, const int x_inc);
void PUBLIC_API cblas_hnrm2(const int n,
                            half* nrm2,
                            const half* x, const int x_inc);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
void PUBLIC_API cblas_sasum(const int n,
                            float* asum,
                            const float* x, const int x_inc);
void PUBLIC_API cblas_dasum(const int n,
                            double* asum,
                            const double* x, const int x_inc);
void PUBLIC_API cblas_scasum(const int n,
                            float2* asum,
                            const float2* x, const int x_inc);
void PUBLIC_API cblas_dzasum(const int n,
                            double2* asum,
                            const double2* x, const int x_inc);
void PUBLIC_API cblas_hasum(const int n,
                            half* asum,
                            const half* x, const int x_inc);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
void PUBLIC_API cblas_ssum(const int n,
                           float* sum,
                           const float* x, const int x_inc);
void PUBLIC_API cblas_dsum(const int n,
                           double* sum,
                           const double* x, const int x_inc);
void PUBLIC_API cblas_scsum(const int n,
                           float2* sum,
                           const float2* x, const int x_inc);
void PUBLIC_API cblas_dzsum(const int n,
                           double2* sum,
                           const double2* x, const int x_inc);
void PUBLIC_API cblas_hsum(const int n,
                           half* sum,
                           const half* x, const int x_inc);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
void PUBLIC_API cblas_isamax(const int n,
                            float* imax,
                            const float* x, const int x_inc);
void PUBLIC_API cblas_idamax(const int n,
                            double* imax,
                            const double* x, const int x_inc);
void PUBLIC_API cblas_icamax(const int n,
                            float2* imax,
                            const float2* x, const int x_inc);
void PUBLIC_API cblas_izamax(const int n,
                            double2* imax,
                            const double2* x, const int x_inc);
void PUBLIC_API cblas_ihamax(const int n,
                            half* imax,
                            const half* x, const int x_inc);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
void PUBLIC_API cblas_ismax(const int n,
                           float* imax,
                           const float* x, const int x_inc);
void PUBLIC_API cblas_idmax(const int n,
                           double* imax,
                           const double* x, const int x_inc);
void PUBLIC_API cblas_icmax(const int n,
                           float2* imax,
                           const float2* x, const int x_inc);
void PUBLIC_API cblas_izmax(const int n,
                           double2* imax,
                           const double2* x, const int x_inc);
void PUBLIC_API cblas_ihmax(const int n,
                           half* imax,
                           const half* x, const int x_inc);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
void PUBLIC_API cblas_ismin(const int n,
                           float* imin,
                           const float* x, const int x_inc);
void PUBLIC_API cblas_idmin(const int n,
                           double* imin,
                           const double* x, const int x_inc);
void PUBLIC_API cblas_icmin(const int n,
                           float2* imin,
                           const float2* x, const int x_inc);
void PUBLIC_API cblas_izmin(const int n,
                           double2* imin,
                           const double2* x, const int x_inc);
void PUBLIC_API cblas_ihmin(const int n,
                           half* imin,
                           const half* x, const int x_inc);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
void PUBLIC_API cblas_sgemv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dgemv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cgemv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* x, const int x_inc,
                            const void* beta,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zgemv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* x, const int x_inc,
                            const void* beta,
                            double2* y, const int y_inc);
void PUBLIC_API cblas_hgemv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* x, const int x_inc,
                            const void* beta,
                            half* y, const int y_inc);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
void PUBLIC_API cblas_sgbmv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dgbmv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cgbmv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* x, const int x_inc,
                            const void* beta,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zgbmv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* x, const int x_inc,
                            const void* beta,
                            double2* y, const int y_inc);
void PUBLIC_API cblas_hgbmv(const Layout layout, const Transpose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* x, const int x_inc,
                            const void* beta,
                            half* y, const int y_inc);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
void PUBLIC_API cblas_chemv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* x, const int x_inc,
                            const void* beta,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zhemv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* x, const int x_inc,
                            const void* beta,
                            double2* y, const int y_inc);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
void PUBLIC_API cblas_chbmv(const Layout layout, const Triangle triangle,
                            const int n, const int k,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* x, const int x_inc,
                            const void* beta,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zhbmv(const Layout layout, const Triangle triangle,
                            const int n, const int k,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* x, const int x_inc,
                            const void* beta,
                            double2* y, const int y_inc);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
void PUBLIC_API cblas_chpmv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const float2* ap,
                            const float2* x, const int x_inc,
                            const void* beta,
                            float2* y, const int y_inc);
void PUBLIC_API cblas_zhpmv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const double2* ap,
                            const double2* x, const int x_inc,
                            const void* beta,
                            double2* y, const int y_inc);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
void PUBLIC_API cblas_ssymv(const Layout layout, const Triangle triangle,
                            const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dsymv(const Layout layout, const Triangle triangle,
                            const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_hsymv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* x, const int x_inc,
                            const void* beta,
                            half* y, const int y_inc);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
void PUBLIC_API cblas_ssbmv(const Layout layout, const Triangle triangle,
                            const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dsbmv(const Layout layout, const Triangle triangle,
                            const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_hsbmv(const Layout layout, const Triangle triangle,
                            const int n, const int k,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* x, const int x_inc,
                            const void* beta,
                            half* y, const int y_inc);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
void PUBLIC_API cblas_sspmv(const Layout layout, const Triangle triangle,
                            const int n,
                            const float alpha,
                            const float* ap,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dspmv(const Layout layout, const Triangle triangle,
                            const int n,
                            const double alpha,
                            const double* ap,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_hspmv(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const half* ap,
                            const half* x, const int x_inc,
                            const void* beta,
                            half* y, const int y_inc);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
void PUBLIC_API cblas_strmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float2* a, const int a_ld,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double2* a, const int a_ld,
                            double2* x, const int x_inc);
void PUBLIC_API cblas_htrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const half* a, const int a_ld,
                            half* x, const int x_inc);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
void PUBLIC_API cblas_stbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const float2* a, const int a_ld,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const double2* a, const int a_ld,
                            double2* x, const int x_inc);
void PUBLIC_API cblas_htbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const half* a, const int a_ld,
                            half* x, const int x_inc);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
void PUBLIC_API cblas_stpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float* ap,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double* ap,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float2* ap,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double2* ap,
                            double2* x, const int x_inc);
void PUBLIC_API cblas_htpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const half* ap,
                            half* x, const int x_inc);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
void PUBLIC_API cblas_strsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float2* a, const int a_ld,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double2* a, const int a_ld,
                            double2* x, const int x_inc);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
void PUBLIC_API cblas_stbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const float2* a, const int a_ld,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n, const int k,
                            const double2* a, const int a_ld,
                            double2* x, const int x_inc);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
void PUBLIC_API cblas_stpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float* ap,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double* ap,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const float2* ap,
                            float2* x, const int x_inc);
void PUBLIC_API cblas_ztpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int n,
                            const double2* ap,
                            double2* x, const int x_inc);

// General rank-1 matrix update: SGER/DGER/HGER
void PUBLIC_API cblas_sger(const Layout layout,
                           const int m, const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           const float* y, const int y_inc,
                           float* a, const int a_ld);
void PUBLIC_API cblas_dger(const Layout layout,
                           const int m, const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           const double* y, const int y_inc,
                           double* a, const int a_ld);
void PUBLIC_API cblas_hger(const Layout layout,
                           const int m, const int n,
                           const void* alpha,
                           const half* x, const int x_inc,
                           const half* y, const int y_inc,
                           half* a, const int a_ld);

// General rank-1 complex matrix update: CGERU/ZGERU
void PUBLIC_API cblas_cgeru(const Layout layout,
                            const int m, const int n,
                            const void* alpha,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc,
                            float2* a, const int a_ld);
void PUBLIC_API cblas_zgeru(const Layout layout,
                            const int m, const int n,
                            const void* alpha,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc,
                            double2* a, const int a_ld);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
void PUBLIC_API cblas_cgerc(const Layout layout,
                            const int m, const int n,
                            const void* alpha,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc,
                            float2* a, const int a_ld);
void PUBLIC_API cblas_zgerc(const Layout layout,
                            const int m, const int n,
                            const void* alpha,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc,
                            double2* a, const int a_ld);

// Hermitian rank-1 matrix update: CHER/ZHER
void PUBLIC_API cblas_cher(const Layout layout, const Triangle triangle,
                           const int n,
                           const float alpha,
                           const float2* x, const int x_inc,
                           float2* a, const int a_ld);
void PUBLIC_API cblas_zher(const Layout layout, const Triangle triangle,
                           const int n,
                           const double alpha,
                           const double2* x, const int x_inc,
                           double2* a, const int a_ld);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
void PUBLIC_API cblas_chpr(const Layout layout, const Triangle triangle,
                           const int n,
                           const float alpha,
                           const float2* x, const int x_inc,
                           float2* ap);
void PUBLIC_API cblas_zhpr(const Layout layout, const Triangle triangle,
                           const int n,
                           const double alpha,
                           const double2* x, const int x_inc,
                           double2* ap);

// Hermitian rank-2 matrix update: CHER2/ZHER2
void PUBLIC_API cblas_cher2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc,
                            float2* a, const int a_ld);
void PUBLIC_API cblas_zher2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc,
                            double2* a, const int a_ld);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
void PUBLIC_API cblas_chpr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const float2* x, const int x_inc,
                            const float2* y, const int y_inc,
                            float2* ap);
void PUBLIC_API cblas_zhpr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const double2* x, const int x_inc,
                            const double2* y, const int y_inc,
                            double2* ap);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
void PUBLIC_API cblas_ssyr(const Layout layout, const Triangle triangle,
                           const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           float* a, const int a_ld);
void PUBLIC_API cblas_dsyr(const Layout layout, const Triangle triangle,
                           const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           double* a, const int a_ld);
void PUBLIC_API cblas_hsyr(const Layout layout, const Triangle triangle,
                           const int n,
                           const void* alpha,
                           const half* x, const int x_inc,
                           half* a, const int a_ld);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
void PUBLIC_API cblas_sspr(const Layout layout, const Triangle triangle,
                           const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           float* ap);
void PUBLIC_API cblas_dspr(const Layout layout, const Triangle triangle,
                           const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           double* ap);
void PUBLIC_API cblas_hspr(const Layout layout, const Triangle triangle,
                           const int n,
                           const void* alpha,
                           const half* x, const int x_inc,
                           half* ap);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
void PUBLIC_API cblas_ssyr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            const float* y, const int y_inc,
                            float* a, const int a_ld);
void PUBLIC_API cblas_dsyr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            const double* y, const int y_inc,
                            double* a, const int a_ld);
void PUBLIC_API cblas_hsyr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const half* x, const int x_inc,
                            const half* y, const int y_inc,
                            half* a, const int a_ld);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
void PUBLIC_API cblas_sspr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            const float* y, const int y_inc,
                            float* ap);
void PUBLIC_API cblas_dspr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            const double* y, const int y_inc,
                            double* ap);
void PUBLIC_API cblas_hspr2(const Layout layout, const Triangle triangle,
                            const int n,
                            const void* alpha,
                            const half* x, const int x_inc,
                            const half* y, const int y_inc,
                            half* ap);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
void PUBLIC_API cblas_sgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const int m, const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* b, const int b_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const int m, const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* b, const int b_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_cgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const int m, const int n, const int k,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* b, const int b_ld,
                            const void* beta,
                            float2* c, const int c_ld);
void PUBLIC_API cblas_zgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const int m, const int n, const int k,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* b, const int b_ld,
                            const void* beta,
                            double2* c, const int c_ld);
void PUBLIC_API cblas_hgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const int m, const int n, const int k,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* b, const int b_ld,
                            const void* beta,
                            half* c, const int c_ld);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
void PUBLIC_API cblas_ssymm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* b, const int b_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dsymm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* b, const int b_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_csymm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* b, const int b_ld,
                            const void* beta,
                            float2* c, const int c_ld);
void PUBLIC_API cblas_zsymm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* b, const int b_ld,
                            const void* beta,
                            double2* c, const int c_ld);
void PUBLIC_API cblas_hsymm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const half* b, const int b_ld,
                            const void* beta,
                            half* c, const int c_ld);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
void PUBLIC_API cblas_chemm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const float2* b, const int b_ld,
                            const void* beta,
                            float2* c, const int c_ld);
void PUBLIC_API cblas_zhemm(const Layout layout, const Side side, const Triangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const double2* b, const int b_ld,
                            const void* beta,
                            double2* c, const int c_ld);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
void PUBLIC_API cblas_ssyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_csyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            const void* beta,
                            float2* c, const int c_ld);
void PUBLIC_API cblas_zsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            const void* beta,
                            double2* c, const int c_ld);
void PUBLIC_API cblas_hsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const void* alpha,
                            const half* a, const int a_ld,
                            const void* beta,
                            half* c, const int c_ld);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
void PUBLIC_API cblas_cherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const float alpha,
                            const float2* a, const int a_ld,
                            const float beta,
                            float2* c, const int c_ld);
void PUBLIC_API cblas_zherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                            const int n, const int k,
                            const double alpha,
                            const double2* a, const int a_ld,
                            const double beta,
                            double2* c, const int c_ld);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
void PUBLIC_API cblas_ssyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const float alpha,
                             const float* a, const int a_ld,
                             const float* b, const int b_ld,
                             const float beta,
                             float* c, const int c_ld);
void PUBLIC_API cblas_dsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const double alpha,
                             const double* a, const int a_ld,
                             const double* b, const int b_ld,
                             const double beta,
                             double* c, const int c_ld);
void PUBLIC_API cblas_csyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const float2* a, const int a_ld,
                             const float2* b, const int b_ld,
                             const void* beta,
                             float2* c, const int c_ld);
void PUBLIC_API cblas_zsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const double2* a, const int a_ld,
                             const double2* b, const int b_ld,
                             const void* beta,
                             double2* c, const int c_ld);
void PUBLIC_API cblas_hsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const half* a, const int a_ld,
                             const half* b, const int b_ld,
                             const void* beta,
                             half* c, const int c_ld);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
void PUBLIC_API cblas_cher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const float2* a, const int a_ld,
                             const float2* b, const int b_ld,
                             const float beta,
                             float2* c, const int c_ld);
void PUBLIC_API cblas_zher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const double2* a, const int a_ld,
                             const double2* b, const int b_ld,
                             const double beta,
                             double2* c, const int c_ld);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
void PUBLIC_API cblas_strmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            float* b, const int b_ld);
void PUBLIC_API cblas_dtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            double* b, const int b_ld);
void PUBLIC_API cblas_ctrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            float2* b, const int b_ld);
void PUBLIC_API cblas_ztrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            double2* b, const int b_ld);
void PUBLIC_API cblas_htrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const half* a, const int a_ld,
                            half* b, const int b_ld);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM/HTRSM
void PUBLIC_API cblas_strsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            float* b, const int b_ld);
void PUBLIC_API cblas_dtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            double* b, const int b_ld);
void PUBLIC_API cblas_ctrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const float2* a, const int a_ld,
                            float2* b, const int b_ld);
void PUBLIC_API cblas_ztrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const double2* a, const int a_ld,
                            double2* b, const int b_ld);
void PUBLIC_API cblas_htrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const half* a, const int a_ld,
                            half* b, const int b_ld);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
void PUBLIC_API cblas_somatcopy(const Layout layout, const Transpose a_transpose,
                                const int m, const int n,
                                const float alpha,
                                const float* a, const int a_ld,
                                float* b, const int b_ld);
void PUBLIC_API cblas_domatcopy(const Layout layout, const Transpose a_transpose,
                                const int m, const int n,
                                const double alpha,
                                const double* a, const int a_ld,
                                double* b, const int b_ld);
void PUBLIC_API cblas_comatcopy(const Layout layout, const Transpose a_transpose,
                                const int m, const int n,
                                const void* alpha,
                                const float2* a, const int a_ld,
                                float2* b, const int b_ld);
void PUBLIC_API cblas_zomatcopy(const Layout layout, const Transpose a_transpose,
                                const int m, const int n,
                                const void* alpha,
                                const double2* a, const int a_ld,
                                double2* b, const int b_ld);
void PUBLIC_API cblas_homatcopy(const Layout layout, const Transpose a_transpose,
                                const int m, const int n,
                                const void* alpha,
                                const half* a, const int a_ld,
                                half* b, const int b_ld);
                                half* b, const size_t b_offset, const size_t b_ld);

// =================================================================================================

#ifdef __cplusplus
} // extern "C"
#endif

// CLBLAST_CLBLAST_BLAS_H_
#endif
