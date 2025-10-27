
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Netlib CBLAS interface to the CLBlast BLAS routines, performing all buffer
// copies automatically and running on the default OpenCL platform and device. For full control over
// performance, it is advised to use the regular clblast.h or clblast_c.h headers instead.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_NETLIB_C_H_
#define CLBLAST_CLBLAST_NETLIB_C_H_

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(CLBLAST_DLL)
#if defined(COMPILING_DLL)
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
typedef enum CLBlastLayout_ { CLBlastLayoutRowMajor = 101, CLBlastLayoutColMajor = 102 } CLBlastLayout;
typedef enum CLBlastTranspose_ {
  CLBlastTransposeNo = 111,
  CLBlastTransposeYes = 112,
  CLBlastTransposeConjugate = 113
} CLBlastTranspose;
typedef enum CLBlastTriangle_ { CLBlastTriangleUpper = 121, CLBlastTriangleLower = 122 } CLBlastTriangle;
typedef enum CLBlastDiagonal_ { CLBlastDiagonalNonUnit = 131, CLBlastDiagonalUnit = 132 } CLBlastDiagonal;
typedef enum CLBlastSide_ { CLBlastSideLeft = 141, CLBlastSideRight = 142 } CLBlastSide;
typedef enum CLBlastKernelMode_ {
  CLBlastKernelModeCrossCorrelation = 141,
  CLBlastKernelModeConvolution = 152
} CLBlastKernelMode;

// For full compatibility with CBLAS
typedef CLBlastLayout CBLAS_ORDER;
typedef CLBlastTranspose CBLAS_TRANSPOSE;
typedef CLBlastTriangle CBLAS_UPLO;
typedef CLBlastDiagonal CBLAS_DIAG;
typedef CLBlastSide CBLAS_SIDE;
#define CblasRowMajor CLBlastLayoutRowMajor
#define CblasColMajor CLBlastLayoutColMajor
#define CblasNoTrans CLBlastTransposeNo
#define CblasTrans CLBlastTransposeYes
#define CblasConjTrans CLBlastTransposeConjugate
#define CblasUpper CLBlastTriangleUpper
#define CblasLower CLBlastTriangleLower
#define CblasNonUnit CLBlastDiagonalNonUnit
#define CblasUnit CLBlastDiagonalUnit
#define CblasLeft CLBlastSideLeft
#define CblasRight CLBlastSideRight

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
void PUBLIC_API cblas_srotg(float* sa, float* sb, float* sc, float* ss);
void PUBLIC_API cblas_drotg(double* sa, double* sb, double* sc, double* ss);

// Generate modified givens plane rotation: SROTMG/DROTMG
void PUBLIC_API cblas_srotmg(float* sd1, float* sd2, float* sx1, float sy1, float* sparam);
void PUBLIC_API cblas_drotmg(double* sd1, double* sd2, double* sx1, double sy1, double* sparam);

// Apply givens plane rotation: SROT/DROT
void PUBLIC_API cblas_srot(int n, float* x, int x_inc, float* y, int y_inc, float cos, float sin);
void PUBLIC_API cblas_drot(int n, double* x, int x_inc, double* y, int y_inc, double cos, double sin);

// Apply modified givens plane rotation: SROTM/DROTM
void PUBLIC_API cblas_srotm(int n, float* x, int x_inc, float* y, int y_inc, float* sparam);
void PUBLIC_API cblas_drotm(int n, double* x, int x_inc, double* y, int y_inc, double* sparam);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
void PUBLIC_API cblas_sswap(int n, float* x, int x_inc, float* y, int y_inc);
void PUBLIC_API cblas_dswap(int n, double* x, int x_inc, double* y, int y_inc);
void PUBLIC_API cblas_cswap(int n, void* x, int x_inc, void* y, int y_inc);
void PUBLIC_API cblas_zswap(int n, void* x, int x_inc, void* y, int y_inc);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
void PUBLIC_API cblas_sscal(int n, float alpha, float* x, int x_inc);
void PUBLIC_API cblas_dscal(int n, double alpha, double* x, int x_inc);
void PUBLIC_API cblas_cscal(int n, const void* alpha, void* x, int x_inc);
void PUBLIC_API cblas_zscal(int n, const void* alpha, void* x, int x_inc);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
void PUBLIC_API cblas_scopy(int n, const float* x, int x_inc, float* y, int y_inc);
void PUBLIC_API cblas_dcopy(int n, const double* x, int x_inc, double* y, int y_inc);
void PUBLIC_API cblas_ccopy(int n, const void* x, int x_inc, void* y, int y_inc);
void PUBLIC_API cblas_zcopy(int n, const void* x, int x_inc, void* y, int y_inc);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
void PUBLIC_API cblas_saxpy(int n, float alpha, const float* x, int x_inc, float* y, int y_inc);
void PUBLIC_API cblas_daxpy(int n, double alpha, const double* x, int x_inc, double* y, int y_inc);
void PUBLIC_API cblas_caxpy(int n, const void* alpha, const void* x, int x_inc, void* y, int y_inc);
void PUBLIC_API cblas_zaxpy(int n, const void* alpha, const void* x, int x_inc, void* y, int y_inc);

// Dot product of two vectors: SDOT/DDOT/HDOT
float PUBLIC_API cblas_sdot(int n, const float* x, int x_inc, const float* y, int y_inc);
double PUBLIC_API cblas_ddot(int n, const double* x, int x_inc, const double* y, int y_inc);

// Dot product of two complex vectors: CDOTU/ZDOTU
void PUBLIC_API cblas_cdotu_sub(int n, const void* x, int x_inc, const void* y, int y_inc, void* dot);
void PUBLIC_API cblas_zdotu_sub(int n, const void* x, int x_inc, const void* y, int y_inc, void* dot);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
void PUBLIC_API cblas_cdotc_sub(int n, const void* x, int x_inc, const void* y, int y_inc, void* dot);
void PUBLIC_API cblas_zdotc_sub(int n, const void* x, int x_inc, const void* y, int y_inc, void* dot);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
float PUBLIC_API cblas_snrm2(int n, const float* x, int x_inc);
double PUBLIC_API cblas_dnrm2(int n, const double* x, int x_inc);
float PUBLIC_API cblas_scnrm2(int n, const void* x, int x_inc);
double PUBLIC_API cblas_dznrm2(int n, const void* x, int x_inc);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
float PUBLIC_API cblas_sasum(int n, const float* x, int x_inc);
double PUBLIC_API cblas_dasum(int n, const double* x, int x_inc);
float PUBLIC_API cblas_scasum(int n, const void* x, int x_inc);
double PUBLIC_API cblas_dzasum(int n, const void* x, int x_inc);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
float PUBLIC_API cblas_ssum(int n, const float* x, int x_inc);
double PUBLIC_API cblas_dsum(int n, const double* x, int x_inc);
float PUBLIC_API cblas_scsum(int n, const void* x, int x_inc);
double PUBLIC_API cblas_dzsum(int n, const void* x, int x_inc);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
int PUBLIC_API cblas_isamax(int n, const float* x, int x_inc);
int PUBLIC_API cblas_idamax(int n, const double* x, int x_inc);
int PUBLIC_API cblas_icamax(int n, const void* x, int x_inc);
int PUBLIC_API cblas_izamax(int n, const void* x, int x_inc);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
int PUBLIC_API cblas_isamin(int n, const float* x, int x_inc);
int PUBLIC_API cblas_idamin(int n, const double* x, int x_inc);
int PUBLIC_API cblas_icamin(int n, const void* x, int x_inc);
int PUBLIC_API cblas_izamin(int n, const void* x, int x_inc);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
int PUBLIC_API cblas_ismax(int n, const float* x, int x_inc);
int PUBLIC_API cblas_idmax(int n, const double* x, int x_inc);
int PUBLIC_API cblas_icmax(int n, const void* x, int x_inc);
int PUBLIC_API cblas_izmax(int n, const void* x, int x_inc);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
int PUBLIC_API cblas_ismin(int n, const float* x, int x_inc);
int PUBLIC_API cblas_idmin(int n, const double* x, int x_inc);
int PUBLIC_API cblas_icmin(int n, const void* x, int x_inc);
int PUBLIC_API cblas_izmin(int n, const void* x, int x_inc);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
void PUBLIC_API cblas_sgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, float alpha,
                            const float* a, int a_ld, const float* x, int x_inc, float beta, float* y, int y_inc);
void PUBLIC_API cblas_dgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, double alpha,
                            const double* a, int a_ld, const double* x, int x_inc, double beta, double* y, int y_inc);
void PUBLIC_API cblas_cgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, const void* alpha,
                            const void* a, int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);
void PUBLIC_API cblas_zgemv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, const void* alpha,
                            const void* a, int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
void PUBLIC_API cblas_sgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, int kl, int ku,
                            float alpha, const float* a, int a_ld, const float* x, int x_inc, float beta, float* y,
                            int y_inc);
void PUBLIC_API cblas_dgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, int kl, int ku,
                            double alpha, const double* a, int a_ld, const double* x, int x_inc, double beta, double* y,
                            int y_inc);
void PUBLIC_API cblas_cgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, int kl, int ku,
                            const void* alpha, const void* a, int a_ld, const void* x, int x_inc, const void* beta,
                            void* y, int y_inc);
void PUBLIC_API cblas_zgbmv(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, int kl, int ku,
                            const void* alpha, const void* a, int a_ld, const void* x, int x_inc, const void* beta,
                            void* y, int y_inc);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
void PUBLIC_API cblas_chemv(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* a,
                            int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);
void PUBLIC_API cblas_zhemv(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* a,
                            int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
void PUBLIC_API cblas_chbmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, int k, const void* alpha,
                            const void* a, int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);
void PUBLIC_API cblas_zhbmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, int k, const void* alpha,
                            const void* a, int a_ld, const void* x, int x_inc, const void* beta, void* y, int y_inc);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
void PUBLIC_API cblas_chpmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* ap,
                            const void* x, int x_inc, const void* beta, void* y, int y_inc);
void PUBLIC_API cblas_zhpmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* ap,
                            const void* x, int x_inc, const void* beta, void* y, int y_inc);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
void PUBLIC_API cblas_ssymv(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* a,
                            int a_ld, const float* x, int x_inc, float beta, float* y, int y_inc);
void PUBLIC_API cblas_dsymv(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* a,
                            int a_ld, const double* x, int x_inc, double beta, double* y, int y_inc);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
void PUBLIC_API cblas_ssbmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, int k, float alpha, const float* a,
                            int a_ld, const float* x, int x_inc, float beta, float* y, int y_inc);
void PUBLIC_API cblas_dsbmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, int k, double alpha, const double* a,
                            int a_ld, const double* x, int x_inc, double beta, double* y, int y_inc);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
void PUBLIC_API cblas_sspmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* ap,
                            const float* x, int x_inc, float beta, float* y, int y_inc);
void PUBLIC_API cblas_dspmv(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* ap,
                            const double* x, int x_inc, double beta, double* y, int y_inc);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
void PUBLIC_API cblas_strmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const float* a, int a_ld, float* x, int x_inc);
void PUBLIC_API cblas_dtrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const double* a, int a_ld, double* x, int x_inc);
void PUBLIC_API cblas_ctrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* a, int a_ld, void* x, int x_inc);
void PUBLIC_API cblas_ztrmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* a, int a_ld, void* x, int x_inc);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
void PUBLIC_API cblas_stbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const float* a, int a_ld, float* x, int x_inc);
void PUBLIC_API cblas_dtbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const double* a, int a_ld, double* x, int x_inc);
void PUBLIC_API cblas_ctbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const void* a, int a_ld, void* x, int x_inc);
void PUBLIC_API cblas_ztbmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const void* a, int a_ld, void* x, int x_inc);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
void PUBLIC_API cblas_stpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const float* ap, float* x, int x_inc);
void PUBLIC_API cblas_dtpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const double* ap, double* x, int x_inc);
void PUBLIC_API cblas_ctpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* ap, void* x, int x_inc);
void PUBLIC_API cblas_ztpmv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* ap, void* x, int x_inc);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
void PUBLIC_API cblas_strsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const float* a, int a_ld, float* x, int x_inc);
void PUBLIC_API cblas_dtrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const double* a, int a_ld, double* x, int x_inc);
void PUBLIC_API cblas_ctrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* a, int a_ld, void* x, int x_inc);
void PUBLIC_API cblas_ztrsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* a, int a_ld, void* x, int x_inc);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
void PUBLIC_API cblas_stbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const float* a, int a_ld, float* x, int x_inc);
void PUBLIC_API cblas_dtbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const double* a, int a_ld, double* x, int x_inc);
void PUBLIC_API cblas_ctbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const void* a, int a_ld, void* x, int x_inc);
void PUBLIC_API cblas_ztbsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, int k, const void* a, int a_ld, void* x, int x_inc);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
void PUBLIC_API cblas_stpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const float* ap, float* x, int x_inc);
void PUBLIC_API cblas_dtpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const double* ap, double* x, int x_inc);
void PUBLIC_API cblas_ctpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* ap, void* x, int x_inc);
void PUBLIC_API cblas_ztpsv(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose,
                            CLBlastDiagonal diagonal, int n, const void* ap, void* x, int x_inc);

// General rank-1 matrix update: SGER/DGER/HGER
void PUBLIC_API cblas_sger(CLBlastLayout layout, int m, int n, float alpha, const float* x, int x_inc, const float* y,
                           int y_inc, float* a, int a_ld);
void PUBLIC_API cblas_dger(CLBlastLayout layout, int m, int n, double alpha, const double* x, int x_inc,
                           const double* y, int y_inc, double* a, int a_ld);

// General rank-1 complex matrix update: CGERU/ZGERU
void PUBLIC_API cblas_cgeru(CLBlastLayout layout, int m, int n, const void* alpha, const void* x, int x_inc,
                            const void* y, int y_inc, void* a, int a_ld);
void PUBLIC_API cblas_zgeru(CLBlastLayout layout, int m, int n, const void* alpha, const void* x, int x_inc,
                            const void* y, int y_inc, void* a, int a_ld);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
void PUBLIC_API cblas_cgerc(CLBlastLayout layout, int m, int n, const void* alpha, const void* x, int x_inc,
                            const void* y, int y_inc, void* a, int a_ld);
void PUBLIC_API cblas_zgerc(CLBlastLayout layout, int m, int n, const void* alpha, const void* x, int x_inc,
                            const void* y, int y_inc, void* a, int a_ld);

// Hermitian rank-1 matrix update: CHER/ZHER
void PUBLIC_API cblas_cher(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const void* x, int x_inc,
                           void* a, int a_ld);
void PUBLIC_API cblas_zher(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const void* x,
                           int x_inc, void* a, int a_ld);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
void PUBLIC_API cblas_chpr(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const void* x, int x_inc,
                           void* ap);
void PUBLIC_API cblas_zhpr(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const void* x,
                           int x_inc, void* ap);

// Hermitian rank-2 matrix update: CHER2/ZHER2
void PUBLIC_API cblas_cher2(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* x,
                            int x_inc, const void* y, int y_inc, void* a, int a_ld);
void PUBLIC_API cblas_zher2(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* x,
                            int x_inc, const void* y, int y_inc, void* a, int a_ld);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
void PUBLIC_API cblas_chpr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* x,
                            int x_inc, const void* y, int y_inc, void* ap);
void PUBLIC_API cblas_zhpr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, const void* alpha, const void* x,
                            int x_inc, const void* y, int y_inc, void* ap);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
void PUBLIC_API cblas_ssyr(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* x,
                           int x_inc, float* a, int a_ld);
void PUBLIC_API cblas_dsyr(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* x,
                           int x_inc, double* a, int a_ld);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
void PUBLIC_API cblas_sspr(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* x,
                           int x_inc, float* ap);
void PUBLIC_API cblas_dspr(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* x,
                           int x_inc, double* ap);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
void PUBLIC_API cblas_ssyr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* x,
                            int x_inc, const float* y, int y_inc, float* a, int a_ld);
void PUBLIC_API cblas_dsyr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* x,
                            int x_inc, const double* y, int y_inc, double* a, int a_ld);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
void PUBLIC_API cblas_sspr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, float alpha, const float* x,
                            int x_inc, const float* y, int y_inc, float* ap);
void PUBLIC_API cblas_dspr2(CLBlastLayout layout, CLBlastTriangle triangle, int n, double alpha, const double* x,
                            int x_inc, const double* y, int y_inc, double* ap);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
void PUBLIC_API cblas_sgemm(CLBlastLayout layout, CLBlastTranspose a_transpose, CLBlastTranspose b_transpose, int m,
                            int n, int k, float alpha, const float* a, int a_ld, const float* b, int b_ld, float beta,
                            float* c, int c_ld);
void PUBLIC_API cblas_dgemm(CLBlastLayout layout, CLBlastTranspose a_transpose, CLBlastTranspose b_transpose, int m,
                            int n, int k, double alpha, const double* a, int a_ld, const double* b, int b_ld,
                            double beta, double* c, int c_ld);
void PUBLIC_API cblas_cgemm(CLBlastLayout layout, CLBlastTranspose a_transpose, CLBlastTranspose b_transpose, int m,
                            int n, int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld,
                            const void* beta, void* c, int c_ld);
void PUBLIC_API cblas_zgemm(CLBlastLayout layout, CLBlastTranspose a_transpose, CLBlastTranspose b_transpose, int m,
                            int n, int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld,
                            const void* beta, void* c, int c_ld);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
void PUBLIC_API cblas_ssymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n, float alpha,
                            const float* a, int a_ld, const float* b, int b_ld, float beta, float* c, int c_ld);
void PUBLIC_API cblas_dsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n,
                            double alpha, const double* a, int a_ld, const double* b, int b_ld, double beta, double* c,
                            int c_ld);
void PUBLIC_API cblas_csymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n,
                            const void* alpha, const void* a, int a_ld, const void* b, int b_ld, const void* beta,
                            void* c, int c_ld);
void PUBLIC_API cblas_zsymm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n,
                            const void* alpha, const void* a, int a_ld, const void* b, int b_ld, const void* beta,
                            void* c, int c_ld);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
void PUBLIC_API cblas_chemm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n,
                            const void* alpha, const void* a, int a_ld, const void* b, int b_ld, const void* beta,
                            void* c, int c_ld);
void PUBLIC_API cblas_zhemm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle, int m, int n,
                            const void* alpha, const void* a, int a_ld, const void* b, int b_ld, const void* beta,
                            void* c, int c_ld);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
void PUBLIC_API cblas_ssyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            float alpha, const float* a, int a_ld, float beta, float* c, int c_ld);
void PUBLIC_API cblas_dsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            double alpha, const double* a, int a_ld, double beta, double* c, int c_ld);
void PUBLIC_API cblas_csyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            const void* alpha, const void* a, int a_ld, const void* beta, void* c, int c_ld);
void PUBLIC_API cblas_zsyrk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            const void* alpha, const void* a, int a_ld, const void* beta, void* c, int c_ld);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
void PUBLIC_API cblas_cherk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            float alpha, const void* a, int a_ld, float beta, void* c, int c_ld);
void PUBLIC_API cblas_zherk(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose a_transpose, int n, int k,
                            double alpha, const void* a, int a_ld, double beta, void* c, int c_ld);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
void PUBLIC_API cblas_ssyr2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, float alpha, const float* a, int a_ld, const float* b, int b_ld, float beta,
                             float* c, int c_ld);
void PUBLIC_API cblas_dsyr2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, double alpha, const double* a, int a_ld, const double* b, int b_ld, double beta,
                             double* c, int c_ld);
void PUBLIC_API cblas_csyr2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld,
                             const void* beta, void* c, int c_ld);
void PUBLIC_API cblas_zsyr2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld,
                             const void* beta, void* c, int c_ld);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
void PUBLIC_API cblas_cher2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld, float beta,
                             void* c, int c_ld);
void PUBLIC_API cblas_zher2k(CLBlastLayout layout, CLBlastTriangle triangle, CLBlastTranspose ab_transpose, int n,
                             int k, const void* alpha, const void* a, int a_ld, const void* b, int b_ld, double beta,
                             void* c, int c_ld);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
void PUBLIC_API cblas_strmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, float alpha,
                            const float* a, int a_ld, float* b, int b_ld);
void PUBLIC_API cblas_dtrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, double alpha,
                            const double* a, int a_ld, double* b, int b_ld);
void PUBLIC_API cblas_ctrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, const void* alpha,
                            const void* a, int a_ld, void* b, int b_ld);
void PUBLIC_API cblas_ztrmm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, const void* alpha,
                            const void* a, int a_ld, void* b, int b_ld);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
void PUBLIC_API cblas_strsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, float alpha,
                            const float* a, int a_ld, float* b, int b_ld);
void PUBLIC_API cblas_dtrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, double alpha,
                            const double* a, int a_ld, double* b, int b_ld);
void PUBLIC_API cblas_ctrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, const void* alpha,
                            const void* a, int a_ld, void* b, int b_ld);
void PUBLIC_API cblas_ztrsm(CLBlastLayout layout, CLBlastSide side, CLBlastTriangle triangle,
                            CLBlastTranspose a_transpose, CLBlastDiagonal diagonal, int m, int n, const void* alpha,
                            const void* a, int a_ld, void* b, int b_ld);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
void PUBLIC_API cblas_shad(int n, float alpha, const float* x, int x_inc, const float* y, int y_inc, float beta,
                           float* z, int z_inc);
void PUBLIC_API cblas_dhad(int n, double alpha, const double* x, int x_inc, const double* y, int y_inc, double beta,
                           double* z, int z_inc);
void PUBLIC_API cblas_chad(int n, const void* alpha, const void* x, int x_inc, const void* y, int y_inc,
                           const void* beta, void* z, int z_inc);
void PUBLIC_API cblas_zhad(int n, const void* alpha, const void* x, int x_inc, const void* y, int y_inc,
                           const void* beta, void* z, int z_inc);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
void PUBLIC_API cblas_somatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, float alpha,
                                const float* a, int a_ld, float* b, int b_ld);
void PUBLIC_API cblas_domatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, double alpha,
                                const double* a, int a_ld, double* b, int b_ld);
void PUBLIC_API cblas_comatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, const void* alpha,
                                const void* a, int a_ld, void* b, int b_ld);
void PUBLIC_API cblas_zomatcopy(CLBlastLayout layout, CLBlastTranspose a_transpose, int m, int n, const void* alpha,
                                const void* a, int a_ld, void* b, int b_ld);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
void PUBLIC_API cblas_sim2col(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const float* im, float* col);
void PUBLIC_API cblas_dim2col(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const double* im, double* col);
void PUBLIC_API cblas_cim2col(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const void* im, void* col);
void PUBLIC_API cblas_zim2col(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const void* im, void* col);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
void PUBLIC_API cblas_scol2im(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const float* col, float* im);
void PUBLIC_API cblas_dcol2im(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const double* col, double* im);
void PUBLIC_API cblas_ccol2im(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const void* col, void* im);
void PUBLIC_API cblas_zcol2im(CLBlastKernelMode kernel_mode, int channels, int height, int width, int kernel_h,
                              int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                              int dilation_w, const void* col, void* im);

// =================================================================================================

#ifdef __cplusplus
}  // extern "C"
#endif

// CLBLAST_CLBLAST_NETLIB_C_H_
#endif
