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


#ifndef COMMON_H_
#define COMMON_H_

#if defined (_MSC_VER)
#define __template_static static
#else   /* _MSC_VER */
#define __template_static
#endif  /* !_MSC_VER */

#define MAX(a, b)  ((a>b)? a: b)

#include <clBLAS.h>
#include <cmdline.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum BlasRoutineID {
		CLBLAS_GEMV,
		CLBLAS_SYMV,
		CLBLAS_GEMM,
		CLBLAS_GEMM2,
		CLBLAS_GEMM_TAIL,
		CLBLAS_TRMM,
		CLBLAS_TRSM,
		CLBLAS_SYRK,
		CLBLAS_SYR2K,
		CLBLAS_TRMV,
        CLBLAS_TPMV,
		CLBLAS_TRSV,
		CLBLAS_TRSV_GEMV,	// Need a Kludge as current "gemv" don't support complex types
		CLBLAS_SYMM,
		CLBLAS_GER,
		CLBLAS_SYR,
		CLBLAS_HER,
		CLBLAS_HER2,
		CLBLAS_HEMM,
		CLBLAS_HERK,
		CLBLAS_SWAP,
		CLBLAS_COPY,
		CLBLAS_DOT,
		CLBLAS_SCAL,
        CLBLAS_AXPY,
		CLBLAS_ROTG,
		CLBLAS_ROTM,
		CLBLAS_ROT,
		CLBLAS_ROTMG,
		CLBLAS_NRM2,
        CLBLAS_ASUM,
        CLBLAS_iAMAX,

		/* ! Must be the last */
		BLAS_FUNCTIONS_NUMBER
	} BlasRoutineID;

	typedef enum BlasFunction {
    FN_SGEMV,
    FN_DGEMV,
    FN_CGEMV,
    FN_ZGEMV,

    FN_SSYMV,
    FN_DSYMV,

    FN_SSPMV,
    FN_DSPMV,

    FN_SGEMM,
    FN_DGEMM,
    FN_CGEMM,
    FN_ZGEMM,

    FN_SGEMM_2,
    FN_DGEMM_2,
    FN_CGEMM_2,
    FN_ZGEMM_2,

    FN_STRMM,
    FN_DTRMM,
    FN_CTRMM,
    FN_ZTRMM,

    FN_STRSM,
    FN_DTRSM,
    FN_CTRSM,
    FN_ZTRSM,

    FN_SSYR2K,
    FN_DSYR2K,
    FN_CSYR2K,
    FN_ZSYR2K,

    FN_SSYRK,
    FN_DSYRK,
    FN_CSYRK,
    FN_ZSYRK,

    FN_STRMV,
    FN_DTRMV,
    FN_CTRMV,
    FN_ZTRMV,

    FN_STPMV,
    FN_DTPMV,
    FN_CTPMV,
    FN_ZTPMV,

    FN_STRSV,
    FN_DTRSV,
    FN_CTRSV,
    FN_ZTRSV,

    FN_STPSV,
    FN_DTPSV,
    FN_CTPSV,
    FN_ZTPSV,

    FN_SSYMM,
    FN_DSYMM,
    FN_CSYMM,
    FN_ZSYMM,

	FN_SSYR,
	FN_DSYR,

    FN_SSPR,
	FN_DSPR,

    FN_SGER,
    FN_DGER,
    FN_CGERU,
    FN_ZGERU,
    FN_CGERC,
    FN_ZGERC,

    FN_CHER,
    FN_ZHER,
	FN_CHER2,
	FN_ZHER2,

    FN_CHPR,
    FN_ZHPR,
	FN_CHPR2,
	FN_ZHPR2,

	FN_SSYR2,
	FN_DSYR2,

    FN_SSPR2,
	FN_DSPR2,

	FN_CHEMV,
	FN_ZHEMV,

    FN_CHPMV,
	FN_ZHPMV,

	FN_CHEMM,
	FN_ZHEMM,

	FN_CHERK,
	FN_ZHERK,

	FN_SGBMV,
	FN_DGBMV,
	FN_CGBMV,
	FN_ZGBMV,

	FN_STBMV,
	FN_DTBMV,
	FN_CTBMV,
	FN_ZTBMV,

	FN_SSBMV,
	FN_DSBMV,

	FN_CHBMV,
	FN_ZHBMV,

	FN_STBSV,
	FN_DTBSV,
	FN_CTBSV,
	FN_ZTBSV,

	FN_CHER2K,
	FN_ZHER2K,

    FN_SCOPY,
    FN_DCOPY,
    FN_CCOPY,
    FN_ZCOPY,

    FN_SSWAP,
    FN_DSWAP,
    FN_CSWAP,
    FN_ZSWAP,

    FN_SDOT,
    FN_DDOT,
    FN_CDOTU,
    FN_ZDOTU,
    FN_CDOTC,
    FN_ZDOTC,

    FN_SSCAL,
    FN_DSCAL,
    FN_CSCAL,
    FN_ZSCAL,
    FN_CSSCAL,
    FN_ZDSCAL,

    FN_SAXPY,
    FN_DAXPY,
    FN_CAXPY,
    FN_ZAXPY,

    FN_SROTG,
    FN_DROTG,
    FN_CROTG,
    FN_ZROTG,

    FN_SROTM,
    FN_DROTM,

	FN_SROT,
    FN_DROT,
	FN_CSROT,
    FN_ZDROT,

    FN_SROTMG,
    FN_DROTMG,

    FN_SNRM2,
    FN_DNRM2,
    FN_SCNRM2,
    FN_DZNRM2,

    FN_SASUM,
    FN_DASUM,
    FN_SCASUM,
    FN_DZASUM,

    FN_iSAMAX,
    FN_iDAMAX,
    FN_iCAMAX,
    FN_iZAMAX,

    BLAS_FUNCTION_END
} BlasFunctionID;

cl_context
getQueueContext(cl_command_queue commandQueue, cl_int *error);

cl_int
waitForSuccessfulFinish(
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_event *events);

cl_int
flushAll(
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues);

const char* orderStr(clblasOrder order);
const char* sideStr(clblasSide side);
const char* uploStr(clblasUplo uplo);
const char* transStr(clblasTranspose trans);
const char* diagStr(clblasDiag diag);

char encodeTranspose(clblasTranspose value);
char encodeUplo(clblasUplo value);
char encodeDiag(clblasDiag value);
char encodeSide(clblasSide value);

int functionBlasLevel(BlasFunctionID funct);

size_t trsmBlockSize(void);

#ifdef __cplusplus
}       // extern "C"
#endif

#ifdef __cplusplus

template <typename T>
static T
convertMultiplier(ComplexLong arg)
{
    return static_cast<T>(arg.re);
}

template<>
__template_static FloatComplex
convertMultiplier(ComplexLong arg)
{
    return floatComplex(
        static_cast<float>(arg.re), static_cast<float>(arg.imag));
}

template<>
__template_static DoubleComplex
convertMultiplier(ComplexLong arg)
{
    return doubleComplex(arg.re, arg.imag);
}

template <typename T>
static cl_double returnMax(T arg)
{
    return static_cast<cl_double>(fabs(arg));
}

 template<>
__template_static cl_double returnMax<FloatComplex> (FloatComplex arg)
{
    return static_cast<cl_double>( MAX( fabs(CREAL(arg)), fabs(CIMAG(arg)) ) );
}

 template<>
__template_static cl_double returnMax<DoubleComplex> (DoubleComplex arg)
{
    return static_cast<cl_double>( MAX( fabs(CREAL(arg)), fabs(CIMAG(arg)) ) );
}

// xGEMM
void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc);

// xTRMM, xTRSM
void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb);

//xTRMV, xTRSV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    size_t lda,
    int incx,
    size_t offa,
    size_t offx);

//xTPMV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    int incx,
    size_t offa,
    size_t offx);

//xSYR xHER
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    size_t offx,
    int incx,
    size_t offa,
    size_t lda);


//xHER2
void
printTestParams(
        clblasOrder order,
        clblasUplo  uplo,
        size_t N,
        bool useAlpha,
        cl_float2 alpha,
        size_t offx,
        int incx,
        size_t offy,
        int incy,
        size_t offa,
        size_t lda);

//xCOPY , xSWAP
void
printTestParams(
        size_t N,
        size_t offx,
        int incx,
        size_t offy,
        int incy);

//xSyr2
void
printTestParams(
	clblasOrder order,
	clblasUplo  uplo,
	size_t N,
	double alpha,
	size_t offx,
	int incx,
	size_t offy,
	int incy,
	size_t offa,
	size_t lda);

//HEMV
void
printTestParams(
    clblasOrder order,
    clblasUplo  uplo,
    size_t N,
    ComplexLong alpha,
    size_t offa,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy);

//xSymm,
void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    bool useBeta,
    ComplexLong beta,
    size_t lda,
    size_t ldb,
    size_t ldc,
    size_t offa,
    size_t offb,
    size_t offc );

//xHEMM
void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    bool useAlpha,
    cl_float2 alpha,
    bool useBeta,
    cl_float2 beta,
    size_t lda,
    size_t ldb,
    size_t ldc,
    size_t offa,
    size_t offb,
    size_t offc );


//xGER , xGERC

void
printTestParams(
    clblasOrder order,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t lda,
    int incx,
    int incy,
    size_t offa,
    size_t offx,
    size_t offy );

// xGEMV
void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    int incx,
    bool useBeta,
    ComplexLong beta,
    int incy);

// xGBMV
void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy);

//xHBMV/xSBMV

void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy);


// xTBMV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    size_t KLU,
    size_t offA,
    size_t lda,
    size_t offx,
    int incx,
    size_t offy,
    int incy);

// xSYMV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    int incx,
    bool useBeta,
    ComplexLong beta,
    int incy);

// xSYR2K
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc);

// xSYRK
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc);

// xSCAL
void
printTestParams(
    size_t N,
    ComplexLong alpha,
    size_t offx,
    int incx);

// xAXPY
void
printTestParams(
    size_t N,
    ComplexLong alpha,
    size_t offx,
    int incx,
    size_t offy,
    int incy);

// For ROT
void
printTestParams(
    size_t N,
    size_t offx,
    int incx,
	size_t offy,
	int incy,
	ComplexLong alpha,
	ComplexLong beta);

// xROTG, check if other ROTs can use this too
void
printTestParams(size_t offSA, size_t offSB, size_t offC, size_t offS);

// xROTM
void
printTestParams(size_t N, size_t offx, int incx, size_t offy, int incy, size_t offParam, ComplexLong sflagParam);

//xROTMG
void
printTestParams(int offX, int offY, int offD1, int offD2, int offParam, ComplexLong sflagParam);

// xNRM2, AMAX and ASUM
void
printTestParams(
    size_t N,
    size_t offx,
    int incx);

#endif  // __cplusplus

#endif  /* COMMON_H_ */
