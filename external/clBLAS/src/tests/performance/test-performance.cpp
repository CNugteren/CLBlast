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


#define DO_SYR
#define DO_SPR
#define DO_SYMM
#define DO_TRMV
#define DO_TPMV
#define DO_TRSV
#define DO_GEMM
#define DO_TRMM
#define DO_TRSM
#define DO_GEMV
#define DO_SYR2K
#define DO_SYRK
#define DO_GER
#define DO_GERC
#define DO_HER
#define DO_HPR
#define DO_SYR2
#define DO_SPR2
#define DO_SPR2
#define DO_SBMV
#define DO_HER2
#define DO_HPR2
#define DO_HEMV
#define DO_HEMM
#define DO_HERK
#define DO_SYMV
#define DO_TPSV
#define DO_HPMV
#define DO_SPMV
#define DO_GBMV
#define DO_HBMV
#define DO_TBMV
#define DO_TBSV
#define DO_HER2K
#define DO_SWAP
#define DO_COPY
#define DO_SCAL
#define DO_AXPY
#define DO_DOT
#define DO_DOTC
#define DO_ROTG
#define DO_ROTM
#define DO_ROT
#define DO_ROTMG
#define DO_NRM2
#define DO_ASUM
#define DO_iAMAX

//#define DO_GEMM_2 - This needs to remain commented.

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <math.h>
#include <float.h>

#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <gemv.h>
#include <symv.h>
#include <gemm.h>
#include <gemm-2.h>
#include <trmm.h>
#include <trsm.h>
#include <syr2k.h>
#include <syrk.h>
#include <trmv.h>
#include <tpmv.h>
#include <trsv.h>
#include <symm.h>
#include <ger.h>
#include <gerc.h>
#include <syr.h>
#include <sbmv.h>
#include <spr.h>
#include <syr2.h>
#include <spr2.h>
#include <her.h>
#include <hpr.h>
#include <her2.h>
#include <hpr2.h>
#include <hemm.h>
#include <hemv.h>
#include <herk.h>
#include <tpsv.h>
#include <hpmv.h>
#include <spmv.h>
#include <gbmv.h>
#include <hbmv.h>
#include <tbmv.h>
#include <tbsv.h>
#include <her2k.h>
#include <swap.h>
#include <copy.h>
#include <scal.h>
#include <axpy.h>
#include <dot.h>
#include <dotc.h>
#include <rotg.h>
#include <rotm.h>
#include <rot.h>
#include <rotmg.h>
#include <nrm2.h>
#include <asum.h>
#include <iamax.h>

#include "PerformanceRecorder.h"

#define EXPECTED_SINGLE_FLOAT_PERF_RATIO 10.0
#define EXPECTED_DOUBLE_FLOAT_PERF_RATIO 4.0

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;
using namespace clMath;

PerformanceRecorder *perfRecorder;

static bool
isDoubleZero(double d)
{
    return (fabs(d) < 0.000001);
}

static const char
*functionToString(BlasFunction function)
{
    const char *s = NULL;

    switch (function) {
    case FN_SGEMV:
        s = "SGEMV";
        break;
    case FN_DGEMV:
        s = "DGEMV";
        break;
    case FN_CGEMV:
        s = "CGEMV";
        break;
    case FN_ZGEMV:
        s = "ZGEMV";
        break;
    case FN_SSYMV:
        s = "SSYMV";
        break;
    case FN_DSYMV:
        s = "DSYMV";
        break;
    case FN_SGEMM:
        s = "SGEMM";
        break;
    case FN_DGEMM:
        s = "DGEMM";
        break;
    case FN_CGEMM:
        s = "CGEMM";
        break;
    case FN_ZGEMM:
        s = "ZGEMM";
        break;
    case FN_SGEMM_2:
        s = "SGEMM_2";
        break;
    case FN_DGEMM_2:
        s = "DGEMM_2";
        break;
    case FN_CGEMM_2:
        s = "CGEMM_2";
        break;
    case FN_ZGEMM_2:
        s = "ZGEMM_2";
        break;
    case FN_STRMM:
        s = "STRMM";
        break;
    case FN_DTRMM:
        s = "DTRMM";
        break;
    case FN_CTRMM:
        s = "CTRMM";
        break;
    case FN_ZTRMM:
        s = "ZTRMM";
        break;
    case FN_STRSM:
        s = "STRSM";
        break;
    case FN_DTRSM:
        s = "DTRSM";
        break;
    case FN_CTRSM:
        s = "CTRSM";
        break;
    case FN_ZTRSM:
        s = "ZTRSM";
        break;
    case FN_SSYR2K:
        s = "SSYR2K";
        break;
    case FN_DSYR2K:
        s = "DSYR2K";
        break;
    case FN_CSYR2K:
        s = "CSYR2K";
        break;
    case FN_ZSYR2K:
        s = "ZSYR2K";
        break;
    case FN_SSYRK:
        s = "SSYRK";
        break;
    case FN_DSYRK:
        s = "DSYRK";
        break;
    case FN_CSYRK:
        s = "CSYRK";
        break;
    case FN_ZSYRK:
        s = "ZSYRK";
        break;
	case FN_STRMV:
		s = "STRMV";
		break;
	case FN_DTRMV:
        s = "DTRMV";
		break;
	case FN_CTRMV:
        s = "CTRMV";
		break;
	case FN_ZTRMV:
        s = "ZTRMV";
	    break;
    case FN_STPMV:
        s = "STPMV";
        break;
    case FN_DTPMV:
        s = "DTPMV";
        break;
    case FN_CTPMV:
        s = "CTPMV";
        break;
    case FN_ZTPMV:
        s = "ZTPMV";
        break;

	case FN_STRSV:
        s = "STRSV";
        break;
    case FN_DTRSV:
        s = "DTRSV";
        break;
    case FN_CTRSV:
        s = "CTRSV";
        break;
    case FN_ZTRSV:
        s = "ZTRSV";
		break;

    case FN_STBSV:
        s = "STBSV";
        break;
    case FN_DTBSV:
        s = "DTBSV";
        break;
    case FN_CTBSV:
        s = "CTBSV";
        break;
    case FN_ZTBSV:
        s = "ZTBSV";
        break;


    case FN_STPSV:
        s = "STPSV";
        break;
    case FN_DTPSV:
        s = "DTPSV";
        break;
    case FN_CTPSV:
        s = "CTPSV";
        break;
    case FN_ZTPSV:
        s = "ZTPSV";
        break;


	case FN_SSYMM:
	    s = "SSYMM";
		break;
	case FN_DSYMM:
        s = "DSYMM";
		break;
	case FN_CSYMM:
        s = "CSYMM";
		break;
	case FN_ZSYMM:
        s = "ZSYMM";
		break;

    case FN_SGER:
        s = "SGER";
        break;
    case FN_DGER:
        s = "DGER";
        break;
    case FN_CGERU:
        s = "CGERU";
        break;
    case FN_ZGERU:
        s = "ZGERU";
        break;
	case FN_CGERC:
        s = "CGERC";
        break;
    case FN_ZGERC:
        s = "ZGERC";
        break;
    case FN_CHER:
        s = "CHER";
        break;
    case FN_ZHER:
        s = "ZHER";
        break;
     case FN_CHPR:
        s = "CHPR";
        break;
    case FN_ZHPR:
        s = "ZHPR";
        break;

	case FN_CHER2:
        s = "CHER2";
        break;
    case FN_ZHER2:
        s = "ZHER2";
        break;
	case FN_SSYR:
		s = "SSYR";
		break;
	case FN_DSYR:
		s = "DSYR";
		break;
    case FN_SSPR2:
        s = "SSPR2";
        break;
    case FN_DSPR2:
        s = "DSPR2";
        break;
    case FN_SSPR:
        s = "SSPR";
        break;
    case FN_DSPR:
        s = "DSPR";
        break;
	case FN_SSYR2:
		s = "SSYR2";
		break;
	case FN_DSYR2:
		s = "DSYR2";
		break;
	case FN_CHEMM:
        s = "CHEMM";
        break;
    case FN_ZHEMM:
        s = "ZHEMM";
        break;
	case FN_CHEMV:
        s = "CHEMV";
        break;
    case FN_ZHEMV:
        s = "ZHEMV";
        break;
    case FN_CHERK:
        s = "CHERK";
        break;
    case FN_ZHERK:
        s = "ZHERK";
        break;
    case FN_SSBMV:
        s = "SSBMV";
        break;
    case FN_DSBMV:
        s = "DSBMV";
        break;
    case FN_CHBMV:
        s = "CHBMV";
        break;
    case FN_ZHBMV:
        s = "ZHBMV";
        break;
    case FN_CHER2K:
        s = "CHER2K";
        break;
    case FN_ZHER2K:
        s = "ZHER2K";
        break;

    case FN_SSWAP:
        s = "SSWAP";
        break;
    case FN_DSWAP:
        s = "DSWAP";
        break;
    case FN_CSWAP:
        s = "CSWAP";
        break;
    case FN_ZSWAP:
        s = "ZSWAP";
        break;

    case FN_SSCAL:
	    s = "SSCAL";
		break;
	case FN_DSCAL:
        s = "DSCAL";
		break;
	case FN_CSCAL:
        s = "CSCAL";
		break;
	case FN_ZSCAL:
        s = "ZSCAL";
		break;
	case FN_CSSCAL:
        s = "CSSCAL";
		break;
	case FN_ZDSCAL:
        s = "ZDSCAL";
		break;

	case FN_SCOPY:
        s = "SCOPY";
        break;
    case FN_DCOPY:
        s = "DCOPY";
        break;
    case FN_CCOPY:
        s = "CCOPY";
        break;
    case FN_ZCOPY:
        s = "ZCOPY";
        break;
	 case FN_SDOT:
        s = "SDOT";
        break;
    case FN_DDOT:
        s = "DDOT";
        break;

    case FN_CDOTU:
        s = "CDOTU";
        break;
    case FN_ZDOTU:
        s = "ZDOTU";
        break;

    case FN_CDOTC:
        s = "CDOTC";
        break;
    case FN_ZDOTC:
        s = "ZDOTC";
        break;

    case FN_SAXPY:
        s = "SAXPY";
        break;
    case FN_DAXPY:
        s = "DAXPY";
        break;
    case FN_CAXPY:
        s = "CAXPY";
        break;
    case FN_ZAXPY:
        s = "ZAXPY";
        break;


    case FN_SROTG:
        s = "SROTG";
        break;
    case FN_DROTG:
        s = "DROTG";
        break;
    case FN_CROTG:
        s = "CROTG";
        break;
    case FN_ZROTG:
        s = "ZROTG";
        break;

    case FN_SROTM:
        s = "SROTM";
        break;
    case FN_DROTM:
        s = "DROTM";
        break;

	case FN_SROT:
        s = "SROT";
        break;
    case FN_DROT:
        s = "DROT";
        break;
	case FN_CSROT:
        s = "CSROT";
        break;
    case FN_ZDROT:
        s = "ZDROT";
        break;

    case FN_SROTMG:
        s = "SROTMG";
        break;
    case FN_DROTMG:
        s = "DROTMG";
        break;

    case FN_SNRM2:
        s = "SNRM2";
        break;
    case FN_DNRM2:
        s = "DNRM2";
        break;
	case FN_SCNRM2:
        s = "SCNRM2";
        break;
    case FN_DZNRM2:
        s = "DZNRM2";
        break;

    case FN_SASUM:
        s = "SASUM";
        break;
    case FN_DASUM:
        s = "DASUM";
        break;
    case FN_SCASUM:
        s = "SCASUM";
        break;
    case FN_DZASUM:
        s = "DZASUM";
        break;

    case FN_iSAMAX:
        s = "iSAMAX";
        break;
    case FN_iDAMAX:
        s = "iDAMAX";
        break;
    case FN_iCAMAX:
        s = "iCAMAX";
        break;
    case FN_iZAMAX:
        s = "iZAMAX";
        break;

    default:
        break;
    }

    return s;
}

static const clblasOrder orderSet[] =
#ifdef PERF_TEST_WITH_ROW_MAJOR
    { clblasColumnMajor, clblasRowMajor };
#else
    { clblasColumnMajor };
#endif
static const clblasTranspose transSet[] =
    { clblasNoTrans, clblasTrans, clblasConjTrans };
static const clblasSide sideSet[] =
    { clblasLeft, clblasRight };
static const clblasUplo uploSet[] =
    { clblasUpper, clblasLower };
static const clblasDiag diagSet[] =
    { clblasUnit, clblasNonUnit };

const int sizeRange[] = { 2048, 2800, 4096, 5600 };
// Since blas-1 contains only vector arrays, huge vectors has to be provided to reach the peak of the card
const int blas1sizeRange[] = {4194304, 7840000, 16777216, 31360000 };
//const int sizeRange[] = { 2800, 4096, 5600};
const int KRange[] = { 2047, 2799, 4095, 5599 };
const int ldaRange[] = { 0, 5496, 5497 };
const int offsetRange[] = { 0, 100 };
const size_t offs[] =	{0, 63, 128, 258 };
const int incRange[] = { 1, 10 };


const double realAlphaRange[] = {(double)50.0, (double)100.0, (double)999999};
const cl_float2 complexAlphaRange[] = {floatComplex(1,2), floatComplex(4,5)};
const ComplexLong alphaBetaRange[] = {{50,50}, {20,20}};

#ifdef DO_GEMV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, GEMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange),
    Values(ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, GEMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
#endif

#ifdef DO_SYMV
// generic symv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, SYMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(sizeRange),
    Values(ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(32),
    Values(ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
#endif

#ifdef DO_GEMM_2
// generic gemm test looking over a set of sizes

INSTANTIATE_TEST_CASE_P(Generic, gemm2, Combine(
    Values(clblasColumnMajor), Values(clblasNoTrans), Values(clblasNoTrans),
    ValuesIn(sizeRange), ValuesIn(sizeRange), ValuesIn(sizeRange),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_GEMM
// generic gemm test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, GEMM, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange), ValuesIn(sizeRange),
    Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, GEMM, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), ValuesIn(transSet),
    Values(32), Values(32), Values(32),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_TRMM
// generic trmm test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, TRMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet), ValuesIn(sizeRange),
    ValuesIn(sizeRange), Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRMM,  Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(32), Values(32),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_TRSM
// generic trsm test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, TRSM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet), ValuesIn(sizeRange),
    ValuesIn(sizeRange), Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRSM,  Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(32), Values(32),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_SYR2K
// generic syr2k test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, SYR2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange),
    Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYR2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_SYRK
// generic syrk test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, SYRK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange),
    Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYRK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_HERK
// generic syrk test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(sizeRange), ValuesIn(sizeRange),ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(32), Values(32), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(ExtraTestSizes()), Values(1)));
#endif


#ifdef DO_TRMV
// generic trmv test looking over a set of sizes
// N, LDA, INCX, OFFA, OFFX, NUMQUEUES
INSTANTIATE_TEST_CASE_P(Generic, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
	ValuesIn(sizeRange), Values(0), Values(1), Values(0,10), Values(0,9), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRMV,  Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
   Values(5000), Values(0), Values(1), Values(0,10), Values(0,9),Values(1)));
#endif

#ifdef DO_TPMV
// generic trmv test looking over a set of sizes
// N, LDA, INCX, OFFA, OFFX, NUMQUEUES
INSTANTIATE_TEST_CASE_P(Generic, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(sizeRange), Values(0),Values(1), Values(0,10), Values(0,9), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TPMV,  Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
   Values(5000), Values(0),Values(1), Values(0,10), Values(0,9),Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(Generic, TRSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
        ValuesIn(sizeRange), Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRSV,  Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
   Values(1024), Values(0), Values(1), Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(Generic, TPSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
        ValuesIn(sizeRange), Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TPSV,  Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
   Values(1024), Values(0), Values(1), Values(0,10), Values(0,9), Values(1)));
#endif


#ifdef DO_SYMM
INSTANTIATE_TEST_CASE_P(Generic, SYMM, Combine(
	ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
	ValuesIn(sizeRange), ValuesIn(sizeRange),
	ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
	Values(1) ) );

INSTANTIATE_TEST_CASE_P(custom, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    Values(1024), Values(1024),
    ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    Values(1) ) );
#endif


#ifdef DO_HEMM
INSTANTIATE_TEST_CASE_P(Generic, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange), ValuesIn(complexAlphaRange),
	ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes()),
	//ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes((size_t)0, (size_t)0, (size_t)0, (size_t)12, (size_t)0, (size_t)1)),
	Values(1) ) );

INSTANTIATE_TEST_CASE_P(custom, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    Values(1024), Values(1024), Values(complexAlphaRange[0]), Values(complexAlphaRange[1]), Values(clMath::ExtraTestSizes((size_t)0, (size_t)0, (size_t)0, (size_t)8, (size_t)0, (size_t)1 )),
	Values(1) ) );
#endif

#ifdef DO_GER
INSTANTIATE_TEST_CASE_P(Generic, GER, Combine(
        ValuesIn(orderSet),ValuesIn(sizeRange), ValuesIn(sizeRange),
        Values(0), Values(1), Values(1), Values(0, 10),
        Values(0, 8),Values(0, 9),Values(1) ) );

INSTANTIATE_TEST_CASE_P(custom, GER, Combine(
         ValuesIn(orderSet),ValuesIn(sizeRange), ValuesIn(sizeRange),
        Values(0), Values(1), Values(1), Values(0, 10),
        Values(0, 8),Values(0, 9), Values(1) ) );
#endif

#ifdef DO_GERC
INSTANTIATE_TEST_CASE_P(Generic, GERC, Combine(
        ValuesIn(orderSet),ValuesIn(sizeRange), ValuesIn(sizeRange),
        Values(0), Values(1), Values(1), Values(0, 10),
        Values(0, 8),Values(0, 9),Values(1) ) );

INSTANTIATE_TEST_CASE_P(custom, GERC, Combine(
         ValuesIn(orderSet),ValuesIn(sizeRange), ValuesIn(sizeRange),
        Values(0), Values(1), Values(1), Values(0, 10),
        Values(0, 8),Values(0, 9), Values(1) ) );
#endif

#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(Generic, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, HER, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );
#endif

#ifdef DO_HPR
INSTANTIATE_TEST_CASE_P(Generic, HPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    Values(0), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, HPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    Values(0), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );
#endif


#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(Generic, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange), ValuesIn(ldaRange),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(ldaRange),
    Values(1) ) );
#endif

#ifdef DO_HPR2
INSTANTIATE_TEST_CASE_P(Generic, HPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange),ValuesIn(offsetRange), ValuesIn(ldaRange),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, HPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(ldaRange),
    Values(1) ) );
#endif

#ifdef DO_SYR
INSTANTIATE_TEST_CASE_P(Generic, SYR, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
	ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange),
	Values(0), Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, SYR, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange),
    ValuesIn(ldaRange), Values(1) ) );
#endif

#ifdef DO_SPR
INSTANTIATE_TEST_CASE_P(Generic, SPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange),
    Values(0), Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, SPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange),
    Values(0), Values(1) ) );
#endif

#ifdef DO_SYR2
INSTANTIATE_TEST_CASE_P(Generic, SYR2, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
	ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
	Values(0), Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, SYR2, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange), ValuesIn(offsetRange),
    ValuesIn(ldaRange), Values(1) ) );
#endif

#ifdef DO_SPR2
INSTANTIATE_TEST_CASE_P(Generic, SPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(0), Values(1) ) );

INSTANTIATE_TEST_CASE_P(Custom, SPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), Values(1), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(0), Values(1) ) );
#endif

#ifdef DO_HEMV
INSTANTIATE_TEST_CASE_P(Generic, HEMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Custom, HEMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
#endif


#ifdef DO_HPMV
INSTANTIATE_TEST_CASE_P(Generic, HPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Custom, HPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
#endif


#ifdef DO_SPMV
INSTANTIATE_TEST_CASE_P(Generic, SPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(sizeRange), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Custom, SPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(4099), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), Values((size_t)0), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
#endif

#ifdef DO_GBMV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    ValuesIn(sizeRange), ValuesIn(sizeRange), ValuesIn(KRange), ValuesIn(KRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    Values(32), Values(32), Values(30), Values(25),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
#endif

#ifdef DO_SBMV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, SBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(sizeRange), ValuesIn(KRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(32), Values(25),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
#endif

#ifdef DO_HBMV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, HBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(sizeRange), ValuesIn(KRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, HBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(32), Values(25),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
#endif


#ifdef DO_TBMV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(sizeRange),ValuesIn(KRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(32),Values(30),Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));
#endif

#ifdef DO_TBSV
// generic gemv test looking over a set of sizes
INSTANTIATE_TEST_CASE_P(Generic, TBSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(sizeRange),ValuesIn(KRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TBSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(32),Values(30),Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));
#endif

#ifdef DO_HER2K

INSTANTIATE_TEST_CASE_P(Generic, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(sizeRange), ValuesIn(sizeRange),ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(32), Values(32), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(ExtraTestSizes()), Values(1)));
#endif

#ifdef DO_SWAP

INSTANTIATE_TEST_CASE_P(Generic, SWAPXY, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SWAPXY, Combine(
    Values(819430), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), Values(1) ) );
#endif

#ifdef DO_DOT

INSTANTIATE_TEST_CASE_P(Generic, DOT, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, DOT, Combine(
    Values(819430), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif

#ifdef DO_DOTC

INSTANTIATE_TEST_CASE_P(Generic, DOTC, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, DOTC, Combine(
    Values(819430), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif


#ifdef DO_COPY

INSTANTIATE_TEST_CASE_P(Generic, COPY, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, COPY, Combine(
    Values(32), ValuesIn(incRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif


#ifdef DO_SCAL

INSTANTIATE_TEST_CASE_P(Generic, SCAL, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), ValuesIn(incRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SCAL, Combine(
    Values(819430), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), Values(1, 2), Values(1) ) );
#endif

#ifdef DO_AXPY

INSTANTIATE_TEST_CASE_P(Generic, AXPY, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(alphaBetaRange),  ValuesIn(offsetRange),
    ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, AXPY, Combine(
    Values(819430), ValuesIn(alphaBetaRange), ValuesIn(offsetRange),
    Values(1, 2), ValuesIn(offsetRange), Values(1, 2), Values(1) ) );
#endif

#ifdef DO_ROTG
INSTANTIATE_TEST_CASE_P(Generic, ROTG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, ROTG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1)));
#endif

#ifdef DO_ROTM
INSTANTIATE_TEST_CASE_P(Generic, ROTM, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(alphaBetaRange), Values(1)));
// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, ROTM, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(alphaBetaRange), Values(1)));
#endif

#ifdef DO_ROT
INSTANTIATE_TEST_CASE_P(Generic, ROT, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, ROT, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(incRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
#endif

#ifdef DO_ROTMG
INSTANTIATE_TEST_CASE_P(Generic, ROTMG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
    ValuesIn(alphaBetaRange), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, ROTMG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
    ValuesIn(alphaBetaRange), Values(1)));
#endif

#ifdef DO_NRM2

INSTANTIATE_TEST_CASE_P(Generic, NRM2, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, NRM2, Combine(
    Values(819430), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif

#ifdef DO_ASUM

INSTANTIATE_TEST_CASE_P(Generic, ASUM, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, ASUM, Combine(
    Values(819430), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif

#ifdef DO_iAMAX

INSTANTIATE_TEST_CASE_P(Generic, iAMAX, Combine(
    ValuesIn(blas1sizeRange), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, iAMAX, Combine(
    Values(819430), ValuesIn(incRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );
#endif

#if 0
// ensure that a TRMM function is faster then the respective GEMM one
static void
checkIsTrmmFaster(BlasFunction trmmFn, BlasFunction gemmFn)
{
    const char *s1, *s2;
    gflops_t gf1, gf2;

    gf1 = perfRecorder->clblasAvgPerf(trmmFn);
    gf2 = perfRecorder->clblasAvgPerf(gemmFn);

    if (isDoubleZero((double)gf1) || isDoubleZero((double)gf2)) {
        // skip, respective tests has not been run
        return;
    }

    s1 = functionToString(trmmFn);
    s2 = functionToString(gemmFn);
    cerr << "Check if the " << s1 << " function is faster than the " <<
            s2 << " one" << endl;

    if (gf1 * 2 > gf2) { // since TRMM has in twice as less operations as GEMM
        cerr << "PASS" << endl << endl;
    }
    else {
        cerr << "FAIL" << endl << endl;
    }
}
#endif

int
main(int argc, char *argv[])
{
    int ret;
    int fn;
    gflops_t gflops1, gflops2;
	gbps_t gbps1,gbps2;
    double ratio;
    const char *name;
    ::clMath::BlasBase *base;
    TestParams params;
#if 0
    BlasFunction estimFuncs[][2] = {
        {FN_SGEMM, FN_CGEMM }, // FN_STRMM, FN_CTRMM},
        {FN_DGEMM, FN_ZGEMM }  // FN_DTRMM, FN_ZTRMM}};
		};
    const char *message[2] = {
        "Check if the resulting average ratio for single float types "
        "(for GEMM and TRMM) matches the expected one ",
        "Check if the resulting average ratio for double float "
        "precision types (for GEMM and TRMM) matches the expected one "};
    double estimRatios[2] = {
        EXPECTED_SINGLE_FLOAT_PERF_RATIO,
        EXPECTED_DOUBLE_FLOAT_PERF_RATIO};
#endif

    if ((argc > 1) && !strcmp(argv[1], "--test-help")) {
        printUsage("test-performance");
        return 0;
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::std::cerr << "Initialize OpenCL and CLBLAS..." << ::std::endl;
    base = ::clMath::BlasBase::getInstance();
    if (base == NULL) {
        ::std::cerr << "Fatal error, OpenCL or clblas initialization failed! "
                       "Leaving the test." << ::std::endl;
        return -1;
    }

    base->setSeed(DEFAULT_SEED);

    if (argc > 1) {
        params.optFlags = NO_FLAGS;
        params.devType = CL_DEVICE_TYPE_GPU;
        params.devName = NULL;
        if (parseBlasCmdLineArgs(argc, argv, &params) != 0) {
            printUsage(argv[0]);
            return 1;
        }
        if (params.optFlags & SET_SEED) {
            base->setSeed(params.seed);
        }
        if (params.optFlags & SET_ALPHA) {
            base->setAlpha(params.alpha);
        }
        if (params.optFlags & SET_BETA) {
            base->setBeta(params.beta);
        }
        if (params.optFlags & SET_M) {
            base->setM(params.M);
        }
        if (params.optFlags & SET_N) {
            base->setN(params.N);
        }
        if (params.optFlags & SET_K) {
            base->setK(params.K);
        }
		if (params.optFlags & SET_INCX) {
            base->setIncX(params.incx);
        }

        if (params.optFlags & SET_DEVICE_TYPE) {
            if (!base->setDeviceType(&params.devType, params.devName)) {
                ::std::cerr << "Fatal error, OpenCL or clblas "
                        "initialization failed! Leaving the test." <<
                        ::std::endl;
                return -1;
            }
        }
        if (params.optFlags & SET_NUM_COMMAND_QUEUES) {
            base->setNumCommandQueues(params.numCommandQueues);
        }
    }

    parseEnv(&params);
    if (params.optFlags & SET_USE_IMAGES) {
        base->setUseImages(params.useImages);
    }

    perfRecorder = new PerformanceRecorder;

	/* Use of image based buffers is deprecated
    if (base->useImages()) {
        if (base->addScratchImages()) {
            std::cerr << "FATAL ERROR, CANNOT CREATE SCRATCH IMAGES!" << std::endl;
        }
    }
	*/

    ret = RUN_ALL_TESTS();

    if (base->useImages()) {
        base->removeScratchImages();
    }

    cerr << endl << endl;
    cerr << "----------------------------------------------" << endl <<
            "Overall performance information:" << endl <<
            "----------------------------------------------" << endl;

    // now, check average speed ratio
    for (fn = 0; fn < BLAS_FUNCTION_END; fn++) {
        name = functionToString(static_cast<BlasFunction>(fn));

        /*
         * For global memory based solutions print only average performance,
         * and for those of image based perform just comparison
         */

        ratio = perfRecorder->avgTimeRatio(static_cast<BlasFunction>(fn));
        if (isDoubleZero(ratio)) {
            // skip, this group of tests has not been run
            continue;
        }
		if (functionBlasLevel(static_cast<BlasFunction>(fn)) != 3) 	//display metrics in GBps if it is a BLAS-1 or BLAS-2 function
		{
	        gbps1 = perfRecorder->etalonAvgGbpsPerf(
	            static_cast<BlasFunction>(fn));
	        gbps2 = perfRecorder->clblasAvgGbpsPerf(
	            static_cast<BlasFunction>(fn));

	        cout << "Average reference " << name << endl <<
	                " performance is " << gbps1 <<
	                " GBps; for CLBLAS implementation: " << endl <<
	                "average performance = " << gbps2 << " GBps, "
	                "average time ratio = " << ratio << endl << endl;
		}
		else						//display metrics in GFlops if its a BLAS-3 function
		{
			gflops1 = perfRecorder->etalonAvgPerf(
                static_cast<BlasFunction>(fn));
            gflops2 = perfRecorder->clblasAvgPerf(
                static_cast<BlasFunction>(fn));

            cout << "Average reference " << name << endl <<
                    " performance is " << gflops1 <<
                    " giga-flops; for CLBLAS implementation: " << endl <<
                    "average performance = " << gflops2 << " giga-flops, "
                    "average time ratio = " << ratio << endl << endl;
		}
    }

    // check if TRMM is faster than GEMM
#if 0
    checkIsTrmmFaster(FN_STRMM, FN_SGEMM);
    checkIsTrmmFaster(FN_DTRMM, FN_DGEMM);
    checkIsTrmmFaster(FN_CTRMM, FN_CGEMM);
    checkIsTrmmFaster(FN_ZTRMM, FN_ZGEMM);

    /*
     * Now, do the final average ratio comparison if there is
     * the image based version. Involve only GEMM and TRMM as using
     * 2 images
     */
    if (base->useImages()) {
        int j;

        for (i = 0; i < 2; i++) {
            ratio = 0;
            nruns = 0;
            for (j = 0; j < 2; j++) {
                r = perfRecorder->avgTimeRatio(estimFuncs[i][j]);
                if (!isDoubleZero(r)) {
                    ratio += r;
                    nruns++;
                }
            }
            if (nruns) {
                ratio /= nruns;
                cerr << message[i] << endl;
                if (ratio >= estimRatios[i]) {
                    cerr << "PASS (" << ratio << ")" << endl << endl;
                }
                else {
                    cerr << "FAIL (" << ratio << ")" << endl << endl;
                }
            }
        }
    }
#endif
    /*
     * Explicitely tell the singleton to release all resources,
     * before we return from main.
     */
    base->release( );

    return ret;
}
