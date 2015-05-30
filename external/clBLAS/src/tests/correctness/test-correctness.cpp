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


#define DO_GEMM
#define DO_TRMM
#define DO_TRSM
#define DO_SYR2K
#define DO_SYRK
#define DO_GEMV
#define DO_SYMV
#define DO_SYMM
#define DO_TRMV
#define DO_TPMV
#define DO_TRSV
#define DO_SYR
#define DO_SPR
#define DO_GER
#define DO_GERC
#define DO_SYR2
#define DO_HER
#define DO_HER2
#define DO_HEMM
#define DO_HEMV
#define DO_HPMV
#define DO_SPMV
#define DO_SBMV
#define DO_HERK
#define DO_TPSV
#define DO_HPR
#define DO_SPR2
#define DO_HPR2
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

//#define DO_SPL - Only used for special case testing (for devel purposes)
//#define DO_GEMM_2 - This needs to remain commented.

#include <gtest/gtest.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <gemm.h>
#include <gemm-2.h>
#include <trmm.h>
#include <trsm.h>
#include <gemv.h>
#include <symv.h>
#include <syr2k.h>
#include <syrk.h>
#include <trsv.h>
#include <trmv.h>
#include <tpmv.h>
#include <symm.h>
#include <syr.h>
#include <sbmv.h>
#include <spr.h>
#include <ger.h>
#include <gerc.h>
#include <syr2.h>
#include <her.h>
#include <her2.h>
#include <hemv.h>
#include <hpmv.h>
#include <spmv.h>
#include <hemm.h>
#include <herk.h>
#include <tpsv.h>
#include <hpr.h>
#include <spr2.h>
#include <hpr2.h>
#include <gbmv.h>
#include <hbmv.h>
#include <tbmv.h>
#include <tbsv.h>
#include <her2k.h>
#include <swap.h>
#include <scal.h>
#include <copy.h>
#include <axpy.h>
#include <dot.h>
#include <asum.h>
#include <dotc.h>
#include <rotg.h>
#include <rotm.h>
#include <rot.h>
#include <rotmg.h>
#include <nrm2.h>
#include <iamax.h>

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;

TestParams globalTestParams;

// Different ranges of test parameters

static const clblasOrder orderSet[] =
    { clblasColumnMajor, clblasRowMajor };
static const clblasTranspose transSet[] =
    { clblasNoTrans, clblasTrans, clblasConjTrans };
static const clblasSide sideSet[] =
    { clblasLeft, clblasRight };
static const clblasUplo uploSet[] =
    { clblasUpper, clblasLower };
static const clblasDiag diagSet[] =
    { clblasUnit, clblasNonUnit };

const size_t ZERO_VAL[1] = { 0 };
const int ONE_VAL[1] = { 1 };
const int verySmallRange[] =
{1, 3, 5, 10, 11, 15, 16, 23, 21, 32, 33, 45, 40, 63, 333, 1024, 1025, 4096, 4223};
const int completeRange[] =
{1, 3, 5, 10, 11, 15, 16, 23, 21, 32, 33, 45, 40, 63, 333, 1024, 1025, 4096, 4223};
#if defined SHORT_TESTS
const int smallRange[] =
    { 63, 128 };

const int numQueues[] =
    { 2 };
#elif defined MEDIUM_TESTS  /* SHORT_TESTS */
const int smallRange[] =
    { 15, 64, 133 };
const int numQueues[] =
    { 3, 4 };
#else                       /* MEDIUM_TESTS */
const int smallRange[] =
    { 15, 16, 33, 40, 62, 64, 128, 129, 256, 258 };
    //{ 15, 16, 32, 33, 63, 64, 128, 129, 256, 257 };
	//{ 3, 4, 15, 16, 32, 33, 63, 64, 128, 129, 256, 257, 333, 566, 787, 1024, 1025, 1113, 1111, 999, 883, 633, 17 };

const int numQueues[] =
    { 2, 3, 4, 5, 6, 7 };
#endif                      /* !SHORT_TESTS && !MEDIUM_TESTS */

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

enum {
    BIG_LDA = 500,
    BIG_LDB = 600,
    BIG_LDC = 700
};

const int incs[] =
    { 33, -33 };

#else                       /* SHORT_TESTS || MEDIUM_TESTS */

enum {
    BIG_LDA = 501,
    BIG_LDB = 602,
    BIG_LDC = 703
};

const int incs[] =
    { 1, -1, 33, -33 };

#endif                      /* !SHORT_TESTS && !MEDIUM_TESTS */

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)
const size_t offs[] =
    { 63, 258 };
#else                       /* !SHORT_TESTS && !MEDIUM_TESTS */
const size_t offs[] =
    {0, 63, 128, 258 };
#endif

const int ldaRange[] = {0, 3192, 4097 };
const int offsetRange[] = { 0, 100 };
const double realAlphaRange[] = {(double)50, (double)100, (double)999999};
const cl_float2 complexAlphaRange[] = {floatComplex(0,1), floatComplex(3,4)};
const cl_float2 complexAlpha = floatComplex(2,3);

const ComplexLong alphaBetaRange[] = {{50,50}, {20,20}};
const ComplexLong alphaBeta = {10,10};
const ComplexLong sflagRange[] = {{-1,0}, {0,0}, {1,0}, {-2,0}};

const ComplexLong rotCosMedium = {0, 3};
const ComplexLong rotSinMedium = {0, 4};

const ComplexLong rotCosShort = {1, 6};
const ComplexLong rotSinShort = {1, 2};

#ifdef DO_SPL

INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeHER2_SPL, HER2, Combine(
        Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(ldaRange),
    Values(1) ) );
#endif



#ifdef DO_HEMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HEMV, HEMV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_0HEMV, HEMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(order_HEMV, HEMV, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), ValuesIn(offs), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(uplo_HEMV, HEMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), ValuesIn(offs), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(alpha_beta_HEMV, HEMV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_0HEMV, HEMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(1500, 5101), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_HEMV, HEMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(alphaBetaRange),
    ValuesIn(alphaBetaRange), ValuesIn(offs), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)),
    Values(1)));

#endif      // Correctness

#endif

#ifdef DO_SWAP
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(SmallRange, SWAPXY, Combine(
        Values(100,50), Values(0), Values(1), Values(0), Values(1), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_SWAP, SWAPXY, Combine(
        Values(64,128,256,512), Values(0,3), Values(1,-1), Values(0,3), Values(1,-1), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_SWAP, SWAPXY, Combine(
        ValuesIn(completeRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs), Values(1)));

#endif
#endif

#ifdef DO_AXPY
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Small_AXPY, AXPY, Combine(
        Values(100,50), ValuesIn(alphaBetaRange), Values(0), Values(1), Values(0), Values(1), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_AXPY, AXPY, Combine(
        Values(64,128,256,512), ValuesIn(alphaBetaRange), Values(0,3), Values(1,-1), Values(0,3), Values(1,-1), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_AXPY, AXPY, Combine(
        ValuesIn(completeRange), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs), Values(1)));

#endif
#endif

#ifdef DO_ROTG
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Small_ROTG, ROTG, Combine(
        Values(1, 5), Values(1, 6), Values(2, 8), Values(3, 7), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_ROTG, ROTG, Combine(
        Values(64,128,256,512), Values(64, 128, 256, 512), Values(0,3), Values(0,3), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_ROTG, ROTG, Combine(
        ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1)));

#endif
#endif

#ifdef DO_ROTM
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Small_ROTM, ROTM, Combine(
        Values(1, 5, 10, 20), Values(1, 6), Values(1, -1), Values(1, 6), Values(1, -1), Values(1, 6), ValuesIn(sflagRange), Values(1)));
#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_ROTM, ROTM, Combine(
        Values(64,128,256,512), Values(0,3), Values(1, -3, 3, 1), Values(0,3), Values(1, -3, 3, 1), Values(0, 3), ValuesIn(sflagRange), Values(1)));
#else
INSTANTIATE_TEST_CASE_P(ALL_ROTM, ROTM, Combine(
        ValuesIn(completeRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs),
        ValuesIn(offsetRange), ValuesIn(sflagRange), Values(1)));
#endif
#endif

#ifdef DO_ROT
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Small_ROT, ROT, Combine(
        Values(1, 5, 10, 20), Values(1, 6), Values(1, -1), Values(1, 6), Values(1, -1), Values(rotCosShort), Values(rotSinShort), Values(1)));
#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_ROT, ROT, Combine(
        Values(64,128,256,512), Values(0,3), Values(1, -3, 3, 1), Values(0,3), Values(1, -3, 3, 1), Values(rotCosMedium), Values(rotSinMedium), Values(1)));
#else
INSTANTIATE_TEST_CASE_P(ALL_ROT, ROT, Combine(
        ValuesIn(completeRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs),
        ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));
#endif
#endif

#ifdef DO_ROTMG
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Small_ROTMG, ROTMG, Combine(
        Values(1, 6), Values(1, 6), Values(1, 6), Values(1, 6), Values(1, 6), ValuesIn(sflagRange), Values(1)));
#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_ROTMG, ROTMG, Combine(
        Values(1, 3, 15), Values(0, 3, 15), Values(0, 3, 15), Values(0, 3, 15), Values(0, 3, 15), ValuesIn(sflagRange), Values(1)));
#else
INSTANTIATE_TEST_CASE_P(ALL_ROTMG, ROTMG, Combine(
        ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
        ValuesIn(offsetRange), ValuesIn(sflagRange), Values(1)));
#endif
#endif

//NRM2

#ifdef DO_NRM2

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_NRM2, NRM2, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1)) );

INSTANTIATE_TEST_CASE_P(SelectedSmall0_NRM2, NRM2, Combine(
    Values(61), Values(4, -11), Values(0), Values(1), Values(1)) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_NRM2, NRM2, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_NRM2, NRM2, Combine(
    Values(4900), Values(1), Values(4), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_NRM2, NRM2, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness
#endif

#ifdef DO_ASUM

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_ASUM, ASUM, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1)) );

INSTANTIATE_TEST_CASE_P(SelectedSmall0_ASUM, ASUM, Combine(
    Values(61), Values(4, -11), Values(0), Values(1), Values(1)) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_ASUM, ASUM, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_ASUM, ASUM, Combine(
    Values(4900), Values(1), Values(4), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_ASUM, ASUM, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness
#endif

#ifdef DO_iAMAX

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_iAMAX, iAMAX, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1)) );

INSTANTIATE_TEST_CASE_P(SelectedSmall0_iAMAX, iAMAX, Combine(
    Values(61), Values(4, -1), Values(0), Values(1), Values(1)) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_iAMAX, iAMAX, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_iAMAX, iAMAX, Combine(
    Values(4900), Values(1), Values(4), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_iAMAX, iAMAX, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness
#endif

#ifdef DO_HPMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HPMV, HPMV, Combine(
	Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_0HPMV, HPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(order_HPMV, HPMV, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), ValuesIn(offs), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(uplo_HPMV, HPMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), ValuesIn(offs), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(alpha_beta_HPMV, HPMV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_0HPMV, HPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(1500, 5101), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_HPMV, HPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), ValuesIn(offs), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)),
    Values(1)));

#endif      // Correctness

#endif

#ifdef DO_SYMM

#if defined(SHORT_TESTS)
/*INSTANTIATE_TEST_CASE_P(Short_SYMM, SYMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),Values(clblasLower), ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)), Values(1)));*/
INSTANTIATE_TEST_CASE_P(SelectedSmall_0SYMM, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),ValuesIn(uploSet), Values(15),Values(15), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
/*INSTANTIATE_TEST_CASE_P(order_SYMM, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),Values(clblasLower), ValuesIn(smallRange),ValuesIn(smallRange) ,ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 9, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(uplo_SYMM, SYMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),ValuesIn(uploSet), ValuesIn(smallRange),ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 9, 0)), Values(1)));*/
INSTANTIATE_TEST_CASE_P(alpha_beta_SYMM, SYMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),Values(clblasLower), Values(64),Values(133), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 3, 7, 11)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_0SYMM, SYMM, Combine(
    ValuesIn(orderSet), Values(clblasLeft),Values(clblasLower), Values(1100),Values(4000), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 0, 0, 0)), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_SYMM_FriendlyOffsets, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),ValuesIn(uploSet), ValuesIn(smallRange),ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 64, 32, 128)),
    Values(1)));
INSTANTIATE_TEST_CASE_P(ALL_SYMM_UnfriendlyOffsets, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),ValuesIn(uploSet), ValuesIn(smallRange),ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 6, 3, 12)),
    Values(1)));

#endif      // Correctness
#endif


#ifdef DO_HEMM

#if defined(SHORT_TESTS)
/*INSTANTIATE_TEST_CASE_P(Short_HEMM, HEMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),Values(clblasLower), ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)), Values(1)));*/
INSTANTIATE_TEST_CASE_P(SelectedSmall_0HEMM, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),ValuesIn(uploSet), Values(15),Values(15), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
/*INSTANTIATE_TEST_CASE_P(order_HEMM, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),Values(clblasLower), ValuesIn(smallRange),ValuesIn(smallRange) ,ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 9, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(uplo_HEMM, HEMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),ValuesIn(uploSet), ValuesIn(smallRange),ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, 0, 0, 0, 9, 0)), Values(1)));*/
INSTANTIATE_TEST_CASE_P(alpha_beta_HEMM, HEMM, Combine(
    Values(clblasRowMajor), Values(clblasLeft),Values(clblasLower), Values(64),Values(133), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 0, 0, 9)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_0HEMM, HEMM, Combine(
    ValuesIn(orderSet), Values(clblasLeft),Values(clblasLower), Values(1010),Values( 4000), Values(complexAlpha),
    Values(complexAlpha), Values(clMath::ExtraTestSizes(0, (size_t)0, (size_t)0, 0, 1, 0)), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_HEMM, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet),ValuesIn(uploSet), ValuesIn(smallRange),ValuesIn(smallRange), ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange), Values(clMath::ExtraTestSizes(0, (size_t)512, (size_t)511, 9, 0, 0)), Values(1)));

#endif      // Correctness
#endif


#ifdef DO_SPMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SPMV, SPMV, Combine(
	Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_0SPMV, SPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(order_SPMV, SPMV, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), ValuesIn(offs), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(uplo_SPMV, SPMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), ValuesIn(offs), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(alpha_beta_SPMV, SPMV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), Values(alphaBeta),
    Values(alphaBeta), Values((size_t)0), Values((size_t)0), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_0SPMV, SPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(1500, 5101), Values(alphaBeta),
	Values(alphaBeta), Values((size_t)0), Values((size_t)0), Values((size_t)0), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#else
INSTANTIATE_TEST_CASE_P(ALL_SPMV, SPMV, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(alphaBetaRange),
	ValuesIn(alphaBetaRange), ValuesIn(offs), ValuesIn(offs), ValuesIn(offs), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)),
    Values(1)));

#endif  // Correctness

#endif


#ifdef DO_GEMM_2
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA_OFF_NX, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasNoTrans), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 1, 3, 10)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA_OFF_TN, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasTrans), Values(clblasNoTrans ),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, (size_t)501, (size_t)502, 3, 2, 1)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA_OFF_HN, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasConjTrans), Values(clblasNoTrans ),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, (size_t)501, (size_t)502, 3, 2, 1)), Values(1)));

#if !defined(SHORT_TESTS) && !defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_NX, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasNoTrans), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_TN, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasTrans), Values(clblasNoTrans ),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_HN, GEMM2, Combine(
    Values(clblasColumnMajor), Values(clblasConjTrans), Values(clblasNoTrans ),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
#endif

#endif //DO_GEMM_2

#ifdef DO_GEMM
// xGEMM tests
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA_OffSet, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 1, 0, 0)), Values(1)));

// Cases for extended versions with offsets

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(67), Values(138), Values(220),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 600, 700)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(67), Values(138), Values(220),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 600, 700)), Values(1)));

#else                               /* SHORT_TESTS || MEDIUM_TESTS */

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(67), Values(135), Values(228),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_1, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(64), Values(64), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_2, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(128), Values(64), Values(77),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_3, GEMM, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(112), Values(86), Values(68),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(67), Values(135), Values(228),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_1, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(64), Values(64), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_2, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(128), Values(64), Values(77),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_3, GEMM, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), ValuesIn(transSet),
    Values(112), Values(86), Values(68),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 502)), Values(1)));

#endif                              /* !SHORT_TESTS || !MEDIUM_TESTS */

// Big matrices
#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasNoTrans), Values(clblasNoTrans),
        Values(2801), Values(2903), Values(3005),
        Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasNoTrans), Values(clblasNoTrans),
        Values(4777), Values(4333), Values(5000),
        Values(clMath::ExtraTestSizes()), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasTrans), Values(clblasNoTrans),
        Values(5777), Values(5333), Values(3000),
        Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasTrans), Values(clblasConjTrans),
        Values(6777), Values(3333), Values(3000),
        Values(clMath::ExtraTestSizes()), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom cases

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasNoTrans), Values(clblasNoTrans),
        Values(1), Values(1), Values(1),
        Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasNoTrans), Values(clblasNoTrans),
        Values(2), Values(1), Values(3),
        Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasTrans), Values(clblasNoTrans),
        Values(3), Values(2), Values(1),
        Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasTrans), Values(clblasConjTrans),
        Values(4), Values(3), Values(2),
        Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, GEMM, Combine(
        ValuesIn(orderSet),
        Values(clblasConjTrans), Values(clblasNoTrans),
        Values(17), Values(13), Values(1),
        Values(clMath::ExtraTestSizes()), Values(1)));

        // Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, GEMM, Combine(
        ValuesIn(orderSet), ValuesIn(transSet), ValuesIn(transSet),
        Values(32), Values(32), Values(32),
        Values(clMath::ExtraTestSizes()), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */

#endif // DO_GEMM


#ifdef DO_TRMM
// xTRMM tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, TRMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, TRMM,  Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 0, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 0, 0, 0, 0)), Values(1)));

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, TRMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(158),
    Values(clMath::ExtraTestSizes(0, 0, 0, 502, 606, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(158),
    Values(clMath::ExtraTestSizes(0, 0, 0, 502, 606, 0)), Values(1)));

#else                        /* SHORT_TESTS || MEDIUM_TESTS */

// Cases for extended versions with offsets
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, TRMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(113),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_1, TRMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(128), Values(66),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_2, TRMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(53), Values(67),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(113),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_1, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(128), Values(66),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_2, TRMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(53), Values(67),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 0)), Values(1)));

#endif                        /* !SHORT_TESTS && !MEDIUM_TESTS */

// Big matrices

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(2801), Values(2903),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(4567), Values(4321),
    Values(clMath::ExtraTestSizes()), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasNonUnit),
    Values(5567), Values(5321),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasLower), Values(clblasTrans),
    Values(clblasUnit),
    Values(6567), Values(3321),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(1), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(2), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasNonUnit),
    Values(3), Values(2),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasLower), Values(clblasTrans),
    Values(clblasUnit),
    Values(4), Values(3),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, TRMM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasUnit),
    Values(17), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRMM,  Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(32), Values(32),
    Values(clMath::ExtraTestSizes()), Values(1)));


#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */
#endif // DO_TRMM

#ifdef DO_TRSM
// xTRSM tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, TRSM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, TRSM,  Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 0, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 0, 0, 0, 0)), Values(1)));

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, TRSM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(158),
    Values(clMath::ExtraTestSizes(0, 0, 0, 502, 606, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(158),
    Values(clMath::ExtraTestSizes(0, 0, 0, 502, 606, 0)), Values(1)));

#else                               /* SHORT_TESTS || MEDIUM_TESTS */

// Cases for extended versions with offsets
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, TRSM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(113),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_1, TRSM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(128), Values(66),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_2, TRSM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(53), Values(67),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(115), Values(113),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_1, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(128), Values(66),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_2, TRSM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(53), Values(67),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 0)), Values(1)));

#endif                              /* !SHORT_TESTS && !MEDIUM_TESTS */

// Big matrices

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(2801), Values(2903),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(4567), Values(4321),
    Values(clMath::ExtraTestSizes()), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasNonUnit),
    Values(5567), Values(5321),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasLower), Values(clblasTrans),
    Values(clblasUnit),
    Values(6567), Values(3321),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(1), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasRight), Values(clblasUpper), Values(clblasTrans),
    Values(clblasNonUnit),
    Values(2), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasNonUnit),
    Values(3), Values(2),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasLower), Values(clblasTrans),
    Values(clblasUnit),
    Values(4), Values(3),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, TRSM, Combine(
    ValuesIn(orderSet),
    Values(clblasLeft), Values(clblasUpper), Values(clblasNoTrans),
    Values(clblasUnit),
    Values(17), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, TRSM,  Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(32), Values(32),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */
#endif // DO_TRSM

#ifdef DO_GEMV
// xGEMV tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, GEMV, Combine(
    Values(clblasColumnMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, GEMV, Combine(
    Values(clblasRowMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, GEMV,  Combine(
    Values(clblasColumnMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, GEMV, Combine(
    Values(clblasRowMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SmallRange_VariousInc, GEMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, incs, incs,
                                   ZERO_VAL, ZERO_VAL, ZERO_VAL)),
    Values(1)));

// Cases for the extended version with offsets
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx, GEMV,  Combine(
    Values(clblasColumnMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, ONE_VAL, ONE_VAL, offs,
                                   ZERO_VAL, ZERO_VAL)),
           Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx, GEMV,  Combine(
    Values(clblasRowMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, ONE_VAL, ONE_VAL, offs,
                                   ZERO_VAL, ZERO_VAL)),
           Values(1)));

// Big matrices
#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(2800), Values(2800),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(4567), Values(4321),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));


INSTANTIATE_TEST_CASE_P(SelectedBig_2, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasNoTrans),
    Values(5567), Values(5321),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(6567), Values(3321),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(1), Values(1),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(2), Values(1),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasNoTrans),
    Values(3), Values(2),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasTrans),
    Values(4), Values(3),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, GEMV, Combine(
    ValuesIn(orderSet), Values(clblasNoTrans),
    Values(17), Values(1),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, GEMV,  Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */
#endif // DO_GEMV

#ifdef DO_SYMV
// xSYMV tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, SYMV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, SYMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, SYMV,  Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, SYMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SmallRange_VariousInc, SYMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, incs, incs,
                                   ZERO_VAL, ZERO_VAL, ZERO_VAL)),
    Values(1)));

// cases for the extended versions with offsets
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx, SYMV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, ONE_VAL, ONE_VAL, offs,
                                   ZERO_VAL, ZERO_VAL)),
             Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx, SYMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(smallRange),
    ValuesIn(clMath::makeContainerETS(ZERO_VAL, ONE_VAL, ONE_VAL, offs,
                                   ZERO_VAL, ZERO_VAL)),
             Values(1)));

// Big matrices
#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(2801),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined MEDIUM_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_1, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(4567),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(5567),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(6567),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(1),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(2),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(3),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(4),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, SYMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(5),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYMV,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(32),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */
#endif

#ifdef DO_SYR2K
// xSYR2K tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, SYR2K,  Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 501, 502, 0, 0, 0)), Values(1)));

// cases for the extended versions with the offsets
#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(254), Values(353),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 602, 704)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(254), Values(353),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 602, 704)), Values(1)));

#else                               /* SHORT_TESTS || MEDIUM_TESTS */

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(255), Values(253),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_1, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(128), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_2, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(75), Values(200),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_3, SYR2K, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(111), Values(256),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(255), Values(253),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_1, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(128), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 501, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_2, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(75), Values(200),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 502)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_3, SYR2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(111), Values(256),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 501, 502)), Values(1)));

#endif                              /* !SHORT_TESTS && !MEDIUM_TESTS */

// Big matrices
#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedBig_0, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(2801), Values(2903),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(4567), Values(4321),
    Values(clMath::ExtraTestSizes()), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(5567), Values(5321),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans),
    Values(6567), Values(3321),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(1), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(2), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(3), Values(2),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans),
    Values(4), Values(3),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, SYR2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(17), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYR2K,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */
#endif // DO_SYR2K

#ifdef DO_HERK
/*
 ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo,      // uplo
        clblasTranspose, // transA
        int,                // N
        int,                // K
        ComplexLong,        // alpha
        ComplexLong,        // beta
        ExtraTestSizes,     // offa, offc, lda, ldc.
        int                 // numCommandQueues
*/
#if !defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(SPL_HERK, HERK, Combine(
    Values(clblasColumnMajor, clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(513), Values(513), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
#endif

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HERK, HERK, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall0_HERK, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasConjTrans),
    Values(14), Values(15), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HERK, HERK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,0,10,0)), Values(1)));
INSTANTIATE_TEST_CASE_P(Uplo_HERK, HERK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));
INSTANTIATE_TEST_CASE_P(Trans_HERK, HERK, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,0,10,0)), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_HERK, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,10,0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig0_HERK, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(2510, 4300), Values(1500,4600), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));

#endif      // Correctness

#endif // DO_HERK


#ifdef DO_HER2K

#if !defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(SPL_HER2K, HER2K, Combine(
    Values(clblasColumnMajor, clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(513), Values(513), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
#endif

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HER2K, HER2K, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall0_HER2K, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasConjTrans),
    Values(14), Values(15), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HER2K, HER2K, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,0,10,0)), Values(1)));
INSTANTIATE_TEST_CASE_P(Uplo_HER2K, HER2K, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));
INSTANTIATE_TEST_CASE_P(Trans_HER2K, HER2K, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,0,10,0)), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_HER2K, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,10,0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig0_HER2K, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(clblasNoTrans, clblasConjTrans),
    Values(2510, 4300), Values(1500,4600), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes(0,0,0,9,0,0)), Values(1)));

#endif      // Correctness

#endif // DO_HER2K


#ifdef DO_SYRK
// xSYRK tests

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange, SYRK, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes()), Values(1)));
// We know, that SmallRange does not have values more that 257,
// so lda is set to 500.
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDA, SYRK,  Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 0, 501, 0, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDA, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(500, 0, 501, 0, 0, 0)), Values(1)));

// cases for the extended versions with the offsets
#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, SYRK, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(252), Values(353),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 702)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(252), Values(353),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 702)), Values(1)));

#else                               /* SHORT_TESTS || MEDIUM_TESTS */

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_0, SYRK, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(255), Values(253),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_1, SYRK, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(128), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 501)), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeEx_2, SYRK, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(75), Values(200),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 501)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_0, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(255), Values(253),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 0)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_1, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(128), Values(64),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 501)), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeEx_2, SYRK, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(transSet),
    Values(75), Values(200),
    Values(clMath::ExtraTestSizes(0, 0, 0, 500, 0, 501)), Values(1)));

#endif                              /* !SHORT_TESTS && !MEDIUM_TESTS */

// Big matrices
#if !defined(SHORT_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_0, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(2801), Values(2903),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedBig_1, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(4567), Values(4321),
    Values(clMath::ExtraTestSizes()), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(5567), Values(5321),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedBig_3, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans),
    Values(6567), Values(3321),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif // !MEDIUM_TESTS
#endif // !SHORT_TESTS

// Small matrices and Custom tests

INSTANTIATE_TEST_CASE_P(SelectedSmall_0, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(1), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined SHORT_TESTS

INSTANTIATE_TEST_CASE_P(SelectedSmall_1, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasTrans),
    Values(2), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

#if !defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(SelectedSmall_2, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(3), Values(2),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_3, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans),
    Values(4), Values(3),
    Values(clMath::ExtraTestSizes()), Values(1)));
INSTANTIATE_TEST_CASE_P(SelectedSmall_4, SYRK, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans),
    Values(17), Values(1),
    Values(clMath::ExtraTestSizes()), Values(1)));

// Custom test - use command line arguments to tweak it
INSTANTIATE_TEST_CASE_P(Custom, SYRK,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(32), Values(32),
    Values(clMath::ExtraTestSizes()), Values(1)));

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */

#endif // DO_SYRK


#ifdef DO_TRMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(ShortTRMV, TRMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_TRMV, TRMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0,9), Values(0), Values(1)));
INSTANTIATE_TEST_CASE_P(Uplo_TRMV, TRMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0,10), Values(1)));
INSTANTIATE_TEST_CASE_P(Trans_TRMV, TRMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    ValuesIn(transSet), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0,9), Values(0), Values(1)));
INSTANTIATE_TEST_CASE_P(Diag_TRMV, TRMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(clblasNoTrans), ValuesIn(diagSet), ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0,10), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),Values(0,4097),
    ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

#endif      // Correctness

#endif

#ifdef DO_TPMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(ShortTPMV, TPMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_TPMV, TPMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0,9), Values(0), Values(1)));
INSTANTIATE_TEST_CASE_P(Uplo_TPMV, TPMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0,10), Values(1)));
INSTANTIATE_TEST_CASE_P(Trans_TPMV, TPMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    ValuesIn(transSet), Values(clblasUnit),ValuesIn(smallRange),Values(0),
    Values(1), Values(0,9), Values(0), Values(1)));
INSTANTIATE_TEST_CASE_P(Diag_TPMV, TPMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(clblasNoTrans), ValuesIn(diagSet), ValuesIn(smallRange),Values(0),
    Values(1), Values(0), Values(0,10), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),Values(0,4097),
    ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

#endif      // Correctness

#endif

#ifdef DO_TRSV

#ifdef SHORT_TESTS

INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeTRSV, TRSV, Combine(
    Values(clblasRowMajor), Values(clblasUpper),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));

#endif

#ifdef MEDIUM_TESTS

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeTRSV, TRSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasTrans), Values(clblasNonUnit), ValuesIn(smallRange),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));

INSTANTIATE_TEST_CASE_P(SmallRange_VariousIncTRSV, TRSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans, clblasConjTrans), Values(clblasUnit), ValuesIn(smallRange),
    Values(0), ValuesIn(incs),  Values(0), Values(0), Values(1)));

#endif

#if !defined SHORT_TESTS && !defined MEDIUM_TESTS

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeTRSV, TRSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeTRSV, TRSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRange_BigLDATRSV, TRSV,  Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(500), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRange_BigLDATRSV, TRSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(500), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(SmallRange_VariousIncTRSV, TRSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

#endif

#endif

#ifdef DO_TPSV

#ifdef SHORT_TESTS

INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeTPSV, TPSV, Combine(
    Values(clblasRowMajor), Values(clblasUpper),
    Values(clblasNoTrans), Values(clblasUnit),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));

#endif

#ifdef MEDIUM_TESTS

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeTPSV, TPSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasTrans), Values(clblasNonUnit), ValuesIn(smallRange),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));

INSTANTIATE_TEST_CASE_P(SmallRange_VariousIncTPSV, TPSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans, clblasConjTrans), Values(clblasUnit), ValuesIn(smallRange),
    Values(0), ValuesIn(incs),  Values(0), Values(0), Values(1)));

#endif

#if !defined SHORT_TESTS && !defined MEDIUM_TESTS

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeTPSV, TPSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeTPSV, TPSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
INSTANTIATE_TEST_CASE_P(SmallRange_VariousIncTPSV, TPSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),ValuesIn(smallRange),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));
#endif

#endif

/*#ifdef DO_SYMM


	  order = ::std::tr1::get<0>(GetParam());
      side = ::std::tr1::get<1>(GetParam());
      uplo = ::std::tr1::get<2>(GetParam());
      M = ::std::tr1::get<3>(GetParam());
      N = ::std::tr1::get<4>(GetParam());
      lda = ::std::tr1::get<5>(GetParam());
      ldb = ::std::tr1::get<6>(GetParam());
  	  ldc = ::std::tr1::get<7>(GetParam());
      offa = ::std::tr1::get<8>(GetParam());
      numCommandQueues = ::std::tr1::get<9>(GetParam());

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeSYMM, SYMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(3192), Values(3192), Values(3192), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeSYMM, SYMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(3192), Values(3192), Values(3192), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(ColumnMajor_VariousLDASYMM, SYMM, Combine(
    Values(clblasColumnMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(ldaRange), ValuesIn(ldaRange), ValuesIn(ldaRange), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_VariousLDASYMM, SYMM, Combine(
    Values(clblasRowMajor), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(ldaRange), ValuesIn(ldaRange), ValuesIn(ldaRange), Values(0),
    Values(1) ) );
#endif
*/

#ifdef DO_SYR
/*
 		clblasOrder,     // order
        clblasUplo,      // uplo
        int,                // N
        double,             //alpha
        int,                // offx
        int,                // incx, should be greater than 0
        int,                // offa
        int,                // lda, 0 - undefined
        int                 // numCommandQueues
*/

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SYR, SYR, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_SYR, SYR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_SYR, SYR, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_SYR, SYR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_SYR, SYR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#else       // Correctness

INSTANTIATE_TEST_CASE_P(ALL, SYR, Combine(ValuesIn(orderSet), ValuesIn(uploSet),
	ValuesIn(smallRange), ValuesIn(realAlphaRange), ValuesIn(offsetRange), ValuesIn(incs),
	ValuesIn(offsetRange), ValuesIn(ldaRange), Values(1) ) );

#endif

#endif


#ifdef DO_SPR

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SPR, SPR, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_SPR, SPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_SPR, SPR, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_SPR, SPR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_SPR, SPR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500, 5101), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_SPR, SPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(ldaRange), Values(1) ) );

#endif      // Correctness

#endif

#ifdef DO_GER

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_GER, GER, Combine(
    Values(clblasRowMajor),ValuesIn(smallRange), ValuesIn(smallRange),
    Values(0), Values(1), Values(1), Values(0), Values(0), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_GER, GER, Combine(
    ValuesIn(orderSet), Values(61), Values(32),
    Values(0), Values(4,-11), Values(-30,1), Values(0), Values(0), Values(0),
    Values(1) ) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_GER, GER, Combine(
    ValuesIn(orderSet), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(0), Values(-10), Values(21), Values(0,9), Values(0), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig0_GER, GER, Combine(
    ValuesIn(orderSet), Values(4900), Values(3999),
    Values(0), Values(4), Values(-33), Values(0), Values(0), Values(0),
    Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_GER, GER, Combine(
    ValuesIn(orderSet), ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );

#endif      // Correctness

#endif


#ifdef DO_GERC

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_GERC, GERC, Combine(
    Values(clblasRowMajor),ValuesIn(smallRange), ValuesIn(smallRange),
    Values(0), Values(1), Values(1), Values(0), Values(0), Values(0),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_GERC, GERC, Combine(
    ValuesIn(orderSet), Values(61), Values(32),
    Values(0), Values(4,-11), Values(-30,1), Values(0), Values(0), Values(0),
    Values(1) ) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_GERC, GERC, Combine(
    ValuesIn(orderSet), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(0), Values(-10), Values(21), Values(0,9), Values(0), Values(0,19),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig0_GERC, GERC, Combine(
    ValuesIn(orderSet), Values(4900), Values(3999),
    Values(0), Values(4), Values(-33), Values(0), Values(0), Values(0),
    Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_GERC, GERC, Combine(
    ValuesIn(orderSet), ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(offsetRange),
    Values(1) ) );

#endif      // Correctness

#endif

#ifdef DO_HER
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HER, HER, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_HER, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HER, HER, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_HER, HER, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1) ) );

#else       // Correctness

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeHER, HER, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeHER, HER, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(ColumnMajor_VariousLDAHER, HER, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_VariousLDAHER, HER, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
#endif

#endif

#ifdef DO_HPR

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HPR, HPR, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_HPR, HPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HPR, HPR, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_HPR, HPR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_HPR, HPR, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500, 5101), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_HPR, HPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness

#endif


#ifdef DO_HER2
#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HER2, HER2, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_HER2, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HER2, HER2, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_HER2, HER2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0,10), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );

#else       // Correctness

INSTANTIATE_TEST_CASE_P(ColumnMajor_SmallRangeHER2, HER2, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(ldaRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_SmallRangeHER2, HER2, Combine(
        Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(ldaRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(ColumnMajor_VariousLDAHER2, HER2, Combine(
        Values(clblasColumnMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(ldaRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(RowMajor_VariousLDAHER2, HER2, Combine(
        Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),ValuesIn(ldaRange),
    Values(1) ) );

#endif

#endif

#ifdef DO_HPR2

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HPR2, HPR2, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_HPR2, HPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HPR2, HPR2, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_HPR2, HPR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    Values(0,10), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_HPR2, HPR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500, 5101), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_HPR2, HPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(ldaRange), Values(1) ) );

#endif      // Correctness

#endif


/*INSTANTIATE_TEST_CASE_P(ALL_HEMM_WITH_OFFSETS_ZERO, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    //Values(clMath::ExtraTestSizes(0, 0, 0, 12, 0, 1)),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(ALL_HEMM, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 12, 13, 15)),
    Values(1) ) );


INSTANTIATE_TEST_CASE_P(SelectedBig_0, HEMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    Values(5600), Values(5600),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    Values(1) ) );


*/



/*
INSTANTIATE_TEST_CASE_P(SYMM_VERYSMALL, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(verySmallRange), ValuesIn(verySmallRange),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    Values(1) ) );*/

/*INSTANTIATE_TEST_CASE_P(ALL_SYMM, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 1, 3, 13)),
    Values(1) ) );

INSTANTIATE_TEST_CASE_P(ALL_SYMM_WITH_OFFSETS_ZERO, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    Values(1) ) );


INSTANTIATE_TEST_CASE_P(SelectedBig_0, SYMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    Values(5600), Values(5600),ValuesIn(complexAlphaRange), ValuesIn(complexAlphaRange),
    Values(clMath::ExtraTestSizes(0, 0, 0, 0, 0, 0)),
    Values(1) ) );
*/


#ifdef DO_SYR2

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SYR2, SYR2, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_SYR2, SYR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_SYR2, SYR2, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_SYR2, SYR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,10), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_SYR2, SYR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500, 2800), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#else       // Correctness

INSTANTIATE_TEST_CASE_P(ALL, SYR2, Combine(ValuesIn(orderSet), ValuesIn(uploSet),
	ValuesIn(smallRange), ValuesIn(realAlphaRange), ValuesIn(offsetRange), ValuesIn(incs),
	ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(ldaRange), Values(1)));

#endif

#endif


#ifdef DO_SPR2

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SPR2, SPR2, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall_SPR2, SPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(15), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_SPR2, SPR2, Combine(
    ValuesIn(orderSet), Values(clblasLower), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,9), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(Uplo_SPR2, SPR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    Values(0,10), Values(1), Values(0,10), Values(0,9), Values(0), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_SPR2, SPR2, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(1500, 5101), ValuesIn(realAlphaRange),
    Values(0), Values(1), Values(0), Values(0), Values(0), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(All_SPR2, SPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(realAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(ldaRange), Values(1) ) );

#endif      // Correctness

#endif


#ifdef DO_GBMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_GBMV, GBMV, Combine(
    Values(clblasRowMajor), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_GBMV, GBMV, Combine(
    ValuesIn(orderSet), Values(clblasConjTrans),
    Values(14), Values(15), Values(10), Values(8),Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_GBMV, GBMV, Combine(
    ValuesIn(orderSet), Values(clblasNoTrans),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(Trans_GBMV, GBMV, Combine(
    Values(clblasRowMajor), ValuesIn(transSet),
    ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_GBMV, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    Values(1, 2, 4, 9), Values(3, 6, 11), Values(5), Values(7),Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig0_GBMV, GBMV, Combine(
    Values(clblasRowMajor), ValuesIn(transSet), Values(2599), Values(999),
    Values(2000), Values(565), Values(clMath::ExtraTestSizes(0,(int)30,(int)1,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_GBMV, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), ValuesIn(smallRange), ValuesIn(smallRange),
    ValuesIn(smallRange), ValuesIn(smallRange),Values(clMath::ExtraTestSizes(0,(int)22,(int)-20,9,10,0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig1_GBMV, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), Values(2510, 2300), Values(1500,2400),
    Values(2509, 2299), Values(1499,2399),Values(clMath::ExtraTestSizes(0,(int)3,(int)-2,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#endif      // Correctness

#endif // DO_GBMV



#ifdef DO_SBMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SBMV, SBMV, Combine(
    Values(clblasRowMajor), Values(clblasUpper),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_SBMV, SBMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(14), Values(10), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_SBMV, SBMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(Uplo__SBMV, SBMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_SBMV, SBMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(7), Values(5),Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig0_SBMV, SBMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(2000), Values(565), Values(clMath::ExtraTestSizes(0,(int)30,(int)1,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_SBMV, SBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0,(int)22,(int)-20,9,10,0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig1_SBMV, SBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2510, 2300), Values(1500,1700),
    Values(clMath::ExtraTestSizes(0,(int)3,(int)-2,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#endif      // Correctness

#endif // DO_SBMV

//HBMV
#ifdef DO_HBMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_HBMV, HBMV, Combine(
    Values(clblasRowMajor), Values(clblasUpper),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_HBMV, HBMV, Combine(
    ValuesIn(orderSet), Values(clblasLower),
    Values(14), Values(10), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_HBMV, HBMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(Trans_HBMV, HBMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    ValuesIn(smallRange), ValuesIn(smallRange),
    Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_HBMV, HBMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper),
    Values(7), Values(5),Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig0_HBMV, HBMV, Combine(
    Values(clblasRowMajor), Values(clblasLower),
    Values(2000), Values(565), Values(clMath::ExtraTestSizes(0,(int)30,(int)1,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_HBMV, HBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(smallRange), ValuesIn(smallRange),
    Values(clMath::ExtraTestSizes(0,(int)22,(int)-20,9,10,0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig1_HBMV, HBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2510, 2300), Values(1500,1700),
    Values(clMath::ExtraTestSizes(0,(int)3,(int)-2,9,0,6)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), Values(1)));

#endif      // Correctness

#endif // DO_HBMV


#ifdef DO_TBMV

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_TBMV, TBMV, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_TBMV, TBMV, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans), Values(clblasUnit),
    Values(14), Values(13), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Order_TBMV, TBMV, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange),Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Uplo_TBMV, TBMV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 10)), Values(1)));

INSTANTIATE_TEST_CASE_P(Trans_TBMV, TBMV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(transSet), Values(clblasUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Diag_TBMV, TBMV, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 8, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_TBMV, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(1, 2, 4, 9), Values(3), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)), Values(1)));

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_TBMV, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(clMath::ExtraTestSizes(0,(int)22,(int)-20,9,10,0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_TBMV, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(2509, 2299), Values(1499,2199), Values(clMath::ExtraTestSizes(0,(int)3,(int)-2,9,0,6)), Values(1)));

#endif      // Correctness

#endif // DO_TBMV


#ifdef DO_TBSV

#if defined(SHORT_TESTS)
/*
INSTANTIATE_TEST_CASE_P(Short_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_TBSV, TBSV, Combine(
    ValuesIn(orderSet), Values(clblasLower), Values(clblasTrans), Values(clblasUnit),
    Values(14), Values(13), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)), Values(1)));
*/

INSTANTIATE_TEST_CASE_P(Short_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedSmall0_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), Values(clblasUnit),
    Values(14), Values(13), Values(ExtraTestSizes(0, (int)-2, (int)1, 9, 0, 0)), Values(1)));

#elif defined(MEDIUM_TESTS)

INSTANTIATE_TEST_CASE_P(Order_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange),Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Uplo_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 10)), Values(1)));

INSTANTIATE_TEST_CASE_P(Trans_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), Values(clblasUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Diag_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 8, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), Values(clblasNoTrans), ValuesIn(diagSet),
    Values(1, 2, 4, 9), Values(3), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)), Values(1)));
/*
INSTANTIATE_TEST_CASE_P(Order_TBSV, TBSV, Combine(
    ValuesIn(orderSet), Values(clblasUpper), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange),Values(ExtraTestSizes(0, (int)1, (int)33, 10, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Uplo_TBSV, TBSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(clblasNoTrans), Values(clblasNonUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 0, 0, 10)), Values(1)));

INSTANTIATE_TEST_CASE_P(Trans_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasLower), ValuesIn(transSet), Values(clblasUnit),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)-33, (int)1, 0, 10, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(Diag_TBSV, TBSV, Combine(
    Values(clblasRowMajor), Values(clblasUpper), Values(clblasNoTrans), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(ExtraTestSizes(0, (int)1, (int)1, 8, 0, 0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedVerySmall_TBSV, TBSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(1, 2, 4, 9), Values(3), Values(ExtraTestSizes(0, (int)-1, (int)1, 9, 0, 0)), Values(1)));
*/
#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_TBSV, TBSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    ValuesIn(smallRange), ValuesIn(smallRange), Values(clMath::ExtraTestSizes(0,(int)22,(int)-20,9,10,0)), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_TBSV, TBSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet),
    Values(2509, 2299), Values(1499,2199), Values(clMath::ExtraTestSizes(0,(int)3,(int)-2,9,0,6)), Values(1)));

#endif      // Correctness

#endif // DO_TBSV

//COPY

#ifdef DO_COPY

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_COPY, COPY, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1), Values(1)) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_COPY, COPY, Combine(
    Values(61), Values(4, -11), Values(1), Values(0), Values(1), Values(1) ) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_COPY, COPY, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_COPY, COPY, Combine(
    Values(4900), Values(1), Values(1), Values(4), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_COPY, COPY, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness

#endif

//DOT

#ifdef DO_DOT

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_DOT, DOT, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1), Values(1), Values(1)) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_DOT, DOT, Combine(
    Values(61), Values(4, -11), Values(1), Values(0), Values(1), Values(1) , Values(1)) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_DOT, DOT, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_DOT, DOT, Combine(
    Values(4900), Values(1), Values(1), Values(4), Values(1), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_DOT, DOT, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness

#endif

#ifdef DO_DOTC

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_DOTC, DOTC, Combine(
    ValuesIn(smallRange), Values(1), Values(1), Values(1), Values(1), Values(1), Values(1)) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_DOTC, DOTC, Combine(
    Values(61), Values(4, -11), Values(1), Values(0), Values(1), Values(1) , Values(1)) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_DOTC, DOTC, Combine(
    ValuesIn(smallRange), Values(-10), Values(1), Values(1), Values(1), Values(1), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_DOTC, DOTC, Combine(
    Values(4900), Values(1), Values(1), Values(4), Values(1), Values(1), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_DOTC, DOTC, Combine(
    ValuesIn(completeRange), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), Values(1) ) );

#endif      // Correctness

#endif


#ifdef DO_SCAL

#if defined(SHORT_TESTS)
INSTANTIATE_TEST_CASE_P(Short_SCAL, SCAL, Combine(
    ValuesIn(smallRange), ValuesIn(alphaBetaRange), Values(0), Values(1), Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedSmall0_SCAL, SCAL, Combine(
    Values(61), ValuesIn(alphaBetaRange), Values(0), Values(4,-11), Values(1) ) );


#elif defined(MEDIUM_TESTS)
INSTANTIATE_TEST_CASE_P(Medium_SCAL, SCAL, Combine(
    ValuesIn(smallRange), ValuesIn(alphaBetaRange), Values(0), Values(-10), Values(1) ) );

INSTANTIATE_TEST_CASE_P(SelectedBig0_SCAL, SCAL, Combine(
    Values(4900), ValuesIn(alphaBetaRange), Values(0), Values(4), Values(1) ) );

#else       // Correctness
INSTANTIATE_TEST_CASE_P(ALL_SCAL, SCAL, Combine(
    ValuesIn(completeRange), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), ValuesIn(incs), Values(1) ) );

#endif      // Correctness

#endif


// Big matrices
#if !defined SHORT_TESTS

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(SelectedBig_0TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(2800),
    Values(0), ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_1TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(4567),
    Values(0), ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(SelectedBig_0TRSV, TRSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(2800),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));

#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(SelectedBig_0TPSV, TPSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(2800),
    Values(0), Values(1),  Values(0), Values(0), Values(1)));
#endif

#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(SelectedBig_0HER, HER, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), Values(2800), Values((double)50),
    Values(0), Values(1), Values(0), Values(0),
    Values(1) ) );
#endif


#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(SelectedBig_0HER2, HER2, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), Values(2800), Values((cl_float2)floatComplex(0,1)),
        Values(0), Values(1), Values(0),
        Values(0), Values(0),Values(1) ) );
#endif


#if !defined(MEDIUM_TESTS)

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(SelectedBig_2TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(5567),
    Values(0), ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_3TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(6567),
    Values(0), ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_4TRMV, TRMV, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(7567),
    Values(0), ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TPMV
INSTANTIATE_TEST_CASE_P(SelectedBig_2TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(5567),Values(0),
    ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_3TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(6567),Values(0),
    ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_4TPMV, TPMV, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet),
   Values(clblasNoTrans), ValuesIn(diagSet), Values(7567),Values(0),
   ValuesIn(incs), Values(0, 10), Values(0, 9), Values(1)));
#endif


#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(SelectedBig_1TRSV, TRSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(4567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2TRSV, TRSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(5567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_3TRSV, TRSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(6567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_4TRSV, TRSV, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(7567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(SelectedBig_1TPSV, TPSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(4567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_2TPSV, TPSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(5567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_3TPSV, TPSV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(6567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));

INSTANTIATE_TEST_CASE_P(SelectedBig_4TPSV, TPSV, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(7567),
    Values(0), ValuesIn(incs),  Values(0,10), Values(0,9), Values(1)));
#endif


#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(SelectedBig_1HER, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(3192), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_2HER, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2048), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_3HER, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(3192), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_4HER, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2055), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    Values(1) ) );
#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(SelectedBig_1HER2, HER2, Combine(
        ValuesIn(orderSet), ValuesIn(uploSet), Values(3192), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),
    ValuesIn(offsetRange), ValuesIn(ldaRange),Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_2HER2, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2048), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
        ValuesIn(ldaRange),Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_3HER2, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(3192), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange),
    ValuesIn(ldaRange),Values(1) ) );
INSTANTIATE_TEST_CASE_P(SelectedBig_4HER2, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(2055), ValuesIn(complexAlphaRange),
    ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),ValuesIn(offsetRange),
    ValuesIn(ldaRange),Values(1) ) );
#endif


#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */

// Small matrices

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_0TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(1),
    Values(0), Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TPMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_0TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(1),Values(0),
    Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(SelectedSmall_0TRSV, TRSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasNonUnit), Values(1),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(SelectedSmall_0TPSV, TPSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasNonUnit), Values(1),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
#endif


#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(SelectedSmall_0HER, HER, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), Values(4), ValuesIn(realAlphaRange),
    Values(0), ValuesIn(incs), Values(0,9), Values(0,11),
    Values(1) ) );
#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(SelectedSmall_0HER2, HER2, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet), Values(4), ValuesIn(complexAlphaRange),
    Values(0,7), ValuesIn(incs), Values(0,9), Values(0,11),
    Values(0),Values(1) ) );
#endif


#if !defined SHORT_TESTS

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_1TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(2),
    Values(0), Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TPMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_1TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(2),Values(0),
    Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(SelectedSmall_1TRSV, TRSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasUnit), Values(2),
    Values(0), Values(1),  Values(10), Values(9), Values(1)));
#endif


#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(SelectedSmall_1HER, HER, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet), Values(12), ValuesIn(realAlphaRange),
    Values(0), ValuesIn(incs), Values(0), Values(1),
    Values(1) ) );

#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(SelectedSmall_1HER2, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(12), ValuesIn(complexAlphaRange),
    Values(0,1), ValuesIn(incs), Values(0),Values(9),
    Values(0),Values(1) ) );
#endif


#if !defined(MEDIUM_TESTS)

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_2TRMV, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(13),
    Values(0), Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TPMV
INSTANTIATE_TEST_CASE_P(SelectedSmall_2TPMV, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(clblasNoTrans), ValuesIn(diagSet), Values(13),Values(0),
    Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(SelectedSmall_2TRSV, TRSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasNoTrans), Values(clblasNonUnit), Values(13),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(SelectedSmall_2TPSV, TPSV, Combine(
    Values(clblasRowMajor), ValuesIn(uploSet),
    Values(clblasTrans), Values(clblasUnit), Values(13),
    Values(0), Values(1), Values(0,10), Values(0,9), Values(1)));
#endif


#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(SelectedSmallHER_2HER, HER, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), Values(65), ValuesIn(realAlphaRange),
    Values(0), ValuesIn(incs), Values(0), Values(0),
    Values(1) ) );

#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(SelectedSmallHER2_2HER2, HER2, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), Values(65), ValuesIn(complexAlphaRange),
   Values(0), ValuesIn(incs), Values(0), Values(0),
   Values(0), Values(1) ) );
#endif

#endif /* !MEDIUM_TESTS */
#endif /* !SHORT_TESTS */

// Custom test - use command line arguments to tweak it
#if !defined SHORT_TESTS && !defined MEDIUM_TESTS
#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(Custom, TRMV,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet), Values(32),
    Values(0), Values(1), Values(0, 10), Values(0, 9), Values(1)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(Custom, TRSV,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet), Values(32),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(Custom, TPSV,  Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet), Values(32),
    Values(0), Values(1),  Values(0,10), Values(0,9), Values(1)));
#endif

#ifdef DO_GER
INSTANTIATE_TEST_CASE_P(Custom, GER, Combine(
    ValuesIn(orderSet),
    Values(32), Values(32),
    Values(0), Values(1), Values(1), Values(0, 9), Values(0, 11), Values(0, 10),
    Values(1) ) );
#endif

#ifdef DO_GERC
INSTANTIATE_TEST_CASE_P(Custom, GERC, Combine(
    ValuesIn(orderSet),
    Values(32), Values(32),
    Values(0), Values(1), Values(1), Values(0, 9), Values(0, 11), Values(0, 10),
    Values(1) ) );
#endif

#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(Custom, HER, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), Values(32), Values(99.0),
    Values(0), Values(1), Values(6, 2), Values(0, 5),
    Values(1) ) );

#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(Custom, HER2, Combine(
   ValuesIn(orderSet), ValuesIn(uploSet), Values(32), ValuesIn(complexAlphaRange),
   Values(0), Values(1), Values(0), Values(0),Values(40), Values(1) ) );
#endif

#endif /* !SHORT_TESTS */
// Multiple command queues tests

#if defined SHORT_TESTS
#define QUEUES_TEST_MATRIX_SIZES 257
#elif defined MEDIUM_TESTS
#define QUEUES_TEST_MATRIX_SIZES 385
#else
#define QUEUES_TEST_MATRIX_SIZES 513,1025
#endif

#if !defined(SHORT_TESTS)

#ifdef DO_GEMM
INSTANTIATE_TEST_CASE_P(MultipleQueues, GEMM, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES),
    Values(QUEUES_TEST_MATRIX_SIZES),
    Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#if !defined(MEDIUM_TESTS)


#ifdef DO_TRMM

INSTANTIATE_TEST_CASE_P(MultipleQueues, TRMM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#ifdef DO_TRSM
INSTANTIATE_TEST_CASE_P(MultipleQueues, TRSM, Combine(
    ValuesIn(orderSet), ValuesIn(sideSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#endif                      /* MEDIUM_TESTS */


#ifdef DO_GEMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, GEMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), ValuesIn(numQueues)));
#endif

#ifdef DO_SYMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, SYMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), ValuesIn(numQueues)));
#endif

#ifdef DO_SYR2K
INSTANTIATE_TEST_CASE_P(MultipleQueues, SYR2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#ifdef DO_SYRK
INSTANTIATE_TEST_CASE_P(MultipleQueues, SYRK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#if !defined MEDIUM_TESTS

#ifdef DO_HERK
INSTANTIATE_TEST_CASE_P(MultipleQueues, HERK, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#ifdef DO_HER2K
INSTANTIATE_TEST_CASE_P(MultipleQueues, HER2K, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange),
    Values(clMath::ExtraTestSizes()), ValuesIn(numQueues)));
#endif

#ifdef DO_TRMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, TRMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1), Values(0, 10), Values(0, 9), ValuesIn(numQueues)));
#endif

#ifdef DO_TPMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, TPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet),
    ValuesIn(transSet), ValuesIn(diagSet),Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1), Values(0, 10), Values(0, 9), ValuesIn(numQueues)));
#endif

#ifdef DO_HEMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, HEMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(alphaBetaRange),
    ValuesIn(alphaBetaRange), Values(0, 10), Values(0, 9), Values(0, 8), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), ValuesIn(numQueues)));
#endif

#ifdef DO_HPMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, HPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(alphaBetaRange),
    ValuesIn(alphaBetaRange), Values(0, 10), Values(0, 9), Values(0, 8), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), ValuesIn(numQueues)));
#endif


#ifdef DO_SPMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, SPMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(alphaBetaRange),
    ValuesIn(alphaBetaRange), Values(0, 10), Values(0, 9), Values(0, 8), Values(clMath::ExtraTestSizes(0, 1, 1, 0, 0, 0)), ValuesIn(numQueues)));
#endif

#ifdef DO_TRSV
INSTANTIATE_TEST_CASE_P(MultipleQueues, TRSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasConjTrans), ValuesIn(diagSet),Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1),  Values(0,10), Values(0,9), ValuesIn(numQueues)));
#endif

#ifdef DO_TPSV
INSTANTIATE_TEST_CASE_P(MultipleQueues, TPSV, Combine(
    Values(clblasColumnMajor), ValuesIn(uploSet),
    Values(clblasTrans), ValuesIn(diagSet),Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1),  Values(0,10), Values(0,9), ValuesIn(numQueues)));
#endif

#ifdef DO_SYR
INSTANTIATE_TEST_CASE_P(MultipleQueues, SYR, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(realAlphaRange),
	ValuesIn(offsetRange), ValuesIn(incs),
	ValuesIn(offsetRange), ValuesIn(ldaRange), ValuesIn(numQueues) ) );

#endif

#ifdef DO_SPR
INSTANTIATE_TEST_CASE_P(MultipleQueues, SPR, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(realAlphaRange),
	ValuesIn(offsetRange), ValuesIn(incs),
	ValuesIn(offsetRange), ValuesIn(ldaRange), ValuesIn(numQueues) ) );

#endif

#ifdef DO_GER
INSTANTIATE_TEST_CASE_P(MultipleQueues, GER, Combine(
    ValuesIn(orderSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1), Values(1), Values(0, 9), Values(0, 11), Values(0, 10),
    ValuesIn(numQueues) ) );
#endif

#ifdef DO_GERC
INSTANTIATE_TEST_CASE_P(MultipleQueues, GERC, Combine(
    ValuesIn(orderSet),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(0), Values(1), Values(1), Values(0, 9), Values(0, 11), Values(0, 10),
    ValuesIn(numQueues) ) );
#endif

#ifdef DO_HER
INSTANTIATE_TEST_CASE_P(MultipleQueues, HER, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), Values(1), Values(0), Values(0),
    ValuesIn(numQueues) ) );

#endif

#ifdef DO_HPR
INSTANTIATE_TEST_CASE_P(MultipleQueues, HPR, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(realAlphaRange),
    ValuesIn(ldaRange), Values(1), Values(0), Values(0),
    ValuesIn(numQueues) ) );

#endif

#ifdef DO_HER2
INSTANTIATE_TEST_CASE_P(MultipleQueues, HER2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0),Values(1), ValuesIn(ldaRange),
    ValuesIn(numQueues) ) );
#endif

#ifdef DO_HPR2
INSTANTIATE_TEST_CASE_P(MultipleQueues, HPR2, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(complexAlphaRange),
    Values(0), Values(1), Values(0),Values(1), ValuesIn(ldaRange),
    ValuesIn(numQueues) ) );
#endif


#ifdef DO_SYR2
#endif

#ifdef DO_SPR2
INSTANTIATE_TEST_CASE_P(MultipleQueues, SPR2, Combine(
	ValuesIn(orderSet), ValuesIn(uploSet), Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(realAlphaRange),
	ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange),
	ValuesIn(offsetRange), ValuesIn(ldaRange), ValuesIn(numQueues) ) );
#endif

#ifdef DO_GBMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, GBMV, Combine(
    ValuesIn(orderSet), ValuesIn(transSet), Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(QUEUES_TEST_MATRIX_SIZES),Values(clMath::ExtraTestSizes(0,(int)1,(int)1,0,0,0)),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), ValuesIn(numQueues)));
#endif

#ifdef DO_TBMV
INSTANTIATE_TEST_CASE_P(MultipleQueues, TBMV, Combine(
    ValuesIn(orderSet), ValuesIn(uploSet), ValuesIn(transSet), ValuesIn(diagSet), Values(QUEUES_TEST_MATRIX_SIZES),
    Values(QUEUES_TEST_MATRIX_SIZES), Values(clMath::ExtraTestSizes(0,(int)1,(int)1,0,0,0)), ValuesIn(numQueues)));
#endif

#ifdef DO_SCAL
INSTANTIATE_TEST_CASE_P(MultipleQueues, SCAL, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(numQueues)));
#endif

#ifdef DO_COPY
INSTANTIATE_TEST_CASE_P(MultipleQueues, COPY, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues)));
#endif

#ifdef DO_SWAP
INSTANTIATE_TEST_CASE_P(MultipleQueues, SWAPXY, Combine(
        Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(offsetRange), ValuesIn(incs),
        ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(numQueues) ) );
#endif

#ifdef DO_DOT
INSTANTIATE_TEST_CASE_P(MultipleQueues, DOT, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues) ) );
#endif

#ifdef DO_DOTC
INSTANTIATE_TEST_CASE_P(MultipleQueues, DOTC, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues) ) );
#endif

#ifdef DO_AXPY
INSTANTIATE_TEST_CASE_P(MultipleQueues, AXPY, Combine(
        Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(alphaBetaRange), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(numQueues)));
#endif

#ifdef DO_ROTG
INSTANTIATE_TEST_CASE_P(MultipleQueues, ROTG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues)));
#endif

#ifdef DO_ROTM
INSTANTIATE_TEST_CASE_P(MultipleQueues, ROTM, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(sflagRange), ValuesIn(numQueues)));
#endif

#ifdef DO_ROT
INSTANTIATE_TEST_CASE_P(MultipleQueues, ROT, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(offsetRange), ValuesIn(incs), ValuesIn(offsetRange), ValuesIn(incs),
    ValuesIn(alphaBetaRange), ValuesIn(alphaBetaRange), ValuesIn(numQueues)));
#endif

#ifdef DO_ROTMG
INSTANTIATE_TEST_CASE_P(MultipleQueues, ROTMG, Combine(
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(offsetRange),
    ValuesIn(offsetRange), ValuesIn(sflagRange), ValuesIn(numQueues)));
#endif

#ifdef DO_NRM2
INSTANTIATE_TEST_CASE_P(MultipleQueues, NRM2, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues) ) );
#endif

#ifdef DO_ASUM
INSTANTIATE_TEST_CASE_P(MultipleQueues, ASUM, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues) ) );
#endif

#ifdef DO_iAMAX
INSTANTIATE_TEST_CASE_P(MultipleQueues, iAMAX, Combine(
    Values(QUEUES_TEST_MATRIX_SIZES), ValuesIn(incs),
    ValuesIn(offsetRange), ValuesIn(offsetRange), ValuesIn(numQueues) ) );
#endif

#endif /* !MEDIUM_TESTS */
#endif /* SHORT_TESTS */

#undef QUEUES_TEST_MATRIX_SIZES

///////////////////////////////////////////////////////////////////////////////

int
main(int argc, char *argv[])
{
    ::clMath::BlasBase *base;
    TestParams params;
    int ret;

    if( (argc > 1) && ( !strcmp(argv[1], "--test-help") || !strcmp(argv[1], "-?") || !strcmp(argv[1], "-h") ) )
	{
        printUsage("test-correctness");
		::testing::InitGoogleTest(&argc, argv);
        return 0;
    }

	//	The library takes an environment variable to control how to cache kernels; automate the setting of this
	//	environment variable in our different test programs to set it to reasonable values
	//	Read environmental variable to limit or disable ( 0 ) the size of the kernel cache in memory
	char* kCacheEnv = getenv( "AMD_CLBLAS_KCACHE_LIMIT_MB" );
	if( kCacheEnv == NULL )
	{
#if defined( SHORT_TESTS )
#else
	putenv( (char*)"AMD_CLBLAS_KCACHE_LIMIT_MB=256" );
#endif
	}

    ::testing::InitGoogleTest(&argc, argv);
    ::std::cerr << "Initialize OpenCL and clblas..." << ::std::endl;
    base = ::clMath::BlasBase::getInstance();
    if (base == NULL) {
        ::std::cerr << "Fatal error, OpenCL or clblas initialization failed! "
                       "Leaving the test." << ::std::endl;
        return -1;
    }

    base->setSeed(DEFAULT_SEED);

    if (argc != 1) {
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
        if (params.optFlags & SET_INCY) {
            base->setIncY(params.incy);
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

	/* Use of image based buffers is deprecated
    if (base->useImages()) {
        if (base->addScratchImages()) {
            std::cerr << "FATAL ERROR, CANNOT CREATE SCRATCH IMAGES!" << std::endl;
        }
    }
	*/

    base->printEnvInfo();
    ret = RUN_ALL_TESTS();

    if (base->useImages()) {
        base->removeScratchImages();
    }

    /*
     * Explicitely tell the singleton to release all resources,
     * before we return from main.
     */
    base->release( );

    return ret;
}
