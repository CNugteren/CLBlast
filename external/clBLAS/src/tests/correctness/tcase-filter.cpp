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


#include "tcase-filter.h"

#if defined(SHORT_TESTS) || defined(MEDIUM_TESTS)

static __inline size_t
selectSize(size_t orig, size_t alt)
{
    return (orig) ? orig : alt;
}

static size_t
nonZeroSize(size_t size1, size_t size2, size_t size3)
{
    size_t r = 0;

    if (size1) {
        r = size1;
    }
    else if (size2) {
        r = size2;
    }
    else {
        r = size3;
    }

    return r;
}

static int
sizeEquCount(size_t size1, size_t size2, size_t size3)
{
    int cnt = 0;

    cnt += static_cast<int>(size1 == size2);
    cnt += static_cast<int>(size2 == size3);
    cnt += static_cast<int>(size1 == size3);

    return cnt;
}

static __inline bool
isEquToAny(size_t size, size_t alt1, size_t alt2, size_t alt3)
{
    return ((size == alt1) || (size == alt2) || (size == alt3));
}

static __inline bool
isRealConjugation(const TestParams *params, bool isComplex)
{
    return !isComplex &&
           ((params->transA == clblasConjTrans) ||
            (params->transB == clblasConjTrans));
}

#endif                          /* SHORT_TESTS || MEDIUM_TESTS */

#if defined(SHORT_TESTS)

bool
canCaseBeSkipped(const TestParams *params, bool isComplex)
{
    size_t s;
    size_t m, n, k, lda, ldb, ldc;

    // skip cases with conjugated transposition for real data
    if (isRealConjugation(params, isComplex)) {
        return true;
    }

    /*
     * Enable only cases at which all the problem dimensions are equal
     * to each other
     */
    s = nonZeroSize(params->M, params->N, params->K);
    m = selectSize(params->M, s);
    n = selectSize(params->N, s);
    k = selectSize(params->K, s);
    if (sizeEquCount(m, n, k) < 3) {
        return true;
    }

    /*
     * filter BigLDA cases
     */
    /*
    s = nonZeroSize(params->lda, params->ldb, params->ldc);
    lda = selectSize(params->lda, s);
    ldb = selectSize(params->ldb, s);
    ldc = selectSize(params->ldc, s);
    if (sizeEquCount(lda, ldb, ldc) < 3) {
        return true;
    }
	
    if (!isEquToAny(lda, m, n, k)) {
        return true;
    }
    */
    return false;
}

#elif defined(MEDIUM_TESTS)     /* SHORT_TESTS */

#include <algorithm>

#include <stdio.h>

/*
 * Evaluate best vector length that buffer with such leading dimension
 * would have for such leading dimension.
 */
static unsigned int
prognozedVecLen(size_t ld)
{
    size_t u = static_cast<size_t>(1) << (sizeof(size_t) * 8 - 1);
    size_t vecLen;

    // typically vecLen will not exceed 8
    ld %= 8;
    if (ld == 0) {
        return 8;
    }
    else if (ld == 1) {
        return 1;
    }

    // find the highest non zero bit
    for (; (u != 0) && !(u & ld); u >>= 1);

    /*
     * Evaluated as minimum of modules based operation results against
     * upper and lower power of 2 bounds
     */
    vecLen = ld - u;
    u >>= 1;
    vecLen = ::std::min(vecLen, u - ld);

    return static_cast<unsigned int>(vecLen);
}

bool
canCaseBeSkipped(const TestParams *params, bool isComplex)
{
    size_t s;
    size_t m, n, k, lda, ldb, ldc;
    int bigCnt = 0;
    unsigned int vecLen;

    // skip cases with conjugated transposition for real data
    if (isRealConjugation(params, isComplex)) {
        return true;
    }

    // set of cases for extended versions is really tiny, so enable them all
    if (params->offA || params->offBX || params->offCY) {
        return false;
    }

    s = nonZeroSize(params->M, params->N, params->K);
    m = selectSize(params->M, s);
    n = selectSize(params->N, s);
    k = selectSize(params->K, s);

    // enable BigLDA cases when problem dimensions all are equal to each other
    s = nonZeroSize(params->lda, params->ldb, params->ldc);
    lda = selectSize(params->lda, s);
    ldb = selectSize(params->ldb, s);
    ldc = selectSize(params->ldc, s);
    bigCnt += static_cast<int>(!isEquToAny(lda, m, n, k));
    bigCnt += static_cast<int>(!isEquToAny(ldb, m, n, k));
    bigCnt += static_cast<int>(!isEquToAny(ldc, m, n, k));
    if (bigCnt) {
        if (sizeEquCount(m, n, k) < 3) {
            return true;
        }
        else {
            return false;
        }
    }

    // enable only cases at which buffers will have the same vectorization
    vecLen = prognozedVecLen(lda);
    if ((prognozedVecLen(ldb) != vecLen) ||
        (prognozedVecLen(ldc) != vecLen)) {

        return true;
    }

    return false;
}

#else                           /* MEDIUM_TESTS */

bool
canCaseBeSkipped(const TestParams *params, bool isComplex)
{
    (void)params;
    (void)isComplex;
    return false;
}

#endif                          /* !SHORT_TESTS && !MEDIUM_TESTS */

