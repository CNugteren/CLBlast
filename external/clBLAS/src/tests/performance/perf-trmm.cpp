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


#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <blas-internal.h>
#include <BlasBase.h>
#include <trmm.h>

#include "TrxmPerformanceTest.cpp"

/*
 * NOTE: operation factor takes into account the same as for
 *       gemm but also the fact that only a half of data is actually
 *       useful
 */

using namespace std;
using namespace clMath;


// strmm performance test case
TEST_P(TRMM, strmm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<float>::runInstance(FN_STRMM, &params);
}

// dtrmm performance test case
TEST_P(TRMM, dtrmm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<double>::runInstance(FN_DTRMM, &params);
}

// ctrmm performance test case
TEST_P(TRMM, ctrmm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<FloatComplex>::runInstance(FN_CTRMM, &params);
}

// ztrmm performance test case
TEST_P(TRMM, ztrmm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<DoubleComplex>::runInstance(FN_ZTRMM, &params);
}
