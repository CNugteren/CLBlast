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
#include <trsm.h>

#include "TrxmPerformanceTest.cpp"

using namespace std;
using namespace clMath;

// strsm performance test case
TEST_P(TRSM, strsm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<float>::runInstance(FN_STRSM, &params);
}

// dtrsm performance test case
TEST_P(TRSM, dtrsm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<double>::runInstance(FN_DTRSM, &params);
}

// ctrsm performance test case
TEST_P(TRSM, ctrsm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<FloatComplex>::runInstance(FN_CTRSM, &params);
}

// ztrsm performance test case
TEST_P(TRSM, ztrsm)
{
    TestParams params;

    getParams(&params);
    TrxmPerformanceTest<DoubleComplex>::runInstance(FN_ZTRSM, &params);
}
