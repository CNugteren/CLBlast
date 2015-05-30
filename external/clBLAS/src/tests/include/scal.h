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


#ifndef SCAL_H_
#define SCAL_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <common.h>

using namespace clMath;
using ::testing::TestWithParam;

class SCAL : public TestWithParam<
    ::std::tr1::tuple<
        int,                // N
        ComplexLong,		// alpha
        int,                // offx
        int,                // incx
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->N = N;
        params->alpha = paramAlpha;
        params->offBX = offx;
        params->incx = incx;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        N = ::std::tr1::get<0>(GetParam());
		paramAlpha = ::std::tr1::get<1>(GetParam());
		offx = ::std::tr1::get<2>(GetParam());
		incx = ::std::tr1::get<3>(GetParam());
        numCommandQueues = ::std::tr1::get<4>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        printTestParams(N, paramAlpha, offx, incx);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    size_t N;
    unsigned int seed;
    size_t offx;
    int incx;
    bool useAlpha;
    ComplexLong paramAlpha;
    ::clMath::BlasBase *base;
    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SCAL_H_
