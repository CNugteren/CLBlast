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

#ifndef SWAP__H_
#define SWAP__H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>

using namespace clMath;
using ::testing::TestWithParam;

// Name SWAP creates problem in gTest
class SWAPXY : public TestWithParam<

    ::std::tr1::tuple<
    int,                // N
    int,                // offBX
    int,                // incx, should not be  0
	int,				//offCY
	int,				//incy, should not be 0
    int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->N = N;
        params->offBX = offBX;
        params->incx = incx;
		params->offCY = offCY;
		params->incy = incy;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        N = ::std::tr1::get<0>(GetParam());
        offBX = ::std::tr1::get<1>(GetParam());
        incx = ::std::tr1::get<2>(GetParam());
		offCY = ::std::tr1::get<3>(GetParam());
		incy = ::std::tr1::get<4>(GetParam());
        numCommandQueues = ::std::tr1::get<5>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useN()) {
            N = base->N();
        }

		printTestParams(N, offBX, incx, offCY, incy);
		::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    size_t N;
    size_t offBX;
    int incx;
    size_t offCY;
	int incy;
	unsigned int seed;

    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif
