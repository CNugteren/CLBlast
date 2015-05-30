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

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-math.h>

using ::testing::TestWithParam;

class DOTC : public TestWithParam<
    ::std::tr1::tuple<
    int,                // N
    int,                // incx, should be greater than 0
    int,                //incy
	int,				//offx
	int,				//offy
	int,				//offa -- for offDP
    int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->N = N;
        params->incx = incx;
        params->incy = incy;
		params->offBX = offx;
		params->offCY = offy;
		params->offa = offDP;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        //size_t lenX;

        N = ::std::tr1::get<0>(GetParam());
        incx = ::std::tr1::get<1>(GetParam());
        incy = ::std::tr1::get<2>(GetParam());
		offx = ::std::tr1::get<3>(GetParam());
		offy = ::std::tr1::get<4>(GetParam());
		offDP = ::std::tr1::get<5>(GetParam());
        numCommandQueues = ::std::tr1::get<6>(GetParam());

        base = ::clMath::BlasBase::getInstance();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useN()) {
            N = base->N();
        }

		printTestParams(N, offx, incx, offy, incy);
        ::std::cerr << "offDP = " << offDP << ::std::endl;
		::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    size_t N;
    int incx;
    int incy;
    size_t offx, offy, offDP;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};


