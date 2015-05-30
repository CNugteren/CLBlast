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

#ifndef ROTM_H_
#define ROTM_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class ROTM : public TestWithParam<
    ::std::tr1::tuple<
	int,				// N
    int,             // offx
    int,                // incx
    int,             // offy
    int,                // incy
	int,				// offParam
    ComplexLong,        // SFLAG Param
    int                 // numCommandQueues
        > >
{
public:
    void getParams(TestParams *params)
    {
        params->N = N;
        params->offa= offa;  // corrosponds to offx
		params->offb = offb; // corrosponds to offy
		params->offc = offc; // corrosponds to offParam
        params->incx = incx;
        params->incy = incy;
        params->alpha = alpha; // corrosponds to sflagparam
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        N = ::std::tr1::get<0>(GetParam());
        offa = ::std::tr1::get<1>(GetParam());
		incx = ::std::tr1::get<2>(GetParam());
		offb = ::std::tr1::get<3>(GetParam());
        incy = ::std::tr1::get<4>(GetParam());
        offc = ::std::tr1::get<5>(GetParam());
        alpha = ::std::tr1::get<6>(GetParam());
        numCommandQueues = ::std::tr1::get<7>(GetParam());

        base = ::clMath::BlasBase::getInstance();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues)
        {
            numCommandQueues = base->numCommandQueues();
        }

		printTestParams(N, offa, incx, offb, incy, offc, alpha);
		::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    size_t N, offa, offb, offc;
    int incx, incy;
    ComplexLong alpha;
    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};
#endif
