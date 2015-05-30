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

#ifndef ROTG_H_
#define ROTG_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class ROTG : public TestWithParam<
    ::std::tr1::tuple<
	int,				//offsa
	int,				//offsb
    int,                //offc
    int,                //offs
    int                 //numCommandQueues
        > >
{
public:
    void getParams(TestParams *params)
    {
        params->offa = offC;
        params->offb = offS;
		params->offBX = offSA;
		params->offCY = offSB;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        offSA = ::std::tr1::get<0>(GetParam());
        offSB = ::std::tr1::get<1>(GetParam());
		offC = ::std::tr1::get<2>(GetParam());
		offS = ::std::tr1::get<3>(GetParam());
        numCommandQueues = ::std::tr1::get<4>(GetParam());

        base = ::clMath::BlasBase::getInstance();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues)
        {
            numCommandQueues = base->numCommandQueues();
        }

		printTestParams(offSA, offSB, offC, offS);
			::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    size_t offSA, offSB, offC, offS;

    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};
#endif
