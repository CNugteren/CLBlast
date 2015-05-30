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

#ifndef ROTMG_H_
#define ROTMG_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-random.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class ROTMG : public TestWithParam<
    ::std::tr1::tuple<
	int,				// offD1
    int,                // offD2
    int,                // offBX
    int,                // offCY
    int,                // offParam
    ComplexLong,        // SFLAG Param
    int                 // numCommandQueues
        > >
{
public:
    void getParams(TestParams *params)
    {
        params->offBX= offBX;   // corrosponds to offx
		params->offCY = offCY;  // corrosponds to offy
		params->offa = offa;    // corrosponds to offD1
        params->offb = offb;    // corrosponds to offD2
        params->offc = offc;    // corrospods to offParam
        params->alpha = alpha;  // corrosponds to sflagparam
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        offBX = ::std::tr1::get<0>(GetParam());
        offCY = ::std::tr1::get<1>(GetParam());
		offa = ::std::tr1::get<2>(GetParam());
		offb = ::std::tr1::get<3>(GetParam());
        offc = ::std::tr1::get<4>(GetParam());
        alpha = ::std::tr1::get<5>(GetParam());
        numCommandQueues = ::std::tr1::get<6>(GetParam());

        base = ::clMath::BlasBase::getInstance();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues)
        {
            numCommandQueues = base->numCommandQueues();
        }

		printTestParams(offBX, offCY, offa, offb, offc, alpha);
		::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    int offa, offb, offc, offBX, offCY;
    ComplexLong alpha;
    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

template <typename T>
static void
randomRotmg(
    T *D1,
    T *D2,
    T *X,
    T *Y,
    T *PARAM
    )
{
    // Since rotmg involves upto 3 multiplication on an element, taking cube-root
    cl_double bound = pow(UPPER_BOUND<T>(), (1.0/3)) / 2.0;

    *D1 = random<T>(bound);
    *D2 = random<T>(bound);
    *X = random<T>(bound);
    *Y = random<T>(bound);

    // Populate PARAM. Flag in PARAM[0] is expected to be set outside this function call
    for(int i=1; i<=4; i++) {
        PARAM[i] = random<T>(bound);
    }
}

#endif
