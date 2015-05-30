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

class iAMAX : public TestWithParam<
    ::std::tr1::tuple<
    int,                // N
    int,                // incx, should be greater than 0
	int,				//offx
	int,				//offa -- for offiAmax
    int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->N = N;
        params->incx = incx;
		params->offBX = offx;
		params->offa = offiAmax;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        N = ::std::tr1::get<0>(GetParam());
        incx = ::std::tr1::get<1>(GetParam());
		offx = ::std::tr1::get<2>(GetParam());
		offiAmax = ::std::tr1::get<3>(GetParam());
        numCommandQueues = ::std::tr1::get<4>(GetParam());

        base = ::clMath::BlasBase::getInstance();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useN()) {
            N = base->N();
        }

		printTestParams(N, offx, incx);
        ::std::cerr << "offiAmax = " << offiAmax << ::std::endl;
    }

    size_t N;
    int incx;
    size_t offx, offiAmax;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};


