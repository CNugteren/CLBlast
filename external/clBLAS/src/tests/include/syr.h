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


#if !defined(SYR_PACKED)
    #ifndef SYR_H
        #define SYR_H
    #else
        #define DUPLICIT
    #endif
#endif

#ifndef DUPLICIT

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-math.h>

using ::testing::TestWithParam;

#ifndef SYR_PACKED
class SYR : public TestWithParam<
#else
class SPR : public TestWithParam<
#endif
    ::std::tr1::tuple<
    	clblasOrder,     // order
		clblasUplo,		// uplo
        int,                // N
		double,				// alpha
		int,				// offx
        int,                // incx, should be greater than 0
		int,				// offa
        int,                // lda, 0 - undefined
        int             	// numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
		params->uplo = uplo;
        params->seed = seed;
        params->N = N;
		params->alpha.re = (long)alpha; // This will cast alpha to long. So the real value that is
										// passed is not the same as what is set in the test case
		params->offBX = offx;
        params->incx = incx;
		params->offa = offa;
        params->lda = lda;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        order = ::std::tr1::get<0>(GetParam());
        uplo  = ::std::tr1::get<1>(GetParam());
        N     = ::std::tr1::get<2>(GetParam());
		alpha = ::std::tr1::get<3>(GetParam());
		offx  = ::std::tr1::get<4>(GetParam());
        incx  = ::std::tr1::get<5>(GetParam());
		offa  = ::std::tr1::get<6>(GetParam());
        lda   = ::std::tr1::get<7>(GetParam());

  	    numCommandQueues = ::std::tr1::get<8>(GetParam());

        #ifndef SYR_PACKED
		lda = ::std::max( lda, N );
        #else
        lda =0;
        #endif

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();
		//base->setAlpha(50);
		//alpha = 50.0;

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useN()) {
            N = base->N();
        }
//		if (base->useAlpha()) {
//			alpha = base->Alpha();
//		}

		printTestParams(order, uplo, N, alpha, offx, incx, offa, lda);

        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
	clblasUplo uplo;
    size_t N;
    size_t lda;
    int incx;
    size_t offx, offa;

    unsigned int seed;

	double alpha;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SYR_H_
