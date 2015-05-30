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


#if !defined(SYR2_PACKED)
    #ifndef SYR2_H
        #define SYR2_H
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

#ifndef SYR2_PACKED
class SYR2 : public TestWithParam<
#else
class SPR2 : public TestWithParam<
#endif

    ::std::tr1::tuple<
    	clblasOrder,     // order
		clblasUplo,		// uplo
        int,                // N
		double,				//alpha
		int,				// offx
        int,                // incx, should be greater than 0
		int,				// offy,
		//int,				// incy, should be greater than 0.
							// Since tuple doesnot allow more than 10 arguments we assume incy = incx;
		int,				// offa
        int,                // lda, 0 - undefined
        int             	// numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
		params->uplo  = uplo;
        params->seed  = seed;
        params->N     = N;
		params->alpha.re = (long)alpha; // This will cast alpha to long. So the real value that is
										// passed is not the same as what is set in the test case
		params->offBX  = offx;
        params->incx  = incx;
		params->offCY  = offy;
		params->incy  = incy;
		params->offa  = offa;
        params->lda   = lda;

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
		offy  = ::std::tr1::get<6>(GetParam());
		offa  = ::std::tr1::get<7>(GetParam());
        lda   = ::std::tr1::get<8>(GetParam());

  	    numCommandQueues = ::std::tr1::get<9>(GetParam());

        incy  = incx; //GTest allows only 10 arguments to be passed and
					  //hence we define incy to be equivalent to incx.

		#ifndef SYR2_PACKED
		lda = ::std::max( lda, N );
        #else
            lda =0;
        #endif

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useN()) {
            N = base->N();
        }

		printTestParams(order, uplo, N, alpha, offx, incx, offy, incy, offa, lda);

        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
	clblasUplo uplo;
    size_t N;
    size_t lda;
    int incx, incy;
    size_t offx, offy, offa;

    unsigned int seed;

	double alpha;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX, imageY;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SYR2_H_
