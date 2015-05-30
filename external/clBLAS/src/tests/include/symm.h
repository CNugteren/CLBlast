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


#ifndef SYMM_H_
#define SYMM_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>

using namespace clMath;
using ::testing::TestWithParam;

class SYMM : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,		 // order
        clblasSide,		 // side
        clblasUplo,		// uplo
        int,                // M
        int,            	 // N
		cl_float2,				//alpha
		cl_float2,				//beta
		ExtraTestSizes,     // to get more than ten parameters in gtest.
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
        params->seed = seed;
		params->side = side;
		params->uplo = uplo;
        params->M = M;
        params->N = N;
        params->lda = lda;
        params->ldb = ldb;
        params->ldc = ldc;
		params->offa = offa;
		params->offb = offb;
		params->offc = offc;
		params->alpha.re = (long)CREAL(alpha);
        params->alpha.imag = (long)CIMAG(alpha);
        params->beta.re = (long)CREAL(beta);
        params->beta.imag = (long)CIMAG(beta);


        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
		ExtraTestSizes extra;
		order = ::std::tr1::get<0>(GetParam());
        side = ::std::tr1::get<1>(GetParam());
        uplo = ::std::tr1::get<2>(GetParam());
        M = ::std::tr1::get<3>(GetParam());
        N = ::std::tr1::get<4>(GetParam());
		alpha = ::std::tr1::get<5>(GetParam());
        beta  = ::std::tr1::get<6>(GetParam());
		extra = ::std::tr1::get<7>(GetParam());

		offa = extra.offA;
        offb = extra.offBX;
        offc = extra.offCY;
		lda = extra.strideA.ld;
		ldb = extra.strideBX.ld;
		ldc = extra.strideCY.ld;

        numCommandQueues = ::std::tr1::get<8>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        if (base->useM()) {
            M = base->M();
        }
        if (base->useN()) {
            N = base->N();
        }

		if( side == clblasLeft )
		{
			lda = ::std::max(lda, M);
		}
		else
		{
			lda = ::std::max(lda, N);
		}


		switch (order) {
        case clblasRowMajor:
            ldb = ::std::max(ldb, N);
            ldc = ::std::max(ldc, N);
            break;
        case clblasColumnMajor:
            ldb = ::std::max(ldb, M);
            ldc = ::std::max(ldc, M);
            break;
        }

		printTestParams(order, side, uplo, M, N, 1, alpha, 1, beta, lda, ldb, ldc, offa, offb, offc);

        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
	clblasSide side;
	clblasUplo uplo;
    size_t M, N;
    size_t lda, ldb, ldc;
    size_t offa, offb, offc;
    unsigned int seed;
    cl_float2 alpha, beta;
    ::clMath::BlasBase *base;
    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SYMM_H_
