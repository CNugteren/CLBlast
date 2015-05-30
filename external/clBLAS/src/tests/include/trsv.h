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

#if !defined(TRSV_PACKED_)
    #ifndef TRSV_H_
        #define TRSV_H_
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

#ifndef TRSV_PACKED_
class TRSV : public TestWithParam<
#else
class TPSV : public TestWithParam<
#endif
    ::std::tr1::tuple<
        clblasOrder,     // order
		clblasUplo,		// uplo
        clblasTranspose, // transA
		clblasDiag,		// diag
        int,                // N
        int,                // lda, 0 - undefined
        int,                // incx, should be greater than 0
		int,				//offa
		int,				//offx
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
		params->uplo = uplo;
        params->transA = transA;
		params->diag = diag;
        params->seed = seed;
        params->N = N;
        params->lda = lda;
        params->incx = incx;
		params->offa = offa;
		params->offBX = offx;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        order = ::std::tr1::get<0>(GetParam());
        uplo = ::std::tr1::get<1>(GetParam());
		transA = ::std::tr1::get<2>(GetParam());
		diag = ::std::tr1::get<3>(GetParam());
        N = ::std::tr1::get<4>(GetParam());
        lda = ::std::tr1::get<5>(GetParam());
        incx = ::std::tr1::get<6>(GetParam());
		offa = ::std::tr1::get<7>(GetParam());
		offx = ::std::tr1::get<8>(GetParam());
        numCommandQueues = ::std::tr1::get<9>(GetParam());


        #ifndef TRSV_PACKED_
		lda = ::std::max( lda, N );
        #else
        lda = 0;
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

	    printTestParams(order, uplo, transA, diag, N, lda, incx, offa, offx);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    clblasTranspose transA;
    clblasDiag diag;
    size_t N;
    size_t lda;
    int incx;
    size_t offx, offa;
    unsigned int seed;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // DUPLICIT
