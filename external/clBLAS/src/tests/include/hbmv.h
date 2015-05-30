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


#ifndef HBMV_H_
#define HBMV_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <blas-random.h>
#include <blas-math.h>
#include <tbmv.h>

using namespace clMath;
using ::testing::TestWithParam;

class HBMV : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo,      // uplo
        int,                // N
        int,                // K
        ExtraTestSizes,
        ComplexLong,		// Alpha
		ComplexLong, 		// Beta
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));

        params->order = order;
        params->uplo = uplo;
        params->seed = seed;
        params->N = N;
        params->K = KLU;
        params->lda = lda;
        params->incx = incx;
        params->incy = incy;
        params->offA = offA;
        params->offa = offA;
        params->offBX = offx;
        params->offCY = offy;
        params->alpha = paramAlpha;
        params->beta = paramBeta;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        ExtraTestSizes extra;

        order = ::std::tr1::get<0>(GetParam());
        uplo = ::std::tr1::get<1>(GetParam());
        N = ::std::tr1::get<2>(GetParam());
        KLU = ::std::tr1::get<3>(GetParam());
        extra = ::std::tr1::get<4>(GetParam());
        offA = extra.offA;
        offx = extra.offBX;
        offy = extra.offCY;
        lda = extra.strideA.ld;
        incx = extra.strideBX.inc;
        incy = extra.strideCY.inc;
        paramAlpha = ::std::tr1::get<5>(GetParam());
		paramBeta  = ::std::tr1::get<6>(GetParam());
        numCommandQueues = ::std::tr1::get<7>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        KLU = KLU % N;
        lda = ::std::max(lda, (KLU+1));

        printTestParams(order, uplo, N, KLU, paramAlpha, offA,
                            lda, offx, incx, paramBeta, offy, incy);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    size_t  N, KLU;
    size_t lda;
    int incx, incy;
    size_t offA, offx, offy;
    unsigned int seed;

    ComplexLong paramAlpha, paramBeta;
    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};



#endif  // HBMV_H_
