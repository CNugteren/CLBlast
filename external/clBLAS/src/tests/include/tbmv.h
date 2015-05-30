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


#ifndef TBMV_H_
#define TBMV_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <blas-random.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class TBMV : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo,      // uplo
        clblasTranspose, // transA
        clblasDiag,      // diag
        int,                // N
        int,                // KL or KU
        ExtraTestSizes,
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));

        params->order = order;
        params->uplo = uplo;
        params->transA = transA;
        params->diag = diag;
        params->seed = seed;
        params->N = N;
        params->K = KLU;
        params->lda = lda;
        params->incx = incx;
        params->offA = offA;
        params->offa = offA;
        params->offBX = offx;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        ExtraTestSizes extra;

        order = ::std::tr1::get<0>(GetParam());
        uplo = ::std::tr1::get<1>(GetParam());
        transA = ::std::tr1::get<2>(GetParam());
        diag = ::std::tr1::get<3>(GetParam());
        N = ::std::tr1::get<4>(GetParam());
        KLU = ::std::tr1::get<5>(GetParam());
        extra = ::std::tr1::get<6>(GetParam());
        offA = extra.offA;
        offx = extra.offBX;
        lda = extra.strideA.ld;
        incx = extra.strideBX.inc;
        numCommandQueues = ::std::tr1::get<7>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        KLU = KLU % N;
        lda = ::std::max(lda, (KLU+1));

        printTestParams(order, uplo, transA, diag, N, KLU, offA,
                            lda, offx, incx, 0, 1);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasTranspose transA;
    clblasUplo uplo;
    clblasDiag diag;
    size_t N, KLU;
    size_t lda;
    int incx;
    size_t offA, offx;
    unsigned int seed;

    ::clMath::BlasBase *base;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

template <typename T>
static void
randomTbmvMatrices(
    size_t N,
    T *A,
    size_t lda,
    T *X,
	int incx
    )
{
    size_t i;
	size_t lenX, lenA;
    cl_double bound;

	// bound is calculated by solving the equation (x^2 + x - UPPER_BOUND) < 0
	bound = UPPER_BOUND<T>();
	bound = sqrt( bound / N );           // (N * bound^2 - UPPER_BOUND) < 0

    lenA = (N) * lda;
    for (i = 0; i < lenA; i++) {
        A[i] = random<T>(bound);
    }

   	lenX = 1 + ((N - 1) * abs(incx));
    if (X != NULL) {
        for (i = 0; i < lenX; i++) {
			X[i] = random<T>(bound);
        }
    }
}

#endif  // TBMV_H_
