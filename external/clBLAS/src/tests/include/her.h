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



#if !defined(HER_PACKED)
    #ifndef HER_H
        #define HER_H
    #else
        #define DUPLICIT
    #endif
#endif

#ifndef DUPLICIT

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-random.h>
#include <blas-math.h>

using ::testing::TestWithParam;

#ifndef HER_PACKED
class HER : public TestWithParam<
#else
class HPR : public TestWithParam<
#endif

    ::std::tr1::tuple<
        clblasOrder,    // order
		clblasUplo,     // uplo
        int,                // N
		double,             //alpha
        int,                // lda
        int,                //incx
        int,                // offx
		int,                // offa			//FIX_ME.. gtest not allowing to add more parameters
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
        params->uplo = uplo;
        params->N = N;
        params->alpha.re = (long)alpha;
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
        N = ::std::tr1::get<2>(GetParam());
        alpha = ::std::tr1::get<3>(GetParam());
        lda = ::std::tr1::get<4>(GetParam());
        incx = ::std::tr1::get<5>(GetParam());
		offa = ::std::tr1::get<6>(GetParam());
		offx = ::std::tr1::get<7>(GetParam());
        numCommandQueues = ::std::tr1::get<8>(GetParam());

        #ifndef HER_PACKED
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

	printTestParams(order, uplo, N, alpha,
			offx, incx, offa, lda );

        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    size_t  N;
    size_t lda;
    int incx;
    size_t offa, offx;
    unsigned int seed;
    double  alpha;
    ComplexLong paramAlpha;
    size_t rowsA, columnsA;
    ::clMath::BlasBase *base;
    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#ifndef RANDOM_HER
#define RANDOM_HER

template <typename T>
static void
randomHerMatrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    T *alpha,
    T *A,
    size_t lda,
    T *X,
	int incx
    )
{
    size_t i, j;
	size_t lengthX;
    cl_double bound, max;

	// bound is calculated by solving the equation (alpha*x^2 + x - UPPER_BOUND) < 0
	bound = UPPER_BOUND<T>();
	if(module(CREAL(*alpha)) > (sqrt(bound) / (2.0)))
		*alpha = random<T>((sqrt(bound) / (2.0)));

	max = module(CREAL(*alpha));
	bound = bound / max / 2.0;
    bound = sqrt( ((((1.0) / max) / (4.0)) / max) + bound) - ((1.0) / ((2.0) * max));

    if( lda )
    {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            setElement<T>(order, clblasNoTrans, i, j, A, lda, random<T>(bound));
        }
    }
    } else {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, random<T>(bound));
            }
        }
    }

	lengthX = 1 + ((N - 1) * abs(incx));
    if (X != NULL) {
        for (i = 0; i < lengthX; i++) {
			X[i] = random<T>(bound);
        }
    }
}
#endif // RANDOM_HER

#endif  // HER_H_
