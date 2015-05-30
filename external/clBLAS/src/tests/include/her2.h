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


#if !defined(HER2_PACKED)
    #ifndef HER2_H
        #define HER2_H
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

#ifndef HER2_PACKED
class HER2 : public TestWithParam<
#else
class HPR2 : public TestWithParam<
#endif

    ::std::tr1::tuple<
    	clblasOrder,     // order
		clblasUplo,		// uplo
        int,                // N
		cl_float2,				//alpha
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
		params->alpha.re = (long)(CREAL(alpha)); // This will cast alpha to long. So the real value that is
		params->alpha.imag = (long)(CIMAG(alpha));								// passed is not the same as what is set in the test case
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

		#ifndef HER2_PACKED
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

		printTestParams(order, uplo, N, 1, alpha, offx, incx, offy, incy, offa, lda);

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
	cl_float2 alpha;
    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX, imageY;
    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#ifndef RANDOM_HER2
#define RANDOM_HER2

template <typename T>
static void
randomHer2Matrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    T *alpha,
    T *A,
    size_t lda,
    T *X,
	int incx,
	T *Y,
	int incy
    )
{
    size_t i, j;
	size_t lengthX;
    size_t lengthY;
	cl_double bound, max;

	// bound is calculated by solving the equation (2*alpha*x^2 + x - UPPER_BOUND) < 0
	bound = UPPER_BOUND<T>();
	max = module( ::std::max( alpha->s[0], alpha->s[1] ) );

	if(max > (sqrt(bound) / (4.0)))
		*alpha = random<T>((sqrt(bound) / (4.0)));
	max = module( ::std::max( alpha->s[0], alpha->s[1] ) );

	bound = bound / ( 2 * max);
    bound = sqrt( ((((1.0) / max) / (16.0)) / max) + bound) - ((1.0) / ((4.0) * max));

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
	lengthY = 1 + (N - 1) * abs(incy);
	if (Y != NULL) {
		for (i = 0; i < lengthY; i++) {
			Y[i] = random<T>(bound);
		}
	}
}
#endif //RANDOM_HER2

#endif  //HER2_H_
