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

#ifndef SPMV_H_
#define SPMV_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <blas-random.h>
#include <ExtraTestSizes.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class SPMV : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo, 		// uplo
        int,                // N
        ComplexLong,		// Alpha
		ComplexLong, 		// Beta
		size_t,				// offA
		size_t,				// offx
		size_t, 			// offy
		ExtraTestSizes,
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
        params->uplo = uplo;
        params->seed = seed;
        params->N = N;
        params->lda = lda;
        params->incx = incx;
        params->incy = incy;
        params->offA = offA;
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

        order 	   = ::std::tr1::get<0>(GetParam());
        uplo 	   = ::std::tr1::get<1>(GetParam());
        N 		   = ::std::tr1::get<2>(GetParam());
        paramAlpha = ::std::tr1::get<3>(GetParam());
		paramBeta  = ::std::tr1::get<4>(GetParam());
		offA	   = ::std::tr1::get<5>(GetParam());
		offx	   = ::std::tr1::get<6>(GetParam());
		offy	   = ::std::tr1::get<7>(GetParam());
		extra 	   = ::std::tr1::get<8>(GetParam());
        lda 	   = 0;
        incx 	   = extra.strideBX.inc;
        incy       = extra.strideCY.inc;

		numCommandQueues = ::std::tr1::get<9>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

		useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        useAlpha = base->useAlpha();
        if (useAlpha != 0) {
            paramAlpha = base->alpha();
        }
        useBeta = base->useBeta();
        if (useBeta != 0) {
            paramBeta = base->beta();
        }
        if (base->useN()) {
            N = base->N();
        }
        if (base->useIncX()) {
            incx = base->incX();
        }
        if (base->useIncY()) {
            incy = base->incY();
        }

        printTestParams(order, uplo, N, paramAlpha, offA,
                        0, offx, incx, paramBeta, offy, incy);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    size_t N;
    size_t lda;
    int incx, incy;
    size_t offA, offx, offy;
    unsigned int seed;

    bool useAlpha, useBeta;
    ComplexLong paramAlpha, paramBeta;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX, imageY;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

template <typename T>
static void
randomSpmvMatrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    T *X,
    int incx,
	bool useBeta,
	T *beta,
    T *Y,
    int incy
    )
{
    size_t i, j;
    size_t lengthX;
    size_t lengthY;
    cl_double bound;
	cl_double fAlpha, fBeta;

    if (!useAlpha) {
        *alpha = random<T>(100);
        if (module(*alpha) == 0.0) {
            *alpha = 1.0;
        }
    }

	if (!useBeta) {
        *beta = random<T>(100);
        if (module(*beta) == 0.0) {
            *beta = 1.0;
        }
    }

    bound = UPPER_BOUND<T>();

    if(module(*alpha) > bound)
        *alpha = random<T>((sqrt(bound) / ((2.0) * N)));
	if (module(*alpha) == 0.0) {
            *alpha = 1.0;
    }

	if(module(*beta) > bound)
        *beta = random<T>((sqrt(bound)));
	if (module(*beta) == 0.0) {
            *beta = 1.0;
    }

	fAlpha = module(*alpha);
	fBeta  = module(*beta);

    bound = bound / (fAlpha * N);

    bound = sqrt( ((((((fBeta * fBeta)) / fAlpha) / (4.0)) / fAlpha) / (N * N)) + bound) - ((fBeta) / ((2.0) * (fAlpha) * N));


    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, random<T>(bound));
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

#endif  // SPMV_H_
