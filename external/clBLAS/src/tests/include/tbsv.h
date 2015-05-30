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


#ifndef TBSV_H_
#define TBSV_H_

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

class TBSV : public TestWithParam<
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
randomTbsvMatrices(
    clblasOrder order,
	clblasUplo uplo,
    clblasDiag diag,
    size_t N,
    size_t K,
    T *A,
    size_t lda,
    T *X,
    int incx)
{
    size_t i, j;
    T min, max, x, y;
    cl_double modMin, modMax, sum, maxDiag;

    min = ZERO<T>();
    max = ZERO<T>();
    incx = abs(incx);
    maxDiag = 1.0;

    cl_double bound;
    bound = (UPPER_BOUND<T>()/(N));

    switch (diag) {
    case clblasUnit:
        for (i = 0; i < N; i++) {
            // must not be accessed
            setElementBanded<T>(order, uplo, i, i, K, A, lda, FNAN<T>());
        }
        break;
    case clblasNonUnit:
        /* Do not allow zeros on A's main diagonal and get a big number which is atleast greater than N/4*/
        maxDiag = ((N/4) > bound) ? (bound/4) : (N/4);
        maxDiag = (1 > (maxDiag)) ? 1 : maxDiag;
        do {
            max = randomTrsv<T>(bound);
        } while ((module(max) < (maxDiag)));
        modMax = module(max);
        min = max / 100;
        modMin = module(min);
        setElementBanded<T>(order, uplo, 0, 0, K, A, lda, max);
        //printf("Diagonals %d ", max);
        for (i = 1; i < N; i++) {
            x = randomTrsv<T>(modMin, modMax);
            if (module(x) < 1) {
                x = max;
            }
            //printf("%d ", x);
            /*if(module(x) < 1)
            {
                printf("WARNING: Diagonal less than one\n");
            }*/
            setElementBanded<T>(order, uplo, i, i, K, A, lda, x);
        }
       // printf("\n");
        break;
    }

    /* Generate a_{ij} for all j <> i. */
    for (i = 0; i < N; i++) {

        if (diag == clblasUnit) {
            sum = module(ONE<T>());
        }
        else {
            T temp;
            temp = getElementBanded<T>(order, uplo, i, i, K, A, lda);
            sum = module(temp);
        }

        for (j = 0; j < N; j++) {
            if ((j == i) || (module((int)(i-j)) > ((int)K)) )    // Diagonal and out-of-band elemnts
            {
                continue;
            }

            if (((uplo == clblasUpper) && (j > i)) ||
                ((uplo == clblasLower) && (j < i)))
            {
                x = randomTrsv<T>(sum/(K + 1)); //Only K + 1 accumulation not N.
                setElementBanded<T>(order, uplo, i, j, K, A, lda, x);
            }
        }
    }

    /* Generate matrix X. */
    sum = TRSM_LIMIT_B<T>();
    for (i = 0; i < N; i++) {
        if(diag == clblasNonUnit)
        {
            sum = module(getElementBanded<T>(order, uplo, i, i, K, A, lda));
        }
        else
        {
            sum = module(ONE<T>());
        }
        y = randomTrsv<T>(sum/(K+1));
        setElement<T>(clblasColumnMajor, clblasNoTrans, (i * abs(incx)), 0, X, (1 + (N-1)*abs(incx)), y);
        if (i == 0) {
            min = y;
        }
        else if (module(y) < module(min)) {
            min = y;
        }
    }
}

#endif  // TBSV_H_
