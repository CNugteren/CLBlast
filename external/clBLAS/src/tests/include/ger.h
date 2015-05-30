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


#ifndef GER_H_
#define GER_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>

using ::testing::TestWithParam;

class GER : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,    // order
        int,                // M
        int,                // N
        int,                // lda
        int,                //incx
        int,                //incy
        int,                // offx
	int,		    // offy
	int,                // offa			//FIX_ME.. gtest not allowing to add more parameters
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        params->order = order;
        params->M = M;
        params->N = N;
        params->lda = lda;
        params->incx = incx;
        params->incy = incy;
	params->offa = offa;
	params->offBX = offx;
	params->offCY = offy;
        params->rowsA = rowsA;
        params->alpha = paramAlpha;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
	order = ::std::tr1::get<0>(GetParam());
        M = ::std::tr1::get<1>(GetParam());
        N = ::std::tr1::get<2>(GetParam());
        lda = ::std::tr1::get<3>(GetParam());
        incx = ::std::tr1::get<4>(GetParam());
	incy = ::std::tr1::get<5>(GetParam());
	offa = ::std::tr1::get<6>(GetParam());
	offx = ::std::tr1::get<7>(GetParam());
	offy = ::std::tr1::get<8>(GetParam());
        numCommandQueues = ::std::tr1::get<9>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();
	ComplexLong fAlpha;
	fAlpha.re = 3, fAlpha.imag = 4;
	base->setAlpha(fAlpha);

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        useAlpha = base->useAlpha();
        if (useAlpha != 0) {
            paramAlpha = base->alpha();
        }
        if (base->useM()) {
            M = base->M();
        }
        if (base->useN()) {
            N = base->N();
        }

        rowsA = M;
        columnsA = N;

	switch (order) {
        case clblasRowMajor:
            lda = ::std::max(lda, columnsA);
            break;
        case clblasColumnMajor:
            lda = ::std::max(lda, rowsA);
            break;
        }


	printTestParams(order, M, N, useAlpha,
                   	base->alpha(),
			lda, incx, incy, offa, offx, offy);

        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    size_t M, N;
    size_t lda;
    int incx, incy;
    size_t offa, offx, offy;
    unsigned int seed;
    bool useAlpha;
    ComplexLong paramAlpha;
    size_t rowsA, columnsA;
    ::clMath::BlasBase *base;
    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // GER_H_
