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


#ifndef SYMV_H_
#define SYMV_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <common.h>
#include <math.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class SYMV : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo,      // uplo
        int,                // N
        ExtraTestSizes,
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
        params->lda = lda;
        params->ldb = ldb;
        params->ldc = ldc;
        params->rowsA = rowsA;
        params->rowsB = rowsB;
        params->rowsC = rowsC;
        params->columnsA = columnsA;
        params->columnsB = columnsB;
        params->columnsC = columnsC;
        params->incx = incx;
        params->incy = incy;
        params->offA = offsetA;
        params->offBX = offsetx;
        params->offCY = offsety;
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
        extra = ::std::tr1::get<3>(GetParam());
        offsetA = extra.offA;
        lda = extra.strideA.ld;
        incx = extra.strideBX.inc;
        incy = extra.strideCY.inc;
        numCommandQueues = ::std::tr1::get<4>(GetParam());

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

        lda = ::std::max(lda, N);

        if (incx == 1 || incx == -1) {
            /* X is row vector for row major matrix B
             * or column vector for column major matrix B */
            ldb = lda;
            offsetx = (N / 2) * ldb;
        }
        else {
            /* X is column vector for row major matrix B
             * or row vector for column major matrix B */
            ldb = ::std::max(N, (size_t)module(incx));
            offsetx = N / 2;
            incx = incx > 0 ? (int)ldb : (int)(0-ldb);
        }

        if (incy == 1 || incy == -1) {
            /* Y is row vector in row major matrix C
             * or column vector in column major matrix C */
            ldc = lda;
            offsety = (N / 2) * ldc;
        }
        else {
            /* Y is column vector in matrix C
             * or row vector in column major matrix C */
            ldc = ::std::max(N, (size_t)module(incy));
            offsety = N / 2;
            incy = incy > 0 ? (int)ldc : (int)(0-ldc);
        }

        switch (order) {
        case clblasRowMajor:
            columnsA = lda;
            columnsB = ldb;
            columnsC = ldc;
            rowsA = N;
            rowsB = N;
            rowsC = N;
            break;
        case clblasColumnMajor:
            rowsA = lda;
            rowsB = ldb;
            rowsC = ldc;
            columnsA = N;
            columnsB = N;
            columnsC = N;
            break;
        }

        printTestParams(order, uplo, N, useAlpha, base->alpha(), offsetA, lda,
                        incx, useBeta, base->beta(), incy);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    size_t N;
    size_t lda, ldb, ldc;
    size_t offsetA, offsetx, offsety;
    int incx, incy;
    unsigned int seed;

    bool useAlpha, useBeta;
    ComplexLong paramAlpha, paramBeta;

    size_t rowsA, columnsA, rowsB, columnsB, rowsC, columnsC;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SYMV_H_
