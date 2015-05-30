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


#ifndef GEMV_H_
#define GEMV_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <blas-math.h>

using namespace clMath;
using ::testing::TestWithParam;

class GEMV : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasTranspose, // transA
        int,                // M
        int,                // N
        ExtraTestSizes,
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));

        params->order = order;
        params->transA = transA;
        params->transB = transB;
        params->transC = transC;
        params->seed = seed;
        params->M = M;
        params->N = N;
        params->K = L;
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
        size_t lenX, lenY;
        bool seqX, seqY;

        order = ::std::tr1::get<0>(GetParam());
        transA = ::std::tr1::get<1>(GetParam());
        M = ::std::tr1::get<2>(GetParam());
        N = ::std::tr1::get<3>(GetParam());
        extra = ::std::tr1::get<4>(GetParam());
        offA = extra.offA;
        lda = extra.strideA.ld;
        incx = extra.strideBX.inc;
        incy = extra.strideCY.inc;
        numCommandQueues = ::std::tr1::get<5>(GetParam());

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
        if (base->useM()) {
            M = base->M();
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

        ldb = ldc = 0;
        offx = offy = 0;

        L = (M + N) / 2; //It doesn't matter, can be any value

        seqX = module(incx) == 1;
        seqY = module(incy) == 1;

        if (transA == clblasNoTrans) {
            lenX = N;
            lenY = M;
        }
        else {
            lenX = M;
            lenY = N;
        }

        rowsA = M;
        columnsA = N;

        switch (order) {
        case clblasRowMajor:

            lda = ::std::max(lda, columnsA);
            columnsA = lda;

            if (seqX) {
                //x is a middle row in row major matrix
                rowsB = L;
                columnsB = lenX;
                ldb = ::std::max(ldb, columnsB);
                transB = clblasTrans;
                offx = (rowsB / 2) * ldb;
            }
            else {
                //x is a middle column column in row major matrix
                rowsB = lenX;
                columnsB = L;
                ldb = ::std::max((size_t)module(incx), columnsB);
                transB = clblasNoTrans;
                offx = columnsB / 2;
            }
            columnsB = ldb;

            if (seqY) {
                //y is a middle row in row major matrix
                rowsC = L;
                columnsC = lenY;
                ldc = ::std::max(ldc, columnsC);
                transC = clblasTrans;
                offy = (rowsC / 2) * ldc;
            }
            else {
                //y is a middle column in row major matrix
                rowsC = lenY;
                columnsC = L;
                ldc = ::std::max((size_t)module(incy), columnsC);
                transC = clblasNoTrans;
                offy = columnsC / 2;
            }
            columnsC = ldc;
            break;
        case clblasColumnMajor:

            lda = ::std::max(lda, rowsA);
            rowsA = lda;

            if (seqX) {
                //x is a middle column in column major matrix
                rowsB = lenX;
                columnsB = L;
                ldb = ::std::max(ldb, rowsB);
                transB = clblasNoTrans;
                offx = (columnsB / 2) * ldb;
            }
            else {
                //x is a middle row in column major matrix
                rowsB = L;
                columnsB = lenX;
                ldb = ::std::max((size_t)module(incx), rowsB);
                transB = clblasTrans;
                offx = rowsB / 2;
            }
            rowsB = ldb;

            if (seqY) {
                //y is a middle column in column major matrix
                rowsC = lenY;
                columnsC = L;
                ldc = ::std::max(ldc, rowsC);
                transC = clblasNoTrans;
                offy = (columnsC / 2) * ldc;
            }
            else {
                //y is a middle row in column major matrix
                rowsC = L;
                columnsC = lenY;
                ldc = ::std::max((size_t)module(incy), rowsC);
                transC = clblasTrans;
                offy = rowsC / 2;
            }
            rowsC = ldc;
            break;
        }

        if (!seqX) {
            incx = incx > 0 ? (int)ldb : (int)(0-ldb);
        }
        if (!seqY) {
            incy = incy > 0 ? (int)ldc : (int)(0-ldc);
        }

        printTestParams(order, transA, M, N, useAlpha, base->alpha(), offA,
                        lda, incx, useBeta, base->beta(), incy);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasTranspose transA, transB, transC;
    size_t M, N, L;
    size_t lda, ldb, ldc;
    int incx, incy;
    size_t offA, offx, offy;
    unsigned int seed;

    bool useAlpha, useBeta;
    ComplexLong paramAlpha, paramBeta;

    size_t rowsA, rowsB, rowsC, columnsA, columnsB, columnsC;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageX;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // GEMV_H_
