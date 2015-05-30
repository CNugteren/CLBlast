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


#ifndef TRSM_H_
#define TRSM_H_

#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>

using namespace clMath;
using ::testing::TestWithParam;

class TRSM : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,         // order
        clblasSide,          // side
        clblasUplo,          // uplo
        clblasTranspose,     // transA
        clblasDiag,          // diag
        int,                    // M
        int,                    // N
        ExtraTestSizes,
        int                     // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));

        params->order = order;
        params->side = side;
        params->uplo = uplo;
        params->transA = transA;
        params->diag = diag;
        params->seed = seed;
        params->M = M;
        params->N = N;
        params->offA = offA;
        params->offBX = offB;
        params->lda = lda;
        params->ldb = ldb;
        params->rowsA = rowsA;
        params->columnsA = columnsA;
        params->rowsB = rowsB;
        params->columnsB = columnsB;
        params->alpha = paramAlpha;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        ExtraTestSizes extra;

        order = ::std::tr1::get<0>(GetParam());
        side = ::std::tr1::get<1>(GetParam());
        uplo = ::std::tr1::get<2>(GetParam());
        transA = ::std::tr1::get<3>(GetParam());
        diag = ::std::tr1::get<4>(GetParam());
        M = ::std::tr1::get<5>(GetParam());
        N = ::std::tr1::get<6>(GetParam());
        extra = ::std::tr1::get<7>(GetParam());
        offA = extra.offA;
        offB = extra.offBX;
        lda = extra.strideA.ld;
        ldb = extra.strideBX.ld;
        numCommandQueues = ::std::tr1::get<8>(GetParam());

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
        if (base->useM()) {
            M = base->M();
        }
        if (base->useN()) {
            N = base->N();
        }

        switch (side) {
        case clblasLeft:
            rowsA = M;
            columnsA = M;
            break;
        case clblasRight:
            rowsA = N;
            columnsA = N;
            break;
        }
        rowsB = M;
        columnsB = N;

        switch (order) {
        case clblasRowMajor:
            lda = ::std::max(lda, columnsA);
            columnsA = lda;
            ldb = ::std::max(ldb, columnsB);
            columnsB = ldb;
            break;
        case clblasColumnMajor:
            lda = ::std::max(lda, rowsA);
            rowsA = lda;
            ldb = ::std::max(ldb, rowsB);
            rowsB = ldb;
            break;
        }

        printTestParams(order, side, uplo, transA, diag, M, N, useAlpha,
                        base->alpha(), offA, lda, offB, ldb);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasSide side;
    clblasUplo uplo;
    clblasTranspose transA;
    clblasDiag diag;
    size_t M, N;
    size_t offA, offB;
    size_t lda, ldb;
    unsigned int seed;

    bool useAlpha;
    ComplexLong paramAlpha;

    size_t rowsA, columnsA;
    size_t rowsB, columnsB;

    ::clMath::BlasBase *base;
    cl_ulong imageA;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // TRSM_H_
