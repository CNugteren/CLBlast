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


/*
 * Symv performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <symv.h>
#include <blas-random.h>

#ifdef PERF_TEST_WITH_ACML
#include <blas-internal.h>
#include <blas-wrapper.h>
#endif

#include "PerformanceTest.h"

/*
 * NOTE: operation factor means overall number
 *       of multiply and add per each operation involving
 *       2 matrix elements
 */

using namespace std;
using namespace clMath;

#define CHECK_RESULT(ret)                                                   \
do {                                                                        \
    ASSERT_GE(ret, 0) << "Fatal error: can not allocate resources or "      \
                         "perform an OpenCL request!" << endl;              \
    EXPECT_EQ(0, ret) << "The OpenCL version is slower in the case" <<      \
                         endl;                                              \
} while (0)

namespace clMath {

template <typename ElemType> class SymvPerformanceTest : public PerformanceTest
{
public:
    virtual ~SymvPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        SymvPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor = (fn == FN_SSYMV) ? sizeof(cl_float) : sizeof(cl_double);

        if ((fn == FN_DSYMV) &&
            !base->isDevSupportDoublePrecision()) {

            std::cerr << ">> WARNING: The target device doesn't support native "
                         "double precision floating point arithmetic" <<
                         std::endl << ">> Test skipped" << std::endl;
            return;
        }

        if (!perfCase.areResourcesSufficient(params)) {
            std::cerr << ">> RESOURCE CHECK: Skip due to insufficient resources" <<
                        std::endl;
        }
        else {
            ret = perfCase.run(opFactor);
        }

        ASSERT_GE(ret, 0) << "Fatal error: can not allocate resources or "
                             "perform an OpenCL request!" << endl;
        EXPECT_EQ(0, ret) << "The OpenCL version is slower in the case" << endl;
    }

private:
    SymvPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType beta_;
    ElemType *A_;
    ElemType *B_;
    ElemType *C_;
    ElemType *backC_;
    cl_mem mobjA_;
    cl_mem mobjB_;
    cl_mem mobjC_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
SymvPerformanceTest<ElemType>::SymvPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn, (problem_size_t)params->N * params->N),
                        params_(*params), mobjA_(NULL), mobjB_(NULL), mobjC_(NULL)
{
    A_ = new ElemType[params_.rowsA * params_.columnsA];
    B_ = new ElemType[params_.rowsB * params_.columnsB];
    C_ = new ElemType[params_.rowsC * params_.columnsC];
    backC_ = new ElemType[params_.rowsC * params_.columnsC];

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
SymvPerformanceTest<ElemType>::~SymvPerformanceTest()
{
    if(A_ != NULL)
    {
        delete[] A_;
    }
    if(B_ != NULL)
    {
        delete[] B_;
    }
    if(C_ != NULL)
    {
        delete[] C_;
    }
    if(backC_ != NULL)
    {
        delete[] backC_;
    }

    if(mobjC_ != NULL)
    {
        clReleaseMemObject(mobjC_);
    }
    if(mobjB_ != NULL)
    {
        clReleaseMemObject(mobjB_);
    }
    if(mobjC_ != NULL)
    {
        clReleaseMemObject(mobjA_);
    }
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
SymvPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize, maxMatrSize;
    size_t n = params->N;

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();

    if((A_ == NULL) || (backC_ == NULL) || (C_ == NULL) || (B_ == NULL))
    {
        return 0;   // Not enough memory for host arrays
    }


    maxMatrSize = gmemSize / 3;

    maxMatrSize = std::min(maxMatrSize, allocSize);

    return (n * n * sizeof(ElemType) < maxMatrSize);
}

template <typename ElemType> int
SymvPerformanceTest<ElemType>::prepare(void)
{
    size_t lenX, lenY;
    bool useAlpha = base_->useAlpha();
    bool useBeta = base_->useBeta();

    if (useAlpha) {
        alpha_ = convertMultiplier<ElemType>(params_.alpha);
    }
    if (useBeta) {
        beta_ = convertMultiplier<ElemType>(params_.beta);
    }

    lenX = params_.N;
    lenY = params_.N;
    randomGemmxMatrices<ElemType>(params_.order, params_.transA, params_.transB,
                           params_.transC, lenY, params_.N, lenX, useAlpha,
                           &alpha_, A_, params_.lda, B_, params_.ldb, useBeta,
                           &beta_, C_, params_.ldc);

    mobjA_ = base_->createEnqueueBuffer(A_, params_.rowsA * params_.columnsA *
                                     sizeof(*A_), params_.offA * sizeof(*A_),
                                     CL_MEM_READ_ONLY);
    mobjB_ = base_->createEnqueueBuffer(B_, params_.rowsB * params_.columnsB *
                                     sizeof(*B_), 0, CL_MEM_READ_ONLY);
    mobjC_ = base_->createEnqueueBuffer(backC_, params_.rowsC * params_.columnsC *
                                     sizeof(*backC_), 0, CL_MEM_READ_WRITE);

    return ((mobjA_ != NULL) && (mobjB_ != NULL) && (mobjC_ != NULL)) ? 0 : -1;
}

template <typename ElemType> nano_time_t
SymvPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
    size_t lda;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    memcpy(C_, backC_, params_.rowsC * params_.columnsC * sizeof(ElemType));
    order = params_.order;
    lda = params_.lda;

#ifdef PERF_TEST_WITH_ACML

// #warning "SYMV performance test not implemented"
    time = NANOTIME_MAX;
    order = order;
    lda = lda;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
SymvPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjC_, CL_TRUE, 0,
                                  params_.rowsC * params_.columnsC *
                                  sizeof(ElemType), backC_, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "Vector Y buffer object enqueuing error, status = " <<
                 status << endl;

        return NANOTIME_ERR;
    }

    status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
        cout << "Wait on event failed, status = " <<
                status << endl;

        return NANOTIME_ERR;
    }

    event = NULL;

//#define TIMING
#ifdef TIMING
    clFinish( queue);

    time = getCurrentTime();
    int iter = 20;
    for ( int i = 1; i <= iter; i++)
    {
#endif
    status = (cl_int)clMath::clblas::symv(params_.order,
        params_.uplo, params_.N, alpha_, mobjA_, params_.offA, params_.lda,
        mobjB_, params_.offBX, params_.incx,
        beta_, mobjC_, params_.offCY, params_.incy,
        1, &queue, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS SYMV function failed, status = " <<
                status << endl;

        return NANOTIME_ERR;
    }

#ifdef TIMING
    } // iter loop
    clFinish( queue);
    time = getCurrentTime() - time;
    time /= iter;
#else
    status = flushAll(1, &queue);
    if (status != CL_SUCCESS) {
        cerr << "clFlush() failed, status = " << status << endl;
        return NANOTIME_ERR;
    }

    time = getCurrentTime();
    status = waitForSuccessfulFinish(1, &queue, &event);
    if (status == CL_SUCCESS) {
        time = getCurrentTime() - time;
    }
    else {
        cerr << "Waiting for completion of commands to the queue failed, "
                "status = " << status << endl;
        time = NANOTIME_ERR;
    }
#endif
    return time;
}

} // namespace clMath

// ssymv performance test
TEST_P(SYMV, ssymv)
{
    TestParams params;

    getParams(&params);
    SymvPerformanceTest<float>::runInstance(FN_SSYMV, &params);
}

// dsymv performance test case
TEST_P(SYMV, dsymv)
{
    TestParams params;

    getParams(&params);
    SymvPerformanceTest<double>::runInstance(FN_DSYMV, &params);
}
