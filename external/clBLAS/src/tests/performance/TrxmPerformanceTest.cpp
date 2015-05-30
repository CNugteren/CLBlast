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
 * Performance test case class implementation for
 * TRMM and TRSM routines
 */

#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <blas-random.h>

#ifdef PERF_TEST_WITH_ACML
#include <blas-internal.h>
#include <blas-wrapper.h>
#endif

#include "PerformanceTest.h"

using namespace std;
using namespace clMath;

namespace clMath {

template <typename ElemType> class TrxmPerformanceTest : public PerformanceTest
{
public:
    virtual ~TrxmPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        TrxmPerformanceTest<ElemType> *perfCase;
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        if (fn == FN_STRMM || fn == FN_DTRMM ||
            fn == FN_STRSM || fn == FN_DTRSM) {

            opFactor = 1;
        }
        else {
            opFactor = 4;
        }

        if ((fn == FN_DTRMM || fn == FN_ZTRMM ||
             fn == FN_DTRSM || fn == FN_ZTRSM) &&
            !base->isDevSupportDoublePrecision()) {

            std::cerr << ">> WARNING: The target device doesn't support native "
                         "double precision floating point arithmetic" <<
                         std::endl << ">> Test skipped" << std::endl;
            return;
        }

        perfCase = new TrxmPerformanceTest<ElemType>(fn, params);
        if (!perfCase->areResourcesSufficient(params)) {
            std::cerr << ">> RESOURCE CHECK: Skip due to unsufficient "
                         "resources" << std::endl;
        }
        else {
            ret = perfCase->run(opFactor);
        }

        delete perfCase;

        ASSERT_GE(ret, 0) << "Fatal error: can not allocate resources or "
                                     "perform an OpenCL request!" << endl;
        EXPECT_EQ(0, ret) << "The OpenCL version is slower in the case" << endl;
    }

private:
    TrxmPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType *A_;
    ElemType *B_;
    ElemType *backB_;
    cl_mem mobjA_;
    cl_mem mobjB_;
    ::clMath::BlasBase *base_;
    bool isTrsm_;

    static problem_size_t problemSize(TestParams *params);
};

} // namespace

template <typename ElemType>
TrxmPerformanceTest<ElemType>::TrxmPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn, problemSize(params)),
                params_(*params), mobjA_(NULL), mobjB_(NULL)
{
    A_ = new ElemType[params_.rowsA * params_.columnsA];
    B_ = new ElemType[params_.rowsB * params_.columnsB];
    backB_ = new ElemType[params_.rowsB * params_.columnsB];

    base_ = ::clMath::BlasBase::getInstance();
    isTrsm_ = (static_cast<int>(fn) >= FN_STRSM);
}

template <typename ElemType>
TrxmPerformanceTest<ElemType>::~TrxmPerformanceTest()
{
    delete[] A_;
    delete[] B_;
    delete[] backB_;

    clReleaseMemObject(mobjB_);
    clReleaseMemObject(mobjA_);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
TrxmPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize, maxMatrSize;
    bool ret = true;
    size_t m = params->M, n = params->N;
    size_t asize;
    clblasSide side = params->side;

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();
    asize = (side == clblasLeft) ? m : n;

    if (base->useImages()) {
        size_t iw;

        // overall 2 images 1/5 of gmemSize each and 2 memory objects
        maxMatrSize = 3 * gmemSize / 10;
        iw = base->scratchImageWidth() * sizeof(cl_float4) / sizeof(ElemType);

        if (isTrsm_) {
            size_t ih, nb;

            // check if matrix A is fitted to the image with 32x32 blocks
            ih = base->scratchImageHeight();
            nb = asize / 32 + (asize % 32 != 0);
            ret = ((asize * asize + nb * 32 * 32) / 2 < iw * ih);
        }
        else {
            ret = (std::max(n, asize) < iw);
        }
    }
    else {
        maxMatrSize = gmemSize / 2;
    }
    maxMatrSize = std::min(maxMatrSize, allocSize);

    if (ret) {
        ret = ((m * n * sizeof(ElemType) < maxMatrSize) &&
               (asize * asize * sizeof(ElemType) < maxMatrSize));
    }

    return ret;
}

template <typename ElemType> int
TrxmPerformanceTest<ElemType>::prepare(void)
{
    bool useAlpha = base_->useAlpha();

    if (useAlpha) {
        alpha_ = convertMultiplier<ElemType>(base_->alpha());
    }

    if (isTrsm_) {
        randomTrsmMatrices<ElemType>(params_.order, params_.side, params_.uplo,
            params_.diag, params_.M, params_.N, useAlpha,
            &alpha_, A_, params_.lda, B_, params_.ldb);
    }
    else {
        randomTrmmMatrices<ElemType>(params_.order, params_.side, params_.uplo,
            params_.diag, params_.M, params_.N, useAlpha,
            &alpha_, A_, params_.lda, B_, params_.ldb);
    }

    mobjA_ = base_->createEnqueueBuffer(A_, params_.rowsA * params_.columnsA *
                                        sizeof(ElemType),
                                        params_.offA * sizeof(ElemType),
                                        CL_MEM_READ_ONLY);
    if (mobjA_) {
        mobjB_ = base_->createEnqueueBuffer(backB_, params_.rowsB *
                                            params_.columnsB * sizeof(ElemType),
                                            params_.offBX * sizeof(ElemType),
                                            CL_MEM_READ_WRITE);
    }

    return (mobjB_) ? 0 : -1;
}

template <typename ElemType> nano_time_t
TrxmPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
    size_t lda, ldb;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    memcpy(B_, backB_, params_.rowsB * params_.columnsB *
           sizeof(ElemType));
    order = params_.order;
    lda = params_.lda;
    ldb = params_.ldb;

#ifdef PERF_TEST_WITH_ACML

    if (order == clblasRowMajor) {
        order = clblasColumnMajor;
        if (params_.side == clblasLeft) {
            lda = params_.M;
        }
        else {
            lda = params_.N;
        }
        ldb = params_.M;
    }

    time = getCurrentTime();
    if (isTrsm_) {
        clMath::blas::trsm(order, params_.side, params_.uplo,
                        params_.transA, params_.diag,
                        params_.M, params_.N,
                        alpha_, A_, lda, B_, ldb);
    }
    else {
        clMath::blas::trmm(order, params_.side, params_.uplo,
                        params_.transA, params_.diag,
                        params_.M, params_.N,
                        alpha_, A_, lda, B_, ldb);
    }
    time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}

template <typename ElemType> nano_time_t
TrxmPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue;

    queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjB_, CL_TRUE, 0,
                                  params_.rowsB * params_.columnsB *
                                  sizeof(ElemType), backB_, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "Matrix B buffer object enqueuing error, status = " <<
                status << endl;

        return NANOTIME_ERR;
    }

    status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
        cerr << "Wait on event failed, status = " <<
                status << endl;

        return NANOTIME_ERR;
    }

    event = NULL;

    if (isTrsm_) {
        status = (cl_int)clMath::clblas::trsm(params_.order, params_.side,
            params_.uplo, params_.transA, params_.diag, params_.M, params_.N,
            alpha_, mobjA_, params_.offA, params_.lda, mobjB_, params_.offBX,
            params_.ldb, 1, &queue, 0, NULL, &event);
    }
    else {
        status = (cl_int)clMath::clblas::trmm(params_.order, params_.side,
            params_.uplo, params_.transA, params_.diag, params_.M, params_.N,
            alpha_, mobjA_, params_.offA, params_.lda, mobjB_, params_.offBX,
            params_.ldb, 1, &queue, 0, NULL, &event);
    }

    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS TRXM function failed, status = " <<
                status << endl;

        return NANOTIME_ERR;
    }
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

    return time;
}

template <typename ElemType> problem_size_t
TrxmPerformanceTest<ElemType>::problemSize(TestParams *params)
{
    problem_size_t size;

    if (params->side == clblasRight) {
        size = (problem_size_t)params->N * params->N * params->M;
    }
    else {
        size = (problem_size_t)params->M * params->M * params->N;
    }

    return size;
}
