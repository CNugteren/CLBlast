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


#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <herk.h>
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

template <typename ElemType> class HerkPerformanceTest : public PerformanceTest
{
public:
    virtual ~HerkPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        HerkPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor = 8;

        if (( fn == FN_ZHERK) &&
            !base->isDevSupportDoublePrecision()) {

            std::cerr << ">> WARNING: The target device doesn't support native "
                         "double precision floating point arithmetic" <<
                         std::endl << ">> Test skipped" << std::endl;
            return;
        }

        if (!perfCase.areResourcesSufficient(params)) {
            std::cerr << ">> RESOURCE CHECK: Skip due to unsufficient resources" <<
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
    HerkPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType beta_;
    ElemType *A_;
    ElemType *C_;
    ElemType *backC_;
    cl_mem mobjA_;
    cl_mem mobjC_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
HerkPerformanceTest<ElemType>::HerkPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn, (problem_size_t)((params->N * params->N
                                            * params->K) / 2) ),
                        params_(*params), mobjA_(NULL), mobjC_(NULL)
{
    A_ = new ElemType[params_.rowsA * params_.columnsA];
    C_ = new ElemType[params_.rowsC * params_.columnsC];
    backC_ = new ElemType[params_.rowsC * params_.columnsC];

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
HerkPerformanceTest<ElemType>::~HerkPerformanceTest()
{
    if(A_!=NULL)
    {
        delete[] A_;
    }
    if(C_!=NULL)
    {
        delete[] C_;
    }
    if(backC_!=NULL)
    {
        delete[] backC_;
    }
    if(mobjC_!=NULL)
    {
        clReleaseMemObject(mobjC_);
    }
    if(mobjA_!=NULL)
    {
        clReleaseMemObject(mobjA_);
    }
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
HerkPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize, maxMatrSize;
    size_t n = params->N, k = params->K;


    if((A_ == NULL) || (backC_ == NULL) || (C_ == NULL))
    {
        return 0;
    }

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();

    maxMatrSize = gmemSize / 2;
    maxMatrSize = std::min(maxMatrSize, allocSize);
    return ((n * k * sizeof(ElemType)) + (n * n * sizeof(ElemType)) < maxMatrSize);

   // bool suff = ( sizeof(ElemType)*n*params->lda < allocSize ) && ((1 + (n-1)*abs(params->incx))*sizeof(ElemType) < allocSize); //for individual allocations
    //suff = suff && ((( n*params->lda + (1 + (n-1)*abs(params->incx)) + (1 + (n-1)*abs(params->incy)))*sizeof(ElemType)) < gmemSize) ; //for total global allocations

    //return suff ;

}

template <typename ElemType> int
HerkPerformanceTest<ElemType>::prepare(void)
{
    alpha_ = convertMultiplier<ElemType>(params_.alpha);
    beta_ = convertMultiplier<ElemType>(params_.beta);

    randomGemmMatrices<ElemType>(params_.order, params_.transA, clblasNoTrans,
        params_.N, params_.N, params_.K, true, &alpha_, A_, params_.lda,
        NULL, 0, true, &beta_, C_, params_.ldc);


    mobjA_ = base_->createEnqueueBuffer(A_, params_.rowsA * params_.columnsA *
                                        sizeof(ElemType),
                                        params_.offA * sizeof(ElemType),
                                        CL_MEM_READ_ONLY);
    if (mobjA_) {
        mobjC_ = base_->createEnqueueBuffer(backC_, params_.rowsC * params_.columnsC *
                                            sizeof(ElemType),
                                            params_.offCY * sizeof(ElemType),
                                            CL_MEM_READ_WRITE);
    }

    return (mobjC_) ? 0 : -1;
}

template <typename ElemType> nano_time_t
HerkPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
    clblasUplo fUplo;
    clblasTranspose fTransA;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    memcpy(C_, backC_, params_.rowsC * params_.columnsC * sizeof(ElemType));
    order = params_.order;
    fUplo = params_.uplo;
    fTransA = params_.transA;


#ifdef PERF_TEST_WITH_ACML
    fTransA = params_.transA;
    fUplo = params_.uplo;

    if (order != clblasColumnMajor)
    {
        fTransA = (params_.transA == clblasNoTrans) ? clblasConjTrans : clblasNoTrans;
        fUplo   = (params_.uplo == clblasUpper) ? clblasLower : clblasUpper;
    }

    time = getCurrentTime();
    clMath::blas::herk(clblasColumnMajor, fUplo, fTransA, params_.N, params_.K, CREAL(alpha_),
                     A_, params_.lda,CREAL( beta_), C_, params_.ldc);
    time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
HerkPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjC_, CL_TRUE, 0,
                                  params_.rowsC * params_.columnsC *
                                  sizeof(ElemType), backC_, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "Matrix C buffer object enqueuing error, status = " <<
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

    #define TIMING
    #ifdef TIMING
    clFinish( queue);
    time = getCurrentTime();

    int iter = 20;
    for ( int i = 1; i <= iter; i++)
    {
    #endif

    status = (cl_int)clMath::clblas::herk(params_.order,
        params_.uplo, params_.transA, params_.N, params_.K, CREAL(alpha_),
        mobjA_, params_.offA, params_.lda, CREAL(beta_), mobjC_, params_.offCY,
        params_.ldc, 1, &queue, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS HERK function failed, status = " <<
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

TEST_P(HERK, cherk)
{
    TestParams params;

    getParams(&params);
    HerkPerformanceTest<FloatComplex>::runInstance(FN_CHERK, &params);
}

TEST_P(HERK, zherk)
{
    TestParams params;

    getParams(&params);
    HerkPerformanceTest<DoubleComplex>::runInstance(FN_ZHERK, &params);
}
