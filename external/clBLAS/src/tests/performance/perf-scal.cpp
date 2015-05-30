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
 * SCAL performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <scal.h>
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

template <typename ElemType> class ScalPerformanceTest : public PerformanceTest
{
public:
    virtual ~ScalPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        ScalPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

        if (((fn == FN_DSCAL) || (fn == FN_ZSCAL) || (fn == FN_ZDSCAL)) &&
            !base->isDevSupportDoublePrecision()) {

            std::cerr << ">> WARNING: The target device doesn't support native "
                         "double precision floating point arithmetic" <<
                         std::endl << ">> Test skipped" << std::endl;
            return;
        }

        if (!perfCase.areResourcesSufficient(params)) {
            std::cerr << ">> RESOURCE CHECK: Skip due to unsufficient resources" <<
                        std::endl;
			return;
        }
        else {
            ret = perfCase.run(opFactor);
        }

        ASSERT_GE(ret, 0) << "Fatal error: can not allocate resources or "
                             "perform an OpenCL request!" << endl;
        EXPECT_EQ(0, ret) << "The OpenCL version is slower in the case" << endl;
    }

private:
    ScalPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType *X_;
    ElemType *backX_;
    cl_mem mobjX_;
    size_t  lengthX;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
ScalPerformanceTest<ElemType>::ScalPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) ( (2 * params->N)  * sizeof(ElemType) ) ), params_(*params), mobjX_(NULL)
{

    X_ = backX_ = NULL;
    mobjX_= NULL;
    lengthX = 1 + (params->N - 1) * abs(params_.incx);
    try {
        X_ = new ElemType[lengthX + params_.offBX];
        backX_ = new ElemType[lengthX + params_.offBX];
    }
    catch(bad_alloc& ba) {
        X_ = backX_ = NULL;     // areResourcesSufficient() will handle the rest and return
        mobjX_= NULL;
        ba = ba;
    }

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
ScalPerformanceTest<ElemType>::~ScalPerformanceTest()
{
	if(X_ != NULL) {
        delete[] X_;
	}
	if(backX_ != NULL) {
        delete[] backX_;
	}
    if( mobjX_ != NULL )
		clReleaseMemObject(mobjX_);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
ScalPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    bool ret;

	if((X_ == NULL) || (backX_ == NULL)) {
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

    ret = ((lengthX + params->offBX) * sizeof(ElemType)) < allocSize;
    ret = ret && ( ((lengthX + params->offBX) * sizeof(ElemType)) < gmemSize);

    return ret;
}

template <typename ElemType> int
ScalPerformanceTest<ElemType>::prepare(void)
{

    alpha_ = convertMultiplier<ElemType>(params_.alpha);
    randomVectors(params_.N, (X_ + params_.offBX), params_.incx);
    memcpy(backX_, X_, (lengthX + params_.offBX)* sizeof(ElemType));
	mobjX_ = base_->createEnqueueBuffer(X_, ((lengthX + params_.offBX) * sizeof(*X_)), 0, CL_MEM_READ_WRITE);

    return (mobjX_ != NULL)? 0 : -1;
}

template <typename ElemType> nano_time_t
ScalPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    bool is_css_zds = (params_.K == 1)? true: false;        // K indicates csscal/zdscal

#ifdef PERF_TEST_WITH_ACML

		time = getCurrentTime();
		clMath::blas::scal(is_css_zds, params_.N, alpha_, X_, params_.offBX, params_.incx);
		time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
ScalPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];
    bool is_css_zds = (params_.K == 1)? true: false;        // K indicates csscal/zdscal

    status = clEnqueueWriteBuffer(queue, mobjX_, CL_TRUE, 0,
                                  (lengthX + params_.offBX) * sizeof(ElemType), backX_, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "Matrix A buffer object enqueuing error, status = " <<
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
    time = getCurrentTime();

#define TIMING
#ifdef TIMING
    clFinish( queue);
    int iter = 50;
    for ( int i=1; i <= iter; i++)
    {
#endif

        status = (cl_int)clMath::clblas::scal(is_css_zds, params_.N, alpha_, mobjX_, params_.offBX, params_.incx,
                            1, &queue, 0, NULL, &event);
        if (status != CL_SUCCESS) {
            cerr << "The CLBLAS SCAL function failed, status = " <<
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

// scal performance test
TEST_P(SCAL, sscal)
{
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<float>::runInstance(FN_SSCAL, &params);
}


TEST_P(SCAL, dscal)
{
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<double>::runInstance(FN_DSCAL, &params);
}

TEST_P(SCAL, cscal)
{
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<FloatComplex>::runInstance(FN_CSCAL, &params);
}


TEST_P(SCAL, zscal)
{
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<DoubleComplex>::runInstance(FN_ZSCAL, &params);
}

TEST_P(SCAL, csscal)
{
    TestParams params;

    getParams(&params);
    params.K = 1;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<FloatComplex>::runInstance(FN_CSSCAL, &params);
}


TEST_P(SCAL, zdscal)
{
    TestParams params;

    getParams(&params);
    params.K = 1;                           // K will indicate wheather routine is csscal/zdscal
    ScalPerformanceTest<DoubleComplex>::runInstance(FN_ZDSCAL, &params);
}
