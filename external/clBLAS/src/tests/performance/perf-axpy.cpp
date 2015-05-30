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
 * AXPY performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <axpy.h>
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

template <typename ElemType> class AxpyPerformanceTest : public PerformanceTest
{
public:
    virtual ~AxpyPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        AxpyPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

        if (((fn == FN_DAXPY) || (fn == FN_ZAXPY)) &&
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
    AxpyPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType *X_;
    ElemType *Y_;
    ElemType *blasX_;
    ElemType *blasY_;
    cl_mem mobjX_;
    cl_mem mobjY_;
    size_t  lengthX;
    size_t  lengthY;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
AxpyPerformanceTest<ElemType>::AxpyPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) ( (3 * params->N)  * sizeof(ElemType) ) ), params_(*params), mobjX_(NULL), mobjY_(NULL)
{

    X_ = blasX_ = Y_ = blasY_ = NULL;

    lengthX = 1 + (params->N - 1) * abs(params_.incx);
    lengthY = 1 + (params->N - 1) * abs(params_.incy);

    try
    {
        X_ = new ElemType[lengthX + params_.offBX];
        blasX_ = new ElemType[lengthX + params_.offBX];
        Y_ = new ElemType[lengthY + params_.offCY];
        blasY_ = new ElemType[lengthY + params_.offCY];
    }
    catch(bad_alloc& ba) {
        X_ = blasX_ = Y_ = blasY_ = NULL;     // areResourcesSufficient() will handle the rest and return
        mobjX_= NULL;
        ba = ba;
    }

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
AxpyPerformanceTest<ElemType>::~AxpyPerformanceTest()
{
	if(X_ != NULL)
    {
        delete[] X_;
	}
	if(blasX_ != NULL)
    {
        delete[] blasX_;
	}
    if( mobjX_ != NULL )
    {
		clReleaseMemObject(mobjX_);
    }
    if(Y_ != NULL)
    {
        delete[] Y_;
	}
	if(blasY_ != NULL)
    {
        delete[] blasY_;
	}
    if( mobjY_ != NULL )
    {
		clReleaseMemObject(mobjY_);
    }
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
AxpyPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize, reqdSize;
    bool ret;

	if((X_ == NULL) || (blasX_ == NULL) || (Y_ == NULL) || (blasY_ == NULL))
    {
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();
    reqdSize = (lengthX + params->offBX + lengthY + params->offCY) * sizeof(ElemType);

    ret = (reqdSize) < allocSize;
    ret = ret && (reqdSize < gmemSize);

    return ret;
}

template <typename ElemType> int
AxpyPerformanceTest<ElemType>::prepare(void)
{

    alpha_ = convertMultiplier<ElemType>(params_.alpha);
    randomVectors(params_.N, (X_ + params_.offBX), params_.incx, (Y_ + params_.offCY), params_.incy);
    memcpy(blasX_, X_, (lengthX + params_.offBX)* sizeof(ElemType));
    memcpy(blasY_, Y_, (lengthY + params_.offCY)* sizeof(ElemType));
	mobjX_ = base_->createEnqueueBuffer(X_, ((lengthX + params_.offBX) * sizeof(ElemType)), 0, CL_MEM_READ_ONLY);
	mobjY_ = base_->createEnqueueBuffer(Y_, ((lengthY + params_.offCY) * sizeof(ElemType)), 0, CL_MEM_READ_WRITE);

    return ((mobjX_ != NULL) && (mobjY_ != NULL))? 0 : -1;
}

template <typename ElemType> nano_time_t
AxpyPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;

#ifdef PERF_TEST_WITH_ACML

		time = getCurrentTime();
		clMath::blas::axpy(params_.N, alpha_, blasX_, params_.offBX, params_.incx, blasY_, params_.offCY, params_.incy);
		time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
AxpyPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjX_, CL_TRUE, 0,
                                  (lengthX + params_.offBX) * sizeof(ElemType), X_, 0, NULL, &event);
    status |= clEnqueueWriteBuffer(queue, mobjY_, CL_TRUE, 0,
                                  (lengthY + params_.offCY) * sizeof(ElemType), Y_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "mobjX_ or mobjY_ buffer object enqueuing error, status = " <<
                 status << endl;

        return NANOTIME_ERR;
    }

    status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS)
    {
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

        status = (cl_int)clMath::clblas::axpy(params_.N, alpha_, mobjX_, params_.offBX, params_.incx, mobjY_, params_.offCY, params_.incy,
                            1, &queue, 0, NULL, &event);
        if (status != CL_SUCCESS) {
            cerr << "The CLBLAS AXPY function failed, status = " <<
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

// axpy performance test
TEST_P(AXPY, saxpy)
{
    TestParams params;

    getParams(&params);
    AxpyPerformanceTest<float>::runInstance(FN_SAXPY, &params);
}


TEST_P(AXPY, daxpy)
{
    TestParams params;

    getParams(&params);
    AxpyPerformanceTest<double>::runInstance(FN_DAXPY, &params);
}

TEST_P(AXPY, caxpy)
{
    TestParams params;

    getParams(&params);
    AxpyPerformanceTest<FloatComplex>::runInstance(FN_CAXPY, &params);
}


TEST_P(AXPY, zaxpy)
{
    TestParams params;

    getParams(&params);
    AxpyPerformanceTest<DoubleComplex>::runInstance(FN_ZAXPY, &params);
}
