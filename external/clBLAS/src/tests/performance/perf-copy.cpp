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
#include <copy.h>
#include <blas-random.h>

#ifdef PERF_TEST_WITH_ACML
#include <blas-internal.h>
#include <blas-wrapper.h>
#endif

#include "PerformanceTest.h"

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

template <typename ElemType> class CopyPerformanceTest : public PerformanceTest
{
public:
    virtual ~CopyPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        CopyPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

        if (((fn == FN_DCOPY) || (fn == FN_ZCOPY)) &&
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
    CopyPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
	ElemType *backY_;
    ElemType *blasX_;
    ElemType *blasY_;
    cl_mem mobjX_;
    cl_mem mobjY_;
    size_t  lengthX;
    size_t  lengthY;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
CopyPerformanceTest<ElemType>::CopyPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) ( (2 * params->N)  * sizeof(ElemType) ) ), params_(*params), mobjX_(NULL), mobjY_(NULL)
{

    blasX_ = NULL;
    blasY_ = NULL;
	backY_ = NULL;
    lengthX = 1 + (params->N - 1) * abs(params_.incx);
    lengthY = 1 + (params->N - 1) * abs(params_.incy);

    try
    {
		backY_ = new ElemType[lengthY + params_.offCY];
        blasX_ = new ElemType[lengthX + params_.offBX];
        blasY_ = new ElemType[lengthY + params_.offCY];
    }
    catch(bad_alloc& ba) {
        backY_ = blasX_ = blasY_ = NULL;     // areResourcesSufficient() will handle the rest and return
        mobjX_= mobjY_ = NULL;
        ba = ba;
    }

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
CopyPerformanceTest<ElemType>::~CopyPerformanceTest()
{
	if(blasX_ != NULL)
    {
        delete[] blasX_;
	}
	if(blasY_ != NULL)
    {
        delete[] blasY_;
	}
	if(backY_ != NULL)
	{
		delete[] backY_;
	}
    if( mobjX_ != NULL )
    {
		clReleaseMemObject(mobjX_);
    }
    if( mobjY_ != NULL )
    {
		clReleaseMemObject(mobjY_);
    }
}
template <typename ElemType> bool
CopyPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    bool ret;
    size_t sizeX, sizeY;

	if((blasX_ == NULL) || (blasY_ == NULL) || (backY_ ==NULL) ) {
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();
    sizeX = (lengthX + params->offBX) * sizeof(ElemType);
    sizeY = (lengthY + params->offCY) * sizeof(ElemType);

    ret = ((sizeX < allocSize) && (sizeY < allocSize));
    ret = (ret && ((sizeX + sizeY) < gmemSize));

    return ret;
}

template <typename ElemType> int
CopyPerformanceTest<ElemType>::prepare(void)
{
    randomVectors(params_.N, (blasX_ + params_.offBX), params_.incx, (blasY_ + params_.offCY), params_.incy);
    memcpy(backY_, blasY_, (lengthY + params_.offCY)* sizeof(ElemType));

	mobjX_ = base_->createEnqueueBuffer(blasX_, ((lengthX + params_.offBX) * sizeof(ElemType)), 0, CL_MEM_READ_WRITE);
	mobjY_ = base_->createEnqueueBuffer(blasY_, ((lengthY + params_.offCY) * sizeof(ElemType)), 0, CL_MEM_READ_WRITE);

    return ((mobjX_ != NULL) && (mobjY_ != NULL))? 0 : -1;
}

template <typename ElemType> nano_time_t
CopyPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;

#ifdef PERF_TEST_WITH_ACML

		time = getCurrentTime();
		clMath::blas::copy(params_.N, blasX_, params_.offBX, params_.incx, blasY_, params_.offCY, params_.incy);
		time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
CopyPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    DataType type;
    type = ( typeid(ElemType) == typeid(float))? TYPE_FLOAT:( typeid(ElemType) == typeid(double))? TYPE_DOUBLE:
										( typeid(ElemType) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    status = clEnqueueWriteBuffer(queue, mobjY_, CL_TRUE, 0,
                                  (lengthY + params_.offCY) * sizeof(ElemType), backY_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector Y buffer object enqueuing error, status = " <<
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
    int iter = 100;
    for ( int i=1; i <= iter; i++)
    {
#endif

        status = (cl_int)clMath::clblas::copy(type, params_.N, mobjX_, params_.offBX, params_.incx,
                             mobjY_, params_.offCY, params_.incy, 1, &queue, 0, NULL, &event);
        if (status != CL_SUCCESS) {
            cerr << "The CLBLAS COPY function failed, status = " <<
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

TEST_P(COPY, scopy)
{
    TestParams params;

    getParams(&params);
    CopyPerformanceTest<float>::runInstance(FN_SCOPY, &params);
}


TEST_P(COPY, dcopy)
{
    TestParams params;

    getParams(&params);
    CopyPerformanceTest<double>::runInstance(FN_DCOPY, &params);
}

TEST_P(COPY, ccopy)
{
    TestParams params;

    getParams(&params);
    CopyPerformanceTest<FloatComplex>::runInstance(FN_CCOPY, &params);
}


TEST_P(COPY, zcopy)
{
    TestParams params;

    getParams(&params);
    CopyPerformanceTest<DoubleComplex>::runInstance(FN_ZCOPY, &params);
}
