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
 * ROTM performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <rotm.h>
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

// ElemType1 for storing general type, ElemType2 to store type of C which is only float/double
template <typename ElemType> class RotmPerformanceTest : public PerformanceTest
{
public:
    virtual ~RotmPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        RotmPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

        if (((fn == FN_DROTM)) &&
            !base->isDevSupportDoublePrecision())
        {
            std::cerr << ">> WARNING: The target device doesn't support native "
                         "double precision floating point arithmetic" <<
                         std::endl << ">> Test skipped" << std::endl;
            return;
        }

        if (!perfCase.areResourcesSufficient(params))
        {
            std::cerr << ">> RESOURCE CHECK: Skip due to unsufficient resources" <<
                        std::endl;
			return;
        }
        else
        {
            ret = perfCase.run(opFactor);
        }

        ASSERT_GE(ret, 0) << "Fatal error: can not allocate resources or "
                             "perform an OpenCL request!" << endl;
        EXPECT_EQ(0, ret) << "The OpenCL version is slower in the case" << endl;
    }

private:
    RotmPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType *X_, *Y_, *PARAM_, *back_X_, *back_Y_, *back_PARAM_;
    size_t lengthx, lengthy;
    cl_mem mobjX_, mobjY_, mobjParam_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
RotmPerformanceTest<ElemType>::RotmPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) ((4 * params->N + 5) * sizeof(ElemType))), params_(*params)
{

    X_ = Y_ = PARAM_ = NULL;
    back_X_ = back_Y_ = back_PARAM_ = NULL;
    mobjX_= mobjY_ = mobjParam_ = NULL;

    lengthx = 1 + (params_.N - 1) * abs(params_.incx);
    lengthy = 1 + (params_.N - 1) * abs(params_.incy);

    try
    {
        X_ = new ElemType[lengthx + params_.offa];
        back_X_ = new ElemType[lengthx + params_.offa];
        Y_ = new ElemType[lengthy + params_.offb];
        back_Y_ = new ElemType[lengthy + params_.offb];
        PARAM_ = new ElemType[5 + params_.offc];
        back_PARAM_ = new ElemType[5 + params_.offc];
    }
    catch(bad_alloc& ba)
    {
        X_ = back_X_ = Y_ = back_Y_ = NULL;     // areResourcesSufficient() will handle the rest and return
        PARAM_ = back_PARAM_ = NULL;
        ba = ba;
    }

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
RotmPerformanceTest<ElemType>::~RotmPerformanceTest()
{
	if(X_ != NULL)
    {
        delete[] X_;
	}
	if(back_X_ != NULL)
    {
        delete[] back_X_;
	}
    if( mobjX_ != NULL )
    {
		clReleaseMemObject(mobjX_);
    }

    if(Y_ != NULL)
    {
        delete[] Y_;
	}
	if(back_Y_ != NULL)
    {
        delete[] back_Y_;
	}
    if( mobjY_ != NULL )
    {
		clReleaseMemObject(mobjY_);
    }

    if(PARAM_ != NULL)
    {
        delete[] PARAM_;
	}
	if(back_PARAM_ != NULL)
    {
        delete[] back_PARAM_;
	}
    if( mobjParam_ != NULL )
    {
		clReleaseMemObject(mobjParam_);
    }
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
RotmPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t offx = params->offa;
    size_t offy = params->offb;
    size_t offParam = params->offc;

    size_t sizex = (lengthx + offx)*sizeof(ElemType);
    size_t sizey = (lengthy + offy)*sizeof(ElemType);
    size_t sizeParam = (5 + offParam)*sizeof(ElemType);

    bool ret;
    size_t sizeRequired = (sizex + sizey + sizeParam);

	if((X_ == NULL) || (back_X_ == NULL) || (Y_ == NULL) || (back_Y_ == NULL) ||
        (PARAM_ == NULL) || (back_PARAM_ == NULL))
    {
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();

    ret = (sizex < allocSize) && (sizey < allocSize);
    ret = ret && (sizeRequired < gmemSize);

    return ret;
}

template <typename ElemType> int
RotmPerformanceTest<ElemType>::prepare(void)
{
    //Filling random values for SA and SB. C & S are only for output sake
    randomVectors(params_.N, (X_ + params_.offa), params_.incx, (Y_ + params_.offb), params_.incy);
    randomVectors(4, (PARAM_ + params_.offc + 1), 1); //1st element is initialized separately

    ElemType sflagParam = convertMultiplier<ElemType>(params_.alpha);
    PARAM_[params_.offc] = sflagParam; // initializing first element

    memcpy(back_X_, X_, (lengthx + params_.offa)*sizeof(ElemType));
    memcpy(back_Y_, Y_, (lengthy + params_.offb)*sizeof(ElemType));
    memcpy(back_PARAM_, PARAM_, (params_.offc)*sizeof(ElemType));

	// Allocate buffers
    mobjX_ = base_->createEnqueueBuffer(X_, (lengthx + params_.offa) * sizeof(ElemType), 0, CL_MEM_READ_WRITE);
    mobjY_ = base_->createEnqueueBuffer(Y_, (lengthy + params_.offb) * sizeof(ElemType), 0, CL_MEM_READ_WRITE);
    mobjParam_  = base_->createEnqueueBuffer(PARAM_,  (5 + params_.offc) * sizeof(ElemType), 0, CL_MEM_READ_ONLY);

    if((mobjX_ == NULL) || (mobjY_ == NULL) || (mobjParam_ == NULL))
    {
        return -1;
    }
    return 0;
}

template <typename ElemType> nano_time_t
RotmPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;

#ifdef PERF_TEST_WITH_ACML

		time = getCurrentTime();
		clMath::blas::rotm(params_.N, back_X_, params_.offa, params_.incx, back_Y_, params_.offb, params_.incy,
                        back_PARAM_, params_.offc);
		time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
RotmPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    DataType type;
    type = ( typeid(ElemType) == typeid(float))? TYPE_FLOAT: TYPE_DOUBLE;

    status = clEnqueueWriteBuffer(queue, mobjX_, CL_TRUE, 0, (lengthx + params_.offa) * sizeof(ElemType), X_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector X buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clEnqueueWriteBuffer(queue, mobjY_, CL_TRUE, 0, (lengthy + params_.offb) * sizeof(ElemType), Y_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector Y buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clEnqueueWriteBuffer(queue, mobjParam_, CL_TRUE, 0, (5 + params_.offc) * sizeof(ElemType), PARAM_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector C buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS)
    {
        cout << "Wait on event failed, status = " << status << endl;
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
        status = (cl_int)clMath::clblas::rotm(type, params_.N, mobjX_, params_.offa, params_.incx, mobjY_, params_.offb, params_.incy,
                                             mobjParam_, params_.offc, 1, &queue, 0, NULL, &event);
        if (status != CL_SUCCESS)
        {
            cerr << "The CLBLAS ROTM function failed, status = " << status << endl;
            return NANOTIME_ERR;
        }
#ifdef TIMING
    } // iter loop
    clFinish( queue);
    time = getCurrentTime() - time;
    time /= iter;
#else

    status = flushAll(1, &queue);
    if (status != CL_SUCCESS)
    {
        cerr << "clFlush() failed, status = " << status << endl;
        return NANOTIME_ERR;
    }

    time = getCurrentTime();
    status = waitForSuccessfulFinish(1, &queue, &event);
    if (status == CL_SUCCESS)
    {
        time = getCurrentTime() - time;
    }
    else
    {
        cerr << "Waiting for completion of commands to the queue failed, "
                "status = " << status << endl;
        time = NANOTIME_ERR;
    }
#endif
    return time;
}

} // namespace clMath

// rotm performance test
TEST_P(ROTM, srotm)
{
    TestParams params;

    getParams(&params);
    RotmPerformanceTest<float>::runInstance(FN_SROTM, &params);
}


TEST_P(ROTM, drotm)
{
    TestParams params;

    getParams(&params);
    RotmPerformanceTest<double>::runInstance(FN_DROTM, &params);
}

