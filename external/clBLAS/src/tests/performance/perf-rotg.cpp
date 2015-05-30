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
 * ROTG performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <rotg.h>
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
template <typename ElemType1, typename ElemType2> class RotgPerformanceTest : public PerformanceTest
{
public:
    virtual ~RotgPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        RotgPerformanceTest<ElemType1, ElemType2> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

        if (((fn == FN_DROTG) || (fn == FN_ZROTG)) &&
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
    RotgPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType1 *SA_, *SB_, *S_, *back_SA_, *back_SB_, *back_S_;
    ElemType2 *C_, *back_C_;
    cl_mem mobjSA_, mobjSB_, mobjC_, mobjS_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType1, typename ElemType2>
RotgPerformanceTest<ElemType1, ElemType2>::RotgPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) (5 * sizeof(ElemType1) + sizeof(ElemType2))), params_(*params)
{

    SA_ = SB_ = S_ = NULL;
    back_SA_ = back_SB_ = back_S_ = NULL;
    C_ = back_C_ = NULL;
    mobjSA_= mobjSB_ = mobjC_ = mobjS_ = NULL;

    try
    {
        SA_ = new ElemType1[1 + params_.offBX];
        back_SA_ = new ElemType1[1 + params_.offBX];
        SB_ = new ElemType1[1 + params_.offCY];
        back_SB_ = new ElemType1[1 + params_.offCY];
        C_ = new ElemType2[1 + params_.offa];
        back_C_ = new ElemType2[1 + params_.offa];
        S_ = new ElemType1[1 + params_.offb];
        back_S_ = new ElemType1[1 + params_.offb];
    }
    catch(bad_alloc& ba)
    {
        SA_ = back_SA_ = SB_ = back_SB_ = NULL;     // areResourcesSufficient() will handle the rest and return
        S_ = back_S_ = NULL;
        C_ = back_C_ = NULL;
        ba = ba;
    }

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType1, typename ElemType2>
RotgPerformanceTest<ElemType1, ElemType2>::~RotgPerformanceTest()
{
	if(SA_ != NULL)
    {
        delete[] SA_;
	}
	if(back_SA_ != NULL)
    {
        delete[] back_SA_;
	}
    if( mobjSA_ != NULL )
    {
		clReleaseMemObject(mobjSA_);
    }

    if(SB_ != NULL)
    {
        delete[] SB_;
	}
	if(back_SB_ != NULL)
    {
        delete[] back_SB_;
	}
    if( mobjSB_ != NULL )
    {
		clReleaseMemObject(mobjSB_);
    }

    if(C_ != NULL)
    {
        delete[] C_;
	}
	if(back_C_ != NULL)
    {
        delete[] back_C_;
	}
    if( mobjC_ != NULL )
    {
		clReleaseMemObject(mobjC_);
    }

    if(S_ != NULL)
    {
        delete[] S_;
	}
	if(back_S_ != NULL)
    {
        delete[] back_S_;
	}
    if( mobjS_ != NULL )
    {
		clReleaseMemObject(mobjS_);
    }
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType1, typename ElemType2> bool
RotgPerformanceTest<ElemType1, ElemType2>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t offSA_ = params->offBX;
    size_t offSB_ = params->offCY;
    size_t offC_ = params->offa;
    size_t offS_ = params->offb;
    bool ret;
    size_t sizeRequired = ((1 + offSA_) + (1 + offSB_) + (1 + offS_)) * sizeof(ElemType1)
                             + ((1 + offC_) * sizeof(ElemType2));

	if((SA_ == NULL) || (back_SA_ == NULL) || (SB_ == NULL) || (back_SB_ == NULL) ||
        (C_ == NULL) || (back_C_ == NULL) || (S_ == NULL) || (back_S_ == NULL))
    {
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

    ret = (sizeRequired) < allocSize;
    ret = ret && (sizeRequired < gmemSize);

    return ret;
}

template <typename ElemType1, typename ElemType2> int
RotgPerformanceTest<ElemType1, ElemType2>::prepare(void)
{
    randomVectors(1, (SA_ + params_.offBX), 1, (SB_ + params_.offCY), 1);
    C_[params_.offa] = back_C_[params_.offa] = ZERO<ElemType2>();
    S_[params_.offb] = back_S_[params_.offb] = ZERO<ElemType1>();
    back_SA_[params_.offBX] = SA_[params_.offBX];
    back_SB_[params_.offCY] = SB_[params_.offCY];

    //printing the inputs, as they change after processing
    ::std::cerr << "A = ";
    printElement<ElemType1>(SA_[params_.offBX]);
    ::std::cerr << "\tB = ";
    printElement<ElemType1>(SB_[params_.offCY]);
    ::std::cerr << "\tC = ";
    printElement<ElemType2>(C_[params_.offa]);
    ::std::cerr << "\tS = ";
    printElement<ElemType1>(S_[params_.offb]);
    ::std::cout << std::endl << std::endl;

	// Allocate buffers
    mobjSA_ = base_->createEnqueueBuffer(SA_, (1 + params_.offBX) * sizeof(ElemType1), 0, CL_MEM_READ_WRITE);
    mobjSB_ = base_->createEnqueueBuffer(SB_, (1 + params_.offCY) * sizeof(ElemType1), 0, CL_MEM_READ_WRITE);
    mobjC_  = base_->createEnqueueBuffer(C_,  (1 + params_.offa ) * sizeof(ElemType2), 0, CL_MEM_WRITE_ONLY);
    mobjS_  = base_->createEnqueueBuffer(S_,  (1 + params_.offb ) * sizeof(ElemType1), 0, CL_MEM_WRITE_ONLY);

    if((mobjSA_ == NULL) || (mobjSB_ == NULL) || (mobjC_ == NULL) || (mobjS_ == NULL))
    {
        return -1;
    }
    return 0;
}

template <typename ElemType1, typename ElemType2> nano_time_t
RotgPerformanceTest<ElemType1, ElemType2>::etalonPerfSingle(void)
{
    nano_time_t time = 0;

#ifdef PERF_TEST_WITH_ACML

		time = getCurrentTime();
		clMath::blas::rotg(back_SA_, params_.offBX, back_SB_, params_.offCY, back_C_, params_.offa, back_S_, params_.offb);
		time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType1, typename ElemType2> nano_time_t
RotgPerformanceTest<ElemType1, ElemType2>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    DataType type;
    type = ( typeid(ElemType1) == typeid(float))? TYPE_FLOAT:( typeid(ElemType1) == typeid(double))? TYPE_DOUBLE:
										( typeid(ElemType1) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    status = clEnqueueWriteBuffer(queue, mobjSA_, CL_TRUE, 0, (1 + params_.offBX) * sizeof(ElemType1), SA_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector SA buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clEnqueueWriteBuffer(queue, mobjSB_, CL_TRUE, 0, (1 + params_.offCY) * sizeof(ElemType1), SB_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector SB buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clEnqueueWriteBuffer(queue, mobjC_, CL_TRUE, 0, (1 + params_.offa) * sizeof(ElemType2), C_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector C buffer object enqueuing error, status = " << status << endl;
        return NANOTIME_ERR;
    }

    status = clEnqueueWriteBuffer(queue, mobjS_, CL_TRUE, 0, (1 + params_.offb) * sizeof(ElemType1), S_, 0, NULL, &event);
    if (status != CL_SUCCESS)
    {
        cerr << "Vector S buffer object enqueuing error, status = " << status << endl;
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
        status = (cl_int)clMath::clblas::rotg(type, mobjSA_, params_.offBX, mobjSB_, params_.offCY, mobjC_, params_.offa, mobjS_, params_.offb,
                                                1, &queue, 0, NULL, &event);
        if (status != CL_SUCCESS)
        {
            cerr << "The CLBLAS ROTG function failed, status = " << status << endl;
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

// rotg performance test
TEST_P(ROTG, srotg)
{
    TestParams params;

    getParams(&params);
    RotgPerformanceTest<float, float>::runInstance(FN_SROTG, &params);
}


TEST_P(ROTG, drotg)
{
    TestParams params;

    getParams(&params);
    RotgPerformanceTest<double, double>::runInstance(FN_DROTG, &params);
}

TEST_P(ROTG, crotg)
{
    TestParams params;

    getParams(&params);
    RotgPerformanceTest<FloatComplex, float>::runInstance(FN_CROTG, &params);
}


TEST_P(ROTG, zrotg)
{
    TestParams params;

    getParams(&params);
    RotgPerformanceTest<DoubleComplex, double>::runInstance(FN_ZROTG, &params);
}
