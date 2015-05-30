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
 * Symm performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <gerc.h>
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

template <typename ElemType> class GercPerformanceTest : public PerformanceTest
{
public:
    virtual ~GercPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        GercPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor =1;

		if (fn == FN_ZGERC &&
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
    GercPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType *A_;
    ElemType *backA_;
    ElemType *x_;
    ElemType *y_;
    cl_mem mobjA_;
    cl_mem mobjx_;
    cl_mem mobjy_;
    int lengthA;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
GercPerformanceTest<ElemType>::GercPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,(problem_size_t) (((2 *  params->M * params->N) +  params->M + params->N ) * sizeof(ElemType) ) ), params_(*params), mobjA_(NULL), mobjx_(NULL), mobjy_(NULL)
{
	//if( params_.side == clblasLeft )
          //      ka = params_.M;
        //else    ka = params_.N;

	if( params_.order == clblasColumnMajor )
			lengthA = params_.N * params_.lda;
		else
			lengthA = params_.M * params_.lda;

    A_ = new ElemType[(lengthA) + params_.offa];
    backA_ = new ElemType[lengthA+ params_.offa];
    x_ = new ElemType[(1 + (params->M - 1) * abs(params_.incx))+ params_.offBX];
    y_ = new ElemType[(1 + (params->N - 1) * abs(params_.incy)) + params_.offCY] ;

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
GercPerformanceTest<ElemType>::~GercPerformanceTest()
{
    if(A_ != NULL)
    {
    delete[] A_;
    }
	if(x_ != NULL)
	{
    delete[] x_;
	}
	if(y_ != NULL)
	{
    delete[] y_;
	}
	if(backA_ != NULL)
	{
    delete[] backA_;
	}

	if( mobjy_ != NULL )
	    clReleaseMemObject(mobjy_);
    if( mobjx_ != NULL )
		clReleaseMemObject(mobjx_);
	if( mobjA_ != NULL )
	    clReleaseMemObject(mobjA_);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
GercPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    bool ret;
    size_t m = params->M, n = params->N;

	if((A_ == NULL) || (backA_ == NULL) || (x_ == NULL) || (y_ == NULL))
	{
        return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

    ret = std::max(m, n) * params_.lda * sizeof(ElemType) < allocSize;
    ret = ret && ( ((1 + (params_.M-1)*abs(params_.incx)))* sizeof(ElemType) < allocSize);
    ret = ret && ( ((1 + (params_.N-1)*abs(params_.incy))) * sizeof(ElemType) < allocSize);

    ret = ret && (((std::max(m, n) * params_.lda) + ((1 + (params_.M-1)*abs(params_.incx))) +  ((1 + (params_.N-1)*abs(params_.incy)))) < gmemSize);

    return ret;
}

template <typename ElemType> int
GercPerformanceTest<ElemType>::prepare(void)
{
    bool useAlpha = base_->useAlpha();

    if (useAlpha) {
        alpha_ = convertMultiplier<ElemType>(params_.alpha);
    }


    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    creationFlags = ( (params_.order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
	BlasRoutineID funcId = CLBLAS_GER;

	populate( A_ + params_.offa, params_.M, params_.N, params_.lda, funcId, creationFlags);
	populate( x_ , (1 + (params_.M-1) * abs(params_.incx) + params_.offBX),1, (1 + (params_.M-1) * abs(params_.incx) + params_.offBX), funcId, 0 );
	populate( y_ , (1 + (params_.N-1) * abs(params_.incy) + params_.offCY),1, (1 + (params_.N-1) * abs(params_.incy) + params_.offCY), funcId, 0 );


        memcpy(backA_, A_, (lengthA + params_.offa)* sizeof(ElemType));

	mobjA_ = base_->createEnqueueBuffer(A_, (lengthA + params_.offa) * sizeof(*A_), 0, CL_MEM_READ_WRITE);
	mobjx_ = base_->createEnqueueBuffer(x_, ( (1 + (params_.M-1) * abs(params_.incx) + params_.offBX)) * sizeof(*x_), 0, CL_MEM_READ_WRITE);
	mobjy_ = base_->createEnqueueBuffer(y_,( (1 + (params_.N-1) * abs(params_.incy) + params_.offCY)) * sizeof(*y_) , 0, CL_MEM_READ_WRITE);

     return ( (mobjA_ != NULL) &&  (mobjx_ != NULL) && (mobjy_ != NULL) ) ? 0 : -1;
}

template <typename ElemType> nano_time_t
GercPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
    size_t lda;
    //int fIncx, fIncy;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    order = params_.order;
    lda = params_.lda;

#ifdef PERF_TEST_WITH_ACML

	 clblasOrder fOrder;
    size_t fN, fM;
    size_t fOffx, fOffy;
    int fIncx, fIncy;
    ElemType  *fX, *fY;
    fOrder = params_.order;
    fM = params_.M;
    fN = params_.N;
    fIncx = params_.incx;
    fIncy = params_.incy;
    fX = x_;
    fY = y_;
    fOffx = params_.offBX;
    fOffy = params_.offCY;

    if (fOrder != clblasColumnMajor) {
           fOrder = clblasColumnMajor;

		   doConjugate( (y_ + params_.offCY), (1 + (params_.N-1) * abs(params_.incy)), 1, 1 );
           fM = params_.N;
           fN = params_.M;
           fX = y_;
           fY = x_;
           fIncx = params_.incy;
           fIncy = params_.incx;
           fOffx = params_.offCY;
           fOffy = params_.offBX;
		   // Note this according to the Legacy guide
		   time = getCurrentTime();
			clMath::blas::ger(fOrder, fM, fN, alpha_, fX , fOffx, fIncx, fY, fOffy, fIncy,  A_, params_.offa, params_.lda);
       }
	else{
		time = getCurrentTime();
		clMath::blas::gerc(order, fM, fN, alpha_, fX, fOffx, params_.incx, fY, fOffy, params_.incy,  A_, params_.offa, lda);
	}
    time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML<F2>

    return time;
}


template <typename ElemType> nano_time_t
GercPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjA_, CL_TRUE, 0,
                                  (lengthA + params_.offa) * sizeof(ElemType), backA_, 0, NULL, &event);
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

        int iter = 20;
        for ( int i = 1; i <= iter; i++)
        {
#endif

    status = (cl_int)clMath::clblas::gerc(params_.order, params_.M, params_.N, alpha_,
         mobjx_, params_.offBX, params_.incx, mobjy_, params_.offCY, params_.incy, mobjA_, params_.offa, params_.lda, 1,
        &queue, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS GERC function failed, status = " <<
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


TEST_P(GERC, cgerc)
{
    TestParams params;

    getParams(&params);
    GercPerformanceTest<FloatComplex>::runInstance(FN_CGERC, &params);
}


TEST_P(GERC, zgerc)
{
    TestParams params;

    getParams(&params);
    GercPerformanceTest<DoubleComplex>::runInstance(FN_ZGERC, &params);
}
