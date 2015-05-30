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
 * Her2 performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <her2.h>
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

template <typename ElemType> class Her2PerformanceTest : public PerformanceTest
{
public:
    virtual ~Her2PerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        Her2PerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        opFactor = 1;

        if ((fn == FN_ZHER2) &&
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
    Her2PerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType *A_;
    ElemType *X_;
	ElemType *Y_;
	ElemType *backA_;
    cl_mem mobjA_;
    cl_mem mobjX_;
	cl_mem mobjY_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
Her2PerformanceTest<ElemType>::Her2PerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn, (problem_size_t)((((params->N * params->N) + (params->N)) * 3 ) * sizeof(ElemType))),
	params_(*params), mobjA_(NULL), mobjX_(NULL), mobjY_(NULL)
{
    A_ = new ElemType[params_.N * params_.lda + params_.offa];
    X_ = new ElemType[ 1 + (params_.N-1) * abs(params_.incx) + params_.offBX];
	Y_ = new ElemType[ 1 + (params_.N-1) * abs(params_.incy) + params_.offCY];
    backA_ = new ElemType[params_.N * params_.lda + params_.offa];

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
Her2PerformanceTest<ElemType>::~Her2PerformanceTest()
{
	if(A_ != NULL)
    {
    delete[] A_;
    }
	if(backA_ != NULL)
	{
		delete[] backA_;
	}
	if(X_ != NULL)
	{
    delete[] X_;
	}
	if(Y_ != NULL)
	{
	delete[] Y_;
	}

	if(mobjX_ != NULL) {
		clReleaseMemObject(mobjX_);
    }

	if(mobjY_ != NULL) {
		clReleaseMemObject(mobjY_);
	}

	if(mobjA_ != NULL) {
		clReleaseMemObject(mobjA_);
	}
}

template <typename ElemType> bool
Her2PerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t n = params->N;

	if((A_ == NULL) || (backA_ == NULL) || (X_ == NULL) || (Y_ == NULL))
	{
        return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();

	bool suff = ( sizeof(ElemType)*n*params->lda < allocSize ) && ((1 + (n-1)*abs(params->incx))*sizeof(ElemType) < allocSize)
				  && ((1 + (n-1)*abs(params->incy))*sizeof(ElemType) < allocSize); //for individual allocations
    suff = suff && ((( n*params->lda + (1 + (n-1)*abs(params->incx)) + (1 + (n-1)*abs(params->incy)))*sizeof(ElemType)) < gmemSize) ; //for total global allocations

    return suff ;
}

template <typename ElemType> int
Her2PerformanceTest<ElemType>::prepare(void)
{
    //bool useAlpha = true;
	size_t lenX = 1 + (params_.N-1) * abs(params_.incx);
	size_t lenY = 1 + (params_.N-1) * abs(params_.incy);

    alpha_ = convertMultiplier<ElemType>(params_.alpha);

	randomHer2Matrices<ElemType>(params_.order, params_.uplo, params_.N, &alpha_, (A_ + params_.offa), params_.lda,
							(X_ + params_.offBX), params_.incx, (Y_ + params_.offCY), params_.incy);

	memcpy(backA_, A_, ((params_.N * params_.lda + params_.offa)* sizeof(ElemType)));

    mobjA_ = base_->createEnqueueBuffer(A_, (params_.N * params_.lda + params_.offa)* sizeof(*A_), 0, CL_MEM_READ_WRITE);
    mobjX_ = base_->createEnqueueBuffer(X_, (lenX + params_.offBX ) * sizeof(*X_), 0, CL_MEM_READ_ONLY);
	mobjY_ = base_->createEnqueueBuffer(Y_, (lenY + params_.offCY ) * sizeof(*Y_), 0, CL_MEM_READ_ONLY);

	return ((mobjA_ != NULL) &&  (mobjX_ != NULL) && (mobjY_ != NULL)) ? 0 : -1;
}

template <typename ElemType> nano_time_t
Her2PerformanceTest<ElemType>::etalonPerfSingle(void)
{
	clblasOrder order;
    clblasUplo fUplo;
    nano_time_t time = 0;
    size_t lda;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    order = params_.order;
    lda = params_.lda;
    fUplo = params_.uplo;

#ifdef PERF_TEST_WITH_ACML

    ElemType *fX, *fY;
    int fIncx, fIncy;
    size_t fOffx, fOffy;
	fX = X_;    fOffx = params_.offBX;  fIncx = params_.incx;
	fY = Y_;    fOffy = params_.offCY;  fIncy = params_.incy;


	if (order != clblasColumnMajor)
    {
		doConjugate( (X_ + params_.offBX), (1 + (params_.N-1) * abs(params_.incx)), 1, 1 );
        doConjugate( (Y_ + params_.offCY), (1 + (params_.N-1) * abs(params_.incy)), 1, 1 );
        order = clblasColumnMajor;
        fUplo = (fUplo == clblasLower)? clblasUpper : clblasLower;
	    fX = Y_;    fOffx = params_.offCY;  fIncx = params_.incy;
	    fY = X_;    fOffy = params_.offBX;  fIncy = params_.incx;
    }

   	time = getCurrentTime();
   	clMath::blas::her2(order, fUplo, params_.N, alpha_, fX, fOffx, fIncx, fY,
					fOffy, fIncy, A_, params_.offa, lda);
	time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
Her2PerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjA_, CL_TRUE, 0,
                                  ((params_.N * params_.lda) + params_.offa) *
                                  sizeof(ElemType), backA_, 0, NULL, &event);
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

#define TIMING
#ifdef TIMING
    clFinish( queue);
    time = getCurrentTime();

    int iter = 20;
    for ( int i = 1; i <= iter; i++)
    {
#endif
    status = (cl_int)clMath::clblas::her2(params_.order, params_.uplo, params_.N, alpha_, mobjX_, params_.offBX, params_.incx,
				mobjY_, params_.offCY, params_.incy, mobjA_, params_.offa, params_.lda, 1, &queue, 0, NULL, &event);

    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS HER2 function failed, status = " <<
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

TEST_P(HER2, cher2)
{
    TestParams params;

    getParams(&params);
    Her2PerformanceTest<FloatComplex>::runInstance(FN_CHER2, &params);
}

TEST_P(HER2, zher2)
{
    TestParams params;

    getParams(&params);
    Her2PerformanceTest<DoubleComplex>::runInstance(FN_ZHER2, &params);
}
