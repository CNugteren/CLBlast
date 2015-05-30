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
 * Gemv performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <tpsv.h>
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

template <typename ElemType> class TpsvPerformanceTest : public PerformanceTest
{
public:
    virtual ~TpsvPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        TpsvPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

		opFactor = 1;

        if ((fn == FN_DTPSV || fn == FN_ZTPSV) &&
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
    TpsvPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
   // ElemType alpha_;
    ElemType *A_;
    ElemType *X_;
    ElemType *backX_;
    cl_mem mobjA_;
    cl_mem mobjX_;
    size_t lengthA;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
TpsvPerformanceTest<ElemType>::TpsvPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest( fn, (problem_size_t)( ( params->N * (params->N+1) * sizeof(ElemType) ) ) ),
    params_(*params), mobjA_(NULL), mobjX_(NULL)
{
	lengthA = (params_.N * (params_.N + 1))/2;
    A_ = 		new ElemType[(lengthA) + params_.offa];
    X_ = 		new ElemType[ 1 + ((params_.N-1) * abs(params_.incx)) + params_.offBX ];
    backX_ = 	new ElemType[ 1 + ((params_.N-1) * abs(params_.incx)) + params_.offBX ];

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
TpsvPerformanceTest<ElemType>::~TpsvPerformanceTest()
{
     if(A_ != NULL)
    {
    delete[] A_;
    }
	if(X_ != NULL)
	{
    delete[] X_;
	}
	if(backX_ != NULL)
	{
    delete[] backX_;
	}

    if( mobjA_ != NULL )
		clReleaseMemObject(mobjA_);
    if( mobjX_ != NULL )
		clReleaseMemObject(mobjX_);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
TpsvPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t n = params->N;

	if ((A_ == NULL) || (X_ == NULL) || (backX_ == NULL))
	{
        return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

	bool suff = ( sizeof(ElemType)*((n*(n+1))/2) < allocSize ) && ((1 + (n-1)*abs(params->incx))*sizeof(ElemType) < allocSize); //for individual allocations
    suff = suff && ((( ((n*(n+1))/2) + (1 + (n-1)*abs(params->incx))*2)*sizeof(ElemType)) < gmemSize) ; //for total global allocations
    return suff ;
}

template <typename ElemType> int
TpsvPerformanceTest<ElemType>::prepare(void)
{
    size_t lenX, N;

    N = params_.N;
    lenX = 1 + ((N-1) *abs(params_.incx)) + params_.offBX;


	randomTrsvMatrices( params_.order, params_.uplo, params_.diag, params_.N, (A_ + params_.offa), 0,
										(X_ + params_.offBX), params_.incx);
    memcpy(backX_, X_, lenX * sizeof(ElemType));
    mobjA_ = base_->createEnqueueBuffer(A_, ((lengthA) + params_.offa) *
                                     sizeof(*A_), 0, CL_MEM_READ_ONLY);
    mobjX_ = base_->createEnqueueBuffer(X_, lenX *
                                     sizeof(*X_), 0, CL_MEM_READ_WRITE);
    return ((mobjA_ != NULL) &&  (mobjX_ != NULL) ) ? 0 : -1;
}

template <typename ElemType> nano_time_t
TpsvPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
	clblasUplo fUplo;
    clblasTranspose fTrans;
#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif
    memcpy(X_, backX_, ((1 + ((params_.N-1) * abs(params_.incx)))+params_.offBX) * sizeof(ElemType));
    order = params_.order;
	fUplo = params_.uplo;
    fTrans = params_.transA;

#ifdef PERF_TEST_WITH_ACML

	if (order != clblasColumnMajor)
    {
        order = clblasColumnMajor;
        fUplo =  (params_.uplo == clblasUpper)? clblasLower : clblasUpper;
        fTrans = (params_.transA == clblasNoTrans)? clblasTrans : clblasNoTrans;

        if( params_.transA == clblasConjTrans )
            doConjugate( A_ + params_.offa, 1, lengthA, 1 );
    }
    //printf("Calling ACML TPSV\n");
    //printf("X Before calling %f %f %f %f\n", X_[0], X_[1], X_[2], X_[3]);
    time = getCurrentTime();
    clMath::blas::tpsv(order, fUplo, fTrans, params_.diag,
                    params_.N, A_, params_.offa, X_, params_.offBX, params_.incx);
    time = getCurrentTime() - time;
    //printf("X After Calling %f %f %f %f\n", X_[0], X_[1], X_[2], X_[3]);
    //printf("time %lu\n", (unsigned long)time );
#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
TpsvPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];
    size_t lenX = 1 + ((params_.N-1) * abs(params_.incx)) + params_.offBX;

    status = clEnqueueWriteBuffer(queue, mobjX_, CL_TRUE, 0,
                                  lenX * sizeof(ElemType), backX_, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "Vector X buffer object enqueuing error, status = " <<
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
    //printf("backX before calling %f %f %f %f\n", backX_[0], backX_[1], backX_[2], backX_[3]);

    DataType type;
    type = ( typeid(ElemType) == typeid(float))? TYPE_FLOAT:( typeid(ElemType) == typeid(double))? TYPE_DOUBLE: ( typeid(ElemType) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

	time = getCurrentTime();
//#define TIMING
#ifdef TIMING
	clFinish( queue);

	int iter = 20;
	for ( int i = 1; i <= iter; i++)
	{
#endif
    status = (cl_int)clMath::clblas::tpsv(type, params_.order, params_.uplo,
        params_.transA, params_.diag, params_.N, mobjA_, params_.offa,
        mobjX_, params_.offBX, params_.incx, 1, &queue, 0, NULL, &event);

    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS TPSV function failed, status = " <<
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

    status = waitForSuccessfulFinish(1, &queue, &event);
    if (status == CL_SUCCESS) {
        time = getCurrentTime() - time;

        clEnqueueReadBuffer(queue, mobjX_, CL_TRUE, 0,
                            lenX * sizeof(ElemType), backX_, 0, NULL, NULL);

        /*
        printf("X Vector is \n");
        for(int i =0 ; i<params_.N; i++)
            printf("%f ", backX_[i]);
        printf("\n");
        printf("backX After calling %4.10f %4.10f %4.10f %4.10f\n", backX_[0], backX_[1], backX_[2], backX_[3]);
        */
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


TEST_P(TPSV, stpsv)
{
    TestParams params;

    getParams(&params);
    TpsvPerformanceTest<float>::runInstance(FN_STPSV, &params);
}


TEST_P(TPSV, dtpsv)
{
    TestParams params;

    getParams(&params);
    TpsvPerformanceTest<double>::runInstance(FN_DTPSV, &params);
}

TEST_P(TPSV, ctpsv)
{
    TestParams params;

    getParams(&params);
    TpsvPerformanceTest<FloatComplex>::runInstance(FN_CTPSV, &params);
}

TEST_P(TPSV, ztpsv)
{
    TestParams params;

    getParams(&params);
    TpsvPerformanceTest<DoubleComplex>::runInstance(FN_ZTPSV, &params);
}

