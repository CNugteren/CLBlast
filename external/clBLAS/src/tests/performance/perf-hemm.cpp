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
#include <hemm.h>
#include <blas-random.h>

#ifdef PERF_TEST_WITH_ACML
#include <blas-internal.h>
#include <blas-wrapper.h>
#endif

#include "PerformanceTest.h"

//#define SHUNT_ACML_RUN
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

template <typename ElemType> class HemmPerformanceTest : public PerformanceTest
{
public:
    virtual ~HemmPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        HemmPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

            opFactor = 8;

        if ((fn == FN_ZHEMM) &&
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
    HemmPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType alpha_;
    ElemType beta_;
    ElemType *A_;
    ElemType *B_;
    ElemType *C_;
    ElemType *backC_;
    cl_mem mobjA_;
    cl_mem mobjB_;
    cl_mem mobjC_;
    size_t ka, kbc;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
HemmPerformanceTest<ElemType>::HemmPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,
        						(problem_size_t) ( params->M * params->N * ( (params->side == clblasLeft)? params->M : params->N ) ) ),
                        params_(*params), mobjA_(NULL), mobjB_(NULL), mobjC_(NULL)
{
	if( params_.side == clblasLeft )
                ka = params_.M;
        else    ka = params_.N;

	if( params_.order == clblasColumnMajor )
				kbc = params_.N;
		else	kbc = params_.M;

	A_ = new ElemType[params_.lda * ka + params_.offA];
    B_ = new ElemType[params_.ldb * kbc + params_.offBX];
    C_ = new ElemType[params_.ldc * kbc + params_.offCY];
    backC_ = new ElemType[params_.ldc * kbc + params_.offCY];

    base_ = ::clMath::BlasBase::getInstance();
}

template <typename ElemType>
HemmPerformanceTest<ElemType>::~HemmPerformanceTest()
{
    if(A_ != NULL)
    {
    delete[] A_;
    }
	if(B_ != NULL)
	{
    delete[] B_;
	}
	if(C_ != NULL)
	{
    delete[] C_;
	}
	if(backC_ != NULL)
	{
    delete[] backC_;
	}

	if( mobjC_ != NULL )
    {
	    clReleaseMemObject(mobjC_);
    }
    if( mobjB_ != NULL )
    {
		clReleaseMemObject(mobjB_);
    }
	if( mobjA_ != NULL )
    {
	    clReleaseMemObject(mobjA_);
}
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
HemmPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    bool ret;
    size_t m = params->M, n = params->N;

	if((A_ == NULL) || (backC_ == NULL) || (C_ == NULL) || (B_ == NULL))
	{
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

    ret = (std::max(m, n) * params_.lda * sizeof(ElemType)) < allocSize;
    ret = (ret && (std::max(m, n) * params_.ldb * sizeof(ElemType)) < allocSize);
    ret = (ret && (std::max(m, n) * params_.ldc * sizeof(ElemType)) < allocSize);
    ret = (ret && (((std::max(m, n) * params_.lda) + (std::max(m, n) * params_.ldb) + (std::max(m, n) * params_.ldc))) < gmemSize);

    return ret;
}

template <typename ElemType> int
HemmPerformanceTest<ElemType>::prepare(void)
{
    //bool useAlpha = base_->useAlpha();
    //bool useBeta = base_->useBeta();

    int creationFlags = 0, AcreationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    creationFlags = ( (params_.order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    AcreationFlags = ( (params_.uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_HEMM;

    populate( A_ + params_.offA, ka, ka, params_.lda, BlasFn, AcreationFlags);
	populate( B_ + params_.offBX, params_.M, params_.N, params_.ldb, BlasFn, creationFlags );
	populate( C_ + params_.offCY, params_.M, params_.N, params_.ldc, BlasFn, creationFlags );

	memcpy( backC_, C_, (kbc * params_.ldc + params_.offCY) * sizeof(ElemType) );

		mobjA_ = base_->createEnqueueBuffer(A_, (params_.lda * ka  + params_.offA) * sizeof(ElemType), 0, CL_MEM_READ_ONLY);
        mobjB_ = base_->createEnqueueBuffer(B_, (params_.ldb * kbc + params_.offBX) * sizeof(ElemType), 0, CL_MEM_READ_ONLY);
        mobjC_ = base_->createEnqueueBuffer(backC_, (params_.ldc * kbc + params_.offCY) * sizeof(ElemType), 0, CL_MEM_READ_WRITE);

    return (mobjC_) ? 0 : -1;
}

template <typename ElemType> nano_time_t
HemmPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
    clblasUplo fUplo;
	clblasSide fSide;
	size_t lda, ldb, ldc, fN, fM;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif

    order = params_.order;
	fUplo = params_.uplo;
	fSide = params_.side;
    lda = params_.lda;
    ldb = params_.ldb;
    ldc = params_.ldc;
	fM = params_.M;
	fN = params_.N;

#ifdef PERF_TEST_WITH_ACML

	if (order != clblasColumnMajor) {

		   order = clblasColumnMajor;
           fM = params_.N;
           fN = params_.M;
           fSide = (params_.side == clblasLeft)? clblasRight: clblasLeft;
           fUplo = (params_.uplo == clblasUpper)? clblasLower: clblasUpper;
       }


    time = getCurrentTime();
    #ifndef SHUNT_ACML_RUN
   clMath::blas::hemm(order, fSide, fUplo, fM, fN, alpha_,
						A_, params_.offA, lda, B_, params_.offBX, ldb, beta_, C_, params_.offCY, ldc);
    #endif
    time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
HemmPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];

    status = clEnqueueWriteBuffer(queue, mobjC_, CL_TRUE, 0,
                                  (params_.ldc * kbc + params_.offCY) * sizeof(ElemType), backC_, 0, NULL, &event);
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

	time = getCurrentTime();
//#define TIMING
#ifdef TIMING
	clFinish( queue);

	int iter = 20;
	for ( int i = 1; i <= iter; i++)
	{
#endif
    status = (cl_int)clMath::clblas::hemm(params_.order,
        params_.side, params_.uplo, params_.M, params_.N, alpha_,
        mobjA_, params_.offA, params_.lda, mobjB_, params_.offBX, params_.ldb, beta_, mobjC_, params_.offCY, params_.ldc, 1,
        &queue, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS HEMM function failed, status = " <<
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

TEST_P(HEMM, chemm)
{
    TestParams params;

    getParams(&params);
    HemmPerformanceTest<FloatComplex>::runInstance(FN_CHEMM, &params);
}


TEST_P(HEMM, zhemm)
{
    TestParams params;

    getParams(&params);
    HemmPerformanceTest<DoubleComplex>::runInstance(FN_ZHEMM, &params);
}
