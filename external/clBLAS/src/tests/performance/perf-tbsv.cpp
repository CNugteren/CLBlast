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
 * Tbsv performance test cases
 */

#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <tbmv.h>
#include <tbsv.h>
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

template <typename ElemType> class TbsvPerformanceTest : public PerformanceTest
{
public:
    virtual ~TbsvPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        TbsvPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor = 1;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        if ((fn == FN_DTBSV || fn == FN_ZTBSV) &&
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
    TbsvPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType *A_;
    ElemType *X_;
    ElemType *backX_;
    cl_mem mobjA_;
    cl_mem mobjX_;
    cl_mem mobjScratch_;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
TbsvPerformanceTest<ElemType>::TbsvPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest(fn,
    (problem_size_t)(   params->N * (params->K+1) * 2           // A & X access
                     - (params->K * (params->K+1) )             // Substract hole-part for A & X
                     + (2*params->N)   /* Y access */  ) * sizeof(ElemType)  ),
                            params_(*params), mobjA_(NULL), mobjX_(NULL), mobjScratch_(NULL)
{
    size_t lenA, lenX;
    lenA = params_.N  * params_.lda + params_.offA;
    lenX = (params_.N  - 1)* params_.incx + 1 + params_.offBX;
    A_ = new ElemType[ lenA ];
    X_ = new ElemType[ lenX ];
    backX_ = new ElemType[ lenX ];

    base_ = ::clMath::BlasBase::getInstance();

	mobjA_ = NULL;
	mobjX_ = NULL;
	mobjScratch_ = NULL;
}

template <typename ElemType>
TbsvPerformanceTest<ElemType>::~TbsvPerformanceTest()
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

    if ( mobjA_ != NULL )
		clReleaseMemObject(mobjA_);
	if ( mobjX_ != NULL )
	    clReleaseMemObject(mobjX_);
	if ( mobjScratch_ != NULL )
		clReleaseMemObject(mobjScratch_);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
TbsvPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t n = params->N, lda = params->lda;
    size_t lenA = (n * lda  + params->offA)* sizeof(ElemType);
    size_t lenX = ((params->N - 1)* params->incx + 1 + params->offBX) * sizeof(ElemType);

    if((A_ == NULL) || (X_ == NULL) || (backX_ == NULL))
	{
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize(0);
    allocSize = (size_t)base->maxMemAllocSize();

    bool suff = (lenA < allocSize) && ( (lenA + 2 * lenX) < gmemSize );

    return suff;
}

template <typename ElemType> int
TbsvPerformanceTest<ElemType>::prepare(void)
{
    size_t lenX, lenA;

    lenA = params_.N * params_.lda + params_.offA;
    lenX = (params_.N - 1)*abs(params_.incx) + 1 + params_.offBX;

    randomTbsvMatrices( params_.order, params_.uplo, params_.diag, params_.N, params_.K,
                            (A_+params_.offA), params_.lda, (X_+params_.offBX), params_.incx );

    memcpy(backX_, X_, lenX * sizeof(ElemType));

    mobjA_ = base_->createEnqueueBuffer(A_, lenA * sizeof(ElemType), 0, CL_MEM_READ_ONLY);
    mobjX_ = base_->createEnqueueBuffer(X_, lenX * sizeof(ElemType), 0, CL_MEM_READ_WRITE);
    mobjScratch_ = base_->createEnqueueBuffer(backX_, lenX * sizeof(ElemType), 0, CL_MEM_READ_WRITE);

    return ((mobjA_ != NULL) && (mobjX_ != NULL) && (mobjScratch_ != NULL)) ? 0 : -1;
}

template <typename ElemType> nano_time_t
TbsvPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder fOrder;
    clblasTranspose fTrans;
    clblasUplo fUplo;
    size_t lda, lenA, lenX;

    lenA = params_.N * params_.lda;
    lenX = (params_.N - 1)* params_.incx + 1 + params_.offBX;

    memcpy(X_, backX_, lenX * sizeof(ElemType));
    fOrder = params_.order;
    fTrans = params_.transA;
    fUplo = params_.uplo;
    lda = params_.lda;

    if (fOrder != clblasColumnMajor)
    {
        fOrder = clblasColumnMajor;
        fTrans = (params_.transA == clblasNoTrans)? clblasTrans : clblasNoTrans;
        fUplo = (params_.uplo == clblasLower)? clblasUpper : clblasLower;

		if( params_.transA == clblasConjTrans )
            doConjugate( (A_+params_.offA), 1, lenA, lda );
   	}

#ifdef PERF_TEST_WITH_ACML

   	time = getCurrentTime();
   	clMath::blas::tbsv(fOrder, fUplo, fTrans, params_.diag, params_.N, params_.K, A_, params_.offA, lda, X_, params_.offBX, params_.incx);
  	time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
TbsvPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    size_t lenX;
    cl_command_queue queue = base_->commandQueues()[0];

    lenX = (params_.N - 1)* params_.incx + 1 + params_.offBX;
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
    DataType type;
    type = ( typeid(ElemType) == typeid(float))? TYPE_FLOAT:( typeid(ElemType) == typeid(double))? TYPE_DOUBLE:
										( typeid(ElemType) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    event = NULL;
    time = getCurrentTime();
    int iter = 20;
	for ( int i = 1; i <= iter; i++)
	{
        status = clMath::clblas::tbsv(type, params_.order, params_.uplo, params_.transA, params_.diag, params_.N, params_.K,
                                        mobjA_, params_.offA, params_.lda, mobjX_, params_.offBX, params_.incx,
                                         1, &queue, 0, NULL, &event);

        if (status != CL_SUCCESS) {
            cerr << "The CLBLAS TBSV function failed, status = " <<
                    status << endl;
            return NANOTIME_ERR;
        }
    }
    clFinish( queue );
    time = getCurrentTime() - time;
	time /= iter;

    return time;
}

} // namespace clMath

TEST_P(TBSV, stbsv)
{
    TestParams params;

    getParams(&params);
    TbsvPerformanceTest<float>::runInstance(FN_STBSV, &params);
}

TEST_P(TBSV, dtbsv)
{
    TestParams params;

    getParams(&params);
    TbsvPerformanceTest<double>::runInstance(FN_DTBSV, &params);
}

TEST_P(TBSV, ctbsv)
{
    TestParams params;

    getParams(&params);
    TbsvPerformanceTest<FloatComplex>::runInstance(FN_CTBSV, &params);
}

TEST_P(TBSV, ztbsv)
{
    TestParams params;

    getParams(&params);
    TbsvPerformanceTest<DoubleComplex>::runInstance(FN_ZTBSV, &params);
}
