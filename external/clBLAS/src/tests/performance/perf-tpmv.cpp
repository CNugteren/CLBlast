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
#include <tpmv.h>
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

template <typename ElemType> class TpmvPerformanceTest : public PerformanceTest
{
public:
    virtual ~TpmvPerformanceTest();

    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

    static void runInstance(BlasFunction fn, TestParams *params)
    {
        TpmvPerformanceTest<ElemType> perfCase(fn, params);
        int ret = 0;
        int opFactor;
        BlasBase *base;

        base = clMath::BlasBase::getInstance();

        /*  *************Important*********************
		if (fn == FN_STRMV || fn == FN_DTRMV) {
            opFactor = 2;
        }
        else {
            opFactor = 8;
        }   this is only for blas-3 routines- operations factor
			FOR BLAS-2(bandwidth intensive) ROUTINES MAKE opFactor AS 1 and pass the appropriate size
			that is read and written in the constructor below           */
		opFactor = 1; //FIX-ME

        if ((fn == FN_DTPMV || fn == FN_ZTPMV) &&
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
    TpmvPerformanceTest(BlasFunction fn, TestParams *params);

    bool areResourcesSufficient(TestParams *params);

    TestParams params_;
    ElemType *AP_;
	ElemType *X_;
    ElemType *backX_;
    cl_mem mobjAP_;
    cl_mem mobjX_;
	cl_mem scratchBuff;
    ::clMath::BlasBase *base_;
};

template <typename ElemType>
TpmvPerformanceTest<ElemType>::TpmvPerformanceTest(
    BlasFunction fn,
    TestParams *params) : PerformanceTest( fn, (problem_size_t)( ( params->N * (params->N+1) * sizeof(ElemType) ) ) ),
    params_(*params), mobjAP_(NULL), mobjX_(NULL)
{

    AP_ = new ElemType[( ( params_.N *( params_.N + 1 ) )/2 ) + params_.offa];
    X_ = new ElemType[ 1 + (params_.N-1) * abs(params_.incx)  + params_.offBX];
    backX_ = new ElemType[ 1 + (params_.N-1) * abs(params_.incx)  + params_.offBX];

    base_ = ::clMath::BlasBase::getInstance();
	mobjAP_ = NULL;
	mobjX_ = NULL;
	scratchBuff = NULL;
}

template <typename ElemType>
TpmvPerformanceTest<ElemType>::~TpmvPerformanceTest()
    // Matrix A
{
    if(AP_ != NULL)
    {
    delete[] AP_;
    }
	if(X_ != NULL)
	{
    delete[] X_;
	}
	if(backX_ != NULL)
	{
    delete[] backX_;
	}

    if ( mobjAP_ != NULL )
		clReleaseMemObject(mobjAP_);
	if ( mobjX_ != NULL )
	    clReleaseMemObject(mobjX_);
	if ( scratchBuff != NULL )
		clReleaseMemObject(scratchBuff);
}

/*
 * Check if available OpenCL resources are sufficient to
 * run the test case
 */
template <typename ElemType> bool
TpmvPerformanceTest<ElemType>::areResourcesSufficient(TestParams *params)
{
    clMath::BlasBase *base;
    size_t gmemSize, allocSize;
    size_t n = params->N;

	if((AP_ == NULL) || (X_ == NULL) || (backX_ == NULL))
	{
		return 0;
	}

    base = clMath::BlasBase::getInstance();
    gmemSize = (size_t)base->availGlobalMemSize( 0 );
    allocSize = (size_t)base->maxMemAllocSize();

    bool suff = ( sizeof(ElemType)*( ( n *( n + 1 ) )/2 )< allocSize ) && ((1 + (n-1)*abs(params->incx))*sizeof(ElemType) < allocSize); //for individual allocations
	suff = suff && ((( ( ( n *( n + 1 ) )/2 ) + (1 + (n-1)*abs(params->incx))*2)*sizeof(ElemType)) < gmemSize) ; //for total global allocations

    return suff ;
}

template <typename ElemType> int
TpmvPerformanceTest<ElemType>::prepare(void)
{
    size_t lenX, n;
	n = params_.N;
    lenX = 1 + (n-1) * abs(params_.incx);


	int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT | PACKED_MATRIX;

    // Default is Column-Major
    creationFlags = ( (this-> params_.order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    creationFlags = ( (this-> params_.uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_TRMV;

     populate( (AP_ + params_.offa), n, n, 0, BlasFn, creationFlags);
     populate( X_ , lenX + params_.offBX, 1, lenX + params_.offBX, BlasFn);
     memcpy(backX_, X_, ((1 + (params_.N-1) * abs(params_.incx))+ params_.offBX )* sizeof(ElemType));


    mobjAP_ = base_->createEnqueueBuffer(AP_,( (( n *( n + 1 ) )/2 ) + params_.offa)* sizeof(*AP_), 0, CL_MEM_READ_ONLY);
    mobjX_ = base_->createEnqueueBuffer(X_, (lenX + params_.offBX )* sizeof(*X_), 0, CL_MEM_WRITE_ONLY);
	scratchBuff = base_->createEnqueueBuffer(NULL , lenX * sizeof(*X_), 0, CL_MEM_READ_ONLY);

    return ( (mobjAP_ != NULL) &&  (mobjX_ != NULL) && (scratchBuff != NULL) ) ? 0 : -1;
}

template <typename ElemType> nano_time_t
TpmvPerformanceTest<ElemType>::etalonPerfSingle(void)
{
    nano_time_t time = 0;
    clblasOrder order;
	clblasUplo fUplo;
	clblasTranspose fTrans;
    //size_t lda;

#ifndef PERF_TEST_WITH_ROW_MAJOR
    if (params_.order == clblasRowMajor) {
        cerr << "Row major order is not allowed" << endl;
        return NANOTIME_ERR;
    }
#endif
    order = params_.order;
	fUplo = params_.uplo;
	fTrans = params_.transA;
    //lda = params_.lda;

#ifdef PERF_TEST_WITH_ACML

	if (order != clblasColumnMajor)
    {
        order = clblasColumnMajor;
		fUplo =  (params_.uplo == clblasUpper)? clblasLower : clblasUpper;
        fTrans = (params_.transA == clblasNoTrans)? clblasTrans : clblasNoTrans;

        if( params_.transA == clblasConjTrans )
            doConjugate( (AP_+params_.offa), (( params_.N * (params_.N + 1)) / 2) , 1, 1 );
    }


   time = getCurrentTime();
   clMath::blas::tpmv(order, fUplo,fTrans, params_.diag,
                    params_.N, AP_, params_.offa, X_, params_.offBX, params_.incx);
    time = getCurrentTime() - time;

#endif  // PERF_TEST_WITH_ACML

    return time;
}


template <typename ElemType> nano_time_t
TpmvPerformanceTest<ElemType>::clblasPerfSingle(void)
{
    nano_time_t time;
    cl_event event;
    cl_int status;
    cl_command_queue queue = base_->commandQueues()[0];
	size_t lenX = 1 + (params_.N-1) * abs(params_.incx);

    status = clEnqueueWriteBuffer(queue, mobjX_, CL_TRUE, 0,
                                  (lenX + params_.offBX )* sizeof(ElemType), backX_, 0, NULL, &event);
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

	DataType type;
    type = ( typeid(ElemType) == typeid(float))? TYPE_FLOAT:( typeid(ElemType) == typeid(double))? TYPE_DOUBLE:
										( typeid(ElemType) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;
	time = getCurrentTime();
#define TIMING
#ifdef TIMING
	clFinish( queue);

	int iter = 20;
	for ( int i = 1; i <= iter; i++)
	{
#endif
    status = (cl_int)clMath::clblas::tpmv(type, params_.order, params_.uplo,
        params_.transA, params_.diag, params_.N, mobjAP_, params_.offa,
        mobjX_, params_.offBX, params_.incx, scratchBuff,
        1, &queue, 0, NULL, &event);


    if (status != CL_SUCCESS) {
        cerr << "The CLBLAS TPMV function failed, status = " <<
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

	//printf("Time elapsed : %lu\n", time);
#endif

    return time;
}

} // namespace clMath

// strmv performance test
TEST_P(TPMV, stpmv)
{
    TestParams params;

    getParams(&params);
    TpmvPerformanceTest<float>::runInstance(FN_STPMV, &params);
}

// dtrmv performance test case
TEST_P(TPMV, dtpmv)
{
    TestParams params;

    getParams(&params);
    TpmvPerformanceTest<double>::runInstance(FN_DTPMV, &params);
}
// ctrmv performance test case
TEST_P(TPMV, ctpmv)
{
    TestParams params;

    getParams(&params);
    TpmvPerformanceTest<FloatComplex>::runInstance(FN_CTPMV, &params);
}
// ztrmv performance test case
TEST_P(TPMV, ztpmv)
{
    TestParams params;

    getParams(&params);
    TpmvPerformanceTest<DoubleComplex>::runInstance(FN_ZTPMV, &params);
}

