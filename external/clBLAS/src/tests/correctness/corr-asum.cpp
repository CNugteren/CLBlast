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
#include <blas-internal.h>
#include <blas-wrapper.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <blas-random.h>
#include <asum.h>

static void
releaseMemObjects(cl_mem objX, cl_mem objAsum, cl_mem objScratch)
{
    if(objX != NULL)
 	{
        clReleaseMemObject(objX);
	}
	if(objAsum != NULL)
    {
        clReleaseMemObject(objAsum);
    }
	if(objScratch != NULL)
    {
        clReleaseMemObject(objScratch);
    }
}

template <typename T> static void
deleteBuffers(T *blasX, T *blasAsum=NULL, T *clblasAsum=NULL)
{
	if(blasX != NULL)
	{
        delete[] blasX;
    }
	if(clblasAsum != NULL)
    {
        delete[] clblasAsum;
    }
	if(blasAsum != NULL)
    {
        delete(blasAsum);
    }
}

template <typename T1, typename T2>
void
asumCorrectnessTest(TestParams *params)
{
    cl_int err;
    T1 *blasX;
    T2 *clblasAsum, *blasAsum;
    cl_mem bufX, bufAsum, scratchBuff;
    clMath::BlasBase *base;
    cl_event *events;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T1) == typeid(cl_double) ||
         typeid(T1) == typeid(DoubleComplex)) &&
        !base->isDevSupportDoublePrecision()) {

        std::cerr << ">> WARNING: The target device doesn't support native "
                     "double precision floating point arithmetic" <<
                     std::endl << ">> Test skipped" << std::endl;
        SUCCEED();
        return;
    }

	printf("number of command queues : %d\n\n", params->numCommandQueues);

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));

    blasX 	= new T1[lengthX + params->offBX ];
	blasAsum = new T2[1];
    clblasAsum = new T2[1 + params->offa];

	if((blasX == NULL) || (clblasAsum == NULL) || (blasAsum == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasAsum,  clblasAsum);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);
    ::std::cerr << "Generating input data... ";

	randomVectors<T1>(params->N, (blasX + params->offBX), params->incx, (T1*)NULL, 0, true);
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(*blasX), 0, CL_MEM_READ_WRITE);
    bufAsum = base->createEnqueueBuffer(NULL, (1 + params->offa) * sizeof(T2), 0, CL_MEM_READ_WRITE);
	scratchBuff = base->createEnqueueBuffer(NULL, (lengthX * sizeof(T1)), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xASUM routine... ";

	*blasAsum = ::clMath::blas::asum( params->N, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufX == NULL) || (bufAsum == NULL) || (scratchBuff == NULL)) {
        releaseMemObjects(bufX, bufAsum, scratchBuff);
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasAsum, clblasAsum);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xASUM routine... ";

    DataType type;
    type = ( typeid(T1) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T1) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T1) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE;

    // Should use bufXTemp as well
    err = (cl_int)::clMath::clblas::asum( type, params->N,  bufAsum, params->offa, bufX,
    					params->offBX, params->incx, scratchBuff, params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufAsum, scratchBuff);
        deleteBuffers<T1>(blasX );
        deleteBuffers<T2>(blasAsum, clblasAsum);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::ASUM() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufAsum, scratchBuff);
        deleteBuffers<T1>(blasX );
        deleteBuffers<T2>(blasAsum, clblasAsum);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufAsum, CL_TRUE, 0,
        (1 + params->offa) * sizeof(*clblasAsum), clblasAsum, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "ASUM: Reading results failed...." << std::endl;
	}
    releaseMemObjects(bufX, bufAsum, scratchBuff);

    compareMatrices<T2>(clblasColumnMajor, 1 , 1, (blasAsum), (clblasAsum+params->offa), 1);
    deleteBuffers<T1>(blasX);
    deleteBuffers<T2>(blasAsum, clblasAsum);
    delete[] events;
}

// Instantiate the test

TEST_P(ASUM, sasum) {
    TestParams params;

    getParams(&params);
    asumCorrectnessTest<cl_float, cl_float>(&params);
}

TEST_P(ASUM, dasum) {
    TestParams params;

    getParams(&params);
    asumCorrectnessTest<cl_double, cl_double>(&params);
}

TEST_P(ASUM, scasum) {
    TestParams params;

    getParams(&params);
    asumCorrectnessTest<FloatComplex, cl_float>(&params);
}

TEST_P(ASUM, dzasum) {
    TestParams params;

    getParams(&params);
    asumCorrectnessTest<DoubleComplex, cl_double>(&params);
}
