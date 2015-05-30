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
#include <iamax.h>

static void
releaseMemObjects(cl_mem objX, cl_mem objiAmax, cl_mem objScratch)
{
    if(objX != NULL)
 	{
        clReleaseMemObject(objX);
	}
	if(objiAmax != NULL)
    {
        clReleaseMemObject(objiAmax);
    }
	if(objScratch != NULL)
    {
        clReleaseMemObject(objScratch);
    }
}

template <typename T> static void
deleteBuffers(T *blasX, int *blasiAmax=NULL, int *clblasiAmax=NULL)
{
	if(blasX != NULL)
	{
        delete[] blasX;
    }
	if(clblasiAmax != NULL)
    {
        delete[] clblasiAmax;
    }
	if(blasiAmax != NULL)
    {
        delete(blasiAmax);
    }
}

template <typename T>
void
iamaxCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasX;
    int *clblasiAmax, *blasiAmax;
    cl_mem bufX, bufiAmax, scratchBuff;
    clMath::BlasBase *base;
    cl_event *events;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(cl_double) ||
         typeid(T) == typeid(DoubleComplex)) &&
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

    blasX 	= new T[lengthX + params->offBX ];
	blasiAmax = new int[1];
    clblasiAmax = new int[1 + params->offa];

	if((blasX == NULL) || (clblasiAmax == NULL) || (blasiAmax == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(blasX, blasiAmax, clblasiAmax);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);
    ::std::cerr << "Generating input data... ";

	randomVectors<T>(params->N, (blasX + params->offBX), params->incx, NULL, 0);
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(T), 0, CL_MEM_READ_ONLY);
    bufiAmax = base->createEnqueueBuffer(NULL, (1 + params->offa) * sizeof(int), 0, CL_MEM_READ_WRITE);
	scratchBuff = base->createEnqueueBuffer(NULL, (2 * lengthX * sizeof(T)), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xiAMAX routine... ";

	*blasiAmax = ::clMath::blas::iamax( params->N, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufX == NULL) || (bufiAmax == NULL) || (scratchBuff == NULL)) {
        releaseMemObjects(bufX, bufiAmax, scratchBuff);
        deleteBuffers<T>(blasX, blasiAmax, clblasiAmax);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xiAMAX routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE;

    // Should use bufXTemp as well
    err = (cl_int)::clMath::clblas::iamax( type, params->N, bufiAmax, params->offa,
                                           bufX, params->offBX, params->incx, scratchBuff,
                                            params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufiAmax, scratchBuff);
        deleteBuffers<T>(blasX, blasiAmax, clblasiAmax);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::iAMAX() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufiAmax, scratchBuff);
        deleteBuffers<T>(blasX, blasiAmax, clblasiAmax);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufiAmax, CL_TRUE, 0,
        (1 + params->offa) * sizeof(*clblasiAmax), clblasiAmax, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "iAMAX: Reading results failed...." << std::endl;
	}

    compareValues<int>((blasiAmax), (clblasiAmax+params->offa), 0);
    releaseMemObjects(bufX, bufiAmax, scratchBuff);
    deleteBuffers<T>(blasX, blasiAmax, clblasiAmax);
    delete[] events;
}

// Instantiate the test

TEST_P(iAMAX, isamax) {
    TestParams params;

    getParams(&params);
    iamaxCorrectnessTest<cl_float>(&params);
}

TEST_P(iAMAX, idamax) {
    TestParams params;

    getParams(&params);
    iamaxCorrectnessTest<cl_double>(&params);
}

TEST_P(iAMAX, icamax) {
    TestParams params;

    getParams(&params);
    iamaxCorrectnessTest<FloatComplex>(&params);
}

TEST_P(iAMAX, izamax) {
    TestParams params;

    getParams(&params);
    iamaxCorrectnessTest<DoubleComplex>(&params);
}
