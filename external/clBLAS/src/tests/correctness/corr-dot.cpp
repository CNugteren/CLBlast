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
#include <dot.h>

static void
releaseMemObjects(cl_mem objX, cl_mem objY, cl_mem objDP, cl_mem objScratch)
{
    if(objX != NULL)
 	{
        clReleaseMemObject(objX);
	}
	if(objY != NULL)
    {
        clReleaseMemObject(objY);
	}
	if(objDP != NULL)
    {
        clReleaseMemObject(objDP);
    }
	if(objScratch != NULL)
    {
        clReleaseMemObject(objScratch);
    }
}

template <typename T> static void
deleteBuffers(T *blasX, T *blasY, T *blasDP, T *clblasDP)
{
	if(blasX != NULL)
	{
        delete[] blasX;
    }
	if(blasY != NULL)
	{
	    delete[] blasY;
	}
	if(clblasDP != NULL)
    {
        delete[] clblasDP;
    }
	if(blasDP != NULL)
    {
        delete(blasDP);
    }
}

template <typename T>
void
dotCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasX, *blasY, *clblasDP, *blasDP;
    cl_mem bufX, bufY, bufDP, scratchBuff;
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
	size_t lengthY = (1 + ((params->N -1) * abs(params->incy)));

    blasX 	= new T[lengthX + params->offBX ];
    blasY 	= new T[lengthY + params->offCY ];
	blasDP = new T[1];
    clblasDP = new T[1 + params->offa];

	if((blasX == NULL) || (blasY == NULL) || (clblasDP == NULL) || (blasDP == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(blasX, blasY, blasDP,  clblasDP);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);
    ::std::cerr << "Generating input data... ";

	randomVectors(params->N, (blasX + params->offBX), params->incx, (blasY + params->offCY), params->incy, true);
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(*blasX), 0, CL_MEM_READ_WRITE);
    bufY = base->createEnqueueBuffer(blasY, (lengthY + params->offCY)* sizeof(*blasY), 0, CL_MEM_READ_WRITE);
    bufDP = base->createEnqueueBuffer(NULL, (1 + params->offa) * sizeof(T), 0, CL_MEM_READ_WRITE);
	scratchBuff = base->createEnqueueBuffer(NULL, (lengthX * sizeof(T)), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xDOT routine... ";

	*blasDP  = ::clMath::blas::dot( params->N, blasX, params->offBX, params->incx, blasY, params->offCY, params->incy);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufX == NULL) || (bufY == NULL) || (bufDP == NULL) || (scratchBuff == NULL)) {
        releaseMemObjects(bufX, bufY, bufDP, scratchBuff);
        deleteBuffers<T>(blasX, blasY, blasDP, clblasDP);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xDOT routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE;

    // Should use bufXTemp as well
    err = (cl_int)::clMath::clblas::dot( type, params->N,  bufDP, params->offa, bufX,
    					params->offBX, params->incx, bufY, params->offCY, params->incy, scratchBuff, params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufY, bufDP, scratchBuff);
        deleteBuffers<T>(blasX, blasY, blasDP, clblasDP);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::DOT() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufY, bufDP, scratchBuff);
        deleteBuffers<T>(blasX, blasY, blasDP, clblasDP);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufDP, CL_TRUE, 0,
        (1 + params->offa) * sizeof(*clblasDP), clblasDP, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "DOT: Reading results failed...." << std::endl;
	}
    releaseMemObjects(bufX, bufY, bufDP, scratchBuff);

    compareMatrices<T>(clblasColumnMajor, 1 , 1, (blasDP), (clblasDP+params->offa), 1);
    deleteBuffers<T>(blasX, blasY, blasDP, clblasDP);
    delete[] events;
}

// Instantiate the test

TEST_P(DOT, sdot) {
    TestParams params;

    getParams(&params);
    dotCorrectnessTest<cl_float>(&params);
}

TEST_P(DOT, ddot) {
    TestParams params;

    getParams(&params);
    dotCorrectnessTest<cl_double>(&params);
}

TEST_P(DOT, cdotu) {
    TestParams params;

    getParams(&params);
    dotCorrectnessTest<FloatComplex>(&params);
}

TEST_P(DOT, zdotu) {
    TestParams params;

    getParams(&params);
    dotCorrectnessTest<DoubleComplex>(&params);
}
