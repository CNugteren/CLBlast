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
#include <swap.h>

static void
releaseMemObjects(cl_mem objX,  cl_mem objY)
{
  if(objX != NULL)
  {
  	clReleaseMemObject(objX);
  }
  if(objY != NULL)
  {
    clReleaseMemObject(objY);
  }
}

template <typename T> static void
deleteBuffers(T *X, T *Y,  T *blasX, T *blasY)
{
    if(X != NULL)
    {
        delete[] X;
    }
	if(Y != NULL)
    {
        delete[] Y;
    }
    if(blasX != NULL)
	{
        delete[] blasX;
	}
	if(blasY != NULL)
	{
        delete[] blasY;
	}
}

template <typename T>
void
swapCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *X, *Y, *blasX, *blasY;
    cl_mem bufX, bufY;
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

    X 		= new T[lengthX + params->offBX ];
    Y 		= new T[lengthY + params->offCY ];
    blasX 	= new T[lengthX + params->offBX ];
    blasY	= new T[lengthY + params->offCY ];

	if((X == NULL) || (blasX == NULL) || (Y == NULL) || (blasY == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(X, Y, blasX, blasY);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    // Populate A and blasX
    randomVectors(params->N, (X+params->offBX), params->incx, (Y+params->offCY), params->incy);

	memcpy(blasX, X, (lengthX + params->offBX) * sizeof(T));
	memcpy(blasY, Y, (lengthY + params->offCY) * sizeof(T));

    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX)* sizeof(T), 0, CL_MEM_READ_WRITE);
    bufY = base->createEnqueueBuffer(Y, (lengthY + params->offCY)* sizeof(T), 0, CL_MEM_READ_WRITE);

	if ((bufX == NULL) || (bufY == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufX, bufY);
        deleteBuffers<T>(X, Y, blasX, blasY);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling reference xSWAP routine... ";

	::clMath::blas::swap( params->N, blasX, params->offBX, params->incx,
						 blasY, params->offCY, params->incy);
    ::std::cerr << "Done" << ::std::endl;


    ::std::cerr << "Calling clblas xSWAP routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT : (( typeid(T) == typeid(cl_double))? TYPE_DOUBLE: (( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE));

    err = (cl_int)::clMath::clblas::swap( type, params->N, bufX, params->offBX, params->incx, bufY, params->offCY, params->incy,
										  params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufY);
        deleteBuffers<T>(X, Y, blasX, blasY);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SWAP() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufY);
        deleteBuffers<T>(X, Y, blasX, blasY);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        (lengthX + params->offBX) * sizeof(T), X, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufY, CL_TRUE, 0,
        (lengthY + params->offCY) * sizeof(T), Y, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "SWAP: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufX, bufY);


    compareMatrices<T>(clblasColumnMajor, lengthX , 1, (blasX + params->offBX), (X + params->offBX), lengthX);
    compareMatrices<T>(clblasColumnMajor, lengthY , 1, (blasY + params->offCY), (Y + params->offCY), lengthY);
    deleteBuffers<T>(X, Y, blasX, blasY);
    delete[] events;
}

// Instantiate the test

TEST_P(SWAPXY, sswap) {
    TestParams params;

    getParams(&params);
    swapCorrectnessTest<cl_float>(&params);
}

TEST_P(SWAPXY, dswap) {
    TestParams params;

    getParams(&params);
    swapCorrectnessTest<cl_double>(&params);
}

TEST_P(SWAPXY, cswap) {
    TestParams params;

    getParams(&params);
    swapCorrectnessTest<FloatComplex>(&params);
}

TEST_P(SWAPXY, zswap) {
    TestParams params;

    getParams(&params);
    swapCorrectnessTest<DoubleComplex>(&params);
}
