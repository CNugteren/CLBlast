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
#include <rot.h>
#include <matrix.h>
//#include "delta.h"

static void
releaseMemObjects(cl_mem bufX, cl_mem bufY)
{
    if(bufX != NULL)
 	{
        clReleaseMemObject(bufX);
	}
	if(bufY != NULL)
    {
        clReleaseMemObject(bufY);
	}
}

template <typename T> static void
deleteBuffers(T *X, T *Y, T *back_X, T *back_Y)
{
	if(X != NULL)
	{
        delete[] X;
    }
	if(Y != NULL)
	{
	    delete[] Y;
	}
    if(back_X != NULL)
	{
        delete[] back_X;
    }
	if(back_Y != NULL)
	{
	    delete[] back_Y;
	}
}

template <typename T>
void
rotCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *X, *Y, *back_X, *back_Y;
    T alpha, beta;
    cl_mem bufX, bufY;
    clMath::BlasBase *base;
    cl_event *events;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(cl_double) || typeid(T) == typeid(DoubleComplex)) &&
        !base->isDevSupportDoublePrecision())
    {
        std::cerr << ">> WARNING: The target device doesn't support native "
                     "double precision floating point arithmetic" <<
                     std::endl << ">> Test skipped" << std::endl;
        SUCCEED();
        return;
    }

	printf("number of command queues : %d\n\n", params->numCommandQueues);

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    size_t lengthx = 1 + (params->N - 1) * abs(params->incx);
    size_t lengthy = 1 + (params->N - 1) * abs(params->incy);

    X 	= new T[lengthx + params->offa];
    Y 	= new T[lengthy + params->offb];

    back_X 	= new T[lengthx + params->offa];
    back_Y 	= new T[lengthy + params->offb];

	if((X == NULL) || (Y == NULL) ||
        (back_X == NULL) || (back_Y == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(X, Y, back_X, back_Y);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    //Filling random values for SA and SB. C & S are only for output sake
    randomVectors(params->N, (X + params->offa), params->incx, (Y+params->offb), params->incy);

    alpha = convertMultiplier<T>(params->alpha);
	beta = convertMultiplier<T>(params->beta);

    memcpy(back_X, X, (lengthx + params->offa) * sizeof(T));
    memcpy(back_Y, Y, (lengthy + params->offb) * sizeof(T));

    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(X, (lengthx + params->offa) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufY = base->createEnqueueBuffer(Y, (lengthy + params->offb) * sizeof(T), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xROT routine... ";

	::clMath::blas::rot(params->N, back_X, params->offa, params->incx, back_Y, params->offb, params->incy,
                 alpha, beta);
    ::std::cerr << "Done" << ::std::endl;

    // Hold X vector

    if ((bufX == NULL) || (bufY == NULL))
    {
        releaseMemObjects(bufX, bufY);
        deleteBuffers(X, Y, back_X, back_Y);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xROT routine... ";


    err = (cl_int)::clMath::clblas::rot( params->N, bufX, params->offa, params->incx, bufY, params->offb, params->incy,
                              alpha, beta, params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufX, bufY);
        deleteBuffers(X, Y, back_X, back_Y);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::ROT() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufX, bufY);
        deleteBuffers(X, Y,  back_X, back_Y );
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        (lengthx + params->offa) * sizeof(T), X, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufY, CL_TRUE, 0,
        (lengthy + params->offb) * sizeof(T), Y, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		::std::cerr << "ROT: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufX, bufY);


    compareMatrices<T>(clblasRowMajor, lengthx , 1, (back_X + params->offa), (X + params->offa), 1);
    compareMatrices<T>(clblasRowMajor, lengthy , 1, (back_Y + params->offb), (Y + params->offb), 1);

    deleteBuffers<T>(X, Y, back_X, back_Y);
    delete[] events;
}

// Instantiate the test
TEST_P(ROT, srot)
{
    TestParams params;

    getParams(&params);
    rotCorrectnessTest<cl_float>(&params);
}

TEST_P(ROT, drot)
{
    TestParams params;

    getParams(&params);
    rotCorrectnessTest<cl_double>(&params);
}

TEST_P(ROT, csrot)
{
    TestParams params;

    getParams(&params);
    rotCorrectnessTest<FloatComplex>(&params);
}

TEST_P(ROT, zdrot)
{
    TestParams params;

    getParams(&params);
    rotCorrectnessTest<DoubleComplex>(&params);
}


