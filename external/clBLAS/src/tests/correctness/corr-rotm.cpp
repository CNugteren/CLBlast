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
#include <rotm.h>
#include <matrix.h>

static void
releaseMemObjects(cl_mem bufX, cl_mem bufY, cl_mem bufParam)
{
    if(bufX != NULL)
 	{
        clReleaseMemObject(bufX);
	}
	if(bufY != NULL)
    {
        clReleaseMemObject(bufY);
	}
    if(bufParam != NULL)
 	{
        clReleaseMemObject(bufParam);
	}
}

template <typename T> static void
deleteBuffers(T *X, T *Y, T *PARAM, T *back_X, T *back_Y, T *back_PARAM)
{
	if(X != NULL)
	{
        delete[] X;
    }
	if(Y != NULL)
	{
	    delete[] Y;
	}
    if(PARAM != NULL)
	{
        delete[] PARAM;
    }
    if(back_X != NULL)
	{
        delete[] back_X;
    }
	if(back_Y != NULL)
	{
	    delete[] back_Y;
	}
    if(back_PARAM != NULL)
	{
        delete[] back_PARAM;
    }
}

template <typename T>
void
rotmCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *X, *Y, *back_X, *back_Y;
    T *PARAM, *back_PARAM;
    T sflagParam;
    cl_mem bufX, bufY, bufParam;
    clMath::BlasBase *base;
    cl_event *events;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(cl_double)) &&
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
    PARAM   = new T[5 + params->offc]; //params always has 5 elements

    back_X 	= new T[lengthx + params->offa];
    back_Y 	= new T[lengthy + params->offb];
    back_PARAM   = new T[5 + params->offc]; //params always has 5 elements

	if((X == NULL) || (Y == NULL) || (PARAM == NULL) ||
        (back_X == NULL) || (back_Y == NULL) || (back_PARAM == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(X, Y, PARAM, back_X, back_Y, back_PARAM);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    randomVectors(params->N, (X + params->offa), params->incx, (Y+params->offb), params->incy);
    randomVectors(4, (PARAM + params->offc + 1), 1); //1st element is initialized separately

    sflagParam = convertMultiplier<T>(params->alpha);
    PARAM[params->offc] = sflagParam; // initializing first element

    memcpy(back_X, X, (lengthx + params->offa)*sizeof(T));
    memcpy(back_Y, Y, (lengthy + params->offb)*sizeof(T));
    memcpy(back_PARAM, PARAM, (params->offc + 5)*sizeof(T));

    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufX = base->createEnqueueBuffer(X, (lengthx + params->offa) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufY = base->createEnqueueBuffer(Y, (lengthy + params->offb) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufParam  = base->createEnqueueBuffer(PARAM,  (5 + params->offc) * sizeof(T), 0, CL_MEM_READ_ONLY);

    ::std::cerr << "Calling reference xROTM routine... ";

	::clMath::blas::rotm(params->N, back_X, params->offa, params->incx, back_Y, params->offb, params->incy,
                 back_PARAM, params->offc);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufX == NULL) || (bufY == NULL) || (bufParam == NULL))
    {
        releaseMemObjects(bufX, bufY, bufParam);
        deleteBuffers(X, Y, PARAM, back_X, back_Y, back_PARAM);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xROTM routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float)) ? TYPE_FLOAT :
            TYPE_DOUBLE;

    err = (cl_int)::clMath::clblas::rotm( type, params->N, bufX, params->offa, params->incx, bufY, params->offb, params->incy,
                              bufParam, params->offc, params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufX, bufY, bufParam);
        deleteBuffers(X, Y, PARAM, back_X, back_Y, back_PARAM);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::ROTM() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufX, bufY, bufParam);
        deleteBuffers(X, Y, PARAM, back_X, back_Y, back_PARAM);
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
		::std::cerr << "ROTM: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufX, bufY, bufParam);

    compareMatrices<T>(clblasColumnMajor, lengthx , 1, (back_X + params->offa), (X + params->offa), lengthx);
    compareMatrices<T>(clblasColumnMajor, lengthy , 1, (back_Y + params->offb), (Y + params->offb), lengthy);

    deleteBuffers<T>(X, Y, PARAM, back_X, back_Y, back_PARAM);
    delete[] events;
}

// Instantiate the test
TEST_P(ROTM, srotm)
{
    TestParams params;

    getParams(&params);
    rotmCorrectnessTest<cl_float>(&params);
}

TEST_P(ROTM, drotm)
{
    TestParams params;

    getParams(&params);
    rotmCorrectnessTest<cl_double>(&params);
}
