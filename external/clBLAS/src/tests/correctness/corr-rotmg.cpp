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
#include <rotmg.h>
#include <matrix.h>
#include "delta.h"

static void
releaseMemObjects(cl_mem bufD1, cl_mem bufD2, cl_mem bufX, cl_mem bufY, cl_mem bufParam)
{
    if(bufD1 != NULL)
 	{
        clReleaseMemObject(bufD1);
	}
	if(bufD2 != NULL)
    {
        clReleaseMemObject(bufD2);
	}
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
deleteBuffers(T *D1, T *D2, T *X, T *Y, T *PARAM)
{
	if(D1 != NULL)
	{
        delete[] D1;
    }
	if(D2 != NULL)
	{
	    delete[] D2;
	}
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
}

template <typename T>
void
rotmgCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *D1, *D2, *X, *Y, *PARAM;
    T *back_D1, *back_D2, *back_X, *back_Y, *back_PARAM;
    T sflagParam;
    cl_mem bufD1, bufD2, bufX, bufY, bufParam;
    clMath::BlasBase *base;
    cl_event *events;
    cl_double deltaForType = 0.0;

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

    X 	    = new T[1 + params->offBX];
    Y 	    = new T[1 + params->offCY];
    D1 	    = new T[1 + params->offa];
    D2 	    = new T[1 + params->offb];
    PARAM   = new T[5 + params->offc]; //params always has 5 elements

    back_X 	    = new T[1 + params->offBX];
    back_Y 	    = new T[1 + params->offCY];
    back_D1 	= new T[1 + params->offa];
    back_D2 	= new T[1 + params->offb];
    back_PARAM  = new T[5 + params->offc]; //params always has 5 elements

	if((D1 == NULL) || (D2 == NULL) || (X == NULL) || (Y == NULL) || (PARAM == NULL) ||
        (back_D1 == NULL) || (back_D2 == NULL) ||(back_X == NULL) || (back_Y == NULL) || (back_PARAM == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(D1, D2, X, Y, PARAM);
        deleteBuffers<T>(back_D1, back_D2, back_X, back_Y, back_PARAM);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    //Filling random values for SA and SB. C & S are only for output sake
    randomRotmg( (D1 + params->offa), (D2 + params->offb),
                (X + params->offBX), (Y + params->offCY), (PARAM + params->offc) );

    sflagParam = convertMultiplier<T>(params->alpha);
    PARAM[params->offc] = sflagParam; // initializing first element

    memcpy(back_X, X, (1 + params->offBX)*sizeof(T));
    memcpy(back_Y, Y, (1 + params->offCY)*sizeof(T));
    memcpy(back_D1, D1, (1 + params->offa)*sizeof(T));
    memcpy(back_D2, D2, (1 + params->offb)*sizeof(T));
    memcpy(back_PARAM, PARAM, (params->offc + 5)*sizeof(T));

    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufD1 = base->createEnqueueBuffer(D1, (1 + params->offa) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufD2 = base->createEnqueueBuffer(D2, (1 + params->offb) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(X, (1 + params->offBX) * sizeof(T), 0, CL_MEM_READ_WRITE);
    bufY = base->createEnqueueBuffer(Y, (1 + params->offCY) * sizeof(T), 0, CL_MEM_READ_ONLY);
    bufParam  = base->createEnqueueBuffer(PARAM,  (5 + params->offc) * sizeof(T), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xROTMG routine... ";

	::clMath::blas::rotmg(back_D1, params->offa, back_D2, params->offb, back_X, params->offBX, back_Y, params->offCY,
                 back_PARAM, params->offc);
    ::std::cerr << "Done" << ::std::endl;

    // Hold X vector

    if ((bufD1 == NULL) || (bufD2 == NULL) || (bufX == NULL) || (bufY == NULL) || (bufParam == NULL))
    {
        releaseMemObjects(bufD1, bufD2, bufX, bufY, bufParam);
        deleteBuffers<T>(D1, D2, X, Y, PARAM);
        deleteBuffers<T>(back_D1, back_D2, back_X, back_Y, back_PARAM);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xROTMG routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float)) ? TYPE_FLOAT :
            TYPE_DOUBLE;

    err = (cl_int)::clMath::clblas::rotmg(  type, bufD1, params->offa, bufD2, params->offb, bufX, params->offBX,
                                            bufY, params->offCY, bufParam, params->offc,
                                            params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufD1, bufD2, bufX, bufY, bufParam);
        deleteBuffers<T>(D1, D2, X, Y, PARAM);
        deleteBuffers<T>(back_D1, back_D2, back_X, back_Y, back_PARAM);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::ROTMG() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS)
    {
        releaseMemObjects(bufD1, bufD2, bufX, bufY, bufParam);
        deleteBuffers<T>(D1, D2, X, Y, PARAM);
        deleteBuffers<T>(back_D1, back_D2, back_X, back_Y, back_PARAM);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufD1, CL_TRUE, 0,
        (1 + params->offa) * sizeof(T), D1, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufD2, CL_TRUE, 0,
        (1 + params->offb) * sizeof(T), D2, 0, NULL, NULL);

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        (1 + params->offBX) * sizeof(T), X, 0, NULL, NULL);

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufY, CL_TRUE, 0,
        (1 + params->offCY) * sizeof(T), Y, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufParam, CL_TRUE, 0,
        (5 + params->offc) * sizeof(T), PARAM, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		::std::cerr << "ROTMG: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufD1, bufD2, bufX, bufY, bufParam);

    deltaForType = DELTA_0<T>();

    #ifndef CORR_TEST_WITH_ACML
    // Acml doesn't store answer in D1, D2 and X1. So skipping those checks
        cl_double delta;
        delta = deltaForType * returnMax<T>(back_D1[params->offa]);
        compareValues<T>( (back_D1 + params->offa), (D1 + params->offa), delta);

        delta = deltaForType * returnMax<T>(back_D2[params->offb]);
        compareValues<T>( (back_D2 + params->offb), (D2 + params->offb), delta);

        delta = deltaForType * returnMax<T>(back_X[params->offBX]);
        compareValues<T>( (back_X + params->offBX), (X + params->offBX), delta);

        delta = deltaForType * returnMax<T>(back_Y[params->offCY]);
        compareValues<T>( (back_Y + params->offCY), (Y + params->offCY), delta);
    #endif

    // Creating delta array for PARAM array
    cl_double deltaArr[5];
    for(int i=0; i<5; i++) {
        deltaArr[i] = deltaForType * returnMax<T>(back_PARAM[i + (params->offc)]);
    }
    compareMatrices<T>(clblasColumnMajor, 5 , 1, (back_PARAM + params->offc), (PARAM + params->offc), 5, deltaArr);

    deleteBuffers<T>(D1, D2, X, Y, PARAM);
    deleteBuffers<T>(back_D1, back_D2, back_X, back_Y, back_PARAM);

    delete[] events;
}

// Instantiate the test
TEST_P(ROTMG, srotmg)
{
    TestParams params;

    getParams(&params);
    rotmgCorrectnessTest<cl_float>(&params);
}

TEST_P(ROTMG, drotmg)
{
    TestParams params;

    getParams(&params);
    rotmgCorrectnessTest<cl_double>(&params);
}
