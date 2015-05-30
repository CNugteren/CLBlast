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
#include <nrm2.h>
#include "delta.h"

static void
releaseMemObjects(cl_mem objX, cl_mem objNrm2, cl_mem objScratch)
{
    if(objX != NULL)
 	{
        clReleaseMemObject(objX);
	}
	if(objNrm2 != NULL)
    {
        clReleaseMemObject(objNrm2);
    }
	if(objScratch != NULL)
    {
        clReleaseMemObject(objScratch);
    }
}

template <typename T> static void
deleteBuffers(T *blasX, T *blasNRM2=NULL, T *clblasNRM2=NULL)
{
	if(blasX != NULL)
	{
        delete[] blasX;
    }
	if(clblasNRM2 != NULL)
    {
        delete[] clblasNRM2;
    }
	if(blasNRM2 != NULL)
    {
        delete(blasNRM2);
    }
}

template <typename T1, typename T2>
void
nrm2CorrectnessTest(TestParams *params)
{
    cl_int err;
    T1 *blasX;
    T2 *clblasNRM2, *blasNRM2;
    cl_mem bufX, bufNRM2, scratchBuff;
    clMath::BlasBase *base;
    cl_event *events;
    cl_double deltaForType = 0.0;

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
	blasNRM2 = new T2[1];
    clblasNRM2 = new T2[1 + params->offa];

	if((blasX == NULL) || (clblasNRM2 == NULL) || (blasNRM2 == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasNRM2,  clblasNRM2);
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
    bufNRM2 = base->createEnqueueBuffer(NULL, (1 + params->offa) * sizeof(T2), 0, CL_MEM_READ_WRITE);
	scratchBuff = base->createEnqueueBuffer(NULL, (lengthX * 2 * sizeof(T1)), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xNRM2 routine... ";

	*blasNRM2  = ::clMath::blas::nrm2( params->N, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufX == NULL) || (bufNRM2 == NULL) || (scratchBuff == NULL)) {
        releaseMemObjects(bufX, bufNRM2, scratchBuff);
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasNRM2,  clblasNRM2);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xNRM2 routine... ";

    DataType type;
    type = ( typeid(T1) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T1) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T1) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE;

    err = (cl_int)::clMath::clblas::nrm2( type, params->N,  bufNRM2, params->offa, bufX,
    					params->offBX, params->incx, scratchBuff, params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufNRM2, scratchBuff);
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasNRM2,  clblasNRM2);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::NRM2() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues, base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX, bufNRM2, scratchBuff);
        deleteBuffers<T1>(blasX);
        deleteBuffers<T2>(blasNRM2,  clblasNRM2);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufNRM2, CL_TRUE, 0,
            (1 + params->offa) * sizeof(*clblasNRM2), clblasNRM2, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		::std::cerr << "NRM2: Reading results failed...." << std::endl;
	}
    releaseMemObjects(bufX, bufNRM2, scratchBuff);

    deltaForType = DELTA_0<T1>();

    // Since every element of X encounters a division, delta would be sum of deltas for every element in X
    cl_double delta = 0;
    for(unsigned int i=0; i<(params->N); i++) {
        delta += deltaForType * returnMax<T1>(blasX[params->offBX + i]);
    }
    compareValues<T2>( (blasNRM2), (clblasNRM2+params->offa), delta);

    deleteBuffers<T1>(blasX);
    deleteBuffers<T2>(blasNRM2,  clblasNRM2);
    delete[] events;
}

// Instantiate the test

TEST_P(NRM2, snrm2) {
    TestParams params;

    getParams(&params);
    nrm2CorrectnessTest<cl_float, cl_float>(&params);
}

TEST_P(NRM2, dnrm2) {
    TestParams params;

    getParams(&params);
    nrm2CorrectnessTest<cl_double, cl_double>(&params);
}

TEST_P(NRM2, scnrm2) {
    TestParams params;

    getParams(&params);
    nrm2CorrectnessTest<FloatComplex, cl_float>(&params);
}

TEST_P(NRM2, dznrm2) {
    TestParams params;

    getParams(&params);
    nrm2CorrectnessTest<DoubleComplex, cl_double>(&params);
}
