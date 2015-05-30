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
#include <rotg.h>
#include <matrix.h>
#include "delta.h"

static void
releaseMemObjects(cl_mem bufSA, cl_mem bufSB, cl_mem bufC, cl_mem bufS)
{
    if(bufSA != NULL)
 	{
        clReleaseMemObject(bufSA);
	}
	if(bufSB != NULL)
    {
        clReleaseMemObject(bufSB);
	}
    if(bufC != NULL)
 	{
        clReleaseMemObject(bufC);
	}
	if(bufS != NULL)
    {
        clReleaseMemObject(bufS);
	}
}

template <typename T> static void
deleteBuffers(T *A, T *B, T *C=NULL, T *D=NULL, T *E=NULL, T *F=NULL)
{
	if(A != NULL)
	{
        delete[] A;
    }
	if(B != NULL)
	{
	    delete[] B;
	}
    if(C != NULL)
	{
        delete[] C;
    }
	if(D != NULL)
	{
	    delete[] D;
	}
    if(E != NULL)
	{
        delete[] E;
    }
	if(F != NULL)
	{
	    delete[] F;
	}
}

// type T1 indicates the basic type,
// while T2 indicates type of buffer C. C is not complex for complex types
template <typename T1, typename T2>
void
rotgCorrectnessTest(TestParams *params)
{
    cl_int err;
    T1 *SA, *SB, *S, *back_SA, *back_SB, *back_S;
    T2 *C, *back_C;
    cl_mem bufSA, bufSB, bufC, bufS;
    clMath::BlasBase *base;
    cl_event *events;
    cl_double deltaForType = 0.0;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T1) == typeid(cl_double) ||
         typeid(T1) == typeid(DoubleComplex)) &&
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

    size_t length = 1;//only one element need to be accessed always

    SA 	= new T1[length + params->offBX ];
    SB 	= new T1[length + params->offCY ];
    C   = new T2[length + params->offa ];
    S   = new T1[length + params->offb ];

    back_SA 	= new T1[length + params->offBX ];
    back_SB 	= new T1[length + params->offCY ];
    back_C      = new T2[length + params->offa ];
    back_S      = new T1[length + params->offb ];

	if((SA == NULL) || (SB == NULL) || (C == NULL) || (S == NULL) ||
        (back_SA == NULL) || (back_SB == NULL) || (back_C == NULL) || (back_S == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T1>(SA, SB, S, back_SA, back_SB, back_S);
        deleteBuffers<T2>(C, back_C);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    //Filling random values for SA and SB. C & S are only for output sake
    randomVectors(1, (SA+params->offBX), 1, (SB+params->offCY), 1);
    S[params->offb] =  back_S[params->offb] = ZERO<T1>();
    C[params->offa] = back_C[params->offa] = ZERO<T2>();

    back_SA[params->offBX] = SA[params->offBX];
    back_SB[params->offCY] = SB[params->offCY];
    ::std::cerr << "Done" << ::std::endl;

    //printing the inputs, as they change after processing
    ::std::cerr << "A = ";
    printElement<T1>(SA[params->offBX]);
    ::std::cerr << "\tB = ";
    printElement<T1>(SB[params->offCY]);
    ::std::cerr << "\tC = ";
    printElement<T2>(C[params->offa]);
    ::std::cerr << "\tS = ";
    printElement<T1>(S[params->offb]);
    ::std::cout << std::endl << std::endl;

	// Allocate buffers
    bufSA = base->createEnqueueBuffer(SA, (length + params->offBX) * sizeof(T1), 0, CL_MEM_READ_WRITE);
    bufSB = base->createEnqueueBuffer(SB, (length + params->offCY) * sizeof(T1), 0, CL_MEM_READ_WRITE);
    bufC  = base->createEnqueueBuffer(C,  (length + params->offa ) * sizeof(T2), 0, CL_MEM_WRITE_ONLY);
    bufS  = base->createEnqueueBuffer(S,  (length + params->offb ) * sizeof(T1), 0, CL_MEM_WRITE_ONLY);

    ::std::cerr << "Calling reference xROTG routine... ";

	::clMath::blas::rotg(back_SA, params->offBX, back_SB, params->offCY, back_C, params->offa, back_S, params->offb);
    ::std::cerr << "Done" << ::std::endl;

    // Hold X vector

    if ((bufSA == NULL) || (bufSB == NULL) || (bufC == NULL) || (bufS == NULL))
    {
        releaseMemObjects(bufSA, bufSB, bufC, bufS);
        deleteBuffers<T1>(SA, SB, S, back_SA, back_SB, back_S);
        deleteBuffers<T2>(C, back_C);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xROTG routine... ";

    DataType type;
    type = ( typeid(T1) == typeid(cl_float)) ? TYPE_FLOAT :
           ( typeid(T1) == typeid(cl_double)) ? TYPE_DOUBLE:
           ( typeid(T1) == typeid(cl_float2)) ? TYPE_COMPLEX_FLOAT:
            TYPE_COMPLEX_DOUBLE;

    err = (cl_int)::clMath::clblas::rotg( type, bufSA, params->offBX, bufSB, params->offCY,
                                         bufC, params->offa, bufS, params->offb,
                                         params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufSA, bufSB, bufC, bufS);
        deleteBuffers<T1>(SA, SB, S, back_SA, back_SB, back_S);
        deleteBuffers<T2>(C, back_C);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::ROTG() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufSA, bufSB, bufC, bufS);
        deleteBuffers<T1>(SA, SB, S, back_SA, back_SB, back_S);
        deleteBuffers<T2>(C, back_C);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufSA, CL_TRUE, 0,
        (length + params->offBX) * sizeof(T1), SA, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufSB, CL_TRUE, 0,
        (length + params->offCY) * sizeof(T1), SB, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufC, CL_TRUE, 0,
        (length + params->offa) * sizeof(T2), C, 0, NULL, NULL);

    err |= clEnqueueReadBuffer(base->commandQueues()[0], bufS, CL_TRUE, 0,
        (length + params->offb) * sizeof(T1), S, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		::std::cerr << "ROTG: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufSA, bufSB, bufC, bufS);

    deltaForType = DELTA_0<T1>();
    cl_double delta;

    delta = deltaForType * returnMax<T1>(back_SA[params->offBX]);
    compareValues<T1>( (back_SA + params->offBX), (SA + params->offBX), delta);

    delta = deltaForType * returnMax<T1>(back_SB[params->offCY]);
    compareValues<T1>( (back_SB + params->offCY), (SB + params->offCY), delta);

    delta = deltaForType * returnMax<T2>(back_C[params->offa]);
    compareValues<T2>( (back_C + params->offa), (C + params->offa), delta);

    delta = deltaForType * returnMax<T1>(back_S[params->offb]);
    compareValues<T1>( (back_S + params->offb), (S + params->offb), delta);

    deleteBuffers<T1>(SA, SB, S, back_SA, back_SB, back_S);
    deleteBuffers<T2>(C, back_C);
    delete[] events;
}

// Instantiate the test

TEST_P(ROTG, srotg) {
    TestParams params;

    getParams(&params);
    rotgCorrectnessTest<cl_float, cl_float>(&params);
}

TEST_P(ROTG, drotg) {
    TestParams params;

    getParams(&params);
    rotgCorrectnessTest<cl_double, cl_double>(&params);
}

TEST_P(ROTG, crotg) {
    TestParams params;

    getParams(&params);
    rotgCorrectnessTest<FloatComplex, cl_float>(&params);
}

TEST_P(ROTG, zrotg) {
    TestParams params;

    getParams(&params);
    rotgCorrectnessTest<DoubleComplex, cl_double>(&params);
}

