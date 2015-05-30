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
#include <spr.h>

static void
releaseMemObjects(cl_mem objAP, cl_mem objX)
{
	if(objAP != NULL)
	{
    clReleaseMemObject(objAP);
	}
	if(objX != NULL)
	{
    clReleaseMemObject(objX);
}

}

template <typename T> static void
deleteBuffers(T *blasAP, T *clblasAP, T *X)
{
	if(blasAP != NULL)
	{
    delete[] blasAP;
	}
	if(clblasAP != NULL)
	{
    delete[] clblasAP;
	}
	if(X != NULL)
	{
	delete[] X;
}
}

template <typename T>
void
sprCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasAP, *clblasAP, *X;
//	T *tempA;
    cl_mem bufAP, bufX;
    clMath::BlasBase *base;
    cl_event *events;
	bool useAlpha;
	T alpha;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(cl_double)) &&
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

    size_t lengthAP = ( ( params->N*( params->N + 1 ) )/2 );
    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));

    blasAP 		= new T[lengthAP + params->offa];
    clblasAP 	= new T[lengthAP + params->offa];
    X		 	= new T[lengthX + params->offBX];
//	tempA 		= new T[lengthA + params->offa ];

    srand(params->seed);

    ::std::cerr << "Generating input data... ";


	memset(blasAP, -1, (lengthAP + params->offa));
	memset(clblasAP, -1, (lengthAP + params->offa));
	memset(X, -1, (lengthX + params->offBX));

	alpha =  convertMultiplier<T>(params->alpha);
	useAlpha = true;

	#ifdef DEBUG_SPR
	printf("ALPHA in CORR_SPR.CPP %f\n", alpha);
	#endif

	if((blasAP == NULL) || (X == NULL) || (clblasAP == NULL))
    {
        ::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
		deleteBuffers<T>(blasAP, clblasAP, X);
        delete[] events;
        SUCCEED();
        return;
    }

	randomSyrMatrices<T>(params->order, params->uplo, params->N, useAlpha, &alpha,
						(blasAP + params->offa), 0, (X + params->offBX), params->incx);

    memcpy(clblasAP, blasAP, (lengthAP + params->offa)* sizeof(*blasAP));

	::std::cerr << "Done" << ::std::endl;

    bufAP = base->createEnqueueBuffer(clblasAP, (lengthAP + params->offa) * sizeof(*clblasAP), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX)* sizeof(*X), 0, CL_MEM_READ_ONLY);

    ::std::cerr << "Calling reference xSPR routine... ";

	clblasOrder order;
    clblasUplo fUplo;
	order = params->order;
    fUplo = params->uplo;


	if (order != clblasColumnMajor)
    {

        order = clblasColumnMajor;
        fUplo =  (params->uplo == clblasUpper)? clblasLower : clblasUpper;

        if( params->transA == clblasConjTrans )
            doConjugate( (blasAP +params->offa), (( params->N * (params->N + 1)) / 2) , 1, 1 );

    }

    clMath::blas::spr( clblasColumnMajor, fUplo, params->N, alpha, X, params->offBX, params->incx, blasAP, params->offa);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufAP == NULL) || (bufX == NULL) ) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufAP, bufX);
        deleteBuffers<T>(blasAP, clblasAP, X);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSPR routine... ";

    err = (cl_int)::clMath::clblas::spr( params->order, params->uplo, params->N, alpha,
						bufX, params->offBX, params->incx, bufAP, params->offa,
						params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX);
        deleteBuffers<T>(blasAP, clblasAP, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SYR() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX);
        deleteBuffers<T>(blasAP, clblasAP, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufAP, CL_TRUE, 0,
        (lengthAP + params->offa) * sizeof(*clblasAP), clblasAP, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "SPR: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufAP, bufX);
	printf("Comparing the results\n");
	compareMatrices<T>(clblasColumnMajor, lengthAP , 1, (blasAP + params->offa), (clblasAP + params->offa),
                       lengthAP);

	deleteBuffers<T>(blasAP, clblasAP, X);
    delete[] events;
}

// Instantiate the test

TEST_P(SPR, sspr) {
    TestParams params;

    getParams(&params);
    sprCorrectnessTest<cl_float>(&params);
}

TEST_P(SPR, dspr) {
    TestParams params;

    getParams(&params);
    sprCorrectnessTest<cl_double>(&params);
}
