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
#include <syr.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objX)
{
	if(objA != NULL)
	{
    clReleaseMemObject(objA);
	}
	if(objX != NULL)
	{
    clReleaseMemObject(objX);
}

}

template <typename T> static void
deleteBuffers(T *blasA, T *clblasA, T *X)
{
	if(blasA != NULL)
	{
    delete[] blasA;
	}
	if(clblasA != NULL)
	{
    delete[] clblasA;
	}
	if(X != NULL)
	{
	delete[] X;
}
}

template <typename T>
void
syrCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasA, *clblasA, *X;
//	T *tempA;
    cl_mem bufA, bufX;
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

    size_t lengthA = params->N * params->lda;
    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));

    blasA 		= new T[lengthA + params->offa ];
    clblasA 	= new T[lengthA + params->offa ];
    X		 	= new T[lengthX + params->offBX ];
//	tempA 		= new T[lengthA + params->offa ];

    srand(params->seed);

    ::std::cerr << "Generating input data... ";


	memset(blasA, -1, (lengthA + params->offa));
	memset(clblasA, -1, (lengthA + params->offa));
	memset(X, -1, (lengthX + params->offBX));

	alpha =  convertMultiplier<T>(params->alpha);
	useAlpha = true;

	#ifdef DEBUG_SYR
	printf("ALPHA in CORR_SYR.CPP %f\n", alpha);
	#endif

	if((blasA == NULL) || (X == NULL) || (clblasA == NULL))
    {
        ::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
		deleteBuffers<T>(blasA, clblasA, X);
        delete[] events;
        SUCCEED();
        return;
    }

	randomSyrMatrices<T>(params->order, params->uplo, params->N, useAlpha, &alpha,
						(blasA + params->offa), params->lda, (X + params->offBX), params->incx);

/*
	// Set data in A and X using populate() routine
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (params-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    creationFlags = ( (params-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_SYR;
    // Populate A and blasX
    populate( blasA + params->offa, params-> N, params-> N, params-> lda, BlasFn, creationFlags);
    populate( X , (lengthX + params->offBX), 1, (lengthX + params->offBX), BlasFn);
*/
    // Copy blasA to clblasA
    memcpy(clblasA, blasA, (lengthA + params->offa)* sizeof(*blasA));
  //  memcpy(tempA, blasA, (lengthA + params->offa)* sizeof(*blasA));

	::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufA = base->createEnqueueBuffer(clblasA, (lengthA + params->offa) * sizeof(*clblasA), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX)* sizeof(*X), 0, CL_MEM_READ_ONLY);

    ::std::cerr << "Calling reference xSYR routine... ";

	clblasOrder order;
    clblasUplo fUplo;

	order = params->order;
    fUplo = params->uplo;

	//printf("\n\n before acml call\nA\n");
   // printMatrixBlock( params->order, 0, 0, params->N, params->N, params->lda, blasA);
    //printf("\nX\n");
    //printMatrixBlock( clblasColumnMajor, 0, 0, lengthX, 1, lengthX, X);

	if (order == clblasColumnMajor)
    {
		::clMath::blas::syr( clblasColumnMajor, fUplo, params->N, alpha, X, params->offBX, params->incx, blasA, params->offa, params->lda);
    }
 	else
	{
        T *reorderedA = new T[lengthA + params->offa];

        //reorderMatrix<T>(clblasRowMajor, params->N, params->lda, blasA, reorderedA);

		fUplo = (fUplo == clblasUpper) ? clblasLower : clblasUpper;
		//::clMath::blas::syr( clblasColumnMajor, fUplo, params->N, alpha, X, params->offBX, params->incx, reorderedA, params->offa, params->lda);

		::clMath::blas::syr( clblasColumnMajor, fUplo, params->N, alpha, X, params->offBX, params->incx, blasA, params->offa, params->lda);

		//reorderMatrix<T>(clblasColumnMajor, params->lda, params->N, reorderedA, blasA);

        delete[] reorderedA;
    }
	//printf("After acml\n");
	//printMatrixBlock( params->order, 0, 0, params->N, params->N, params->lda, blasA);

    ::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufX == NULL) ) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(blasA, clblasA, X);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSYR routine... ";

    err = (cl_int)::clMath::clblas::syr( params->order, params->uplo, params->N, alpha,
						bufX, params->offBX, params->incx, bufA, params->offa, params->lda,
						params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(blasA, clblasA, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SYR() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(blasA, clblasA, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufA, CL_TRUE, 0,
        (lengthA + params->offa) * sizeof(*clblasA), clblasA, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "SYR: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufA, bufX);
	//printMatrixBlock( params->order, 0, 0, params->N, params->N, params->lda, clblasA);
	//getchar();

//	printf("Comparing with the temp buffer\n");
//    compareMatrices<T>(clblasColumnMajor, 1, (params->lda - params->N), (blasA + params->offa + params->N), (tempA + params->offa + params->N),
//    					params->lda);
//	delete[] tempA;
	printf("Comparing the results\n");
	compareMatrices<T>(params->order, params->N , params->N, (blasA + params->offa), (clblasA + params->offa),
                       params->lda);

	deleteBuffers<T>(blasA, clblasA, X);
    delete[] events;
}

// Instantiate the test

TEST_P(SYR, ssyr) {
    TestParams params;

    getParams(&params);
    syrCorrectnessTest<cl_float>(&params);
}

TEST_P(SYR, dsyr) {
    TestParams params;

    getParams(&params);
    syrCorrectnessTest<cl_double>(&params);
}
