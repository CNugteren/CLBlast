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
#include <tpmv.h>

static void
releaseMemObjects(cl_mem objAP, cl_mem objX,  cl_mem objXTemp)
{
   if(objAP != NULL)
 	{
    clReleaseMemObject(objAP);
	}
	if(objX != NULL)
    {
    clReleaseMemObject(objX);
	}
	if(objXTemp != NULL)
	{
    clReleaseMemObject(objXTemp);
}
}

template <typename T> static void
deleteBuffers(T *AP, T *blasX, T *clblasX)
{
    if(AP != NULL)
    {
    delete[] AP;
    }
	if(blasX != NULL)
	{
    delete[] blasX;
    }
	if(clblasX != NULL)
	{
		delete[] clblasX;
	}
}

template <typename T>
void
tpmvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *AP, *blasX, *clblasX;
    cl_mem bufAP, bufX, bufXTemp;
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

    size_t lengthAP = (params->N *( params->N + 1 ))/2 ;
    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));

    AP 		= new T[lengthAP + params->offa ];
    blasX 	= new T[lengthX + params->offBX ];
    clblasX = new T[lengthX + params->offBX ];

	if((AP == NULL) || (blasX == NULL) || (clblasX == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(AP, blasX, clblasX);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    // Set data in A and X using populate() routine
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT | PACKED_MATRIX;

    // Default is Column-Major
    creationFlags = ( (params-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    creationFlags = ( (params-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_TRMV;

    // Populate A and blasX
    populate( AP + params->offa, params-> N, params-> N, 0, BlasFn, creationFlags);
    populate( blasX , (lengthX + params->offBX), 1, (lengthX + params->offBX), BlasFn);

    // Copy blasX to clblasX
    memcpy(clblasX, blasX, (lengthX + params->offBX)* sizeof(*blasX));
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufAP = base->createEnqueueBuffer(AP, (lengthAP + params->offa)* sizeof(*AP), 0, CL_MEM_READ_ONLY);
    bufX = base->createEnqueueBuffer(clblasX, (lengthX + params->offBX)* sizeof(*clblasX), 0, CL_MEM_WRITE_ONLY);
    bufXTemp = base->createEnqueueBuffer(NULL, lengthX * sizeof(*clblasX), 0, CL_MEM_READ_ONLY);

	//printData( "bufX", blasX, lengthX, 1, lengthX);
	//printData( "clblasX", clblasX, lengthX, 1, lengthX);

    ::std::cerr << "Calling reference xTPMV routine... ";


	clblasOrder order;
    clblasUplo fUplo;
    clblasTranspose fTrans;

	order = params->order;
    fUplo = params->uplo;
    fTrans = params->transA;

	if (order != clblasColumnMajor)
    {
        order = clblasColumnMajor;
        fUplo =  (params->uplo == clblasUpper)? clblasLower : clblasUpper;
        fTrans = (params->transA == clblasNoTrans)? clblasTrans : clblasNoTrans;

        if( params->transA == clblasConjTrans )
            doConjugate( (AP +params->offa), (( params->N * (params->N + 1)) / 2) , 1, 1 );
    }

	::clMath::blas::tpmv( order, fUplo, fTrans, params->diag, params->N, AP, params->offa, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    // Hold X vector

    if ((bufAP == NULL) || (bufX == NULL) || (bufXTemp == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufAP, bufX, bufXTemp);
        deleteBuffers<T>(AP, blasX, clblasX);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTPMV routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT:TYPE_COMPLEX_DOUBLE;

    // Should use bufXTemp as well
    err = (cl_int)::clMath::clblas::tpmv( type, params->order, params->uplo, params->transA, params->diag, params->N, bufAP,
    					params->offa, bufX, params->offBX, params->incx, bufXTemp, params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX, bufXTemp);
        deleteBuffers<T>(AP, blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TPMV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX, bufXTemp);
        deleteBuffers<T>(AP, blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        (lengthX + params->offBX) * sizeof(*clblasX), clblasX, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "TPMV: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufAP, bufX, bufXTemp);

    compareMatrices<T>(clblasColumnMajor, lengthX , 1, (blasX + params->offBX), (clblasX + params->offBX),
                       lengthX);
    deleteBuffers<T>(AP, blasX, clblasX);
    delete[] events;
}

// Instantiate the test

TEST_P(TPMV, stpmv) {
    TestParams params;

    getParams(&params);
    tpmvCorrectnessTest<cl_float>(&params);
}

TEST_P(TPMV, dtpmv) {
    TestParams params;

    getParams(&params);
    tpmvCorrectnessTest<cl_double>(&params);
}

TEST_P(TPMV, ctpmv) {
    TestParams params;

    getParams(&params);
    tpmvCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TPMV, ztpmv) {
    TestParams params;

    getParams(&params);
    tpmvCorrectnessTest<DoubleComplex>(&params);
}
