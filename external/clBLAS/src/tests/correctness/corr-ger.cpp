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
#include <ger.h>
#include<cltypes.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objx, cl_mem objy)
{
    if( objA!=NULL)
	clReleaseMemObject(objA);
    if( objx!=NULL)
    	clReleaseMemObject(objx);
    if( objy!=NULL)
    	clReleaseMemObject(objy);
}

template <typename T> static void
deleteBuffers(T *A, T *x, T *y, T *backA)
{

	if(A != NULL)
    {
	delete[] A;
    }
	if(backA != NULL)
	{
		delete[] backA;
	}
	if(x != NULL)
	{
    delete[] x;
	}
	if(y != NULL)
	{
    delete[] y;
}
}

template <typename T>
void
gerCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *x, *y, *backA;
    //size_t N, M;

    T alpha_;
    cl_mem bufA, bufx, bufy;
    clMath::BlasBase *base;
    cl_event *events;
//	int ka, kxy;

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

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

	size_t lengthA;
	if( params->order == clblasColumnMajor )
			lengthA = params->N  * params->lda;
	else	lengthA = params->M  * params->lda;

    size_t lengthx = (1 + (((params->M)-1) * abs(params->incx)));
    size_t lengthy = (1 + (((params->N)-1) * abs(params->incy)));

    bool useAlpha = base->useAlpha();

    if (useAlpha) {
        alpha_ = convertMultiplier<T>(params->alpha);
    }


    A 		= new T[lengthA + params->offa];
    x   	= new T[lengthx + params->offBX];
    y   	= new T[lengthy + params->offCY];
    backA       = new T[lengthA + params->offa];

	if((A == NULL) || (backA == NULL) || (x == NULL) || (y == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(A, backA, x, y);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;
    creationFlags = ( (params-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
	BlasRoutineID BlasFn = CLBLAS_GER;

    populate( (A + params->offa), params->M, params->N, params-> lda, BlasFn, creationFlags);
    populate( (x + params->offBX), lengthx, 1, lengthx, BlasFn );
    populate( (y + params->offCY), lengthy, 1, lengthy, BlasFn );

    // Copy C to backX
    memcpy(backA, A, (lengthA + params->offa) * sizeof(T));

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offa) * sizeof(*A), 0, CL_MEM_READ_WRITE);
    bufx = base->createEnqueueBuffer(x, (lengthx + params->offBX) * sizeof(*x), 0, CL_MEM_READ_ONLY);
    bufy = base->createEnqueueBuffer(y, (lengthy + params->offCY) * sizeof(*y), 0, CL_MEM_READ_ONLY);


    ::std::cerr << "Done" << ::std::endl;
    ::std::cerr << "Calling reference xGER routine... ";


	clblasOrder fOrder;
    size_t fN, fM;
    size_t fOffx, fOffy;
    int fIncx, fIncy;
    T *fX, *fY;
    fOrder = params->order;
    fM = params->M;
    fN = params->N;
    fIncx = params->incx;
    fIncy = params->incy;
    fX = x;
    fY = y;
    fOffx = params->offBX;
    fOffy = params->offCY;

    if (fOrder != clblasColumnMajor) {

           fOrder = clblasColumnMajor;
           fM = params->N;
           fN = params->M;
           fX = y;
           fY = x;
           fIncx = params->incy;
           fIncy = params->incx;
           fOffx = params->offCY;
           fOffy = params->offBX;
       }

    // Call reference blas routine
    clMath::blas::ger(fOrder, fM, fN, alpha_, fX , fOffx, fIncx, fY, fOffy, fIncy,  A, params->offa, params->lda);
	::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufx == NULL) || (bufy == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufx, bufy);
        deleteBuffers<T>(A, x, y, backA);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xGER routine... ";

    err = (cl_int)::clMath::clblas::ger( params->order, params->M, params->N, alpha_,
                            bufx, params->offBX, params->incx, bufy, params->offCY, params->incy,bufA, params->offa, params->lda,
							params->numCommandQueues, base->commandQueues(), 0, NULL, events );

    if (err != CL_SUCCESS) {

   	releaseMemObjects(bufA, bufx, bufy);
        deleteBuffers<T>(A, x, y, backA);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::GER() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {

        releaseMemObjects(bufA, bufx, bufy);
        deleteBuffers<T>(A, x, y, backA);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufA, CL_TRUE, 0,
        (lengthA + params->offa)* sizeof(*backA), backA, 0,
        NULL, NULL);

    releaseMemObjects(bufA, bufx, bufy);

    // handle lda correctly based on row-major/col-major..
    compareMatrices<T>(params->order, params->M , params->N, A+ params->offa, backA + params->offa, params->lda);
    deleteBuffers<T>(A, x, y, backA);
    delete[] events;
}

// Instantiate the test


TEST_P(GER, sger) {
    TestParams params;

    getParams(&params);
    gerCorrectnessTest<cl_float>(&params);
}

TEST_P(GER, dger) {
    TestParams params;

    getParams(&params);
    gerCorrectnessTest<cl_double>(&params);
}


TEST_P(GER, cgeru) {
    TestParams params;

    getParams(&params);
    gerCorrectnessTest<FloatComplex>(&params);
}

TEST_P(GER, zgeru) {
    TestParams params;

    getParams(&params);
    gerCorrectnessTest<DoubleComplex>(&params);
}
