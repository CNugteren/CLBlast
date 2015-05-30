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
#include <scal.h>

static void
releaseMemObjects(cl_mem objX)
{
    if(objX != NULL)
    {
        clReleaseMemObject(objX);
	}
}

template <typename T> static void
deleteBuffers(T *blasX, T *clblasX)
{
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
void scalCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasX, *clblasX;
    cl_mem bufX;
    clMath::BlasBase *base;
    cl_event *events;
    T alpha;

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
    bool is_css_zds = (params->K == 1)? true: false;        // K indicates csscal/zdscal

    blasX = new T[lengthX + params->offBX ];
    clblasX = new T[lengthX + params->offBX ];

	if( (blasX == NULL) || (clblasX == NULL) )
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(blasX, clblasX);
		delete[] events;
		SUCCEED();
        return;
	}

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

    randomVectors(params->N, (blasX+params->offBX), params->incx);
    alpha = convertMultiplier<T>(params->alpha);
    memcpy(clblasX, blasX, (lengthX + params->offBX)* sizeof(*blasX));
    ::std::cerr << "Done" << ::std::endl;
    bufX = base->createEnqueueBuffer(clblasX, (lengthX + params->offBX)* sizeof(*clblasX), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xSCAL routine... ";
    // Both blas and clBlas wrapper functions consider the real part of alpha in case of css/zdscal
    // This is to make sure both get the same scalar alpha. check wrapper functions
    ::clMath::blas::scal(is_css_zds, params->N, alpha, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if (bufX == NULL) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufX);
        deleteBuffers<T>(blasX, clblasX);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSCAL routine... ";
    // Both blas and clBlas wrapper functions consider the real part of alpha in case of css/zdscal
    // This is to make sure both get the same scalar alpha. check wrapper functions
    err = (cl_int)::clMath::clblas::scal(is_css_zds, params->N, alpha, bufX, params->offBX,
                    params->incx, params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX);
        deleteBuffers<T>(blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SCAL() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues, base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufX);
        deleteBuffers<T>(blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
                   (lengthX + params->offBX) * sizeof(*clblasX), clblasX, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		::std::cerr << "SCAL: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufX);

    compareMatrices<T>(clblasColumnMajor, lengthX , 1, (blasX + params->offBX),
                        (clblasX + params->offBX), lengthX);
    deleteBuffers<T>(blasX, clblasX);
    delete[] events;
}

// Instantiate the test

TEST_P(SCAL, sscal) {
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<cl_float>(&params);
}

TEST_P(SCAL, dscal) {
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<cl_double>(&params);
}

TEST_P(SCAL, cscal) {
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<FloatComplex>(&params);
}

TEST_P(SCAL, zscal) {
    TestParams params;

    getParams(&params);
    params.K = 0;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<DoubleComplex>(&params);
}


// For these 2 routines alpha is scalar
TEST_P(SCAL, csscal) {
    TestParams params;

    getParams(&params);
    params.K = 1;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<FloatComplex>(&params);
}

TEST_P(SCAL, zdscal) {
    TestParams params;

    getParams(&params);
    params.K = 1;                           // K will indicate wheather routine is csscal/zdscal
    scalCorrectnessTest<DoubleComplex>(&params);
}
