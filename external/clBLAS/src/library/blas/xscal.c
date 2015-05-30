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


//#define DEBUG_SCAL

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"


clblasStatus
doScal(
	CLBlasKargs *kargs,
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		cl_int err;
		ListHead seq;
        clblasStatus retCode = clblasSuccess;

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

        retCode = checkMemObjects(X, X, X, false, X_VEC_ERRSET, X_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		///////////////////////////////////////////////////////////////

		if ((commandQueues == NULL) || (numCommandQueues == 0))
		{
			return clblasInvalidValue;
		}

		/* numCommandQueues will be hardcoded to 1 as of now. No multi-gpu support */
		numCommandQueues = 1;
		if (commandQueues[0] == NULL)
		{
			return clblasInvalidCommandQueue;
		}

		if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
		{
			return clblasInvalidEventWaitList;
		}

		kargs->N = N;
		kargs->A = X;
		kargs->offBX = offx;
		kargs->ldb.vector = incx;	// Will be using this as incx

		if(incx < 0) {    // According to Netlib - return for negative incx
		    return clblasSuccess;
		}

		#ifdef DEBUG_SCAL
		printf("Calling makeSolutionSeq from DoScal: SCAL\n");
		#endif

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_SCAL, kargs, numCommandQueues, commandQueues,
        					        numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}





clblasStatus
clblasSscal(
    size_t N,
    float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_SCAL
		printf("\nSSCAL Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_FLOAT;
        kargs.alpha.argFloat = alpha;

		return doScal(&kargs, N, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDscal(
    size_t N,
    double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_SCAL
		printf("\nDSCAL Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_DOUBLE;
        kargs.alpha.argDouble = alpha;

		return doScal(&kargs, N, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCscal(
    size_t N,
    cl_float2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_SCAL
		printf("\nCSCAL Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_FLOAT;
        kargs.alpha.argFloatComplex = alpha;

		return doScal(&kargs, N, X, offx, incx,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZscal(
    size_t N,
    cl_double2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_SCAL
		printf("\nZSCAL Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_DOUBLE;
        kargs.alpha.argDoubleComplex = alpha;

		return doScal(&kargs, N, X, offx, incx,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCsscal(
    size_t N,
    float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        CLBlasKargs kargs;
        FloatComplex fAlpha;

        #ifdef DEBUG_SSCAL
        printf("\nCSSCAL Called\n");
        #endif

        CREAL(fAlpha) = alpha;
        CIMAG(fAlpha) = 0.0f;

        memset(&kargs, 0, sizeof(kargs));
        kargs.alpha.argFloatComplex = fAlpha;
        kargs.dtype = TYPE_COMPLEX_FLOAT;

        return doScal(&kargs, N, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clblasZdscal(
    size_t N,
    double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
         CLBlasKargs kargs;
        DoubleComplex fAlpha;

        #ifdef DEBUG_SSCAL
        printf("\nZDSCAL Called\n");
        #endif

        CREAL(fAlpha) = alpha;
        CIMAG(fAlpha) = 0.0f;

        memset(&kargs, 0, sizeof(kargs));
        kargs.alpha.argDoubleComplex = fAlpha;
        kargs.dtype = TYPE_COMPLEX_DOUBLE;

        return doScal(&kargs, N, X, offx, incx, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

