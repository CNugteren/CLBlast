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


//#define DEBUG_AXPY

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"


clblasStatus
doAxpy(
	CLBlasKargs *kargs,
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
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

        retCode = checkMemObjects(X, Y, X, false, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {
			#ifdef DEBUG_AXPY
			printf("Invalid mem object..\n");
			#endif
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET))) {
			#ifdef DEBUG_AXPY
			printf("Invalid Size for X\n");
			#endif
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, N, Y, offy, incy, Y_VEC_ERRSET))) {
			#ifdef DEBUG_AXPY
			printf("Invalid Size for Y\n");
			#endif
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

		/*
 		 * ASSUMPTION:
 		 * doTRMV assumes "commandQueue" of 0. The same is reflected in
		 * "makeSolutionSeq" as well. If either of them changes in future,
		 * this code needs to be revisited.
  		 */

		kargs->N = N;
		kargs->A = X;
		kargs->offBX = offx;
		kargs->ldb.vector = incx;	// Will be using this as incx
		kargs->B = Y;
		kargs->offCY = offy;
		kargs->ldc.vector = incy;	// Will be using this as incy

		#ifdef DEBUG_AXPY
		printf("Calling makeSolutionSeq from DoAxpy: AXPY\n");
		#endif

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_AXPY, kargs, numCommandQueues, commandQueues,
        					        numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}





clblasStatus
clblasSaxpy(
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_AXPY
		printf("\nSAXPY Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_FLOAT;
        kargs.alpha.argFloat = alpha;

		return doAxpy(&kargs, N, X, offx, incx, Y, offy, incy,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDaxpy(
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_AXPY
		printf("\nDAXPY Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_DOUBLE;
        kargs.alpha.argDouble = alpha;

		return doAxpy(&kargs, N, X, offx, incx, Y, offy, incy,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCaxpy(
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int  incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_AXPY
		printf("\nCAXPY Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_FLOAT;
        kargs.alpha.argFloatComplex = alpha;

		return doAxpy(&kargs, N, X, offx, incx, Y, offy, incy,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZaxpy(
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		#ifdef DEBUG_AXPY
		printf("\nZAXPY Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_DOUBLE;
        kargs.alpha.argDoubleComplex = alpha;

		return doAxpy(&kargs, N, X, offx, incx, Y, offy, incy,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}
