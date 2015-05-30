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


#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"


clblasStatus
doRotm(
	CLBlasKargs *kargs,
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_mem param,
    size_t offParam,
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

        retCode = checkMemObjects(X, Y, param, true, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, N, Y, offy, incy, Y_VEC_ERRSET))) {
			printf("Invalid Size for Y\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 5, param, offParam, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for PARAM\n"); // PARAM is of minimum length 5
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
		kargs->B = Y;
		kargs->offCY = offy;
		kargs->ldc.vector = incy;	// Will be using this as incy
		kargs->D = param;
		kargs->offd = offParam;
		kargs->pigFuncID = CLBLAS_ROTM;

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_ROTM, kargs, numCommandQueues, commandQueues,
        					        numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}





clblasStatus
clblasSrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem SPARAM,
    size_t offSparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_FLOAT;

		return doRotm(&kargs, N, X, offx, incx, Y, offy, incy, SPARAM, offSparam,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem DPARAM,
    size_t offDparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_DOUBLE;

		return doRotm(&kargs, N, X, offx, incx, Y, offy, incy, DPARAM, offDparam,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

