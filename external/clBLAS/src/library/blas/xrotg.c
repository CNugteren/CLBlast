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

//#define DEBUG_ROTG

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"


clblasStatus
doRotg(
	CLBlasKargs *kargs,
    cl_mem A,
    size_t offA,
    cl_mem B,
    size_t offB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		cl_int err;
		ListHead seq;
        clblasStatus retCode = clblasSuccess;

        // C is of real type even for complex numbers
        DataType cType = (kargs->dtype == TYPE_COMPLEX_FLOAT)? TYPE_FLOAT :
                            ((kargs->dtype == TYPE_COMPLEX_DOUBLE)? TYPE_DOUBLE : (kargs->dtype));

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

        retCode = checkMemObjects(A, B, A, false, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {      // for mem objects A, B
			printf("Invalid mem object..\n");
            return retCode;
		}
		retCode = checkMemObjects(C, S, C, false, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {      // for mem objects C, S
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, 1, A, offA, 1, X_VEC_ERRSET))) {
			printf("Invalid Size for A\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, B, offB, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for B\n");
            return retCode;
		}

		if ((retCode = checkVectorSizes(cType, 1, C, offC, 1, X_VEC_ERRSET))) {
			printf("Invalid Size for C\n");
            return retCode;
		}

		if ((retCode = checkVectorSizes(kargs->dtype, 1, S, offS, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for S\n");
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

		kargs->A = A;
    	kargs->B = B;
		kargs->C = C;
    	kargs->D = S;
		kargs->offa = offA;
		kargs->offb = offB;
        kargs->offc = offC;
        kargs->offd = offS;

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_ROTG, kargs, numCommandQueues, commandQueues,
        					        numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}



clblasStatus
clblasSrotg(
    cl_mem SA,
    size_t offSA,
    cl_mem SB,
    size_t offSB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_FLOAT;

		return doRotg(&kargs, SA, offSA, SB, offSB, C, offC, S, offS,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDrotg(
    cl_mem DA,
    size_t offDA,
    cl_mem DB,
    size_t offDB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_DOUBLE;

		return doRotg(&kargs, DA, offDA, DB, offDB, C, offC, S, offS,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_FLOAT;

		return doRotg(&kargs, CA, offCA, CB, offCB, C, offC, S, offS,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_DOUBLE;

		return doRotg(&kargs, CA, offCA, CB, offCB, C, offC, S, offS,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}
