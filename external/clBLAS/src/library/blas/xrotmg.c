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
doRotmg(
	CLBlasKargs *kargs,
    cl_mem D1,
    size_t offD1,
    cl_mem D2,
    size_t offD2,
    cl_mem X1,
    size_t offX1,
    cl_mem Y1,
    size_t offY1,
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

        retCode = checkMemObjects(D1, D2, X1, true, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {      // for mem objects A, B
			printf("Invalid mem object..\n");
            return retCode;
		}
		retCode = checkMemObjects(Y1, param, Y1, false, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {      // for mem objects C, S
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, 1, D1, offD1, 1, X_VEC_ERRSET))) {
			printf("Invalid Size for D1\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, D2, offD2, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for D2\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, X1, offX1, 1, X_VEC_ERRSET))) {
			printf("Invalid Size for X1\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, Y1, offY1, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for Y1\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, param, offParam, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for PARAM\n");
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

		kargs->A = D1;
    	kargs->B = D2;
		kargs->C = X1;
    	kargs->D = Y1;
    	kargs->E = param;
		kargs->offa = offD1;
		kargs->offb = offD2;
        kargs->offc = offX1;
        kargs->offd = offY1;
        kargs->offe = offParam;

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_ROTMG, kargs, numCommandQueues, commandQueues,
        					        numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}



clblasStatus
clblasSrotmg(
    cl_mem SD1,
    size_t offSD1,
    cl_mem SD2,
    size_t offSD2,
    cl_mem SX1,
    size_t offSX1,
    const cl_mem SY1,
    size_t offSY1,
    cl_mem SPARAM,
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

		return doRotmg(&kargs, SD1, offSD1, SD2, offSD2, SX1, offSX1, SY1, offSY1, SPARAM, offSparam,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDrotmg(
    cl_mem DD1,
    size_t offDD1,
    cl_mem DD2,
    size_t offDD2,
    cl_mem DX1,
    size_t offDX1,
    const cl_mem DY1,
    size_t offDY1,
    cl_mem DPARAM,
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

		return doRotmg(&kargs, DD1, offDD1, DD2, offDD2, DX1, offDX1, DY1, offDY1, DPARAM, offDparam,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

