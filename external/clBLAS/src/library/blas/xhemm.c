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

extern clblasStatus
doSymm( CLBlasKargs *kargs, clblasOrder order, clblasUplo uplo, clblasSide side,
        size_t M, size_t N,
        const cl_mem A, size_t offa, size_t lda,
        const cl_mem B, size_t offb, size_t ldb,
        cl_mem C, size_t offc, size_t ldc,
        cl_uint numCommandQueues, cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList, const cl_event *eventWaitList,
        cl_event *events,
        BlasFunctionID symm_or_hemm);


clblasStatus
clblasChemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
        CLBlasKargs kargs;

		#ifdef DEBUG_HEMM
		printf("Chemm called\n");
		#endif
    	memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_FLOAT;
    	kargs.alpha.argFloatComplex = alpha;
    	kargs.beta.argFloatComplex  = beta;
    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList,
						eventWaitList, events, CLBLAS_HEMM);
	}

clblasStatus
clblasZhemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
        CLBlasKargs kargs;

		#ifdef DEBUG_HEMM
		printf("Zhemm called\n");
		#endif
    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_COMPLEX_DOUBLE;
    	kargs.alpha.argDoubleComplex = alpha;
    	kargs.beta.argDoubleComplex  = beta;

    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList,
						eventWaitList, events, CLBLAS_HEMM);
	}

