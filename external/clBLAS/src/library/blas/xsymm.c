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

#define SYMM_USING_GEMM
//#define DEBUG_SYMM
extern clblasStatus executeGEMM( CLBlasKargs *kargs, cl_uint numCommandQueues,
                                    cl_command_queue *commandQueues,
                                    cl_uint numEventsInWaitList, const cl_event *eventWaitList,
                                    cl_event *events);

clblasStatus
doSymm(	CLBlasKargs *kargs, clblasOrder order, clblasUplo uplo, clblasSide side,
		size_t M, size_t N,
		const cl_mem A, size_t offa, size_t lda,
		const cl_mem B, size_t offb, size_t ldb,
		cl_mem C, size_t offc, size_t ldc,
		cl_uint numCommandQueues, cl_command_queue *commandQueues,
		cl_uint numEventsInWaitList, const cl_event *eventWaitList,
		cl_event *events,
        BlasFunctionID symm_or_hemm)
{
    cl_int err;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, B, C, true, A_MAT_ERRSET, B_MAT_ERRSET, C_MAT_ERRSET))) {
		//printf("SYMM:- Invalid mem object..\n");
        return retCode;
    }


    if (side == clblasLeft)
    {
        // MxM x MxN
        if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, M, A, offa, lda, A_MAT_ERRSET))) {
            //printf("Invalid Size for A\n");
            return retCode;
        }
    } else {
        // MxN x NxN
        if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, N, N, A, offa, lda, A_MAT_ERRSET))) {
            //printf("Invalid Size for A\n");
            return retCode;
        }
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, N, B, offb, ldb, B_MAT_ERRSET))) {
		//printf("Invalid Size for B\n");
        return retCode;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, N, C, offc, ldc, C_MAT_ERRSET))) {
		//printf("Invalid Size for C\n");
        return retCode;
    }

	#ifdef DEBUG_SYMM
	printf("DoSymm being called...\n");
	#endif

	if ((commandQueues == NULL) || (numCommandQueues == 0))
	{
		return clblasInvalidValue;
	}

	if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
	{
		return clblasInvalidEventWaitList;
	}

	numCommandQueues = 1;
    kargs->order = order;
    kargs->uplo = uplo;
	kargs->side = side;
    kargs->pigFuncID = symm_or_hemm;
    kargs->M = M;
	if (kargs->side == clblasLeft)
	{
		kargs->K = M;
	} else {
		kargs->K = N;
	}
    kargs->N = N;
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = B;
    kargs->ldb.matrix = ldb;
    kargs->C = C;
    kargs->ldc.matrix = ldc;
	kargs->offA = offa;
	kargs->offa = offa;
    kargs->offA = offa;
    kargs->offBX = offb;
    kargs->offCY = offc;
    kargs->offsetM = 0;
    kargs->offsetN = 0;
    //kargs->offsetK = 0;   FIXME: not found offsetK in new AMD structure!
    kargs->scimage[0] = 0;
    kargs->scimage[1] = 0;
	if (kargs->order == clblasRowMajor)
	{
		kargs->order = clblasColumnMajor;
		kargs->M = N;
		kargs->N = M;

		if (kargs->side == clblasLeft)
		{
			kargs->side = clblasRight;
		} else {
			kargs->side = clblasLeft;
		}

		if (kargs->uplo == clblasUpper)
		{
			kargs->uplo = clblasLower;
		} else {
			kargs->uplo = clblasUpper;
		}
	}

#ifndef SYMM_USING_GEMM
	#ifdef DEBUG_SYMM
	printf("Calling makeSolutionSeq : SYMM \n");
	#endif
    {
        ListHead seq;

        listInitHead(&seq);
        err = makeSolutionSeq(CLBLAS_SYMM, kargs, numCommandQueues, commandQueues,
            				  numEventsInWaitList, eventWaitList, events, &seq);
        if (err == CL_SUCCESS) {
       	    err = executeSolutionSeq(&seq);
        }
        freeSolutionSeq(&seq);
    }
#else
    //
    // SYMM_USING_GEMM
    //
    {
        CLBlasKargs GEMMNArgs, GEMMTArgs, GEMMDArgs;
        cl_event gemmNEvent, gemmTEvent ;
        FloatComplex cBeta;
        DoubleComplex zBeta;
        clblasTranspose transposeFunction = clblasTrans;

        memcpy(&GEMMNArgs, kargs, sizeof(CLBlasKargs));
        memcpy(&GEMMTArgs, kargs, sizeof(CLBlasKargs));
        memcpy(&GEMMDArgs, kargs, sizeof(CLBlasKargs));

        switch(symm_or_hemm)
        {
            case CLBLAS_SYMM:
                transposeFunction = clblasTrans;
                GEMMDArgs.pigFuncID = CLBLAS_SYMM_DIAGONAL;
                break;

            case CLBLAS_HEMM:
                transposeFunction = clblasConjTrans;
                GEMMDArgs.pigFuncID = CLBLAS_HEMM_DIAGONAL;
                break;

            default:
                printf("WARNING: doSymm():  Neither SYMM nor HEMM is calling this function.");
                break;
        }


        //
        // It is the diagonal piggy back for GEMMD. For others, it is just CLBLAS_SYMM
        //

        //
        // Set the Transpose for GEMM'T' and GEMM'D'
        // The other two do not have transpose by default
        //
        switch(kargs->side)
        {
            case clblasLeft:
                GEMMTArgs.transA = transposeFunction;
                if (kargs->uplo == clblasUpper)
                {
                    //
                    // This is for proper TAIL handling for Right Lower case alone
                    // For all other cases, NN kernel is good enough to handle tails
                    //
                   GEMMDArgs.transA = transposeFunction;
                }
                break;

            case clblasRight:
                GEMMTArgs.transB = transposeFunction;
                if (kargs->uplo == clblasLower)
                {
                    //
                    // This is for proper TAIL handling for Right Lower case alone
                    // For all other cases, NN kernel is good enough to handle tails
                    //
                    GEMMDArgs.transB = transposeFunction;
                }
                break;

            default:
                break;
        }

        //
        // Set the BETA multiplier to 1 for GEMMT and GEMMD
        //
        memset(&GEMMTArgs.beta, 0, sizeof(GEMMTArgs.beta));
        memset(&GEMMDArgs.beta, 0, sizeof(GEMMDArgs.beta));
        switch(kargs->dtype)
        {
            case TYPE_FLOAT:
            GEMMTArgs.beta.argFloat = 1.0f;
            GEMMDArgs.beta.argFloat = 1.0f;
            break;

            case TYPE_DOUBLE:
            GEMMTArgs.beta.argDouble = 1.0;
            GEMMDArgs.beta.argDouble = 1.0;
            break;

            case TYPE_COMPLEX_FLOAT:
            CREAL(cBeta) = 1.0f;
            CIMAG(cBeta) = 0.0f;
            GEMMTArgs.beta.argFloatComplex = cBeta;
            GEMMDArgs.beta.argFloatComplex = cBeta;
            break;

            case TYPE_COMPLEX_DOUBLE:
            CREAL(zBeta) = 1.0;
            CIMAG(zBeta) = 0.0;
            GEMMTArgs.beta.argDoubleComplex = zBeta;
            GEMMDArgs.beta.argDoubleComplex = zBeta;
            break;
        }

        //
        // GEMM Handler will notice the "pigFuncID" and set appropriate flags
        //
        err = executeGEMM(&GEMMNArgs, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, &gemmNEvent);
        if (err == CL_SUCCESS)
        {
            err = executeGEMM(&GEMMTArgs, numCommandQueues, commandQueues, 1, &gemmNEvent, &gemmTEvent);
            if (err == CL_SUCCESS)
            {
                err = executeGEMM(&GEMMDArgs, numCommandQueues, commandQueues, 1, &gemmTEvent, events);
            }
        }
    }
#endif
    return (clblasStatus)err;
}

clblasStatus
clblasSsymm(
	clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    float beta,
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

    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_FLOAT;
    	kargs.alpha.argFloat = alpha;
    	kargs.beta.argFloat = beta;

		#ifdef DEBUG_SYMM
		printf("Ssymm called\n");
		#endif
    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events, CLBLAS_SYMM);
	}

clblasStatus
clblasDsymm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    double beta,
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

		#ifdef DEBUG_SYMM
		printf("Dsymm called\n");
		#endif
    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_DOUBLE;
    	kargs.alpha.argDouble = alpha;
    	kargs.beta.argDouble = beta;

    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList,
						eventWaitList, events, CLBLAS_SYMM);
	}

clblasStatus
clblasCsymm(
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

		#ifdef DEBUG_SYMM
		printf("Csymm called\n");
		#endif
    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_COMPLEX_FLOAT;
    	kargs.alpha.argFloatComplex = alpha;
    	kargs.beta.argFloatComplex  = beta;

    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList,
						eventWaitList, events, CLBLAS_SYMM);
	}

clblasStatus
clblasZsymm(
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

		#ifdef DEBUG_SYMM
		printf("Zsymm called\n");
		#endif
    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_COMPLEX_DOUBLE;
    	kargs.alpha.argDoubleComplex = alpha;
    	kargs.beta.argDoubleComplex  = beta;

    	return doSymm(	&kargs, order, uplo, side, M, N, A, offa, lda, B, offb, ldb, C, offc, ldc,
						numCommandQueues, commandQueues, numEventsInWaitList,
						eventWaitList, events, CLBLAS_SYMM);
	}

