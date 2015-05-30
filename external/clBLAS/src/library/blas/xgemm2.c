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

//#define DEBUG_GEMM_2

int
gemmHasMTail(size_t M,  int vecLen, clblasOrder order, clblasTranspose transA, clblasTranspose transB)
{
	transB = transB;    // Dummy- to remove warning
    if (order == clblasColumnMajor)
	{
		if (transA == clblasNoTrans)
		{
			return (M % vecLen);
		} else {
			return 0;
		}
	} else {
		printf("gemmHasMTail: Not handling Row Major - FIXME\n");
		return 0;
	}
}

int
gemmHasNTail(size_t N, int vecLen, clblasOrder order, clblasTranspose transA, clblasTranspose transB)
{
	if (order == clblasColumnMajor)
	{
		if (transA == clblasNoTrans)
		{
			if (transB == clblasNoTrans)
			{
				return 0;
			} else {
				return (N % vecLen);
			}
		} else {
			if (transB == clblasNoTrans)
			{
				return 0;
			} else {
				return (N % vecLen);
			}
		}
	} else {
		printf("gemmHasNTail: Not handling Row Major - FIXME\n");
		return 0;
	}
}

int
gemmHasTails(size_t M,  size_t N, size_t K, int vecLen, clblasOrder order, clblasTranspose transA, clblasTranspose transB)
{
	K = K;  // Dummy- to remove warning
    if (order == clblasColumnMajor)
	{
		if (transA == clblasNoTrans)
		{
			if (transB == clblasNoTrans)
			{
				return (M % vecLen);
			} else {
				return ((M % vecLen) || (N % vecLen));
			}
		} else {
			if (transB == clblasNoTrans)
			{
				//
				// Vectoring on A is on K dimension and we handle tail directly in the kernel
				//
				return 0;
			} else {
				return (N % vecLen);
			}
		}
	} else {
		printf("gemmHasTails: Not handling Row Major - FIXME\n");
		return 0;
	}
}

clblasStatus executeGEMM( CLBlasKargs *kargs, cl_uint numCommandQueues, cl_command_queue *commandQueues, cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList, cl_event *events)
{
    cl_int err = CL_SUCCESS;
    ListHead seq, tailSeq;
	cl_event nontail;
	cl_uint gemmVeclen;
	CLBLASKernExtra *kextra;
    size_t M, N, K;

    M = kargs->M; N = kargs->N; K = kargs->K;
    #ifdef DEBUG_GEMM_2
    printf("executeGEMM Called\n");
    #endif
    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GEMM2, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, &nontail, &seq);
    if (err == CL_SUCCESS) {
	    ListNode *f = listNodeFirst(&seq);
		SolutionStep *gemm2;
		size_t tailStartM, tailStartN;
		bool processTails;

		gemm2 = container_of(f, node, SolutionStep);
		kextra = gemm2->kernels[CLBLAS_COMPUTING_KERNEL]->extra;
		gemmVeclen = kextra->vecLen;

		if (gemmHasTails(M, N, K, gemmVeclen, kargs->order, kargs->transA, kargs->transB) == 0)
		{
			#ifdef DEBUG_GEMM_2
			printf("No M or N Tails to process..\n");
			#endif
			processTails = false;
			gemm2->event = events;
		} else {
			processTails = true;
			if (gemmHasMTail(M, gemmVeclen, kargs->order, kargs->transA, kargs->transB))
			{
				tailStartM = M - (M%gemmVeclen);
			} else {
				tailStartM = M;
			}

			if (gemmHasNTail(N, gemmVeclen, kargs->order, kargs->transA, kargs->transB))
			{
				tailStartN = N - (N%gemmVeclen);
			} else {
				tailStartN = N;
            }
		}
        err = executeSolutionSeq(&seq);
		if ((err == CL_SUCCESS) && (processTails == true))
		{
			CLBlasKargs targs;

			memcpy(&targs, &gemm2->args, sizeof(CLBlasKargs));
			targs.tailStartM = tailStartM;
			targs.tailStartN = tailStartN;
			#ifdef DEBUG_GEMM_2
			printf("Processing Tails\n");
			#endif
    		listInitHead(&tailSeq);
    		err = makeSolutionSeq(CLBLAS_GEMM_TAIL, &targs, numCommandQueues, commandQueues,
        						  1, &nontail, events, &tailSeq);
			if (err == CL_SUCCESS)
			{
				err = executeSolutionSeq(&tailSeq);
			}
			freeSolutionSeq(&tailSeq);
		}
    }
    freeSolutionSeq(&seq);
    return (clblasStatus) err;
}

static clblasStatus
doGemm(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus err;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, B, C, true, A_MAT_ERRSET, B_MAT_ERRSET, C_MAT_ERRSET))) {
        return retCode;
    }
    if (K != 0) {
        if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, M, K, A, offA, lda, A_MAT_ERRSET))) {
            return retCode;
        }
        if ((retCode = checkMatrixSizes(kargs->dtype, order, transB, K, N, B, offB, ldb, B_MAT_ERRSET))) {
            return retCode;
        }
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, N, C, offC, ldc, C_MAT_ERRSET))) {
            return retCode;
    }

	numCommandQueues = 1;
	#ifdef DEBUG_2
	printf("DoGemm being called...\n");
	#endif
    kargs->pigFuncID = CLBLAS_GEMM2;
    kargs->order = order;
    kargs->transA = transA;
    kargs->transB = transB;
    kargs->M = M;
    kargs->N = N;
    kargs->K = K;
    kargs->A = A;
    kargs->offA = offA;
    kargs->offa = offA;
    kargs->lda.matrix = lda;
    kargs->B = B;
    kargs->offBX = offB;
    kargs->ldb.matrix = ldb;
    kargs->C = C;
    kargs->offCY = offC;
    kargs->ldc.matrix = ldc;

    kargs->offsetM = 0;
    kargs->offsetN = 0;
    kargs->scimage[0] = 0;
    kargs->scimage[1] = 0;

    err = executeGEMM(kargs, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    return err;
			}

/*
clblasStatus
clblasSgemmV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
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

    return doGemm(&kargs, order, transA, transB, M, N, K, A, 0, lda, B, 0, ldb,
                  C, 0, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDgemmV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.alpha.argDouble = alpha;
    kargs.beta.argDouble = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, 0, lda, B, 0, ldb,
                  C, 0, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCgemmV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, 0, lda, B, 0, ldb,
                  C, 0, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZgemmV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, 0, lda, B, 0, ldb,
                  C, 0, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasSgemmExV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
	size_t offA,
    size_t lda,
    const cl_mem B,
	size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
	size_t offC,
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

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda, B, offB, ldb,
                  C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDgemmExV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
	size_t offA,
    size_t lda,
    const cl_mem B,
	size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
	size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.alpha.argDouble = alpha;
    kargs.beta.argDouble = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda, B, offB, ldb,
                  C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCgemmExV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
	size_t offA,
    size_t lda,
    const cl_mem B,
	size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
	size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda, B, offB, ldb,
                  C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZgemmExV2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
	size_t offA,
    size_t lda,
    const cl_mem B,
	size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
	size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda, B, offB, ldb,
                  C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
*/
