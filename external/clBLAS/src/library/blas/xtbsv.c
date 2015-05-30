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
#include <stdlib.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

//#define DEBUG_TBSV
static clblasUplo

getUpLo(CLBlasKargs *kargs)
{
    if (kargs->order == clblasRowMajor)
    {
        return kargs->uplo;
    }

    if (kargs->uplo == clblasUpper)
    {
        return clblasLower;
    }
    return clblasUpper;
}

static clblasStatus
orchestrateNonTransposeTBSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gbmvSeq, cl_uint numEventsInWaitList,
                const cl_event *eventWaitList, cl_event *events)
{
    clblasStatus err;
    SolutionStep *trtri, *gbmv;
    size_t nLoops, i;
    cl_event *triangleEventArray, *rectangleEventArray;
    size_t TARGET_ROWS;
    bool gbmvExecute;
    size_t temp;

    ListNode *f = listNodeFirst(trtriSeq);
    trtri = container_of(f, node, SolutionStep);
    f = listNodeFirst(gbmvSeq);
    gbmv = container_of(f, node, SolutionStep);

    TARGET_ROWS = trtri->subdims->y;
    TARGET_ROWS = (TARGET_ROWS > kargs->K) ? kargs->K : TARGET_ROWS;
    TARGET_ROWS = (TARGET_ROWS == 0) ? 1 : TARGET_ROWS;

    trtri->numEventsInWaitList = numEventsInWaitList;
    trtri->eventWaitList = eventWaitList;

    if (kargs->N <= TARGET_ROWS)
    {
        trtri->event = events;
        trtri->args.startRow = 0;
        trtri->args.endRow = (cl_int)((kargs->N)-1);
        err = executeSolutionSeq(trtriSeq);
        return err;
    }

    //
    // Allocate Event Chain
    //
    nLoops = ((kargs->N) / TARGET_ROWS);
    if ((kargs->N % TARGET_ROWS))
    {
        nLoops++;
    }
    //
    // Allocate Event Arrays to order the orchestration
    //
    triangleEventArray = malloc(nLoops*sizeof(cl_event));
    rectangleEventArray = malloc(nLoops*sizeof(cl_event));
    if ((triangleEventArray == NULL) || (rectangleEventArray == NULL))
    {
        if (triangleEventArray)
        {
            free (triangleEventArray);
        }
        if (rectangleEventArray)
        {
            free (rectangleEventArray);
        }
        return clblasOutOfHostMemory;
    }
    //
    //  Solve 1 Triangle using Triangle Kernel Followed by Rectangle Kernels
    //
    trtri->event = &triangleEventArray[0];
    if (getUpLo(kargs) == clblasUpper)
    {
        trtri->args.startRow = (cl_int)((kargs->N) - TARGET_ROWS);
        trtri->args.endRow = (cl_int)((kargs->N)-1);
    } else {
        trtri->args.startRow = 0;
        trtri->args.endRow = (cl_int)(TARGET_ROWS-1);
    }
    err = executeSolutionSeq(trtriSeq);

/*#define GET_OFFA(offa, lda, r, c, k)\
if(r < k) \
offa = r * lda + col + k - r;\
else if (r == k) \
offa = r * lda + col;\
else\
offa = r * lda + col - (r - k);
*/
#define GET_OFFA_LOWER(offa, lda, row, col, kl) (offa) = ((row) * (lda)) + (col) + (kl) - (row);
#define GET_OFFA_UPPER(offa, lda, row, col) (offa) = ((row) * (lda)) + (col) - (row);

if (err == CL_SUCCESS)
    {
        //
        // Solve the Rectangles one by one
        //
        //nLoops = 1;
        for(i=1; i<nLoops; i++)
        {
            #ifdef DEBUG_TBSV
                printf("Calling gbmv-");
            #endif
            gbmv->numEventsInWaitList = 1;
            gbmv->eventWaitList = &triangleEventArray[i-1];
            gbmv->event = &rectangleEventArray[i-1];

            if (getUpLo(kargs) == clblasUpper)
            {
                gbmv->args.N = TARGET_ROWS;
                gbmv->args.M = ((trtri->args.startRow) >= (int)(kargs->K)) ? kargs->K : (size_t)trtri->args.startRow;
                gbmv->args.startRow = (trtri->args.startRow - gbmv->args.M);
                gbmv->args.endRow = (trtri->args.startRow - 1);
                gbmv->args.KU = (trtri->args.startRow >= (int)(kargs->K)) ? 0 : (kargs->K - trtri->args.startRow);
                gbmv->args.KL = gbmv->args.M - 1;

                GET_OFFA_UPPER(gbmv->args.offA, kargs->lda.matrix, gbmv->args.startRow, trtri->args.startRow);
                gbmv->args.offA -= gbmv->args.KL;
                gbmv->args.offA += kargs->offA;
                gbmv->args.offa = gbmv->args.offA;

                if(kargs->ldb.vector < 0)
                {
                    gbmv->args.offBX = kargs->offBX + ((i-1) * TARGET_ROWS) * abs(kargs->ldb.vector);
                    gbmv->args.offCY = kargs->offBX + ((i * TARGET_ROWS) ) * abs(kargs->ldb.vector);
                }
                else
                {
                    gbmv->args.offBX = kargs->offBX + (trtri->args.startRow) * kargs->ldb.vector;
                    gbmv->args.offCY = kargs->offBX + (gbmv->args.startRow) * kargs->ldb.vector;
                }

            } else {
                gbmv->args.startRow = (cl_int)((i)*TARGET_ROWS);
                gbmv->args.endRow   = (cl_int)((((TARGET_ROWS*i) + kargs->K) > kargs->N) ? kargs->N : (TARGET_ROWS*i + kargs->K));
                gbmv->args.N = TARGET_ROWS;
                gbmv->args.M = (gbmv->args.endRow - gbmv->args.startRow);
                gbmv->args.KU = TARGET_ROWS - 1;
                gbmv->args.KL = ((trtri->args.startRow + kargs->K) < kargs->N) ? (kargs->K - TARGET_ROWS) : (kargs->N - trtri->args.startRow - 1 - TARGET_ROWS);

                GET_OFFA_LOWER(gbmv->args.offA, kargs->lda.matrix, gbmv->args.startRow, trtri->args.startRow, kargs->K);
                gbmv->args.offA -= gbmv->args.KL;
                gbmv->args.offA += kargs->offA;
                gbmv->args.offa = gbmv->args.offA;
                if(kargs->ldb.vector < 0)
                {
                    gbmv->args.offBX = kargs->offBX + (kargs->N - gbmv->args.startRow) * abs(kargs->ldb.vector);
                    gbmv->args.offCY = kargs->offBX + (kargs->N - (gbmv->args.startRow + gbmv->args.M) ) * abs(kargs->ldb.vector);
                }
                else
                {
                    gbmv->args.offBX = kargs->offBX + (trtri->args.startRow) * kargs->ldb.vector;
                    gbmv->args.offCY = kargs->offBX + (gbmv->args.startRow) * kargs->ldb.vector;
                }

            }

            #ifdef DEBUG_TBSV
            printf("GBMV ITER %d, startRow %d, endRow %d, N %d, M %d , KU %d, KL %d, offBX %d, offA %d, offCY %d\n", i-1, gbmv->args.startRow, gbmv->args.endRow, \
                                            gbmv->args.N, gbmv->args.M, gbmv->args.KU, gbmv->args.KL, gbmv->args.offBX, gbmv->args.offA, gbmv->args.offCY);
            #endif
            // This is required when KL or KU is 0 for TBSV.
            gbmvExecute = (gbmv->args.M != 0);
            if(gbmvExecute)
            {
                if(kargs->order == clblasColumnMajor) //GBMV Swaps it back while assigning
                {
                    temp = gbmv->args.N;
                    gbmv->args.N = gbmv->args.M;
                    gbmv->args.M = temp;
                    temp = gbmv->args.KU;
                    gbmv->args.KU = gbmv->args.KL;
                    gbmv->args.KL = temp;
                }
                err = executeSolutionSeq(gbmvSeq);
            }

            if (err != CL_SUCCESS)
            {
                printf("TBSV: WARNING: GBMV LOOP: Breaking after %d iterations  !!!\n", (int)i);
                break;
            }

            #ifdef DEBUG_TBSV
                printf("Calling TBSV\n");
            #endif
            if (getUpLo(kargs) == clblasUpper)
            {
                trtri->args.startRow = (cl_int)(((int)trtri->args.startRow - (int)TARGET_ROWS) >= 0) ? (trtri->args.startRow - TARGET_ROWS) : 0;
                trtri->args.endRow = (cl_int)(gbmv->args.endRow);
            } else {
                trtri->args.startRow = gbmv->args.startRow;
                trtri->args.endRow = (cl_int)(((gbmv->args.startRow + TARGET_ROWS-1) < kargs->N) ? (gbmv->args.startRow + TARGET_ROWS-1) : kargs->N-1);
            }
            #ifdef DEBUG_TBSV
            printf("TRSV ITER %d, startRow %d , endRow %d\n", i, trtri->args.startRow, trtri->args.endRow);
            #endif
            trtri->event = &triangleEventArray[i];
            if (i == (nLoops-1))
            {
                //
                // TRTRI's last iteration must be tied to the "event" that the API
                // user will choose to wait on.
                //
                trtri->event = events;
            }
            //
            // For first iteration, TRTRI waits on what the API user has specified.
            // Subsequent iterations will wait on the previous iteration's rectangle
            // counterpart
            //

            trtri->numEventsInWaitList =1;
            if(gbmvExecute)
            {
                trtri->eventWaitList = &rectangleEventArray[i-1];
            }
            else //GBMV is not executed when KL or KU of the band in TBSV is 0.
            {
                trtri->eventWaitList = &triangleEventArray[i-1];
            }

            err = executeSolutionSeq(trtriSeq);
            if (err != CL_SUCCESS)
            {
                printf("TBSV: WARNING: TRSV LOOP: Breaking after %d iterations  !!!\n", (int)i);
                break;
            }

        }
    }

    free(triangleEventArray);
    free(rectangleEventArray);
    return err;
}

static clblasStatus
orchestrateTransposeTBSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gbmvSeq, cl_uint numEventsInWaitList,
                const cl_event *eventWaitList, cl_event *events)
{
    clblasStatus err;
    SolutionStep *trtri, *gbmv;
    size_t nLoops, i;
    cl_event *triangleEventArray, *rectangleEventArray;
    size_t TARGET_ROWS;
    bool gbmvExecute;
    size_t temp;
    int TR_ER, N_SR, SD_ER;

    ListNode *f = listNodeFirst(trtriSeq);
    trtri = container_of(f, node, SolutionStep);
    f = listNodeFirst(gbmvSeq);
    gbmv = container_of(f, node, SolutionStep);

    TARGET_ROWS = trtri->subdims->y;
    TARGET_ROWS = (TARGET_ROWS > kargs->K) ? kargs->K : TARGET_ROWS;
    TARGET_ROWS = (TARGET_ROWS == 0) ? 1 : TARGET_ROWS;

    trtri->numEventsInWaitList = numEventsInWaitList;
    trtri->eventWaitList = eventWaitList;

    if (kargs->N <= TARGET_ROWS)
    {
        trtri->event = events;
        trtri->args.startRow = 0;
        trtri->args.endRow = (cl_int)((kargs->N));
        err = executeSolutionSeq(trtriSeq);
        return err;
    }

    //
    // Allocate Event Chain
    //
    nLoops = ((kargs->N) / TARGET_ROWS);
    if ((kargs->N % TARGET_ROWS))
    {
        nLoops++;
    }
    //
    // Allocate Event Arrays to order the orchestration
    //
    triangleEventArray = malloc(nLoops*sizeof(cl_event));
    rectangleEventArray = malloc(nLoops*sizeof(cl_event));
    if ((triangleEventArray == NULL) || (rectangleEventArray == NULL))
    {
        if (triangleEventArray)
        {
            free (triangleEventArray);
        }
        if (rectangleEventArray)
        {
            free (rectangleEventArray);
        }
        return clblasOutOfHostMemory;
    }
    //
    //  Solve 1 Triangle using Triangle Kernel Followed by Rectangle Kernels
    //
    trtri->event = &triangleEventArray[0];
    if (getUpLo(kargs) == clblasUpper)
    {
        trtri->args.startRow = 0;
        trtri->args.endRow = (cl_int)(TARGET_ROWS);
    } else {
        trtri->args.startRow = (cl_int)((kargs->N) - TARGET_ROWS);
        trtri->args.endRow = (cl_int)((kargs->N));
    }
    err = executeSolutionSeq(trtriSeq);

/*#define GET_OFFA(offa, lda, r, c, k)\
if(r < k) \
offa = r * lda + col + k - r;\
else if (r == k) \
offa = r * lda + col;\
else\
offa = r * lda + col - (r - k);
*/
#define GET_OFFA_LOWER(offa, lda, row, col, kl) (offa) = ((row) * (lda)) + (col) + (kl) - (row);
#define GET_OFFA_UPPER(offa, lda, row, col) (offa) = ((row) * (lda)) + (col) - (row);

    if (err == CL_SUCCESS)
    {
        //
        // Solve the Rectangles one by one
        //
        //nLoops = 1;
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))


        for(i=1; i<nLoops; i++)
        {
            #ifdef DEBUG_TBSV
                printf("Calling gbmv-");
            #endif
            gbmv->numEventsInWaitList = 1;
            gbmv->eventWaitList = &triangleEventArray[i-1];
            gbmv->event = &rectangleEventArray[i-1];

            if (getUpLo(kargs) == clblasUpper)
            {
                TR_ER = trtri->args.endRow - 1;
                gbmv->args.N = max(0, min(((int)kargs->K), ((int)kargs->N - 1 - TR_ER)));
                gbmv->args.M = TARGET_ROWS;
                gbmv->args.startRow = (trtri->args.startRow);
                gbmv->args.endRow = trtri->args.endRow;
                N_SR = max(0, min(((int)kargs->K), ((int)kargs->N - 1 - (int)trtri->args.startRow)));
                gbmv->args.KU = N_SR - TARGET_ROWS;
                gbmv->args.KL = gbmv->args.M - 1;

                GET_OFFA_UPPER(gbmv->args.offA, kargs->lda.matrix, gbmv->args.startRow, (gbmv->args.endRow));
                gbmv->args.offA -= gbmv->args.KL;
                gbmv->args.offA += kargs->offA;
                gbmv->args.offa = gbmv->args.offA;

                if(kargs->ldb.vector < 0)
                {
                    gbmv->args.offBX = kargs->offBX + (kargs->N - (gbmv->args.endRow)) * abs(kargs->ldb.vector);
                    gbmv->args.offCY = kargs->offBX + (kargs->N - (gbmv->args.endRow + gbmv->args.N) ) * abs(kargs->ldb.vector);
                }
                else
                {
                    gbmv->args.offBX = kargs->offBX + (gbmv->args.startRow) * kargs->ldb.vector;
                    gbmv->args.offCY = kargs->offBX + (gbmv->args.endRow) * kargs->ldb.vector;
                }


            } else {

#define SUBDIAGS(r, k) ((r) <= (k)) ? (r) : (k);
                gbmv->args.startRow = trtri->args.startRow;
                gbmv->args.endRow   = trtri->args.endRow;

                gbmv->args.N = SUBDIAGS((int)trtri->args.startRow, (int)kargs->K);
                gbmv->args.M = TARGET_ROWS;
                gbmv->args.KU = gbmv->args.N - 1;
                SD_ER = SUBDIAGS((int)(trtri->args.endRow - 1), (int)kargs->K);
                gbmv->args.KL = SD_ER - gbmv->args.N;

                GET_OFFA_LOWER(gbmv->args.offA, kargs->lda.matrix, gbmv->args.startRow, (gbmv->args.startRow - gbmv->args.N), kargs->K);
                gbmv->args.offA -= gbmv->args.KL;
                gbmv->args.offA += kargs->offA;
                gbmv->args.offa = gbmv->args.offA;
                if(kargs->ldb.vector < 0)
                {
                    gbmv->args.offBX = kargs->offBX + (kargs->N - gbmv->args.endRow) * abs(kargs->ldb.vector);
                    gbmv->args.offCY = kargs->offBX + (kargs->N - (gbmv->args.startRow) ) * abs(kargs->ldb.vector);
                }
                else
                {
                    gbmv->args.offBX = kargs->offBX + (gbmv->args.startRow) * kargs->ldb.vector;
                    gbmv->args.offCY = kargs->offBX + (gbmv->args.startRow - gbmv->args.N) * kargs->ldb.vector;
                }

            }
            #ifdef DEBUG_TBSV
            printf("GBMV ITER %d, startRow %d, endRow %d, N %d, M %d , KU %d, KL %d, offBX %d, offA %d, offCY %d\n", i-1, gbmv->args.startRow, gbmv->args.endRow, \
                                            gbmv->args.N, gbmv->args.M, gbmv->args.KU, gbmv->args.KL, gbmv->args.offBX, gbmv->args.offA, gbmv->args.offCY);
            #endif
            // This is required when KL or KU is 0 for TBSV.
            gbmvExecute = (gbmv->args.N != 0);
            if(gbmvExecute)
            {
                if(kargs->order == clblasColumnMajor) //GBMV Swaps it back while assigning
                {
                    temp = gbmv->args.N;
                    gbmv->args.N = gbmv->args.M;
                    gbmv->args.M = temp;
                    temp = gbmv->args.KU;
                    gbmv->args.KU = gbmv->args.KL;
                    gbmv->args.KL = temp;
                }
                err = executeSolutionSeq(gbmvSeq);
            }

            if (err != CL_SUCCESS)
            {
                printf("TBSV: WARNING: GBMV LOOP: Breaking after %d iterations  !!!\n", (int)i);
                break;
            }

            #ifdef DEBUG_TBSV
                printf("Calling TBSV\n");
            #endif
            if (getUpLo(kargs) == clblasUpper)
            {
                trtri->args.startRow = (cl_int)(trtri->args.endRow);
                trtri->args.endRow = (cl_int)(((int)trtri->args.endRow + (int)TARGET_ROWS) <= (int)kargs->N) ? (trtri->args.endRow + TARGET_ROWS) : kargs->N;
            } else {
                trtri->args.endRow = trtri->args.startRow;
                trtri->args.startRow = (cl_int)((((int)trtri->args.startRow - (int)TARGET_ROWS) > 0) ? (trtri->args.startRow - TARGET_ROWS) : 0);
            }
            #ifdef DEBUG_TBSV
            printf("TRSV ITER %d, startRow %d , endRow %d\n", i, trtri->args.startRow, trtri->args.endRow);
            #endif
            trtri->event = &triangleEventArray[i];
            if (i == (nLoops-1))
            {
                //
                // TRTRI's last iteration must be tied to the "event" that the API
                // user will choose to wait on.
                //
                trtri->event = events;
            }
            //
            // For first iteration, TRTRI waits on what the API user has specified.
            // Subsequent iterations will wait on the previous iteration's rectangle
            // counterpart
            //

            trtri->numEventsInWaitList =1;
            if(gbmvExecute)
            {
                trtri->eventWaitList = &rectangleEventArray[i-1];
            }
            else //GBMV is not executed when KL or KU of the band in TBSV is 0.
            {
                trtri->eventWaitList = &triangleEventArray[i-1];
            }

            err = executeSolutionSeq(trtriSeq);
            if (err != CL_SUCCESS)
            {
                printf("TBSV: WARNING: TRSV LOOP: Breaking after %d iterations  !!!\n", (int)i);
                break;
            }

        }
    }

    free(triangleEventArray);
    free(rectangleEventArray);
    return err;
}

static clblasStatus
orchestrateTBSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gbmvSeq, cl_uint numEventsInWaitList,
                const cl_event *eventWaitList, cl_event *events)
{
    clblasStatus err = clblasNotImplemented;

    if  (   ((kargs->order == clblasRowMajor) && (kargs->transA == clblasNoTrans)) ||
            ((kargs->order == clblasColumnMajor) && (kargs->transA != clblasNoTrans))
        )
    {
        #ifdef DEBUG_TBSV
        printf("Orchestrating the NO-Transpose case..\n");
        #endif
        err = orchestrateNonTransposeTBSV(kargs, trtriSeq, gbmvSeq, numEventsInWaitList, eventWaitList, events);
    } else {
        #ifdef DEBUG_TRSV
        printf("Orchestrating the Transpose case..\n");
        #endif
        err = orchestrateTransposeTBSV(kargs, trtriSeq, gbmvSeq, numEventsInWaitList, eventWaitList, events);
    }

    return err;
}


clblasStatus
doTbsv(
	CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem x,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err = clblasNotImplemented;
    ListHead seq;
	CLBlasKargs gbmvKargs;
	ListHead gbmvSeq;
	//cl_context c;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    retCode = checkMemObjects(A, x, (cl_mem) NULL, false, A_MAT_ERRSET, X_VEC_ERRSET, END_ERRSET);
    if (retCode != clblasSuccess) {
		#ifdef DEBUG_TBSV
		printf("Invalid mem object..\n");
		#endif
        return retCode;
    }

	/*
	 * PENDING:
 	 * checkMatrixSizes() does not account for "offa" argument.
 	 * Need to pass "offa" when "checkMatrixSizes()" is changed.
	 */
    retCode = checkBandedMatrixSizes(kargs->dtype, order, trans, N, N, K, 0, A, offa, lda, A_MAT_ERRSET );
    if (retCode != clblasSuccess) {
		#ifdef DEBUG_TBSV
		printf("Invalid Size for A\n");
		#endif
        return retCode;
    }
    retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET );
    if (retCode != clblasSuccess) {
		#ifdef DEBUG_TBSV
		printf("Invalid Size for X\n");
		#endif
        return retCode;
    }

	#ifdef DEBUG_TBSV
	printf("DoTbsv being called...\n");
	#endif

	if ((commandQueues == NULL) || (numCommandQueues == 0))
	{
		return clblasInvalidValue;
	}

	if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
	{
		return clblasInvalidEventWaitList;
	}

    if (commandQueues[0] == NULL)
	{
		return clblasInvalidCommandQueue;
	}


	numCommandQueues = 1; // NOTE: Hard-coding the number of command queues to 1i
    kargs->order = order;
    kargs->uplo = uplo;
    kargs->transA = trans;
	kargs->diag = diag;
    kargs->M = N; // store Original N
    kargs->N = N;
    kargs->K = K;
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->ldb.vector = incx;
    kargs->offBX = offx;
	kargs->offa = offa;
	kargs->offA = offa;
    kargs->C = x;
    kargs->offCY = offx;
    kargs->ldc.vector = incx;
    kargs->startRow = 0;

    if(trans == clblasNoTrans)
    {
        kargs->endRow = (order == clblasRowMajor) ?  N-1 : N;
    }
    else
    {
        kargs->endRow = (order == clblasRowMajor) ?  N : N-1;
    }

    memcpy(&gbmvKargs, kargs, sizeof(CLBlasKargs));
    gbmvKargs.pigFuncID = CLBLAS_GBMV;

    listInitHead(&seq);
    listInitHead(&gbmvSeq);

    err = makeSolutionSeq(CLBLAS_TRSV, kargs, numCommandQueues, commandQueues,
                          numEventsInWaitList, eventWaitList, events, &seq);

    if (err == CL_SUCCESS) {

        err = makeSolutionSeq(CLBLAS_GBMV, &gbmvKargs, numCommandQueues, commandQueues,
                                0, NULL, NULL, &gbmvSeq);
        if (err == CL_SUCCESS)
        {
            err = orchestrateTBSV(kargs, &seq, &gbmvSeq, numEventsInWaitList, eventWaitList, events);
        }
    }

    freeSolutionSeq(&seq);
    freeSolutionSeq(&gbmvSeq);
    return (clblasStatus)err;
}

clblasStatus
clblasStbsv(
clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
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
    #ifdef DEBUG_TBSV
    printf("STBSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TBSV;
    kargs.alpha.argFloat = -1.0;
    kargs.beta.argFloat = 1.0;

    return doTbsv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
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
    #ifdef DEBUG_TBSV
    printf("DTBSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TBSV;
    kargs.alpha.argDouble = -1.0;
    kargs.beta.argDouble = 1.0;

    return doTbsv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
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
    FloatComplex alpha, beta;
    #ifdef DEBUG_TBSV
    printf("CTBSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TBSV;

    CREAL(alpha) = -1.0;
    CIMAG(alpha) = 0.0;
    CREAL(beta) = 1.0;
    CIMAG(beta) = 0.0;

    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;

    return doTbsv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
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
    DoubleComplex alpha, beta;
    #ifdef DEBUG_TBSV
    printf("ZTBSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TBSV;

    CREAL(alpha) = -1.0;
    CIMAG(alpha) = 0.0;
    CREAL(beta) = 1.0;
    CIMAG(beta) = 0.0;

    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;

    return doTbsv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

