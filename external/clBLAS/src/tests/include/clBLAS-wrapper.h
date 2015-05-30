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


#ifndef CLBLAS_WRAPPER_H_
#define CLBLAS_WRAPPER_H_

#include <clBLAS.h>
#include <cltypes.h>

namespace clMath {

class clblas {
public:
    // GEMV wrappers
    static clblasStatus
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        FloatComplex beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemv(
        clblasOrder order,
        clblasTranspose transA,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    // SYMV wrappers
    static clblasStatus
    symv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    symv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    // GEMM wrappers
    static clblasStatus
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemm(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemm(
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
        cl_event *events);

    static clblasStatus
    gemm(
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
        cl_event *events);

    static clblasStatus
    gemm2(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemm2(
        clblasOrder order,
        clblasTranspose transA,
        clblasTranspose transB,
        size_t M,
        size_t N,
        size_t K,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    gemm2(
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
        cl_event *events);

    static clblasStatus
    gemm2(
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
        cl_event *events);

    // TRMM wrappers
    static clblasStatus
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trmm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    // TRSM wrappers
    static clblasStatus
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    trsm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        clblasTranspose transA,
        clblasDiag diag,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        cl_mem B,
        size_t offB,
        size_t ldb,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    // SYR2K wrappers
    static clblasStatus
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transAB,
        size_t N,
        size_t K,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transAB,
        size_t N,
        size_t K,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        const cl_mem B,
        size_t offB,
        size_t ldb,
        double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transAB,
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
        cl_event *events);

    static clblasStatus
    syr2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transAB,
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
        cl_event *events);

    // SYRK wrappers
    static clblasStatus
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        FloatComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        FloatComplex beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    syrk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        DoubleComplex beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


    static clblasStatus
    trmv(
	    DataType type,
    	clblasOrder order,
    	clblasUplo uplo,
	    clblasTranspose trans,
	    clblasDiag diag,
    	size_t N,
    	const cl_mem A,
    	size_t ffa,
    	size_t lda,
    	cl_mem X,
    	size_t offx,
    	int incx,
	    cl_mem scratchBuff,
    	cl_uint numCommandQueues,
    	cl_command_queue *commandQueues,
    	cl_uint numEventsInWaitList,
    	const cl_event *eventWaitList,
    	cl_event *events);

	static clblasStatus
    trsv(
        DataType type,
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
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
        cl_event *events);

    static clblasStatus
    tpsv(
        DataType type,
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose trans,
        clblasDiag diag,
        size_t N,
        const cl_mem A,
        size_t offa,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

	static clblasStatus
	symm(
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
    	cl_event *events);

	static clblasStatus
    symm(
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
        cl_event *events);

	static clblasStatus
    symm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        FloatComplex beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

	static clblasStatus
    symm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        DoubleComplex beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

	static clblasStatus
	syr(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		float alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		cl_mem A,
		size_t offa,
		size_t lda,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

static clblasStatus
	syr(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		double alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		cl_mem A,
		size_t offa,
		size_t lda,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);


static clblasStatus
        ger(
            clblasOrder order,
            size_t M,
            size_t N,
            float alpha,
            const cl_mem X,
            size_t offx,
            int incx,
            const cl_mem Y,
            size_t offy,
            int incy,
            cl_mem A,
            size_t offa,
            size_t lda,
            cl_uint numCommandQueues,
            cl_command_queue *commandQueues,
            cl_uint numEventsInWaitList,
            const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    ger(
        clblasOrder order,
        size_t M,
        size_t N,
        double alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
	cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    ger(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    ger(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    gerc(
        clblasOrder order,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    her(
        clblasOrder order,
	clblasUplo uplo,
        size_t N,
        float alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    her(
        clblasOrder order,
	clblasUplo uplo,
        size_t N,
        double alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
	syr2(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		float alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		const cl_mem Y,
		size_t offy,
		int incy,
		cl_mem A,
		size_t offa,
		size_t lda,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

static clblasStatus
	syr2(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		double alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		const cl_mem Y,
		size_t offy,
		int incy,
		cl_mem A,
		size_t offa,
		size_t lda,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

//HER2 wrappers
 static clblasStatus
    her2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueue,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    her2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem A,
        size_t offa,
        size_t lda,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueue,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
        hemv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offa,
            size_t lda,
            const cl_mem X,
            size_t offx,
            int incx,
        FloatComplex beta,
            cl_mem Y,
        size_t offy,
            int incy,
        cl_uint numCommandQueues,
            cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
            const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    hemv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//HEMM
static clblasStatus
    hemm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        FloatComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        FloatComplex beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    hemm(
        clblasOrder order,
        clblasSide side,
        clblasUplo uplo,
        size_t M,
        size_t N,
        DoubleComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        DoubleComplex beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
	cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

// HERK wrappers
 static clblasStatus
    herk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        float alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        float beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
    herk(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        double alpha,
        const cl_mem A,
        size_t offA,
        size_t lda,
        double beta,
        cl_mem C,
        size_t offC,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

// TPMV wrappers
static  clblasStatus
	tpmv(
		DataType type,
		clblasOrder order,
		clblasUplo uplo,
		clblasTranspose trans,
		clblasDiag diag,
		size_t N,
		const cl_mem AP,
		size_t offa,
		cl_mem X,
		size_t offx,
		int incx,
		cl_mem scratchBuff,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueues,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

// SPMV wrappers
    static clblasStatus
    spmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        cl_float alpha,
        const cl_mem AP,
        size_t offa,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

    static clblasStatus
    spmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        cl_double alpha,
        const cl_mem AP,
        size_t offa,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

// HPMV wrappers
static clblasStatus
    hpmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        const cl_mem AP,
        size_t offa,
        const cl_mem X,
        size_t offx,
        int incx,
        FloatComplex beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    hpmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        const cl_mem AP,
        size_t offa,
        const cl_mem X,
        size_t offx,
        int incx,
        DoubleComplex beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

// SPR wrappers
static clblasStatus
	spr(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		float alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		cl_mem AP,
		size_t offa,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

static clblasStatus
	spr(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		double alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		cl_mem AP,
		size_t offa,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

// HPR wrappers
static clblasStatus
    hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        float alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_mem AP,
        size_t offa,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    hpr(
        clblasOrder order,
	    clblasUplo uplo,
        size_t N,
        double alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_mem AP,
        size_t offa,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

// SPR2 wrappers
static clblasStatus
	spr2(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		float alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		const cl_mem Y,
		size_t offy,
		int incy,
		cl_mem AP,
		size_t offa,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

static clblasStatus
	spr2(
		clblasOrder order,
		clblasUplo uplo,
		size_t N,
		double alpha,
		const cl_mem X,
		size_t offx,
		int incx,
		const cl_mem Y,
		size_t offy,
		int incy,
		cl_mem AP,
		size_t offa,
		cl_uint numCommandQueues,
		cl_command_queue *commandQueue,
		cl_uint numEventsInWaitList,
		const cl_event *eventWaitList,
		cl_event *events);

//HPR2 wrappers
 static clblasStatus
    hpr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        FloatComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem AP,
        size_t offa,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueue,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    hpr2(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        DoubleComplex alpha,
        const cl_mem X,
        size_t offx,
        int incx,
        const cl_mem Y,
        size_t offy,
        int incy,
        cl_mem AP,
        size_t offa,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueue,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_float alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_double alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_float2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_double2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    tbmv(
        DataType type,
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
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//SBMV

static clblasStatus
    sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_float alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_double alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//HBMV

static clblasStatus
    hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_float2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_double2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//TBSV

static clblasStatus
    tbsv(
        DataType type,
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
        //cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    her2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        cl_float2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        cl_float beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
    her2k(
        clblasOrder order,
        clblasUplo uplo,
        clblasTranspose transA,
        size_t N,
        size_t K,
        cl_double2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem B,
        size_t offb,
        size_t ldb,
        cl_double beta,
        cl_mem C,
        size_t offc,
        size_t ldc,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
    scal(
        bool is_css_zds,
        size_t N,
        cl_float alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
    scal(
        bool is_css_zds,
        size_t N,
        cl_double alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


 static clblasStatus
    scal(
        bool is_css_zds,
        size_t N,
        FloatComplex alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
    scal(
        bool is_css_zds,
        size_t N,
        DoubleComplex alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 //swap calls
 static clblasStatus
	swap(
        DataType type,
		size_t N,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


//copy
    static clblasStatus
    copy(
        DataType type,
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        //cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//DOT

static clblasStatus
    dot(
        DataType type,
        size_t N,
        cl_mem dotProduct,
        size_t offDP,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//ASUM
static clblasStatus
    asum(
        DataType type,
        size_t N,
        cl_mem asum,
        size_t offAsum,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//DOTC
static clblasStatus
    dotc(
        DataType type,
        size_t N,
        cl_mem dotProduct,
        size_t offDP,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 //axpy calls
 static clblasStatus
	axpy(
		size_t N,
        cl_float alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
	axpy(
		size_t N,
        cl_double alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
	axpy(
		size_t N,
        FloatComplex alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 static clblasStatus
	axpy(
		size_t N,
        DoubleComplex alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rotmg(
        DataType type,
        cl_mem D1,
        size_t offD1,
        cl_mem D2,
        size_t offD2,
        cl_mem X1,
        size_t offX1,
        cl_mem Y1,
        size_t offY1,
        cl_mem PARAM,
        size_t offParam,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rotm(
        DataType type,
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_mem PARAM,
        size_t offParam,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rotg(
        DataType type,
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
        cl_event *events);

static clblasStatus
    rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_float C,
        cl_float S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_double C,
        cl_double S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        FloatComplex C,
        FloatComplex S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

static clblasStatus
    rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        DoubleComplex C,
        DoubleComplex S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

//AMAX
static clblasStatus
    iamax(
        DataType type,
        size_t N,
        cl_mem iMax,
        size_t offiMax,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);


static clblasStatus
    nrm2(
        DataType type,
        size_t N,
        cl_mem NRM2,
        size_t offNRM2,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events);

 }; // clblas class
}   // namespace clMath

#endif  // CLBLAS_WRAPPER_H_

