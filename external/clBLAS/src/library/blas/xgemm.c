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


#include <string.h>
#include <clBLAS.h>
#include <stdlib.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

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
    cl_int err;
    ListHead seq;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, B, C, true, A_MAT_ERRSET, B_MAT_ERRSET, C_MAT_ERRSET))) {
        return retCode;
    }
    if (K != 0) {
        if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, M,
                                        K, A, offA, lda, A_MAT_ERRSET ))) {
            return retCode;
        }
        if ((retCode = checkMatrixSizes(kargs->dtype, order, transB,
                                        K, N, B, offB, ldb, B_MAT_ERRSET ))) {
            return retCode;
        }
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans,
                                    M, N, C, offC, ldc, C_MAT_ERRSET ))) {
            return retCode;
    }

	#ifdef DEBUG_2
	printf("DoGemm being called...\n");
	#endif
    kargs->order = order;
    kargs->transA = transA;
    kargs->transB = transB;
    kargs->M = M;
    kargs->N = N;
    kargs->K = K;
    kargs->A = A;
    kargs->offA = offA;
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

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GEMM, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);

    return (clblasStatus)err;
}


static ssize_t
TransposeKernel(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   void *extra)
{
/*
 *Transpose kernel generator
 *a typical kernel for mod4 sizes in both direction looks like below

                   "// micro tile size is 4 x 4 \n"
				   "// matrix are of column major \n"
                   "#pragma OPENCL EXTENSION cl_amd_printf : enable \n"
                   "void __kernel \n"
                   "transpose(  uint X, \n"
			       "uint Y, \n"
			       "uint ld, \n"
			       "uint offset, \n"
			       "const __global float *restrict mat, \n"
			       "__global float *transposed_mat) \n"
                   "{ \n"
				   "transposed_mat += offset; \n"
				   "mat += offset; \n"
		           "transposed_mat += ( (uint)get_global_id(1) * Y + (uint)get_global_id(0) ) << 2; \n"
		           "mat += ( (uint)get_global_id(0) * ld + (uint)get_global_id(1) ) << 2; \n"
		           "//transpose inside the block \n"
		           "transposed_mat[0] = mat[0]; \n"
		           "transposed_mat[1] = mat[ld]; \n"
		           "transposed_mat[2] = mat[ld*2]; \n"
		           "transposed_mat[3] = mat[ld*3]; \n"
				   "\n"
		           "transposed_mat[Y] = mat[1]; \n"
		           "transposed_mat[Y+1] = mat[1+ld]; \n"
		           "transposed_mat[Y+2] = mat[1+ld*2]; \n"
		           "transposed_mat[Y+3] = mat[1+ld*3]; \n"
				   "\n"
		           "transposed_mat[2*Y] = mat[2]; \n"
		           "transposed_mat[2*Y+1] = mat[2+ld]; \n"
		           "transposed_mat[2*Y+2] = mat[2+ld*2]; \n"
		           "transposed_mat[2*Y+3] = mat[2+ld*3]; \n"
				   "\n"
		           "transposed_mat[3*Y] = mat[3]; \n"
		           "transposed_mat[3*Y+1] = mat[3+ld]; \n"
		           "transposed_mat[3*Y+2] = mat[3+ld*2]; \n"
		           "transposed_mat[3*Y+3] = mat[3+ld*3]; \n"
                   "}";
*/
    struct KgenContext *ctx;
    ssize_t ret = 0;
    char tmp[2048];
    int modX = subdims->x;
    int modY = subdims->y;

    ctx = createKgenContext(buf, buflen, true);

    sprintf(tmp, "// micro tile size is 4 x 4 \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "// matrix are of column major \n");
    kgenAddStmt(ctx, tmp);

    //kernel declartion
    sprintf(tmp, "void __kernel \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "transpose(  uint X, \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint Y, \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint ld, \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint offset, \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "const __global float *restrict mat, \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "__global float *transposed_mat) \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "{ \n");
    kgenAddStmt(ctx, tmp);

    //kernel body
    sprintf(tmp, "uint global_id_0 = (uint)get_global_id(0); \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint global_id_1 = (uint)get_global_id(1); \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint global_size_0 = (uint)get_global_size(0); \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "uint global_size_1 = (uint)get_global_size(1); \n");
    kgenAddStmt(ctx, tmp);

    sprintf(tmp, "transposed_mat += offset; \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "mat += offset; \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "transposed_mat += ( global_id_1 * Y + global_id_0 ) << 2; \n");
    kgenAddStmt(ctx, tmp);
    sprintf(tmp, "mat += ( global_id_0 * ld + global_id_1 ) << 2; \n");
    kgenAddStmt(ctx, tmp);

    sprintf(tmp, "//transpose inside the block \n");
    kgenAddStmt(ctx, tmp);
    //first block
    sprintf(tmp, "transposed_mat[0] = mat[0]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[1] = mat[ld]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2 )
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[2] = mat[ld*2]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[3] = mat[ld*3]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "\n");
    kgenAddStmt(ctx, tmp);

    //second block
    if(modX == 1)
    {
        sprintf(tmp, "if( global_id_1 < global_size_1 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[Y] = mat[1]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[Y+1] = mat[1+ld]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[Y+2] = mat[1+ld*2]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[Y+3] = mat[1+ld*3]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "\n");
    kgenAddStmt(ctx, tmp);
    if(modX == 1)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }

	//third block
    if(modX == 1 || modX == 2)
    {
        sprintf(tmp, "if( global_id_1 < global_size_1 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[2*Y] = mat[2]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[2*Y+1] = mat[2+ld]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[2*Y+2] = mat[2+ld*2]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[2*Y+3] = mat[2+ld*3]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "\n");
    kgenAddStmt(ctx, tmp);
    if(modX == 1 || modX == 2)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
	}

	//fourth block
    if(modX == 1 || modX == 2 || modX == 3)
    {
        sprintf(tmp, "if( global_id_1 < global_size_1 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[3*Y] = mat[3]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[3*Y+1] = mat[3+ld]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY == 1)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[3*Y+2] = mat[3+ld*2]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "if( global_id_0 < global_size_0 - 1 ) \n");
        kgenAddStmt(ctx, tmp);
        sprintf(tmp, "{ \n");
        kgenAddStmt(ctx, tmp);
    }
    sprintf(tmp, "transposed_mat[3*Y+3] = mat[3+ld*3]; \n");
    kgenAddStmt(ctx, tmp);
    if(modY ==1 || modY == 2 || modY == 3)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }
    if(modX == 1 || modX == 2 || modX == 3)
    {
        sprintf(tmp, "} \n");
        kgenAddStmt(ctx, tmp);
    }


	sprintf(tmp, "} \n");
    kgenAddStmt(ctx, tmp);

    if (!ret) {
        ret = (ssize_t)kgenSourceSize(ctx) + 1;
    }
    destroyKgenContext(ctx);
	return ret;
}
void 
transposeMemObject(
    clblasOrder order,
    size_t X,
    size_t Y,
    size_t ld,
    size_t offset,
    const cl_mem src,
    cl_mem dst,
    cl_context context,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
/**
 *transposition of a 2D MemObject
 * @param[in] order     Row/column order.
 * @param[in] X         Number of columns in transposed matrix / rows in input matrix.
 * @param[in] Y         Number of rows in transposed matrix / columns in input matrix.
 * @param[in] ld        Leading dimension of input matrix.
 * @param[in] offset    offset size.
 * @param[in] src       Input matrix of the transposition.
 * @param[in] dst       Output matrix of the transposition.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 */


    char *source;
    cl_int err;
    cl_kernel kernel;
    Kernel *transpose_kernel;
    solver_id_t sid;
    KernelKey key;
    CLBLASKernExtra extra;
    BlasFunctionID funcID = CLBLAS_TRANSPOSE;
    char *log;
    const cl_uint workDim = 2;
    const size_t localWorkSize[2] = { 8, 8 };
    size_t globalWorkSize[2];
    ssize_t size;

    sid = makeSolverID(funcID, 1);
    memset(key.subdims, 0, sizeof(key.subdims));
    key.nrDims = 2;
    key.subdims[0].x = X%4;
    key.subdims[0].y = Y%4;
    key.subdims[0].bwidth = 1; 
    key.subdims[0].itemX = 4;
    key.subdims[0].itemY = 4;
    memset(&extra, 0, sizeof(extra));

    err = getQueueDevice(*commandQueues, &key.device);
    err = getQueueContext(*commandQueues, &key.context);

    //look for the kernel from cache first
    if (areKernelsCacheable()) 
    {
        transpose_kernel = findKernel(clblasKernelCache, sid, &key, &extra);
    }

    // if transpose_kernel was not found from cache, create the kernel
    if (transpose_kernel == NULL)
    {
        transpose_kernel = allocKernel();
        log = malloc(65536);
        if (log) {
            log[0] = '\0';
        }
        //kernel source auto generation
        //call size = TransposeKernel(NULL, 0, ...)
        //then allocate buffer and call TransposeKernel again
        size = TransposeKernel(NULL, 0, &key.subdims[0], &extra);
        source = calloc(1, size);
        TransposeKernel(source, size, &key.subdims[0], &extra);
        //printf("transpose source: %s\n", source);
        transpose_kernel->program = buildClProgram(source, NULL, key.context, key.device,
                                     log, 65536, &err);
        transpose_kernel->extraSize = sizeof(CLBLASKernExtra);
        transpose_kernel->extra = calloc(1, transpose_kernel->extraSize);// memory freed by clblasTeardown
        *(CLBLASKernExtra*)(transpose_kernel->extra) = extra;

        //save the kernel in cache 
        getKernel(transpose_kernel);
        if (addKernelToCache(clblasKernelCache, sid, transpose_kernel, &key,
                             clblasKernelExtraCmp)) 
        {
            putKernel(clblasKernelCache, transpose_kernel);
        }
        free(log);
        free(source);
    }

    //launch the kernel
    err = clCreateKernelsInProgram(transpose_kernel->program, 1, &kernel, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_uint), &X);
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &Y);
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &ld);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &offset);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &src);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &dst);

    globalWorkSize[0] = (Y+4-1)/4;
    globalWorkSize[1] = (X+4-1)/4;

    err = clEnqueueNDRangeKernel(*commandQueues, kernel, workDim, NULL,
        globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(*commandQueues);
	
    clReleaseKernel(kernel);


}

clblasStatus
clblasSgemm(
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
    cl_context context;
    cl_device_id device;
    cl_int err;
    cl_mem transposed_A;
    clblasStatus status;
    float *transposed_A_host;
    size_t device_size;
    char* device_name;
    int fast_sgemmtn = 0;


    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.alpha.argFloat = alpha;
    kargs.beta.argFloat = beta;

    err = clGetCommandQueueInfo( *commandQueues, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
    if( err < 0)
        return clblasInvalidCommandQueue;
    
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_size);
    device_name = (char*)malloc(device_size * sizeof(char));
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, device_size, device_name, NULL);
    if( err < 0)
        return clblasInvalidDevice;

    if(getenv("CLBLAS_FAST_SGEMM_TN") != NULL)
        fast_sgemmtn = *getenv("CLBLAS_FAST_SGEMM_TN") - '0';

	
	//if the env CLBLAS_FAST_SGEMMTN is set to 1
	//and if transA = T, transB = N and order = clblasColumnMajor 
	//and if the devices are Spectre, Hawaii or Tahiti
	//do the transpose first 
	//and then call the NN sgemm is a fater apporach. 
	//the cost of this approach is the use of an extra cl_mem object
	if( ( fast_sgemmtn == 1 ) && ( strcmp(device_name, "Spectre") || strcmp(device_name, "Hawaii") || strcmp(device_name, "Tahiti") )  && (transA == clblasTrans && transB == clblasNoTrans && order == clblasColumnMajor) )
	{
        //do the transpose on A
        //only transpose the leading part of the matrix
        //update ldb and transB would be necessary
        free(device_name);
        err = clGetCommandQueueInfo( *commandQueues, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);

        transposed_A = clCreateBuffer(context, CL_MEM_READ_WRITE, (M * K + offA) * sizeof(float), NULL, &err);
        if( err < 0 )
            return clblasOutOfResources;

        transposeMemObject(order, K, M, lda, offA, A, transposed_A, context, numCommandQueues, commandQueues,
							numEventsInWaitList, eventWaitList, events);

        //transA should be reset to clblasNoTrans
        transA = clblasNoTrans;
        //update lda to the minimal size 
        lda = M;
        //now call doGemm with transposed A, updated lda and updated transA
        status = doGemm(&kargs, order, transA, transB, M, N, K, transposed_A, offA, lda,
                  B, offB, ldb, C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
		
        clReleaseMemObject( transposed_A );
		
        return status;
    }
    else
    {
        free(device_name);
        return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda,
                  B, offB, ldb, C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
    }


}

clblasStatus
clblasDgemm(
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

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda,
                  B, offB, ldb, C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCgemm(
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

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda,
                  B, offB, ldb, C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZgemm(
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

    return doGemm(&kargs, order, transA, transB, M, N, K, A, offA, lda,
                  B, offB, ldb, C, offC, ldc, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
