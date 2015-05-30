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

// TRSV Column-Major Upper Kernel
//#include <TRSV.h>


const char * trsv_CU_SolveTriangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#ifdef PACKED
    #define A( row, col) (*( A + (((col)*((col)+1))/2 + (row))))
#elif defined(BANDED)
    #define A( row, col) A[ (row) * lda + (col)]
#else
    #define A( row, col) A[ (row) + (col) * lda]
#endif
// Only one workgroup of threads launched
__kernel void %PREFIXtrsv_CU_SolveTriangle_kernel( __global %TYPE const * restrict _A, __global %TYPE* _xnew, uint N, int incx, int isUnity,
                                                   uint lda, int doConj, int startRow, int startCol, uint offa, uint offx
#ifdef BANDED
                                                   , uint KU
#endif
                                                 )
{
    __global %TYPE* xnew;
    __global %TYPE const * restrict A = _A + offa;

    if ( incx < 0 ) // Goto end of vector
    {
        xnew     = _xnew + offx + ( N - 1) * abs(incx);
    }
    else
    {
        xnew     = _xnew + offx;
    }

    __local %TYPE  xShared; // To share solved x value with other threads..

    size_t gIdx     = get_global_id(0);
    size_t bIdx     = get_group_id(0);
    size_t threadIdx= get_local_id(0);

    %TYPE sum     = %MAKEVEC(0.0);
    %TYPE xVal    = %MAKEVEC(0.0);
    %TYPE loadedA     = %MAKEVEC(0.0);

    int targetCol     = startCol;
    int targetRow     = startRow + threadIdx;
    int loops     = (startCol - startRow) + 1;

#ifdef BANDED
    int bandCol = (loops - 1) - threadIdx;
#endif

    for( int i=0; i < loops; i++)
    {
        if ( targetRow == targetCol)
        {
            xVal = xnew[ targetRow * incx];
            %SUB(sum, xVal, sum);

            if( isUnity)
            {
                xShared = sum;
            }
            else // Handle diagonal element
            {
#ifdef BANDED
                loadedA = A((targetRow), (bandCol));
#else
                loadedA = A((targetRow), (targetCol));
#endif
                %CONJUGATE(doConj, loadedA);
                %DIV(xShared, sum, loadedA);
            }
            xnew[ targetRow * incx ] = xShared;
        }
        // Sync so that xShared it available to all threads
        barrier(CLK_LOCAL_MEM_FENCE);

        if ( targetRow < targetCol)
        {
#ifdef BANDED
            loadedA = A((targetRow), (bandCol));
#else
            loadedA = A((targetRow), (targetCol));
#endif
            %CONJUGATE(doConj, loadedA);
            %MAD(sum, loadedA, xShared);
        }

        // Avoid Race...
        barrier(CLK_LOCAL_MEM_FENCE);
        targetCol--;
#ifdef BANDED
        bandCol--;
#endif
    }
}";


const char * trsv_CL_SolveTriangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
    #define A(row, col) (*( A + ((( (col) *((2*N) + 1 - (col))) / 2) + ((row) - (col)))))
#elif defined(BANDED)
    #define A(row, col) A[ (row) * lda + (col)]
#else
    #define A(row, col) A[ (row) + (col) * lda]
#endif
#pragma OPENCL EXTENSION cl_amd_printf : enable
// Only one block of threads launched
__kernel void %PREFIXtrsv_CL_SolveTriangle_kernel( __global const %TYPE* _A, __global %TYPE* _xnew, uint N, int incx, int isUnity,
                                                   uint lda, int doConj, int startCol, int endRow, uint offa, uint offx
#ifdef BANDED
                                                   , uint KL
#endif
                                                            )
{
    __global %TYPE* xnew;
    __global %TYPE* A = _A + offa;

    if ( incx < 0 ) // Goto end of vector
    {
        xnew     = _xnew + offx + ( N - 1) * abs(incx);
    }
    else
    {
        xnew     = _xnew + offx;
    }

    __local %TYPE  xShared; // To share solved x value with other threads..

    size_t gIdx     = get_global_id(0);
    size_t bIdx     = get_group_id(0);
    size_t threadIdx= get_local_id(0);

    %TYPE sum     = %MAKEVEC(0.0);
    %TYPE xVal    = %MAKEVEC(0.0);
    %TYPE loadedA     = %MAKEVEC(0.0);

    int targetCol     = startCol;
    int targetRow     = endRow - threadIdx;
    int loops     = (endRow - startCol) + 1;
#ifdef BANDED
    int bandCol = (KL + 1) - loops + threadIdx;
#endif

//    printf(\"%u : bandCol %d targetCol %d targetRow %d loops %d KL %d\\n\", threadIdx, bandCol, targetCol, targetRow, loops, KL);

    for( int i=0; i < loops; i++)
    {
        if ( targetRow == targetCol)
        {
            xVal = xnew[ targetRow * incx];
            //printf(\"Before1 %u : xShared %f, sum %f\\n\", threadIdx, xShared, sum);
            %SUB(sum, xVal, sum);
            //printf(\"Before2 %u : xShared %f, sum %f XvAL %f, targetRow %d\\n\", threadIdx, xShared, sum, xVal, targetRow);

            if( isUnity)
            {
                xShared = sum;
            }
            else // Handle diagonal element
            {
#ifndef BANDED
                loadedA = A((targetRow), (targetCol));
#else
                loadedA = A((targetRow), (bandCol));
#endif
                %CONJUGATE(doConj, loadedA);
                %DIV(xShared, sum, loadedA);
            }
            //printf(\"After %u : xShared %f, sum %f\\n\", threadIdx, xShared, sum);
            xnew[ targetRow * incx ] = xShared;
        }
        // Sync so that xShared it available to all threads
        barrier(CLK_LOCAL_MEM_FENCE);

        if ( targetRow > targetCol)
        {
#ifndef BANDED
                loadedA = A((targetRow), (targetCol));
#else
                loadedA = A((targetRow), (bandCol));
#endif
            %CONJUGATE(doConj, loadedA);
            //printf(\"%u : xShared %f, sum %f loadedA %f\\n\", threadIdx, xShared, sum, loadedA);
            %MAD(sum, loadedA, xShared);
        }

        // Avoid Race...
        barrier(CLK_LOCAL_MEM_FENCE);
        targetCol++;
#ifdef BANDED
        bandCol++;
#endif
    }
}
";

const char * trsv_CUT_SolveTriangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
    #define A( row, col) (*( A + (((col)*((col)+1))/2 + (row))))
#elif defined(BANDED)
    #define A(row, col) A[ (row) * lda + (col)]
#else
    #define A( row, col) A[ (row) + (col) * lda]
#endif
#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void %PREFIXtrsv_CUT_SolveTriangle_kernel(     __global const %TYPE* _A,
                                                __global %TYPE* _xnew,
                                                uint N,
                                                int incx,
                                                int isUnity,
                                                uint lda,
                                                int doConj,
                                                int startRow, int endRow, uint offa, uint offx
#ifdef BANDED
                                                , uint KU
#endif
                                                         )
{
        __global %TYPE* xnew;
        __global const %TYPE* A = _A + offa;
        if ( incx < 0 ) // Goto end of vector
        {
            xnew     = _xnew + offx  + ( N - 1) * abs(incx);
        }
        else
        {
            xnew     = _xnew + offx;
        }

        int blockSize = get_local_size(0);
        int threadID = get_local_id(0);
        int targetRow;
#ifdef BANDED
        int bandRow = startRow;
        int bandCol = threadID;
//        printf(\"threadID %d, bandRow %d bandCol %d\\n\",threadID, bandRow, bandCol);
#endif
        __local volatile %TYPE saccShared[%TRIANGLE_HEIGHT];

        targetRow = startRow + threadID;
        //#pragma unroll
        for( int idx = threadID; (idx < %TRIANGLE_HEIGHT) && ((startRow + idx) < endRow); idx += blockSize)
        {
            saccShared[idx] = xnew[ (startRow + idx) * incx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        %TYPE diagA = %INIT(0.0);
        if(targetRow < endRow)
        {
#ifndef BANDED
            diagA = A((targetRow), (targetRow));
#else
            diagA = A((startRow + threadID), (0));
#endif
            %CONJUGATE(doConj, diagA);
        }
        %TYPE tempA, tempS;
        for(int i = 0; i < %TRIANGLE_HEIGHT; i++)
        {
            if((i <= threadID) && (i > 0) && (targetRow < endRow))
            {
#ifndef BANDED
                tempA = A((startRow + i - 1), (targetRow));
#else
                tempA = A((bandRow - 1), (bandCol + 1));
  //              printf(\"threadID %d, bandRow %d bandCol %d A %f\\n\",threadID, bandRow, bandCol, tempA);
#endif
                %CONJUGATE(doConj, tempA);
                %MUL(tempS, tempA, saccShared[i-1]);
                %SUB(saccShared[threadID], saccShared[threadID], tempS);
            }
            if((i == threadID) && (targetRow < endRow) && (!isUnity))
            {
                tempS = saccShared[threadID];
    //            printf(\"threadID %d, saccShared %f, diagA %f\\n\", threadID, tempS, diagA);
                %DIV(saccShared[threadID], tempS, diagA);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#ifdef BANDED
            bandRow++; bandCol--;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(targetRow < endRow)
        {
            xnew[(targetRow * incx)] = saccShared[threadID];
        }
}
";

const char * trsv_CLT_SolveTriangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
    #define A(row, col) (*( A + ((( (col) *((2*N) + 1 - (col))) / 2) + ((row) - (col)))))
#elif defined(BANDED)
    #define A(row, col) A[ (row) * lda + (col)]
#else
    #define A(row, col) A[ (row) + (col) * lda]
#endif
#pragma OPENCL EXTENSION cl_amd_printf : enable

// Column-Major Lower Non-Unity case
// StartRow points to actual Row to start from( absolute Column number)
// endRow points to actual Row to stop + 1( absolute Column number)
__kernel void %PREFIXtrsv_CLT_SolveTriangle_kernel(     __global const %TYPE* _A,
                                                __global %TYPE* _xnew,
                                                uint N,
                                                int incx,
                                                int isUnity,
                                                uint lda,
                                                int doConj,
                                                int startRow, int endRow, uint offa, uint offx
#ifdef BANDED
                                                ,uint KL
#endif
                    )
{
        __global %TYPE* xnew;
        __global const %TYPE *A = _A + offa;
        if ( incx < 0 ) // Goto end of vector
        {
            xnew     = _xnew  + offx + ( N - 1) * abs(incx);
        }
        else
        {
            xnew     = _xnew + offx;
        }

        int blockSize = get_local_size(0);
        int threadID = get_local_id(0);
        __local volatile %TYPE saccShared[%TRIANGLE_HEIGHT];
        int targetRow;
        targetRow = (endRow - 1) - threadID;

#ifdef BANDED
        int bandRow = (endRow - 1);
        int bandCol = (KL) - threadID;
#endif

        //#pragma unroll
        for( int idx = threadID; (idx < %TRIANGLE_HEIGHT) && (((endRow - 1) - idx) >= startRow); idx += blockSize)
        {
            saccShared[idx] = xnew[((endRow - 1) - idx) * incx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        %TYPE diagA = %INIT(0.0);
        if(targetRow >= startRow)
        {
#ifndef BANDED
            diagA = A((targetRow), (targetRow));
#else
            diagA = A((bandRow - threadID), (KL));
//            printf(\"ThreadID %d, bandRow %d bandCol %d\\n\", threadID, bandRow, bandCol);
#endif
            %CONJUGATE(doConj, diagA);
        }
        %TYPE tempA, tempS;

        for( int i = (endRow - 1); i >= startRow; i--)
        {
            if((targetRow == i) && (!isUnity))
            {
                tempS = saccShared[threadID];
                %DIV(saccShared[threadID], tempS, diagA);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if((targetRow < i) && (targetRow >= startRow))
            {
#ifndef BANDED
                tempA = A((i), (targetRow));
#else
                tempA = A((bandRow), (bandCol));
#endif
                %CONJUGATE(doConj, tempA);
                %MUL(tempS, tempA, saccShared[(endRow - 1) - i]);
                %SUB(saccShared[threadID], saccShared[threadID], tempS);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#ifdef BANDED
            bandRow--; bandCol++;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(targetRow >= startRow)
        {
            xnew[(targetRow * incx)] = saccShared[threadID];
        }
}
";

