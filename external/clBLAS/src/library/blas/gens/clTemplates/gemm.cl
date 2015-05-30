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

//
// Few Observations:
// 1. Vector Length of 4 increases the performance of DGEMM to 250GFLOPS.
//    Coupled with 8x8 block without "barrier", max performance seen is around 267GFLOPS on Cayman.
//    Otherwise, it is at 225GFLOPS max (16x8 block, with barrier, vector size of 2)
//    However, a change of tile-size with vector length of just 2 yields 330GFLOPS consistently.
//    So, 330 is our sweetspot for DGEMM now.
// 2. When MxN is not completely divisible by [subdimy x subdimx] then a workgroup size of
//    8x8 yields the best performance.
//    Even in this case, if extra threads exit - the performance should be better.
//    Thread exit should be done only if a barrier is NOT used. Otherwise, it will result in a hang.
// 3. When processing non-tail run, workgroups processing full tiles can be grouped together and run
//    However, this did not yield any significant performance on tail processing
//    Sometimes, performance degradation was also seen. So, this idea will not be pursued.
//
// Pending Enhancements for GEMM:
// -4. TN Kernel Performance can be improved. The prototype code shows better performance than the templated
//     code. The templated code slightly differs from the prototype code. This can be fixed to get more
//     performance.
// -2. PENDING BUG FIX on the Unroll Factor for NN kernel - Configurable PANEL Implementation Introduced it.
//     Currently panel of %V only supported
//  0. When workgroup size == WAVEFRONT Size,  GEMM_NEEDS_BARRIER need not be defined.
//     This saves a few milliseconds depending on the problem size.
//  1. Support for VLOADA, VLOADB and VLOADC in KPRINTF required. Currently, if any one of the matrices
//     are vector unfriendly, the kernel translates to a completely scalar kernel.
//     This is pretty easy to implement in KPRINTF.
//  2. Panel Width == %V in the current implementation. It should be a separate config define
//     that can has to be a multiple of %V.
//     Currently only NxN kernel has %PANEL support implemented. TN and NT needs to be enhanced.
//     This will be required for tuning and also for high performance for D,C and ZGEMMs
//  3. "actualRow" based improvement can be used in KTail Processing as well for NN Kernel
//  4. A.B^T can be optimzed for cases where ITEMY > 4. Successive threads are now ITEMX apart
//     Instead, we can make them float4 apart to get highest L1 cache bandwidth
//  5. A.B^T - actualCol, actualRow optimization
//
static const char *GEMM_NN_KERNEL = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void GEMM_NN__KERNEL ( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *_C,
                                     uint M, uint N, uint _K, uint _lda, uint _ldb, uint ldc, uint offa, uint offb, uint offc,
                                %TYPE alpha, %TYPE beta
                                #ifdef TAIL_RUN
                                , uint tailStartM, uint tailStartN
                                #endif
                                )
{
    const int V = %V;
    __global %TYPE const *restrict A;
    __global %TYPE const *restrict B;
    __global %TYPE *C = _C + offc;
    uint K = _K;
    uint lda, ldb;
    uint rowA, colA, rowB, colB, rowC, colC;
    uint numGroupsOnY;
    uint row, col;
    uint tid = get_local_id(0);
    int panel;
    int ACOLSTART, ACOLEND;
    uint MV;

    //
    // %WIDTH - Preferably 16
    // %ITEMY, %ITEMX - 1 Thread is responsible for %ITEMY * %ITEMX sub-matrix in C
    //                    %ITEMY must be divisible by %V for NN kernel
    // The entire workgroup loops-together to complete ITEMY-ITEMX sub-matrix
    //
    uint threadsY = %WIDTH;
    uint threadsX = get_local_size(0)/threadsY;

    //
    // Column-Major ordering of Workgroups
    //
    // %ITEMY - Number of elements , a workitem processes in Y direction.
    // %ITEMX - Number of elements , a workitem processes in X direction.
    //
    // %V     - Vectoring Width
    // %PANEL(*) - Panel Width to access Rows of A and Columns of B
    //               Right now, %V is assumed to be the panel width.
    //               We dont use %PANEL in the current implementation.
    //
    MV = M;
    #ifndef TAIL_RUN
    {
        uint bidX, bidY;
        uint blockDimY;

        #ifdef M_TAIL_PRESENT
        MV = M - (M % (%V));
        #endif
        if (MV == 0)
        {
            return;
        }
        blockDimY = ((M-1) / (threadsY * %ITEMY)) + 1;
        bidY = ( get_group_id(0) % ( blockDimY));
        bidX = ( get_group_id(0) / ( blockDimY));
        //
        // Note:
        // Using the new Map function does not yeild any performnce gain.
        // In fact, it degraded the performance
        // Keep this commented.
        //
        //mapWorkGroupToTileNumber(M, N, &bidY, &bidX);

        //
        // <row,col> is the left-top of the TILE region
        // in the output C matrix that will be determined
        // by this workgroup
        //
        row =  (bidY * (threadsY * %ITEMY));
        col =  (bidX * (threadsX * %ITEMX));
    }
    #else
    {
        uint nWorkGroupsAY, nWorkGroupsAX, nWorkGroupsA;
        uint bidY, bidX;

        if (M == tailStartM)
        {
            nWorkGroupsA = 0;
        } else {
            nWorkGroupsAY = ((M - tailStartM - 1)/threadsY + 1);
            nWorkGroupsAX = ((tailStartN - 1)/threadsX + 1);
            nWorkGroupsA = nWorkGroupsAY * nWorkGroupsAX;
        }
        if (get_group_id(0) < nWorkGroupsA)
        {
            bidY = get_group_id(0) % (nWorkGroupsAY);
            bidX = get_group_id(0) / nWorkGroupsAY;
            row = tailStartM + (bidY * threadsY * %ITEMY);
            col = (bidX * threadsX * %ITEMX);
        } else {
            uint nWorkGroupsBY, nWorkGroupsBX;

            nWorkGroupsBY = ((M-1)/threadsY) + 1;
            nWorkGroupsBX = ((N-tailStartN-1)/threadsX) + 1;
            bidY = (get_group_id(0) - nWorkGroupsA) % (nWorkGroupsBY);
            bidX = (get_group_id(0) - nWorkGroupsA) / nWorkGroupsBY;
            row = (bidY * threadsY * %ITEMY);
            col = tailStartN + (bidX * threadsX * %ITEMX);
        }

    }
    #endif

    //
    // ACOLSTART, ACOLEND
    // SYMM Matrix  multiplication proceeds by multiplying panels on A's block-row
    // with panels on B's block-column.
    // However due to symmetric nature of A/B matrix compounded by the fact that
    // only upper OR lower triangle of the symm matrix is available, vector-loads
    // are not possible while traversing certain regions of the matrix.
    // ACOLStart and ACOLEnd - signify what portion of SYMM can be achieved through
    // this NN kernel. The SYMM handler has to compose the SYMM in-terms of GEMM kernels
    //
#ifdef __SYMM_LEFT__
    // MxM * MxN
    A = _A + offa;
    lda = _lda;
    B = _B + offb;
    ldb = _ldb;
    K = M;
    #ifndef __SYMM_DIAGONAL__
    #ifdef __SYMM_LOWER__
    ACOLSTART = 0;
    ACOLEND = row;
    #elif defined(__SYMM_UPPER__)
    ACOLSTART = row + (threadsY*(%ITEMY));
    ACOLEND = K;
    #else
    #error GEMM_NN_KERNEL
    #endif
    #else
        ACOLSTART = row;
        ACOLEND = row + (threadsY*(%ITEMY));
    #endif
    if (ACOLEND > K)
    {
        ACOLEND = K;
    }
    /*
    if (get_local_id(0) == 0)
    {
        printf(\" GEMM_NN_KERNEL : SYMM_LEFT: Setting ACOLSTART to %d and ACOLEND to %d \\n \" , ACOLSTART, ACOLEND);
    }
    */
#elif defined(__SYMM_RIGHT__)
    // MxN * NxN
    A = _B + offb;
    lda = _ldb;
    B = _A + offa;
    ldb = _lda;
    K = N;
    #ifndef __SYMM_DIAGONAL__
    #ifdef __SYMM_UPPER__
    ACOLSTART = 0;
    ACOLEND = col;
    #elif defined(__SYMM_LOWER__)
    ACOLSTART =  col + (threadsX*(%ITEMX));
    ACOLEND = K;
    #else
    #error GEMM_NN_KERNEl
    #endif
#else
        ACOLSTART = col;
        ACOLEND =  col + (threadsX*(%ITEMX));
    #endif
    if (ACOLEND > K)
    {
        ACOLEND = K;
    }
#else
    A = _A + offa;
    B = _B + offb;
    K = _K;
    lda = _lda;
    ldb = _ldb;
    ACOLSTART = 0;
    ACOLEND = K;
#endif

    uint offsetY = (tid % threadsY) * %V;
    uint offsetX = (tid / threadsY) * %ITEMX;
    rowA     =     row + offsetY;
       colB     =     (col+offsetX);
    #ifndef TAIL_RUN
    bool tailBlock = ((row  >= M) || (col >= N));
    #else
    bool tailBlock = (row >= tailStartM);
    #endif


    /*
    #ifdef TAIL_RUN
    if ((rowA >= M) || (colB >= N))
    {
        return;
    }
    #endif
    */

    #ifndef TAIL_RUN
    // Non-tail RUN
    if (tailBlock == true)
    {
        return;
    }
    #elif defined(TAIL_RUN)
    // TAIL RUN
    if (tailBlock == false)
    {
        return;
    }
    #else
    #error GEMM_NN_KERNEL
    #endif

    %TYPE%V AVAL[%V][(%ITEMY_BY_V)]; // 8
    #ifdef COMPLEX
        %TYPE%HV AVALEVEN[%V][(%ITEMY_BY_V)]; // 8
        %TYPE%HV AVALODD[%V][(%ITEMY_BY_V)]; // 8
    #endif

    %TYPE%V   BVAL[%ITEMX];
    #ifdef COMPLEX
        %TYPE%HV   BVALEVEN[%ITEMX];
        %TYPE%HV   BVALODD[%ITEMX];
    #endif

    %TYPE%V CVAL[(%ITEMY_BY_V)][%ITEMX];
    #ifdef COMPLEX
        %TYPE%HV CVALEVEN[(%ITEMY_BY_V)][%ITEMX];
        %TYPE%HV CVALODD[(%ITEMY_BY_V)][%ITEMX];
    #endif

    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            CVAL[i][j] = (%TYPE%V) 0;
            #ifdef COMPLEX
                CVALEVEN[i][j] = (%TYPE%HV) 0;
                CVALODD[i][j] = (%TYPE%HV) 0;
            #endif
        }
    }

    uint ACOL;
    for(ACOL=ACOLSTART; ((ACOL+ %V -1) < ACOLEND); ACOL += %V)
    {
        {
            //
            // Load B values
            //
            %IF(%ITEMX) #pragma unroll %ITEMX
            for(uint bcol = 0; bcol < %ITEMX; bcol++)
            {
                #ifdef N_TAIL_PRESENT
                uint actualCol;
                actualCol = ((colB + bcol) >= N) ? (N-1) : (colB + bcol);
                #endif

                #if !defined(__SYMM_DIAGONAL__) || defined(__SYMM_LEFT__)
                    #ifndef N_TAIL_PRESENT
                        BVAL[bcol] = %VLOAD(0, (&B[ACOL + (colB + bcol)*ldb]));
                    #else
                        BVAL[bcol] = %VLOAD(0, (&B[ACOL + (actualCol)*ldb]));
                    #endif
                #else
                    // defined(__SYMM_DIAGONAL__) && defined(__SYMM_RIGHT__)
                    #ifndef N_TAIL_PRESENT
                        BVAL[bcol] = SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, ACOL, (colB + bcol));
                    #else
                        BVAL[bcol] = SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, ACOL, actualCol);
                    #endif
                #endif
                //
                // If Complex data, load the real and imaginary parts into separate register banks
                //
                #ifdef COMPLEX
                    BVALEVEN[bcol] = BVAL[bcol].even;
                    BVALODD[bcol] =  BVAL[bcol].odd;
                #endif
            }
        }

        {
            //
            // Load A values
            //
            //
            // PENDNG BUG FIX: Unroll Factor should be according to PANEL Size
            //                 Previoously PANEL was size of V. So ITEMY worked
            // Current Workaround - Panel same as %V - See gemm_cached.cpp
            //
            %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
            for(uint j=0; j< (%ITEMY_BY_V); j++)
            {
                #pragma unroll %V
                for(uint i = 0; i < %V; i++)
                {
                    uint actualRow;

                    #if !defined(__SYMM_DIAGONAL__) || defined(__SYMM_RIGHT__)
                        #ifndef M_TAIL_PRESENT
                            AVAL[i][j] = %VLOAD(0, (&A[(rowA + j*threadsY*(V)) + (ACOL + i)*lda]) );
                        #else
                            actualRow = ((rowA + j*threadsY*(V)) >= MV) ? (MV-%V) : (rowA + j*threadsY*(V));
                            AVAL[i][j] = %VLOAD(0, (&A[actualRow + (ACOL + i)*lda]) );
                        #endif
                    #else
                        // CASE: SYMM_DIAGONAL && SYMM_LEFT
                        #ifndef M_TAIL_PRESENT
                            //AVAL[c][r] = %VLOAD(0, (&A[(rowA + r*threadsY*(V)) + (ACOL + c)*lda]) );
                            AVAL[i][j] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, (rowA + j*threadsY*(V)) ,  (ACOL + i));
                        #else
                            actualRow = ((rowA + j*threadsY*(V)) >= MV) ? (MV-%V) : (rowA + j*threadsY*(V));
                            AVAL[i][j] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, actualRow, (ACOL + i));
                        #endif
                    #endif
                    //
                    // If Complex data, load the real and imaginary parts into separate register banks
                    //
                    #ifdef COMPLEX
                        AVALEVEN[i][j] = AVAL[i][j].even;
                        AVALODD[i][j] = AVAL[i][j].odd;
                    #endif
                }
            }
        }

        {
            %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
            for(uint i=0; i<(%ITEMY_BY_V); i++)
            {
                %IF(%ITEMX) #pragma unroll %ITEMX
                for(uint j=0; j<(%ITEMX); j++)
                {
                    #ifndef COMPLEX
                        %VFOR_REAL
                        {
                            CVAL[i][j] = mad(AVAL[%VFORINDEX][i], BVAL[j]%VFORSUFFIX, CVAL[i][j]);
                        }
                    #else
                        //
                        // Pending - Replace by %COMPLEX_VMAD()
                        //
                        %VFOR_REAL
                        {
                            //
                            // PENDING Needs a FIX
                            //
                            CVALEVEN[i][j]  = mad(AVALEVEN[%VFORINDEX][i],  BVALEVEN[j]%VFORSUFFIX, CVALEVEN[i][j]);
                            CVALODD[i][j]   = mad(AVALEVEN[%VFORINDEX][i],  BVALODD[j]%VFORSUFFIX,  CVALODD[i][j]);
                            CVALEVEN[i][j]  = mad(AVALODD[%VFORINDEX][i],   -BVALODD[j]%VFORSUFFIX,  CVALEVEN[i][j]);
                            CVALODD[i][j]   = mad(AVALODD[%VFORINDEX][i],   BVALEVEN[j]%VFORSUFFIX,  CVALODD[i][j]);
                        }
                    #endif
                }
            }
        }

        #ifdef GEMM_NEEDS_BARRIER
        barrier(CLK_LOCAL_MEM_FENCE);
        #endif
    }

    #ifdef COMPLEX
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            %COMPLEX_JOIN(CVAL[i][j], CVALEVEN[i][j], CVALODD[i][j]);
        }
    }
    #endif

    //
    // Tail blocks never execute this FOR loop as they execute with Vector Width of 1
    //


    for(; ACOL < ACOLEND; ACOL ++)
    {
        //
        // Load B values
        //
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint bcol = 0; bcol < %ITEMX; bcol++)
        {
            //
            // PENDING: PANEL iteration to Load the Panel Depth iterating by %V
            //
            #if !defined(__SYMM_DIAGONAL__) || defined(__SYMM_LEFT__)
            {
                %TYPE SCAL;
                #ifndef N_TAIL_PRESENT
                    SCAL = B[ACOL + (colB + bcol)*ldb];
                    BVAL[bcol] = %VMAKEVEC(SCAL);
                #else
                    SCAL = B[ACOL + ((colB + bcol)%(N))*ldb];
                    BVAL[bcol] = %VMAKEVEC(SCAL);
                #endif
            }
           #else
                // SYMM_DIAGONAL && SYMM_RIGHT
            {
                %TYPE SCAL;

                #ifndef N_TAIL_PRESENT
                    SCAL = SYMM_SCALAR_LOAD(B, N, ldb, ACOL,  (colB + bcol));
                    BVAL[bcol] = %VMAKEVEC(SCAL);
                #else
                    SCAL = SYMM_SCALAR_LOAD(B, N, ldb, ACOL, ((colB + bcol)%(N)));
                    BVAL[bcol] = %VMAKEVEC(SCAL);
                #endif
            }
           #endif
        }

        //
        // Load A values
        //
        %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
        for(uint i = 0; i < (%ITEMY_BY_V); i++) // 1 * ITEMY/V
        {
            #if !defined(__SYMM_DIAGONAL__) || defined(__SYMM_RIGHT__)
            #ifndef M_TAIL_PRESENT
            AVAL[0][i] = %VLOAD(0, (&A[(rowA + i*threadsY*(V)) + (ACOL)*lda]) );
            #else
            AVAL[0][i] = %VLOAD(0, (&A[(((rowA + i*threadsY*(V))) % (MV)) + (ACOL)*lda]) );
            #endif
            #else
                // defined(DIAGONAL) && (LEFT)
                #ifndef M_TAIL_PRESENT
                AVAL[0][i] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, (rowA + i*threadsY*(V)) , (ACOL));
                #else
                AVAL[0][i] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, ((rowA + i*threadsY*(V)) % (MV)), (ACOL));
                #endif
            #endif
        }

        {
            %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
            for(uint i=0; i<(%ITEMY_BY_V); i++)
            {
                %IF(%ITEMX) #pragma unroll %ITEMX
                for(uint j=0; j<(%ITEMX); j++)
                {
                    %VMAD(CVAL[i][j] ,  AVAL[0][i] , BVAL[j]);
                }
            }
        }
    }


    /*
    if ((get_group_id(0) == 0) && (get_local_id(0) == 0))
    {
        printf(\"Updating C Matrix: Alpha = %f, Beta = %f\\n\", alpha, beta);
    }
    */
    //
    // STORE Result in C
    //
    %TYPE%V reg , betareg, alphareg;
    %TYPE%V alphav, betav;
    alphav = %VMAKEVEC(alpha);
    betav = %VMAKEVEC(beta);

    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            #if !defined(M_TAIL_PRESENT) && !defined(N_TAIL_PRESENT)
            reg = %VLOAD(0, (&C[rowA + i*threadsY*V +     (colB+j)*ldc]));
            %VMUL(betareg, betav, reg);
            %VMUL(alphareg, alphav, CVAL[i][j]);
            %ADD( reg, betareg, alphareg);
            %VSTORE(reg, 0, (&C[(rowA + i*threadsY*V) + (colB+j)*ldc]));
            #else
                if (((rowA + i*threadsY*V) < MV) && ((colB + j) < N))
                {
                    reg = %VLOAD(0, (&C[rowA + i*threadsY*V +     (colB+j)*ldc]));
                    %VMUL(betareg, betav, reg);
                    %VMUL(alphareg, alphav, CVAL[i][j]);
                    %ADD( reg, betareg, alphareg);
                    %VSTORE(reg, 0, (&C[(rowA + i*threadsY*V) + (colB+j)*ldc]));
        }
            #endif
    }
    }
    return;
}
";

static const char *GEMM_NT_KERNEL = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

//#undef COMPLEX
//#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void GEMM_NT__KERNEL ( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *_C,
                                uint M, uint N, uint _K, uint _lda, uint _ldb, uint ldc, uint offa, uint offb, uint offc,
                                %TYPE alpha, %TYPE beta
                                #ifdef TAIL_RUN
                                , uint tailStartM, uint tailStartN
                                #endif
                                )
{
    const int V = %V;
    __global %TYPE const *restrict A;
    __global %TYPE const *restrict B;
    __global %TYPE *C = _C + offc;
    uint K = _K;
    uint lda, ldb;
    uint rowA, colA, rowB, colB, rowC, colC;
    uint numGroupsOnY;
    uint row, col;
    uint tid = get_local_id(0);
    int panel;
    int ACOLSTART, ACOLEND;
    uint MV, NV;

    //
    // %WIDTH - Preferably 16
    // %ITEMY, %ITEMX - 1 Thread is responsible for %ITEMY * %ITEMX sub-matrix in C
    //                    %ITEMY and %ITEMX must be divisible by %V for NT kernel
    // The entire workgroup loops-together to complete ITEMY-ITEMX sub-matrix
    //
    uint threadsY = %WIDTH;
    uint threadsX = get_local_size(0)/threadsY;

    //
    // Column-Major ordering of Workgroups
    //
    // %ITEMY - Number of elements , a workitem processes in Y direction.
    // %ITEMX - Number of elements , a workitem processes in X direction.
    //
    // %V     - Vectoring Width
    // %PANEL(*) - Panel Width to access Rows of A and Columns of B
    //               Right now, %V is assumed to be the panel width.
    //               We dont use %PANEL in the current implementation.
    //
    MV = M;
    NV = N;
    #ifndef TAIL_RUN
    {
        uint bidX, bidY;
        uint blockDimY;

        #ifdef M_TAIL_PRESENT
        MV = M - (M % (%V));
        if (MV == 0)
        {
            return;
        }
        #endif
        #ifdef N_TAIL_PRESENT
        NV = N - (N% (%V));
        if (NV == 0)
        {
            return;
        }
        #endif
        blockDimY = ((M-1) / (threadsY * %ITEMY)) + 1;
        uint blockID = get_group_id(0);
        getBlockNumber(blockDimY, blockID, &bidY, &bidX, 1);

        //
        // <row,col> is the left-top of the TILE region
        // in the output C matrix that will be determined
        // by this workgroup
        //
        row =  (bidY * (threadsY * %ITEMY));
        col =  (bidX * (threadsX * %ITEMX));
    }
    #else
    {
        uint nWorkGroupsAY, nWorkGroupsAX, nWorkGroupsA;
        uint bidY, bidX;

        MV = M;
        if (M == tailStartM)
        {
            nWorkGroupsA = 0;
        } else {
            nWorkGroupsAY = ((M - tailStartM - 1)/threadsY + 1);
            nWorkGroupsAX = ((tailStartN - 1)/threadsX + 1);
            nWorkGroupsA = nWorkGroupsAY * nWorkGroupsAX;
        }
        if (get_group_id(0) < nWorkGroupsA)
        {
            bidY = get_group_id(0) % (nWorkGroupsAY);
            bidX = get_group_id(0) / nWorkGroupsAY;
            row = tailStartM + (bidY * threadsY * %ITEMY);
            col = (bidX * threadsX * %ITEMX);
            NV = tailStartN;
        } else {
            uint nWorkGroupsBY, nWorkGroupsBX;

            nWorkGroupsBY = ((M-1)/threadsY) + 1;
            nWorkGroupsBX = ((N-tailStartN-1)/threadsX) + 1;
            bidY = (get_group_id(0) - nWorkGroupsA) % (nWorkGroupsBY);
            bidX = (get_group_id(0) - nWorkGroupsA) / nWorkGroupsBY;
            row = (bidY * threadsY * %ITEMY);
            col = tailStartN + (bidX * threadsX * %ITEMX);
            NV = N;
        }

    }
    #endif

    //
    // ACOLSTART, ACOLEND
    // SYMM Matrix  multiplication proceeds by multiplying panels on A's block-row
    // with panels on B's block-column.
    // However due to symmetric nature of A matrix compounded by the fact that
    // only upper OR lower triangle of the symm matrix is available, vector-loads
    // are not possible while traversing certain regions of the matrix.
    // ACOLStart and ACOLEnd - signify what portion of SYMM can be achieved through
    // this NT kernel. The SYMM handler has to compose the SYMM in-terms of GEMM kernels
    //
#ifdef __SYMM_LEFT__
    #error GEMM_NT_KERNEL Should not be called in __SYMM_LEFT__ case!
#elif defined(__SYMM_RIGHT__)
    // MxN * NxN
    A = _B + offb;
    lda = _ldb;
    B = _A + offa;
    ldb = _lda;
    K = N;
    #ifndef __SYMM_DIAGONAL__
    #ifdef __SYMM_UPPER__
    ACOLSTART =  col + (threadsX*(%ITEMX));
    ACOLEND = K;
    #elif defined(__SYMM_LOWER__)
    ACOLSTART = 0;
    ACOLEND = col;
    #else
    #error GEMM_NT_KERNEL : Neither SYMM_UPPER nor SYMM_LOWER is defined!
    #endif
    #else
        ACOLSTART = col;
        ACOLEND =  col + (threadsX*(%ITEMX));
    #endif
    if (ACOLEND > K)
    {
        ACOLEND = K;
    }
#else // GEMM
    A = _A + offa;
    B = _B + offb;
    K = _K;
    lda = _lda;
    ldb = _ldb;
    ACOLSTART = 0;
    ACOLEND = K;
#endif

    uint offsetY = (tid % threadsY) * %V;
    uint offsetX = (tid / threadsY) * %ITEMX;
    rowA     =     row + offsetY;
       colB     =     col + offsetX;
    #ifndef TAIL_RUN
    bool tailBlock = ((row >= M) || (col >= N));
    #else
    bool tailBlock = ((row >= tailStartM) || (col >= tailStartN));
    #endif

    /* Should be handled with TAIL_PRESENT Macros.
    if ((rowA >= M) || (colB >= N))
    {
        return;
    }
    */

    #ifndef TAIL_RUN
    // Non-tail RUN
    if (tailBlock == true)
    {
        return;
    }
    #else
    // TAIL RUN - This case never happens.
    if (tailBlock == false)
    {
        return;
    }
    #endif

    %TYPE%V CVAL[(%ITEMY_BY_V)][%ITEMX];
    #ifdef COMPLEX
    %TYPE%HV    CVALEVEN[(%ITEMY_BY_V)][%ITEMX];
    %TYPE%HV    CVALODD[(%ITEMY_BY_V)][%ITEMX];
    #endif

    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            CVAL[i][j] = (%TYPE%V) 0;
            #ifdef COMPLEX
            CVALEVEN[i][j] = (%TYPE%HV) 0;
            CVALODD[i][j] = (%TYPE%HV) 0;
            #endif
        }
    }

    uint ACOL;
    for(ACOL=ACOLSTART; ((ACOL+%V-1) < ACOLEND); ACOL += %V /* %PANEL */)
    {
        %TYPE%V AVAL[%V][(%ITEMY_BY_V)];     // [%PANEL][%ITEMY_BY_V]
        %TYPE%V BVAL[%ITEMX_BY_V][%V];        // [%PANEL][%ITEMX]
        #ifdef COMPLEX
        %TYPE%HV    AVALEVEN[%V][(%ITEMY_BY_V)];     // [%PANEL][%ITEMY_BY_V]
        %TYPE%HV    AVALODD[%V][(%ITEMY_BY_V)];     // [%PANEL][%ITEMY_BY_V]
        %TYPE%HV    BVALEVEN[%ITEMX_BY_V][%V];        // [%PANEL][%ITEMX]
        %TYPE%HV    BVALODD[%ITEMX_BY_V][%V];        // [%PANEL][%ITEMX]
        #endif

        {
            //
            // Load B values
            //
            %IF(%V) #pragma unroll %V
            for(uint panel=0; panel < %V; panel++)
            {
                %IF(%ITEMX_BY_V) #pragma unroll %ITEMX_BY_V
                for(uint bcol = 0; bcol < %ITEMX_BY_V; bcol++)
                {
                    //
                    // PENDING: PANEL iteration to Load the Panel Depth iterating by %V
                    //
                    #ifndef __SYMM_DIAGONAL__
                        #ifndef N_TAIL_PRESENT
                        BVAL[bcol][panel] = %VLOAD(0, (&B[(ACOL + panel)*ldb + (colB + bcol*(V))]));
                        #else
                        BVAL[bcol][panel] = %VLOAD(0, (&B[(ACOL + panel)*ldb + ((colB + bcol*V) % NV)]));
                        #endif
                    #else
                        #ifndef N_TAIL_PRESENT
                        BVAL[bcol][panel] = SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, (colB + bcol*(V)), (ACOL + panel));
                        #else
                        BVAL[bcol][panel] =
                            SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, ((colB + bcol*V) % NV), (ACOL + panel));
                        #endif
                    #endif

                    #ifdef CONJUGATE_B
                        %TYPE%V conjTemp = BVAL[bcol][panel];
                        %CONJUGATE(1, conjTemp);
                        BVAL[bcol][panel] = conjTemp;
                    #endif
                    #ifdef COMPLEX
                    {
                        BVALEVEN[bcol][panel] = BVAL[bcol][panel].even;
                        BVALODD[bcol][panel]  = BVAL[bcol][panel].odd;
                    }
                    #endif
                }
            }

            //
            // Load A values
            //
            %IF(%ITEMY) #pragma unroll %ITEMY
            for(uint i = 0; i < (%V * (%ITEMY_BY_V)) /* PANEL * ITEMY/V */; i++)
            {
                const uint yiterations = %ITEMY_BY_V;
                uint c = (i / yiterations);
                uint r = (i % yiterations);

                #ifndef M_TAIL_PRESENT
                AVAL[c][r] = %VLOAD(0, (&A[(rowA + r*threadsY*(V)) + (ACOL + c)*lda]) );
                #else
                AVAL[c][r] = %VLOAD(0, (&A[((rowA + r*threadsY*(V)) % MV) + (ACOL + c)*lda]) );
                #endif

                #ifdef COMPLEX
                AVALEVEN[c][r] = AVAL[c][r].even;
                AVALODD[c][r] = AVAL[c][r].odd;
                #endif
            }
        }

        %IF(%V) #pragma unroll %V
        for(uint panel=0; panel<(%V); panel++)
        {
            %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
            for(uint i=0; i<(%ITEMY_BY_V); i++)
            {
                %IF(%ITEMX_BY_V) #pragma unroll %ITEMX_BY_V
                for(uint j=0; j<(%ITEMX_BY_V); j++)
                {
                    const int CX = j * (%V);

                    #ifndef COMPLEX
                    %VFOR_REAL
                    {
                        CVAL[i][CX + %VFORINDEX] = mad(AVAL[panel][i], BVAL[j][panel]%VFORSUFFIX,
                                                        CVAL[i][CX + %VFORINDEX]);
                    }
                    #else
                        //
                        // PENDING: Replace with %COMPLEX_MAD op
                        //
                        %VFOR_REAL
                        {
                            CVALEVEN[i][CX + %VFORINDEX] =
                                mad(AVALEVEN[panel][i], BVALEVEN[j][panel]%VFORSUFFIX, CVALEVEN[i][CX + %VFORINDEX]);
                            CVALODD[i][CX + %VFORINDEX]  =
                                mad(AVALEVEN[panel][i], BVALODD[j][panel]%VFORSUFFIX,  CVALODD[i][CX + %VFORINDEX]);
                            CVALEVEN[i][CX + %VFORINDEX] =
                                mad(AVALODD[panel][i], -BVALODD[j][panel]%VFORSUFFIX,  CVALEVEN[i][CX + %VFORINDEX]);
                            CVALODD[i][CX + %VFORINDEX] =
                                mad(AVALODD[panel][i], BVALEVEN[j][panel]%VFORSUFFIX,  CVALODD[i][CX + %VFORINDEX]);
                        }
                    #endif
                }
            }
        }

        #ifdef GEMM_NEEDS_BARRIER
        barrier(CLK_LOCAL_MEM_FENCE);
        #endif
    }

    #ifdef COMPLEX
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            %COMPLEX_JOIN(CVAL[i][j], CVALEVEN[i][j], CVALODD[i][j]);
        }
    }
    #endif

    //
    // Tail blocks never execute this FOR loop as they execute with Vector Width of 1
    //

    for(; ACOL < ACOLEND; ACOL ++)
    {
        %TYPE%V AVAL[(%ITEMY_BY_V)];    // [%PANEL][%ITEMY_BY_V]
        %TYPE   BVAL[%ITEMX];               // [%PANEL][%ITEMX]

        //
        // Load B values
        //
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint bcol = 0; bcol < %ITEMX; bcol++)
        {
            %TYPE SCALAR;
            //
            // PENDING: PANEL iteration to Load the Panel Depth iterating by %V
            //
            {
                #ifndef __SYMM_DIAGONAL__
                    #ifndef N_TAIL_PRESENT
                        SCALAR = B[ACOL*ldb + (colB + bcol)];
                    #else
                        SCALAR = B[ACOL*ldb + ((colB + bcol) % NV)];
                    #endif
                #else
                    #ifndef N_TAIL_PRESENT
                        SCALAR = SYMM_SCALAR_LOAD(B, N, ldb, (colB + bcol), ACOL );
                    #else
                        SCALAR = SYMM_SCALAR_LOAD(B, N, ldb, ((colB + bcol) % NV), ACOL);
                    #endif
                #endif

                #ifdef CONJUGATE_B
                    %CONJUGATE(1, SCALAR);
                #endif
                BVAL[bcol] = (SCALAR);
            }
        }

        //
        // Load A values
        //
        %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
        for(uint i = 0; i < (%ITEMY_BY_V); i++) // 1 * ITEMY/V
        {
            #ifndef M_TAIL_PRESENT
            AVAL[i] = %VLOAD(0, (&A[(rowA + i*threadsY*(V)) + (ACOL)*lda]) );
            #else
            AVAL[i] = %VLOAD(0, (&A[((rowA + i*threadsY*(V)) % MV) + (ACOL)*lda]) );
            #endif
        }

        {
            %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
            for(uint i=0; i<(%ITEMY_BY_V); i++)
            {
                %IF(%ITEMX) #pragma unroll %ITEMX
                for(uint j=0; j<(%ITEMX); j++)
                {
                    %VMAD(CVAL[i][j] ,  AVAL[i] , BVAL[j]);
                }
            }
        }
    }

    //
    // STORE Result in C
    //
    %TYPE%V reg , betareg, alphareg;
    %TYPE%V alphav, betav;
    alphav = %VMAKEVEC(alpha);
    betav = %VMAKEVEC(beta);

    #ifndef HERK
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            #if !defined(M_TAIL_PRESENT) && !defined(N_TAIL_PRESENT)
            reg = %VLOAD(0, (&C[rowA + i*threadsY*V +     (colB+j)*ldc]));
            %VMUL(betareg, betav, reg);
            %VMUL(alphareg, alphav, CVAL[i][j]);
            %ADD( reg, betareg, alphareg);
            %VSTORE(reg, 0, (&C[(rowA + i*threadsY*V) + (colB+j)*ldc]));
            #else
                if (((rowA + i*threadsY*V) < MV) && ((colB+j) < NV))
                {
                    reg = %VLOAD(0, (&C[rowA + i*threadsY*V +     (colB+j)*ldc]));
                    %VMUL(betareg, betav, reg);
                    %VMUL(alphareg, alphav, CVAL[i][j]);
                    %ADD( reg, betareg, alphareg);
                    %VSTORE(reg, 0, (&C[(rowA + i*threadsY*V) + (colB+j)*ldc]));
        }
            #endif
    }
    }
    #else
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i<(%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            int actualRow = rowA + i*threadsY*V;
            int actualCol = colB + j;
            #if !defined(M_TAIL_PRESENT) && !defined(N_TAIL_PRESENT)
                {
                    %VMUL(alphareg, alphav, CVAL[i][j]);
                    //%TYPE temp[%V];
                    //*(__private %TYPE%V *)(&temp) = alphareg;
                    //#pragma unroll %V
                    //for(uint r = 0; r < %V; r++)
                    %VFOR
                    {
                        #ifdef HERK_LOWER_TRIANGLE
                        if((actualRow + %VFORINDEX) >= (actualCol))
                        #else
                        if((actualRow + %VFORINDEX) <= (actualCol))
                        #endif
                        {
                            %TYPE C_s =  C[%VFORINDEX + actualRow + actualCol * ldc];
                            %TYPE beta_s;
                            %MUL(beta_s, beta, C_s);
                            C_s = alphareg%VFORSUFFIX + beta_s;
                            if((%VFORINDEX + actualRow) == actualCol)
                            {
                                 C_s.odd = 0.0f;
                            }
                            C[%VFORINDEX + actualRow + actualCol * ldc] = C_s;
                        }
                    }
                }
            #else
                {
                    if (((rowA + i*threadsY*V) < MV) && ((colB+j) < NV))
                    {
                        %VMUL(alphareg, alphav, CVAL[i][j]);
                        //%TYPE temp[%V];
                        //*(__private %TYPE%V *)(&temp) = alphareg;
                        //#pragma unroll %V
                        //for(uint r = 0; r < %V; r++)
                        %VFOR
                        {
                            #ifdef HERK_LOWER_TRIANGLE
                            if((%VFORINDEX + actualRow) >= (actualCol))
                            #else
                            if((%VFORINDEX + actualRow) <= (actualCol))
                            #endif
                            {
                                %TYPE C_s =  C[%VFORINDEX + actualRow + actualCol * ldc];
                                %TYPE beta_s;
                                %MUL(beta_s, beta, C_s);
                                C_s = alphareg%VFORSUFFIX + beta_s;
                                if((%VFORINDEX + actualRow) == actualCol)
                                {
                                    C_s.odd = 0.0f;
                                }
                                C[%VFORINDEX + actualRow + actualCol * ldc] = C_s;
                            }
                        }
                    }
                }
            #endif
        }
    }
    #endif
    return;
}
";


static const char *GEMM_TN_KERNEL = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

//#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void GEMM_TN__KERNEL ( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *_C,
                                     uint M, uint N, uint _K, uint _lda, uint _ldb, uint ldc, uint offa, uint offb, uint offc,
                                %TYPE alpha, %TYPE beta
                                #ifdef TAIL_RUN
                                , uint tailStartM, uint tailStartN
                                #endif
                                )
{
    const int V = %V;
    const int ITEMY = %ITEMY;
    __global %TYPE const *restrict A;
    __global %TYPE const *restrict B;
    __global %TYPE *C = _C + offc;
    uint K = _K;
    uint lda, ldb;
    uint rowA, colA, rowB, colB, rowC, colC;
    uint numGroupsOnY;
    uint row, col;
    uint tid = get_local_id(0);
    int panel;
    int ACOLSTART, ACOLEND;
    uint MV, bidX;
    uint bidY;
    uint blockDimX;

    //
    // %WIDTH - Preferably 16
    // %ITEMY, %ITEMX - 1 Thread is responsible for %ITEMY * %ITEMX sub-matrix in C
    //                    %ITEMY must be divisible by %V for NN kernel
    // The entire workgroup loops-together to complete ITEMY-ITEMX sub-matrix
    //
    uint threadsY = %WIDTH;
    uint threadsX = get_local_size(0)/threadsY;

    //
    // Row-Major ordering of Workgroups
    //
    // %ITEMY - Number of elements , a workitem processes in Y direction.
    // %ITEMX - Number of elements , a workitem processes in X direction.
    //
    // %V     - Vectoring Width
    // %PANEL(*) - Panel Width to access Rows of A and Columns of B
    //               Right now, %V is assumed to be the panel width.
    //               We dont use %PANEL in the current implementation.
    //
    MV = M;
    #ifndef TAIL_RUN
    {

        blockDimX = ((N-1) / (threadsX * %ITEMX)) + 1;
        uint blockID = get_group_id(0);
        getBlockNumber(blockDimX, blockID, &bidY, &bidX, 0);

        //
        // <row,col> is the left-top of the TILE region
        // in the output C matrix that will be determined
        // by this workgroup
        //
        row =  (bidY * (threadsY * %ITEMY));
        col =  (bidX * (threadsX * %ITEMX));
    }
    #else
    #error GEMM_TN_KERNEL: TAIL_RUN is NOT needed for TN Kernel!
    #endif

    //
    // ACOLSTART, ACOLEND
    // SYMM Matrix  multiplication proceeds by multiplying panels on A's block-row
    // with panels on B's block-column.
    // However due to symmetric nature of A/B matrix compounded by the fact that
    // only upper OR lower triangle of the symm matrix is available, vector-loads
    // are not possible while traversing certain regions of the matrix.
    // ACOLStart and ACOLEnd - signify what portion of SYMM can be achieved through
    // this TN kernel. The SYMM handler has to compose the SYMM in-terms of GEMM kernels
    // SYMMETRIC LOAD routines are used when traversing the diaognal region wherease normal rules
    // hold good otherwise.
    //
#ifdef __SYMM_LEFT__
    // MxM * MxN
    A = _A + offa;
    lda = _lda;
    B = _B + offb;
    ldb = _ldb;
    K = M;
    #ifndef __SYMM_DIAGONAL__
    #ifdef __SYMM_LOWER__
    ACOLSTART = row + (threadsY * %ITEMY);
    ACOLEND = K;
        /*
        if (get_local_id(0) == 0)
        {
            printf(\"GEMM_TN_KERNEL: SYMM_LOWER: Setting ACOLSTART to %d, ACOLEND = %d\\n\", ACOLSTART, ACOLEND);
        }
        */
    #elif defined(__SYMM_UPPER__)
    ACOLSTART = 0;
    ACOLEND = row;
    #else
    #error GEMM_TN_KERNEL
    #endif
    #else
        ACOLSTART = row;
        ACOLEND = row + (threadsY * %ITEMY);
    #endif
    if (ACOLEND > K)
    {
        ACOLEND = K;
    }
#elif defined(__SYMM_RIGHT__)
    // MxN * NxN
    #error GEMM_TN_KERNEL: Internal Error: Should not be called in SYMM_RIGHT case! Right is Wrong!
#else
    // GEMM Case
    A = _A + offa;
    B = _B + offb;
    K = _K;
    lda = _lda;
    ldb = _ldb;
    ACOLSTART = 0;
    ACOLEND = K;
#endif

    uint offsetX = (tid % threadsX) * %ITEMX;
    uint offsetY = (tid / threadsX) * %ITEMY;
    rowA     =     (row + offsetY);
    colB     =     (col + offsetX);
    #ifndef TAIL_RUN
    bool tailBlock = ((row  >= M) || (col >= N));
    #else
    #error GEMM_TN_KERNEL: No TAIL_RUN for TN case
    #endif

    %TYPE%V AVAL[%ITEMY]; // %ITEMY * %PANEL
    #ifdef COMPLEX
    %TYPE%HV AVALEVEN[%ITEMY]; // %ITEMY * %PANEL
    %TYPE%HV AVALODD[%ITEMY]; // %ITEMY * %PANEL
    #endif

    %TYPE%V BVAL[%ITEMX];
    #ifdef COMPLEX
    %TYPE%HV BVALEVEN[%ITEMX]; // %ITEMY * %PANEL
    %TYPE%HV BVALODD[%ITEMX]; // %ITEMY * %PANEL
    #endif

    %TYPE   CVAL[%ITEMY][%ITEMX];
    #ifdef COMPLEX
    %TYPE%HV CVALEVEN[%ITEMY][%ITEMX]; // %ITEMY * %PANEL
    %TYPE%HV CVALODD[%ITEMY][%ITEMX]; // %ITEMY * %PANEL
    #endif

    %IF(%ITEMY) #pragma unroll %ITEMY
    for(uint i=0; i< (%ITEMY); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            #ifdef COMPLEX
            CVAL[i][j] = (%TYPE) 0;
            CVALEVEN[i][j] = (%TYPE%HV) 0;
            CVALODD[i][j] = (%TYPE%HV) 0;
            #else
            CVAL[i][j] = (%TYPE) 0;
            #endif
        }
    }

    int ACOL;
    uint actualCol;
    uint actualRow;
    int ACOLENDV;
    int numIterations = (ACOLEND - ACOLSTART) / (%V) ;

    if (numIterations >= 0)
    {
        ACOLENDV = ACOLSTART + (numIterations * (%V));
    } else {
        ACOLENDV = ACOLEND;
    }


    if (ldb % (512) == 0) // PENDING: 512 needs to be a configurable
    {
        //
        // ASSUMPTION(SYMM Variants): \"ACOLSTART\" is perfectly divisble by \"%V\"
        // ACOLSTART depends on the tile size on Y direction
        // Since Vector-sizes are hardly 1, 2,4, 8 or 16, we can assume that
        // this is indeed the case
        //

        //
        // Assumption is that 32/16/8 is divisble by any value in %V
        //
        int num32Iterations = (ACOLENDV - ACOLSTART) / (32/(sizeof(%TYPE)/sizeof(float)));
        if (num32Iterations <= 0)
        {
            ACOL = ACOLSTART;
        } else {
            int startIteration = bidX % num32Iterations;
            ACOL = ACOLSTART + ( startIteration * (32/(sizeof(%TYPE)/sizeof(float))));
        }
    } else {
        ACOL = ACOLSTART;
    }

    for(int itr=0; itr<numIterations; itr++)
    {
        {
            //
            // Load A values
            //
            %IF(%ITEMY) #pragma unroll %ITEMY
            for(int i = 0; i < %ITEMY; i++)
            {
                #ifndef __SYMM_DIAGONAL__
                #ifndef M_TAIL_PRESENT
                    AVAL[i] = %VLOAD(0, (&A[(rowA + i)*lda + ACOL]) );
                #else
                    actualRow = ((rowA + i) >= MV) ? (MV-1) : (rowA + i);
                    AVAL[i] = %VLOAD(0, (&A[actualRow*lda + ACOL]) );
                #endif
                #else
                    #ifndef M_TAIL_PRESENT
                        AVAL[i] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, ACOL, (rowA+i));
                    #else
                        actualRow = ((rowA + i) >= MV) ? (MV-1) : (rowA + i);
                        AVAL[i] = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, ACOL, actualRow);
                    #endif
                #endif

                #ifdef CONJUGATE_A
                    %TYPE%V conjTemp = AVAL[i];
                    %CONJUGATE(1, conjTemp);
                    AVAL[i] = conjTemp;
                #endif

                #ifdef COMPLEX
                AVALEVEN[i] = AVAL[i].even;
                AVALODD[i] = AVAL[i].odd;
                #endif
            }

            //
            // Load B values
            //
            %IF(%ITEMX) #pragma unroll %ITEMX
            for(int j=0; j<(%ITEMX); j++)
            {
                #ifndef N_TAIL_PRESENT
                        BVAL[j] = %VLOAD(0, (&B[ACOL + (colB + j)*ldb]));
                #else
                        actualCol = ((colB + j) >= N) ? (N-1) : (colB + j);
                        BVAL[j] = %VLOAD(0, (&B[ACOL + (actualCol)*ldb]));
                #endif

                #ifdef COMPLEX
                BVALEVEN[j] = BVAL[j].even;
                BVALODD[j] = BVAL[j].odd;
                #endif
            }
        } // LOAD A and B Over


        // MATH Begin
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(int j=0; j<(%ITEMX); j++)
        {
            %IF(%ITEMY) #pragma unroll %ITEMY
            for(int i=0; i<(%ITEMY); i++)
            {
                #ifndef COMPLEX
                %VMAD_AND_REDUCE(CVAL[i][j] ,  AVAL[i], BVAL[j]);
                #else
                CVALEVEN[i][j] = mad(AVALEVEN[i], BVALEVEN[j], CVALEVEN[i][j]);
                CVALEVEN[i][j] = mad(AVALODD[i], -BVALODD[j], CVALEVEN[i][j]);
                CVALODD[i][j]  = mad(AVALEVEN[i], BVALODD[j], CVALODD[i][j]);
                CVALODD[i][j]  = mad(AVALODD[i],  BVALEVEN[j], CVALODD[i][j]);
                /*
                EVENSUM = AVALEVEN[i] * BVALEVEN[j];
                EVENSUM = mad(AVALODD[i], -BVALODD[j], EVENSUM);
                ODDSUM  = AVALEVEN[i]*BVALODD[j];
                ODDSUM  = mad(AVALODD[i],  BVALEVEN[j], ODDSUM);
                CVAL[i][j].S0 += EVENSUM.S0 + EVENSUM.S1;
                CVAL[i][j].S1 += ODDSUM.S0 + ODDSUM.S1;
                */
                #endif
            }
        }

        ACOL = ((ACOL + %V) == ACOLENDV) ? ACOLSTART : (ACOL + %V); //%PANEL
    }

    #ifdef COMPLEX
    {
        %IF(%ITEMY) #pragma unroll %ITEMY
        for(uint i=0; i< (%ITEMY); i++)
        {
            %IF(%ITEMX) #pragma unroll %ITEMX
            for(uint j=0; j<(%ITEMX); j++)
            {
                CVAL[i][j].even =   %REDUCE_SUM_REAL_HV(CVALEVEN[i][j]);
                CVAL[i][j].odd =    %REDUCE_SUM_REAL_HV(CVALODD[i][j]);
            }
        }
    }
    #endif

    ACOL = ACOLENDV;

    for(; ACOL < ACOLEND; ACOL ++)
    {
        //
        // Load B values
        //
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint bcol = 0; bcol < %ITEMX; bcol++)
        {
            //
            // PENDING: PANEL iteration to Load the Panel Depth iterating by %V
            //
            #ifndef N_TAIL_PRESENT
            BVAL[bcol] = %VMAKEVEC(B[ACOL + (colB + bcol)*ldb]);
            #else
            BVAL[bcol] = %VMAKEVEC(B[ACOL + ((colB + bcol)%(N))*ldb]);
            #endif
        }

        //
        // Load A values
        //
        %IF(%ITEMY) #pragma unroll %ITEMY
        for(uint i = 0; i < (%ITEMY); i++) // 1 * ITEMY/V
        {
            #ifndef __SYMM_DIAGONAL__
            {
                #ifndef M_TAIL_PRESENT
                AVAL[i] = %VMAKEVEC(A[(rowA + i)*lda + ACOL]);
                #else
                AVAL[i] = %VMAKEVEC(A[((rowA + i) % MV)*lda + ACOL]);
                #endif
            }
            #else
            {
                %TYPE t;
                #ifndef M_TAIL_PRESENT
                t = SYMM_SCALAR_LOAD(A, M, lda, ACOL, (rowA+i) );
                #else
                t = SYMM_SCALAR_LOAD(A, M, lda, ACOL, ((rowA + i) % MV));
                #endif
                AVAL[i] = %VMAKEVEC(t);
            }
            #endif
            #ifdef CONJUGATE_A
                %CONJUGATE(1, AVAL[i]);
            #endif
        }

        {
            %IF(%ITEMY) #pragma unroll %ITEMY
            for(uint i=0; i<(%ITEMY); i++)
            {
                %IF(%ITEMX) #pragma unroll %ITEMX
                for(uint j=0; j<(%ITEMX); j++)
                {
                    %MAD_AND_REDUCE(CVAL[i][j] ,  AVAL[i] , BVAL[j]);
                }
            }
        }
    }


    //
    // STORE Result in C
    //
    %TYPE%V reg , betareg, alphareg;
    %TYPE reg_s , betareg_s, alphareg_s;
    %TYPE%V alphav, betav;
    alphav = %VMAKEVEC(alpha);
    betav = %VMAKEVEC(beta);
    //%TYPE CVALV_TEMP[%V];
    %TYPE%V CVALV;

    #ifndef HERK
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
                //#pragma unroll %V
                //for(uint k=0; k< (%V); k++)
                %VFOR
                {
                    CVALV%VFORSUFFIX = CVAL[i*V + %VFORINDEX][j];
                }
                //CVALV = *(__private %TYPE%V *)CVALV_TEMP;

            #if !defined(M_TAIL_PRESENT) && !defined(N_TAIL_PRESENT)
                reg = %VLOAD(0, (&C[(rowA + i*V) +     (colB+j)*ldc]));
                %VMUL(betareg, betav, reg);
                %VMUL(alphareg, alphav, CVALV);
                %ADD( reg, betareg, alphareg);
                %VSTORE(reg, 0, (&C[(rowA + i*V) + (colB+j)*ldc]));
            #else
                if (((rowA + i*V + V - 1) < M) && ((colB + j) < N))
                {
                    reg = %VLOAD(0, (&C[rowA + i*V +     (colB+j)*ldc]));
                    %VMUL(betareg, betav, reg);
                    %VMUL(alphareg, alphav, CVALV);
                    %ADD( reg, betareg, alphareg);
                    %VSTORE(reg, 0, (&C[(rowA + i*V) + (colB+j)*ldc]));
                } else {
                    if ((colB + j) < N)
                    {
                        //%TYPE TEMP[%V];
                        //*(__private %TYPE%V *) TEMP = CVALV;
                        //#pragma unroll %V
                        //for(uint v=0; ((v< %V) && ((rowA + (i * %V) + v) < M) ); v++)
                        %VFOR
                        {
                            if (((rowA + (i * %V) + %VFORINDEX) < M) )
                            {
                                %TYPE c;

                                c = C[rowA + i*V + %VFORINDEX + (colB+j)*ldc];
                                %MUL(betareg_s, c, beta);
                                c = CVALV%VFORSUFFIX;
                                %MUL(alphareg_s, c, alpha);
                                %ADD(c, betareg_s, alphareg_s);
                                C[rowA + i*V + %VFORINDEX + (colB+j)*ldc] = c;
                           }
                        }
                    }
                }
            #endif
        }
    }
    #else
    %IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
    for(uint i=0; i< (%ITEMY_BY_V); i++)
    {
        %IF(%ITEMX) #pragma unroll %ITEMX
        for(uint j=0; j<(%ITEMX); j++)
        {
            int actualRow = rowA + i*V;
            int actualCol = colB + j;

            //#pragma unroll %V
            //for(uint k=0; k< (%V); k++)
            %VFOR
            {
                CVALV%VFORSUFFIX = CVAL[i*V + %VFORINDEX][j];
            }
            //CVALV = *(__private %TYPE%V *)CVALV_TEMP;

            #if !defined(M_TAIL_PRESENT) && !defined(N_TAIL_PRESENT)
                %VMUL(alphareg, alphav, CVALV);
                //%TYPE temp[%V];
                //*(__private %TYPE%V *)(&temp) = alphareg;
                //#pragma unroll %V
                //for(uint r = 0; r < %V; r++)
                %VFOR
                {
                    #ifdef HERK_LOWER_TRIANGLE
                    if((%VFORINDEX + actualRow) >= (actualCol))
                    #else
                    if((%VFORINDEX + actualRow) <= (actualCol))
                    #endif
                    {
                        %TYPE C_s =  C[%VFORINDEX + actualRow + actualCol * ldc];
                        %TYPE beta_s;
                        %MUL(beta_s, beta, C_s);
                        C_s = alphareg%VFORSUFFIX + beta_s;
                        if((%VFORINDEX + actualRow) == actualCol)
                        {
                            C_s.odd = 0.0f;
                        }
                        C[%VFORINDEX + actualRow + actualCol * ldc] = C_s;
                    }
                }
            #else
                if (((rowA + i*V + V - 1) < M) && ((colB + j) < N))
                {
                    %VMUL(alphareg, alphav, CVALV);
                    //%TYPE temp[%V];
                    //*(__private %TYPE%V *)(&temp) = alphareg;
                    //#pragma unroll %V
                    //for(uint r = 0; r < %V; r++)
                    %VFOR
                    {
                        #ifdef HERK_LOWER_TRIANGLE
                        if((%VFORINDEX + actualRow) >= (actualCol))
                        #else
                        if((%VFORINDEX + actualRow) <= (actualCol))
                        #endif
                        {
                            %TYPE C_s =  C[%VFORINDEX + actualRow + actualCol * ldc];
                            %TYPE beta_s;
                            %MUL(beta_s, beta, C_s);
                            C_s = alphareg%VFORSUFFIX + beta_s;
                            if((%VFORINDEX + actualRow) == actualCol)
                            {
                                C_s.odd = 0.0f;
                            }
                            C[%VFORINDEX + actualRow + actualCol * ldc] = C_s;
                        }
                    }
                }
                else
                {
                    if ((colB + j) < N)
                    {
                        //%TYPE TEMP[%V];

                        //*(__private %TYPE%V *)(&TEMP) = CVALV;
                        //#pragma unroll %V
                        //for(uint r=0; ((r< %V) && ((rowA + (i * %V) + r) < M) ); r++)
                        %VFOR
                        {
                            if (((rowA + (i * %V) + %VFORINDEX) < M))
                            {
                                #ifdef HERK_LOWER_TRIANGLE
                                if((%VFORINDEX + actualRow) >= (actualCol))
                                #else
                                if((%VFORINDEX + actualRow) <= (actualCol))
                                #endif
                                {
                                    %TYPE c;
                                    c = C[%VFORINDEX + actualRow + (actualCol)*ldc];
                                    %MUL(betareg_s, c, beta);
                                    c = CVALV%VFORSUFFIX;
                                    %MUL(alphareg_s, c, alpha);
                                    %ADD(c, betareg_s, alphareg_s);
                                    if((%VFORINDEX + actualRow) == (actualCol))
                                    {
                                        c.odd = 0.0f;
                                    }
                                    C[%VFORINDEX + actualRow  + actualCol * ldc] = c;
                                }
                            }
                        }
                    }
                }
            #endif
        }
    }
    #endif
    return;
}
";

