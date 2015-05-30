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


// Row-Major Non-transpose case
static const char *gbmv_RNT_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define  H  ( %DEF_H )
#define  TARGET_ROWS  ( %DEF_TARGET_ROWS )

__kernel void %PREFIXgbmv_RNT_kernel( __global const %TYPE * _A, __global %TYPE * _y_vector, __global %TYPE const* restrict _x_vector,
                                        uint M, uint N, uint KL, uint KU, uint lda, int incx, int incy, uint offa, uint offx, uint offy
#ifndef TBMV_ONLY
                                    ,%TYPE alpha, %TYPE beta
#endif
                                                              )
{
    __global const %TYPE * A;
	__global %TYPE const* restrict X;
	__global %TYPE* Y;
	__local %TYPE localRed[ (H+1) * TARGET_ROWS ];  // To avoid bank-conflict

	A = _A + offa;
	if ( incx < 0 )                     // Goto end of vector
		X = _x_vector + offx + ( N - 1 ) * abs(incx);
	else
		X = _x_vector + offx;

	if( incy < 0 )
		Y = _y_vector + offy + ( M - 1 ) * abs(incy);
    else
		Y = _y_vector + offy;

    int gId = get_group_id( 0 );
    int lId = get_local_id( 0 );
    int threadRow = (lId / H);
    int threadCol = (lId % H);
    int row = ( gId * TARGET_ROWS ) + threadRow;
    int AStartColIndex = max( (int)(KL-row), 0 );
    int XStartIndex = ( row <= KL )? 0: (int)(row-KL);
    bool diagPresent = ( row < N ) ? true: false;
    int numSubDiags = min( row, max( 0, min( ((int)min( KL, N )), (int)(N+KL-row) ) ) );
    int numSupDiags = max( 0, min( (int)KU, (int)(N-1-row) ) );
    %TYPE reg1, reg2, sum;

    if( row < M )
    {
        sum = %MAKEVEC(0.0);
        localRed[ lId ] = %MAKEVEC(0.0);
        // Sub-diagonal iteration
        #ifdef GIVEN_SHBMV_UPPER
            int symmStartRow = max( 0, (row - (int)KU) );       // row - (BW-1) = KU
            int symmStartCol = min( (int)KU, row );             // row - (BW-1) = KU
        #endif
        for( int i=threadCol; i<numSubDiags; i+= H )
        {
            #ifdef GIVEN_SHBMV_UPPER
                reg1 = A[ ((symmStartRow+i) * lda) + (symmStartCol - i) ];
                %CONJUGATE(1 , reg1);                           // Hermitian transpose- will be ignored for real cases
            #else
                reg1 = A[ (row * lda) + (AStartColIndex + i) ];
            #endif

            #ifdef DO_CONJ
                %CONJUGATE(1 , reg1);
            #endif
            reg2 = X[ (XStartIndex + i) * incx ];
            %MAD( sum, reg1, reg2 );
        }
        #ifdef GIVEN_SHBMV_UPPER
            AStartColIndex = 0;
        #else
            AStartColIndex += numSubDiags;
        #endif
        XStartIndex += numSubDiags;

        // Calculate diagonal component -- only by first thread of the row
        if( diagPresent )
        {
            if( threadCol == 0 )
            {
                reg2 = X[ XStartIndex * incx ];
                #ifndef UNIT_DIAG
                    reg1 = A[ (row * lda) + AStartColIndex ];
                    #ifdef DO_CONJ
                        %CONJUGATE(1 , reg1);
                    #endif
                    #ifdef HBMV_ONLY
                        reg1.odd = 0.0;                 // Imaginary part of diagonal is assumed to be zero
                    #endif
                    %MAD( sum, reg1, reg2 );
                #else
                    sum += reg2;
                #endif
            }
            AStartColIndex ++;
            XStartIndex ++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if( row < M )
    {
	    // Super-diagonal iteration
	    #ifdef GIVEN_SHBMV_LOWER
            AStartColIndex = (KL+1) - 2;            // KL+1 is BW
        #endif

        for( int i=threadCol; i<numSupDiags; i+= H )
        {
            #ifdef GIVEN_SHBMV_LOWER
                reg1 = A[ ((row+i+1) * lda) + (AStartColIndex - i) ];
                %CONJUGATE(1 , reg1);                           // Hermitian transpose- will be ignored for real cases
            #else
                reg1 = A[ (row * lda) + (AStartColIndex + i) ];
            #endif

            #ifdef DO_CONJ
                %CONJUGATE(1 , reg1);
            #endif
            reg2 = X[ (XStartIndex + i) * incx ];
            %MAD( sum, reg1, reg2 );
        }
        localRed[ (threadRow * (H+1)) + threadCol ] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Update the Y vector
    if( (threadCol == 0) && (row < M) )
    {
        sum = %MAKEVEC(0.0);
        for( int i=0; i<H; i++ )
        {
            %ADD( sum, sum, localRed[ (threadRow * (H+1)) + i ] );
        }
        #ifndef TBMV_ONLY
            %MUL( reg1, alpha, sum );
            %MUL( reg2, beta, Y [ row * incy ] );
            %ADD( Y[ row * incy ], reg1, reg2 );
        #else
            Y[ row * incy ] = sum;
        #endif
    }

}
";



// Row-Major Transpose case
static const char *gbmv_RT_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define TARGET_ROWS  ( %DEF_TARGET_ROWS )
#define HEIGHT ( %DEF_H)
#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void %PREFIXgbmv_RT_kernel( __global const %TYPE * _A, __global %TYPE * _y_vector, __global %TYPE const* restrict _x_vector,
                                    uint M, uint N, uint KL, uint KU, uint lda, int incx, int incy, uint offa, uint offx, uint offy
#ifndef TBMV_ONLY
                                    ,%TYPE alpha, %TYPE beta
#endif
                                    )
{
    __global const %TYPE * A = _A + offa;
    __global %TYPE const * restrict X;
    __global %TYPE *Y;

    if ( incx < 0 ) // Goto end of X vector
    {
        X = _x_vector + offx + ( M - 1) * abs(incx);
    }
    else
    {
        X = _x_vector + offx;
    }

    if( incy < 0 ) // Goto end of Y vector
    {
        Y = _y_vector + offy + ( N - 1) * abs(incy);
    }
    else
    {
        Y = _y_vector + offy;
    }

    int blkID, thrID;
    int blkColIndx, blkStrtCol, blkStrtRow, blkOffset;
    int thrRow, thrCol;
    int bandWidth = KL + KU + 1;

    blkID = get_group_id(0);
    thrID = get_local_id(0);

    //Find the block start column and start row.
    blkOffset = blkID * HEIGHT;
    blkColIndx = (blkOffset) + KL;
    blkStrtCol = (blkColIndx >= bandWidth) ? (bandWidth - 1) : blkColIndx;
    blkStrtRow = ((blkColIndx - (bandWidth - 1)) < 0) ? 0 : (blkColIndx - (bandWidth - 1));
    %TYPE thrSum = %MAKEVEC(0.0); //Private sum for each thread
    %TYPE reg1, reg2;

    if(((blkOffset) + (thrID % HEIGHT)) < N)
    {
        thrRow = blkStrtRow + ((int)thrID / (HEIGHT));
        thrCol = blkStrtCol + ((int)thrID % (HEIGHT)) - ((int)thrID / (HEIGHT));
        while((thrRow < M) && (thrCol >= 0))
        {
            if(thrCol < bandWidth)
            {
                reg2 = X[ thrRow * incx];
                #ifdef UNIT_DIAG
                    if(thrCol == ((int)KL))
                    {
                        thrSum += reg2;
                    }
                    else
                    {
                        reg1 = A[(thrRow*lda) + thrCol];
                        #ifdef DO_CONJ
                            %CONJUGATE(1 , reg1);
                        #endif
                        %MAD(thrSum, reg1, reg2);
                    }
                #else
                    reg1 = A[(thrRow*lda) + thrCol];
                    #ifdef DO_CONJ
                        %CONJUGATE(1 , reg1);
                    #endif
                    %MAD(thrSum, reg1, reg2);
                #endif
                //thrSum += A[(thrRow*lda) + thrCol ] * X[ thrRow ];
            }
            thrRow += TARGET_ROWS;
            thrCol -= TARGET_ROWS;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Store the results in a temporary local buffer and accumulate the same.
    __local %TYPE sum[(TARGET_ROWS * HEIGHT)];
    sum[thrID] = thrSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if((thrID < HEIGHT) && ((blkOffset + (thrID % HEIGHT)) < N))
    {
        int iY = (blkOffset) + thrID;
        %TYPE tempSum = %MAKEVEC(0.0);
        for(int i = 0; i < TARGET_ROWS; i++)
        {
            reg1 = sum[thrID + (i * HEIGHT)];
            %ADD(tempSum, tempSum, reg1);
            //tempSum += sum[thrID + (i * HEIGHT)];
        }
        #ifndef TBMV_ONLY
            %MUL(reg1, alpha, tempSum);
            %MUL(reg2, beta, Y[iY * incy]);
            %ADD(Y[iY * incy], reg1, reg2);
        #else
            Y[ iY * incy ] = tempSum;
        #endif
        //Y[iY] = ((alpha * tempSum) + (beta * Y[iY]));
    }
}
";


