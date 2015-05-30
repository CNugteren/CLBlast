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


// Column-Major Case

static const char *ger_C_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define BH %BH_DEF
#define BW %BW_DEF

__kernel void %PREFIXger_C_kernel( __global %TYPE const* restrict _X, __global %TYPE const* restrict _Y, __global %TYPE* _A,
				uint M, uint N, uint offx, int incx, uint offy, int incy, uint offa, uint lda,
				%TYPE alpha, int doConj )
{
	__global %TYPE* A;
	__global %TYPE const* restrict X;
	__global %TYPE const* restrict Y;

	A = _A + offa;
	X = _X + offx;
	Y = _Y + offy;

	if ( incx < 0 ) // Goto end of vector
	{
		X = X + ( M - 1) * abs(incx);
	}

	if ( incy < 0 ) // Goto end of vector
	{
		Y = Y + ( N - 1) * abs(incy);
	}

	// create local memory
	__local %TYPE localX[ BH * %V ];
	__local %TYPE localY[ BW ];

	uint lID = get_local_id( 0 );
	uint gID = get_group_id( 0 );

	uint tIDy = lID & ( BH-1 );  //get y coordinate of a thread in 1D workgroup
	uint tIDx = lID / BH;        //get x coordinate of a thread in !D workgroup
    uint nBlocksX = (( N + BW - 1) / BW );
    uint nBlocksY = (( M + BH * %V - 1 ) / ( BH * %V ));

	uint gIDy = gID % nBlocksY;	//get y coordinate of a workgroup in 1D grid
    uint gIDx = gID / nBlocksY;	// get x coordinate of a workgroup in a 1D grid

    uint row = (( BH * gIDy)+  tIDy) * %V;
    uint col = (( BW * gIDx)+  tIDx);


    if( (gIDx != (nBlocksX-1)) && (gIDy != (nBlocksY-1)) )       // Completely vector blocks
    {
        //populate local memory
        for( int i = lID; i< ( BH * %V); i+= get_local_size(0) )
        {
            int idx = i + ( gIDy * BH * %V);
            localX[ i ] = *(X + (idx * incx));
        }

        for( int i = lID; i< BW; i+= get_local_size(0) )
        {
            int idx = i + ( gIDx * BW);
            localY[ i ] = *(Y + (idx * incy));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        %TYPE%V prevA, temp;
        %TYPE yReg = localY[  tIDx ];
        %TYPE%V xReg = *(__local %TYPE%V*)(&localX[ tIDy * %V]);

        prevA = %VLOAD( 0, ( A + col*lda + row ) );
        %CONJUGATE(doConj, yReg);
        %VMUL( temp, xReg, alpha );
        %VMAD( prevA, temp, yReg);
        %VSTORE( prevA, 0 , ( A + col*lda + row ) );

    }
    else                            // Border blocks in both X & Y direction
    {
    	//populate local memory
        for( int i = lID; i< ( BH * %V); i+= get_local_size(0) )
        {
    		int idx = i + ( gIDy * BH * %V);
           	if ( idx < M )
    		{
    			localX[ i ] = *(X + (idx * incx));
        	}
    	}

        for( int i = lID; i< BW; i+= get_local_size(0) )
        {
    		int idx = i + ( gIDx * BW);
    		if ( idx < N)
    		{
           		localY[ i ] = *(Y + (idx * incy));
        	}
    	}
    	barrier(CLK_LOCAL_MEM_FENCE);

    	uint gTIDx = (gIDx * BW) + tIDx;
        if ( gTIDx < N)  // if whithin last column
    	{
    		if( (row + %V - 1) < M )  // if the next V rows are still within M, then do vector math
	    	{
    			%TYPE%V prevA, temp;
    			%TYPE yReg = localY[  tIDx ];
    			%TYPE%V xReg = *(__local %TYPE%V*)(&localX[ tIDy * %V]);

	    		prevA = %VLOAD( 0, ( A + col*lda + row ) );
		    	%CONJUGATE(doConj, yReg);
			    %VMUL( temp, xReg, alpha );
    			%VMAD( prevA, temp, yReg);
    			%VSTORE( prevA, 0 , ( A + col*lda + row ) );

    		}
	    	else if( row < M  )  //else do scalar multiplication
		    {
    			%TYPE xRegS, yReg, prevA, temp;
    			for( int i=row; i<M; i++ )
    			{
    				yReg  = localY[ tIDx ];
    				xRegS = localX[ (tIDy * %V) + (i-row) ];
    				prevA = A[ col*lda + i];
	    			%CONJUGATE(doConj, yReg);
    				%MUL( temp, xRegS, alpha );
    				%MAD( prevA, temp, yReg );
	    			A[ col*lda + i ] = prevA;
		    	}
		    }
	    }
    }
}
\n";



//Row major kernel

static const char *ger_R_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define BH %BH_DEF
#define BW %BW_DEF

__kernel void %PREFIXger_R_kernel( __global %TYPE const* restrict _X, __global %TYPE const* restrict _Y, __global %TYPE* _A,
				uint M, uint N, uint offx, int incx, uint offy, int incy, uint offa, uint lda,
				%TYPE alpha, int doConj )
{
	__global %TYPE* A;
	__global %TYPE const* restrict X;
	__global %TYPE const* restrict Y;

	A = _A + offa;
	X = _X + offx;
	Y = _Y + offy;

	if ( incx < 0 ) // Goto end of vector
	{
		X = X + ( M - 1) * abs(incx);
	}

	if ( incy < 0 ) // Goto end of vector
	{
		Y = Y + ( N - 1) * abs(incy);
	}

    __local %TYPE localX[ BH ];
    __local %TYPE localY[ BW * %V ];

    uint lID = get_local_id( 0 );
    uint gID = get_group_id( 0 );

    uint tIDy = lID / BW;
    uint tIDx = lID & ( BW - 1);
    uint nBlocksY = (( M + BH - 1) / BH );
    uint nBlocksX = (( N + BW * %V - 1 ) / ( BW * %V ));

    uint gIDy = gID / nBlocksX;
    uint gIDx = gID % nBlocksX;

    uint row = (( BH * gIDy)+  tIDy);
    uint col = (( BW * gIDx)+  tIDx) * %V;

    if( (gIDy != (nBlocksY-1)) && (gIDx != (nBlocksX-1)) )         // Perfectly vector blocks
    {
        //populate local memory
        for( int i = lID; i< ( BW * %V); i+= get_local_size(0) )
        {
            int idx = i + ( gIDx * BW * %V);
            localY[ i ] = *(Y + (idx * incy));
        }

        for( int i = lID; i< BH; i+= get_local_size(0) )
        {
            int idx = i + ( gIDy * BH);
            localX[ i ] = *(X + (idx * incx));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        %TYPE%V prevA, temp;
        %TYPE xReg = localX[ tIDy ];
        %TYPE%V yRegS = *(__local %TYPE%V*)(&localY[ tIDx * %V]);

        prevA = %VLOAD( 0, ( A + row*lda + col ) );
        %CONJUGATE(doConj, yRegS);
        %VMUL( temp, yRegS, alpha );
        %VMAD( prevA, temp, xReg );
        %VSTORE( prevA, 0 , ( A + row*lda + col ) );
    }
    else
    {
        //populate local memory
        for( int i = lID; i< ( BW * %V); i+= get_local_size(0) )
        {
            int idx = i + ( gIDx * BW * %V);
            if ( idx < N)
            {
                localY[ i ] = *(Y + (idx * incy));
            }
        }

        for( int i = lID; i< BH; i+= get_local_size(0) )
        {
            int idx = i + ( gIDy * BH);
            if ( idx < M)
            {
                localX[ i ] = *(X + (idx * incx));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint gTIDy = gIDy * BH + tIDy;
        if ( gTIDy < M)
        {
            if( (col + %V - 1) < N )
            {
                %TYPE%V prevA, temp;
                %TYPE xReg = localX[ tIDy ];
                %TYPE%V yRegS = *(__local %TYPE%V*)(&localY[ tIDx * %V]);

                prevA = %VLOAD( 0, ( A + row*lda + col ) );
                %CONJUGATE(doConj, yRegS);
    			%VMUL( temp, yRegS, alpha );
    			%VMAD( prevA, temp, xReg );
                %VSTORE( prevA, 0 , ( A + row*lda + col ) );

            }
            else if( col < N  )
            {
                %TYPE xReg, yRegS, prevA, temp;
                for( int i=col; i<N; i++ )
                {
                    yRegS = localY[ (tIDx * %V) + (i-col) ];
                    xReg  = localX[ tIDy ];
                    prevA = A[ row*lda + i];
                    %CONJUGATE(doConj, yRegS);
    				%MUL( temp, yRegS, alpha );
    				%MAD( prevA, temp, xReg );
    				A[ row*lda + i ] = prevA;
                }
            }
        }
    }
}
\n";
