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

/************************************************/
//NOTE: THIS FILE IS NOT USED. SEE SYR2_HER2.CLT
//      THIS FILE IS FOR LEGACY PURPOSES.

//Column Major Lower
static const char *her2_CL_kernel = "

#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define TARGET_ROWS_BY_VEC 	(%TARGET_ROWS / %V)
#define TARGET_WIDTH		(%BLOCKSIZE / TARGET_ROWS_BY_VEC )
#define TARGET_HEIGHT		(%BLOCKSIZE / TARGET_ROWS_BY_VEC )
// Column-Major Lower
// nBlocks 	= (N - 1)/ TR + 1
// totalBlocks 	= (nBlocks * ( nBlocks + 1)) / 2
__kernel void %PREFIXher2_CL_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

	__global const %TYPE* X;
	__global const %TYPE* Y;
	__global %TYPE* A;

	__local %TYPE xShared[%TARGET_ROWS];
	__local %TYPE yShared[%TARGET_ROWS];
	__local %TYPE xSharedConj[%TARGET_ROWS];
	__local %TYPE ySharedConj[%TARGET_ROWS];

	if( (alpha.even == 0.0) && (alpha.odd == 0.0) )
		return;

	int nBlocks = ((N - 1) / %TARGET_ROWS) + 1;

	A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		X	 = _X + offx - ( N - 1) * incx;
	}
	else
	{
		X = _X + offx;
	}

	if ( incy < 0 ) // Goto end of vector
	{
		Y	 = _Y + offy - ( N - 1) * incy;
	}
	else
	{
		Y = _Y + offy;
	}

	int blockID  = get_group_id(0);
	int threadID = get_local_id(0);

	__local int iShared;
	__local int jShared;

	// Get (i,j) of Block
	if ( threadID == 0)
	{
		int _i = 0, _j = 0;
		for ( _j = (blockID / nBlocks); _j < nBlocks; _j++)
		{
			_i = blockID - ((_j*((2* nBlocks) + 1 - _j))/2) + _j;
			if ( _i < nBlocks && ( _i >= 0) )
			{
				break;
			}
		}

		iShared = _i;
		jShared = _j;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int i = iShared;
	int j = jShared;


	int ref_x = i * %TARGET_ROWS;
	int ref_y = j * %TARGET_ROWS;
	%TYPE conjAlpha	 = alpha;
	%CONJUGATE( 1, conjAlpha );

	// Load data into xShared and yShared
	// Not a common task among blocks in the present implementation..
	// Diagonal Blocks : Should handle not reading diagonal element complex value
	// Diagonal blocks : Should handle the last block as well
	// Scalar code in Present implementation
	if ( i == j)
	{
		int ncols = ((ref_y + %TARGET_ROWS) < N) ? %TARGET_ROWS : (N-ref_y);
        int nrows = ((ref_x + %TARGET_ROWS) < N) ? %TARGET_ROWS : (N-ref_x);
        int nElements = ((nrows) * ((ncols) + 1)) >> 1;
        nrows -= 1;
        ncols -= 1;
        for(i = threadID; i < nElements; i += get_local_size(0))
        {
            int r = -1, c = -1;
            for(int k = 1; (k <= %TARGET_ROWS); k ++)
            {
                int temp = ((k - 1) * k) >> 1;
                r = ((i >= temp) && (i <= (temp + k - 1))) ? k - 1 : r;
            }
            c = i - (((r + 1) * r) >> 1);

            r = ref_x + r;
            c = ref_y + c;

            %TYPE res1, res2, res3, res4, res5;
            res1 = X[r * incx];
            res2 = X[c * incx];
            #ifdef HER2_ROWMAJOR
				%CONJUGATE(1, res1);
			#endif
            %MUL( res5, alpha, res1 );
            res1 = Y[c * incx];
            res3 = Y[r * incx];
            #ifndef HER2_ROWMAJOR
				%CONJUGATE(1, res1);
            #endif
            %MUL( res4, res5, res1 );
            #ifdef HER2_ROWMAJOR
				%CONJUGATE(1, res3);
			#else
				%CONJUGATE(1, res2);
			#endif
            %MUL( res5, conjAlpha, res3 );
            %MAD( res4, res5, res2 );
            res1 = A[r + c * lda];
            %ADD( res2, res1, res4 );
 /* HER2 defn: On output, if alpha not equal to 0.0, then imaginary part of A is set to zero. */

			res2.odd = (r == c) ? 0.0 : res2.odd;


			A[r + c * lda] = res2;
        }
	}
	else if ( i == (nBlocks - 1)) // Last Row Strip blocks ( May not fit into target region)
	{

		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x + threadID); i < N; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifdef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xShared[i - ref_x] = loadedX;
			yShared[i - ref_x] = loadedY;
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y  + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifndef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xSharedConj[(i - ref_y) ] = loadedX;
			ySharedConj[(i - ref_y) ] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_WIDTH)  + 1;

		int colShift = threadID / TARGET_ROWS_BY_VEC;
		int rowShift = (threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V;


		int startRow = ref_x + rowShift;
		%TYPE%V  loadedX, loadedY;

		if ( startRow  < (N - (%V - 1)) )
		{
			loadedX=  *((__local %TYPE%V*)( xShared + rowShift));
			loadedY=  *((__local %TYPE%V*)( yShared + rowShift));
		}

		for( int i= 1; i <=  nLoops; i++)
		{
			int startCol = ref_y + colShift + ( i - 1 ) * TARGET_WIDTH;

			if ( ( startRow  < N ) && ( startCol  < (ref_y + %TARGET_ROWS ) ) )// threads that fall into target region
			{
				if(( startRow + %V) > N )// Loop serially as can't do VLOAD
				{
					%TYPE yValueConj = ySharedConj[ startCol - ref_y];
					%TYPE xValueConj = xSharedConj[ startCol - ref_y];

					for(int row = startRow; row < N; row++)
					{
						%TYPE xValue = xShared[ row - ref_x];
						%TYPE yValue = yShared[ row - ref_x];

						%TYPE res1, res2;
						// X * Y(H)
						%MUL(res1, alpha, yValueConj);
						%MUL( res2, res1,  xValue);

						// Y * X(H)
						%MUL(res1, conjAlpha, xValueConj);
						%MAD( res2, res1,  yValue);
						A[ row + startCol * lda] += res2;
					}
				}
				else
				{
					loadedA  	= %VLOAD( 0, (&A[ startRow + startCol * lda]));

					%TYPE 	 loadedYConj = ySharedConj[ startCol - ref_y];
					%TYPE 	 loadedXConj = xSharedConj[ startCol - ref_y];
					%TYPE 	 res;

					// X * Y(H)
					%MUL(res, loadedYConj, alpha);
					%TYPE%V  resVec;
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedX, resVec);

					// Y * X(H)
					%MUL(res, loadedXConj, conjAlpha);
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedY, resVec);

					%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
				}
			}
		}
	}
	else // blocks that fit exactly.
	{

		%TYPE%V loadedA;

		// Populating xShared
		for( int i = (ref_x + threadID); (i - ref_x) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifdef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xShared[i - ref_x] = loadedX;
			yShared[i - ref_x] = loadedY;
		}

		// Populating yShared
		for( int i = (ref_y + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifndef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xSharedConj[i - ref_y] = loadedX;
			ySharedConj[i - ref_y] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_WIDTH)  + 1;

		int colShift = threadID / TARGET_ROWS_BY_VEC;
		int rowShift = (threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V;

		int startRow = ref_x + rowShift;
		int startCol = ref_y + colShift;
		%TYPE%V  loadedX, loadedY;

		if ( startCol < ( ref_y + %TARGET_ROWS) ) // threads that fall into target region
		{
			loadedX	 =  *((__local %TYPE%V*)( xShared + rowShift));
			loadedY	 =  *((__local %TYPE%V*)( yShared + rowShift));
		}

		//#pragma unroll
		for( int i= 1; i <= nLoops; i++)
		{
			startCol = ref_y + colShift + ( i - 1 ) * TARGET_WIDTH;

			if ( startCol < ( ref_y + %TARGET_ROWS) ) // threads that fall into target region
			{
				loadedA  	 = %VLOAD( 0, (&A[ startRow + startCol * lda]));
				%TYPE 	 loadedYConj = ySharedConj[ startCol - ref_y];
				%TYPE 	 loadedXConj = xSharedConj[ startCol - ref_y];

				// X * Y(H)
				%TYPE 	 res;
				%MUL(res, loadedYConj, alpha);
				%TYPE%V  resVec;
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedX, resVec);

				// Y * X(H)
				%MUL(res, loadedXConj, conjAlpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedY, resVec);
				%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
			}
		}
	}

}
\n";

//Column Major Upper
static const char *her2_CU_kernel = "

#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define TARGET_ROWS_BY_VEC  (%TARGET_ROWS / %V)
#define TARGET_WIDTH        (%BLOCKSIZE / TARGET_ROWS_BY_VEC )
#define TARGET_HEIGHT       (%BLOCKSIZE / TARGET_ROWS_BY_VEC )
// Column-Major Upper
// nBlocks 	= (N - 1)/ TR + 1
// totalBlocks 	= (nBlocks * ( nBlocks + 1)) / 2
__kernel void %PREFIXher2_CU_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

    __global const %TYPE* X;
    __global const %TYPE* Y;
    __global %TYPE* A;

    __local %TYPE xShared[%TARGET_ROWS];
    __local %TYPE yShared[%TARGET_ROWS];
    __local %TYPE xSharedConj[%TARGET_ROWS];
    __local %TYPE ySharedConj[%TARGET_ROWS];

	if( (alpha.even == 0.0) && (alpha.odd == 0.0) )
		return;

    int nBlocks = ((N - 1) / %TARGET_ROWS) + 1;

    A = _A + offa;

    if ( incx < 0 ) // Goto end of vector
    {
        X    = _X + offx - ( N - 1) * incx;
    }
    else
    {
        X = _X + offx;
    }

    if ( incy < 0 ) // Goto end of vector
    {
        Y    = _Y + offy - ( N - 1) * incy;
    }
    else
    {
        Y = _Y + offy;
    }

	int blockID  = get_group_id(0);
	int threadID = get_local_id(0);

	__local int iShared;
	__local int jShared;

	// Get (i,j) of Block
	if ( threadID == 0)
	{
		int _i = 0, _j = 0;
		//for ( _j = 0; _j < nBlocks; _j++)
		for ( _j = (blockID / nBlocks); _j < nBlocks; _j++)
		{
			_i = blockID - ((_j*((2* nBlocks) + 1 - _j))/2) + _j;
			if ( _i < nBlocks && ( _i >= 0) )
			{
				break;
			}
		}

		iShared = _i;
		jShared = _j;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int i = iShared;
	int j = jShared;
	%TYPE conjAlpha	 = alpha;
	%CONJUGATE( 1, conjAlpha );

	int ref_x = (N- 1) - i * %TARGET_ROWS;
	int ref_y = (N- 1) - j * %TARGET_ROWS;

	// Load data into xShared and yShared
	// Not a common task among blocks in the present implementation..

	// Diagonal Blocks : Should handle not reading diagonal element complex value
	// Diagonal blocks : Should handle the last block as well
	// Scalar code in Present implementation
	if ( i == j)
	{
 		int ncols = ((ref_y - %TARGET_ROWS) >= 0) ? %TARGET_ROWS : (ref_y+1);
        int nrows = ((ref_x - %TARGET_ROWS) >= 0) ? %TARGET_ROWS : (ref_x+1);
        int nElements = ((nrows) * ((ncols) + 1)) >> 1;
        nrows -= 1;
        ncols -= 1;
        for(i = threadID; i < nElements; i += get_local_size(0))
        {
            int r, c = -1;
            for(int k = 1; (k <= %TARGET_ROWS); k ++)
            {
                int temp = ((k - 1) * k) >> 1;
                c = ((i >= temp) && (i <= (temp + k - 1))) ? k - 1 : c;
            }
            r = i - (((c + 1) * c) >> 1);

            r = ref_x - (nrows) + r;
            c = ref_y - (ncols) + c;

            %TYPE res1, res2, res3, res4, res5;
            res1 = X[r * incx];
            res2 = X[c * incx];
            #ifdef HER2_ROWMAJOR
				%CONJUGATE(1, res1);
			#endif
            %MUL( res5, alpha, res1 );
            res1 = Y[c * incx];
            res3 = Y[r * incx];
            #ifndef HER2_ROWMAJOR
				%CONJUGATE(1, res1);
            #endif
            %MUL( res4, res5, res1 );
            #ifdef HER2_ROWMAJOR
				%CONJUGATE(1, res3);
			#else
				%CONJUGATE(1, res2);
			#endif
            %MUL( res5, conjAlpha, res3 );
            %MAD( res4, res5, res2 );
            res1 = A[r + c * lda];
            %ADD( res2, res1, res4 );
 /* HER2 defn: On output, if alpha not equal to 0.0, then imaginary part of A is set to zero. */

			res2.odd = (r == c) ? 0.0 : res2.odd;


			A[r + c * lda] = res2;
        }
	}
	else if ( i == (nBlocks - 1)) // Last Row Strip blocks ( May not fit into target region)
	{

		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x - threadID); i >= 0; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifdef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xShared[(%TARGET_ROWS - 1) - threadID] = loadedX;
			yShared[(%TARGET_ROWS - 1) - threadID] = loadedY;
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifndef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xSharedConj[(ref_y - i)] = loadedX;
			ySharedConj[(ref_y - i)] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_WIDTH)  + 1;

		int colShift = threadID / TARGET_ROWS_BY_VEC;
		int rowShift = ((threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V) + (%V - 1);

		int startRow = ref_x - rowShift;
		%TYPE%V  loadedX, loadedY;

		if ( startRow  >= 0 )
		{
			loadedX=  *((__local %TYPE%V*)( &xShared[ (%TARGET_ROWS - 1) - rowShift]));
			loadedY=  *((__local %TYPE%V*)( &yShared[ (%TARGET_ROWS - 1) - rowShift]));
		}

		for( int i= 1; i <=  nLoops; i++)
		{
			int startCol = ref_y - colShift - ( i - 1 ) * TARGET_WIDTH;

			// threads that fall into target region
			if( ( startRow  > -(%V) ) && (startCol > (ref_y - %TARGET_ROWS)) )
			{
				if( startRow  < 0 )// Loop serially as can't do VLOAD
				{
					%TYPE yValueConj = ySharedConj[ ref_y - startCol];
					%TYPE xValueConj = xSharedConj[ ref_y - startCol];

					for(int row = startRow + (%V - 1); row >= 0; row--)
					{
						%TYPE xValue = xShared[ %TARGET_ROWS - 1 - (ref_x - row)];
						%TYPE yValue = yShared[ %TARGET_ROWS - 1 - (ref_x - row)];

						%TYPE res1, res2;

						// X * Y(H)
						%MUL(res1, alpha, yValueConj);
						%MUL( res2, res1,  xValue);

						// Y * X(H)
						%MUL(res1, conjAlpha, xValueConj);
						%MAD( res2, res1,  yValue);
						A[ row + startCol * lda] += res2;
					}
				}
				else
				{
					loadedA  = %VLOAD( 0, (&A[ startRow + startCol * lda]));

					%TYPE 	 loadedXConj = xSharedConj[ ref_y - startCol];
					%TYPE 	 loadedYConj = ySharedConj[ ref_y - startCol];
					%TYPE 	 res;
					// X * Y(H)
					%MUL(res, loadedYConj, alpha);
					%TYPE%V  resVec;
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedX, resVec);

					// Y * X(H)
					%MUL(res, loadedXConj, conjAlpha);
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedY, resVec);
					%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
				}
			}
		}
	}
	else // blocks that fit exactly.
	{
		%TYPE%V loadedA;

		// Populating xShared
		for( int i = (ref_x - threadID); ((ref_x - i) < %TARGET_ROWS); i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifdef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xShared[ (%TARGET_ROWS - 1) - (ref_x - i)] = loadedX;
			yShared[ (%TARGET_ROWS - 1) - (ref_x - i)] = loadedY;
		}

		// Populating yShared
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			#ifndef HER2_ROWMAJOR
				%CONJUGATE(1, loadedX);		//taking conjugate while loading
				%CONJUGATE(1, loadedY);
			#endif
			xSharedConj[(ref_y - i)] = loadedX;
			ySharedConj[(ref_y - i)] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_WIDTH)  + 1;

		int colShift = threadID / TARGET_ROWS_BY_VEC;
		int rowShift = ((threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V) + (%V - 1);


		int startRow = ref_x - rowShift;
		int startCol = ref_y - colShift;
		%TYPE%V  loadedX, loadedY;
		// Not all threads should do this..
		// Depends on whether blocksize width is > target_rows
		if ( startCol > ( ref_y - %TARGET_ROWS) ) // threads that fall into target region
		{
			loadedX=  *((__local %TYPE%V*)( &xShared[ (%TARGET_ROWS - 1)- rowShift]));
			loadedY=  *((__local %TYPE%V*)( &yShared[ (%TARGET_ROWS - 1)- rowShift]));
		}

		for( int i= 1; i <= nLoops; i++)
		{
			startCol = ref_y - colShift - ( i - 1 ) * TARGET_WIDTH;

			if ( startCol > ( ref_y - %TARGET_ROWS) ) // threads that fall into target region
			{
				loadedA  		 	 = %VLOAD( 0, (&A[ startRow + startCol * lda]));
				%TYPE 	 loadedYConj = ySharedConj[ ref_y - startCol];
	 			%TYPE 	 loadedXConj = xSharedConj[ ref_y - startCol];
				%TYPE 	 res;
				// X * Y(H)
				%MUL(res, loadedYConj, alpha);
				%TYPE%V  resVec;
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedX, resVec);

				// Y * X(H)
				%MUL(res, loadedXConj, conjAlpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedY, resVec);
				%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
			}
		}

	}
}
\n";
