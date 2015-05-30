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
static const char *syr2_CL_kernel = "

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
__kernel void %PREFIXsyr2_CL_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

	__global const %TYPE* X;
	__global const %TYPE* Y;
	__global %TYPE* A;

	__local %TYPE xShared[%TARGET_ROWS];
	__local %TYPE yShared[%TARGET_ROWS];
	__local %TYPE xSharedConj[%TARGET_ROWS];
	__local %TYPE ySharedConj[%TARGET_ROWS];

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

            %TYPE res1, res2;
            res1 = alpha * X[c * incx];
			res2 = alpha * X[r * incx];
            res1 = res1 * Y[r * incx];
			res2 = res2 * Y[c * incx];

			A[r + c * lda] += (res1 + res2);
        }
	}
	else if ( i == (nBlocks - 1)) // Last Row Strip blocks ( May not fit into target region)
	{

		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x + threadID); i < N; i += get_local_size(0))
		{
			xShared[i - ref_x] = X[ i * incx];
			yShared[i - ref_x] = Y[ i * incy];
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y  + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
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
						%MUL(res1, alpha, xValueConj);
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
					%MUL(res, loadedXConj, alpha);
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
			xShared[i - ref_x] = X[ i * incx];
			yShared[i - ref_x] = Y[ i * incy];
		}

		// Populating yShared
		for( int i = (ref_y + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
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
				%MUL(res, loadedXConj, alpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedY, resVec);
				%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
			}
		}
	}

}
\n";

//Column Major Upper
static const char *syr2_CU_kernel = "

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
__kernel void %PREFIXsyr2_CU_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

    __global const %TYPE* X;
    __global const %TYPE* Y;
    __global %TYPE* A;

    __local %TYPE xShared[%TARGET_ROWS];
    __local %TYPE yShared[%TARGET_ROWS];
    __local %TYPE xSharedConj[%TARGET_ROWS];
    __local %TYPE ySharedConj[%TARGET_ROWS];

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

            %TYPE res1, res2;
            res1 = alpha * X[c * incx];
			res2 = alpha * X[r * incx];
            res1 = res1 * Y[r * incy];
			res2 = res2 * Y[c * incy];
			A[r + c * lda] += (res1 + res2);
        }
	}
	else if ( i == (nBlocks - 1)) // Last Row Strip blocks ( May not fit into target region)
	{

		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x - threadID); i >= 0; i -= get_local_size(0))
		{
			xShared[(%TARGET_ROWS - 1) - threadID] = X[ i * incx];
			yShared[(%TARGET_ROWS - 1) - threadID] = Y[ i * incy];
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
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
						%MUL(res1, alpha, xValueConj);
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
					%MUL(res, loadedXConj, alpha);
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
			xShared[ (%TARGET_ROWS - 1) - (ref_x - i)] = X[ i * incx];
			yShared[ (%TARGET_ROWS - 1) - (ref_x - i)] = Y[ i * incy];
		}

		// Populating yShared
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
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
				%MUL(res, loadedXConj, alpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedY, resVec);
				%VSTORE(  loadedA, 0, (&A[ startRow + startCol * lda]));
			}
		}

	}
}
";

/*
//Row Major Lower
static const char *syr2_RL_kernel = "

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
//
//nBlocks = 4
//
//Blocks:	9
//	7 8
//	4 5 6
//	0 1 2 3
//

// Row-Major Lower
// nBlocks 	= (N - 1)/ TR + 1
// totalBlocks 	= (nBlocks * ( nBlocks + 1)) / 2
__kernel void %PREFIXsyr2_RL_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

    __global const %TYPE* X;
    __global const %TYPE* Y;
    __global %TYPE* A;

    __local %TYPE xShared[%TARGET_ROWS];
    __local %TYPE yShared[%TARGET_ROWS];
    __local %TYPE xSharedConj[%TARGET_ROWS];
    __local %TYPE ySharedConj[%TARGET_ROWS];

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


	int ref_x = ((nBlocks - 1) * %TARGET_ROWS) - (j * %TARGET_ROWS);
	int ref_y = (i -j) * %TARGET_ROWS;

	// Load data into xShared and yShared
	// Not a common task among blocks in the present implementation..

	// Diagonal Blocks : Should handle not reading diagonal element complex value
	// Diagonal blocks : Should handle the last block as well
	// Scalar code in Present implementation
	if ( ref_x == ref_y )
	{
		// Need only xShared, not using yShared
		for( int i = (ref_x + threadID); (i < N && (i - ref_x) < %TARGET_ROWS); i += get_local_size(0))
		{
			xShared[(i - ref_x)] = X[ i * incx];
			yShared[(i - ref_x)] = Y[ i * incy];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int rowShift = threadID / %TARGET_ROWS;
		int colShift = (threadID & ( %TARGET_ROWS - 1));

		int target_height = %BLOCKSIZE / %TARGET_ROWS;

		int nLoops = ((%TARGET_ROWS - 1)/ target_height)  + 1;
		int startRow = ref_x + rowShift;

		%TYPE yValue, xValue, yValueRead, xValueRead, yValueConj, xValueConj;

		// startCol remains constant on looping
		// Therefore, following code is based on startCol
		int startCol = ref_y + colShift;

		bool readXY		  = false;
		for( int i= 1; i <=  nLoops; i++)
		{
			startRow = ref_x + rowShift + ( i - 1 ) * target_height;

			bool activeThread =  ( startCol < N) && ( startRow < N) && ((startRow -  ref_x) < %TARGET_ROWS);

			if (activeThread)
			{
				// Avoid reading Y again if already read using readXY
				if (( startRow >= startCol) && (!readXY))
				{
					yValueRead = yShared[ startCol - ref_y];
					xValueRead = xShared[ startCol - ref_y];
					readXY 	   = true;
				}

				if ( startRow > startCol )
				{
					%TYPE res1, res2;
					yValueConj = yValueRead;
					xValueConj = xValueRead;

					xValue = xShared[ startRow - ref_x ];
					yValue = yShared[ startRow - ref_x ];

					// X * Y(H)
					%MUL(res1, alpha, yValueConj);
					%MUL( res2, res1,  xValue);

					// Y * X(H)
					%MUL(res1, alpha, xValueConj);
					%MAD( res2, res1, yValue);
					A[ startRow * lda + startCol] += res2;
				}
				else if ( startRow == startCol) // Diagonal
				{
					yValueConj = yValueRead;
					xValueConj = xValueRead;
					yValue 	   = yValueRead;
					xValue     = xValueRead;

					%TYPE res1, res2;
					// X * Y(H)
					%MUL(res1, alpha, yValueConj);
					%MUL( res2, res1,  xValue);

					// Y * X(H)
					%MUL(res1, alpha, xValueConj);
					%MAD( res2, res1, yValue);

					// Discard the imaginary component of A
					%ADD(A[ startRow * lda + startCol], A[ startRow * lda + startCol], res2);
				}
			}
		}
	}
	else if ( ref_x == ((nBlocks - 1) * %TARGET_ROWS)) // Last Row Strip blocks ( May not fit into target region)
	{

		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x + threadID); i < N; i += get_local_size(0))
		{
			xShared[i - ref_x] = X[ i * incx];
			yShared[i - ref_x] = Y[ i * incy];
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y  + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			xSharedConj[(i - ref_y) ] = loadedX;
			ySharedConj[(i - ref_y) ] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_HEIGHT)  + 1;

		int rowShift = threadID / TARGET_ROWS_BY_VEC;
		int colShift = (threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V;


		int startRow = ref_x + rowShift;
		int startCol = ref_y + colShift; // Remains fixed
		%TYPE%V  loadedYConj, loadedXConj;

		if ( startRow  < N )
		{
			loadedYConj =  *((__local %TYPE%V*)( ySharedConj + colShift));
			loadedXConj =  *((__local %TYPE%V*)( xSharedConj + colShift));
		}

		for( int i= 1; i <=  nLoops; i++)
		{
			int startRow = ref_x + rowShift + ( i - 1 ) * TARGET_HEIGHT;

			if (  startRow  < N  )// threads that fall into target region
			{
					loadedA  	= %VLOAD( 0, (&A[ startRow * lda + startCol]));

					%TYPE 	 loadedX= xShared[ startRow - ref_x];
					%TYPE 	 loadedY= yShared[ startRow - ref_x];
					%TYPE 	 res;
					// X * Y(H)
					%MUL(res, loadedX, alpha);
					%TYPE%V  resVec;
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedYConj, resVec);

					// Y * X(H)
					%MUL(res, loadedY, alpha);
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedXConj, resVec);
					%VSTORE(  loadedA, 0, (&A[ startRow * lda + startCol]));
			}
		}
	}
	else // blocks that fit exactly.
	{

		%TYPE%V loadedA;

		// Populating xShared
		for( int i = (ref_x + threadID); (i - ref_x) < %TARGET_ROWS; i += get_local_size(0))
		{
			xShared[i - ref_x] = X[ i * incx];
			yShared[i - ref_x] = Y[ i * incy];
		}

		// Populating yShared
		for( int i = (ref_y + threadID); (i - ref_y) < %TARGET_ROWS; i += get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			xSharedConj[i - ref_y] = loadedX;
			ySharedConj[i - ref_y] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_HEIGHT)  + 1;

		int rowShift = threadID / TARGET_ROWS_BY_VEC;
		int colShift = (threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V;

		int startRow = ref_x + rowShift;
		int startCol = ref_y + colShift;
		%TYPE%V  loadedYConj, loadedXConj;

		if ( startRow < ( ref_x + %TARGET_ROWS) ) // threads that fall into target region
		{
			loadedYConj	 =  *((__local %TYPE%V*)( ySharedConj + colShift));
			loadedXConj	 =  *((__local %TYPE%V*)( xSharedConj + colShift));
		}

		//#pragma unroll
		for( int i= 1; i <= nLoops; i++)
		{
			startRow = ref_x + rowShift + ( i - 1 ) * TARGET_HEIGHT;

			if ( startRow < ( ref_x + %TARGET_ROWS) ) // threads that fall into target region
			{
				loadedA  	= %VLOAD( 0, (&A[ startRow * lda + startCol]));

				%TYPE 	 loadedX = xShared[ startRow - ref_x];
				%TYPE 	 loadedY = yShared[ startRow - ref_x];

				%TYPE 	 res;
				// X * Y(H)
				%MUL(res, loadedX, alpha);
				%TYPE%V  resVec;
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedYConj, resVec);

				// X * Y(H)
				%MUL(res, loadedY, alpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedXConj, resVec);

				%VSTORE(  loadedA, 0, (&A[ startRow * lda + startCol]));
			}
		}
	}
}
\n";

//Row Major Upper
static const char *syr2_RU_kernel = "

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
//
//nBlocks = 4
//
//Blocks:	9
//
//	3 2 1 0
//	  6 5 4
//	    7 8
//		  9
//
//	-----------
//	|<-threadID|
//	| 3 2 1 0  |
//	----------(ref_x, ref_y)
//

// Row-Major Upper
// nBlocks 	= (N - 1)/ TR + 1
// totalBlocks 	= (nBlocks * ( nBlocks + 1)) / 2
__kernel void %PREFIXher2_RU_kernel( __global const %TYPE* _A, __global const %TYPE* _X, __global const %TYPE* _Y, int N, int offx, int incx, int offy, int incy, int offa, int lda, %TYPE alpha)
{

    __global const %TYPE* X;
    __global const %TYPE* Y;
    __global %TYPE* A;

    __local %TYPE xShared[%TARGET_ROWS];
    __local %TYPE yShared[%TARGET_ROWS];
    __local %TYPE xSharedConj[%TARGET_ROWS];
    __local %TYPE ySharedConj[%TARGET_ROWS];

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


	int ref_x = (N - 1) - ((nBlocks - 1 - j) * %TARGET_ROWS);
	int ref_y = (N - 1) - (i -j) * %TARGET_ROWS;

	// Load data into xShared and yShared
	// Not a common task among blocks in the present implementation..

	// Diagonal Blocks : Should handle not reading diagonal element complex value
	// Diagonal blocks : Should handle the last block as well
	// Scalar code in Present implementation
	if ( ref_x == ref_y )
	{
		// Need only xShared, not using yShared
		// Can use ref_x or ref_y
		for( int i =  (ref_x - threadID); (i >= 0 && ((ref_x - i) < %TARGET_ROWS)); i -= get_local_size(0))
		{
			xShared[ (ref_x - i)] = X[ i * incx];
			yShared[ (ref_x - i)] = Y[ i * incy];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int rowShift = threadID / %TARGET_ROWS;
		int colShift = (threadID & ( %TARGET_ROWS - 1));

		int target_height = %BLOCKSIZE / %TARGET_ROWS;

		int nLoops = ((%TARGET_ROWS - 1)/ target_height)  + 1;
		int startRow = ref_x - rowShift;

		%TYPE yValue, xValue, yValueRead, xValueRead, xValueConj, yValueConj;

		int startCol = ref_y - colShift; // remains fixed on looping

		bool readXY = false;
		for( int i= 1; i <=  nLoops; i++)
		{
			startRow = ref_x - rowShift - ( i - 1 ) * target_height;
			bool activeThread = ( startRow >= 0) && ( startCol >= 0) && (( ref_x - startRow) < %TARGET_ROWS);

			if ( activeThread)
			{
				// Avoid reading yValue again for threads that have it already while looping
				if (( startCol >= startRow ) && (!readXY))
				{
					xValueRead = xShared[ ref_y - startCol];
					yValueRead = yShared[ ref_y - startCol];
					readXY = true;
				}

				if ( startCol > startRow )
				{
					%TYPE res1, res2;
					yValueConj = yValueRead;
					xValueConj = xValueRead;

					xValue = xShared[ ref_x - startRow];
					yValue = yShared[ ref_x - startRow];

					// X * Y(H)
					%MUL(res1, alpha, yValueConj);
					%MUL(res2, res1, xValue);

					// X * Y(H)
					%MUL(res1, alpha, xValueConj);
					%MAD( res2, res1,  yValue);

					A[ startRow * lda + startCol] += res2;
				}
				else if (startRow == startCol) // Diagonal
				{
					// The y Values can be obtained from  xValues
					xValue     = xValueRead;
					yValue 	   = yValueRead;
					xValueConj = xValueRead;
					yValueConj = yValueRead;

					%TYPE res1, res2;
					// X * Y(H)
					%MUL(res1, alpha, yValueConj);
					%MUL(res2, res1,  xValue);
					// Y * X(H)
					%MUL(res1, alpha, xValueConj);
					%MAD(res2, res1,  yValue);

					// Discard the imaginary component of A
					%ADD(A[ startRow * lda + startCol], A[ startRow * lda + startCol], res2);
				}
			}
		}
	}
	else if ( ref_x == (( N - 1) - ((nBlocks - 1) * %TARGET_ROWS))) // First Row Strip blocks ( May not fit into target region)
	{
		%TYPE%V loadedA;

		// Populating xShared: May not fit into target region
		for( int i = (ref_x - threadID); i >= 0; i -= get_local_size(0))
		{
			xShared[ref_x - i] = X[ i * incx];
			yShared[ref_x - i] = Y[ i * incy];
		}

		// Populating yShared: Always fits well..
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			xSharedConj[(%TARGET_ROWS - 1) - (ref_y - i)] = loadedX;
			ySharedConj[(%TARGET_ROWS - 1) - (ref_y - i)] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_HEIGHT)  + 1;

		int rowShift = threadID / TARGET_ROWS_BY_VEC;
		int colShift = ((threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V) + (%V - 1);

		int startRow = ref_x - rowShift;
		int startCol = ref_y - colShift;

		%TYPE%V  loadedYConj, loadedXConj;

		if ( startRow  >= 0 )
		{
			loadedYConj =  *((__local %TYPE%V*)( &ySharedConj[ (%TARGET_ROWS - 1) - colShift]));
			loadedXConj =  *((__local %TYPE%V*)( &xSharedConj[ (%TARGET_ROWS - 1) - colShift]));
		}

		for( int i= 1; i <=  nLoops; i++)
		{
			int startRow = ref_x - rowShift - ( i - 1 ) * TARGET_HEIGHT;

			if ( startRow  >= 0 )
			{
					loadedA  = %VLOAD( 0, (&A[ startRow * lda + startCol]));

					%TYPE 	 loadedX = xShared[ ref_x - startRow];
					%TYPE 	 loadedY = yShared[ ref_x - startRow];
					%TYPE 	 res;
					// X * Y(H)
					%MUL(res, loadedX, alpha);
					%TYPE%V  resVec;
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedYConj, resVec);

					// Y * X(H)
					%MUL(res, loadedY, alpha);
					resVec = %VMAKEVEC(res);
					%VMAD( loadedA, loadedXConj, resVec);

					%VSTORE(  loadedA, 0, (&A[ startRow * lda + startCol]));
			}
		}
	}
	else // blocks that fit exactly.
	{
		%TYPE%V loadedA;

		// Populating xShared
		for( int i = (ref_x - threadID); ((ref_x - i) < %TARGET_ROWS); i -= get_local_size(0))
		{
			xShared[(ref_x - i)] = X[ i * incx];
			yShared[(ref_x - i)] = Y[ i * incy];
		}

		// Populating yShared
		for( int i = (ref_y - threadID); (ref_y - i) < %TARGET_ROWS; i -= get_local_size(0))
		{
			%TYPE loadedX = X[ i * incx];
			%TYPE loadedY = Y[ i * incy];
			xSharedConj[(%TARGET_ROWS - 1) - (ref_y - i)] = loadedX;
			ySharedConj[(%TARGET_ROWS - 1) - (ref_y - i)] = loadedY;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop %TARGET_ROWS / TARGET_WIDTH times
		int nLoops = ((%TARGET_ROWS - 1)/ TARGET_HEIGHT)  + 1;

		int rowShift = threadID / TARGET_ROWS_BY_VEC;
		int colShift = ((threadID & ( TARGET_ROWS_BY_VEC - 1)) * %V) + (%V - 1);


		int startRow = ref_x - rowShift;
		int startCol = ref_y - colShift;

		%TYPE%V  loadedYConj, loadedXConj;
		// Not all threads should do this..
		// Depends on whether blocksize width is > target_rows
		if ( startRow > ( ref_x - %TARGET_ROWS) ) // threads that fall into target region
		{
			loadedYConj =  *((__local %TYPE%V*)( &ySharedConj[ (%TARGET_ROWS - 1) - colShift]));
			loadedXConj =  *((__local %TYPE%V*)( &xSharedConj[ (%TARGET_ROWS - 1) - colShift]));
		}

		for( int i= 1; i <= nLoops; i++)
		{
			startRow = ref_x - rowShift - ( i - 1 ) * TARGET_HEIGHT;

			if ( startRow > ( ref_x - %TARGET_ROWS) ) // threads that fall into target region
			{
				loadedA  = %VLOAD( 0, (&A[ startRow * lda + startCol]));

				%TYPE 	 loadedX = xShared[ ref_x - startRow];
				%TYPE 	 loadedY = yShared[ ref_x - startRow];
				%TYPE 	 res;
				// X * Y(H)
				%MUL(res, loadedX, alpha);
				%TYPE%V  resVec;
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedYConj, resVec);

				// Y * X(H)
				%MUL(res, loadedY, alpha);
				resVec = %VMAKEVEC(res);
				%VMAD( loadedA, loadedXConj, resVec);

				%VSTORE(  loadedA, 0, (&A[ startRow * lda + startCol]));

			}
		}

	}
}
\n";
*/
