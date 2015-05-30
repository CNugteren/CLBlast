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

// Compute Rectangle + Traingle

const char * trsv_CU_ComputeRectangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#define TARGET_ROWS_BY_VEC ((%TARGET_ROWS)/(%V))
#ifdef PACKED
    #define A( row, col) (*( A + (((col)*((col)+1))/2 + (row))))
#else
    #define A( row, col) A[ (row) + (col) * lda]
#endif

__kernel void %PREFIXtrsv_CU_ComputeRectangle_kernel( __global %TYPE const * restrict _A, __global %TYPE* _xnew, uint N, int incx, int isUnity, uint lda, int doConj, int startCol, int rowsLeft, uint offa, uint offx)
{
	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	size_t bIdx 	= get_group_id(0);
	size_t threadIdx= get_local_id(0);

	// Get total blocks launched
	size_t nBlocks  = ((rowsLeft - 1) / %TARGET_ROWS) + 1;

	%TYPE sum 	= %MAKEVEC( 0.0);
	%TYPE loadedA 	= %MAKEVEC( 0.0);

	// First Block does scalar stuff...
	// Only this gets executed if nBlocks == 1
	if ( bIdx == 0)
	{
		int targetCol 	= startCol;
		int targetRow 	= threadIdx;
		int lastRow	= rowsLeft - ( nBlocks - 1) * %TARGET_ROWS - 1;

		if ( nBlocks > 1)
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol--;
				}

				%SUB(xnew[ targetRow * incx], xnew[targetRow * incx], sum);
			}
		}
		else // Solve the traingle -- no more kernel launches required
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol--;
				}
			}

			// Change targetCol to point to Triangle last column for all threads
			// As the above condition ( targetRow <= lastRow) changes targetCol for only threads with condition true
			targetCol 	= startCol - %TARGET_ROWS;

			__local %TYPE  xShared; // To share solved x value with other threads..

			for( int i=0; i < (lastRow + 1); i++)
			{
				if ( targetRow == targetCol)
				{
					%TYPE xVal = xnew[ targetRow * incx];
					%SUB(sum, xVal, sum);
					xShared = sum;
					xnew[ targetRow * incx ] = xShared;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (  targetRow < targetCol)
				{
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xShared);
				}

				// Avoid Race
				barrier(CLK_LOCAL_MEM_FENCE);
				targetCol--;
			}
		}
	}
	else
	{
		size_t rowShift = ((threadIdx % ( TARGET_ROWS_BY_VEC  )) * %V);
		size_t colShift = threadIdx / TARGET_ROWS_BY_VEC;

		int rowStart 	= rowsLeft  - ( %TARGET_ROWS * (nBlocks - bIdx) );
		int row		= rowStart + rowShift;

		%TYPE   sumTemp = %MAKEVEC(0.0);
		%TYPE%V sum	= %VMAKEVEC(sumTemp);

		__local %TYPE xData[ %TARGET_WIDTH];

		//#pragma unroll
		for( int i=1; i <= %NLOOPS; i++)
		{
			// Put startCol to start of BLOCKSIZE Block
			int startColp	= startCol - (%TARGET_WIDTH * i) + 1;

			if ( threadIdx < %TARGET_WIDTH)
			{
				xData[threadIdx] = xnew[ (startColp + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int col 	= startColp + colShift;

			%TYPE xDataVal	= xData[ colShift ];
			%TYPE%V xDataVec= %VMAKEVEC( xDataVal);

			%TYPE%V loadedA  = %VLOAD( 0, &A((row), (col)));
			%CONJUGATE(doConj, loadedA);
			%VMAD(sum, loadedA, xDataVec);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		__local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * %TARGET_WIDTH];
		//__local %TYPE* sData = sDataTemp;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		//int TARGET_ROWS		= %TARGET_ROWS;

		// Last Block
		// Do Scalar reduction for last block
		// Followed by solving the triangle
		if ( bIdx == ( nBlocks - 1))
		{
			%TYPE sumTemp 	    = %MAKEVEC(0.0);
			%TYPE%V sumVec      = %VMAKEVEC(sumTemp);
		        %TYPE%V loadedAVec  = %VMAKEVEC(sumTemp);

			//int targetRow = rowStart + threadIdx;
			int targetCol = startCol- %TARGET_ROWS; // Col where triangle last col overlaps

			// Do vector reduction
			if ( threadIdx <  TARGET_ROWS_BY_VEC )
			{
				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(sumVec, sumVec, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}
			}

			__local %TYPE xShared[%V];

			int targetRowTemp = rowStart + threadIdx * %V;
			int VECTOR_SIZE   = %V;

			//#pragma unroll
			for( int i=0; i < (TARGET_ROWS_BY_VEC); i++)
			{
				if ( threadIdx == (TARGET_ROWS_BY_VEC - 1 - i))
			    {
					// Read X-vector
					%TYPE xVal[%V];
					//#pragma unroll
					for( int j = 0; j < %V; j++)
					{
						xVal[j] = xnew[ (targetRowTemp + j)* incx];
					}

					// Read A %Vx%V region into reg
					%TYPE reg[%V][%V];
					//#pragma unroll
					for( int idx = 0; idx < ( %V * %V); idx++)
					{
						int m = idx / ( %V ); // Row : Col-Major idx...
						int n = idx % ( %V );    // Col
						if ( n > m )
						{
							reg[m][n] = A( (targetRowTemp + m), (targetCol -( %V - 1 - n)));
							%CONJUGATE(doConj, reg[m][n]);
						}
					}

					%TYPE sumVecReg[%V];
					%VSTOREWITHINCX(sumVecReg, sumVec, 1);

					// Solve for first x - Do the rest in loop
					%TYPE x[%V];
					%SUB(x[VECTOR_SIZE - 1], xVal[VECTOR_SIZE - 1], sumVecReg[VECTOR_SIZE - 1]);
					xShared[%V - 1] = x[%V - 1];
					xnew[ (targetRowTemp + %V - 1)* incx ] = x[%V - 1];

					//#pragma unroll
					for(int m= ( %V - 2); m >=0; m--)
					{
						%SUB(x[m], xVal[m], sumVecReg[m]);
					}

					//#pragma unroll
					for( int idx = (( ( %V * %V) - 1) - %V); idx > 0; idx--)
					{
						int m = idx / %V;       // Row : Row-Major idx, x[3] is solved before x[2]
						int n = idx % ( %V );// Col
						if ( n > m)
						{
							//x[m] = x[m] - reg[m][n] * x[n];
							%MAD(x[m], reg[m][n], (-x[n]));
						}
					}

					// Store results
					//#pragma unroll
					for(int m = 0; m < %V; m++)
					{
						xShared[m] = x[m];
						xnew[ (targetRowTemp + m)* incx ] = x[m];
					}
			    }


			    // Sync so that xShared it available to all threads
			    barrier(CLK_LOCAL_MEM_FENCE);

			    if ( threadIdx < (TARGET_ROWS_BY_VEC - 1 - i))
				{
						//#pragma unroll
						for( int j=0; j < %V; j++)
						{
							//sumVec += vload4( 0, &A((targetRowTemp), (targetCol -j))) * xShared[%V - 1 -j];
							%TYPE%V loadedAVec  = %VLOAD( 0, &A((targetRowTemp), (targetCol -j)));
							%CONJUGATE(doConj, loadedAVec);
							%VMAD(sumVec, loadedAVec, xShared[VECTOR_SIZE - 1 -j]);
						}
				}

				targetCol = targetCol - %V;
			        // Avoid Race...
			    barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		else
		{
			// Do Vector Reduction on each block except the last Block
			if ( threadIdx < TARGET_ROWS_BY_VEC)
			{
				%TYPE   accTemp = %MAKEVEC(0.0);
				%TYPE%V acc 	= %VMAKEVEC(accTemp);

				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(acc, acc, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}

				// Store the result
				int targetRow = rowStart + threadIdx * %V;

				__global %TYPE* xNewPtr =  xnew + targetRow * incx;
				//float4 value = (float4)( xNewPtr[0], xNewPtr[incx], xNewPtr[incx * 2], xNewPtr[incx *3]);
				%TYPE%V value;
				%VLOADWITHINCX(value, xNewPtr, incx);

				// Compute result
				%SUB(value, value, acc);

				// Store results
				//VSTOREWITHINCX( xNewPtr, value, incx);
				%VSTOREWITHINCX(xNewPtr, value, incx);
			}
		}
	}
}
";

const char *trsv_CU_ComputeRectangle_NonUnity_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#define TARGET_ROWS_BY_VEC ((%TARGET_ROWS)/(%V))
#ifdef PACKED
    #define A( row, col) (*( A + (((col)*((col)+1))/2 + (row))))
#else
    #define A( row, col) A[ (row) + (col) * lda]
#endif
// Compute Rectangle + Traingle
__kernel void %PREFIXtrsv_CU_ComputeRectangle_NonUnity_kernel( __global %TYPE const * restrict _A, __global %TYPE* _xnew, uint N, int incx, int isUnity, uint lda, int doConj, int startCol, int rowsLeft, uint offa, uint offx)
{
	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	size_t bIdx 	= get_group_id(0);
	size_t threadIdx= get_local_id(0);

	// Get total blocks launched
	size_t nBlocks  = (rowsLeft - 1) / %TARGET_ROWS + 1;

	%TYPE sum 	= %MAKEVEC( 0.0);
	%TYPE loadedA 	= %MAKEVEC( 0.0);

	// First Block does scalar stuff...
	// Only this gets executed if nBlocks == 1
	if ( bIdx == 0)
	{
		int targetCol 	= startCol;
		int targetRow 	= threadIdx;
		int lastRow	= rowsLeft - ( nBlocks - 1) * %TARGET_ROWS - 1;

		if ( nBlocks > 1)
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol--;
				}

				%SUB(xnew[ targetRow * incx], xnew[targetRow * incx], sum);
			}
		}
		else // Solve the traingle -- no more kernel launches required
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol--;
				}
			}

			// Change targetCol to point to Triangle last column for all threads
			// As the above condition ( targetRow <= lastRow) changes targetCol for only threads with condition true
			targetCol 	= startCol - %TARGET_ROWS;

			__local %TYPE  xShared; // To share solved x value with other threads..

			for( int i=0; i < (lastRow + 1); i++)
			{
				if ( targetRow == targetCol)
				{
					%TYPE xVal = xnew[ targetRow * incx];
					sum  =  xVal -  sum;

					// Handle diagonal element
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%DIV(xShared, sum, loadedA);

					xnew[ targetRow * incx ] = xShared;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (  targetRow < targetCol)
				{
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xShared);
				}

				// Avoid Race
				barrier(CLK_LOCAL_MEM_FENCE);
				targetCol--;
			}
		}
	}
	else
	{
		size_t rowShift = ((threadIdx % ( TARGET_ROWS_BY_VEC )) * %V);
		size_t colShift = threadIdx / TARGET_ROWS_BY_VEC;

		int rowStart 	= rowsLeft  - ( %TARGET_ROWS * (nBlocks - bIdx) );
		int row		= rowStart + rowShift;

		%TYPE   sumTemp = %MAKEVEC(0.0);
		%TYPE%V sum	= %VMAKEVEC(sumTemp);

		__local %TYPE xData[ %TARGET_WIDTH];

		//#pragma unroll
		for( int i=1; i <= %NLOOPS; i++)
		{
			// Put startCol to start of BLOCKSIZE Block
			int startColp	= startCol - (%TARGET_WIDTH * i) + 1;

			if ( threadIdx < %TARGET_WIDTH)
			{
				xData[threadIdx] = xnew[ (startColp + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int col 	= startColp + colShift;

			%TYPE xDataVal	= xData[ colShift ];
			%TYPE%V xDataVec= %VMAKEVEC( xDataVal);

			%TYPE%V loadedA  = %VLOAD( 0, &A((row), (col)));
			%CONJUGATE(doConj, loadedA);
			%VMAD(sum, loadedA, xDataVec);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		__local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * %TARGET_WIDTH];
		//__local %TYPE* sData = sDataTemp;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		//int TARGET_ROWS		= %TARGET_ROWS;

		// Last Block
		// Do Scalar reduction for last block
		// Followed by solving the triangle
		if ( bIdx == ( nBlocks - 1))
		{
			%TYPE sumTemp 	    = %MAKEVEC(0.0);
			%TYPE%V sumVec      = %VMAKEVEC(sumTemp);
		        %TYPE%V loadedAVec  = %VMAKEVEC(sumTemp);

			//int targetRow = rowStart + threadIdx;
			int targetCol = startCol- %TARGET_ROWS; // Col where triangle last col overlaps

			// Do vector reduction
			if ( threadIdx <  TARGET_ROWS_BY_VEC )
			{
				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(sumVec, sumVec, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}
			}

			__local %TYPE xShared[%V];

			int targetRowTemp = rowStart + threadIdx * %V;
			int VECTOR_SIZE   = %V;

			//#pragma unroll
			for( int i=0; i < (TARGET_ROWS_BY_VEC); i++)
			{
				if ( threadIdx == (TARGET_ROWS_BY_VEC - 1 - i))
			        {
					// Read X-vector
					%TYPE xVal[%V];
					//#pragma unroll
					for( int j = 0; j < %V; j++)
					{
						xVal[j] = xnew[ (targetRowTemp + j)* incx];
					}

					// Read A %Vx%V region into reg
					%TYPE reg[%V][%V];
					//#pragma unroll
					for( int idx = 0; idx < ( %V * %V); idx++)
					{
						int m = idx % ( %V ); // Row : Col-Major idx...
						int n = idx / ( %V );    // Col
						if ( n >= m )
						{
							reg[m][n] = A((targetRowTemp + m), (targetCol -( %V - 1 - n)));
							%CONJUGATE(doConj, reg[m][n]);
						}
					}

					%TYPE sumVecReg[%V];
					%VSTOREWITHINCX(sumVecReg, sumVec, 1);

					// Solve for first x - Do the rest in loop
					%TYPE x[%V];
					%SUB(x[VECTOR_SIZE - 1], xVal[VECTOR_SIZE - 1], sumVecReg[VECTOR_SIZE - 1]);
					%DIV(sumVecReg[VECTOR_SIZE - 1], x[VECTOR_SIZE -1], reg[VECTOR_SIZE - 1][VECTOR_SIZE - 1]);
					x[VECTOR_SIZE -1] = sumVecReg[VECTOR_SIZE - 1];
					xShared[%V - 1] = x[%V - 1];
					xnew[ (targetRowTemp + %V - 1)* incx ] = x[%V - 1];

					//#pragma unroll
					for(int m = ( %V - 2); m >=0; m--)
					{
						%SUB(x[m], xVal[m], sumVecReg[m]);
					}

					//#pragma unroll
					for( int idx = (( ( %V * %V) - 1) - %V); idx >= 0; idx--)
					{
						int m = idx / %V;       // Row : Row-Major idx, x[3] is solved before x[2]
						int n = idx % ( %V );// Col
						if ( n > m)
						{
							//x[m] = x[m] - reg[m][n] * x[n];
							%MAD(x[m], reg[m][n], (-x[n]));
						}
						else if ( m == n)
						{
							%DIV(sumVecReg[m], x[m], reg[m][m]);
							x[m] = sumVecReg[m];
						}
					}

					// Store results
					//#pragma unroll
					for(int m = 0; m < %V; m++)
					{
						xShared[m] = x[m];
						xnew[ (targetRowTemp + m)* incx ] = x[m];
					}
			        }


			        // Sync so that xShared it available to all threads
			        barrier(CLK_LOCAL_MEM_FENCE);

			      	if ( threadIdx < (TARGET_ROWS_BY_VEC - 1 - i))
				{
						//#pragma unroll
						for( int j=0; j < %V; j++)
						{
							//sumVec += vload4( 0, &A((targetRowTemp), (targetCol -j))) * xShared[%V - 1 -j];
							%TYPE%V loadedAVec  = %VLOAD( 0, &A((targetRowTemp), (targetCol -j)));
							%CONJUGATE(doConj, loadedAVec);
							%VMAD(sumVec, loadedAVec, xShared[VECTOR_SIZE - 1 -j]);
						}
				}

				targetCol = targetCol - %V;
			        // Avoid Race...
			        barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		else
		{
			// Do Vector Reduction on each block except the last Block
			if ( threadIdx < TARGET_ROWS_BY_VEC)
			{
				%TYPE   accTemp = %MAKEVEC(0.0);
				%TYPE%V acc 	= %VMAKEVEC(accTemp);

				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(acc, acc, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}

				// Store the result
				int targetRow = rowStart + threadIdx * %V;

				__global %TYPE* xNewPtr =  xnew + targetRow * incx;
				//float4 value = (float4)( xNewPtr[0], xNewPtr[incx], xNewPtr[incx * 2], xNewPtr[incx *3]);
				%TYPE%V value;
				%VLOADWITHINCX(value, xNewPtr, incx);

				// Compute result
				%SUB(value, value, acc);

				// Store results
				//VSTOREWITHINCX( xNewPtr, value, incx);
				%VSTOREWITHINCX(xNewPtr, value, incx);
			}
		}
	}

}
";


const char *trsv_CL_ComputeRectangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#define TARGET_ROWS_BY_VEC ((%TARGET_ROWS)/(%V))
#ifdef PACKED
    #define A(row, col) (*( A + ((( (col) *((2*N) + 1 - (col))) / 2) + ((row) - (col)))))
#else
    #define A(row, col) A[ (row) + (col) * lda]
#endif
// Compute Rectangle + Traingle
__kernel void %PREFIXtrsv_CL_ComputeRectangle_kernel( __global const %TYPE* _A, __global %TYPE* _xnew, uint N, int incx, int isUnity, uint lda, int doConj, int startCol, int rowsLeft, uint offa, uint offx)
{
	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	size_t bIdx 	= get_group_id(0);
	size_t threadIdx= get_local_id(0);

	// Get total blocks launched
	size_t nBlocks  = (rowsLeft - 1) / %TARGET_ROWS + 1;

	%TYPE sum 	= %MAKEVEC( 0.0);
	%TYPE loadedA 	= %MAKEVEC( 0.0);

	// Last Block does scalar stuff...
	// Only this gets executed if nBlocks == 1
	if ( bIdx == (nBlocks - 1))
	{
		int targetCol 	= startCol;
		int startRow 	= (N - rowsLeft) + ( bIdx) * %TARGET_ROWS;
		int targetRow 	= startRow  +  threadIdx;
		int lastRow	= startRow + rowsLeft - ( nBlocks - 1) * %TARGET_ROWS - 1;

		if ( nBlocks > 1)
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol++;
				}

				%SUB(xnew[ targetRow * incx], xnew[targetRow * incx], sum);
			}
		}
		else // Solve the traingle -- no more kernel launches required
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol++;
				}
			}

			// Change targetCol to point to Triangle last column for all threads
			// As the above condition ( targetRow <= lastRow) changes targetCol for only threads with condition true
			targetCol 	= startCol + %TARGET_ROWS;

			__local %TYPE  xShared; // To share solved x value with other threads..

			for( int i=0; i < ((lastRow -startRow) + 1); i++)
			{
				if ( targetRow == targetCol)
				{
					%TYPE xVal = xnew[ targetRow * incx];
					sum  =  xVal -  sum;

					if( isUnity)
					{
						xShared = sum;
					}
					else // Handle diagonal element
					{
						loadedA = A((targetRow), (targetCol));
						%CONJUGATE(doConj, loadedA);
						%DIV(xShared, sum, loadedA);
					}

					xnew[ targetRow * incx ] = xShared;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (  targetRow <= lastRow)
				{
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xShared);
				}

				// Avoid Race
				barrier(CLK_LOCAL_MEM_FENCE);
				targetCol++;
			}
		}
	}
	else
	{
		size_t rowShift = ((threadIdx % ( TARGET_ROWS_BY_VEC )) * %V);
		size_t colShift = threadIdx / TARGET_ROWS_BY_VEC;

		int rowStart 	= (N - rowsLeft) + ( bIdx) * %TARGET_ROWS;
		int row		= rowStart + rowShift;

		%TYPE   sumTemp = %MAKEVEC(0.0);
		%TYPE%V sum	= %VMAKEVEC(sumTemp);

		__local %TYPE xData[ %TARGET_WIDTH];

		//#pragma unroll
		for( int i=1; i <= %NLOOPS; i++)
		{
			// Put startCol to start of BLOCKSIZE Block
			int startColp	= startCol + (%TARGET_WIDTH * (i - 1));

			if ( threadIdx < %TARGET_WIDTH)
			{
				xData[threadIdx] = xnew[ (startColp + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int col 	= startColp + colShift;

			%TYPE xDataVal	= xData[ colShift ];
			%TYPE%V xDataVec= %VMAKEVEC( xDataVal);

			%TYPE%V loadedA  = %VLOAD( 0, &A((row), (col)));
			%CONJUGATE(doConj, loadedA);
			%VMAD(sum, loadedA, xDataVec);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		__local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * %TARGET_WIDTH];
		//__local %TYPE* sData = sDataTemp;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		//int TARGET_ROWS		= %TARGET_ROWS;

		// Last Block
		// Do Scalar reduction for last block
		// Followed by solving the triangle
		if ( bIdx == 0 )
		{
			%TYPE sumTemp 	    = %MAKEVEC(0.0);
			%TYPE%V sumVec      = %VMAKEVEC(sumTemp);
		        %TYPE%V loadedAVec  = %VMAKEVEC(sumTemp);

			//int targetRow = rowStart + threadIdx;
			int targetCol = startCol + %TARGET_ROWS; // Col where triangle last col overlaps

			// Do vector reduction
			if ( threadIdx <  TARGET_ROWS_BY_VEC )
			{
				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(sumVec, sumVec, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}
			}

			__local %TYPE xShared[%V];

			int targetRowTemp = rowStart + threadIdx * %V;
			int VECTOR_SIZE   = %V;

			//#pragma unroll
			for( int i=0; i < (TARGET_ROWS_BY_VEC); i++)
			{
				if ( threadIdx == i )
			        {
					// Read X-vector
					%TYPE xVal[%V];
					//#pragma unroll
					for( int j = 0; j < %V; j++)
					{
						xVal[j] = xnew[ (targetRowTemp + j)* incx];
					}

					// Read A %Vx%V region into reg
					%TYPE reg[%V][%V];
					//#pragma unroll
					for( int idx = 0; idx < ( %V * %V); idx++)
					{
						int m = idx % ( %V ); // Row : Col-Major idx...
						int n = idx / ( %V );    // Col
						if ( m > n )
						{
							reg[m][n] = A((targetRowTemp + m), (targetCol + n));
							%CONJUGATE(doConj, reg[m][n]);
						}
					}

					%TYPE sumVecReg[%V];
					%VSTOREWITHINCX(sumVecReg, sumVec, 1);

					// Solve for first x - Do the rest in loop
					%TYPE x[%V];
					%SUB(x[0], xVal[0], sumVecReg[0]);
					xShared[0] = x[0];
					xnew[ (targetRowTemp)* incx ] = x[0];

					//#pragma unroll
					for(int m = 1; m < %V; m++)
					{
						%SUB(x[m], xVal[m], sumVecReg[m]);
					}

					//#pragma unroll
					for( int idx =  %V; idx < (( %V * %V) - 1); idx++)
					{
						int m = idx / %V;       // Row : Row-Major idx, x[1] is solved before x[2]
						int n = idx % ( %V );// Col
						if ( m > n)
						{
							//x[m] = x[m] - reg[m][n] * x[n];
							%MAD(x[m], reg[m][n], (-x[n]));
						}
					}

					// Store results
					//#pragma unroll
					for(int m = 0; m < %V; m++)
					{
						xShared[m] = x[m];
						xnew[ (targetRowTemp + m)* incx ] = x[m];
					}
			        }


			        // Sync so that xShared it available to all threads
			        barrier(CLK_LOCAL_MEM_FENCE);
			      	if ( (threadIdx > i) && ( threadIdx < (TARGET_ROWS_BY_VEC)) )
				{
						//#pragma unroll
						for( int j=0; j < %V; j++)
						{
							%TYPE%V loadedAVec  = %VLOAD( 0, &A((targetRowTemp), (targetCol +j)));
							%CONJUGATE(doConj, loadedAVec);
							%VMAD(sumVec, loadedAVec, xShared[j]);
						}
				}

				targetCol = targetCol + %V;
			        // Avoid Race...
			        barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		else
		{
			// Do Vector Reduction on each block except the last Block
			if ( threadIdx < TARGET_ROWS_BY_VEC)
			{
				%TYPE   accTemp = %MAKEVEC(0.0);
				%TYPE%V acc 	= %VMAKEVEC(accTemp);

				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(acc, acc, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}

				// Store the result
				int targetRow = rowStart + threadIdx * %V;

				__global %TYPE* xNewPtr =  xnew + targetRow * incx;
				//float4 value = (float4)( xNewPtr[0], xNewPtr[incx], xNewPtr[incx * 2], xNewPtr[incx *3]);
				%TYPE%V value;
				%VLOADWITHINCX(value, xNewPtr, incx);

				// Compute result
				%SUB(value, value, acc);

				// Store results
				%VSTOREWITHINCX(xNewPtr, value, incx);
			}
		}
	}
}
";

const char *trsv_CL_ComputeRectangle_NonUnity_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#define TARGET_ROWS_BY_VEC ((%TARGET_ROWS)/(%V))
#ifdef PACKED
    #define A(row, col) (*( A + ((( (col) *((2*N) + 1 - (col))) / 2) + ((row) - (col)))))
#else
    #define A(row, col) A[ (row) + (col) * lda]
#endif
// Compute Rectangle + Traingle
__kernel void %PREFIXtrsv_CL_ComputeRectangle_NonUnity_kernel( __global const %TYPE* _A, __global %TYPE* _xnew, uint N, int incx, int isUnity, uint lda, int doConj, int startCol, int rowsLeft, uint offa, uint offx)
{
	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	size_t bIdx 	= get_group_id(0);
	size_t threadIdx= get_local_id(0);

	// Get total blocks launched
	size_t nBlocks  = (rowsLeft - 1) / %TARGET_ROWS + 1;

	%TYPE sum 	= %MAKEVEC( 0.0);
	%TYPE loadedA 	= %MAKEVEC( 0.0);

	// Last Block does scalar stuff...
	// Only this gets executed if nBlocks == 1
	if ( bIdx == (nBlocks - 1))
	{
		int targetCol 	= startCol;
		int startRow 	= (N - rowsLeft) + ( bIdx) * %TARGET_ROWS;
		int targetRow 	= startRow  +  threadIdx;
		int lastRow	= startRow + rowsLeft - ( nBlocks - 1) * %TARGET_ROWS - 1;

		if ( nBlocks > 1)
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol++;
				}

				%SUB(xnew[ targetRow * incx], xnew[targetRow * incx], sum);
			}
		}
		else // Solve the traingle -- no more kernel launches required
		{
			if ( targetRow <= lastRow)
			{
				for( int i=0; i < %TARGET_ROWS; i++)
				{
					// All threads look at same xnew
					// Should use Shared Memory ..
					%TYPE xVal =  xnew[ targetCol * incx];
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xVal);
					targetCol++;
				}
			}

			// Change targetCol to point to Triangle last column for all threads
			// As the above condition ( targetRow <= lastRow) changes targetCol for only threads with condition true
			targetCol 	= startCol + %TARGET_ROWS;

			__local %TYPE  xShared; // To share solved x value with other threads..

			for( int i=0; i < ((lastRow -startRow) + 1); i++)
			{
				if ( targetRow == targetCol)
				{
					%TYPE xVal = xnew[ targetRow * incx];
					sum  =  xVal -  sum;

					// Handle diagonal element
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%DIV(xShared, sum, loadedA);
					xnew[ targetRow * incx ] = xShared;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (  targetRow <= lastRow)
				{
					loadedA = A((targetRow), (targetCol));
					%CONJUGATE(doConj, loadedA);
					%MAD(sum, loadedA, xShared);
				}

				// Avoid Race
				barrier(CLK_LOCAL_MEM_FENCE);
				targetCol++;
			}
		}
	}
	else
	{
		size_t rowShift = ((threadIdx % ( TARGET_ROWS_BY_VEC )) * %V);
		size_t colShift = threadIdx / TARGET_ROWS_BY_VEC;

		int rowStart 	= (N - rowsLeft) + ( bIdx) * %TARGET_ROWS;
		int row		= rowStart + rowShift;

		%TYPE   sumTemp = %MAKEVEC(0.0);
		%TYPE%V sum	= %VMAKEVEC(sumTemp);

		__local %TYPE xData[ %TARGET_WIDTH];

		//#pragma unroll
		for( int i=1; i <= %NLOOPS; i++)
		{
			// Put startCol to start of BLOCKSIZE Block
			int startColp	= startCol + (%TARGET_WIDTH * (i - 1));

			if ( threadIdx < %TARGET_WIDTH)
			{
				xData[threadIdx] = xnew[ (startColp + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int col 	= startColp + colShift;

			%TYPE xDataVal	= xData[ colShift ];
			%TYPE%V xDataVec= %VMAKEVEC( xDataVal);

			%TYPE%V loadedA  = %VLOAD( 0, &A((row), (col)));
			%CONJUGATE(doConj, loadedA);
			%VMAD(sum, loadedA, xDataVec);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		__local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * %TARGET_WIDTH];
		//__local %TYPE* sData = sDataTemp;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		//int TARGET_ROWS		= %TARGET_ROWS;

		// Last Block
		// Do Scalar reduction for last block
		// Followed by solving the triangle
		if ( bIdx == 0 )
		{
			%TYPE sumTemp 	    = %MAKEVEC(0.0);
			%TYPE%V sumVec      = %VMAKEVEC(sumTemp);
		        %TYPE%V loadedAVec  = %VMAKEVEC(sumTemp);

			//int targetRow = rowStart + threadIdx;
			int targetCol = startCol + %TARGET_ROWS; // Col where triangle last col overlaps

			// Do vector reduction
			if ( threadIdx <  TARGET_ROWS_BY_VEC )
			{
				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(sumVec, sumVec, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}
			}

			__local %TYPE xShared[%V];

			int targetRowTemp = rowStart + threadIdx * %V;
			int VECTOR_SIZE   = %V;

			//#pragma unroll
			for( int i=0; i < (TARGET_ROWS_BY_VEC); i++)
			{
				if ( threadIdx == i )
			        {
					// Read X-vector
					%TYPE xVal[%V];
					//#pragma unroll
					for( int j = 0; j < %V; j++)
					{
						xVal[j] = xnew[ (targetRowTemp + j)* incx];
					}

					// Read A %Vx%V region into reg
					%TYPE reg[%V][%V];
					//#pragma unroll
					for( int idx = 0; idx < ( %V * %V); idx++)
					{
						int m = idx % ( %V ); // Row : Col-Major idx...
						int n = idx / ( %V );    // Col
						if ( m >= n )
						{
							reg[m][n] = A((targetRowTemp + m), (targetCol + n));
							%CONJUGATE(doConj, reg[m][n]);
						}
					}

					%TYPE sumVecReg[%V];
					%VSTOREWITHINCX(sumVecReg, sumVec, 1);

					// Solve for first x - Do the rest in loop
					%TYPE x[%V];
					%SUB(x[0], xVal[0], sumVecReg[0]);
					%DIV(sumVecReg[0], x[0], reg[0][0]);
					x[0] = sumVecReg[0];
					xShared[0] = sumVecReg[0];
					xnew[ (targetRowTemp)* incx ] = sumVecReg[0];

					//#pragma unroll
					for(int m = 1; m < %V; m++)
					{
						%SUB(x[m], xVal[m], sumVecReg[m]);
					}

					//#pragma unroll
					for( int idx =  %V; idx < (%V * %V); idx++)
					{
						int m = idx / %V;       // Row : Row-Major idx, x[1] is solved before x[2]
						int n = idx % ( %V );// Col
						if ( m > n)
						{
							//x[m] = x[m] - reg[m][n] * x[n];
							%MAD(x[m], reg[m][n], (-x[n]));
						}
						else if ( m == n)
						{
							%DIV(sumVecReg[m], x[m], reg[m][m]);
							x[m] = sumVecReg[m];
						}
					}

					// Store results
					//#pragma unroll
					for(int m = 1; m < %V; m++)
					{
						xShared[m] = x[m];
						xnew[ (targetRowTemp + m)* incx ] = x[m];
					}
			        }


			        // Sync so that xShared it available to all threads
			        barrier(CLK_LOCAL_MEM_FENCE);
			      	if ( (threadIdx > i) && ( threadIdx < (TARGET_ROWS_BY_VEC)) )
				{
						//#pragma unroll
						for( int j=0; j < %V; j++)
						{
							%TYPE%V loadedAVec  = %VLOAD( 0, &A((targetRowTemp), (targetCol +j)));
							%CONJUGATE(doConj, loadedAVec);
							%VMAD(sumVec, loadedAVec, xShared[j]);
						}
				}

				targetCol = targetCol + %V;
			        // Avoid Race...
			        barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		else
		{
			// Do Vector Reduction on each block except the last Block
			if ( threadIdx < TARGET_ROWS_BY_VEC)
			{
				%TYPE   accTemp = %MAKEVEC(0.0);
				%TYPE%V acc 	= %VMAKEVEC(accTemp);

				//#pragma unroll
				for( int j=0; j < %TARGET_WIDTH; j++)
				{
					%ADD(acc, acc, sDataTemp[ threadIdx + j * TARGET_ROWS_BY_VEC]);
				}

				// Store the result
				int targetRow = rowStart + threadIdx * %V;

				__global %TYPE* xNewPtr =  xnew + targetRow * incx;
				//float4 value = (float4)( xNewPtr[0], xNewPtr[incx], xNewPtr[incx * 2], xNewPtr[incx *3]);
				%TYPE%V value;
				%VLOADWITHINCX(value, xNewPtr, incx);

				// Compute result
				%SUB(value, value, acc);

				// Store results
				%VSTOREWITHINCX(xNewPtr, value, incx);
			}
		}
	}
}
";


const char *trsv_CUT_ComputeRectangle_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
    #define A( row, col) (*( A + (((col)*((col)+1))/2 + (row))))
#else
    #define A( row, col) A[ (row) + (col) * lda]
#endif
__kernel void %PREFIXtrsv_CUT_ComputeRectangle_kernel(__global const %TYPE* _A,			__global %TYPE* _xnew,
												uint N,
												int incx,
												int isUnity,
												uint lda,
												int doConj,
												int startRow, int endRow, uint offa, uint offx)
{
	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;
	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew  + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	int threadID = get_local_id(0);
	int threadID_Y, threadID_X;
	int blockSize = %BLOCKSIZE, blockSize_x, blockSize_y;
	int blkid = get_group_id(0);
	int V= %V;

	__local %TYPE solved[%TRIANGLE_HEIGHT];
	__local %TYPE reduce[%TARGET_HEIGHT][ %BLOCKSIZE / %TARGET_HEIGHT];
	__local %TYPE%V *solved_vec;
	int blockStartRow;
	int triangleHeight;
	%TYPE%V acc;
	%TYPE%V loadedAVec;
	%TYPE sacc;
	%TYPE accTemp;

	triangleHeight = endRow - startRow;
/*
	if ((triangleHeight != %TRIANGLE_HEIGHT) || ((triangleHeight % V) != 0))
	{
		// throw -1;

		//
		// It is the caller's responsibility to solve triangle whose width
		// is a multiple of VECTOR SIZE before calling this routine.
		// This makes the width of the rectangle to be multiple of VECTOR SIZE.
		// Thus threads can iterate without looking out for vector-unfriendly
		// dimensions.
		// This condition can be maintained for any dimension of the input matrix
		// So, generality is not broken here.
		//
		*(__global int*)0 = 0;
	}

	if (( %BLOCKSIZE % %TARGET_HEIGHT) != 0)
	{
		// throw -1;

		//
		// Awkward Block Size. Impossible to write neat code.
		// The set of threads belonging to the last threadID_X will not have
		// blockSize_Y number of threads.
		//
		*(__global int*)0 = 0;
	}
*/
	blockSize_y = %TARGET_HEIGHT;
	blockSize_x = %BLOCKSIZE / %TARGET_HEIGHT;

	threadID_Y = threadID % %TARGET_HEIGHT;
	threadID_X = threadID / %TARGET_HEIGHT;

	blockStartRow = endRow + (blkid * blockSize_x);
	blockStartRow += threadID_X;

	for(int i=threadID; i< %TRIANGLE_HEIGHT; i+=blockSize)
	{
		solved[i] = xnew[(startRow + i)*incx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	solved_vec = solved;
	accTemp = %INIT(0.0);
	acc = %VMAKEVEC( accTemp);

	if (blockStartRow < N)
	{
		for(int i=threadID_Y; i<(triangleHeight/V); i+=blockSize_y)
		{
			loadedAVec = %VLOAD(0, &A((startRow + i*V), (blockStartRow)));
			%CONJUGATE(doConj, loadedAVec);
			%VMAD(acc, solved_vec[i], loadedAVec); //startRow == startCol as well.
		}
		sacc = %REDUCE_SUM(acc);

		// Put stuff in shared memory for final reduction
		reduce[threadID_Y][threadID_X] = sacc;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if ( threadID < blockSize_x)
	{
		sacc = %INIT(0.0);
		//#pragma unroll
		for( int i=0; i < %TARGET_HEIGHT; i++)
		{
			%ADD(sacc, sacc, reduce[i][threadID]);
		}

		blockStartRow = endRow + (blkid * blockSize_x);
		blockStartRow += threadID;
		if ( blockStartRow < N)
		{
			%SUB(xnew[(blockStartRow)*incx], xnew[(blockStartRow)*incx], sacc);
		}
	}
}
";

const char *trsv_CLT_ComputeRectangle_kernel="
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
    #define A(row, col) (*( A + ((( (col) *((2*N) + 1 - (col))) / 2) + ((row) - (col)))))
#else
    #define A(row, col) A[ (row) + (col) * lda]
#endif
__kernel void %PREFIXtrsv_CLT_ComputeRectangle_kernel( 	__global const %TYPE* _A,
												__global %TYPE* _xnew,
												uint N,
												int incx,
												int isUnity,
												uint lda,
												int doConj,
												int startRow, int endRow, uint offa, uint offx)
{

	__global %TYPE* xnew;
	__global %TYPE* A = _A + offa;
	if ( incx < 0 ) // Goto end of vector
	{
		xnew	 = _xnew  + offx + ( N - 1) * abs(incx);
	}
	else
	{
		xnew 	= _xnew + offx;
	}

	int threadID = get_local_id(0);
	int threadID_Y, threadID_X;
	int blockSize = %BLOCKSIZE, blockSize_x, blockSize_y;
	int blkid = get_group_id(0);
	int V= %V;

	__local %TYPE solved[%TRIANGLE_HEIGHT];
	__local %TYPE reduce[%TARGET_HEIGHT][ %BLOCKSIZE / %TARGET_HEIGHT];
	__local %TYPE%V *solved_vec;
	int blockStartRow;
	int triangleHeight;
	%TYPE%V acc;
	%TYPE%V loadedAVec;
	%TYPE sacc;
	%TYPE accTemp;

	triangleHeight = endRow - startRow;

	blockSize_y = %TARGET_HEIGHT;
	blockSize_x = %BLOCKSIZE / %TARGET_HEIGHT;

	threadID_Y = threadID % %TARGET_HEIGHT;
	threadID_X = threadID / %TARGET_HEIGHT;

	blockStartRow = startRow - 1 - (blkid * blockSize_x);
	blockStartRow -= threadID_X;

	for(int i=threadID; i< %TRIANGLE_HEIGHT; i+=blockSize)
	{
		solved[i] = xnew[(startRow + i)*incx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	solved_vec = solved;
	accTemp = %INIT(0.0);
	acc = %VMAKEVEC( accTemp);

	if (blockStartRow >= 0)
	{
		for(int i=threadID_Y; i<(triangleHeight/V); i+=blockSize_y)
		{
			loadedAVec = %VLOAD(0, &A((startRow+ i*V) , (blockStartRow)));
			%CONJUGATE(doConj, loadedAVec);
			%VMAD(acc, solved_vec[i], loadedAVec); //startRow == startCol as well.
		}
		sacc = %REDUCE_SUM(acc);

		// Put stuff in shared memory for final reduction
		reduce[threadID_Y][threadID_X] = sacc;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if ( threadID < blockSize_x)
	{
		sacc = %INIT(0.0);
		//#pragma unroll
		for( int i=0; i < %TARGET_HEIGHT; i++)
		{
			%ADD(sacc, sacc, reduce[i][threadID]);
		}

		blockStartRow = startRow - 1 - (blkid * blockSize_x);
		blockStartRow -= threadID;
		if ( blockStartRow < N)
		{
			%SUB(xnew[(blockStartRow)*incx], xnew[(blockStartRow)*incx], sacc);
		}
	}
}
";

