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



// Column-Major Upper Case
static const char *trmv_CU_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#ifdef PACKED
	#define A( row, col) (*( A + ((col*(col+1))/2 + row)))
#else
	#define A( row, col) A[ row + col * lda]
#endif

#define TARGET_ROWS_BY_VEC  ((%TARGET_ROWS)/(%V))
#define TARGET_WIDTH ((%BLOCKSIZE)/(TARGET_ROWS_BY_VEC))

__kernel void %PREFIXtrmv_CU_kernel( __global %TYPE const* restrict _A, __global %TYPE * _xnew, __global %TYPE const* restrict _x_vector, uint N,
									int incx, int isUnity, uint lda, int doConj, uint offa, uint offx
#ifdef HEMV_ONLY
, int incy, uint offy, %TYPE alpha, %TYPE beta
#endif
 )
{
	__global %TYPE const* x_vector;
	__global %TYPE* xnew;
	__global %TYPE const* restrict A;

	A = _A + offa;
	if ( incx < 0 ) // Goto end of vector
	{
		#ifdef HEMV_ONLY
			x_vector = _x_vector + offx + ( N - 1) * abs(incx);
		#else
			x_vector = _x_vector + ( N - 1) * abs(incx);
			xnew	 = _xnew  + (N - 1) * abs(incx) + offx;
		#endif
	}
	else
	{
		#ifdef HEMV_ONLY
			x_vector = _x_vector + offx;
		#else
			x_vector = _x_vector;
			xnew 	 = _xnew + offx;
		#endif
	}

	#ifdef HEMV_ONLY
	if(incy < 0)
		xnew  = _xnew + offy + ( N - 1) * abs(incy);
	else
		xnew = _xnew + offy;
	#endif


	__local %TYPE  sXData[ TARGET_WIDTH ]; // Each column is multiplied with a common x_vector element

	const int gIdx = get_global_id(0);
	const int bIdx = get_group_id(0);
	const int threadIdx = get_local_id(0);
	const int TARGET_ROWS  = %TARGET_ROWS;

	// Last block always targets the top rows
	// which may be less than or equal to 64
	int nBlocks = (N-1)/ %TARGET_ROWS + 1;

	if( bIdx == (nBlocks-1))
	{
		// Variables that don't change while looping
		int startRow = bIdx * %TARGET_ROWS;
		int destRow  = (startRow + threadIdx) ;
		if( destRow >= N)
		{
			return;
		}

		//float acc = 0.0f;
		%TYPE acc 	= %MAKEVEC( 0.0);
		%TYPE accTemp 	= %MAKEVEC( 0.0);

		for ( int j= ( N - 1 ) ; j > destRow ; j--)
		{
			//acc += A( destRow, j) * x_vector[ j * incx];
			accTemp = A( destRow, j);
			%CONJUGATE(doConj, accTemp);
			%MAD(acc, accTemp, x_vector[ j * incx]);
		}

		if ( isUnity )
		{
			#ifdef HEMV_ONLY
				%TYPE acc1, temp;
                %MUL(acc1, acc, alpha);
                temp = xnew[ destRow * incy];
                %ADD(xnew[ destRow * incy], temp, acc1);
			#else
				%ADD(xnew[ destRow * incx] , acc, x_vector[ destRow * incx]);
			#endif
		}
		else
		{
			//xnew[ destRow * incx] = acc + A( destRow , destRow) * x_vector[ destRow * incx];
			accTemp = A( destRow, destRow);

			#ifdef HEMV_ONLY
                #ifndef SPMV_ONLY
				    // accTemp.odd = 0.0f;
                    %CLEAR_IMAGINARY( accTemp );
			    #endif
            #endif

			%CONJUGATE(doConj, accTemp);
			%MAD(acc, accTemp, x_vector[ destRow * incx]);

			#ifdef HEMV_ONLY
				%TYPE temp, acc1;
				%MUL(temp, xnew[ destRow * incy], beta);
				%MUL(acc1, acc, alpha);
				%ADD(xnew[ destRow * incy], temp, acc1);
			#else
				xnew[ destRow * incx] = acc;
			#endif
		}
	}
	else
	{
		%TYPE sumTemp= %MAKEVEC( 0.0);
		%TYPE%V sum = %VMAKEVEC( sumTemp);

		// Variables that don't change while looping
		int startRow = bIdx * %TARGET_ROWS;
		//int rowShift = ((threadIdx & ( TARGET_ROWS_BY_VEC -1 )) * %V);
		int rowShift = ((threadIdx %  (TARGET_ROWS_BY_VEC)) * %V);
		int colShift = threadIdx / TARGET_ROWS_BY_VEC;

		int row	= startRow + rowShift;

		// gIdx is not destination row.

		// startRow may be less than 4
		// So nLoops will be negative
		// and the FOR loop doesn't execute
		int nLoops = (( N - (startRow + %TARGET_ROWS))/ TARGET_WIDTH) - 1;

		for( int j=0; j <= (nLoops); j++)
		{
			int startCol	= N - (j + 1)* TARGET_WIDTH;
			int col 	= startCol + colShift;

			//
			// Only TARGET_WIDTH threads points are to be read from X-vector
			// We dont't use VLOAD here because incx could be > 1
			// Minimal prototyping shows that having separate loading code
			// for incx value of 1 does not change anything in performance
			// In fact, the extra IF costs us.
			//
			barrier(CLK_LOCAL_MEM_FENCE);
			if (threadIdx < TARGET_WIDTH)
			{
				sXData[threadIdx] = x_vector[(startCol + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			// TARGET_ROWS_BY_VEC way bank-conflict : May broadcast if TARGET_ROWS = BLOCKSIZE, which reduces occupancy
			// And we loose performance as we don't have enough blocks to hide memory access and compute latenties per MP
			%TYPE xData =  sXData[colShift];

			//sum += vload4(0, &A( row, col)) * ((float4)( xData, xData, xData, xData));
			// ((float4)( xData, xData, xData, xData));
			%TYPE%V loadedA = %VLOAD(0, (&A( row, col)));
			%CONJUGATE(doConj, loadedA);

			%TYPE%V xDataTemp = %VMAKEVEC(xData);
			%VMAD( sum, loadedA, xDataTemp);
		}


		volatile __local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * TARGET_WIDTH];
		volatile __local %TYPE* sData = sDataTemp;
		//sDataTemp[(threadIdx & ( TARGET_ROWS_BY_VEC -1 )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		// Reduce each block by DTARGET_ROWS threads to generate DTARGET_ROWS acc values
		if ( threadIdx < %TARGET_ROWS)
		{
			//float acc = 0.0f;
			%TYPE acc 	= %MAKEVEC( 0.0);
			%TYPE accTemp 	= %MAKEVEC( 0.0);
			int desRow  = (bIdx * %TARGET_ROWS)+ threadIdx;

			//#pragma unroll TARGET_WIDTH
			for( int j=0; j < TARGET_WIDTH; j++)
			{
				//acc += sData[ threadIdx + j * FTARGET_ROWS];
				%ADD(acc, acc, sData[ threadIdx + j * TARGET_ROWS]);
			}

			for ( int j= (N  - (nLoops+1)* TARGET_WIDTH - 1) ; j > desRow; j--)
			{
				//acc += A( desRow, j) * x_vector[ j * incx];
				accTemp = A( desRow, j);
				%CONJUGATE(doConj, accTemp);
				%MAD(acc, accTemp, x_vector[ j * incx]);
			}

			if ( isUnity )
			{
				//%ADD(xnew[ desRow * incx], acc, x_vector[ desRow * incx]);
				#ifdef HEMV_ONLY
		            %TYPE acc1, temp;
                    %MUL(acc1, acc, alpha);
                    temp = xnew[ desRow * incy];
                    %ADD(xnew[ desRow * incy], temp, acc1);
            	#else
                	%ADD(xnew[ desRow * incx] , acc, x_vector[ desRow * incx]);
            	#endif
			}
			else
			{
				// xnew[ desRow * incx] =  acc + A( desRow, desRow) * x_vector[ desRow * incx];
				accTemp = A( desRow, desRow );

            	#ifdef HEMV_ONLY
                    #ifndef SPMV_ONLY
            	        //accTemp.odd = 0.0f;
                        %CLEAR_IMAGINARY( accTemp );
            	    #endif
                #endif

				%CONJUGATE(doConj, accTemp);
				%MAD(acc, accTemp, x_vector[ desRow * incx]);

	            #ifdef HEMV_ONLY
    	            %TYPE temp, acc1;
            	    %MUL(temp, xnew[ desRow * incy], beta);
					%MUL(acc1, acc, alpha);
               		%ADD(xnew[ desRow * incy], temp, acc1);
            	#else
                	xnew[ desRow * incx] = acc;
            	#endif
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}";

// Column-Major Lower Case

static const char *trmv_CL_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#ifdef PACKED
	#define A( row, col) (*( A + ((( col *((2*N) + 1 - col)) / 2) + (row - col))))
#else
	#define A( row, col) A[ row + col * lda]
#endif

#define TARGET_ROWS_BY_VEC  ((%TARGET_ROWS)/(%V))
#define TARGET_WIDTH ((%BLOCKSIZE)/(TARGET_ROWS_BY_VEC))
__kernel void %PREFIXtrmv_CL_kernel( __global %TYPE const* restrict _A, __global %TYPE* _xnew, __global %TYPE const* restrict _x_vector,
									uint N, int incx, int isUnity, uint lda, int doConj, uint offa, uint offx
#ifdef HEMV_ONLY
, int incy, uint offy, %TYPE alpha, %TYPE beta
#endif
 )
{
	__global %TYPE* x_vector;
	__global %TYPE* xnew;
	__global %TYPE const * restrict A;

	A = _A + offa;
	if ( incx < 0 ) // Goto end of vector
	{
		#ifdef HEMV_ONLY
            x_vector = _x_vector + offx + ( N - 1) * abs(incx);
        #else
            x_vector = _x_vector + ( N - 1) * abs(incx);
            xnew     = _xnew + offx + ( N - 1) * abs(incx);
        #endif
	}
	else
	{
		#ifdef HEMV_ONLY
            x_vector = _x_vector + offx;
        #else
            x_vector = _x_vector;
            xnew     = _xnew + offx;
        #endif
	}


    #ifdef HEMV_ONLY
    if(incy < 0)
        xnew  = _xnew + offy + ( N - 1) * abs(incy);
    else
        xnew = _xnew + offy;
    #endif

	__local %TYPE sXData[ TARGET_WIDTH ]; // Each column is multiplied with a common x_vector element

	size_t gIdx = get_global_id(0);
	size_t bIdx = get_group_id(0);
	size_t threadIdx = get_local_id(0);
	int TARGET_ROWS  = %TARGET_ROWS;

	// Last block always targets the top rows
	// which may be less than or equal to 64
	size_t nBlocks = (N-1)/ %TARGET_ROWS + 1;


	if( bIdx == (nBlocks-1))
	{
		// Target row of xNew is given by threadIdx
		size_t lastRow  = (N - (nBlocks -1) * %TARGET_ROWS) -1;

		if( threadIdx > lastRow )
		{
			return;
		}

		//float acc = 0.0f;
		%TYPE acc 	= %MAKEVEC( 0.0);
		%TYPE accTemp 	= %MAKEVEC( 0.0);

		for ( int j= 0 ; j < threadIdx; j++)
		{
			//acc += A(threadIdx, j) * x_vector[ j * incx];
			accTemp = A(threadIdx, j);
			%CONJUGATE(doConj, accTemp);
			%MAD(acc, accTemp, x_vector[ j * incx]);
		}

		if ( isUnity )
		{
			#ifdef HEMV_ONLY
            	%TYPE acc1, temp;
                %MUL(acc1, acc, alpha);
                temp = xnew[ threadIdx * incy];
                %ADD(xnew[ threadIdx * incy], temp, acc1);
            #else
                %ADD(xnew[ threadIdx * incx] , acc, x_vector[ threadIdx * incx]);
            #endif
		}
		else
		{	//xnew[ threadIdx * incx] =  acc +  A(threadIdx, threadIdx) * x_vector[ threadIdx * incx];
			accTemp = A(threadIdx, threadIdx);

            #ifdef HEMV_ONLY
                #ifndef SPMV_ONLY
                    //accTemp.odd = 0.0f;
                    %CLEAR_IMAGINARY( accTemp );
                #endif
            #endif

			%CONJUGATE(doConj, accTemp);
			%MAD(acc, accTemp, x_vector[ threadIdx * incx]);

            #ifdef HEMV_ONLY
                %TYPE temp, acc1;
                %MUL(temp, xnew[ threadIdx * incy], beta);
                %MUL(acc1, acc, alpha);
	            %ADD(xnew[ threadIdx * incy], temp, acc1);
            #else
                xnew[ threadIdx * incx] = acc;
            #endif
		}
	}
	else
	{
		%TYPE sumTemp= %MAKEVEC( 0.0);
		%TYPE%V sum = %VMAKEVEC( sumTemp);

		// Variables that don't change while looping
		size_t startRow = N - (bIdx + 1)* %TARGET_ROWS;
		//size_t rowShift = ((threadIdx & ( TARGET_ROWS_BY_VEC -1 )) * %V);
		size_t rowShift = ((threadIdx % ( TARGET_ROWS_BY_VEC  )) * %V);
		size_t colShift = threadIdx / TARGET_ROWS_BY_VEC;

		size_t row	= startRow + rowShift;

		// gIdx is not destination row.
		size_t desRow  = startRow + threadIdx;

		// startRow may be less than 4
		// So nLoops will be negative
		// and the FOR loop doesn't execute
		int nLoops = ( startRow / TARGET_WIDTH) - 1;

		for( int j=0; j <= (nLoops); j++)
		{
			size_t startCol	= j * TARGET_WIDTH;
			size_t col 	= startCol + colShift;

			//
			// Only TARGET_WIDTH threads points are to be read from X-vector
			// We dont't use VLOAD here because incx could be > 1
			// Minimal prototyping shows that having separate loading code
			// for incx value of 1 does not change anything in performance
			// In fact, the extra IF costs us.
			//
			barrier(CLK_LOCAL_MEM_FENCE);
			if (threadIdx < TARGET_WIDTH)
			{
				sXData[threadIdx] = x_vector[(startCol + threadIdx) * incx];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			// TARGET_ROWS_BY_VEC way bank-conflict : May broadcast if TARGET_ROWS = BLOCKSIZE, which reduces occupancy
			// And we loose performance as we don't have enough blocks to hide memory access and compute latenties per MP
			%TYPE xData =  sXData[colShift];

			//sum += vload4(0, &A( row, col)) * ((float4)( xData, xData, xData, xData));
			// ((float4)( xData, xData, xData, xData));
			%TYPE%V loadedA = %VLOAD(0, (&A( row, col)));
			%CONJUGATE(doConj, loadedA);

			%TYPE%V xDataTemp = %VMAKEVEC(xData);
			%VMAD(sum, loadedA, xDataTemp);
		}


		__local %TYPE%V sDataTemp[TARGET_ROWS_BY_VEC * TARGET_WIDTH];
		__local %TYPE* sData = sDataTemp;
		//sDataTemp[(threadIdx & ( TARGET_ROWS_BY_VEC -1 )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		sDataTemp[(threadIdx % ( TARGET_ROWS_BY_VEC  )) + (colShift * TARGET_ROWS_BY_VEC)] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		// Reduce each block by DTARGET_ROWS threads to generate DTARGET_ROWS acc values
		if ( threadIdx < %TARGET_ROWS)
		{
			//float acc = 0.0f;
			%TYPE acc 	= %MAKEVEC( 0.0);
			%TYPE accTemp 	= %MAKEVEC( 0.0);

			//#pragma unroll TARGET_WIDTH
			for( int j=0; j < TARGET_WIDTH; j++)
			{
				//acc += sData[ threadIdx + j * FTARGET_ROWS];
				%ADD(acc, acc, sData[ threadIdx + j * TARGET_ROWS]);
			}

			for ( int j= ((nLoops+1)* TARGET_WIDTH) ; j < desRow; j++)
			{
				//acc += A(desRow, j) * x_vector[ j * incx];
				accTemp = A(desRow, j);
				%CONJUGATE(doConj, accTemp);
				%MAD(acc, accTemp, x_vector[ j * incx]);
			}

			if ( isUnity )
			{
				 #ifdef HEMV_ONLY
                    %TYPE acc1, temp;
                    %MUL(acc1, acc, alpha);
                    temp = xnew[ desRow * incy];
                    %ADD(xnew[ desRow * incy], temp, acc1);
                #else
                    %ADD(xnew[ desRow * incx] , acc, x_vector[ desRow * incx]);
                #endif
			}
			else
			{
				// xnew[ desRow * incx] =  acc + A(desRow, desRow) * x_vector[ desRow * incx];
				accTemp = A(desRow, desRow);

                #ifdef HEMV_ONLY
                    #ifndef SPMV_ONLY
                        //accTemp.odd = 0.0f;
                        %CLEAR_IMAGINARY( accTemp );
                    #endif
                #endif

				%CONJUGATE(doConj, accTemp);
				%MAD(acc, accTemp, x_vector[ desRow * incx]);

                #ifdef HEMV_ONLY
                    %TYPE temp, acc1;
                    %MUL(temp, xnew[ desRow * incy], beta);
                    %MUL(acc1, acc, alpha);
	               	%ADD(xnew[ desRow * incy], temp, acc1);
                #else
                    xnew[ desRow * incx] = acc;
                #endif
			}
		}
	}
}";

// Column-Major Lower Transpose
// Threads : %PREFIXBLOCKSIZET, Blocks launched = (N -1) / %PREFIXTARGET_ROWST + 1
/*
#define %PREFIXVECTOR_SIZET %V
#define %PREFIXTARGET_WIDTH_BY_VECT ( %PREFIXBLOCKSIZET / %PREFIXTARGET_ROWST )
#define %PREFIXTARGET_WIDTHT ( %PREFIXTARGET_WIDTH_BY_VECT * %PREFIXVECTOR_SIZET )
*/

static const char *trmv_CLT_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#ifdef PACKED
	#define A( row, col) (*( A + ((( col *((2*N) + 1 - col)) / 2) + (row - col))))
#else
	#define A( row, col) A[ row + col * lda]
#endif

#define TARGET_WIDTH_BY_VEC ((%BLOCKSIZE) / (%TARGET_ROWS) )
#define TARGET_WIDTH ((TARGET_WIDTH_BY_VEC) * (%V))
__kernel void %PREFIXtrmv_CLT_kernel( __global %TYPE const* restrict _A, __global %TYPE * _xnew, __global %TYPE const* restrict _x_vector,
									  uint N, int incx, int isUnity, uint lda, int doConj, uint offa, uint offx
#ifdef HEMV_ONLY
, int incy, uint offy, %TYPE alpha, %TYPE beta
#endif
 )
{
	__global %TYPE* x_vector;
	__global %TYPE* xnew;
	__global %TYPE const * restrict A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		 #ifdef HEMV_ONLY
            x_vector = _x_vector + offx + ( N - 1) * abs(incx);
        #else
            x_vector = _x_vector + ( N - 1) * abs(incx);
            xnew     = _xnew + offx + ( N - 1) * abs(incx);
        #endif
	}
	else
	{
		#ifdef HEMV_ONLY
            x_vector = _x_vector + offx;
        #else
            x_vector = _x_vector;
            xnew     = _xnew + offx;
        #endif
	}


    #ifdef HEMV_ONLY
    if(incy < 0)
        xnew  = _xnew + offy + ( N - 1) * abs(incy);
    else
        xnew = _xnew + offy;
    #endif

	int gIdx 	= get_global_id(0);
	int blockIdx	= get_group_id(0);
	int blockDim  	= get_local_size(0);
	int threadIdx 	= get_local_id(0);

	__local %TYPE xShared[TARGET_WIDTH];

	int startCol  	= blockIdx * %TARGET_ROWS;

	%TYPE accTemp= %INIT( 0.0);
	%TYPE%V acc  = %VMAKEVEC( accTemp);

	//size_t rowShift = ((threadIdx & ( TARGET_WIDTH_BY_VEC -1 )) * %V);
	size_t rowShift = ((threadIdx % ( TARGET_WIDTH_BY_VEC  )) * %V);
	size_t colShift = threadIdx / TARGET_WIDTH_BY_VEC;
	size_t col	= startCol + colShift;
	int startRow;

	for( startRow = (N - TARGET_WIDTH); ( startCol + %TARGET_ROWS - 1 ) < startRow; startRow = (startRow - TARGET_WIDTH))
	{
		// Load X data into Shared memory
		barrier(CLK_LOCAL_MEM_FENCE);
		if (threadIdx < TARGET_WIDTH)
		{
			xShared[threadIdx] = x_vector[ (startRow + threadIdx) * incx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//float4 xData = (float4)(xShared[ rowShift ], xShared[ rowShift + 1], xShared[ rowShift + 2], xShared[ rowShift + 3]);
		%TYPE%V xData;
		__local %TYPE%V* xSharedTemp = (xShared + rowShift);
		xData = *(xSharedTemp);

		int row	= startRow + rowShift;
		//acc   	+= vload4(0, &A(row, col)) * xData;
		%TYPE%V loadedA = %VLOAD( 0, (&A(row, col)) );
		%CONJUGATE(doConj, loadedA);
		%VMAD(acc, loadedA, xData);
	}
	// Restore startRow
	startRow += TARGET_WIDTH;

	__local %TYPE%V sDataTemp[TARGET_WIDTH_BY_VEC * %TARGET_ROWS];
	__local %TYPE* sData = sDataTemp;

	// blocks that did vectorLoads
	bool vectorBlocks = ( startRow != N);
	if ( vectorBlocks )
	{

		//sDataTemp[ ( threadIdx & ( TARGET_WIDTH_BY_VEC -1 ) ) + (colShift * TARGET_WIDTH_BY_VEC) ] = acc;
		sDataTemp[ ( threadIdx % ( TARGET_WIDTH_BY_VEC  ) ) + (colShift * TARGET_WIDTH_BY_VEC) ] = acc;
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	%TYPE sum 	= %MAKEVEC( 0.0);
	%TYPE loadedA 	= %MAKEVEC( 0.0);

	if( threadIdx < %TARGET_ROWS && ( (startCol + threadIdx) < N))
	{
		if ( vectorBlocks )
		{
			//#pragma unroll	TARGET_WIDTH
			for( int i=0 ; i < TARGET_WIDTH; i++)
			{
				%ADD(sum, sum, sData[i + (threadIdx * TARGET_WIDTH )]);
			}

		}

		int destRow = blockIdx * %TARGET_ROWS + threadIdx;

		// Loop from startRow - 1 till destRow
		for( int i= ( startRow - 1); i > destRow; i--)
		{
			loadedA = A(i, destRow);
			%CONJUGATE(doConj, loadedA);
			%MAD(sum, loadedA, x_vector[ i * incx]);
		}
		if ( isUnity)
		{
			#ifdef HEMV_ONLY
                %TYPE acc1, temp;
                %MUL(acc1, sum, alpha);
                temp = xnew[ destRow * incy];
                %ADD(xnew[ destRow * incy], temp, acc1);
            #else
            	%ADD(xnew[ destRow * incx] , sum, x_vector[ destRow * incx]);
            #endif
		}
		else
		{
			loadedA = A(destRow, destRow);

            #ifdef HEMV_ONLY
                #ifndef SPMV_ONLY
            	    //loadedA.odd = 0.0f;
                    %CLEAR_IMAGINARY( loadedA );
                #endif
            #endif

			%CONJUGATE(doConj, loadedA);
			%MAD(sum, loadedA, x_vector[ destRow * incx]);

			#ifdef HEMV_ONLY
				%TYPE temp, acc1;
				%MUL(temp, xnew[ destRow * incy], beta);
				%MUL(acc1, sum, alpha);
				%ADD(xnew[ destRow * incy], temp, acc1);
			#else
				xnew[ destRow * incx] = sum;
			#endif
		}
	}
}";



// Column-Major Upper Transpose
// Threads : %PREFIXBLOCKSIZET, Blocks launched = (N -1) / %PREFIXTARGET_ROWST + 1
static const char *trmv_CUT_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#ifdef PACKED
	#define A( row, col) (*( A + ((col*(col+1))/2 + row)))
#else
	#define A( row, col) A[ row + col * lda]
#endif

#define TARGET_WIDTH_BY_VEC ((%BLOCKSIZE) / (%TARGET_ROWS) )
#define TARGET_WIDTH ((TARGET_WIDTH_BY_VEC) * (%V))

__kernel void %PREFIXtrmv_CUT_kernel( __global %TYPE const* restrict _A, __global %TYPE * _xnew, __global  %TYPE const* restrict _x_vector,
									  uint N, int incx, int isUnity, uint lda, int doConj, uint offa, uint offx
#ifdef HEMV_ONLY
, int incy, uint offy, %TYPE alpha, %TYPE beta
#endif
 )
{
	__global %TYPE* x_vector;
	__global %TYPE* xnew;
	__global %TYPE const* restrict A = _A + offa;

	if ( incx < 0 ) // Goto end of vector
	{
		#ifdef HEMV_ONLY
            x_vector = _x_vector + offx + ( N - 1) * abs(incx);
        #else
            x_vector = _x_vector + ( N - 1) * abs(incx);
            xnew     = _xnew + offx + ( N - 1) * abs(incx);
        #endif
	}
	else
	{
		#ifdef HEMV_ONLY
            x_vector = _x_vector + offx;
        #else
            x_vector = _x_vector;
            xnew     = _xnew + offx;
        #endif
	}


    #ifdef HEMV_ONLY
    if(incy < 0)
        xnew  = _xnew + offy + ( N - 1) * abs(incy);
    else
        xnew = _xnew + offy;
    #endif

	int gIdx 	= get_global_id(0);
	int blockIdx	= get_group_id(0);
	int blockDim  	= get_local_size(0);
	int threadIdx 	= get_local_id(0);

	__local %TYPE xShared[TARGET_WIDTH];

	int startRow  	= 0;
	int startCol  	= N - (blockIdx + 1)* %TARGET_ROWS;

	// Do scalar if this condition is true
	if ( (startRow + TARGET_WIDTH - 1 ) >= startCol)
	{
		int destRow = (startCol + threadIdx) ;

		if( (threadIdx < %TARGET_ROWS) && ( destRow >= 0))
		{
			%TYPE sum = %MAKEVEC(0.0);
			%TYPE accTemp = %MAKEVEC(0.0);

			// Loop from (startRow - 1) till destRow
			for( int i= 0; i < destRow; i++)
			{
				accTemp = A(i, destRow);
				%CONJUGATE(doConj, accTemp);
				%MAD(sum, accTemp, x_vector[ i * incx]);
			}
			if ( isUnity)
			{
				#ifdef HEMV_ONLY
              		%TYPE acc1, temp;
                    %MUL(acc1, sum, alpha);
                    temp = xnew[ destRow * incy];
                    %ADD(xnew[ destRow * incy], temp, acc1);
           		#else
                	%ADD(xnew[ destRow * incx] , sum, x_vector[ destRow * incx]);
            	#endif
			}
			else
			{
				accTemp = A(destRow, destRow);

	            #ifdef HEMV_ONLY
                    #ifndef SPMV_ONLY
                	    //accTemp.odd = 0.0f;
                        %CLEAR_IMAGINARY( accTemp );
            	    #endif
                #endif

				%CONJUGATE(doConj, accTemp);
				%MAD(sum, accTemp, x_vector[ destRow * incx]);

    	        #ifdef HEMV_ONLY
        	        %TYPE temp, acc1;
	                %MUL(temp, xnew[ destRow * incy], beta);
                	%MUL(acc1, sum, alpha);
            	    %ADD(xnew[ destRow * incy], temp, acc1);
        	    #else
                	xnew[ destRow * incx] = sum;
            	#endif
			}
		}
	}
	else
	{
		// float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		%TYPE accTemp = %MAKEVEC( 0.0);
		%TYPE%V acc   = %VMAKEVEC(accTemp);

		//size_t rowShift = ((threadIdx & ( TARGET_WIDTH_BY_VEC -1 )) * %V);
		size_t rowShift = ((threadIdx % ( TARGET_WIDTH_BY_VEC  )) * %V);
		size_t colShift = threadIdx / TARGET_WIDTH_BY_VEC;

		int col	     = startCol + colShift;

		for( int i=0; ; i++)
		{
			// Load X data into Shared memory
			barrier(CLK_LOCAL_MEM_FENCE);
			if (threadIdx < TARGET_WIDTH)
			{
				xShared[threadIdx] = x_vector[ (startRow + threadIdx) * incx];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			//float4 xData = (float4)(xShared[ rowShift ], xShared[ rowShift + 1], xShared[ rowShift + 2], xShared[ rowShift + 3]);
			%TYPE%V xData;
			__local %TYPE%V* xSharedTemp = (xShared + rowShift);
			xData = *(xSharedTemp);

			int row	= startRow + rowShift;
			// acc   	+= vload4(0, &A(row,col)) * xData;
			%TYPE%V loadedA = %VLOAD( 0, (&A(row, col)));
			%CONJUGATE(doConj, loadedA);
			%VMAD(acc, loadedA, xData);

			startRow = startRow + TARGET_WIDTH;
			if ( (startRow + TARGET_WIDTH - 1) >= startCol)
			{
				break;
			}
		}

		//__local float4 sData[16][4];
		//sData[(threadIdx & 15)][colShift] = acc;
		//barrier(CLK_LOCAL_MEM_FENCE);
		__local %TYPE%V sDataTemp[TARGET_WIDTH_BY_VEC * %TARGET_ROWS];
		__local %TYPE* sData = sDataTemp;

		//sDataTemp[ ( threadIdx & ( TARGET_WIDTH_BY_VEC -1 ) ) + (colShift * TARGET_WIDTH_BY_VEC) ] = acc;
		sDataTemp[ ( threadIdx % ( TARGET_WIDTH_BY_VEC  ) ) + (colShift * TARGET_WIDTH_BY_VEC) ] = acc;
		barrier(CLK_LOCAL_MEM_FENCE);

		//acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		%TYPE sum 	= %MAKEVEC( 0.0);
		%TYPE loadedA 	= %MAKEVEC( 0.0);


		if( threadIdx < %TARGET_ROWS )
		{
			//#pragma unroll	TARGET_WIDTH
			for( int i=0 ; i < TARGET_WIDTH; i++)
			{
				%ADD(sum, sum, sData[i + (threadIdx * TARGET_WIDTH )]);
			}

			int destRow = (startCol + threadIdx) ;

			// Loop from startRow - 1 till destRow
			for( int i= startRow; i < destRow; i++)
			{
				loadedA = A(i, destRow);
				%CONJUGATE(doConj, loadedA);
				%MAD(sum, loadedA, x_vector[ i * incx]);
			}
			if ( isUnity)
			{
				#ifdef HEMV_ONLY
                    %TYPE acc1, temp;
                    %MUL(acc1, sum, alpha);
					temp = xnew[ destRow * incy];
                    %ADD(xnew[ destRow * incy], temp, acc1);
                #else
                    %ADD(xnew[ destRow * incx] , sum, x_vector[ destRow * incx]);
                #endif
			}
			else
			{
				loadedA = A(destRow, destRow);

                #ifdef HEMV_ONLY
                    #ifndef SPMV_ONLY
                        //loadedA.odd = 0.0f;
                        %CLEAR_IMAGINARY( loadedA );
                    #endif
                #endif

				%CONJUGATE(doConj, loadedA);
				%MAD(sum, loadedA, x_vector[ destRow * incx]);

                #ifdef HEMV_ONLY
                    %TYPE temp, acc1;
                    %MUL(temp, xnew[ destRow * incy], beta);
                    %MUL(acc1, sum, alpha);
                    %ADD(xnew[ destRow * incy], temp, acc1);
                #else
                    xnew[ destRow * incx] = sum;
                #endif
			}
		}
	}
}";


