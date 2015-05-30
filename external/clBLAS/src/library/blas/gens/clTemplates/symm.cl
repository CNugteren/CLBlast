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
// NOTE:                                  OUTDATED FILE - NOT USED
//
// BUG NOTE:
// The SYMM_C_KERNEL() suffers from TAIL BUG. Does not handle TAILS properly on the M and N side.
// Needs to be treated like GEMM2 - Having a separate TAIL_RUN and trimming M and N on Non-tail Runs.
// However, SYMM is now composed from GEMM. Only handling the diaognal portion depends on this kernel.
// So, we will fix LOADA_SECOND and LOADB_SECOND appropriately and use this kernel.
// This kernel should not be used at all.
// In essence, one should review this kernel only for the __SYMM_DIAGONAL__ portion.
//

const char *SYMM_C_KERNEL= "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#if !defined(__SYMM_UPPER__) && !defined(__SYMM_LOWER__)
	#error Upper or Lower must be defined
#endif

#if defined(__SYMM_UPPER__) && defined(__SYMM_LOWER__)
	#error Both Upper and Lower cannot be defined together
#endif

#if !defined(__SYMM_LEFT__) && !defined(__SYMM_RIGHT__)
	#error Neither Left nor Right defined
#endif

#if defined(__SYMM_LEFT__) && defined(__SYMM_RIGHT__)
	#error Both LEFT and RIGHT cannot be defined together
#endif

#if defined(__SYMM_COLMAJOR__)
	void VECTOR_STORE(%TYPE%V data, __global %TYPE *A, uint M, uint N, uint lda, uint row, uint col)
	{
		if ( ((row + %V -1) < M) && (col < N) )
		{
			%VSTORE( data, 0, (&A[col*lda + row]));
			return;
		}

		//
		// Construct from SCALAR
		//
		if ((row < M) && (col < N))
		{
			int i=0;
			%TYPE temp[%V];

			//
			// FIXME: ENDIAN ISSUES - Currently for Little endian
			//		  Needs fixes for Big Endian
			//
			*(__private %TYPE%V *)temp = data;

			for(; i< (M-row); i++)
			{
				A[col*lda + row + i] = temp[i];
			}
		}
		return;
	}

	%TYPE%V VECTOR_LOAD(__global %TYPE *A, uint M, uint N, uint lda, uint row, uint col)
	{
		%TYPE temp[%V];
		%TYPE%V retval = (%TYPE%V) 0;

		if ( ((row + %V -1) < M) && (col < N) )
		{
			retval = %VLOAD(0, (&A[col*lda + row]));
			return retval;
		}

		//
		// Construct from SCALAR
		//
		if ((row < M) && (col < N))
		{
			int i=0;

			for(; i< (M-row); i++)
			{
				temp[i] = A[col*lda + row + i];
			}
			for(; i< (%V);  i++)
			{
				temp[i] = 0;
			}
			%VLOADWITHINCX(retval, temp, 1);
		}
		return retval;
	}

	%TYPE%V SYMM_VECTOR_LOAD_USING_SCALAR(__global %TYPE *A, uint M, uint lda, uint row, uint col)
	{
		%TYPE temp[%V];
		%TYPE%V retval;

		for(uint i=0; i< (%V); i++)
		{
			if (((row + i) < M) && (col < M))
			{
				#ifdef __SYMM_UPPER__
				if ((row + i) <= col)
				#else
				if ((row + i) >= col)
				#endif
				{
					temp[i] = A[col*lda + row + i];
				} else {
					temp[i] = A[(row+i)*lda + col];
				}
			} else {
				temp[i] = (%TYPE) 0;
			}
		}
		%VLOADWITHINCX(retval, temp, 1 );
		return retval;
	}

	%TYPE%V SYMM_VECTOR_LOAD(__global %TYPE *A, uint M, uint lda, uint row, uint col)
	{
		%TYPE%V retval = (%TYPE%V) 0;

		bool validAddress = ((row >= M) || (col >=M)) ? false : true;
		bool fullyWithinUpperTriangle = validAddress && ((row + %V -1) <= col);
		bool fullyWithinLowerTriangle = validAddress && (row > col) && ((row + %V -1) < M);
		bool protrudingLowerTriangle  = validAddress && ((row + %V -1) >= M);
		bool inBetweenDiagonal  	  = validAddress && (!fullyWithinUpperTriangle) && (!fullyWithinLowerTriangle) && (!protrudingLowerTriangle);
		if (fullyWithinLowerTriangle || fullyWithinUpperTriangle)
		{
			#ifdef __SYMM_UPPER__
			if (fullyWithinUpperTriangle)
			#else
			if (fullyWithinLowerTriangle)
			#endif
			{
				retval = %VLOAD(0, (&A[(col)*lda + (row)]));
			} else {
				retval = %VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda);
			}
		} else {
			if (protrudingLowerTriangle || inBetweenDiagonal)
			{
				retval = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, row, col);
			}
		}
		return retval;
	}

	#ifdef __SYMM_LEFT__
	// (A) MxM * (B) MxN
		%TYPE%V LOADA(__global %TYPE *A, uint M, uint K, uint lda, uint row, uint col)
		{
			return SYMM_VECTOR_LOAD(A, M, lda, row, col);
		}
		#ifdef __SYMM_LOWER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_FIRST(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*lda + (row)]))
		#elif defined(__SYMM_UPPER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_FIRST(A,M,K,lda,row,col) 	%VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda)
		#endif
		#define LOADA_SECOND(A,M,K,lda,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, row, col)
		#ifdef __SYMM_LOWER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_THIRD(A,M,K,lda,row, col)	%VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda)
		#elif defined(__SYMM_UPPER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_THIRD(A,M,K,lda,row, col)	%VLOAD(0, (&A[(col)*lda + (row)]))
		#endif
		#define LOADA_TAIL(A,M,K,lda,row,col) 		SYMM_VECTOR_LOAD_USING_SCALAR(A,M,lda,row,col)

		%TYPE%V LOADB(__global %TYPE *B, uint K, uint N, uint ldb, uint row, uint col)
		{
			return VECTOR_LOAD(B, K, N, ldb, row, col );
		}
		#define LOADB_FIRST(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_SECOND(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_THIRD(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_TAIL(B,K,N,ldb,row,col)	VECTOR_LOAD(B, K, N, ldb, row, col)

	#elif defined(__SYMM_RIGHT__)
		// (A)MxN * (B)NxN
		%TYPE%V LOADA(__global %TYPE *A, uint M, uint K, uint lda, uint row, uint col)
		{
			return VECTOR_LOAD(A, M, K, lda, row, col );
		}
		#define LOADA_FIRST(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_SECOND(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_THIRD(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_TAIL(A,M,K,lda,row,col)	VECTOR_LOAD(A, M, K, lda, row, col)

		%TYPE%V LOADB(__global %TYPE *B, uint K, uint N, uint ldb, uint row, uint col)
		{
			return SYMM_VECTOR_LOAD(B, N, ldb, row, col);
		}
		#ifdef __SYMM_UPPER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_FIRST(B,K,N,ldb,row,col)	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#elif defined(__SYMM_LOWER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_FIRST(B,K,N,ldb,row,col)	%VLOADWITHINCXV2(0, (&B[(row)*(ldb)  + (col)]), ldb)
		#endif
		#define LOADB_SECOND(B,K,N,ldb,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, row, col)
		#ifdef __SYMM_UPPER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_THIRD(B,K,N,ldb,row,col)	%VLOADWITHINCXV2(0, (&B[(row)*(ldb) + (col)]), ldb)
		#elif defined(__SYMM_LOWER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_THIRD(B,K,N,ldb,row,col)	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#endif
		#define LOADB_TAIL(B,K,N,ldb,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(B, N,ldb,row,col)
	#endif // Left, Right

	__kernel void symm_C_kernel( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *_C,
			       			  	uint M, uint N, uint _lda, uint _ldb, int ldc, uint offa, uint offb, uint offc, %TYPE alpha, %TYPE beta)
	{
		__global %TYPE const *restrict A;
		__global %TYPE const *restrict B;
		__global %TYPE *C;
		uint K;
		uint lda, ldb;
		uint indexA, indexB, indexC;
		uint rowA, colA, rowB, colB, rowC, colC;
		uint numGroupsOnY;
		uint bidX, bidY;
		uint row, col;
		uint REDColStart, REDColEnd; // As the panel traverses these columns, it will slow down - Hence RED.
		uint tid = get_local_id(0);
		int panel;
		uint blockDimY;
		C = _C + offc;
	#ifdef __SYMM_LEFT__
		// MxM * MxN
		A = _A + offa;
		lda = _lda;
		B = _B + offb;
		ldb = _ldb;
		K = M;
	#elif defined(__SYMM_RIGHT__)
		// MxN * NxN
		A = _B + offb;
		lda = _ldb;
		B = _A + offa;
		ldb = _lda;
		K = N;
	#endif

		//
		// %WIDTH - Preferably 16
		// %ITEMY, %ITEMX - 1 Thread is responsible for %ITEMY * %ITEMX sub-matrix in C
		//					%ITEMY must be divisible by %V
		// The entire workgroup loops-together to complete ITEMY-ITEMX sub-matrix
		//
		uint threadsY = %WIDTH;
		uint threadsX = get_local_size(0)/threadsY;
		uint offsetY = (tid % threadsY) * %V;
		uint offsetX = (tid / threadsY);

		//
		// Column-Major ordering of Workgroups
		//
		// %ITEMY - Number of elements , a workitem processes in Y direction.
		// %ITEMX - Number of elements , a workitem processes in X direction.
		//
		// %V 	- Vectoring Width
		// %PANEL(*) - Panel Width to access Rows of A and Columns of B
		//		   Right now, %V is assumed to be the panel width.
		//		   We dont use %PANEL in the current implementation.
		//
		blockDimY = ((M-1) / (threadsY * %ITEMY)) + 1;
		bidY = ( get_group_id(0) % ( blockDimY));
		bidX = ( get_group_id(0) / ( blockDimY));

		//
		// <row,col> is the left-top of the TILE region
		// in the output C matrix that will be determined
		// by this workgroup
		//
		row =  (bidY * (threadsY * %ITEMY));
		col =  (bidX * (threadsX * %ITEMX));

		//
		// REDColStart, REDColEnd:
		// SYMM Matrix  multiplication proceeds by multiplying panels on A's block-row
		// with panels on B's block-column.
		// However due to symmetric nature of A/B matrix compounded by the fact that
		// only upper OR lower triangle of the symm matrix is available, vector-loads
		// are not possible while traversing certain regions of the matrix.
		// REDColStart, REDColEnd identifies that region in which the panel crosses
		// the diagonal. This region will be the slowest portion of the kernel next to
		// processing the TAIL part.
		//
		#ifdef __SYMM_LEFT__
			REDColStart = row;
			REDColEnd = row  + (threadsY*(%ITEMY));
		#elif defined(__SYMM_RIGHT__)
			REDColStart = col;
			REDColEnd = col + (threadsX*(%ITEMX));
		#endif
		rowA 	= 	row + offsetY;
	   	colB 	= 	(col+offsetX);
		indexC 	= 	(col+offsetX)*ldc + (row + offsetY);
		bool tailBlock = ((row + threadsY*(%ITEMY)) > M) || ((col + threadsX*(%ITEMX)) > N);

		%TYPE%V AVAL[%V][(%ITEMY_BY_V)]; // 8
		%TYPE BVAL[%ITEMX][%V];
		%TYPE%V CVAL[(%ITEMY_BY_V)][%ITEMX];

		%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
		for(uint i=0; i< (%ITEMY_BY_V); i++)
		{
			%IF(%ITEMX) #pragma unroll %ITEMX
			for(uint j=0; j<(%ITEMX); j++)
			{
				CVAL[i][j] = (%TYPE%V) 0;
			}
		}

		uint ACOL=0;
		//
		// 		SYMM
		//
		for(ACOL=0; ((tailBlock == false) && ((ACOL+%V-1) < K)); ACOL += %V /* %PANEL */)
		{

			if ((ACOL+%V-1) < REDColStart)
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
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_FIRST(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_FIRST(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			} else if (ACOL < REDColEnd)
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
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_SECOND(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_SECOND(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			} else {
				//
				// Load B values
				//
				%IF(%ITEMX) #pragma unroll %ITEMX
				for(uint bcol = 0; bcol < %ITEMX; bcol++)
				{
					//
					// PENDING: PANEL iteration to Load the Panel Depth iterating by %V
					//
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_THIRD(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_THIRD(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			}

			%IF(%V) #pragma unroll %V
			for(uint panel=0; panel < %V; panel++)
			{
				%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
				for(uint i=0; i<(%ITEMY_BY_V); i++)
				{
					%IF(%ITEMX) #pragma unroll %ITEMX
					for(uint j=0; j<(%ITEMX); j++)
					{
						%VMAD(CVAL[i][j] ,  AVAL[panel][i] , BVAL[j][panel]);
					}
				}
			}

			#ifdef SYMM_NEEDS_BARRIER
			barrier(CLK_LOCAL_MEM_FENCE);
			#endif
		}

		//
		//  SYMM - 	The Tail....
		//		The tail can wag past M and N. The LOAD routines clamp those accesses
		//
		for(; ACOL < K; ACOL += %V /* %PANEL */)
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
				*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_TAIL(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

				AVAL[c][r] = LOADA_TAIL(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
			}

			%IF(%V) #pragma unroll %V
			for(uint panel=0; panel < %V; panel++)
			{
				%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
				for(uint i=0; i<(%ITEMY_BY_V); i++)
				{
					%IF(%ITEMX) #pragma unroll %ITEMX
					for(uint j=0; j<(%ITEMX); j++)
					{
						%VMAD(CVAL[i][j] ,  AVAL[panel][i] , BVAL[j][panel]);
					}
				}
			}

			#ifdef SYMM_NEEDS_BARRIER
			barrier(CLK_LOCAL_MEM_FENCE);
			#endif
		}


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
				reg = VECTOR_LOAD(C, M, N, ldc, rowA + i*threadsY*%V, colB+(j*threadsX));
				%VMUL(betareg, betav, reg);
				%VMUL(alphareg, alphav, CVAL[i][j]);
				%ADD( reg, betareg, alphareg);
				VECTOR_STORE(reg, C, M, N, ldc, rowA + i*threadsY*%V, colB+(j*threadsX));
			}
		}
		return;
	}
#else
#error COLMAJOR Not Defined while compiling SYMM_C_KERNEL
#endif
";

const char *SYMM_C_KERNEL_WORKING_EXCEPT_CSYMM_PROBLEM = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif
#if !defined(__SYMM_UPPER__) && !defined(__SYMM_LOWER__)
	#error Upper or Lower must be defined
#endif

#if defined(__SYMM_UPPER__) && defined(__SYMM_LOWER__)
	#error Both Upper and Lower cannot be defined together
#endif

#if !defined(__SYMM_LEFT__) && !defined(__SYMM_RIGHT__)
	#error Neither Left nor Right defined
#endif

#if defined(__SYMM_LEFT__) && defined(__SYMM_RIGHT__)
	#error Both LEFT and RIGHT cannot be defined together
#endif

#if defined(__SYMM_COLMAJOR__)
	void VECTOR_STORE(%TYPE%V data, __global %TYPE *A, uint M, uint N, uint lda, uint row, uint col)
	{
		if ( ((row + %V -1) < M) && (col < N) )
		{
			%VSTORE( data, 0, (&A[col*lda + row]));
			return;
		}

		//
		// Construct from SCALAR
		//
		if ((row < M) && (col < N))
		{
			int i=0;
			%TYPE temp[%V];

			//
			// FIXME: ENDIAN ISSUES - Currently for Little endian
			//		  Needs fixes for Big Endian
			//
			*(__private %TYPE%V *)temp = data;

			for(; i< (M-row); i++)
			{
				A[col*lda + row + i] = temp[i];
			}
		}
		return;
	}

	%TYPE%V VECTOR_LOAD(__global %TYPE *A, uint M, uint N, uint lda, uint row, uint col)
	{
		%TYPE temp[%V];
		%TYPE%V retval = (%TYPE%V) 0;

		if ( ((row + %V -1) < M) && (col < N) )
		{
			retval = %VLOAD(0, (&A[col*lda + row]));
			return retval;
		}

		//
		// Construct from SCALAR
		//
		if ((row < M) && (col < N))
		{
			int i=0;

			for(; i< (M-row); i++)
			{
				temp[i] = A[col*lda + row + i];
			}
			for(; i< (%V);  i++)
			{
				temp[i] = 0;
			}
			%VLOADWITHINCX(retval, temp, 1);
		}
		return retval;
	}

	%TYPE%V SYMM_VECTOR_LOAD_USING_SCALAR(__global %TYPE *A, uint M, uint lda, uint row, uint col)
	{
		%TYPE temp[%V];
		%TYPE%V retval;

		for(uint i=0; i< (%V); i++)
		{
			if (((row + i) < M) && (col < M))
			{
				#ifdef __SYMM_UPPER__
				if ((row + i) <= col)
				#else
				if ((row + i) >= col)
				#endif
				{
					temp[i] = A[col*lda + row + i];
				} else {
					temp[i] = A[(row+i)*lda + col];
				}
			} else {
				temp[i] = (%TYPE) 0;
			}
		}
		%VLOADWITHINCX(retval, temp, 1 );
		return retval;
	}

	%TYPE%V SYMM_VECTOR_LOAD(__global %TYPE *A, uint M, uint lda, uint row, uint col)
	{
		%TYPE%V retval = (%TYPE%V) 0;

		bool validAddress = ((row >= M) || (col >=M)) ? false : true;
		bool fullyWithinUpperTriangle = validAddress && ((row + %V -1) <= col);
		bool fullyWithinLowerTriangle = validAddress && (row > col) && ((row + %V -1) < M);
		bool protrudingLowerTriangle  = validAddress && ((row + %V -1) >= M);
		bool inBetweenDiagonal  	  = validAddress && (!fullyWithinUpperTriangle) && (!fullyWithinLowerTriangle) && (!protrudingLowerTriangle);
		if (fullyWithinLowerTriangle || fullyWithinUpperTriangle)
		{
			#ifdef __SYMM_UPPER__
			if (fullyWithinUpperTriangle)
			#else
			if (fullyWithinLowerTriangle)
			#endif
			{
				retval = %VLOAD(0, (&A[(col)*lda + (row)]));
			} else {
				retval = %VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda);
			}
		} else {
			if (protrudingLowerTriangle || inBetweenDiagonal)
			{
				retval = SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, row, col);
			}
		}
		return retval;
	}

	#ifdef __SYMM_LEFT__
	// (A) MxM * (B) MxN
		%TYPE%V LOADA(__global %TYPE *A, uint M, uint K, uint lda, uint row, uint col)
		{
			return SYMM_VECTOR_LOAD(A, M, lda, row, col);
		}
		#ifdef __SYMM_LOWER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_FIRST(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*lda + (row)]))
		#elif defined(__SYMM_UPPER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_FIRST(A,M,K,lda,row,col) 	%VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda)
		#endif
		#define LOADA_SECOND(A,M,K,lda,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(A, M, lda, row, col)
		#ifdef __SYMM_LOWER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_THIRD(A,M,K,lda,row, col)	%VLOADWITHINCXV2(0, (&A[(row)*lda + (col)]), lda)
		#elif defined(__SYMM_UPPER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADA_THIRD(A,M,K,lda,row, col)	%VLOAD(0, (&A[(col)*lda + (row)]))
		#endif
		#define LOADA_TAIL(A,M,K,lda,row,col) 		SYMM_VECTOR_LOAD_USING_SCALAR(A,M,lda,row,col)

		%TYPE%V LOADB(__global %TYPE *B, uint K, uint N, uint ldb, uint row, uint col)
		{
			return VECTOR_LOAD(B, K, N, ldb, row, col );
		}
		#define LOADB_FIRST(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_SECOND(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_THIRD(B,K,N,ldb,row,col) 	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#define LOADB_TAIL(B,K,N,ldb,row,col)	VECTOR_LOAD(B, K, N, ldb, row, col)

	#elif defined(__SYMM_RIGHT__)
		// (A)MxN * (B)NxN
		%TYPE%V LOADA(__global %TYPE *A, uint M, uint K, uint lda, uint row, uint col)
		{
			return VECTOR_LOAD(A, M, K, lda, row, col );
		}
		#define LOADA_FIRST(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_SECOND(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_THIRD(A,M,K,lda,row,col)	%VLOAD(0, (&A[(col)*(lda) + (row)]))
		#define LOADA_TAIL(A,M,K,lda,row,col)	VECTOR_LOAD(A, M, K, lda, row, col)

		%TYPE%V LOADB(__global %TYPE *B, uint K, uint N, uint ldb, uint row, uint col)
		{
			return SYMM_VECTOR_LOAD(B, N, ldb, row, col);
		}
		#ifdef __SYMM_UPPER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_FIRST(B,K,N,ldb,row,col)	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#elif defined(__SYMM_LOWER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_FIRST(B,K,N,ldb,row,col)	%VLOADWITHINCXV2(0, (&B[(row)*(ldb)  + (col)]), ldb)
		#endif
		#define LOADB_SECOND(B,K,N,ldb,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(B, N, ldb, row, col)
		#ifdef __SYMM_UPPER__
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_THIRD(B,K,N,ldb,row,col)	%VLOADWITHINCXV2(0, (&B[(row)*(ldb) + (col)]), ldb)
		#elif defined(__SYMM_LOWER__)
			// CHECK: KPRINTF Behaviour with so many parantheses - If fails, use parantheses in the caller
			#define LOADB_THIRD(B,K,N,ldb,row,col)	%VLOAD(0, (&B[(col)*(ldb) + (row)]))
		#endif
		#define LOADB_TAIL(B,K,N,ldb,row,col)		SYMM_VECTOR_LOAD_USING_SCALAR(B, N,ldb,row,col)
	#endif // Left, Right

	__kernel void symm_C_kernel( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *C,
			       			  	uint M, uint N, uint _lda, uint _ldb, int ldc, %TYPE alpha, %TYPE beta)
	{
		__global %TYPE const *restrict A;
		__global %TYPE const *restrict B;
		uint K;
		uint lda, ldb;
		uint indexA, indexB, indexC;
		uint rowA, colA, rowB, colB, rowC, colC;
		uint numGroupsOnY;
		uint bidX, bidY;
		uint row, col;
		uint REDColStart, REDColEnd; // As the panel traverses these columns, it will slow down - Hence RED.
		uint tid = get_local_id(0);
		int panel;
		uint blockDimY;
	#ifdef __SYMM_LEFT__
		// MxM * MxN
		A = _A;
		lda = _lda;
		B = _B;
		ldb = _ldb;
		K = M;
	#elif defined(__SYMM_RIGHT__)
		// MxN * NxN
		A = _B;
		lda = _ldb;
		B = _A;
		ldb = _lda;
		K = N;
	#endif

		//
		// %WIDTH - Preferably 16
		// %ITEMY, %ITEMX - 1 Thread is responsible for %ITEMY * %ITEMX sub-matrix in C
		//					%ITEMY must be divisible by %V
		// The entire workgroup loops-together to complete ITEMY-ITEMX sub-matrix
		//
		uint threadsY = %WIDTH;
		uint threadsX = get_local_size(0)/threadsY;
		uint offsetY = (tid % threadsY) * %V;
		uint offsetX = (tid / threadsY);

		//
		// Column-Major ordering of Workgroups
		//
		// %ITEMY - Number of elements , a workitem processes in Y direction.
		// %ITEMX - Number of elements , a workitem processes in X direction.
		//
		// %V 	- Vectoring Width
		// %PANEL(*) - Panel Width to access Rows of A and Columns of B
		//		   Right now, %V is assumed to be the panel width.
		//		   We dont use %PANEL in the current implementation.
		//
		blockDimY = ((M-1) / (threadsY * %ITEMY)) + 1;
		bidY = ( get_group_id(0) % ( blockDimY));
		bidX = ( get_group_id(0) / ( blockDimY));

		//
		// <row,col> is the left-top of the TILE region
		// in the output C matrix that will be determined
		// by this workgroup
		//
		row =  (bidY * (threadsY * %ITEMY));
		col =  (bidX * (threadsX * %ITEMX));

		//
		// REDColStart, REDColEnd:
		// SYMM Matrix  multiplication proceeds by multiplying panels on A's block-row
		// with panels on B's block-column.
		// However due to symmetric nature of A/B matrix compounded by the fact that
		// only upper OR lower triangle of the symm matrix is available, vector-loads
		// are not possible while traversing certain regions of the matrix.
		// REDColStart, REDColEnd identifies that region in which the panel crosses
		// the diagonal. This region will be the slowest portion of the kernel next to
		// processing the TAIL part.
		//
		#ifdef __SYMM_LEFT__
			REDColStart = row;
			REDColEnd = row  + (threadsY*(%ITEMY));
		#elif defined(__SYMM_RIGHT__)
			REDColStart = col;
			REDColEnd = col + (threadsX*(%ITEMX));
		#endif
		rowA 	= 	row + offsetY;
	   	colB 	= 	(col+offsetX);
		indexC 	= 	(col+offsetX)*ldc + (row + offsetY);
		bool tailBlock = ((row + threadsY*(%ITEMY)) > M) || ((col + threadsX*(%ITEMX)) > N);

		%TYPE%V AVAL[%V][(%ITEMY_BY_V)]; // 8
		%TYPE BVAL[%ITEMX][%V];
		%TYPE%V CVAL[(%ITEMY_BY_V)][%ITEMX];

		%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
		for(uint i=0; i< (%ITEMY_BY_V); i++)
		{
			%IF(%ITEMX) #pragma unroll %ITEMX
			for(uint j=0; j<(%ITEMX); j++)
			{
				CVAL[i][j] = (%TYPE%V) 0;
			}
		}

		uint ACOL=0;
		//
		// 		SYMM
		//
		for(ACOL=0; ((tailBlock == false) && ((ACOL+%V-1) < K)); ACOL += %V /* %PANEL */)
		{

			if ((ACOL+%V-1) < REDColStart)
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
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_FIRST(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_FIRST(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			} else if (ACOL < REDColEnd)
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
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_SECOND(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_SECOND(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			} else {
				//
				// Load B values
				//
				%IF(%ITEMX) #pragma unroll %ITEMX
				for(uint bcol = 0; bcol < %ITEMX; bcol++)
				{
					//
					// PENDING: PANEL iteration to Load the Panel Depth iterating by %V
					//
					*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_THIRD(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

					AVAL[c][r] = LOADA_THIRD(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
				}
			}

			%IF(%V) #pragma unroll %V
			for(uint panel=0; panel < %V; panel++)
			{
				%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
				for(uint i=0; i<(%ITEMY_BY_V); i++)
				{
					%IF(%ITEMX) #pragma unroll %ITEMX
					for(uint j=0; j<(%ITEMX); j++)
					{
						%VMAD(CVAL[i][j] ,  AVAL[panel][i] , BVAL[j][panel]);
					}
				}
			}

			#ifdef SYMM_NEEDS_BARRIER
			barrier(CLK_LOCAL_MEM_FENCE);
			#endif
		}

		//
		//  SYMM - 	The Tail....
		//		The tail can wag past M and N. The LOAD routines clamp those accesses
		//
		for(; ACOL < K; ACOL += %V /* %PANEL */)
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
				*(__private %TYPE%V *)(&BVAL[bcol]) = LOADB_TAIL(B, K, N , ldb, ACOL, colB + (threadsX*bcol));
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

				AVAL[c][r] = LOADA_TAIL(A, M, K, lda, rowA + r*threadsY*(%V), ACOL + c );
			}

			%IF(%V) #pragma unroll %V
			for(uint panel=0; panel < %V; panel++)
			{
				%IF(%ITEMY_BY_V) #pragma unroll %ITEMY_BY_V
				for(uint i=0; i<(%ITEMY_BY_V); i++)
				{
					%IF(%ITEMX) #pragma unroll %ITEMX
					for(uint j=0; j<(%ITEMX); j++)
					{
						%VMAD(CVAL[i][j] ,  AVAL[panel][i] , BVAL[j][panel]);
					}
				}
			}

			#ifdef SYMM_NEEDS_BARRIER
			barrier(CLK_LOCAL_MEM_FENCE);
			#endif
		}


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
				reg = VECTOR_LOAD(C, M, N, ldc, rowA + i*threadsY*%V, colB+(j*threadsX));
				%VMUL(betareg, betav, reg);
				%VMUL(alphareg, alphav, CVAL[i][j]);
				%ADD( reg, betareg, alphareg);
				VECTOR_STORE(reg, C, M, N, ldc, rowA + i*threadsY*%V, colB+(j*threadsX));
			}
		}
		return;
	}
#else
#error COLMAJOR Not Defined while compiling SYMM_C_KERNEL
#endif
";

