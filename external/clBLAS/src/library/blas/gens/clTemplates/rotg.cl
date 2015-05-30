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



static const char *rotg_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define ZERO (%TYPE)0.0
#define PZERO (%PTYPE)0.0

// CABS(A) returns SQRT(REALPART(A)**2+IMAGPART(A)**2) -- opencl function length() computes the same
#define CABS( arg )  length( arg )

__kernel void %PREFIXrotg_kernel( __global %TYPE *_A, __global %TYPE *_B, __global %PTYPE *_C,
                                __global %TYPE *_S, uint offa, uint offb, uint offc, uint offs )
{
    %TYPE Areg, Breg, Sreg;
    %PTYPE Creg;

	Areg = _A[offa];
	Breg = _B[offb];

	if(get_global_id(0) == 0)       // Only 1 thread will work
	{
	    #ifndef COMPLEX         // Real and complex math for rotg are different according to netlib
	        %TYPE R, Z, roe, scale, absA, absB;

	        absA = fabs(Areg);
	        absB = fabs(Breg);

	        roe = (isgreater(absA, absB))? Areg: Breg;
	        scale = absA + absB;

	        if(isequal(scale, ZERO))
	        {
	            Creg = 1.0;
	            Sreg = ZERO;
	            R = ZERO;
	            Z = ZERO;
	        }
	        else
	        {
	            // R = scale * sqrt( pown((Areg/scale), 2) + pown((Breg/scale), 2) );
	            // gentype hypot (gentype x, gentype y) -- Computes the value of the
	            //          square root of x2+ y2 without undue overflow or underflow.
	            R = scale * hypot( (Areg/scale), (Breg/scale) );
	            R = (isless(roe, ZERO))? (-R): R;
	            Creg = Areg / R;
	            Sreg = Breg / R;
	            Z = (isgreater(absA, absB))? Sreg:
	                    ( (isnotequal(Creg, ZERO))? (1.0/Creg): 1.0 );
	        }
	        _A[offa] = R;
	        _B[offb] = Z;
	        _C[offc] = Creg;
	        _S[offs] = Sreg;
	    #else           // For comlpex type
	        %TYPE alpha, temp;
	        %PTYPE norm, scale, cabsA, cabsB;

	        cabsA = CABS(Areg);
	        cabsB = CABS(Breg);

	        if(isequal(cabsA, PZERO))
	        {
	            Creg = PZERO;
	            Sreg = (%TYPE)(1.0, 0.0);
	            Areg = Breg;
	        }
	        else
	        {
	            scale = cabsA + cabsB;
	            // norm = scale * sqrt( pown( CABS(Areg/scale), 2 ) + pown( CABS(Breg/scale), 2 ) );
	            norm = scale * hypot( CABS(Areg/scale), CABS(Breg/scale) );
	            alpha = Areg / cabsA;
	            Creg = cabsA / norm;

	            temp = Breg;
	            %CONJUGATE(1, temp);
	            %MUL( Sreg, alpha, temp );
	            Sreg = Sreg / norm;

	            Areg = alpha * norm;
	        }
	        _C[offc] = Creg;
	        _S[offs] = Sreg;
	        _A[offa] = Areg;
	    #endif      // COMPLEX
    }
}
\n";

