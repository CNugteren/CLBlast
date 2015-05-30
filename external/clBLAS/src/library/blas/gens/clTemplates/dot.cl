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



static const char *dot_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXdot_kernel( __global %TYPE *_X, __global %TYPE *_Y, __global %TYPE *scratchBuff,
                                        uint N, uint offx, int incx, uint offy, int incy, int doConj )
{
	__global %TYPE *X = _X + offx;
	__global %TYPE *Y = _Y + offy;
    %TYPE dotP = (%TYPE) 0.0;

    if ( incx < 0 ) {
        X = X + (N - 1) * abs(incx);
    }
    if ( incy < 0 ) {
        Y = Y + (N - 1) * abs(incy);
    }

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1, vReg2, res;

        #ifdef INCX_NONUNITY
            %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
        #else
            vReg1 = %VLOAD( 0, (X + gOffset) );
        #endif

        #ifdef INCY_NONUNITY
            %VLOADWITHINCX( vReg2, (Y + (gOffset*incy)), incy);
        #else
            vReg2 = %VLOAD( 0, (Y + gOffset) );
        #endif

        %CONJUGATE(doConj, vReg1);
        %VMUL( res, vReg1, vReg2 );
        dotP += %REDUCE_SUM( res );          // Add-up elements in the vector to give a scalar
    }

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1, sReg2, res;
        sReg1 = X[gOffset * incx];
        sReg2 = Y[gOffset * incy];

        %CONJUGATE(doConj, sReg1);
            %MUL( res, sReg1, sReg2 );
            %ADD( dotP, dotP, res );
        }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    // dotP of work-item 0 will have the final reduced item of the work-group
    %REDUCTION_BY_SUM( dotP );

    if( (get_local_id(0)) == 0 ) {
        scratchBuff[ get_group_id(0) ] = dotP;
    }
}
\n";

