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



static const char *swap_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXswap_kernel( __global %TYPE *_X, __global %TYPE *_Y, uint N, uint offx, int incx, uint offy, int incy )
{
	__global %TYPE *X = _X + offx;
	__global %TYPE *Y = _Y + offy;

    if ( incx < 0 ) {
        X = X + (N - 1) * abs(incx);
    }
    if ( incy < 0 ) {
        Y = Y + (N - 1) * abs(incy);
    }

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1, vReg2;

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

        #ifdef INCX_NONUNITY
            %VSTOREWITHINCX( (X + (gOffset * incx)), vReg2, incx );
        #else
            %VSTORE( vReg2, 0 ,(X + (gOffset * incx)) );
        #endif

        #ifdef INCY_NONUNITY
            %VSTOREWITHINCX( (Y + (gOffset * incy)), vReg1, incy );
        #else
            %VSTORE( vReg1, 0 ,(Y + (gOffset * incy)) );
        #endif
    }

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1, sReg2;
        sReg1 = X[gOffset * incx];
        sReg2 = Y[gOffset * incy];

        X[gOffset * incx] = sReg2;
        Y[gOffset * incy] = sReg1;
        }
}
\n";

