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



static const char *scal_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXscal_kernel( %TYPE alpha, __global %TYPE *_X, uint N, uint offx, int incx )
{
    if(incx < 0) {
        return;
    }

	__global %TYPE *X = _X + offx;
    uint global_offset = get_global_id(0) * %V;
    bool isVectorWI = ((global_offset + (%V-1)) < N) && (incx == 1);

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1, temp;

        #ifdef INCX_NONUNITY
            %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
        #else
            vReg1 = %VLOAD( 0, (X + gOffset) );
        #endif

        %VMUL( temp, vReg1, alpha );

        #ifdef INCX_NONUNITY
            %VSTOREWITHINCX( (X + (gOffset * incx)), temp, incx );
        #else
            %VSTORE( temp, 0 ,(X + (gOffset * incx)) );
        #endif
    }

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1, temp;
        sReg1 = X[gOffset * incx];
        %MUL( temp, sReg1, alpha );
        X[gOffset * incx] = temp;
        }
}
\n";

