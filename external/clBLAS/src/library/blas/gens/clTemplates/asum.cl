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

static const char *asum_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXasum_kernel( __global %TYPE *_X, __global %PTYPE *scratchBuff, uint N, uint offx, int incx)
{
	__global %TYPE *X = _X + offx;
    %TYPE asum = (%TYPE) 0.0;

    #ifdef INCX_NEGATIVE
        if( get_global_id(0) == 0 ) {
            scratchBuff[0] = (%PTYPE)0.0;
        }
        return;
    #endif


    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1;

        #ifdef INCX_NONUNITY
            %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
        #else
            vReg1 = %VLOAD( 0, (X + gOffset) );
        #endif
        vReg1 = fabs( vReg1 );

        asum += %REDUCE_SUM( vReg1 );          // Add-up elements in the vector to give a scalar
    }

    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1 = X[gOffset * incx];
        sReg1 = fabs( sReg1 );
        //%TYPE res;
        %ADD( asum, asum, sReg1 );
    }

    %REDUCTION_BY_SUM(asum);

    %PTYPE answer;

    #ifdef COMPLEX
        answer = asum.even + asum.odd;
    #else
        answer = asum;
    #endif


    if( (get_local_id(0)) == 0 ) {
        scratchBuff[ get_group_id(0) ] = answer;
    }
}
\n";

