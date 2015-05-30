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

static const char *iamax_kernel = "
#pragma OPENCL EXTENSION cl_amd_printf:enable
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
    #define MIN 0x1.0p-1022         // Min in case of d/z (values from khronos site)
#else
    #define MIN 0x1.0p-126f         // Min in case od s/c
#endif
/******************************************************
 *  Implementations available for REDUCTION_BY_MAX
     0 - ATOMIC_FLI
     1 - REG_FLI,
     2 - ATOMIC_FHI,
     3 - REG_FHI

    Implementation available for REDUCE_MAX
    0 - FHI
    1 - FLI
 ***************************************************/

__kernel void i%PREFIXamax_kernel( __global %TYPE *_X, __global %PTYPE *_scratchBuf,
                                        uint N, uint offx, int incx)
{
	__global %TYPE *X = _X + offx;
    __global %PTYPE *scratchBufVal = _scratchBuf;
    int numGrps = get_num_groups(0);
    __global uint *scratchBufIndex = (__global uint*)(&_scratchBuf[numGrps]);

    #ifdef RETURN_ON_INVALID
        // Incase of incx<1, index will be zero
        if( get_global_id(0) == 0 ) {
            scratchBufVal[0] = (%PTYPE)0.0;
            scratchBufIndex[0] = 0;
        }
        return;
    #endif

    %PTYPE maxVal = MIN, val = MIN;
    uint index = 0, maxIndex = 0;
    %TYPE%V vReg1;
    %PTYPE%V pReg1;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        #ifdef INCX_NONUNITY
            %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
        #else
            vReg1 = %VLOAD( 0, (X + (gOffset * incx)) );
        #endif

        pReg1 = %VABS(vReg1);

        %REDUCE_MAX(pReg1,val,index,1); // Find max within a vector

        if(val > maxVal)
        {
            maxVal = val;
            maxIndex = (gOffset + index);
        }
    }

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1;
        sReg1 = X[gOffset * incx];
        if(%VABS(sReg1) > maxVal)
        {
            maxVal = %VABS(sReg1);
            maxIndex = gOffset;
        }
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
#ifdef REDUCE_MAX_WITH_INDEX_ATOMICS
    %REDUCTION_BY_MAX(maxVal,maxIndex,0);
#else
    %REDUCTION_BY_MAX(maxVal,maxIndex,1);
#endif

    if(get_local_id(0) == 0)
    {
        scratchBufVal[get_group_id(0)] = maxVal;
        scratchBufIndex[get_group_id(0)] = maxIndex + 1; // because 0 is reserved for error
    }
}";
