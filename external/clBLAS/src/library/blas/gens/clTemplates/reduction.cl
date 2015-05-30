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



static const char *red_sum_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXred_sum_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global %TYPE *res = _res + offRes;
    %TYPE redVal = (%TYPE) 0.0;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V reg1;
        reg1 = %VLOAD( 0, (X + gOffset) );
        redVal +=  %REDUCE_SUM( reg1 );
        }
    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        redVal += X[gOffset];
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    %REDUCTION_BY_SUM( redVal );

    if( (get_local_id(0)) == 0 ) {
        res[0] = redVal;
    }
}
\n";


static const char *red_max_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
    #define MAX 0x1.fffffffffffffp1023      // Max in case of d/z (values from khronos site)
#else
    #define MAX 0x1.fffffep127f             // Max in case of s/c
#endif

__kernel void %PREFIXred_max_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global %TYPE *res = _res + offRes;
    %TYPE redVal = (%TYPE) - MAX;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V reg1;
        reg1 = %VLOAD( 0, (X + gOffset) );
        %TYPE scalarMax = %REDUCE_MAX( reg1 );
        redVal =  fmax( redVal, scalarMax );
        }
    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        redVal = fmax( redVal, X[gOffset] );
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    %REDUCTION_BY_MAX( redVal );

    if( (get_local_id(0)) == 0 ) {
        res[0] = redVal;
    }
}
\n";

static const char *red_min_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif

    #define MAX 0x1.fffffffffffffp1023      // Max in case of d/z (values from khronos site)
#else
    #define MAX 0x1.fffffep127f             // Max in case of s/c
#endif

__kernel void %PREFIXred_min_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global %TYPE *res = _res + offRes;
    %TYPE redVal = (%TYPE) MAX;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V reg1;
        reg1 = %VLOAD( 0, (X + gOffset) );
        %TYPE scalarMin = %REDUCE_MIN( reg1 );
        redVal =  fmin( redVal, scalarMin );
        }
    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        redVal = fmin( redVal, X[gOffset] );
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    %REDUCTION_BY_MIN( redVal );

    if( (get_local_id(0)) == 0 ) {
        res[0] = redVal;
    }
}
\n";


static const char *red_with_index_kernel = "

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


__kernel void %PREFIXred_with_index_kernel( __global %TYPE *_X, __global uint *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global uint *XIndex = (__global uint*)(&X[N]);
    __global uint *res = _res + offRes;
    %TYPE maxVal = (%TYPE)MIN, val = (%TYPE)MIN;
    uint maxIndex = 0, index = 0;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1 = %VLOAD( 0, (X + gOffset) );

        %REDUCE_MAX(vReg1,val,index,1); // Find max within a vector
        if(val > maxVal)
        {
            maxVal = val;
            maxIndex = XIndex[(gOffset + index)];
    }
    }
    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sreg1 = X[gOffset];
        if(sreg1 > maxVal)
        {
            maxVal = sreg1;
            maxIndex = XIndex[gOffset];
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
        res[0] = maxIndex;
    }
}
\n";


static const char *red_hypot_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

__kernel void %PREFIXred_hypot_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global %TYPE *res = _res + offRes;
    %TYPE redVal = (%TYPE) 0.0;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V reg1;
        reg1 = %VLOAD( 0, (X + gOffset) );
        %TYPE scalarHypot = %REDUCE_HYPOT( reg1 );
        redVal =  hypot( redVal, scalarHypot );
    }
    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        redVal = hypot( redVal, X[gOffset] );
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    %REDUCTION_BY_HYPOT( redVal );

    if( (get_local_id(0)) == 0 ) {
        res[0] = redVal;
    }
}
\n";

static const char *red_ssq_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
    #define MAX 0x1.fffffffffffffp1023      // Max in case of d/z (values from khronos site)
#else
    #define MAX 0x1.fffffep127f             // Max in case of s/c
#endif

#define ZERO (%TYPE)0.0

// Since scale & ssq are always of primitive type,
// This kernel will always be called only for float/double

__kernel void %PREFIXred_ssq_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                    uint N, uint offx, uint offRes )
{
 	__global %TYPE *X = _X + offx;
    __global %TYPE *res = _res + offRes;
    %TYPE scale = -MAX;

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V scale1;
        scale1 = %VLOAD( 0, (X + gOffset) );

        %TYPE regMax = %REDUCE_MAX( scale1 );
        scale = fmax( scale, regMax );
    }

    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg;
        sReg = X[gOffset];
        scale = fmax( scale, sReg );
    }

    %REDUCTION_BY_MAX( scale );

    __local %TYPE _scaleOfWG;

    if( (get_local_id(0)) == 0 ) {
        _scaleOfWG = scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // At this point we have scale.
    // Now we calculate ssq by loading the array again and dividing the
    // elements by scale and squaring it.

    %TYPE ssq = (%TYPE) 0.0;
    %TYPE scaleOfWG = _scaleOfWG;

    // If scale was zero, that means the whole array encountered before was filled with zeroes
    // Note: scale is a local variable, either all enter or none
    if(isnotequal(scaleOfWG, ZERO))
    {
        for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
        {
            %TYPE%V scale1, ssq1;
            scale1 = %VLOAD( 0, (X + gOffset) );
            ssq1 = %VLOAD( 0, (X + gOffset + N) );

            %TYPE%V tempSsq = (scale1 / scaleOfWG) * (scale1 / scaleOfWG) * ssq1;

            ssq += %REDUCE_SUM( tempSsq );
        }

        for( ; gOffset<N; gOffset++ )
        {
            %TYPE scale1, ssq1;
            scale1 = X[gOffset];
            ssq1 = X[gOffset + N];

            ssq += (scale1 / scaleOfWG) * (scale1 / scaleOfWG) * ssq1;
        }

        %REDUCTION_BY_SUM( ssq );
    }

    if( (get_local_id(0)) == 0 ) {
        res[0] = scaleOfWG * sqrt(ssq);
    }
}
\n";

