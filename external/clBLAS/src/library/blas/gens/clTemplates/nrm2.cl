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


static const char *nrm2_hypot_kernel = "
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

__kernel void %PREFIXnrm2_hypot_kernel( __global %TYPE *_X, __global %PTYPE *scratchBuff,
                                        uint N, uint offx, int incx )
{
	__global %TYPE *X = _X + offx;

    #ifdef RETURN_ON_INVALID
        // Incase of incx<1, NRM2 will be zero
        if( get_global_id(0) == 0 ) {
            scratchBuff[0] = (%PTYPE)0.0;
        }
        return;
    #endif

    int gOffset;
    %TYPE%V res = (%TYPE%V) 0.0;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1;

        #ifdef INCX_NONUNITY
            %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
        #else
            vReg1 = %VLOAD( 0, (X + gOffset) );
        #endif

        res = hypot( res, vReg1 );
    }
    %TYPE nrm2 = %REDUCE_HYPOT( res );

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1;
        sReg1 = X[gOffset * incx];
        nrm2 = hypot( nrm2, sReg1 );
    }

    // Note: this has to be called outside any if-conditions- because REDUCTION uses barrier
    // dotP of work-item 0 will have the final reduced item of the work-group
    %REDUCTION_BY_HYPOT( nrm2 );

    %PTYPE nrm2_ptype;
    #ifdef COMPLEX
        nrm2_ptype = hypot( nrm2.even, nrm2.odd );
    #else
        nrm2_ptype = nrm2;
    #endif


    if( (get_local_id(0)) == 0 ) {
        scratchBuff[ get_group_id(0) ] = nrm2_ptype;
    }
}
\n";

static const char *nrm2_ssq_kernel = "
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

#define PZERO (%PTYPE)0.0
#define ZERO (%TYPE)0.0
#define VZERO (%TYPE%V)0.0

//
// Same scratch buffer will be used both scale and ssq.
// So a scratch buffer of size 2*N is needed.
// scale will be stored in scratch-buffer from [0] to [get_num_groups(0) - 1]
// ssq will be stored from [get_num_groups(0)] to [2*get_num_groups(0) - 1]
//

__kernel void %PREFIXnrm2_ssq_kernel( __global %TYPE *_X, __global %PTYPE *scratchBuff,
                                        uint N, uint offx, int incx )
{
	__global %TYPE *X = _X + offx;
    uint numWGs = get_num_groups(0);

    #ifdef RETURN_ON_INVALID
        // Incase of incx<1, NRM2 will be zero
        if( get_global_id(0) == 0 ) {
            scratchBuff[0] = PZERO;
            scratchBuff[numWGs] = PZERO;
        }
        return;
    #endif

    // First we find the max element in the whole work-group
    // i.e calculating scale
    %TYPE maxFound = (%TYPE) -MAX;

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
        %TYPE regMax = %REDUCE_MAX( vReg1 );
        maxFound = fmax( maxFound, regMax );
    }

    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1;

        sReg1 = X[gOffset * incx];
        sReg1 = fabs( sReg1 );
        maxFound = fmax( maxFound, sReg1 );
    }

    %REDUCTION_BY_MAX( maxFound );

    __local %PTYPE _scale;

    if( (get_local_id(0)) == 0 ) {
        #ifdef COMPLEX
            _scale = fmax( maxFound.even, maxFound.odd );
        #else
            _scale = maxFound;
        #endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // At this point we have scale.
    // Now we calculate ssq by loading the array again and dividing the
    // elements by scale and squaring it.

    %TYPE ssq = ZERO;
    %PTYPE scaleOfWG = _scale;

    // If scaleOfWG was zero, that means the whole array encountered before was filled with zeroes
    // Note: _scale is a local variable, either all enter or none
    if(isnotequal(scaleOfWG, PZERO))
    {
        for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
        {
            %TYPE%V vReg1;

            #ifdef INCX_NONUNITY
                %VLOADWITHINCX( vReg1, (X + (gOffset*incx)), incx);
            #else
                vReg1 = %VLOAD( 0, (X + gOffset) );
            #endif

            vReg1 = fabs( vReg1 );
            %TYPE%V tempSsq = (vReg1 / scaleOfWG) * (vReg1 / scaleOfWG);

            ssq += %REDUCE_SUM( tempSsq );
        }

        for( ; gOffset<N; gOffset++ )
        {
            %TYPE sReg1;

            sReg1 = X[gOffset * incx];
            sReg1 = fabs( sReg1 );

            ssq += (sReg1 / scaleOfWG) * (sReg1 / scaleOfWG);
        }

        %REDUCTION_BY_SUM( ssq );
    }

    if( (get_local_id(0)) == 0 ) {
        scratchBuff[ get_group_id(0) ] = scaleOfWG;

        #ifdef COMPLEX
            scratchBuff[ numWGs + get_group_id(0) ] = ssq.even + ssq.odd;
        #else
            scratchBuff[ numWGs + get_group_id(0) ] = ssq;
        #endif
    }
}
\n";

