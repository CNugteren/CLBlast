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



static const char *rotm_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

#define ZERO    (%TYPE)0.0
#define ONE     (%TYPE)1.0
#define TWO     (%TYPE)2.0

__kernel void %PREFIXrotm_kernel( __global %TYPE *_X, __global %TYPE *_Y, uint N,
                                uint offx, int incx, uint offy, int incy
#ifndef DO_ROT
                                , __global %TYPE *_param, uint offParam             // Rotm parameters
#else
                                , %PTYPE C,  %PTYPE S                               // Rot parameters
#endif
                                )
{
	__global %TYPE *X = _X + offx;
	__global %TYPE *Y = _Y + offy;

    if ( incx < 0 ) {
        X = X + (N - 1) * abs(incx);
    }
    if ( incy < 0 ) {
        Y = Y + (N - 1) * abs(incy);
    }

    %PTYPE H11, H21, H12, H22, flag;    // All these are of PTYPE for rot and rotm

    #ifndef DO_ROT
    // Incase of Rotm
        flag = _param[offParam];
        H11 = _param[offParam+1];
        H21 = _param[offParam+2];
        H12 = _param[offParam+3];
        H22 = _param[offParam+4];

        (flag == (ZERO))? (H11 = ONE, H22 = ONE)                            : 1;    // 1 is dummy here to avoid compilation error
        (flag == (ONE) )? (H21 = -ONE, H12 = ONE)                           : 1;
        (flag == (-TWO))? (H11 = ONE, H21 = ZERO, H12 = ZERO, H22 = ONE)    : 1;
    #else   // ROT
        H11 = C;
        H12 = S;
        H21 = -S;
        H22 = C;
    #endif

    int gOffset;
    for( gOffset=(get_global_id(0) * %V); (gOffset + %V - 1)<N; gOffset+=( get_global_size(0) * %V ) )
    {
        %TYPE%V vReg1, vReg2, temp;

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

        temp = (vReg1 * H11) + (vReg2 * H12);
        vReg2 = (vReg1 * H21) + (vReg2 * H22);

        #ifdef INCX_NONUNITY
            %VSTOREWITHINCX( (X + (gOffset * incx)), temp, incx );
        #else
            %VSTORE( temp, 0 ,(X + (gOffset * incx)) );
        #endif

        #ifdef INCY_NONUNITY
            %VSTOREWITHINCX( (Y + (gOffset * incy)), vReg2, incy );
        #else
            %VSTORE( vReg2, 0 ,(Y + (gOffset * incy)) );
        #endif
    }

    // Loop for the last thread to handle the tail part of the vector
    // Using the same gOffset used above
    for( ; gOffset<N; gOffset++ )
    {
        %TYPE sReg1, sReg2, temp;
        sReg1 = X[gOffset * incx];
        sReg2 = Y[gOffset * incy];

        temp = (sReg1 * H11) + (sReg2 * H12);
        sReg2 = (sReg1 * H21) + (sReg2 * H22);

        X[gOffset * incx] = temp;
        Y[gOffset * incy] = sReg2;
        }
}
\n";

