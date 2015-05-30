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



static const char *rotmg_kernel = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

// Rotmg exists only for S/D
#define ZERO    (%TYPE)0.0
#define ONE     (%TYPE)1.0
#define TWO     (%TYPE)2.0

#define GAM     (%TYPE)4096.0
#define GAMSQ   (%TYPE)( GAM * GAM )
#define RGAMSQ  (%TYPE)( 1.0 / GAMSQ )

__kernel void %PREFIXrotmg_kernel( __global %TYPE *_D1, __global %TYPE *_D2, __global %TYPE *_X1,
                                __global %TYPE *_Y1, __global %TYPE *_param,
                                uint offD1, uint offD2, uint offX1, uint offY1, uint offParam )
{
	%TYPE D1, D2, X1, Y1;
	%TYPE flag, H11, H12, H21, H22;                 // elements of PARAM
	__global %TYPE *param = _param + offParam;

    if(get_global_id(0) == 0)       // Only 1 thread will work
	{
        %TYPE P1, P2, Q1, Q2, temp, U;

        D1 = _D1[offD1];
        D2 = _D2[offD2];
        X1 = _X1[offX1];
        Y1 = _Y1[offY1];

        if(isless(D1, ZERO))
        {
            flag = -ONE;
            H11 = ZERO;
            H12 = ZERO;
            H21 = ZERO;
            H22 = ZERO;
            D1 = ZERO;
            D2 = ZERO;
            X1 = ZERO;
        }
        else                                // CASE D1 NONNEGATIVE
        {
            P2 = D2 * Y1;
            if(isequal(P2, ZERO))
            {
                flag = -TWO;
                param[0] = flag;
                return;
            }
            // Regular case
            P1 = D1 * X1;
            Q2 = P2 * Y1;
            Q1 = P1 * X1;

            if(isgreater( fabs(Q1), fabs(Q2) ))
            {
                H21 = -Y1 / X1;
                H12 = P2 / P1;
                U = ONE - (H12 * H21);

                if(isgreater( U, ZERO ))
                {
                    flag = ZERO;
                    D1 = D1 / U;
                    D2 = D2 / U;
                    X1 = X1 * U;
                }
            }
            else
            {
                if(isless(Q2, ZERO))
                {
                    flag = -ONE;
                    H11 = ZERO;
                    H12 = ZERO;
                    H21 = ZERO;
                    H22 = ZERO;
                    D1 = ZERO;
                    D2 = ZERO;
                    X1 = ZERO;
                }
                else
                {
                    flag = ONE;
                    H11 = P1 / P2;
                    H22 = X1 / Y1;
                    U = ONE + (H11 * H22);
                    temp = D2 / U;
                    D2 = D1 / U;
                    D1 = temp;
                    X1 = Y1 * U;
                }
            }
            if(isnotequal(D1, ZERO))
            {
                while(isless(D1, RGAMSQ) || isgreater(D1, GAMSQ))
                {
                    if(isequal(flag, ZERO))
                    {
                        H11 = ONE;
                        H22 = ONE;
                        flag = -ONE;
                    }
                    else
                    {
                        H21 = -ONE;
                        H12 = ONE;
                        flag = -ONE;
                    }
                    if(isless(D1, RGAMSQ))
                    {
                        D1 = D1 * GAMSQ;
                        X1 = X1 / GAM;
                        H11 = H11 / GAM;
                        H12 = H12 / GAM;
                    }
                    else
                    {
                        D1 = D1 / GAMSQ;
                        X1 = X1 * GAM;
                        H11 = H11 * GAM;
                        H12 = H12 * GAM;
                    }
                }   // End of while
            }

            if(isnotequal(D2, ZERO))
            {
                while(isless( fabs(D2), RGAMSQ ) || isgreater( fabs(D2), GAMSQ ))
                {
                    if(isequal(flag, ZERO))
                    {
                        H11 = ONE;
                        H22 = ONE;
                        flag = -ONE;
                    }
                    else
                    {
                        H21 = -ONE;
                        H12 = ONE;
                        flag = -ONE;
                    }
                    if(isless( fabs(D2), RGAMSQ ))
                    {
                        D2 = D2 * GAMSQ;
                        H21 = H21 / GAM;
                        H22 = H22 / GAM;
                    }
                    else
                    {
                        D2 = D2 / GAMSQ;
                        H21 = H21 * GAM;
                        H22 = H22 * GAM;
                    }
                }   // End of while
            }
        }

        if(isless(flag, ZERO))
        {
            param[1] = H11;
            param[2] = H21;
            param[3] = H12;
            param[4] = H22;
        }
        else if(isequal(flag, ZERO))
        {
            param[2] = H21;
            param[3] = H12;
        }
        else
        {
            param[1] = H11;
            param[4] = H22;
        }

        param[0] = flag;
        _D1[offD1] = D1;
        _D2[offD2] = D2;
        _X1[offX1] = X1;
    }   // global_id(0) == 0
}
\n";

