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

static const char *SYMM_HEMM_HELPER = "
#ifdef DOUBLE_PRECISION
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
#endif

%TYPE SYMM_SCALAR_LOAD(__global %TYPE const * restrict A, uint M, uint lda, uint row, uint col)
{
	%TYPE retval;

    //PENDING: Remove this Check for M. This will never happen
	if (((row) < M) && (col < M))
	{
		#ifdef __SYMM_UPPER__
		if ((row) <= col)
		#else
		if ((row) >= col)
		#endif
		{
			retval = A[(col)*lda + row];
            #ifdef __HEMM__
            if (row == col) { retval.odd = 0; }
            #endif
		} else {
			retval = A[(row)*lda + col];
            #ifdef __HEMM__
            %CONJUGATE(1, retval);
            #endif
		}
	} else {
		retval = (%TYPE) 0;
	}
	return retval;
}

%TYPE%V SYMM_VECTOR_LOAD_USING_SCALAR(__global %TYPE const * restrict A, uint M, uint lda, uint row, uint col)
{
	//%TYPE symm_vec_load_temp[%V];
	%TYPE%V symm_vec_retval;

    //#pragma unroll %V
	//for(uint index_i=0; index_i< (%V); index_i++)
    %VFOR
	{
        //PENDING: Remove this Check for M. This will never happen
		if (((row + %VFORINDEX) < M) && (col < M))
		{
			#ifdef __SYMM_UPPER__
			if ((row + %VFORINDEX) <= col)
			#else
			if ((row + %VFORINDEX) >= col)
			#endif
			{
				//symm_vec_load_temp[index_i] = A[(col)*(lda) + ((row) + index_i)];
				symm_vec_retval%VFORSUFFIX = A[(col)*(lda) + ((row) + %VFORINDEX)];
                #ifdef __HEMM__
                //if ((row + index_i) == col) { symm_vec_load_temp[index_i].odd = 0; }
                if ((row + %VFORINDEX) == col) { (symm_vec_retval%VFORSUFFIX).odd = 0; }
                #endif
			} else {
				//symm_vec_load_temp[index_i] = A[((row)+index_i)*(lda) + (col)];
				symm_vec_retval%VFORSUFFIX = A[((row)+ %VFORINDEX )*(lda) + (col)];
                #ifdef __HEMM__
                //CONJUGATE(1, (symm_vec_load_temp[index_i]));
                {
                    %TYPE SCALAR;

                    SCALAR = symm_vec_retval%VFORSUFFIX;
                    %CONJUGATE(1, SCALAR);
                    symm_vec_retval%VFORSUFFIX = SCALAR;
                }
                #endif
			}
		} else {
			//symm_vec_load_temp[index_i] = (%TYPE) 0;
			symm_vec_retval%VFORSUFFIX = (%TYPE) 0;
		}
	}
	//%VLOADWITHINCX(symm_vec_retval, symm_vec_load_temp, 1 );
    //symm_vec_retval = *(__private %TYPE%V *)symm_vec_load_temp;
	return symm_vec_retval;
}
\n";
