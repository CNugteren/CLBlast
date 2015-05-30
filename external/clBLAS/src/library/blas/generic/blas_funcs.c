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


#include <blas_funcs.h>

int
funcBlasLevel(BlasFunctionID funcID)
{
    switch(funcID)
    {
        case CLBLAS_SWAP:
        case CLBLAS_SCAL:
        case CLBLAS_COPY:
        case CLBLAS_AXPY:
        case CLBLAS_DOT:
        case CLBLAS_REDUCTION_EPILOGUE:
        case CLBLAS_ROTG:
        case CLBLAS_ROTMG:
        case CLBLAS_ROT:
        case CLBLAS_ROTM:
        case CLBLAS_iAMAX:
        case CLBLAS_NRM2:
        case CLBLAS_ASUM:
                            return 1;

        case CLBLAS_GEMV:
        case CLBLAS_SYMV:
        case CLBLAS_TRMV:
        case CLBLAS_TRSV:
        case CLBLAS_TRSV_GEMV:
        case CLBLAS_HEMV:
        case CLBLAS_SYR:
        case CLBLAS_SYR2:
        case CLBLAS_GER:
        case CLBLAS_HER:
        case CLBLAS_HER2:
        case CLBLAS_TPMV:
        case CLBLAS_SPMV:
        case CLBLAS_HPMV:
        case CLBLAS_TPSV:
        case CLBLAS_SPR:
        case CLBLAS_SPR2:
        case CLBLAS_HPR:
        case CLBLAS_HPR2:
        case CLBLAS_GBMV:
        case CLBLAS_TBMV:
        case CLBLAS_SBMV:
        case CLBLAS_HBMV:
        case CLBLAS_TBSV:
                            return 2;

        default:            return 3;
    }
}

bool
funcHasBeta(BlasFunctionID funcID)
{
    return !funcHasTriangMatrix(funcID);
}

bool
funcHasTriangMatrix(BlasFunctionID funcID)
{
    bool ret = false;

    switch (funcID) {
    // go through
    case CLBLAS_TRMM:
    case CLBLAS_TRSM:
	case CLBLAS_TRMV:
	case CLBLAS_HEMV:
	case CLBLAS_TRSV:
        ret = true;
        break;
    default:
        /* do nothing */
        break;
    }

    return ret;
}
