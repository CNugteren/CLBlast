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

#include "tuned_numbers.h"

#define USE_TUNED_NUMBERS

typedef enum callType
{
    GEMM_NN_CALL,    // A-Non trans, B-Non trans
    GEMM_NT_CALL,    // A-Non trans, B-Trans
    GEMM_TN_CALL,    // A-Trans, B-Non trans
    GEMM_TT_CALL,    // A-Trans, B-Trans

    HERK_UN_CALL,    // Upper, Non-trans
    HERK_UC_CALL,    // Upper, Conj-trans
    HERK_LN_CALL,    // Lower, Non-trans
    HERK_LC_CALL,    // Lower, Conj-trans

    SYMM_LU_CALL,   // Left, Upper
    SYMM_RU_CALL,   // Right, Upper
    SYMM_LL_CALL,   // Left, Lower
    SYMM_RL_CALL,   // Right, Lower

    HEMM_LU_CALL,   // Left, Upper
    HEMM_RU_CALL,   // Right, Upper
    HEMM_LL_CALL,   // Left, Lower
    HEMM_RL_CALL,   // Right, Lower

    NUM_CALL_TYPES

} callType;




blockSizes bestBlockSizeForDevice( SolutionStep *step )
{
    blockSizes temp;
    callType currCall;
    CLBlasKargs *kargs = &(step->args);
    TargetDevice *kDevice = &(step->device);
    size_t maxWGSize;


///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// QUICK FIX: changing code using fast regex search-replace:
	// Removing the tagged array-of-structs initialization  - which works only with gcc
	// moving the global static variable locally and assiging the values as individual statements
	// this is not thread-safe; fix-this if thread safety is needed


	static blockSizes bestBlockSizes [NUM_DEVICE_CHIPS][4][NUM_CALL_TYPES];         // [NUM_DEVICE_CHIPS][NUM_DATATYPES][NUM_CALL_TYPES]

	// Block sizes for unknows devices -- using default numbers

	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][GEMM_TN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][GEMM_TT_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][GEMM_TN_CALL] = t; }
	{ blockSizes t = { 8, 16, 2, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][GEMM_TT_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][GEMM_TN_CALL] = t; }
	{ blockSizes t = { 8, 16, 2, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][GEMM_TT_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][GEMM_TN_CALL] = t; }
	{ blockSizes t = { 8, 16, 1, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][GEMM_TT_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HERK_UN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HERK_UC_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HERK_LN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HERK_LC_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HERK_UN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HERK_UC_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HERK_LN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HERK_LC_CALL] = t; }

	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 8, 8, 4, 1 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_FLOAT][HEMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CHIP_UNKNOWN][TYPE_COMPLEX_DOUBLE][HEMM_RL_CALL] = t; }

	#ifdef USE_TUNED_NUMBERS

	// Block sizes for Cayman
	{ blockSizes t = { 32, 4, 4, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 8, 1 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 2, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 2, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 2, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 1 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 4, 16, 8, 2, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][HERK_UN_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 32, 4, 4, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 8, 0 }; bestBlockSizes[CAYMAN][TYPE_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 4, 0 }; bestBlockSizes[CAYMAN][TYPE_COMPLEX_FLOAT][HEMM_RU_CALL] = t; }

	// Block sizes for Tahiti
	{ blockSizes t = { 32, 8, 4, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 32, 8, 8, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 4, 1 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 4, 1 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 1 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 4, 1 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 2, 1 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 2, 1 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HERK_UN_CALL] = t; }
	{ blockSizes t = { 4, 16, 8, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HERK_UC_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HERK_LN_CALL] = t; }
	{ blockSizes t = { 8, 32, 8, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HERK_LC_CALL] = t; }

	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HERK_UN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HERK_UC_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HERK_LN_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HERK_LC_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 8, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 32, 8, 8, 4, 0 }; bestBlockSizes[TAHITI][TYPE_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 4, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_FLOAT][HEMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 2, 0 }; bestBlockSizes[TAHITI][TYPE_COMPLEX_DOUBLE][HEMM_RL_CALL] = t; }

	// Block-sizes for Cypress
	{ blockSizes t = { 32, 8, 4, 8, 1 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 8, 1 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 8, 1 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 16, 8, 4, 1 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 4, 16, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 8, 32, 4, 4, 1 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 16, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 8, 32, 8, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HERK_UN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HERK_UC_CALL] = t; }
	{ blockSizes t = { 8, 16, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HERK_LN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HERK_LC_CALL] = t; }

	{ blockSizes t = { 8, 8, 8, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 8, 16, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 8, 16, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 8, 8, 8, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 8, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 8, 8, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 8, 8, 4, 8, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 32, 4, 8, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 4, 32, 4, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 32, 4, 8, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_FLOAT][HEMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 32, 4, 8, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 4, 16, 4, 4, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 32, 4, 8, 2, 0 }; bestBlockSizes[CYPRESS][TYPE_COMPLEX_DOUBLE][HEMM_RL_CALL] = t; }

	// Block-sizes for GeForce GTX 580
	{ blockSizes t = { 16, 32, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 32, 4, 8, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 32, 16, 8, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 16, 8, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 16, 32, 8, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 32, 16, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][GEMM_NT_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][GEMM_TN_CALL] = t; }

	{ blockSizes t = { 16, 32, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][GEMM_NN_CALL] = t; }
	{ blockSizes t = { 32, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][GEMM_NT_CALL] = t; }

	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HERK_UN_CALL] = t; }
	{ blockSizes t = { 16, 32, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HERK_UC_CALL] = t; }
	{ blockSizes t = { 16, 16, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HERK_LN_CALL] = t; }
	{ blockSizes t = { 16, 32, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HERK_LC_CALL] = t; }

	{ blockSizes t = { 16, 32, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][HERK_UN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][HERK_UC_CALL] = t; }
	{ blockSizes t = { 16, 32, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][HERK_LN_CALL] = t; }
	{ blockSizes t = { 8, 16, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][HERK_LC_CALL] = t; }

	{ blockSizes t = { 32, 16, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 8, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 8, 8, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 8, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 4, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][SYMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][SYMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][SYMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_DOUBLE][SYMM_RL_CALL] = t; }

	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HEMM_LU_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HEMM_RU_CALL] = t; }
	{ blockSizes t = { 16, 8, 4, 2, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HEMM_LL_CALL] = t; }
	{ blockSizes t = { 16, 4, 4, 4, 0 }; bestBlockSizes[GEFORCE_GTX_580][TYPE_COMPLEX_FLOAT][HEMM_RL_CALL] = t; }

	#endif      // USE_TUNED_NUMBERS


///////////////////////////////////////////////////////////////////////////////////////////////////////////

    identifyDevice( kDevice );          // Query device name and stores it in the structure

    if( kargs->pigFuncID == CLBLAS_GEMM2 )
    {
        if( kargs->transA == clblasNoTrans )
        {
            if( kargs->transB == clblasNoTrans )
                    currCall = GEMM_NN_CALL;
            else    currCall = GEMM_NT_CALL;
        }
        else
        {
            if( kargs->transB == clblasNoTrans )
                    currCall = GEMM_TN_CALL;
            else    currCall = GEMM_TT_CALL;
        }
    }
    else if( kargs->pigFuncID == CLBLAS_HERK )
    {
        if( kargs->uplo == clblasUpper )
        {
            if( kargs->transA == clblasNoTrans )
                    currCall = HERK_UN_CALL;
            else    currCall = HERK_UC_CALL;
        }
        else
        {
            if( kargs->transA == clblasNoTrans )
                    currCall = HERK_LN_CALL;
            else    currCall = HERK_LC_CALL;
        }
    }
    else if( (kargs->pigFuncID == CLBLAS_SYMM) || (kargs->pigFuncID == CLBLAS_SYMM_DIAGONAL) )
    {
        if( kargs->side == clblasLeft )
        {
            if( kargs->uplo == clblasUpper )
                    currCall = SYMM_LU_CALL;
            else    currCall = SYMM_LL_CALL;
        }
        else
        {
            if( kargs->uplo == clblasUpper )
                    currCall = SYMM_RU_CALL;
            else    currCall = SYMM_RL_CALL;
        }
    }
    else if( (kargs->pigFuncID == CLBLAS_HEMM) || (kargs->pigFuncID == CLBLAS_HEMM_DIAGONAL) )
    {
        if( kargs->side == clblasLeft )
        {
            if( kargs->uplo == clblasUpper )
                    currCall = HEMM_LU_CALL;
            else    currCall = HEMM_LL_CALL;
        }
        else
        {
            if( kargs->uplo == clblasUpper )
                    currCall = HEMM_RU_CALL;
            else    currCall = HEMM_RL_CALL;
        }
    }

    temp = bestBlockSizes [ (kDevice->ident).chip ] [kargs->dtype] [currCall];

    if( (temp.TY == 0) || (temp.TX == 0) || (temp.ITEMY == 0) || (temp.ITEMX == 0) )
    {
        // If optimal block-sizes for the device is not available,
        // we take default block-sizes
        temp = bestBlockSizes [CHIP_UNKNOWN] [kargs->dtype] [currCall];
    }

    maxWGSize = deviceMaxWorkgroupSize( (kDevice->id), NULL );

    while( ( ((size_t)temp.TY)*((size_t)temp.TX) ) > maxWGSize )   // FIXME check this
    {
       if( temp.TX < temp.TY )
               temp.TX /= 2;
       else    temp.TY /= 2;
    }

    return temp;
}

