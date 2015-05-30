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

#ifndef _TESTDG_H_
#define _TESTDG_H_

// Coming from testDG.hpp

enum TRIANGLE_OPERATIONS {
	LTOU,
	UTOL,
	SWAP
};


enum RealMatrixCreationFlags {
		//NO_FLAGS			= 0,
		ROW_MAJOR_ORDER 		= 1,
		PACKED_MATRIX 			= 2,
		SYMMETRIC_MATRIX		= 4,
		UPPER_HALF_ONLY			= 8,
		LOWER_HALF_ONLY			= 16,
		NO_ALIGNMENT			= 32,
		UNIT_DIAGONAL			= 64,
		RANDOM_INIT			= 128,
		ZERO_DIAGONAL			= 256
	};

#define setDiagonalUnity() 	setDiagonalUnityOrNonUnity(1, data, rows, cols, lda, vectorLength, creationFlags, bound) // Unity diagonal
#define setDiagonalRandom() 	setDiagonalUnityOrNonUnity(2, data, rows, cols, lda, vectorLength, creationFlags, bound) // Random values
#define setDiagonalZero()	setDiagonalUnityOrNonUnity(0, data, rows, cols, lda, vectorLength, creationFlags, bound) // Zero diagonal

// Column-Major is i,j replaced and RML is CMU
// So CMU(i,j) will be RML(j,i)
// The following is Row-Major packed
#define RMLPacked(i,j) ((T*)data + ((i*(i+1))/2 + j) * vectorLength)
#define RMUPacked(i,j) ((T*)data + ((i*((2* rows) + 1 - i))/2 + (j -i))* vectorLength )

#define CMUPacked(i,j) ((T*)data + ((j*(j+1))/2 + i)* vectorLength)
#define CMLPacked(i,j) ((T*)data + ((j*((2*rows) + 1 - j))/2 + (i - j))* vectorLength)


#endif
