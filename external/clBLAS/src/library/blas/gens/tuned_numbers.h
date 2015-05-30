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


#ifndef _TUNED_NUMBERS_
#define _TUNED_NUMBERS_

#include <clBLAS.h>
#include <cltypes.h>
#include <devinfo.h>
#include <solution_seq.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blockSizes
{
    unsigned char TY;               // Not more than 32
    unsigned char TX;
    unsigned char ITEMY:7;          // Not more than 8
    unsigned char ITEMX:7;
    unsigned char useBarrier:1;
} blockSizes;

blockSizes bestBlockSizeForDevice( SolutionStep *step );

#ifdef __cplusplus
}       /* extern "C" { */
#endif

#endif // _TUNED_NUMBERS_
