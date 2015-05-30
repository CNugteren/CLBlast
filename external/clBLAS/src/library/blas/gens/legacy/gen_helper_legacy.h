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


#ifndef GEN_HELPER_LEGACY_H_
#define GEN_HELPER_LEGACY_H_

#include <kerngen.h>
#include <dblock_kgen.h>
#include <matrix_props.h>

#include "../blas_kgen.h"

typedef struct CopyImgFuncs {
    char localToImage[2][FUNC_NAME_MAXLEN];
    char globalToImage[2][FUNC_NAME_MAXLEN];
    char globalToLocalTransposed[2][FUNC_NAME_MAXLEN];
    char globalToLocalTransposedGeneric[2][FUNC_NAME_MAXLEN];
    char globalToLocal[2][FUNC_NAME_MAXLEN];
    char globalToLocalGeneric[2][FUNC_NAME_MAXLEN];
    char zeroBlock[2][FUNC_NAME_MAXLEN];
} CopyImgFuncs;

int
generateImageCopyFuncs(
    CopyImgFuncs *copyFuncs,
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset);

int
generateResultUpdateOld(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName);

int
genResultUpdateWithFlagsOld(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    UpdateResultFlags flags,
    const char *optFuncName,
    const char *genericFuncName,
    const char *cachedName);

int generateUpresFuncs(
    struct KgenContext *ctx,
    BlasFunctionID funcID,
    const BlasGenSettings *gset,
    char optFuncName[FUNC_NAME_MAXLEN],
    char genericFuncName[FUNC_NAME_MAXLEN]);

int
genUpresFuncsWithFlags(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    UpdateResultFlags flags,
    char optFuncName[FUNC_NAME_MAXLEN],
    char genericFuncName[FUNC_NAME_MAXLEN]);

#endif /* GEN_HELPER_LEGACY_H_ */
