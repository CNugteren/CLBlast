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


#ifndef TRSM_KGEN_LEGACY_H_
#define TRSM_KGEN_LEGACY_H_

void
genUpdateIntermTrsmResult(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const char *optFuncName,
    const char *genericFuncName,
    bool withMhitCond);

void
genHeapTrsmResultToLDS(
    struct KgenContext *ctx,
    const BlasGenSettings *gset,
    const char *funcName,
    const char *dstName);

void
genInvertingBlockFunc(
    struct KgenContext *ctx,
    size_t pitch,
    DataType dtype,
    KernelExtraFlags kflags);

#endif /* TRSM_KGEN_LEGACY_H_ */
