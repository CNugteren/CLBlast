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


/*
 * Generators initialization
 */

#include <blas_mempat.h>

#include "clblas-internal.h"
#include "init.h"

unsigned int
initGemmMemPatterns(MemoryPattern *mempats)
{
    initGemmLdsPattern(&mempats[0]);
    initGemmImgPattern(&mempats[1]);
	InitGEMMCachedBlockPattern(&mempats[2]);
	InitGEMMCachedSubgroupPattern(&mempats[3]);
    return 4;
}

int
getGemmMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {
	case clblasLdsBlockGemm:				return  0;
    case clblasImageBlockGemm:			return  1;
    case clblasBlockGemmWithCaching:		return  2;
    case clblasSubgroupGemmWithCaching:	return	3;
	default:								return -1;
    }
}

clblasImplementation
getGemmPreferredPattern(void)
{
    switch (clblasSolvers[CLBLAS_GEMM].defaultPattern) {
    case 0:  return clblasLdsBlockGemm;
    case 1:  return clblasImageBlockGemm;
    case 2:  return clblasBlockGemmWithCaching;
    case 3:  return clblasSubgroupGemmWithCaching;
    default: return clblasDefaultGemm;
    }
}

unsigned int
initGemvMemPatterns(MemoryPattern *mempats)
{
    initGemvPattern(mempats);

    return 1;
}

int
getGemvMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {
    default:    return -1;
    }
}

unsigned int
initSymvMemPatterns(MemoryPattern *mempats)
{
    initSymvPattern(mempats);

    return 1;
}

int
getSymvMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {
    default:    return -1;
    }
}

unsigned int
initTrmmMemPatterns(MemoryPattern *mempats)
{
    initTrmmLdsPattern(mempats);
    initTrmmImgPattern(&mempats[1]);
    initTrmmCachedBlockPattern(&mempats[2]);
    initTrmmCachedSubgroupPattern(&mempats[3]);

    return 4;
}

int
getTrmmMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {

        case clblasLdsBlockTrmm:             return  0;
        case clblasImageBlockTrmm:           return  1;
        case clblasBlockTrmmWithCaching:     return  2;
        case clblasSubgroupTrmmWithCaching:  return 3;

        default: return -1;
    }
}

clblasImplementation
getTrmmPreferredPattern(void)
{
    switch (clblasSolvers[CLBLAS_TRMM].defaultPattern) {

        case 0: return clblasLdsBlockTrmm;
        case 1: return clblasImageBlockTrmm;
        case 2: return clblasBlockTrmmWithCaching;
        case 3: return clblasSubgroupTrmmWithCaching;

        default: return clblasDefaultTrmm;
    }
}

unsigned int
initTrsmMemPatterns(MemoryPattern *mempats)
{
    initTrsmLdsPattern(mempats);
    initTrsmImgPattern(&mempats[1]);
    initTrsmLdsLessCachedPattern(&mempats[2]);
    initTrsmCachedPattern(&mempats[3]);

    return 4;
}

int
getTrsmMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {
    case clblasLdsBlockTrsm:         return  0;
    case clblasImageBlockTrsm:       return  1;
    case clblasBlockTrsmWithoutLds:  return  2;
    case clblasBlockTrsmWithCaching: return  3;
    default:                            return -1;
    }
}

clblasImplementation
getTrsmPreferredPattern(void)
{
    switch (clblasSolvers[CLBLAS_TRSM].defaultPattern) {
    case 0:  return clblasLdsBlockTrsm;
    case 1:  return clblasImageBlockTrsm;
    case 2:  return clblasBlockTrsmWithoutLds;
    case 3:  return clblasBlockTrsmWithCaching;
    default: return clblasDefaultTrsm;
    }
}

unsigned int
initSyrkMemPatterns(MemoryPattern *mempats)
{
    initSyrkBlockPattern(&mempats[0]);
    initSyrkSubgPattern(&mempats[1]);

    return 2;
}

clblasImplementation
getSyrkPreferredPattern(void)
{
    switch (clblasSolvers[CLBLAS_SYRK].defaultPattern) {

    case 0:  return clblasBlockSyrk;
    case 1:  return clblasSubgSyrk;
    default: return clblasDefaultSyrk;

    }
}

int
getSyrkMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {

    case clblasBlockSyrk: return 0;
    case clblasSubgSyrk: return 1;
    default:    return -1;

    }
}

unsigned int
initSyr2kMemPatterns(MemoryPattern *mempats)
{
    initSyr2kBlockPattern(&mempats[0]);
    initSyr2kSubgPattern(&mempats[1]);

    return 2;
}

clblasImplementation
getSyr2kPreferredPattern(void)
{
    switch (clblasSolvers[CLBLAS_SYR2K].defaultPattern) {

    case 0:  return clblasBlockSyr2k;
    case 1:  return clblasSubgSyr2k;
    default: return clblasDefaultSyr2k;

    }
}

int
getSyr2kMemPatternIndex(clblasImplementation impl)
{
    switch (impl) {

    case clblasBlockSyr2k: return 0;
    case clblasSubgSyr2k: return 1;
    default:    return -1;

    }
}

unsigned int
initTrmvMemPatterns(MemoryPattern *mempats)
{
	initTrmvRegisterPattern(&mempats[0]);
	return 1;
}

int
getTrmvMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initTrsvMemPatterns(MemoryPattern *mempats)
{
	initTrsvDefaultPattern(&mempats[0]);
	return 1;
}

int
getTrsvMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initSyrMemPatterns(MemoryPattern *mempats)
{
    initSyrDefaultPattern(&mempats[0]);
    return 1;
}

int
getSyrMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initSyr2MemPatterns(MemoryPattern *mempats)
{
	initSyr2DefaultPattern(&mempats[0]);
	return 1;
}

int
getSyr2MemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initTrsvGemvMemPatterns(MemoryPattern *mempats)
{
	initTrsvGemvDefaultPattern(&mempats[0]);
	return 1;
}

int
getTrsvGemvMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initSymmMemPatterns(MemoryPattern *mempats)
{
	initSymmDefaultPattern(&mempats[0]);
	return 1;
}


int
getSymmMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initGemmV2MemPatterns(MemoryPattern *mempats)
{
	initGemmV2CachedPattern(mempats);
	return 1;
}

int
getGemmV2MemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
		default: return -1;
	}
}

unsigned int
initGemmV2TailMemPatterns(MemoryPattern *mempats)
{
	initGemmV2TailCachedPattern(mempats);
	return 1;
}

int
getGemmV2TailMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
		default: return -1;
	}
}

unsigned int
initGerMemPatterns(MemoryPattern *mempats)
{
	initGerRegisterPattern(&mempats[0]);
	return 1;
}

int
getGerMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initHerMemPatterns(MemoryPattern *mempats)
{
    initHerDefaultPattern(&mempats[0]);
    return 1;
}

int
getHerMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initHer2MemPatterns(MemoryPattern *mempats)
{
	initHer2DefaultPattern(&mempats[0]);
	return 1;
}

int
getHer2MemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initGbmvMemPatterns(MemoryPattern *mempats)
{
	initGbmvRegisterPattern(&mempats[0]);
	return 1;
}

int
getGbmvMemPatternIndex(clblasImplementation impl)
{
	switch(impl) {
	default: return -1;
	}
}

unsigned int
initSwapMemPatterns(MemoryPattern *mempats)
{
    initSwapRegisterPattern(&mempats[0]);
    return 1;
}

int
getSwapMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initScalMemPatterns(MemoryPattern *mempats)
{
    initScalRegisterPattern(&mempats[0]);
    return 1;
}


int
getScalMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initCopyMemPatterns(MemoryPattern *mempats)
{
    initCopyRegisterPattern(&mempats[0]);
    return 1;
}

int
getCopyMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initAxpyMemPatterns(MemoryPattern *mempats)
{
    initAxpyRegisterPattern(&mempats[0]);
    return 1;
}

int
getAxpyMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initDotMemPatterns(MemoryPattern *mempats)
{
    initDotRegisterPattern(&mempats[0]);
    return 1;
}

int
getDotMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initReductionMemPatterns(MemoryPattern *mempats)
{
    initReductionRegisterPattern(&mempats[0]);
    return 1;
}

int
getReductionMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initRotgMemPatterns(MemoryPattern *mempats)
{
    initRotgRegisterPattern(&mempats[0]);
    return 1;
}

int
getRotgMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initRotmgMemPatterns(MemoryPattern *mempats)
{
    initRotmgRegisterPattern(&mempats[0]);
    return 1;
}

int
getRotmgMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initRotmMemPatterns(MemoryPattern *mempats)
{
    initRotmRegisterPattern(&mempats[0]);
    return 1;
}

int
getRotmMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initiAmaxMemPatterns(MemoryPattern *mempats)
{
    initiAmaxRegisterPattern(&mempats[0]);
    return 1;
}

int
getiAmaxMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initNrm2MemPatterns(MemoryPattern *mempats)
{
    initNrm2RegisterPattern(&mempats[0]);
    return 1;
}

int
getNrm2MemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}

unsigned int
initAsumMemPatterns(MemoryPattern *mempats)
{
    initAsumRegisterPattern(&mempats[0]);
    return 1;
}

int
getAsumMemPatternIndex(clblasImplementation impl)
{
    switch(impl) {
    default: return -1;
    }
}
