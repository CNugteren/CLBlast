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
 * Related to BLAS memory patterns
 */

#ifndef BLAS_MEMPAT_H_
#define BLAS_MEMPAT_H_

#include <clBLAS.h>
#include <mempat.h>
#include <clkern.h>
#include <kern_cache.h>

/**
 * @brief Type of internal function implementation
 */
typedef enum clblasImplementation {

    clblasDefaultGemm,           /**< Default: let the library decide what to use. */
    clblasLdsBlockGemm,          /**< Use blocked GEMM with LDS optimization. */
    clblasImageBlockGemm,        /**< Use blocked GEMM with image-based... */
    clblasBlockGemmWithCaching,  /**< Use blocked GEMM with cache-usage optimization. */
    clblasSubgroupGemmWithCaching,/**< Use subgroup GEMM with cache-usage optimization. */

    clblasDefaultTrmm,           /**< Default: let the library decide what to use. */
    clblasLdsBlockTrmm,          /**< Use blocked TRMM with LDS optimization. */
    clblasImageBlockTrmm,        /**< Use blocked TRMM with image-based... */
    clblasBlockTrmmWithCaching,  /**< Use blocked TRMM with cache-usage optimization. */
    clblasSubgroupTrmmWithCaching,/**< Use subgroup TRMM with cache-usage optimization. */

    clblasDefaultTrsm,           /**< Default: let the library decide what to use. */
    clblasLdsBlockTrsm,          /**< Use blocked TRSM with LDS optimization. */
    clblasImageBlockTrsm,        /**< Use blocked TRSM with image-based... */
    clblasBlockTrsmWithCaching,  /**< Use blocked TRSM with cache-usage optimization. */
    clblasBlockTrsmWithoutLds,

    clblasDefaultSyrk,
    clblasBlockSyrk,
    clblasSubgSyrk,

    clblasDefaultSyr2k,
    clblasBlockSyr2k,
    clblasSubgSyr2k

} clblasImplementation;

/**
 * @internal
 * @brief extra information for a memory pattern
 *        used for BLAS problem solving
 * @ingroup BLAS_SOLVERIF_SPEC
 */
typedef struct CLBLASMpatExtra {
    /** memory levels used to store blocks of matrix A */
    meml_set_t aMset;
    /** memory levels used to store blocks of matrix B */
    meml_set_t bMset;
    CLMemType mobjA;
    CLMemType mobjB;
} CLBLASMpatExtra;

/*
 * init memory patterns for the xGEMM functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initGemmMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xGEMM pattern
 */
int
getGemmMemPatternIndex(clblasImplementation impl);

/*
 * Get preferred xGEMM pattern
 */
clblasImplementation
getGemmPreferredPattern(void);

/*
 * init memory patterns for the xGEMV functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initGemvMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xGEMV pattern
 */
int
getGemvMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xSYMV functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initSymvMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYMV pattern
 */
int
getSymvMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xTRMM functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initTrmmMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xTRMM pattern
 */
int
getTrmmMemPatternIndex(clblasImplementation impl);

/*
 * Get preferred xTRMM pattern
 */
clblasImplementation
getTrmmPreferredPattern(void);

/*
 * init memory patterns for the xTRSM functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initTrsmMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xTRSM pattern
 */
int
getTrsmMemPatternIndex(clblasImplementation impl);

/*
 * Get preferred xTRSM pattern
 */
clblasImplementation
getTrsmPreferredPattern(void);

/*
 * init memory patterns for the xSYR2K functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initSyr2kMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYR2K pattern
 */
int
getSyr2kMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xSYRK functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initSyrkMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYRK pattern
 */
int
getSyrkMemPatternIndex(clblasImplementation impl);

/*
 * init memory patters for TRMV routine
 * Returns the number of inited patterns
 */
unsigned int
initTrmvMemPatterns(MemoryPattern *mempats);

int
getTrmvMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for TRSV TRTRI routine
 * Returns the number of inited patterns
 */
unsigned int
initTrsvMemPatterns(MemoryPattern *mempats);

int
getTrsvMemPatternIndex(clblasImplementation impl);

unsigned int
initTrsvGemvMemPatterns(MemoryPattern *mempats);

int
getTrsvGemvMemPatternIndex(clblasImplementation impl);

unsigned int
initSymmMemPatterns(MemoryPattern *mempats);

int
getSymmMemPatternIndex(clblasImplementation impl);

unsigned int
initGemmV2MemPatterns(MemoryPattern *mempats);

int
getGemmV2MemPatternIndex(clblasImplementation impl);

unsigned int
initGemmV2TailMemPatterns(MemoryPattern *mempats);

int
getGemmV2TailMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xSYR functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initSyrMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYR pattern
 */
int
getSyrMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xSYR2 functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initSyr2MemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYR2 pattern
 */
int
getSyr2MemPatternIndex(clblasImplementation impl);


/*
 * init memory patters for GER routine
 * Returns the number of inited patterns
 */
unsigned int
initGerMemPatterns(MemoryPattern *mempats);

int
getGerMemPatternIndex(clblasImplementation impl);

unsigned int
initHerMemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xSYR pattern
 */
int
getHerMemPatternIndex(clblasImplementation impl);

/*
 * init memory patterns for the xHER2 functions
 *
 * Returns number of the initialized patterns
 */
unsigned int
initHer2MemPatterns(MemoryPattern *mempats);

/*
 * Get index of the specific xHER2 pattern
 */
int
getHer2MemPatternIndex(clblasImplementation impl);

unsigned int
initGbmvMemPatterns(MemoryPattern *mempats);

int
getGbmvMemPatternIndex(clblasImplementation impl);

unsigned int
initSwapMemPatterns(MemoryPattern *mempats);

int
getSwapMemPatternIndex(clblasImplementation impl);

unsigned int
initScalMemPatterns(MemoryPattern *mempats);

int
getScalMemPatternIndex(clblasImplementation impl);

unsigned int
initCopyMemPatterns(MemoryPattern *mempats);

int
getCopyMemPatternIndex(clblasImplementation impl);

unsigned int
initDotMemPatterns(MemoryPattern *mempats);

int
getDotMemPatternIndex(clblasImplementation impl);

unsigned int
initAxpyMemPatterns(MemoryPattern *mempats);

int
getAxpyMemPatternIndex(clblasImplementation impl);

unsigned int
initReductionMemPatterns(MemoryPattern *mempats);

int
getReductionMemPatternIndex(clblasImplementation impl);

unsigned int
initRotgMemPatterns(MemoryPattern *mempats);

int
getRotgMemPatternIndex(clblasImplementation impl);

unsigned int
initRotmgMemPatterns(MemoryPattern *mempats);

int
getRotmgMemPatternIndex(clblasImplementation impl);

unsigned int
initRotmMemPatterns(MemoryPattern *mempats);

int
getRotmMemPatternIndex(clblasImplementation impl);

unsigned int
initiAmaxMemPatterns(MemoryPattern *mempats);

int
getiAmaxMemPatternIndex(clblasImplementation impl);

unsigned int
initNrm2MemPatterns(MemoryPattern *mempats);

int
getNrm2MemPatternIndex(clblasImplementation impl);

unsigned int
initAsumMemPatterns(MemoryPattern *mempats);

int
getAsumMemPatternIndex(clblasImplementation impl);

#endif /* BLAS_MEMPAT_H_ */
