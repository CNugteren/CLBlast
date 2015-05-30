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
 * Memory usage pattern related definitions
 */

#ifndef MEMPAT_H_
#define MEMPAT_H_

#include <solver.h>

enum {
    MAX_MEMORY_PATTERNS = 16
};

/**
 * @internal
 * @brief Memory level identifiers
 *
 * @ingroup SOLVERIF
 */
typedef enum CLMemLevel {
    CLMEM_LEVEL_LDS = 0x01,        /**< Local data storage */
    CLMEM_LEVEL_L1 = 0x02,         /**< L1 cache */
    CLMEM_LEVEL_L2 = 0x04          /**< L2 cache */
} CLMemLevel;

/**
 * @internal
 * @brief Memory type identifiers
 *
 * @ingroup SOLVERIF
 */
typedef enum CLMemType {
    CLMEM_GLOBAL_MEMORY,
    CLMEM_LOCAL_MEMORY,
    CLMEM_IMAGE,
    // FIXME: it's for backward compatibility, remove after blkmul deprecation
    CLMEM_BUFFER = CLMEM_LOCAL_MEMORY
} CLMemType;

// memory levels set
typedef unsigned int meml_set_t;

/*
 * FIXME: deprecate cuLevel and thLevel
 */

/**
 * @internal
 * @brief Solver memory pattern description structure
 *
 * The structure decribes memory using features and used
 * by frontend at choosing of solving strategy and decomposition
 * block sizes
 *
 * @ingroup SOLVERIF
 */
typedef struct MemoryPattern {
    const char *name;           /**< Pattern's name */
    unsigned int nrLevels;      /**< Decomposition levels number */
    /** Level a problem is decomposed among compute units at */
    int cuLevel;
    /** Level a problem is decomposed among threads within single compute unit */
    int thLevel;
    SolverOps *sops;            /**< Solver operations */
    /** extra information specific for the application field */
    void *extra;
} MemoryPattern;

#endif /* MEMPAT_H_ */
