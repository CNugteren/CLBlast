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
 * Data and execution granulation
 */

#ifndef GRANULATION_H_
#define GRANULATION_H_

/**
 * @internal
 * @brief Decomposition axis
 * @ingroup PROBLEM_DECOMPOSITION
 */
typedef enum DecompositionAxis {
    DECOMP_AXIS_Y,
    DECOMP_AXIS_X
} DecompositionAxis;

/**
 * @internal
 * @brief Data parallelism granularity
 * @ingroup PROBLEM_DECOMPOSITION
 */
typedef struct PGranularity {
    /** work group sizes */
    unsigned int wgSize[2];
    /** work group dimension */
    unsigned int wgDim;
    /** wavefront size */
    unsigned int wfSize;
    /** Record number of work-groups spawned */
    unsigned int numWGSpawned[2];
	/** max number of work group size */
	unsigned int maxWorkGroupSize;
} PGranularity;

/**
 * @internal
 * @brief Subproblem dimensions
 *
 * The structure represents how a problem is decomposed during
 * the computation. The decomposition is made in terms of
 * resulting data. It describes as well what portion of work each
 * computing item gets as what chunk it evaluates at a time.
 * The chunk processed at a time is typically bound by amount
 * of resources consumed at this level of decomposition while
 * the whole portion is bound of amount of more high level resources
 * to be available, and can also be used for the purpose of work
 * balancing.
 *
 * @ingroup PROBLEM_DECOMPOSITION
 */
typedef struct SubproblemDim {
    size_t x;       /**< Subproblem step size in X dimension */
    size_t y;       /**< Subproblem step size in Y dimension */
    /** Width of data blocks processed consecutively
     * to evaluate a subproblem of 'x' by 'y' size */
    size_t bwidth;
    size_t itemX;   /**< Size of the whole subproblem in X dimension
                        evaluated by a computing item */
    size_t itemY;   /**< Size of the whole subproblem in Y dimension
                        evaluated by a computing item */
} SubproblemDim;

#endif /* GRANULATION_H_ */
