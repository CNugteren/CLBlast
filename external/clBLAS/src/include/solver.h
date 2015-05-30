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


#ifndef SOLVER_H_
#define SOLVER_H_

#include <defbool.h>

#include <cltypes.h>
#include <kerngen.h>
#include <clkern.h>
#include <clBLAS.h>
#include <kernel_extra.h>

struct Kernel;

// OpenCL solver ID
typedef int solver_id_t;

/**
 * @internal
 * @defgroup SOLVERIF Solver interface
 *
 * This interface binds the library frontend to the library backend
 */
/*@{*/

/**
 * @internal
 * @brief Solver flags
 */
typedef enum SolverFlags {
    /** supports 1D work space */
    SF_WSPACE_1D = 0x01,
    /** supports 2D work space */
    SF_WSPACE_2D = 0x02,
    /** input data blocks at the top level must be square */
    SF_TOP_INPUT_SQUARE_BLOCKS = 0x04
} SolverFlags;

typedef enum PatternPerformance{
    PPERF_NOT_SUPPORTED = -1,
    PPERF_POOR = 0,
    PPERF_AVERAGE,
    PPERF_GOOD,
    PPERF_BEST
} PatternPerformance;

typedef enum CheckCalcPGran{
    PGRAN_CHECK = 0,
    PGRAN_CALC
} CheckCalcPGran;

/**
 * @internal
 * @brief type of function generating kernel source for an
 *         OpenCL based solver
 *
 * @param[out] buf         Pointer to a buffer to store a generated kenrel to
 * @param[in] buflen       Length of the buffer
 * @param[in] subdims      Subproblem dimensions to generate an optimal kernel
 * @param[in] pgran        Data parallelism granularity
 * @param[in] extra        Generator extra information depending on the
 *                          application fields
 *
 * If the pointer to the buffer is NULL, the function should just calculate
 * needed size of the buffer to fit the code in.
 *
 * @return size of the generated kernel source on success; negated error code
 *         otherwise
 *   - -ENOMEM: enough of memory to allocated internal structures
 *   - -EOVERFLOW: generated source exceeds the buffer size
 *   - -EINVAL: invalid argument is passed
 */
typedef ssize_t
(*SolverKgen)(
   char *buf,
   size_t buflen,
   const SubproblemDim *subdims,
   const PGranularity *pgran,
   void *extra);

/**
 * @internal
 * @brief Solver operations
 *
 * The 'args' parameter for 'calcPrepWorkGroups',
 * and the second parameter for the 'assignKargs' methods plays the role of pointer
 * to a kernel arguments structure depending on the application field.
 */
typedef struct SolverOps {
    /** Kernel generator */
    SolverKgen genKernel;

    /** Assign kernel arguments; the first argument is kernel argument batch
     *  passed immediately to a kernel */
    void (*assignKargs)(KernelArg*, const void* args, const void *extra);

    /** Check if available LDS size is enough to fit all needed data at such
     *  granulation; 'kernelArgs' - kernel arguments depending on the
     *  application fields */
    bool (*isFitToLDS)(
        SubproblemDim *dims,
        DataType,
        cl_ulong ldsSize,
        const void *args);

    /** Get the pattern`s performance estimation for specified flags,
     * arguments and granulation.
     * Is used for selecting most suitable pattern current problem */
    int (*getPatternPerf)(
        unsigned int kflags,
        const void *args);

    /**
     * Inner decomposition axis matching to the fastest moving OpenCL
     * work dimension. Used only for those patterns which use 2 dimensional
     * decomposition
     */
    DecompositionAxis (*innerDecompositionAxis)(const void *args);

    /** Calculate number of needed global threads to execute a kernel */
    void (*calcThreads)(
        size_t threads[2],
        const SubproblemDim *subdims,
        const PGranularity *pgran,
        const void *args,
        const void *extra);

    /** Set number of lines of the same top level block stored into the image
     *  together and the direction of blocks storing. A solver that uses images
     *  and stores data to images by blocks must provide the method */
    void (*imgPackMode)(
        const void *extra,
        const SubproblemDim *subdims,
        int dataID,
        unsigned int *rate,
        clblasOrder *order);

    /** Get solver flags */
    SolverFlags (*getFlags)(void);

    /** Correct problem arguments anr extra kernel parameters
     *  depending on solver specifics. Basically, a solver should not
     *  change any arguments that come from the API level to avoid any
     *  confusing points */
    void (*fixupArgs)(void *args, SubproblemDim* pSubDims, void *extra);

    /** Function, returning default decomposition for the pattern */
    int ( *getDefaultDecomp)(
        PGranularity *pgran,
        SubproblemDim *subdims,
        unsigned int subdimsNum,
        void *pArgs);

    /** Perform validation of decomposition.
      * If "check" flag set to true: validate specified decomposition and
      * check, if specified granulation is valid for it.
      * If "check" flag set to false: calculate granulation,
      * fitting the specified decomposition, if possible */
    bool (*checkCalcDecomp)(
        PGranularity *pgran,
        SubproblemDim *subdims,
        unsigned int subdimsNum,
        DataType dtype,
        int check);


	/*
	 SetBuildOptions
	*/
	void (*setBuildOptions)( char *buildOptsStr, const void *args);

	/*
  	 * selectVectorization
	*/
    KernelExtraFlags (*selectVectorization)( void *kargs, unsigned int vlen);
} SolverOps;

/*@}*/

#endif /* SOLVER_H_ */
