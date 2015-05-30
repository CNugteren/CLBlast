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


#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>

#include <clblas_stddef.h>
#include <clblas-internal.h>
#include <toolslib.h>
#include <events.h>

#include "matrix_dims.h"
#include "solution_assert.h"
#include "solution_seq.h"

#define DECOMPOSITION_THRESHOLD(type) (2560 * sizeof(cl_float) / dtypeSize(type))

/* From solution_seq.c */
bool VISIBILITY_HIDDEN isMatrixInImage(MemoryPattern *pattern, MatrixRole mrole);
void VISIBILITY_HIDDEN releaseStepImgs(SolutionStep *step);

#define isMatrixCached(pattern, mrole)                              \
    checkMatrixMemLevelSet(pattern, mrole, (CLMEM_LEVEL_L2 | CLMEM_LEVEL_L1))

#define isLdsUsed(pattern)                                          \
    (checkMatrixMemLevelSet(pattern, MATRIX_A, CLMEM_LEVEL_LDS) ||  \
     checkMatrixMemLevelSet(pattern, MATRIX_B, CLMEM_LEVEL_LDS))

enum {
    DEFAULT_BUFS_LSIZE_0 = 8,
    DEFAULT_BUFS_LSIZE_1 = 8,
    DEFAULT_CACHED_BUFS_LSIZE_0 = 8,
    DEFAULT_CACHED_BUFS_LSIZE_1 = 8
};

static cl_uint getQueueMaxImages(cl_command_queue queue);

static bool checkMatrixMemLevelSet(MemoryPattern *pattern, MatrixRole mrole,
    meml_set_t mask);

static void stripeDivision(BlasFunctionID funcID, const CLBlasKargs *args,
    ListHead *seq, cl_uint totalCUs);
static void rectDivision(BlasFunctionID funcID, const CLBlasKargs *args,
    ListHead *seq, cl_uint totalCUs);
static void triMatrixStripeDivision(BlasFunctionID funcID,
    const CLBlasKargs *args, ListHead *seq, cl_uint totalCUs);

static cl_bool findBestPattern(SolutionStep *step);

static void getDefaultStepGranulation(SolutionStep *step);
static bool avoidLoadFromStorage(SolutionStep *step);

static bool getStepResources(SolutionStep *step);
static void getSuitableImageSizes(size_t *minWidth, size_t *minHeight,
    size_t *bestHeight, MatrixRole mrole, CLBlasKargs *kargs, unsigned int vecLen,
    SubproblemDim *subdims);

static ListNode* decomposeTRXMStep(SolutionStep *step);
static ListNode* decomposeSYRKStep(SolutionStep *step);
static ListNode* decomposeSYR2KStep(SolutionStep *step);

// Find vector length which lda and tile width is divisible on
unsigned int
appropriateVecLen(size_t ld, unsigned int tsize, size_t twidth, int funcLevel)
{
    unsigned int vlen = sizeof(cl_float4) / tsize;

    if (funcLevel == 3) {
        vlen *= 2;
    }
    while (vlen > twidth) {
        vlen /= 2;
    }

    while ((ld % vlen) || (twidth % vlen)) {
        vlen /= 2;
    }

    return vlen;
}

/*
 * Select an appropriate vectorization to perform computation with.
 * It's done based upon the problem sizes and device type. The device type
 * is taken into account as well since not all devices allow not aligned
 * access to vector data.
 */

cl_int
selectVectorization(
    const SolutionStep *step,
    CLBLASKernExtra *kextra)
{
    const TargetDevice *device = &step->device;
    cl_device_type devType;
    cl_int err;
    size_t tw;
    bool tra;
    size_t checkedSizes[3];
    int i, j;
    const CLBlasKargs *kargs = &step->args;
    KernelExtraFlags kflags = kextra->flags;
    KernelExtraFlags vecFlags[3] = { KEXTRA_NO_COPY_VEC_A, KEXTRA_NO_COPY_VEC_B,
                                     KEXTRA_NO_COPY_VEC_C };
    unsigned int vlen;
    unsigned int tsize;
    MemoryPattern *mempat;
    const SubproblemDim *dim = &step->subdims[1];
    int funcLevel;

    mempat = &clblasSolvers[step->funcID].memPatterns[step->patternID];
    err = clGetDeviceInfo(device->id, CL_DEVICE_TYPE, sizeof(devType),
                          &devType, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }

    if (isLdsUsed(mempat)) {
        kextra->vecLenC = kextra->vecLen = sizeof(cl_float4) /
                                                  dtypeSize(step->args.dtype);
        kextra->vecLenA = kextra->vecLenB = kextra->vecLen;
    }
    else {
        kextra->vecLenA = kextra->vecLenB = 0;
    }

    // select vectorization based upon leading dimensions and starting offsets
    for (i = 0; i < 2; i++) {
        if (!i) {
           // check by leading dimensions
           checkedSizes[0] = kargs->lda.matrix;
           if (funcBlasLevel(step->funcID) == 2) {
               checkedSizes[1] = checkedSizes[2] = 0;
           }
           else {
               checkedSizes[1] = kargs->ldb.matrix;
               checkedSizes[2] = kargs->ldc.matrix;
           }
        }
        else {
            // check by offsets
            checkedSizes[0] = kargs->offA;
            checkedSizes[1] = kargs->offBX;
            checkedSizes[2] = kargs->offCY;
        }

        if (funcHasTriangMatrix(step->funcID)) {
            checkedSizes[2] = checkedSizes[1];
        }

        vlen = sizeof(cl_float4) / dtypeSize(step->args.dtype);

        /*
         * Disable vectorization at load from the global memory to LDS
         * if matrix width is not aligned on the boundary of the float4
         */
        for (j = 0; j < 3; j++) {
            if (checkedSizes[j] % vlen) {
                kflags |= vecFlags[j];
            }
        }

		if ((step->funcID == CLBLAS_TRMV) || (step->funcID == CLBLAS_HEMV))
		{
		   if ( ( ((kflags & KEXTRA_UPPER_TRIANG)==0) && (kflags & KEXTRA_COLUMN_MAJOR) ) ||
		        ( ((kflags & KEXTRA_UPPER_TRIANG)) && ((kflags & KEXTRA_COLUMN_MAJOR) == 0)) )

			{
				if( (kargs->N) % vlen)
				{
					kflags |= KEXTRA_NO_COPY_VEC_A;
				}
			}
		}

		if(mempat->sops->selectVectorization != NULL)
		{
			kflags |= mempat->sops->selectVectorization((void *)kargs, vlen);
		}

		if ((step->funcID == CLBLAS_TRSV) || (step->funcID == CLBLAS_TRSV_GEMV))
		{
			//
			// TRTRI, GEMV Part - Only Scalar loads
			// PENDING:
			// Analyze Case by Case and selectively enable/disable
			//
			kflags |= KEXTRA_NO_COPY_VEC_A;
			kflags |= KEXTRA_NO_COPY_VEC_B;
		}

		//
		// Routines that Use LDS should be above this IF statement
		//
		if (isLdsUsed(mempat)) {
            continue;
        }

		//
		// Routines that dont use LDS have to be below the isLdsUsed() code
		//
		if (step->funcID == CLBLAS_GEMM2)
		{
			if ((step->subdims[0].y > step->args.M) || (step->subdims[0].x > step->args.N))
			{
				kextra->vecLen = 1;
			} else {
        	    kextra->vecLen = sizeof(cl_float4) / dtypeSize(step->args.dtype);
			}
        	kextra->vecLenA = kextra->vecLen;
        	kextra->vecLenB = kextra->vecLen;
        	kextra->vecLenC = kextra->vecLen;
			continue;
		}

		if (step->funcID == CLBLAS_GEMM_TAIL)
		{
        	kextra->vecLen =  1;
        	kextra->vecLenA = 1;
        	kextra->vecLenB = 1;
        	kextra->vecLenC = 1;
			continue;
		}
    	funcLevel = funcBlasLevel(step->funcID);
        funcLevel = funcBlasLevel(step->funcID);

        /*
         * If the step's pattern uses LDS, it is responsible for alignment.
         * Otherwise it's needed to provide appropriate vector length
         */
        tsize = dtypeSize(step->args.dtype);
        tra = isMatrixAccessColMaj(step->funcID, kflags, MATRIX_A);
        tw = (tra) ? dim->y : dim->bwidth;
        vlen = appropriateVecLen(checkedSizes[0], tsize, tw, funcLevel);
        kextra->vecLenA = (kextra->vecLenA) ? umin(kextra->vecLenA, vlen) :
                                              vlen;

        tra = isMatrixAccessColMaj(step->funcID, kflags, MATRIX_B);
        tw = ((funcLevel == 2) || !tra) ? dim->bwidth : dim->x;
        vlen = appropriateVecLen(checkedSizes[1], tsize, tw, funcLevel);
        kextra->vecLenB = (kextra->vecLenB) ? umin(kextra->vecLenB, vlen) :
                                              vlen;

        tra = isMatrixAccessColMaj(step->funcID, kflags, MATRIX_C );
        tw = ((funcLevel == 2) || tra) ? dim->y : dim->x;
        vlen = appropriateVecLen( checkedSizes[2],
            tsize,
            tw,
            funcLevel );
        kextra->vecLenC = kextra->vecLenC ? umin(vlen,kextra->vecLenC) :
                                            vlen;

        kextra->vecLen = umin(kextra->vecLenA, kextra->vecLenB);
        kextra->vecLen = umin(kextra->vecLenC, kextra->vecLen);
    }

    kextra->flags = kflags;

    return CL_SUCCESS;
}

/*
 * Replace 'offsetM' and 'offsetN' field with respective extra offset at
 * 'offA', 'offBX', 'offCY' and taking into accoutn offset along K
 */
void VISIBILITY_HIDDEN
fixupGemmOffsets(CLBlasKargs *kargs, KernelExtraFlags kflags, size_t offsetK)
{
    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_A)) {
        kargs->offA += offsetK * kargs->lda.matrix + kargs->offsetM;
    }
    else {
        kargs->offA += kargs->offsetM * kargs->lda.matrix + offsetK;
    }
    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_B)) {
        kargs->offBX += offsetK * kargs->ldb.matrix + kargs->offsetN;
    }
    else {
        kargs->offBX += kargs->offsetN * kargs->ldb.matrix + offsetK;
    }
    if (isMatrixAccessColMaj(CLBLAS_GEMM, kflags, MATRIX_C)) {
        kargs->offCY += kargs->offsetN * kargs->ldc.matrix + kargs->offsetM;
    }
    else {
        kargs->offCY += kargs->offsetM * kargs->ldc.matrix + kargs->offsetN;
    }
    kargs->offsetM = kargs->offsetN = 0;
}

ListNode
*decomposeProblemStep(SolutionStep *step)
{
    ListNode *node;

    switch (step->funcID) {
    case CLBLAS_TRMM:
    case CLBLAS_TRSM:
        node = decomposeTRXMStep(step);
        break;
    case CLBLAS_SYRK:
        node = decomposeSYRKStep(step);
        break;
    case CLBLAS_SYR2K:
        node = decomposeSYR2KStep(step);
        break;
    default:
        node = &step->node;
        break;
    }

    return node;
}

cl_int
makeSolutionSeq(
    BlasFunctionID funcID,
    const CLBlasKargs *args,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events,
    ListHead *seq)
{
    cl_int err;
    cl_uint j, totalCUs, numDevicesWithoutDoubles;
    bool hasDouble;
    SolutionStep *step;
    CLBLASKernExtra extra;
    ListNode *i;
    MemoryPattern *pattern;
    solver_id_t sid;
    KernelKey key;
    bool need[MAX_CLBLAS_KERNELS_PER_STEP] = {true};
    CLBlasKernelType ktype;
    Kernel *kernel;
    bool loadData = false;
    unsigned char* buffer[MAX_CLBLAS_KERNELS_PER_STEP];
    size_t sizeBuffer[MAX_CLBLAS_KERNELS_PER_STEP];
    char bopts[BUILD_OPTS_MAXLEN]; // Moving bopts up. See the comments before findKernel()
    int ik;
    // first subdimension index in the subproblem dims array
    int firstDimIdx;

    if ((numCommandQueues == 0) || (commandQueues == NULL)) {
        return CL_INVALID_VALUE;
    }

    memset(buffer, 0, sizeof(buffer));
    listInitHead(seq);

    totalCUs = 0;
    numDevicesWithoutDoubles = 0;
    for (j = 0; j < numCommandQueues; j++) {
        cl_device_id devID;

        err = getQueueDevice(commandQueues[j], &devID);
        if (err != CL_SUCCESS) {
            continue;
        }
        if (isDoubleBasedType(args->dtype)) {
            hasDouble = deviceHasNativeDouble(devID, &err);
            if (err != CL_SUCCESS) {
                continue;
            }
            if (!hasDouble) {
                numDevicesWithoutDoubles++;
                continue;
            }
        }

        step = calloc(1, sizeof(SolutionStep));
        if (step == NULL) {
            freeSolutionSeq(seq);
            return CL_OUT_OF_HOST_MEMORY;
        }

        step->funcID = funcID;
        step->args = *args;
        step->args.addrBits = deviceAddressBits(devID, &err);
        step->cmdQueue = commandQueues[j];
        step->numEventsInWaitList = numEventsInWaitList;
        step->eventWaitList = eventWaitList;
        step->event = NULL;
        if (events != NULL) {
            step->event = events + j;
        }
        step->pgran.wfSize = deviceWavefront(devID, &err);
        step->extraFlags = clblasArgsToKextraFlags(args, step->funcID);
        if (step->funcID == CLBLAS_SYR2K) {
            step->extraFlags |= KEXTRA_SYRK_2K_RANK;
        }

        step->device.id = devID;
        err = identifyDevice(&step->device);
        if (err != CL_SUCCESS) {
            freeSolutionSeq(seq);
            return err;
        }

        totalCUs += deviceComputeUnits(devID, &err);
        listAddToTail(seq, &step->node);
    }
    if (totalCUs == 0) {
        return (numDevicesWithoutDoubles == numCommandQueues) ?
                    CL_INVALID_DEVICE : CL_INVALID_COMMAND_QUEUE;
    }

    memset(&extra, 0, sizeof(extra));
    memset(bopts, 0, BUILD_OPTS_MAXLEN*sizeof(char));
    extra.dtype = args->dtype;

    /* Split task between multiple command queues */

    if (funcID == CLBLAS_GEMM) {
        rectDivision(funcID, args, seq, totalCUs);
    }
    else if ((funcID == CLBLAS_SYRK) || (funcID == CLBLAS_SYR2K)) {
        triMatrixStripeDivision(funcID, args, seq, totalCUs);
    }
    else {
        stripeDivision(funcID, args, seq, totalCUs);
    }

    /* Some steps can be decomposed into several sequential substeps */

    parseEnvImplementation();

    // Function level decomposition
    for (i = listNodeFirst(seq); i != seq; i = i->next) {
        step = container_of(i, node, SolutionStep);
        if (step->cmdQueue == NULL) {
            continue;
        }

        if (step->funcID == CLBLAS_GEMM) {
            fixupGemmOffsets(&step->args, step->extraFlags, 0);
            continue;
        }

        i = decomposeProblemStep(step);
    }

	#ifdef DEBUG_2
	printf("Finding a kernel for each step\n");
	#endif

    /* Find a kernel for each step */

    for (i = listNodeFirst(seq); (i != seq) && (err == CL_SUCCESS);
         i = i->next) {

        DeviceIdent *ident;

        step = container_of(i, node, SolutionStep);
        if (step->cmdQueue == NULL) {
            continue;
        }

        ident = &step->device.ident;

        /*
         * Set vendor dependent flags
         *
         * FIXME: thrown this kludge away when generator interface will
         *        support passing ident info
         */
        if (ident->vendor == VENDOR_AMD) {
            step->extraFlags |= (KEXTRA_VENDOR_AMD | KEXTRA_ENABLE_MAD);
        }

        if (!findBestPattern(step)) {
            err = CL_OUT_OF_RESOURCES;
            break;
        }

		#ifdef DEBUG_2
		printf("Find best pattern finished\n");
		#endif


        pattern = &(clblasSolvers[step->funcID].memPatterns[step->patternID]);
        firstDimIdx = 2 - pattern->nrLevels;
        sid = makeSolverID(step->funcID, step->patternID);

        err = getQueueDevice(step->cmdQueue, &key.device);
        err = getQueueContext(step->cmdQueue, &key.context);

        detectProblemTails(step);

        extra.flags = step->extraFlags;
        if (pattern->sops->fixupArgs) {
            pattern->sops->fixupArgs(&step->args, &step->subdims[firstDimIdx],
                                     &extra);
        }
        step->extraFlags = extra.flags;

        key.nrDims = pattern->nrLevels;
        memset(key.subdims, 0, sizeof(key.subdims));
        memcpy(key.subdims, &step->subdims[firstDimIdx],
               sizeof(SubproblemDim) * key.nrDims);

        detectOffsets(step);

        extra.flags = step->extraFlags;

        need[CLBLAS_PREP_A_KERNEL] = isMatrixInImage(pattern, MATRIX_A);
        need[CLBLAS_PREP_B_KERNEL] = isMatrixInImage(pattern, MATRIX_B);

        /*
         * Now, find and enqueue each kernel. Generate and build the kernel
         * on the fly if this kernel is not presented neither in the cache
         * no in the storage
         */
        for (ktype = CLBLAS_COMPUTING_KERNEL;
             ktype < MAX_CLBLAS_KERNELS_PER_STEP; ktype++) {
			 SubproblemDim prepDims[2];

            if (!need[ktype]) {
                continue;
            }

            extra.kernType = ktype;

            err = selectVectorization(step, &extra);
            if (err != CL_SUCCESS) {
                break;
            }

            kernel = NULL;

            //
            // Now that the build options is a part of EXTRA structure,
            // it is also a part of the kernelKey
            // Setting of build options need to be done before
            // findKernel()
            //
            memset(bopts, 0, BUILD_OPTS_MAXLEN*sizeof(char));
            setupBuildOpts(bopts, key.device, pattern);
            if (pattern->sops->setBuildOptions)
            {
                pattern->sops->setBuildOptions(bopts, (void*)(step));
            }
            memcpy(extra.buildOptions, bopts, BUILD_OPTS_MAXLEN);

            if (areKernelsCacheable()) {
                kernel = findKernel(clblasKernelCache, sid, &key, &extra);
            }
            if (kernel == NULL) {
                if (!loadData && !avoidLoadFromStorage(step)) {
                    size_t MNK = (step->args.M + step->args.N + step->args.K) / 3;
                    loadData = !getKernelInfo(&step->device, pattern->name,
                        extra.dtype, step->extraFlags, (int)MNK, &buffer[0],
                        &sizeBuffer[0]);
                }
                if (buffer[ktype] != NULL){
                    kernel = loadKernel((const unsigned char**)&buffer[ktype],
                                        sizeBuffer[ktype], &key, &extra, &err);
                }
                else {
                    SubproblemDim *dims;

                    dims = (ktype == CLBLAS_COMPUTING_KERNEL) ? step->subdims :
                                                                prepDims;

					#ifdef DEBUG_2
					printf("Build options used : %s\n", bopts);
					#endif

                    kernel = makeKernel(key.device, key.context,
                                        pattern->sops->genKernel,
                                        &dims[firstDimIdx], &step->pgran,
                                        &extra, bopts, &err);
                }

                if (kernel == NULL) {
                    break;
                }

                if (areKernelsCacheable()) {
                    getKernel(kernel);
                    if (addKernelToCache(clblasKernelCache, sid, kernel, &key,
                                         clblasKernelExtraCmp)) {
                        putKernel(clblasKernelCache, kernel);
                    }
                }
            } else {
				#ifdef DEBUG_CONTEXT
				printf("KERNEL FOUND IN CACHE\n");
				#endif
			}
            step->kernels[ktype] = kernel;
        }
    }

    if (err != CL_SUCCESS) {
        freeSolutionSeq(seq);
    }

    // free binary kernels
    for (ik = 0; ik < MAX_CLBLAS_KERNELS_PER_STEP; ++ik) {
        free(buffer[ik]);
    }
    return err;
}

static cl_uint
getQueueMaxImages(cl_command_queue queue)
{
    cl_int err;
    cl_device_id device;
    cl_command_queue_properties props;
    cl_bool imageSupport;

    imageSupport = CL_FALSE;
    err = getQueueDevice(queue, &device);
    if (err != CL_SUCCESS) {
        return 0;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport),
        &imageSupport, NULL);
    if (!imageSupport) {
        return 0;
    }

    props = 0;
    err = getQueueProperties(queue, &props);
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        return 0;
    }

    return 2;
}

static bool
isTransBUsed(BlasFunctionID funcID)
{
    if ((CLBLAS_GEMM == funcID) || (CLBLAS_GEMM2 == funcID) || (CLBLAS_GEMM_TAIL == funcID)) {
        return true;
    }
    else {
        return false;
    }
}

KernelExtraFlags
clblasArgsToKextraFlags(const CLBlasKargs *args, BlasFunctionID funcID)
{
    KernelExtraFlags flags = KEXTRA_NO_FLAGS;

    if (args->transA != clblasNoTrans) {
        flags |= KEXTRA_TRANS_A;
    }

    if (isTransBUsed(funcID) && args->transB != clblasNoTrans) {
        flags |= KEXTRA_TRANS_B;
    }

    if (isComplexType(args->dtype)) {
        if (args->transA == clblasConjTrans) {
            flags |= KEXTRA_CONJUGATE_A;
        }
        if (isTransBUsed(funcID) && args->transB == clblasConjTrans) {
            flags |= KEXTRA_CONJUGATE_B;
        }
    }

    if (args->order == clblasColumnMajor) {
        flags |= KEXTRA_COLUMN_MAJOR;
    }
    if ((funcID != CLBLAS_TRMM) && (funcID != CLBLAS_TRSM)) {
        // check if beta is zero
        ArgMultiplier z;

        memset(&z, 0, sizeof(z));
        if (!memcmp(&args->beta, &z, sizeof(z))) {
            flags |= KEXTRA_BETA_ZERO;
        }
    }

    if (funcID != CLBLAS_GEMM) {
        if (args->uplo == clblasUpper) {
            flags |= KEXTRA_UPPER_TRIANG;
        }
        if (args->side == clblasRight) {
            flags |= KEXTRA_SIDE_RIGHT;
        }
        if (args->diag == clblasUnit) {
            flags |= KEXTRA_UNIT_DIAGONAL;
        }
    }
    if (funcID == CLBLAS_GEMV || funcID == CLBLAS_SYMV) {
        if (args->ldb.vector == 1) {
            flags |= KEXTRA_INCX_ONE;
        }
        if (args->ldc.vector == 1) {
            flags |= KEXTRA_INCY_ONE;
        }
    }

    return flags;
}

static bool
checkMatrixMemLevelSet(
    MemoryPattern *pattern,
    MatrixRole mrole,
    meml_set_t mask)
{
    const CLBLASMpatExtra *extra = (const CLBLASMpatExtra*)pattern->extra;
    meml_set_t mset;

    if (mrole == MATRIX_C || extra == NULL) {
        return false;
    }

    switch (mrole) {
    case MATRIX_A:
        mset = extra->aMset;
        break;
    case MATRIX_B:
        mset = extra->bMset;
        break;
    default:
        break;
    }

    return ((mset & mask) != 0);
}

/* Next three functions: stripeDivision(), rectDivision() and
 * triMatrixStripeDivision(), split output matrix into set of non-intersected
 * rectangles. Area of each rectangle depends on the number of Compute Units,
 * available on a device of the given queue.
 * Division is also aligned on the DIVISION_ALIGNMENT boundary. It is measured
 * in number of elements.
 */

/* This constant is used in:
 *     - stripeDivision()
 *     - rectDivision()
 *     - triMatrixStripeDivision()
 *     - decomposeTRXMStep()
 */
static const size_t DIVISION_ALIGNMENT = 128;

static size_t
align(
    size_t value,
    size_t alignment)
{
    /* This implementation assumes that alignment is the power of 2. */
    return (value + (alignment >> 1)) & (~(alignment - 1));
}

/* Stripe division is done according to the picture:
 *
 *      +------+--+----+--+
 *      |      |  |    |  |
 *      |      |  |    |  |
 *      |  1   | 2|  3 | 4|
 *      |      |  |    |  |
 *      |      |  |    |  |
 *      +------+--+----+--+
 */
static void
stripeDivision(
    BlasFunctionID funcID,
    const CLBlasKargs *args,
    ListHead *seq,
    cl_uint totalCUs)
{
    SolutionStep *step;
    ListNode *i;
    cl_int err;
    cl_device_id device;
    cl_uint nrCU;
    SubproblemDim size, offset, stepSize;
    bool first = true;

    kargsToProbDims(&offset, funcID, args, true);
    kargsToProbDims(&size, funcID, args, false);

    for (i = listNodeFirst(seq); i != seq; i = i->next) {
        step = container_of(i, node, SolutionStep);
        err = getQueueDevice(step->cmdQueue, &device);
        nrCU = deviceComputeUnits(device, &err);

        if (totalCUs == 0) {
            step->cmdQueue = NULL;
            continue;
        }

        stepSize = size;
        if (!first) {
            probDimsToKargs(&(step->args), funcID, &offset, true);
        }

        if (funcID == CLBLAS_GEMV) {
            if (totalCUs != nrCU) {
                stepSize.y = (size_t)(size.y * (double)nrCU / totalCUs + 0.5);
                stepSize.y = align(stepSize.y, DIVISION_ALIGNMENT);
                if (stepSize.y == 0) {
                    step->cmdQueue = NULL;
                }
                else if (stepSize.y > size.y) {
                    stepSize.y = size.y;
                    totalCUs = nrCU;
                }
            }

            offset.y += stepSize.y;
            size.y -= stepSize.y;
        }
        else {
            if (totalCUs != nrCU) {
                stepSize.x = (size_t)(size.x * (double)nrCU / totalCUs + 0.5);
                stepSize.x = align(stepSize.x, DIVISION_ALIGNMENT);
                if (stepSize.x == 0) {
                    step->cmdQueue = NULL;
                }
                else if (stepSize.x > size.x) {
                    stepSize.x = size.x;
                    totalCUs = nrCU;
                }
            }
            offset.x += stepSize.x;
            size.x -= stepSize.x;
        }

        totalCUs -= nrCU;
        probDimsToKargs(&(step->args), funcID, &stepSize, false);
        first = false;
    }
}

/* Rectangular division is done according to the picture:
 *
 *      +------+-----+
 *      |      |  2  |
 *      |      |     |
 *      |  1   +--+--+
 *      |      |3 | 4|
 *      |      |  |  |
 *      +------+--+--+
 *
 * The longest side is divided first.
 */
static void
rectDivision(
     BlasFunctionID funcID,
     const CLBlasKargs *args,
     ListHead *seq,
     cl_uint totalCUs)
 {
     SolutionStep *step, **sortedSteps;
     ListNode *i, *j;
     cl_int err;
     cl_device_id device;
     cl_uint nrCU, k, l;
     SubproblemDim size, offset, stepSize;
     unsigned int nrSteps = 0;

     /* 1. Sort steps according to the number of CU they have */
     /* NOTE: We expect small number of steps, so simple insertion sort
      *       would be enough.
      */

     sortedSteps = calloc(listLength(seq), sizeof(*sortedSteps));
     // assert(sortedSteps != NULL);

     k = 0;
     for (i = listNodeFirst(seq); i != seq; i = i->next, nrSteps++) {
         step = container_of(i, node, SolutionStep);
         err = getQueueDevice(step->cmdQueue, &device);

         sortedSteps[k] = step;
         nrCU = deviceComputeUnits(device, &err);

         for (j = i->next; j != seq; j = j->next) {
             step = container_of(i, node, SolutionStep);
             err = getQueueDevice(step->cmdQueue, &device);

             if (nrCU < deviceComputeUnits(device, &err)) {
                 sortedSteps[k] = step;
                 nrCU = deviceComputeUnits(device, &err);
             }
         }

         k++;
     }

     /* 2. Calculate rectangle sizes */

     kargsToProbDims(&offset, funcID, args, true);
     kargsToProbDims(&size, funcID, args, false);
     stepSize = size;

     for (l = 0; l < k; l++) {
         step = sortedSteps[l];
         err = getQueueDevice(step->cmdQueue, &device);
         nrCU = deviceComputeUnits(device, &err);

         if (totalCUs == 0) {
             step->cmdQueue = NULL;
             continue;
         }

         stepSize = size;
         if (l) {
             probDimsToKargs(&(step->args), funcID, &offset, true);
         }

         if (size.y > size.x) {
             if (totalCUs != nrCU) {
                 stepSize.y = (size_t)(size.y * (double)nrCU / totalCUs + 0.5);
                 stepSize.y = align(stepSize.y, DIVISION_ALIGNMENT);
                 if (stepSize.y > size.y) {
                     stepSize.y = size.y;
                     totalCUs = nrCU;
                 }
                 else if (stepSize.y == 0) {
                     step->cmdQueue = NULL;
                 }
             }
             size.y -= stepSize.y;
             offset.y += stepSize.y;
         }
         else {
             if (totalCUs != nrCU) {
                 stepSize.x = (size_t)(size.x * (double)nrCU / totalCUs + 0.5);
                 stepSize.x = align(stepSize.x, DIVISION_ALIGNMENT);
                 if (stepSize.x > size.x) {
                     stepSize.x = size.x;
                     totalCUs = nrCU;
                 }
                 else if (stepSize.x == 0) {
                     step->cmdQueue = NULL;
                 }
             }
             size.x -= stepSize.x;
             offset.x += stepSize.x;
         }

         probDimsToKargs(&(step->args), funcID, &stepSize, false);

         #ifdef DEBUG_2
         printf("RectDivision:\n");
         printf("\t offM=%d, offN=%d, M=%d, N=%d\n", step->args.offsetM, step->args.offsetN, step->args.M, step->args.N);
         #endif
         totalCUs -= nrCU;
     }

     free(sortedSteps);
}

/* Dividing triangular matrix (N x N) horizontally:
 *
 *      +----+
 *      |\   |
 *      +-\--+
 *      |  \ |
 *      |   \|
 *      +----+
 *
 * Take into consideration the areas of triangles/trapezoids rather than
 * areas of stripes.
 */
static void
triMatrixStripeDivision(
    BlasFunctionID funcID,
    const CLBlasKargs *args,
    ListHead *seq,
    cl_uint totalCUs)
{
    SolutionStep *step;
    ListNode *i;
    cl_int err;
    cl_device_id device;
    cl_uint nrCU;
    SubproblemDim size, offset, stepSize, stepOffset;
    size_t top;

    kargsToProbDims(&offset, funcID, args, true);
    kargsToProbDims(&size, funcID, args, false);
    top = 0;

    if (args->uplo == clblasUpper) {
        offset.y += size.y;
    }
    stepSize = size;

    for (i = listNodeFirst(seq); i != seq; i = i->next) {
        step = container_of(i, node, SolutionStep);
        err = getQueueDevice(step->cmdQueue, &device);
        nrCU = deviceComputeUnits(device, &err);

        if (totalCUs == 0) {
            step->cmdQueue = NULL;
            continue;
        }

        if (args->uplo == clblasLower) {
            stepOffset = offset;
        }

        if (totalCUs != nrCU) {
            stepSize.y = (size_t)(
                sqrt(top * top + (double)nrCU / totalCUs * size.y * (top + size.x)) - top);
            stepSize.y = align(stepSize.y, DIVISION_ALIGNMENT);
            if ((stepSize.y == 0) || (stepSize.y > size.y)) {
                stepSize.y = size.y;
                totalCUs = nrCU;
            }
            else if (stepSize.y == 0) {
                step->cmdQueue = NULL;
            }
            /* We have to add special check because the direction of
             * splitting is 'bottom -> top' for UPLO = clblasUpper.
             */
            else if (offset.y != align(offset.y, DIVISION_ALIGNMENT)) {
                size_t o = align(offset.y - stepSize.y, DIVISION_ALIGNMENT);
                if (o > offset.y) {
                    o -= 2 * DIVISION_ALIGNMENT;
                }
                stepSize.y = offset.y - o;
            }
        }
        else {
            stepSize.y = size.y;
        }

        size.y -= stepSize.y;
        top += stepSize.y;
        if (args->uplo == clblasLower) {
            offset.y += stepSize.y;
        }
        else {
            offset.y -= stepSize.y;
            stepOffset = offset;
        }

        probDimsToKargs(&(step->args), funcID, &stepOffset, true);
        probDimsToKargs(&(step->args), funcID, &stepSize, false);

        totalCUs -= nrCU;
    }
}

static cl_bool
findBestPattern(SolutionStep *step)
{
    cl_uint maxImages;

    maxImages = getQueueMaxImages(step->cmdQueue);

    do {
        /* It may be non first attempt. Ensure that there are not
         * hold images for this step
         */
        releaseStepImgs(step);

        step->patternID = selectPattern( step, maxImages );

        assert(step->patternID != (unsigned int)-1);

		#ifdef DEBUG_2
		printf("select Pattern Done\n");
		#endif

        getStepGranulation(step);
		#ifdef DEBUG_2
		printf("getStepGranulation done \n");
		#endif

        assertGranulation(step->subdims, mempat->nrLevels,
                          &step->pgran, mempat->thLevel);
        if (getStepResources(step))
            break;
    } while (maxImages-- != 0);

    return (maxImages != (cl_uint)-1) ? CL_TRUE : CL_FALSE;
}

void
detectProblemTails(SolutionStep *step)
{
    SubproblemDim globDim, offDim;
    SubproblemDim *subdim;
    KernelExtraFlags kflags = KEXTRA_NO_FLAGS;

    subdim = step->subdims;

    kargsToProbDims(&globDim, step->funcID, &step->args, false);
    kargsToProbDims(&offDim, step->funcID, &step->args, true);

	#ifdef DEBUG_2
	printf("detectProblemTails: subdimy=%d, subdimx=%d, subdimBwidth=%d\n", subdim->y, subdim->x, subdim->bwidth);
	#endif
    if (globDim.y % subdim->y) {
        kflags |= KEXTRA_TAILS_M;
    }
    if (globDim.x % subdim->x) {
        kflags |= KEXTRA_TAILS_N;
    }
    if (globDim.bwidth % subdim->bwidth) {
        kflags |= KEXTRA_TAILS_K;
    }
    if (clblasSolvers[step->funcID].memPatterns[step->patternID].nrLevels > 1) {
        if (globDim.y % subdim[1].y) {
            kflags |= KEXTRA_TAILS_M_LOWER;
        }
        if (globDim.x % subdim[1].x) {
            kflags |= KEXTRA_TAILS_N_LOWER;
        }
        if (globDim.bwidth % subdim[1].bwidth) {
            kflags |= KEXTRA_TAILS_K_LOWER;
        }
    }
    else {
        kflags |= (kflags & KEXTRA_TAILS_M) != 0 ? KEXTRA_TAILS_M_LOWER : 0;
        kflags |= (kflags & KEXTRA_TAILS_N) != 0 ? KEXTRA_TAILS_N_LOWER : 0;
        kflags |= (kflags & KEXTRA_TAILS_K) != 0 ? KEXTRA_TAILS_K_LOWER : 0;
    }

    // clean tails flags
    step->extraFlags &= ~(KEXTRA_TAILS_M | KEXTRA_TAILS_N | KEXTRA_TAILS_K
                          | KEXTRA_TAILS_M_LOWER
                          | KEXTRA_TAILS_N_LOWER
                          | KEXTRA_TAILS_K_LOWER);
    // set tails flags
    step->extraFlags |= kflags;
}

void
detectOffsets(SolutionStep *step)
{
    const CLBlasKargs *args = &(step->args);
    KernelExtraFlags kflags = step->extraFlags;

    if (args->offsetM) {
        kflags |= KEXTRA_STARTM_NOT_ZERO;
    }
    if (args->offsetN) {
        kflags |= KEXTRA_STARTN_NOT_ZERO;
    }
    if (args->offA) {
        kflags |= KEXTRA_A_OFF_NOT_ZERO;
    }
    if (args->offBX) {
        kflags |= KEXTRA_BX_OFF_NOT_ZERO;
    }
    if (args->offCY) {
        kflags |= KEXTRA_CY_OFF_NOT_ZERO;
    }

    step->extraFlags = kflags;
}

//-----------------------------------------------------------------------------

static unsigned int
legacySelectPattern(
    BlasFunctionID funcID,
    unsigned int maxImages)
{
    unsigned int id, i, n;
    MatrixRole mrole;
    MemoryPattern *pat;
    int score, maxScore = -1;

    id = -1;
    /*
     * Lookup all patterns, and assign a score per each matrix for
     * each pattern:
     * 0 - matrix is not cached
     * 2 - matrix is cached and stored in an image
     * 3 - matrix is cached and not stored in an image
     *
     * Find the pattern with the best score
     */
    pat = clblasSolvers[funcID].memPatterns;

    for (i = 0; i < clblasSolvers[funcID].nrPatterns; i++, pat++) {
        score = 0;
        n = 0;

        for (mrole = MATRIX_A; mrole <= MATRIX_B; mrole++) {
            if (isMatrixCached(pat, mrole)) {
                if (isMatrixInImage(pat, mrole)) {
                    n++;
                    score += 2;
                }
                else {
                    score += 3;
                }
            }
        }

        if (n > maxImages) {
            continue;
        }

        if (score > maxScore) {
            maxScore = score;
            id = i;
        }
    }

    return id;
}
//-----------------------------------------------------------------------------

unsigned int
selectPattern( SolutionStep* pStep,
    unsigned int maxImages )
{
    unsigned int i = 0;
    int selPatt = -1;
    int perf = -1;
    int maxPerf = -1;
    int funcID = pStep->funcID;
    unsigned int kflags = pStep->extraFlags;

    if (clblasSolvers[funcID].defaultPattern != -1) {
// assert(clblasSolvers[funcID].defaultPattern < clblasSolvers[funcID].nrPatterns);
        return clblasSolvers[funcID].defaultPattern;
    }

	// select best-performing pattern for current case
	for( i = 0; i < clblasSolvers[funcID].nrPatterns; i++ ){

		if( NULL != clblasSolvers[funcID].memPatterns[i].sops->getPatternPerf ){

	        perf = clblasSolvers[funcID].memPatterns[i].sops->getPatternPerf(
                kflags,
                (void*)&pStep->args);

            if( perf > maxPerf ){
                selPatt = i;
                maxPerf = perf;
            }
	    }
	    // if not all patterns provide performace estimation functions
	    // use legacy pattern selection
	    else{
	        return legacySelectPattern( funcID, maxImages );
	    }
	}

    return selPatt;
}

//-----------------------------------------------------------------------------

/*
 * Check if tile sizes exceed the entire problem and adjust them
 * accordingly if yes
 */
bool
dimensionsExceedProblemSize(SolutionStep *step) {
    SubproblemDim probDim;
    SubproblemDim *dims = step->subdims;
    BlasFunctionID funcID = step->funcID;
    MemoryPattern *mempat =
            &clblasSolvers[funcID].memPatterns[step->patternID];

    /*
     * Looks like kernels of other functions handle the case themselves
     * and don't expect that everyone can adjust chosen decomposition
     */
    if (!( (funcID == CLBLAS_GEMV) ||
           (funcID == CLBLAS_SYMV) ||
           (funcID == CLBLAS_GEMM) ||
           (funcID == CLBLAS_TRMM) ||
           (funcID == CLBLAS_TRSM) ||
           (funcID == CLBLAS_SYRK) ||
           (funcID == CLBLAS_SYR2K)) ) {

        return false;
    }


    kargsToProbDims(&probDim, step->funcID, &step->args, false);

    if (mempat->nrLevels != 2) {
        return false;
    }
    dims = &dims[1];

    if (dims->x > probDim.x ||
        dims->y > probDim.y ||
        dims->bwidth > probDim.bwidth) {
        return true;
    }

    return false;
}

void
getMinimalStepGranulation(SolutionStep *step)
{
    SubproblemDim *decompDims = NULL;
    SubproblemDim probDims[2];
    size_t factor = 0;

    // EINVAL
    if( NULL == step ){
        return;
    }

    if (step->funcID == CLBLAS_GEMM2)
	{
		return;
	}

    kargsToProbDims( probDims, step->funcID, &step->args, false);
    decompDims = step->subdims;

    // All exceeding dimensions are set to 1

     if ( decompDims[1].itemX > probDims->x ) {

         factor = decompDims[1].itemX;
         decompDims[1].itemX = 1;
         decompDims[1].x /= factor;
         decompDims[0].itemX /= factor;
         decompDims[0].x /= factor;
     }

     if ( decompDims[1].itemY > probDims->y ) {

         factor = decompDims[1].itemY;
         decompDims[1].itemY = 1;
         decompDims[1].y /= factor;
         decompDims[0].itemY /= factor;
         decompDims[0].y /= factor;
     }

     if( decompDims[1].bwidth > probDims->bwidth ){
         decompDims[0].bwidth /= decompDims[1].bwidth;
         decompDims[1].bwidth = 1;
     }
}

void
getStepGranulation(SolutionStep *step)
{
	SubproblemDim *dims = step->subdims;
    cl_device_id devID;
    double time;
    int status = GF_ERROR;
    size_t MNK;

	#ifdef DEBUG_2
	printf("getStepGranulation called........\n");
	#endif

    MemoryPattern *mempat =
            &clblasSolvers[step->funcID].memPatterns[step->patternID];

	#ifdef DEBUG_2
	printf("Got mempat structure.........0x%p\n", mempat);
	#endif


	#ifdef DEBUG_2
	if ( mempat == NULL)
	{
		printf("mempat pointer is NULL...\n");
	} else {
		printf("mempat pointer is non-null..\n");
		if (mempat->sops == NULL)
			printf("sops is NULL\n");
		else
			if (mempat->sops->getFlags == NULL)
				printf("getFlags() is NULL\n");
		fflush(stdout);
	}
	#endif

	getQueueDevice(step->cmdQueue, &devID);

	#ifdef DEBUG_2
	printf("QueueDevice done...\n");
	#endif


    // try to load decomposition info from the storage

    /*
     * FIXME: It's a workaround so that to avoid getting some decomposition
     *         sizes leading to strange hang ups
     */
    if (!avoidLoadFromStorage(step)) {
		#ifdef DEBUG_2
		printf("!avoidLoadFromStorage...Inside if\n");
		#endif

        MNK = (step->args.M + step->args.N + step->args.K)/3;
        if (mempat->sops->innerDecompositionAxis) {
            size_t ld;
            // bas - banks aligned size, in bytes, should be
            // number of channels * bytes per channel
            // here it is set to 8*256 = 2048 = 512 floats
            size_t bas = 8*256;
            if (mempat->sops->innerDecompositionAxis(&step->args) ==
                    DECOMP_AXIS_X) {
                ld = step->args.ldb.matrix;
            }
            else {
                ld = step->args.lda.matrix;
            }

            if ((ld * dtypeSize(step->args.dtype)) % bas == 0) {
                //special bad case
                MNK = 0;
            }
        }

		if( step->funcID != CLBLAS_GEMM2 )
		{
			status = getGranularityInfo(&step->device, mempat->name,
										step->args.dtype, step->extraFlags,
										(int)MNK, dims, &step->pgran, &time);
		}
        /*
         * Disable blocking for implementations dealing with cache reads
         * from the global memory
         */
        //if (!(isLdsUsed(mempat) || (square && mempat->nrLevels == 2))) {
        //    dims[0].bwidth = dims[1].bwidth;
        //}
    }
	#ifdef DEBUG_2
	printf("isLoadFromStorage done..\n");
	#endif

	//Query solver for default granulation
    if (status == GF_ERROR) {
		// temporary mock, untill all solvers will return required default problem granulation
		// TODO: deprecate the getDefaultStepGranulation(step) function
		if(NULL==mempat->sops->getDefaultDecomp)
		{
			getDefaultStepGranulation(step);
		}
		else
		{
			mempat->sops->getDefaultDecomp( &step->pgran,
			    step->subdims,
			    MAX_SUBDIMS,
			    (void*)&step->args);
		}
    }
    if (dimensionsExceedProblemSize(step)) {
        getMinimalStepGranulation(step);
    }
}

void
getDefaultStepGranulation(SolutionStep *step)
{
    unsigned int nrFloats;
    MemoryPattern *mempat =
            &clblasSolvers[step->funcID].memPatterns[step->patternID];
    SubproblemDim *dims = step->subdims;
    cl_ulong ldsSize;
    size_t wgX, wgY;
    bool square;
    SDimComponent component = SDIM_BWIDTH;
    DataType dtype = step->args.dtype;
    size_t tsize = dtypeSize(dtype);
    unsigned int i;
    SolverFlags sflags;
    unsigned int bcoeff;
    bool bothCached, fixedBw = false;
    cl_device_id devID;
    PGranularity *pgran = &step->pgran;
	size_t maxWorkGroupSize;
	int vecLen;
	size_t subdimyFactor = 1;
	size_t subdimxFactor = 1;

	#ifdef DEBUG_2
	printf("getDefaultStepGranualtion called...\n");
	#endif
    nrFloats = (unsigned int)(dtypeSize(dtype) / sizeof(cl_float));
    square = ((mempat->sops->getFlags() & SF_TOP_INPUT_SQUARE_BLOCKS) != 0);
    bothCached = isMatrixCached(mempat, MATRIX_A) &&
                 isMatrixCached(mempat, MATRIX_B);
    if (step->cmdQueue != NULL) {
        getQueueDevice(step->cmdQueue, &devID);
    }
    else {
        devID = step->device.id;
    }
    clGetDeviceInfo(devID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ldsSize),
                    &ldsSize, NULL);
	clGetDeviceInfo(devID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
					sizeof(size_t), &maxWorkGroupSize, NULL);

    /*
     * Setup dimensions allowing to use more or less effectively the local
     * memory or cache;
     */

    if (square) {
        dims[0].x = (dtype == TYPE_COMPLEX_DOUBLE) ? 16 : 32;
        /*
         * FIXME: for now, we restrict ourselves with square blocks due
         *        to compilation issues
         */
        dims[0].y = dims[0].x; //(dtype == TYPE_FLOAT) ? 32 : 16
        dims[0].bwidth = dims[0].y;
        bcoeff = nrFloats;
        wgY = DEFAULT_BUFS_LSIZE_0;
        wgX = DEFAULT_BUFS_LSIZE_1;
	} else {
        bcoeff = (dtype == TYPE_COMPLEX_DOUBLE) ? 2 : 1;

        if (bothCached) {
            wgY = DEFAULT_CACHED_BUFS_LSIZE_0;
            wgX = DEFAULT_CACHED_BUFS_LSIZE_1;
        }
        else {
            wgY = DEFAULT_BUFS_LSIZE_0;
            wgX = DEFAULT_BUFS_LSIZE_1;
        }

		if (step->funcID == CLBLAS_GEMM2)
		{
			subdimyFactor = 2;
			subdimxFactor = 1;
            bcoeff = 4; // 16/bcoeff = 4 - Thats the panel width we want
		}

   		if ((step->funcID == CLBLAS_TRMV) || (step->funcID == CLBLAS_HEMV))  {
			if (maxWorkGroupSize >= 256)
			{
				wgX = 16;
				wgY = 16;
			} else if (maxWorkGroupSize >= 128)
			{
				wgX = 8;
				wgY = 16;
			} else {
				//
				// PENDING: What if maxWorkGroupSize < 64 ????
				//
				wgX = 8;
				wgY = 8;
			}
		}


        /*
         * Set block sizes such so the work group would access the whole
         * memory channel or not exceed cache associativity for the modern
         * AMD GPU families.
         *
         * FIXME: throw the hardcoded constants away
         */
        if (isMatrixInImage(mempat, MATRIX_A) ||
            isMatrixAccessColMaj(step->funcID, step->extraFlags, MATRIX_A)) {

            dims[0].y = (64 * subdimyFactor) / nrFloats;
            fixedBw = true;
        }
        else {
            dims[0].y = (32 * subdimyFactor);
        }

        if (isMatrixInImage(mempat, MATRIX_B) ||
            isMatrixAccessColMaj(step->funcID, step->extraFlags, MATRIX_B)) {

            dims[0].x = (64 * subdimxFactor) / nrFloats;
            fixedBw = true;
        }
        else {
            dims[0].x = (32 * subdimxFactor);
        }

   		if (step->funcID == CLBLAS_GEMM2)  {
			int count=0;

            //
			// NOTE:
			// wgX and wgY setting for this function must be the same as
			// CLBLAS_GEMM_TAIL below.
			//
			//vecLen = sizeof(cl_float4) / dtypeSize(step->args.dtype);
            //
            // PENDING: 16x16 works best on CYPRESS and 16x8 for Cayman
            //
			wgY = 8*subdimyFactor;
			wgX = 8*subdimxFactor;
			while((wgY * wgX) > maxWorkGroupSize)
			{
				if (count & 1)
				{
					wgY /= 2;
					dims[0].y /= 2;
			    } else {
					wgX /= 2;
					dims[0].x /= 2;
			    }
				count++;
		    }
		}

   		if (step->funcID == CLBLAS_GEMM_TAIL)  {
			//
			// NOTE: wgY and wgX must be same as what is set for CLBLAS_GEMM2 above
			//
			vecLen = 1;

				//
				// PENDING: What if maxWorkGroupSize < 64 ????
				//
				wgY = 8;
				wgX = 8;
				dims[0].y = wgY ;
				dims[0].x = wgX ;
			}

        if((step->funcID == CLBLAS_TRSV) || (step->funcID == CLBLAS_TRSV_GEMV))
        {
            wgY = 8;
            wgX = 8;
            dims[0].y = 64;
            dims[0].x = 64;
        }

        dims[0].bwidth = 16 / bcoeff;
    }

    /*
     * Prevent using more than 1/2 of LDS so as to have at least 2 work groups
     * per compute unit
     */
    if (ldsSize && mempat->sops->isFitToLDS) {
        ldsSize /= 2;

        while (!mempat->sops->isFitToLDS(dims, dtype, ldsSize, &step->args)) {
            /*
             * decrease current component and setup this one to decrease
             * on the next step; do not grow down block width below the
             * value with which the block line takes size of a float4 vector
             */
            if (square) {
                dims[0].x /= 2;
                dims[0].y /= 2;
                dims[0].bwidth /= 2;
            }
            else {
                switch (component) {
                case SDIM_X:
                    dims[0].x /= 2;
                    if (dims[0].bwidth * tsize == sizeof(cl_float4)) {
                        component = SDIM_Y;
                    }
                    else {
                        component = SDIM_BWIDTH;
                    }
                    break;
                case SDIM_Y:
                    dims[0].y /= 2;
                    component = SDIM_X;
                    break;
                case SDIM_BWIDTH:
                    dims[0].bwidth /= 2;
                    component = SDIM_Y;
                    break;
                }
            }
        }

        assert(dims[0].x > 0 && dims[0].y > 0 &&
               dims[0].bwidth * tsize >= sizeof(cl_float4));
    }

    /*
     * adjust local size if a subproblem is not divisible
     * between all local threads
     */
    for (; (wgY > 1) && (dims[0].y < wgY); wgY /= 2) { }
    for (; (wgX > 1) && (dims[0].x < wgX); wgX /= 2) { }

    sflags = mempat->sops->getFlags();
    if (sflags & SF_WSPACE_2D) {
        pgran->wgDim = 2;
        dims[0].itemY = dims[0].y;
        pgran->wgSize[0] = (unsigned int)wgY;
        pgran->wgSize[1] = (unsigned int)wgX;
    }
    else {
        pgran->wgDim = 1;
        pgran->wgSize[0] = (unsigned int)(wgX * wgY);
        pgran->wgSize[1] = 1;
    }

    /*
     * Divide the work between threads
     */
    dims[1].itemX = dims[0].x / wgX;
    dims[1].itemY = dims[0].y / wgY;
    dims[1].x = dims[1].itemX;
    dims[1].y = dims[1].itemY;

    if ((mempat->nrLevels == 1) && square) {
        dims[1].bwidth = dims[1].y;
    }
    else {
        i = fixedBw ? 4 : (8 / nrFloats);
        dims[1].bwidth = szmin(i, dims[0].bwidth);
    }

    dims[0].itemX = dims[0].x;
    dims[0].itemY = dims[0].y;

    /*
     * FIXME: Now, there are issues with generating kernels with non square
     *        tiles in LDS less TRSM due to some fundamental restriction
     *        of the core generator logic. Deprecate this kludge when
     *        they will be eliminated
     */
#if 1
    if ((step->funcID == CLBLAS_TRSM) && (step->patternID == 2)) {
        dims[1].bwidth = dims[1].y;
    }
#endif
    if (funcHasTriangMatrix(step->funcID) && (pgran->wgDim == 1)) {
        dims[0].itemY = SUBDIM_UNUSED;
        if (mempat->nrLevels == 1) {
            dims[1].itemY = SUBDIM_UNUSED;
        }
    }

    if (!(isLdsUsed(mempat) || (square && mempat->nrLevels == 2))) {
        dims[0].bwidth = dims[1].bwidth;
    }
    /*
     * Ensure decomposition size for vectors in case
     * of level 2 routines equal to 1.
     */
    if (funcBlasLevel(step->funcID) == 2) {
        size_t xBlocks;

        xBlocks = dims[0].x / dims[1].x;
        dims[0].x = 1;
        dims[1].itemX = 1;
        dims[1].x = 1;
        dims[0].bwidth = dims[1].bwidth * xBlocks;
    }

    // fixup work group size in respect with desired work dispatch order
    if ((pgran->wgDim == 2) && mempat->sops->innerDecompositionAxis) {
        if (mempat->sops->innerDecompositionAxis(&step->args) ==
            DECOMP_AXIS_X) {

            unsigned int u;

            u = pgran->wgSize[0];
            pgran->wgSize[0] = pgran->wgSize[1];
            pgran->wgSize[1] = u;
        }
    }
    //printf("GDSG: suby = %lu, subx = %lu, bwidth0=%lu, bwidth1=%lu\n", dims[0].y, dims[0].x, dims[0].bwidth, dims[1].bwidth);
}

static bool
avoidLoadFromStorage(SolutionStep *step)
{
    bool notDiv;
    MemoryPattern *mempat =
            &clblasSolvers[step->funcID].memPatterns[step->patternID];
    bool bothCached = isMatrixCached(mempat, MATRIX_A) &&
                      isMatrixCached(mempat, MATRIX_B);

    if (bothCached) {
        return false;
    }

    if ((step->funcID == CLBLAS_GEMM2) && ((step->args.pigFuncID == CLBLAS_SYMM) || (step->args.pigFuncID == CLBLAS_HEMM)) )
    {
        // FIXME: Assuming that returning "true" will load defaultDecomposition sizes
        //        But the statement below on TRSM is a bit confusing.
        //        Returning FALSE  here will load from storage in getStepGranulation()
        return true;
    }

    /*
     * don't load from storage data for LDS gemm,
     * not integrally divisible
     */
    notDiv = (step->args.M % 64) || (step->args.N % 64) || (step->args.K % 64);

    return ((step->funcID == CLBLAS_GEMM) && notDiv);
}

static bool
getStepResources(SolutionStep *step)
{
    int i = 0;
    size_t tsize;
    unsigned int vecLen;
    size_t minWidth, minHeight, bestHeight, minSize, bestSize;
    MatrixRole mrole;
    cl_device_id devID;
    cl_context ctx;
    MemoryPattern *mempat;
    SubproblemDim probDim;
    CLBlasKargs *kargs = &step->args;
    bool ret = true;

    tsize = dtypeSize(kargs->dtype);
    vecLen = (unsigned int)(sizeof(cl_float4) / tsize);
    kargsToProbDims(&probDim, step->funcID, &step->args, false);
    getQueueContext(step->cmdQueue, &ctx);
    getQueueDevice(step->cmdQueue, &devID);

    mempat = &(clblasSolvers[step->funcID].memPatterns[step->patternID]);

    for (mrole = MATRIX_A, i = 0; mrole < MATRIX_C; mrole++) {
        if (isMatrixInImage(mempat, mrole)) {
            if (step->funcID == CLBLAS_TRSM) {
                //blocks
                unsigned int packRate;
                clblasOrder packOrder;
                size_t pitch;
                size_t matrWidth, matrHeight;
                CLBLASKernExtra extra;

                memset(&extra, 0, sizeof(extra));
                extra.dtype = kargs->dtype;
                extra.flags = step->extraFlags;

                mempat->sops->imgPackMode(&extra,
                                          step->subdims, mrole,
                                          &packRate, &packOrder);

                // minimal size parameters
                pitch = matrBlockPitch(step->subdims, mrole, kargs->dtype,
                                        kargs->side);
                matrWidth = matrBlockPitch(&probDim, mrole, kargs->dtype,
                                           kargs->side);
                matrHeight = matrBlockHeight(&probDim, mrole, kargs->side);

                //One panel should fit to image
                if (packOrder == clblasRowMajor) {
                    minWidth = divRoundUp(matrWidth, pitch) * pitch / vecLen;
                    minHeight = packRate;

                    minSize = minWidth * minHeight;
                    // size of image to store all blocks
                    bestSize = minHeight * (minWidth + pitch / vecLen) *
                               divRoundUp(matrHeight, packRate) / 2;
                }
                else {
                    minWidth = pitch / vecLen;
                    minHeight = divRoundUp(matrHeight, packRate) * packRate;

                    minSize = minWidth * minHeight;
                    bestSize = minWidth * (minHeight + packRate) *
                               divRoundUp(matrWidth, pitch) / 2;
                }
                minSize = bestSize;
            }
            else {
                //panels
                getSuitableImageSizes(&minWidth, &minHeight, &bestHeight,
                                      mrole, kargs, vecLen, step->subdims);
                minSize = minWidth * minHeight;
                bestSize = minWidth * bestHeight;
            }

            kargs->scimage[i] = getSCImage(ctx, devID, bestSize,
                                           minSize, minWidth);
            if (kargs->scimage[i] == NULL) {
                ret = false;
                break;
            }

            i++;
        }
    }

    return ret;
}

static void
getSuitableImageSizes(
    size_t *minWidth,
    size_t *minHeight,
    size_t *bestHeight,
    MatrixRole mrole,
    CLBlasKargs *kargs,
    unsigned int vecLen,
    SubproblemDim *subdims)
{
    size_t alignedM, alignedN, alignedK;
    alignedM = divRoundUp(kargs->M, subdims->y);
    alignedM *= subdims->y;
    alignedN = divRoundUp(kargs->N, subdims->x);
    alignedN *= subdims->x;
    alignedK = divRoundUp(kargs->K, subdims->bwidth);
    alignedK *= subdims->bwidth;
    switch (mrole) {
        case MATRIX_A:
            *minWidth = alignedK / vecLen;
            *bestHeight = alignedM;
            *minHeight = subdims->y;
             break;
        case MATRIX_B:
            *minWidth = alignedK / vecLen;
            *bestHeight = alignedN;
            *minHeight = subdims->x;
            break;
        case MATRIX_C:
            *minWidth = alignedN / vecLen;
            *bestHeight = alignedM;
            *minHeight = subdims->y;
            break;
        default:
            break;
    }
}

/*
 * TRxM -> TRxM + GEMM + TRxM
 *
 * When talking about matrix A splitting the following numbering is used:
 *
 *     +---+---+
 *     | 1 | 2 |
 *     +---+---+
 *     | 3 | 4 |
 *     +---+---+
 */
static ListNode*
decomposeTRXMStep(SolutionStep *step)
{
    CLBlasKargs *kargs = &(step->args);
    SolutionStep *trxm1 = NULL, *gemm = NULL, *trxm2 = NULL, *tmp;
    clblasUplo position;
    SubproblemDim size, offset;
    int swap;
    cl_float f;
    cl_double d;
    clblasImplementation impl = clblasDefaultGemm;
    size_t offsetK = 0;

    // skip decomposition for a trmm case which works faster without it
    if (step->funcID == CLBLAS_TRMM && !isDoubleBasedType(step->args.dtype) &&
        isMatrixAccessColMaj(step->funcID, step->extraFlags, MATRIX_B)) {
        return &(step->node);
    }

    /* Implementation specific checks */

    if ((getGemmPreferredPattern() != clblasDefaultGemm) &&
        (getGemmPreferredPattern() != clblasBlockGemmWithCaching)) {

        return &(step->node);
    }
    if (step->funcID == CLBLAS_TRMM) {
        impl = getTrmmPreferredPattern();
        if ((impl != clblasDefaultTrmm) &&
            (impl != clblasBlockTrmmWithCaching)) {

            return &(step->node);
        }
    }
    else {
        impl = getTrsmPreferredPattern();
        if ((impl != clblasDefaultTrsm) &&
            (impl != clblasBlockTrsmWithCaching) &&
            (impl != clblasBlockTrsmWithoutLds)) {

            return &(step->node);
        }
    }

    if ((kargs->side == clblasLeft) &&
        (kargs->M < DECOMPOSITION_THRESHOLD(step->args.dtype))) {
        return &(step->node);
    }
    if ((kargs->side == clblasRight) &&
        (kargs->N < DECOMPOSITION_THRESHOLD(step->args.dtype))) {
        return &(step->node);
    }

    trxm1 = calloc(1, sizeof(SolutionStep));
    gemm = calloc(1, sizeof(SolutionStep));
    trxm2 = calloc(1, sizeof(SolutionStep));
    if ((trxm1 == NULL) || (gemm == NULL) || (trxm2 == NULL)) {
        if (trxm1 != NULL) {
            free(trxm1);
        }
        if (gemm != NULL) {
            free(gemm);
        }
        if (trxm2 != NULL) {
            free(trxm2);
        }
        return &(step->node);
    }
    memcpy(trxm1, step, sizeof(SolutionStep));
    memcpy(gemm, step, sizeof(SolutionStep));
    memcpy(trxm2, step, sizeof(SolutionStep));

    gemm->funcID = CLBLAS_GEMM;
    gemm->args.C = kargs->B;
    gemm->args.ldc.matrix = kargs->ldb.matrix;
    gemm->args.offCY = kargs->offBX;
    switch (kargs->dtype) {
    case TYPE_FLOAT:
        if (step->funcID == CLBLAS_TRSM) {
            if (gemm->args.alpha.argFloat != 0.0f) {
                gemm->args.alpha.argFloat = -1 / gemm->args.alpha.argFloat;
            }
        }
        gemm->args.beta.argFloat = 1.0f;
        break;
    case TYPE_DOUBLE:
        if (step->funcID == CLBLAS_TRSM) {
            if (gemm->args.alpha.argDouble != 0.0f) {
                gemm->args.alpha.argDouble = -1 / gemm->args.alpha.argDouble;
            }
        }
        gemm->args.beta.argDouble = 1.0f;
        break;
    case TYPE_COMPLEX_FLOAT:
        if (step->funcID == CLBLAS_TRSM) {
            f = CREAL(gemm->args.alpha.argFloatComplex) *
                CREAL(gemm->args.alpha.argFloatComplex) +
                CIMAG(gemm->args.alpha.argFloatComplex) *
                CIMAG(gemm->args.alpha.argFloatComplex);
            if (f != 0.0f) {
                gemm->args.alpha.argFloatComplex = floatComplex(
                    -CREAL(gemm->args.alpha.argFloatComplex) / f,
                     CIMAG(gemm->args.alpha.argFloatComplex) / f);
            }
        }
        gemm->args.beta.argFloatComplex = floatComplex(1.0f, 0.0f);
        break;
    case TYPE_COMPLEX_DOUBLE:
        if (step->funcID == CLBLAS_TRSM) {
            d = CREAL(gemm->args.alpha.argDoubleComplex) *
                CREAL(gemm->args.alpha.argDoubleComplex) +
                CIMAG(gemm->args.alpha.argDoubleComplex) *
                CIMAG(gemm->args.alpha.argDoubleComplex);
            if (d != 0.0f) {
                gemm->args.alpha.argDoubleComplex = doubleComplex(
                    -CREAL(gemm->args.alpha.argDoubleComplex) / d,
                     CIMAG(gemm->args.alpha.argDoubleComplex) / d);
            }
        }
        gemm->args.beta.argDoubleComplex = doubleComplex(1.0f, 0.0f);
        break;
    }

    /* Actual position of matrix A's data to use */
    if (kargs->transA == clblasNoTrans) {
        position = kargs->uplo;
    }
    else {
        position = (kargs->uplo == clblasUpper) ? clblasLower :
                        clblasUpper;
    }

    /* Map trxm1 to A1 */
    kargsToProbDims(&size, trxm1->funcID, &(trxm1->args), false);
    size.y = align(size.y / 2, DIVISION_ALIGNMENT);
    probDimsToKargs(&(trxm1->args), trxm1->funcID, &size, false);

    /* Map trxm2 to A4 */
    kargsToProbDims(&offset, trxm2->funcID, &(trxm2->args), true);
    kargsToProbDims(&size, trxm2->funcID, &(trxm2->args), false);
    offset.y += align(size.y / 2, DIVISION_ALIGNMENT);
    size.y -= align(size.y / 2, DIVISION_ALIGNMENT);
    probDimsToKargs(&(trxm2->args), trxm2->funcID, &offset, true);
    probDimsToKargs(&(trxm2->args), trxm2->funcID, &size, false);


    if (kargs->side == clblasLeft) {
        trxm1->args.K = trxm1->args.M;
        trxm2->args.K = trxm2->args.M;

        gemm->args.transB = clblasNoTrans;

        if (position == clblasUpper) {
            /* Map gemm to A2 */
            kargsToProbDims(&size, gemm->funcID, &(gemm->args), false);
            size.y = align(size.y / 2, DIVISION_ALIGNMENT);
            probDimsToKargs(&(gemm->args), gemm->funcID, &size, false);
            offsetK = align(gemm->args.K / 2, DIVISION_ALIGNMENT);
            gemm->args.K -= align(gemm->args.K / 2, DIVISION_ALIGNMENT);
        }
        else {
            /* Map gemm to A3 */
            kargsToProbDims(&offset, gemm->funcID, &(gemm->args), true);
            kargsToProbDims(&size, gemm->funcID, &(gemm->args), false);
            offset.y += align(size.y / 2, DIVISION_ALIGNMENT);
            size.y -= align(size.y / 2, DIVISION_ALIGNMENT);
            probDimsToKargs(&(gemm->args), gemm->funcID, &offset, true);
            probDimsToKargs(&(gemm->args), gemm->funcID, &size, false);
            gemm->args.K = align(gemm->args.K / 2, DIVISION_ALIGNMENT);
        }
    }
    else {
        trxm1->args.K = trxm1->args.N;
        trxm2->args.K = trxm2->args.N;

        gemm->args.transA = clblasNoTrans;
        gemm->args.A = kargs->B;
        gemm->args.lda.matrix = kargs->ldb.matrix;
        gemm->args.offA = kargs->offBX;
        gemm->args.transB = kargs->transA;
        gemm->args.B = kargs->A;
        gemm->args.ldb.matrix = kargs->lda.matrix;
        gemm->args.offBX = kargs->offA;

        if (position == clblasUpper) {
            /* Map gemm to A2 */
            kargsToProbDims(&offset, gemm->funcID, &(gemm->args), true);
            kargsToProbDims(&size, gemm->funcID, &(gemm->args), false);
            offset.x += align(size.x / 2, DIVISION_ALIGNMENT);
            size.x -= align(size.x / 2, DIVISION_ALIGNMENT);
            probDimsToKargs(&(gemm->args), gemm->funcID, &offset, true);
            probDimsToKargs(&(gemm->args), gemm->funcID, &size, false);
            gemm->args.K = align(gemm->args.K / 2, DIVISION_ALIGNMENT);
        }
        else {
            /* Map gemm to A3 */
            kargsToProbDims(&size, gemm->funcID, &(gemm->args), false);
            size.x = align(size.x / 2, DIVISION_ALIGNMENT);
            probDimsToKargs(&(gemm->args), gemm->funcID, &size, false);
            offsetK = align(gemm->args.K / 2, DIVISION_ALIGNMENT);
            gemm->args.K -= align(gemm->args.K / 2, DIVISION_ALIGNMENT);
        }
    }

    trxm1->extraFlags = clblasArgsToKextraFlags(&(trxm1->args), trxm1->funcID);
    gemm->extraFlags = clblasArgsToKextraFlags(&(gemm->args), gemm->funcID);
    trxm2->extraFlags = clblasArgsToKextraFlags(&(trxm2->args), trxm2->funcID);

    fixupGemmOffsets(&gemm->args, gemm->extraFlags, offsetK);

    /* Swap trxm1 and trxm2 if needed. */

    swap = 0;
    if (kargs->side == clblasLeft) {
        if ((step->funcID == CLBLAS_TRMM) && (position == clblasLower)) {
            swap = 1;
        }
        if ((step->funcID == CLBLAS_TRSM) && (position == clblasUpper)) {
            swap = 1;
        }
    }
    else {
        if ((step->funcID == CLBLAS_TRMM) && (position == clblasUpper)) {
            swap = 1;
        }
        if ((step->funcID == CLBLAS_TRSM) && (position == clblasLower)) {
            swap = 1;
        }
    }
    if (swap) {
        tmp = trxm1;
        trxm1 = trxm2;
        trxm2 = tmp;
    }
    /* Tie the sequence trmm1 - gemm - trmm2 together. */

    trxm1->event = decomposeEventsAlloc();
    trxm1->node.next = &(gemm->node);

    gemm->numEventsInWaitList = 1;
    gemm->eventWaitList = trxm1->event;
    gemm->event = decomposeEventsAlloc();
    gemm->node.prev = &(trxm1->node);
    gemm->node.next = &(trxm2->node);

    trxm2->numEventsInWaitList = 1;
    trxm2->eventWaitList = gemm->event;
    trxm2->node.prev = &(gemm->node);

    /* Insert new sequence instead of current step */

    trxm1->node.prev = step->node.prev;
    (trxm1->node.prev)->next = &(trxm1->node);
    step->node.prev = NULL;

    trxm2->node.next = step->node.next;
    (trxm2->node.next)->prev = &(trxm2->node);
    step->node.next = NULL;

    freeSolutionStep(&(step->node));

    return &(trxm2->node);
}

/*
 *  Decompose a SYRK problem in order to evaluate the diagonal part
 *  separately. It's useful since the compiler allocates huge number
 *  of registers for a code processing the diagonal.
 */
static ListNode*
decomposeSYRKStep(SolutionStep *step)
{
    CLBlasKargs *kargs = &step->args;
    SolutionStep *syrk2 = NULL;
    size_t thresh;
    ListNode *next;

    /*
     * Tail prediction. Believe that tile sizes will not exceed 8.
     * Disable decomposition if there are not subproblem tails at
     * the tile level because it can likely slowdown since diagonal
     * update is optimized. Actual tail detection is done after
     * the math decomposition. So the kludge is forced.
     */
    if ((kargs->M % 8 == 0) && (kargs->N % 8 == 0)) {
        return &(step->node);
    }

    thresh = DECOMPOSITION_THRESHOLD(step->args.dtype);
    if (kargs->M < thresh / 2) {
        return &(step->node);
    }

    syrk2 = malloc(sizeof(SolutionStep));
    if (syrk2 == NULL) {
        return &(step->node);
    }

    step->extraFlags |= KEXTRA_SYRK_SEPARATE_DIAGONAL;
    memcpy(syrk2, step, sizeof(SolutionStep));
    syrk2->extraFlags |= KEXTRA_SYRK_EVALUATE_DIAGONAL;

    next = step->node.next;

    /* Synchronize the steps */

    /*
     * This is to not disturb synchronization between the current and the next
     * step or to put the output user event to the tail of the chain if syrk2
     * is the last step
     */
    syrk2->event = step->event;
    step->event = decomposeEventsAlloc();
    syrk2->numEventsInWaitList = 1;
    syrk2->eventWaitList = step->event;

    /* Insert the additional step to the list */
    step->node.next = &syrk2->node;
    syrk2->node.prev = &step->node;
    syrk2->node.next = next;
    next->prev = &syrk2->node;

    return &(syrk2->node);
}

static ListNode*
decomposeSYR2KStep(SolutionStep *step)
{
    CLBlasKargs *kargs = &(step->args);
    SolutionStep *syrk1 = NULL, *syrk2 = NULL;
    size_t thresh;
    ListNode *node;

    /* SYR2K implementation is done as blocked with cache-usage optimization
     * only. Therefore, no implementation specific checks.
     */

    thresh = DECOMPOSITION_THRESHOLD(step->args.dtype);
    if (kargs->M < thresh / 2) {
        return &(step->node);
    }

    syrk1 = calloc(1, sizeof(SolutionStep));
    syrk2 = calloc(1, sizeof(SolutionStep));
    if ((syrk1 == NULL) || (syrk2 == NULL)) {
        if (syrk1 != NULL) {
            free(syrk1);
        }
        if (syrk2 != NULL) {
            free(syrk2);
        }
        return &(step->node);
    }
    memcpy(syrk1, step, sizeof(SolutionStep));
    memcpy(syrk2, step, sizeof(SolutionStep));

    syrk2->args.A = kargs->B;
    syrk2->args.lda.matrix = kargs->ldb.matrix;
    syrk2->args.offA = kargs->offBX;
    syrk2->args.B = kargs->A;
    syrk2->args.ldb.matrix = kargs->lda.matrix;
    syrk2->args.offBX = kargs->offA;
    switch (kargs->dtype) {
    case TYPE_FLOAT:
        syrk2->args.beta.argFloat = 1.0f;
        break;
    case TYPE_DOUBLE:
        syrk2->args.beta.argDouble = 1.0f;
        break;
    case TYPE_COMPLEX_FLOAT:
        syrk2->args.beta.argFloatComplex = floatComplex(1.0f, 0.0f);
        break;
    case TYPE_COMPLEX_DOUBLE:
        syrk2->args.beta.argDoubleComplex = doubleComplex(1.0f, 0.0f);
        break;
    }

    syrk1->extraFlags = clblasArgsToKextraFlags(&(syrk1->args), syrk1->funcID);
    syrk1->extraFlags &= ~KEXTRA_SYRK_2K_RANK;
    syrk2->extraFlags = clblasArgsToKextraFlags(&(syrk2->args), syrk2->funcID);
    syrk2->extraFlags &= ~KEXTRA_SYRK_2K_RANK;

    /* Tie the sequence syrk1 - syrk2 together. */

    syrk1->event = decomposeEventsAlloc();
    syrk1->node.next = &(syrk2->node);

    syrk2->numEventsInWaitList = 1;
    syrk2->eventWaitList = syrk1->event;
    syrk2->node.prev = &(syrk1->node);

    /* Insert new sequence instead of current step */

    syrk1->node.prev = step->node.prev;
    (syrk1->node.prev)->next = &(syrk1->node);
    step->node.prev = NULL;

    syrk2->node.next = step->node.next;
    (syrk2->node.next)->prev = &(syrk2->node);
    step->node.next = NULL;

    freeSolutionStep(&(step->node));

    /*
     * Now, decompose each of these steps to evaluate the diagonal
     * part in a dedicated kernel
     */
    decomposeSYRKStep(syrk1);
    node = decomposeSYRKStep(syrk2);
    return node;
}
