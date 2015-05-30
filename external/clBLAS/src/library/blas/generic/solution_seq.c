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


#include <stdlib.h>
#include <string.h>
#include <clblas_stddef.h>

#include "matrix_dims.h"
#include "problem_iter.h"
#include "solution_assert.h"
#include "solution_seq.h"

bool VISIBILITY_HIDDEN isMatrixInImage(MemoryPattern *pattern, MatrixRole mrole);
void VISIBILITY_HIDDEN releaseStepImgs(SolutionStep *step);

static cl_int
enqueueKernel(
    SolutionStep *step,
    const Kernel *kernel,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *event);

static void
splitSolutionStep(
    SolutionStep *rem,
    SolutionStep *cut,
    SDimComponent component,
    size_t chunk,
    bool backward);

static cl_int
executeImageStep(
    SolutionStep *step,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *event);

void
freeSolutionSeq(ListHead *seq)
{
    listDoForEachSafe(seq, freeSolutionStep);
    listInitHead(seq);
}

cl_int
executeSolutionSeq(const ListHead *seq)
{
    cl_int err = CL_SUCCESS;
    ListNode *i;
    SolutionStep *step;


    /* Enqueue computing kernels */
    for (i = listNodeFirst(seq); (i != seq) && (err == CL_SUCCESS);
         i = i->next) {

        step = container_of(i, node, SolutionStep);
        if (step->cmdQueue == NULL) {
            continue;
        }

        if (step->args.scimage[0]) {
            err = executeImageStep(step, step->numEventsInWaitList,
                                   step->eventWaitList, step->event);
        }
        else {
			#ifdef DEBUG_2
			printf("enqueueKernel from executreSolutionSeq...\n");
			#endif

            err = enqueueKernel(step,
                                step->kernels[CLBLAS_COMPUTING_KERNEL],
                                step->numEventsInWaitList, step->eventWaitList,
                                step->event);
        }
    }

    return err;
}

/* private functions */

void VISIBILITY_HIDDEN
freeSolutionStep(ListNode *node)
{
    SolutionStep *step = container_of(node, node, SolutionStep);
    int i;

    for (i = 0; i < MAX_CLBLAS_KERNELS_PER_STEP; i++) {
        if (step->kernels[i] != NULL) {
            putKernel(clblasKernelCache, step->kernels[i]);
        }
    }
    releaseStepImgs(step);
    free(step);
}

static cl_int
enqueueKernel(
    SolutionStep *step,
    const Kernel *kernel,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *event)
{
    cl_int err;
    KernelDesc kernelDesc;
    KernelErrorInfo errInfo;
    MemoryPattern *pattern;
    const CLBLASKernExtra *kextra = (const CLBLASKernExtra*)kernel->extra;
    SubproblemDim subdims[MAX_SUBDIMS];

    step->args.kernType = kextra->kernType;
    pattern = &clblasSolvers[step->funcID].memPatterns[step->patternID];
    kernelDesc.workDim = step->pgran.wgDim;

    memcpy(subdims, step->subdims, sizeof(step->subdims));

    if(NULL==pattern->sops->calcThreads)
    {
        SubproblemDim globDim;
        const PGranularity *pgran;

        pgran = (pattern->nrLevels == 1) ? NULL : &step->pgran;
        kargsToProbDims(&globDim, step->funcID, &step->args, false);

        // fixup dimensions in respect with desired work dispatch order
        if ((step->pgran.wgDim == 2) && pattern->sops->innerDecompositionAxis) {
            if (pattern->sops->innerDecompositionAxis(&step->args) ==
                DECOMP_AXIS_X) {

                /*
                 * these dimensions will not used more anywhere, so we can
                 * just swap them
                 */
                swapDimXY(&subdims[0]);
                swapDimXY(&subdims[1]);
                swapDimXY(&globDim);
            }
        }

        calcGlobalThreads(kernelDesc.globalThreads, subdims,
                          pgran, globDim.y, globDim.x);
    }
    else
    {
		#ifdef DEBUG_2
		printf("calcThreads is defined\n");
		#endif

		pattern->sops->calcThreads(	kernelDesc.globalThreads,
									subdims,
									&step->pgran,
									&step->args,
									kextra);
    }

    //
    // Store the numWGSpawned for this kernel
    // This size can be used by sequence-steps down the line
    // e.g. Reduction of intermediate results of each work group
    //
    step->pgran.numWGSpawned[0] = kernelDesc.globalThreads[0] / step->pgran.wgSize[0];
    step->pgran.numWGSpawned[1] = kernelDesc.globalThreads[1] / step->pgran.wgSize[1];

    kernelDesc.localThreads[0] = step->pgran.wgSize[0];
    kernelDesc.localThreads[1] = step->pgran.wgSize[1];
    kernelDesc.workDim = step->pgran.wgDim;
    kernelDesc.waitListSize = numEventsInWaitList;
    kernelDesc.eventWaitList = eventWaitList;
    kernelDesc.nowait = 1;
    kernelDesc.event = event;
    kernelDesc.needExecTime = 0;

    memset(kernelDesc.args, 0, sizeof(KernelArg) * MAX_KERNEL_ARGS);
    pattern->sops->assignKargs(kernelDesc.args, (const void*)&(step->args),
                               kextra);

    errInfo.wrongArg = 0;
    errInfo.phase = 0;

    /*
     * TODO: log launchClKernel errors
     */
    dumpKernel(step, kextra->kernType);

    err = clCreateKernelsInProgram(kernel->program, 1, &kernelDesc.kernel,
                                   NULL);
    if (err == CL_SUCCESS) {
        err = launchClKernel(&kernelDesc, step->cmdQueue, &errInfo);
        clReleaseKernel(kernelDesc.kernel);
    }

    return err;
}

bool VISIBILITY_HIDDEN
isMatrixInImage(
    MemoryPattern *pattern,
    MatrixRole mrole)
{
    const CLBLASMpatExtra *extra = (const CLBLASMpatExtra*)pattern->extra;
    bool ret = false;

    if (extra != NULL) {
        switch (mrole) {
        case MATRIX_A:
            ret = (extra->mobjA == CLMEM_IMAGE);
            break;
        case MATRIX_B:
            ret = (extra->mobjB == CLMEM_IMAGE);
            break;
        default:
            break;
        }
    }

    return ret;
}

void VISIBILITY_HIDDEN
releaseStepImgs(SolutionStep *step)
{
    int i;
    cl_mem *imgs = step->args.scimage;
    cl_device_id devID = NULL;;

    for (i = 0; (i < 2) && (imgs[i] != NULL); i++) {
        if (devID == NULL) {
            getQueueDevice(step->cmdQueue, &devID);
        }
        putSCImage(devID, imgs[i]);
        imgs[i] = NULL; //to avoid double release
    }
}

static cl_int
executeImageStep(
    SolutionStep *step,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *event)
{
    SolutionStep outerStep, innerStep, execStep;
    cl_int err = CL_SUCCESS;
    int currImg = 0;
    size_t imgWidth, imgHeight;
    size_t ha, hb;
    size_t maxPanels[MATRIX_ROLES_NUMBER], maxBlocks[MATRIX_ROLES_NUMBER];
    size_t off;
    SubproblemDim wholeDim;
    MatrixRole mrole;
    CLBlasKargs *kargs = &step->args;
    cl_mem *imgs = kargs->scimage;
    MemoryPattern *mempat = &clblasSolvers[step->funcID].memPatterns[step->patternID];
    ProblemIterator innerIter, outerIter;
    int oend = 0, iend;
    SDimComponent comp[2];
    bool backward;
    ListHead doneSteps;
    CLBlasKernelType ktype;

    kargsToProbDims(&wholeDim, step->funcID, kargs, false);
    memset(maxPanels, 0, sizeof(maxPanels));
    memset(maxBlocks, 0, sizeof(maxPanels));

    memcpy(&outerStep, step, sizeof(SolutionStep));
    memcpy(&execStep, step, sizeof(SolutionStep));
    listInitHead(&doneSteps);

    /*
     * Cover the whole problem with dimension which matrix blocks are
     * fitted to images at.
     */

    for (mrole = MATRIX_A; mrole < MATRIX_C; mrole++) {
        if (!isMatrixInImage(mempat, mrole)) {
            continue;
        }

        clGetImageInfo(imgs[currImg], CL_IMAGE_WIDTH, sizeof(imgWidth),
                       &imgWidth, NULL);
        clGetImageInfo(imgs[currImg], CL_IMAGE_HEIGHT, sizeof(imgHeight),
                       &imgHeight, NULL);

        if (step->funcID == CLBLAS_TRSM) {
            maxPanels[mrole] = 0;
            maxBlocks[mrole] = 0;
        } else {
            maxPanels[mrole] = imgHeight / matrBlockHeight(step->subdims, mrole,
                                                           clblasLeft);
        }
        currImg++;
    }

    /*
     * for GEMM function we can take both the matrices as outer, it depends on
     * their sizes and image sizes
     */
    if (step->funcID == CLBLAS_GEMM) {
        size_t dx, dy;

        // FIXME: check which of them use really an image

        ha = matrBlockHeight(&wholeDim, MATRIX_A, clblasLeft);
        hb = matrBlockHeight(&wholeDim, MATRIX_B, clblasLeft);

        dx = maxPanels[MATRIX_B] * matrBlockHeight(step->subdims, MATRIX_B,
                                                   clblasLeft);
        dy = maxPanels[MATRIX_A] * matrBlockHeight(step->subdims, MATRIX_A,
                                                   clblasLeft);

        // hb + (hb*ha)/dx < ha + (ha*hb)/dy
        if ((hb / ha) < (1 + hb / dy) / (1 + ha / dx)) {
            mrole = MATRIX_B;
        }
        else {
            mrole = MATRIX_A;
        }
    }
    else {
        mrole = MATRIX_B;
    }
    /*
     * Let's cover the whole image based step.
     * Pattern iterator is used for traversing
     */
    initProblemIterator(&outerIter, step->funcID, mrole, kargs,
                        maxPanels[mrole], maxBlocks[mrole], step->subdims);
    if (mrole == MATRIX_B) {
        comp[0] = SDIM_X;
        comp[1] = SDIM_Y;
        mrole = MATRIX_A;
    }
    else {
        comp[0] = SDIM_Y;
        comp[1] = SDIM_X;
        mrole = MATRIX_B;
    }
    initProblemIterator(&innerIter, step->funcID, mrole,
                        kargs, maxPanels[mrole], maxBlocks[mrole],
                        step->subdims);
    backward = isIterBackward(&innerIter);

    /*
     * Difference in overflowing checking in the outer and inner loops
     * is due to
     */
    do {
        iteratorReset(&innerIter);
        iend = 0;
        oend = iterateProblem(&outerIter);
        off = iterLastOffset(&outerIter);

        splitSolutionStep(&outerStep, &execStep, comp[0],
                                  off, false);
        if (execStep.funcID == CLBLAS_GEMM) {
            fixupGemmOffsets(&execStep.args, execStep.extraFlags, 0);
        }

        memcpy(&innerStep, &execStep, sizeof(SolutionStep));

        ktype = (comp[0] == SDIM_Y) ? CLBLAS_PREP_A_KERNEL :
                                      CLBLAS_PREP_B_KERNEL;

        if (execStep.kernels[ktype] != NULL) {
            err = enqueueKernel(&execStep, execStep.kernels[ktype],
                                numEventsInWaitList, eventWaitList, event);
            if (err != CL_SUCCESS) {
                 break;
            }
        }

        do {
            iend = iterateProblem(&innerIter);
            off = iterLastOffset(&innerIter);
            splitSolutionStep(&innerStep, &execStep,
                              comp[1], off, backward);
            if (execStep.funcID == CLBLAS_GEMM) {
                fixupGemmOffsets(&execStep.args, execStep.extraFlags, 0);
            }

            assertImageSubstep(step, &execStep, &doneSteps);

            ktype = (comp[1] == SDIM_Y) ? CLBLAS_PREP_A_KERNEL :
                                          CLBLAS_PREP_B_KERNEL;
            if (execStep.kernels[ktype] != NULL) {
                err = enqueueKernel(&execStep, execStep.kernels[ktype],
                                    numEventsInWaitList, eventWaitList, event);
            }
            if (err == CL_SUCCESS) {
                err = enqueueKernel(&execStep,
                                    execStep.kernels[CLBLAS_COMPUTING_KERNEL],
                                    numEventsInWaitList, eventWaitList,
                                    event);
            }
        } while (!iend && (err == CL_SUCCESS));
    } while (!oend && (err == CL_SUCCESS));

    if (err == CL_SUCCESS) {
        assertImageStep(step, &doneSteps);
    }
    releaseImageAssertion(&doneSteps);

    return err;
}

static void
splitSolutionStep(
    SolutionStep *rem,
    SolutionStep *cut,
    SDimComponent component,
    size_t chunk,
    bool backward)
{
    SubproblemDim remDim, cutDim;
    SubproblemDim remDimOff, cutDimOff;

    kargsToProbDims(&remDimOff, rem->funcID, &rem->args, true);
    kargsToProbDims(&remDim, rem->funcID, &rem->args, false);
    memcpy(&cutDim, &remDim, sizeof(SubproblemDim));
    memcpy(&cutDimOff, &remDimOff, sizeof(SubproblemDim));

    memcpy(cut, rem, sizeof(SolutionStep));
    if (component == SDIM_Y) {
        if (backward) {
            cutDimOff.y += remDim.y - chunk;
        }
        else {
            remDimOff.y += chunk;
        }
        cutDim.y = chunk;
        remDim.y -= chunk;
    }
    else {
        if (backward) {
            cutDimOff.x += remDim.x - chunk;
        }
        else {
            remDimOff.x += chunk;
        }
        cutDim.x = chunk;
        remDim.x -= chunk;
    }

    probDimsToKargs(&rem->args, rem->funcID, &remDimOff, true);
    probDimsToKargs(&rem->args, rem->funcID, &remDim, false);
    probDimsToKargs(&cut->args, cut->funcID, &cutDimOff, true);
    probDimsToKargs(&cut->args, cut->funcID, &cutDim, false);
}
