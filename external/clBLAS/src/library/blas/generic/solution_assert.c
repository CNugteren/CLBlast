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
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "solution_assert.h"

#define ASSERT_GREQ_AND_DIV(a, b) assert(((a) >= (b)) && ((a) % (b) == 0))

// solution area
typedef struct SolArea {
    size_t offsetM;
    size_t M;
    size_t offsetN;
    size_t N;
    ListNode node;
} SolArea;

#ifdef ASSERT_GRANULATION

// check the found dimensions are not wrong
void VISIBILITY_HIDDEN
assertGranulation(
    SubproblemDim *dims,
    unsigned int nrDims,
    PGranularity *pgran,
    unsigned int thLevel)
{
    unsigned int i;
    size_t gsize;

    /*
     * subproblem dimensions on all levels must meet the following requirements:
     *
     * 1) Item work piece is greater then a processing step
     * 2) Item work piece is integrally divisible on the processing step
     * 3) Work pieces and processing steps don't grows at forwarding to the bottom level
     * 4) At passing to the thread level, the subproblem must be strict divisible among
     *    all the threads
     */

    gsize = pgran->wgSize[0] * pgran->wgSize[1];

    for (i = 0; i < nrDims; i++) {
        if (i || dims[i].itemX != SUBDIM_UNUSED) {
            ASSERT_GREQ_AND_DIV(dims[i].itemX, dims[i].x);
        }
        if (i || dims[i].itemY != SUBDIM_UNUSED) {
            ASSERT_GREQ_AND_DIV(dims[i].itemY, dims[i].y);
        }
        if (i) {
            ASSERT_GREQ_AND_DIV(dims[i - 1].x, dims[i].itemX);
            ASSERT_GREQ_AND_DIV(dims[i - 1].y, dims[i].itemY);
            ASSERT_GREQ_AND_DIV(dims[i - 1].bwidth, dims[i].bwidth);
        }
    }

    assert((dims[thLevel].itemX * dims[thLevel].itemY) * gsize ==
           dims[thLevel - 1].x * dims[thLevel - 1].y);
}

#endif  // ASSERT_GRANULATION

#ifdef ASSERT_IMAGE_STEPS

static __inline void
assertEnclosed(size_t off1, size_t size1, size_t off2, size_t size2)
{
    bool enc = ((off1 >= off2) && (off1 < off2 + size2) &&
                (off1 + size1 > off2) && (off1 + size1 <= off2 + size2));
    assert(enc);
}

static __inline bool
isIntersected(size_t off1, size_t size1, size_t off2, size_t size2)
{
    return ((off1 >= off2 && off1 < off2 + size2) ||
            (off1 + size1 > off2 && off1 + size1 <= off2 + size2));
}

static void
freeSolAreaNode(ListNode *node)
{
    SolArea *area = container_of(node, node, SolArea);

    free(area);
}

static void
accProcessed(ListNode *node, void *priv)
{
    SolArea *a1 = container_of(node, node, SolArea);
    SolArea *a2 = (SolArea*)priv;

    if (!isIntersected(a1->offsetM, a1->M, a2->offsetM, a2->M)) {
        a2->M += a1->M;
        if (a2->offsetM > a1->offsetM) {
            a2->offsetM = a1->offsetM;
        }
    }
    if (!isIntersected(a1->offsetN, a1->N, a2->offsetN, a2->N)) {
        a2->N += a1->N;
        if (a2->offsetN > a1->offsetN) {
            a2->offsetN = a1->offsetN;
        }
    }
}

static int
solAreaCmp(ListNode *a, const void *b)
{
    SolArea *area = container_of(a, node, SolArea);
    const CLBlasKargs *kargs = (const CLBlasKargs*)b;
    int ret;

    ret = isIntersected(kargs->offsetM, kargs->M,
                        area->offsetM, area->M);
    ret = ret && isIntersected(kargs->offsetN, kargs->N,
                               area->offsetN, area->N);

    return !ret;
}

void VISIBILITY_HIDDEN
assertImageSubstep(
    SolutionStep *wholeStep,
    SolutionStep *substep,
    ListHead *doneSubsteps)
{
    CLBlasKargs *kargs1 = &substep->args;
    CLBlasKargs *kargs2 = &wholeStep->args;
    ListNode *node;
    SolArea *area;

    assertEnclosed(kargs1->offsetM, kargs1->M, kargs2->offsetM, kargs2->M);
    assertEnclosed(kargs1->offsetN, kargs1->N, kargs2->offsetN, kargs2->N);
    node = listNodeSearch(doneSubsteps, (const void*)&substep->args,
                          solAreaCmp);
    assert(!node);
    area = malloc(sizeof(SolArea));
    if (area == NULL) {
        fprintf(stderr, "[%s, line %d]: Failed to allocate memory for image "
                        "step assertion!\n", __FILE__, __LINE__);
    }
    else {
        area->offsetM = substep->args.offsetM;
        area->M = substep->args.M;
        area->offsetN = substep->args.offsetN;
        area->N = substep->args.N;
        listAddToTail(doneSubsteps, &area->node);
    }
}

void VISIBILITY_HIDDEN
assertImageStep(SolutionStep *wholeStep, ListHead *doneSubsteps)
{
    SolArea area;

    area.offsetM = SIZE_MAX;
    area.M = 0;
    area.offsetN = SIZE_MAX;
    area.N = 0;
    listDoForEachPriv(doneSubsteps, accProcessed, &area);
    assert((area.offsetM == wholeStep->args.offsetM) &&
           (area.M == wholeStep->args.M) &&
           (area.offsetM ==wholeStep->args.offsetM) &&
           (area.N == wholeStep->args.N));
}

void VISIBILITY_HIDDEN
releaseImageAssertion(ListHead *doneSubsteps)
{
    listDoForEachSafe(doneSubsteps, freeSolAreaNode);
    listInitHead(doneSubsteps);
}

#endif   /* ASSERT_IMAGE_STEPS */

