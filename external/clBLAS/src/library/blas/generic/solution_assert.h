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


#ifndef SOLUTION_ASSERT_H_
#define SOLUTION_ASSERT_H_

#include "solution_seq.h"

#ifdef ASSERT_GRANULATION

void
assertGranulation(
    SubproblemDim *dims,
    unsigned int nrDims,
    PGranularity *pgran,
    unsigned int thLevel);

#else   // ASSERT_GRANULATION

// stub, do nothing
#define assertGranulation(dims, nrDims, pgran, thLevel)

#endif  // !ASSERT_GRANULATION

#ifdef ASSERT_IMAGE_STEPS

void
assertImageSubstep(
    SolutionStep *wholeStep,
    SolutionStep *substep,
    ListHead *doneSubsteps);

void
assertImageStep(SolutionStep *wholeStep, ListHead *doneSubsteps);

void
releaseImageAssertion(ListHead *doneSubsteps);

#else   /* ASSERT_IMAGE_STEPS */

// stubs

#define assertImageSubstep(wholeStep, substep, doneSubsteps)
#define assertImageStep(wholeStep, doneSubsteps)
#define releaseImageAssertion(doneSubsteps)

#endif  /* !ASSERT_IMAGE_STEPS */

#endif /* SOLUTION_ASSERT_H_ */
