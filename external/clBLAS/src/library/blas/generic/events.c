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
#include <clBLAS.h>

#include <mutex.h>
#include <events.h>

static const size_t ALLOCATION_STEP = 100;

static mutex_t *lock = NULL;
static cl_event *decomposeEvents = NULL;
static size_t numDecomposeEvents = 0;
static size_t maxDecomposeEvents = 0;

void
decomposeEventsSetup(void)
{
    lock = mutexInit();
}

void
decomposeEventsTeardown(void)
{
    mutexLock(lock);

    if (decomposeEvents != NULL) {
        free(decomposeEvents);
    }

    decomposeEvents = NULL;
    numDecomposeEvents = 0;
    maxDecomposeEvents = 0;

    mutexDestroy(lock);
    lock = NULL;
}

cl_event*
decomposeEventsAlloc(void)
{
    cl_event* e;

    mutexLock(lock);

    if (numDecomposeEvents == maxDecomposeEvents) {
        e = realloc(decomposeEvents,
            (maxDecomposeEvents + ALLOCATION_STEP) * sizeof(cl_event));
        if (e == NULL) {
            mutexUnlock(lock);
            return NULL;
        }
        decomposeEvents = e;
        maxDecomposeEvents += ALLOCATION_STEP;
    }
    e = &(decomposeEvents[numDecomposeEvents++]);

    mutexUnlock(lock);
    return e;
}
