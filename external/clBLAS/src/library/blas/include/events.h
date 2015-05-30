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
 * Events used during SolutionStep decomposition internally.
 */

#ifndef EVENTS_H_
#define EVENTS_H_

void decomposeEventsSetup(void);
void decomposeEventsTeardown(void);
cl_event* decomposeEventsAlloc(void);

#endif  /* EVENTS_H_ */
