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


#ifndef TIMER_H_
#define TIMER_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)

typedef unsigned long long nano_time_t;
#define NANOTIME_MAX (~0ULL - 1)

#elif defined(__APPLE__)
#include <stdint.h>

typedef uint64_t nano_time_t;
#define NANOTIME_MAX (UINT64_MAX - 1)

#else

typedef unsigned long nano_time_t;
#define NANOTIME_MAX (~0UL - 1)

#endif

#define NANOTIME_ERR (NANOTIME_MAX + 1)

nano_time_t
conv2millisec(nano_time_t t);

nano_time_t
conv2microsec(nano_time_t t);

nano_time_t
conv2nanosec(nano_time_t t);

nano_time_t
getCurrentTime(void);

void
sleepTime(nano_time_t t);

#ifdef __cplusplus
}   /* extern "C" { */
#endif

#endif  /* TIMER_H_ */
