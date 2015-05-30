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
 * Define simple functionality to track memory leaks in order to separate
 * library leaks from leaks in the other components and to take info in
 * a human friendly format
 */

#ifndef TRACE_MALLOC_H_
#define TRACE_MALLOC_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(TRACE_MALLOC)

#define malloc(size)            debugMalloc(size, __FILE__, __LINE__)
#define calloc(nmemb, size)     debugCalloc(size * nmemb, __FILE__, __LINE__)
#define realloc(ptr, size)      debugRealloc(ptr, size, __FILE__, __LINE__)
#define free(ptr)               debugFree(ptr)

void initMallocTrace(void);
void *debugMalloc(size_t size, const char *file, int line);
void *debugCalloc(size_t size, const char *file, int line);
void *debugRealloc(void *ptr, size_t size, const char *file, int line);
void debugFree(void *ptr);
void printMallocStatistics(void);
void printMemLeaksInfo(void);
void releaseMallocTrace(void);

#else       /* TRACE_MALLOC */

static __inline void initMallocTrace(void)
{
    /* do nothing */
}

static __inline void printMallocStatistics(void)
{
    /* do nothing */
}

static __inline void printMemLeaksInfo(void)
{
    /* do nothing */
}

static __inline void releaseMallocTrace(void)
{
    /* do nothing */
}

#endif      /* !TRACE_MALLOC */

#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* TRACE_MALLOC_H_ */
