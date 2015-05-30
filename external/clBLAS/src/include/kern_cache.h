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
 * OpenCL kernel cache
 */

#ifndef KERN_CACHE_H_
#define KERN_CACHE_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <defbool.h>
#include <list.h>
#include <kerngen.h>
#include <solver.h>
#include <mutex.h>
#include <trace_malloc.h>

struct KernelCache;

/* Unique kernel characteristics */
typedef struct KernelKey {
    cl_device_id device;
    cl_context context;
    unsigned int nrDims;
    SubproblemDim subdims[MAX_SUBDIMS];
} KernelKey;

/*
 * structure describing an optimal CL kernel for some
 * memory pattern and subproblem dimensions
 */
typedef struct Kernel {
    cl_program program;     // program the kernel belongs to
    /* extra information specific for the application field */
    void *extra;
    size_t extraSize;
    void (*dtor)(struct Kernel *kern);
    int noSource;
} Kernel;

typedef int
(*KernelExtraCmpFn)(const void *extra, const void *extraKey);


/*
 * Create kernel cache
 *
 * @nrSolvers:  total solvers amount to store kernels of in a cache
 * @sizeLimit:  limit of the cache in bytes;
 *              if set to 0 the cache size is
 *              unlimited
 *
 * On success returns pointer to kernel cache object;
 * On error returns NULL, if it has not succeeded to allocated need resources
 */
struct KernelCache
*createKernelCache(
    unsigned int nrSolvers,
    size_t sizeLimit);

void
destroyKernelCache(struct KernelCache *kcache);

/*
 * Allocate kernel
 *
 * After allocation fill the structure with zero bytes
 * and set the kernel's reference counter to 1.
 *
 * return pointer to a just created kernel,
 * return NULL if there is not enough memory
 * to allocate a kernel
 */
Kernel
*allocKernel(void);

/*
 * Get reference to kernel not yet added to a cache
 */
void
getKernel(Kernel *kern);

/*
 * Decrement reference counter of this kernel
 *
 * @kcache: the cache the kernel inserted to;
 *          may be NULL if the kernel is not yet
 *          added to a cache, it is ignored in the case
 *
 * When there are no more references to the kernel, it is automatically
 * destroyed
 */
void
putKernel(struct KernelCache *kcache, Kernel *kern);

/*
 * Add new generated kernel to cache
 *
 * @kcache: cache to add the kernel to
 * @sid: solver ID to add the kernel for
 * @kern: kernel to add
 * @key: kernel characteristics
 *
 * On success returns 0.
 * On error returns -1, in on of the following cases:
 *      kernel size is larger than the maximum cache size,
 *      or there is not enough memory to allocate internal
 *      structures,
 *      or the passed solver ID is wrong,
 *      or 'nrDims' is wrong,
 */
int
addKernelToCache(
    struct KernelCache *kcache,
    solver_id_t sid,
    Kernel *kern,
    const KernelKey *key,
    KernelExtraCmpFn extraCmp);

/*
 * Find the kernel for the given OpenCL solver and
 * subproblem dimensions, and increment reference counter to it
 *
 * On success returns the kernel being actually stored in the cache.
 * On error returns NULL; it means the passed solver ID
 * is wrong, or any kernel for the given solver and subprolem
 * dimensions is not stored in the cache
 */
Kernel
*findKernel(
    struct KernelCache *kcache,
    solver_id_t sid,
    const KernelKey *key,
    const void *extraKey);

/*
 * Get available size in the kernel cache
 */
size_t
availKernelCacheSize(struct KernelCache *kcache);

/*
 * Remove all kernels from the cache
 */
void
cleanKernelCache(struct KernelCache *kcache);

size_t
fullKernelSize(struct Kernel *kern);


#if defined(TRACE_MALLOC)

void
printKernelCacheSize(struct KernelCache *kcache);

#else       /* TRACE_MALLOC */

static __inline void
printKernelCacheSize(struct KernelCache *kcache)
{
    /* do nothing */
    (void)kcache;
}

#endif      /* !TRACE_MALLOC */

#endif /* KERN_CACHE_H_ */
