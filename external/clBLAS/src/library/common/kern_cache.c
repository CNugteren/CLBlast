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
 * Kernel cache implementation
 */

/*
 *  TODO: more efficient data structure to search
 *        by dimensions (red black tree, for example) (?)
 */


#include<stdlib.h>
#include <string.h>
#include <assert.h>

#include <kern_cache.h>
#include <kerngen.h>
#include <mempat.h>

#define KCACHE_LOCK(kache)      mutexLock((kcache)->mutex)
#define KCACHE_UNLOCK(kcache)   mutexUnlock((kcache)->mutex)
#define UNLIMITED_CACHE_SIZE    (~0UL)

enum {
    KNODE_MAGIC = 0x3CED50C5,
    TRUNC_AHEAD_FACTOR = 4,
    MAX_OPENCL_DEVICES = 64
};

// prime is chosen such overflowing on multiply on is very likely
const unsigned long long prime = 100000000000000889LL;

typedef struct KernelNode {
    unsigned long magic;
    unsigned long refcnt;
    Kernel kern;
    unsigned long hash;
    // key data the kernel is based on
    KernelKey key;
    // function comparing kernel extra information
    KernelExtraCmpFn extraCmp;
    // node to store in a memory pattern related list
    ListNode dimNode;
    ListNode lruNode;
} KernelNode;

typedef struct KcacheKey {
    unsigned long hash;
    KernelKey key;
    const void *extra;
} KcacheKey;

struct KernelCache {
    size_t totalSize;
    size_t sizeLimit;
    // total amount of solvers
    unsigned int nrSolvers;
    // lists to search by subproblem dimensions
    ListHead *dimKern;
    // least recently used kernels list
    ListHead lruKern;
    mutex_t *mutex;
};

// update kernel hash using the dimension size
static __inline unsigned long
updateHash(unsigned long hash, unsigned long size)
{
    if (size != SUBDIM_UNUSED) {
        hash = (hash << 5) | size;
    }

    return hash;
}

// hash kernel subproblem dimensions
static unsigned long
kernHash(const SubproblemDim *subdims, unsigned int nrDims)
{
    unsigned int i;
    unsigned long hash = 0;

    for (i = 0; i < nrDims; i++) {
        hash = updateHash(hash, (unsigned long)subdims[i].x);
        hash = updateHash(hash, (unsigned long)subdims[i].y);
        hash = updateHash(hash, (unsigned long)subdims[i].bwidth);
        hash = updateHash(hash, (unsigned long)subdims[i].itemX);
        hash = updateHash(hash, (unsigned long)subdims[i].itemY);
    }

    return (unsigned long)(hash * prime);
}

// comparison function to look for a kernel node in the cache
static int
knodeCmp(const ListNode *node, const void *key)
{
    KcacheKey *kkey = (KcacheKey*)key;
    KernelNode *knode = container_of(node, dimNode, KernelNode);

    KernelKey *a = &(kkey->key);
    KernelKey *b = &(knode->key);

    if ((a->device != b->device) || (a->context != b->context) ||
                (a->nrDims != b->nrDims)) {
        return 1;
    }
    if (memcmp(a->subdims, b->subdims, a->nrDims * sizeof(SubproblemDim)) != 0) {
        return 1;
    }

    if (knode->extraCmp != NULL) {
        return knode->extraCmp(knode->kern.extra, kkey->extra);
    }

    return 0;
}

static void
removeKernels(ListHead *truncList, struct KernelCache *kcache, size_t truncSize)
{
    size_t remSize = 0;
    size_t ksize;
    ListNode *l;
    KernelNode *knode;

    listInitHead(truncList);

    while (remSize < truncSize) {
        l = listNodeLast(&kcache->lruKern);
        if (l == &kcache->lruKern) {
            break;
        }

        knode = container_of(l, lruNode, KernelNode);
        listDel(l);
        listDel(&knode->dimNode);
        listAddToTail(truncList, &knode->lruNode);
        ksize = fullKernelSize(&knode->kern);
        remSize += ksize;
        kcache->totalSize -= ksize;
    }
}

static void
putRemovedKernels(struct KernelCache *kcache, ListHead *truncList)
{
    struct ListNode *l;
    struct KernelNode *knode;

    while (1) {
        l = listNodeFirst(truncList);
        if (l == truncList) {
            break;
        }

        knode = container_of(l, lruNode, KernelNode);
        listDel(l);
        putKernel(kcache, &knode->kern);
    }
}

Kernel
*allocKernel(void)
{
    KernelNode *knode;

    knode = malloc(sizeof(KernelNode));
    if (knode == NULL) {
        return NULL;
    }

    memset(knode, 0, sizeof(KernelNode));
    knode->refcnt = 1;
    knode->magic = KNODE_MAGIC;

    return &knode->kern;
}

void
getKernel(Kernel *kern)
{
    KernelNode *knode;

    knode = container_of(kern, kern, KernelNode);
    assert(knode->magic == KNODE_MAGIC);
    knode->refcnt++;
}

void
putKernel(struct KernelCache *kcache, Kernel *kern)
{
    KernelNode *knode;
    unsigned long refcnt;

    if (kern == NULL) {
        return;
    }

    knode = container_of(kern, kern, KernelNode);
    assert(knode->magic == KNODE_MAGIC);

    if (kcache) {
        KCACHE_LOCK(kcache);
    }
    refcnt = --knode->refcnt;
    if (kcache) {
        KCACHE_UNLOCK(kcache);
    }

    if (!refcnt) {
        if (kern->dtor) {
            kern->dtor(kern);
        }
        clReleaseProgram(kern->program);
        clReleaseContext(knode->key.context);
        free(knode);
    }
}

struct KernelCache
*createKernelCache(
    unsigned int nrSolvers,
    size_t sizeLimit)
{
    int err = 0;
    unsigned int i;
    struct KernelCache *kcache;

    kcache = malloc(sizeof(struct KernelCache));
    if (kcache == NULL) {
        return NULL;
    }

    memset(kcache, 0, sizeof(struct KernelCache));

    kcache->nrSolvers = nrSolvers;
    kcache->dimKern = malloc(kcache->nrSolvers * sizeof(ListHead));
    if (kcache->dimKern == NULL) {
        err = -1;
    }
    else {
        for (i = 0; i < kcache->nrSolvers; i++) {
            listInitHead(&kcache->dimKern[i]);
        }
        listInitHead(&kcache->lruKern);

        kcache->sizeLimit = sizeLimit;
        kcache->totalSize = 0;

        kcache->mutex = mutexInit();
        err = (kcache->mutex == NULL);
    }

    if (err) {
        if (kcache->dimKern) {
            free(kcache->dimKern);
        }
        free(kcache);
        kcache = NULL;
    }

    return kcache;
}

void
destroyKernelCache(struct KernelCache *kcache)
{
    cleanKernelCache(kcache);
    free(kcache->dimKern);
    mutexDestroy(kcache->mutex);
    free(kcache);
}

int
addKernelToCache(
    struct KernelCache *kcache,
    solver_id_t sid,
    Kernel *kern,
    const KernelKey *key,
    KernelExtraCmpFn extraCmp)
{
    size_t ksize;
    KernelNode *knode;
    ListHead truncList;

    knode = container_of(kern, kern, KernelNode);
    assert(knode->magic == KNODE_MAGIC);

    if ((unsigned)sid >= kcache->nrSolvers || key->nrDims > MAX_SUBDIMS) {
        return -1;
    }

    listInitHead(&truncList);
    ksize = fullKernelSize(kern);

    KCACHE_LOCK(kcache);

    if (kcache->sizeLimit) {
        if (ksize > kcache->sizeLimit) {
            KCACHE_UNLOCK(kcache);
            return -1;
        }
        else if (ksize > kcache->sizeLimit - kcache->totalSize) {
            removeKernels(&truncList, kcache, ksize * TRUNC_AHEAD_FACTOR);
        }
    }

    knode->hash = kernHash(key->subdims, key->nrDims);
    knode->extraCmp = extraCmp;

    knode->key.device = key->device;
    knode->key.context = key->context;
    clRetainContext(knode->key.context);
    knode->key.nrDims = key->nrDims;
    memset(knode->key.subdims, 0, sizeof(knode->key.subdims));
    memcpy(knode->key.subdims, key->subdims, sizeof(SubproblemDim) *
           knode->key.nrDims);

    listAddToTail(&kcache->dimKern[sid], &knode->dimNode);
    listAddToHead(&kcache->lruKern, &knode->lruNode);
    kcache->totalSize += ksize;

    KCACHE_UNLOCK(kcache);

    if (!isListEmpty(&truncList)) {
        putRemovedKernels(kcache, &truncList);
    }

    return 0;
}

Kernel
*findKernel(
    struct KernelCache *kcache,
    solver_id_t sid,
    const KernelKey *key,
    const void *extraKey)
{
    Kernel *kern = NULL;
    KcacheKey kkey;
    KernelNode *knode;
    ListNode *lnode;

    if ((unsigned)sid >= kcache->nrSolvers || key->nrDims > MAX_SUBDIMS) {
        return NULL;
    }

    kkey.hash = kernHash(key->subdims, key->nrDims);
    kkey.extra = extraKey;

    kkey.key.device = key->device;
    kkey.key.context = key->context;
    kkey.key.nrDims = key->nrDims;
    memset(kkey.key.subdims, 0, sizeof(kkey.key.subdims));
    memcpy(kkey.key.subdims, key->subdims, sizeof(SubproblemDim) * kkey.key.nrDims);

    KCACHE_LOCK(kcache);
    lnode = listNodeSearch(&kcache->dimKern[sid], &kkey, knodeCmp);
    if (lnode) {
        knode = container_of(lnode, dimNode, KernelNode);
        knode->refcnt++;
        kern = &knode->kern;

        // move the kernel to the top of the LRU list
        listDel(&knode->lruNode);
        listAddToHead(&kcache->lruKern, &knode->lruNode);
    }
    KCACHE_UNLOCK(kcache);

    return kern;
}

size_t
availKernelCacheSize(struct KernelCache *kcache)
{
    size_t size;

    KCACHE_LOCK(kcache);
    size = (kcache->sizeLimit) ? (kcache->sizeLimit - kcache->totalSize) :
           ~(size_t)0;
    KCACHE_UNLOCK(kcache);

    return size;
}

void
cleanKernelCache(struct KernelCache *kcache)
{
    ListHead truncList;

    KCACHE_LOCK(kcache);
    removeKernels(&truncList, kcache, kcache->totalSize);
    KCACHE_UNLOCK(kcache);

    putRemovedKernels(kcache, &truncList);
}

size_t
fullKernelSize(Kernel *kern)
{
    size_t allSizes[MAX_OPENCL_DEVICES], size = 0;
    size_t i, retSize;

    clGetProgramInfo(kern->program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(allSizes), &allSizes, &retSize);
    retSize /= sizeof(size);
    for (i = 0; i < retSize; i++) {
        size += allSizes[i];
    }

    if (!kern->noSource) {
        clGetProgramInfo(kern->program, CL_PROGRAM_SOURCE, 0, NULL, &retSize);
    }

    return (size + retSize + sizeof(Kernel) + kern->extraSize);
}

#if defined(TRACE_MALLOC)

#include <stdio.h>

void
printKernelCacheSize(struct KernelCache *kcache)
{
    printf("[KERNEL CACHE] My size is %lu MiB\n", kcache->totalSize / 1048576);
}

#endif
