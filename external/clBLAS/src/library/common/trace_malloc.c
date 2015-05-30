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
#include <list.h>
#include <assert.h>

#include <trace_malloc.h>
#include <mutex.h>

#if defined(TRACE_MALLOC)

#if _MSC_VER
#include <msvc.h>
#endif

// use standard malloc/free though
#undef malloc
#undef calloc
#undef realloc
#undef free

enum {
    MTRACE_NODE_MAGIC = 0x5A20286D
};

#define MTRACE_LOCK()     mutexLock(mutex)
#define MTRACE_UNLOCK()   mutexUnlock(mutex)
#define KIB 1024
#define MIB KIB*1024

typedef struct MtraceNode {
    unsigned long magic;
    char *file;
    int line;
    void *ptr;
    size_t size;
    ListNode node;
} MtraceNode;

static mutex_t *mutex;
static size_t tracedSize;
static size_t rawSize;
ListHead traceList;

static
int cmpNode(const ListNode *node, const void *key)
{
    const MtraceNode *mtnode = container_of(node, node, MtraceNode);

    return !(mtnode->ptr == key);
}

static __inline size_t
rawTracedSize(MtraceNode *mtnode)
{
    return mtnode->size + sizeof(MtraceNode) + strlen(mtnode->file) + 1;
}

static MtraceNode
*searchMtraceNode(void *ptr)
{
    ListNode *node;

    MTRACE_LOCK();
    node = listNodeSearch(&traceList, ptr, cmpNode);
    MTRACE_UNLOCK();

    return (node) ? container_of(node, node, MtraceNode) : NULL;
}

static void
freeNode(ListNode *node)
{
    MtraceNode *mtnode = container_of(node, node, MtraceNode);

    if (mtnode->file != NULL) {
        free(mtnode->file);
    }
    if (mtnode->ptr != NULL) {
        free(mtnode->ptr);
    }
    free(mtnode);
}

static void
sprintfTracedSize(char *str, size_t size)
{
    const char *suffix;

    if (size < KIB * 10) {
        suffix = "bytes";
    }
    else if (size < MIB * 10) {
        suffix = "KiB";
        size /= KIB;
    }
    else {
        suffix = "MIB";
        size /= MIB;
    }

    sprintf(str, "%lu %s", size, suffix);
}

static void
printNodeInfo(ListNode *node)
{
    MtraceNode *mtnode = container_of(node, node, MtraceNode);
    char s[1024];

    sprintfTracedSize(s, mtnode->size);
    printf("%s at %s line %d\n", s, mtnode->file, mtnode->line);
}

void
initMallocTrace(void)
{
    listInitHead(&traceList);
    tracedSize = rawSize = 0;
    mutex = mutexInit();
}

void
*debugMalloc(size_t size, const char *file, int line)
{
    void *ret = NULL;
    MtraceNode *mtnode;

    mtnode = calloc(1, sizeof(MtraceNode));
    if (mtnode == NULL) {
        return NULL;
    }

    mtnode->magic = MTRACE_NODE_MAGIC;
    mtnode->file = strdup(file);
    if (mtnode->file != NULL) {
        ret = mtnode->ptr = malloc(size);
    }

    if (ret != NULL) {
        mtnode->line = line;
        mtnode->size = size;

        MTRACE_LOCK();
        tracedSize += size;
        rawSize += rawTracedSize(mtnode);
        listAddToTail(&traceList, &mtnode->node);
        MTRACE_UNLOCK();
    }
    else {
        freeNode(&mtnode->node);
    }

    return ret;
}

void
*debugCalloc(size_t size, const char *file, int line)
{
    void *ret;

    ret = debugMalloc(size, file, line);
    if (ret != NULL) {
        memset(ret, 0, size);
    }

    return ret;
}

void
*debugRealloc(void *ptr, size_t size, const char *file, int line)
{
    void *ret;

    if (ptr == NULL) {
        ret = debugMalloc(size, file, line);
    }
    else {
        MtraceNode *mtnode;

        mtnode = searchMtraceNode(ptr);
        assert((mtnode != NULL) && (mtnode->magic == MTRACE_NODE_MAGIC));
        ret = realloc(ptr, size);
        if (ret != NULL) {
            ssize_t delta = (ssize_t)size - (ssize_t)mtnode->size;

            mtnode->ptr = ret;
            mtnode->size = size;
            MTRACE_LOCK();
            tracedSize += delta;
            rawSize += delta;
            MTRACE_UNLOCK();
        }
        else {
            debugFree(ptr);
        }
    }

    return ret;
}

void
debugFree(void *ptr)
{
    MtraceNode *mtnode;

    if (ptr == NULL) {
        return;
    }

    mtnode = searchMtraceNode(ptr);
    assert((mtnode != NULL) && (mtnode->magic == MTRACE_NODE_MAGIC));

    MTRACE_LOCK();
    tracedSize -= mtnode->size;
    rawSize -= rawTracedSize(mtnode);
    listDel(&mtnode->node);
    MTRACE_UNLOCK();

    freeNode(&mtnode->node);
}

void
printMallocStatistics(void)
{
    char s[1024];

    sprintfTracedSize(s, tracedSize);
    printf("[MALLOC TRACE] Totally %s is allocated\n", s);
}

void
printMemLeaksInfo(void)
{
    puts("\n");
    if (!tracedSize) {
        puts("[MALLOC TRACE] Hurray! There are not memory leaks!");
    }
    else {
        char s1[1024], s2[1024];

        sprintfTracedSize(s1, tracedSize);
        sprintfTracedSize(s2, rawSize);
        printf("[MALLOC TRACE] Totally %s is lost, raw traced size is %s\n",
               s1, s2);
        puts("Detailed report:\n"
             "------------------------------------------------------------");
        assert(!isListEmpty(&traceList));
        listDoForEach(&traceList, printNodeInfo);
    }
}

void
releaseMallocTrace(void)
{
    listDoForEachSafe(&traceList, freeNode);
    mutexDestroy(mutex);
}

#endif       /* TRACE_MALLOC */

