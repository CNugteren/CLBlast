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

#include <kerngen.h>
#include <list.h>
#include <dis_warning.h>

typedef struct FuncNode {
    void *pattern;
    char funcName[FUNC_NAME_MAXLEN];
    ListNode node;
} FuncNode;

typedef struct FuncNodeKey {
    const void *pattern;
    size_t patSize;
} FuncNodeKey;

struct KgenGuard {
    struct KgenContext *ctx;
    int (*genCallback)(struct KgenContext*, const void*);
    size_t patSize;
    ListHead funcs;
};

static int
funcNodeCmp(const ListNode *n, const void *key)
{
    const FuncNode *fnode = container_of(n, node, FuncNode);
    const FuncNodeKey *fkey = (FuncNodeKey*)key;

    return memcmp(fnode->pattern, fkey->pattern, fkey->patSize);
}

static void
destroyFuncNode(ListNode *node)
{
    FuncNode *fnode = container_of(node, node, FuncNode);

    free(fnode->pattern);
    free(fnode);
}

struct KgenGuard
*createKgenGuard(
    struct KgenContext *ctx,
    int (*genCallback)(struct KgenContext *ctx, const void *pattern),
    size_t patSize)
{
    struct KgenGuard *guard;

    guard = malloc(sizeof(struct KgenGuard));
    if (guard != NULL) {
        guard->ctx = ctx;
        guard->genCallback = genCallback;
        guard->patSize = patSize;
        listInitHead(&guard->funcs);
    }

    return guard;
}

void
reinitKgenGuard(
    struct KgenGuard *guard,
    struct KgenContext *ctx,
    int (*genCallback)(struct KgenContext *ctx, const void *pattern),
    size_t patSize)
{
    listDoForEachSafe(&guard->funcs, destroyFuncNode);
    listInitHead(&guard->funcs);
    guard->ctx = ctx;
    guard->genCallback = genCallback;
    guard->patSize = patSize;
}

/*
 * Invokes generator to generate a function
 * matching to the 'pattern' pattern or just
 * returns its name if the function is already
 * generated
 */
int
findGenerateFunction(
    struct KgenGuard *guard,
    const void *pattern,
    char *name,
    size_t nameLen)
{
    ListNode *n;
    FuncNode *fnode = NULL;

    FuncNodeKey fkey = {pattern, guard->patSize};
    int ret = 0;

    n = listNodeSearch(&guard->funcs, &fkey, funcNodeCmp);
    if (n == NULL) {
        ret = guard->genCallback(guard->ctx, pattern);
        if (!ret) {
            fnode = malloc(sizeof(FuncNode));
            if (fnode == NULL) {
                ret = -ENOMEM;
            }
            else {
                fnode->pattern = malloc(guard->patSize);
                if (fnode->pattern == NULL) {
                    free(fnode);
                    ret = -ENOMEM;
                }
                else {
                    memcpy(fnode->pattern, pattern, guard->patSize);
                    kgenGetLastFuncName(fnode->funcName,
                                        sizeof(fnode->funcName),
                                        guard->ctx);
                    fnode->funcName[FUNC_NAME_MAXLEN - 1] = '\0';
                    listAddToTail(&guard->funcs, &fnode->node);
                }
            }
        }
        else {
            ret = -EOVERFLOW;
        }
    }
    else {
        fnode = container_of(n, node, FuncNode);
    }

    if (!ret) {
        strncpy(name, fnode->funcName, nameLen);
        name[nameLen - 1] = '\0';
    }

    return ret;
}

void
destroyKgenGuard(struct KgenGuard *guard)
{
    listDoForEachSafe(&guard->funcs, destroyFuncNode);
    free(guard);
}

