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
 * Work with circular double linked lists
 */

#ifndef LIST_H_
#define LIST_H_

#include <defbool.h>

#if defined (_WIN64)
typedef unsigned long long prt_size_t;
#else
typedef unsigned long prt_size_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define offset_of(field, type)                                 \
    (prt_size_t)(&((type*)0)->field)

#define container_of(node, field, type)                        \
    (type*)((prt_size_t)(node) - offset_of(field, type))

typedef struct ListNode {
    struct ListNode *prev;
    struct ListNode *next;
} ListNode;

typedef ListNode ListHead;
typedef void (*ListAction)(ListNode *node);
typedef void (*ListPrivAction)(ListNode *node, void *priv);

/*
 *  Type of function comparing list node contents with a key.
 *  On equality such a function must return 0
 */
typedef int (*ListCmpFn)(const ListNode *node, const void *key);

static __inline
bool isListEmpty(ListHead *list)
{
    return (list->next == list);
}

static __inline ListNode
*listNodeFirst(const ListHead *head)
{
    return head->next;
}

static __inline ListNode
*listNodeLast(const ListHead *head)
{
    return head->prev;
}

static __inline void
listInitHead(ListHead *head)
{
    head->prev = head;
    head->next = head;
}

void
listAddToTail(ListHead *head, ListNode *node);

void
listAddToHead(ListHead *head, ListNode *node);

void listDel(ListNode *node);

ListNode
*listDelFromTail(ListHead *head);

void
listDoForEach(ListHead *head, ListAction act);

void
listDoForEachSafe(ListHead *head, ListAction act);

void
listDoForEachPriv(const ListHead *head, ListPrivAction act, void *actPriv);

void
listDoForEachPrivSafe(const ListHead *head, ListPrivAction act, void *actPriv);

ListNode
*listNodeSearch(const ListHead *head, const void *key, ListCmpFn cmp);

size_t
listLength(const ListHead *head);

#ifdef __cplusplus
}
#endif

#endif /* LIST_H_ */
