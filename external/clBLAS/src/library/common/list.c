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


#include <stddef.h>
#include <list.h>
#include <assert.h>

static __inline
void listAddAfter(ListNode *prev, ListNode *node)
{
    ListNode *next = prev->next;

    prev->next = node;
    node->prev = prev;
    node->next = next;
    next->prev = node;
}

void
listAddToTail(ListHead *head, ListNode *node)
{
    listAddAfter(head->prev, node);
}

void
listAddToHead(ListHead *head, ListNode *node)
{
    listAddAfter(head, node);
}

void
listDel(ListNode *node)
{
#ifdef DEBUG
    // check if it's not really the list head
    assert(node->next != node->prev);
#endif

    node->prev->next = node->next;
    node->next->prev = node->prev;
}

ListNode
*listDelFromTail(ListHead *head)
{
    ListNode *node = head->prev;

    listDel(node);

    return node;
}

void
listDoForEach(ListHead *head, ListAction act)
{
    ListNode *node;

    for (node = listNodeFirst(head); node != head; node = node->next) {
        act(node);
    }
}

void
listDoForEachSafe(ListHead *head, ListAction act)
{
    ListNode *node, *save;

    for (node = listNodeFirst(head), save = node->next; node != head;
         node = save, save = node->next) {

        act(node);
    }
}

void
listDoForEachPriv(const ListHead *head, ListPrivAction act, void *actPriv)
{
    ListNode *node;

    for (node = listNodeFirst(head); node != head; node = node->next) {
        act(node, actPriv);
    }
}

void
listDoForEachPrivSafe(const ListHead *head, ListPrivAction act, void *actPriv)
{
    ListNode *node, *save;

    for (node = listNodeFirst(head), save = node->next; node != head;
         node = save, save = node->next) {

        act(node, actPriv);
    }
}

ListNode
*listNodeSearch(const ListHead *head, const void *key, ListCmpFn cmp)
{
    ListNode *node;

    for (node = listNodeFirst(head); node != head; node = node->next) {
        if (!cmp(node, key)) {
            break;
        }
    }

    return (node == head) ? NULL : node;
}

size_t
listLength(const ListHead *head)
{
    size_t length = 0;
    ListNode *node;

    for (node= listNodeFirst(head); node != head; node = node->next) {
        length++;
    }

    return length;
}
