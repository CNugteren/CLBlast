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


#include <mutex.h>

#if defined(_MSC_VER)

#pragma warning(push,3)
#include <windows.h>
#pragma warning(pop)

mutex_t*
mutexInit(void)
{
    HANDLE mutex;

    mutex = CreateMutex(NULL, FALSE, NULL);
    return (mutex_t*)mutex;
}

int
mutexDestroy(mutex_t *_mutex)
{
    HANDLE mutex = (HANDLE)_mutex;

    if (CloseHandle(mutex) == FALSE) {
        /* Bad mutex, etc. */
        return 1;
    }
    return 0;
}

int
mutexLock(mutex_t *_mutex)
{
    HANDLE mutex = (HANDLE)_mutex;
    DWORD rc;

    rc = WaitForSingleObjectEx(mutex, INFINITE, FALSE);
    if (rc != WAIT_OBJECT_0) {
        /* Bad mutex, etc. */
        return 1;
    }
    return 0;
}

int
mutexUnlock(mutex_t *_mutex)
{
    HANDLE mutex = (HANDLE)_mutex;

    if (ReleaseMutex(mutex) == FALSE) {
        /* Bad mutex, etc. */
        return 1;
    }
    return 0;
}

#else /* defined(_MSC_VER) */

#include <stdlib.h>
#include <pthread.h>

mutex_t*
mutexInit(void)
{
    pthread_mutex_t *mutex;

    mutex = calloc(1, sizeof(pthread_mutex_t));
    if (mutex == NULL)
        return NULL;
    if (pthread_mutex_init(mutex, NULL) != 0) {
        free(mutex);
        return NULL;
    }

    return (mutex_t*)mutex;
}

int
mutexDestroy(mutex_t *_mutex)
{
    pthread_mutex_t *mutex = (pthread_mutex_t*)_mutex;

    if (mutex == NULL) {
        /* Mutex is invalid */
        return 1;
    }
    if (pthread_mutex_destroy(mutex) != 0) {
        /* Mutex is busy or invalid */
        return 1;
    }

    free(mutex);
    return 0;
}

int
mutexLock(mutex_t *_mutex)
{
    pthread_mutex_t *mutex = (pthread_mutex_t*)_mutex;

    return (pthread_mutex_lock(mutex) == 0) ? 0 : 1;
}

int
mutexUnlock(mutex_t *_mutex)
{
    pthread_mutex_t *mutex = (pthread_mutex_t*)_mutex;

    return (pthread_mutex_unlock(mutex) == 0) ? 0 : 1;
}

#endif  /* defined (_MSC_VER) */
