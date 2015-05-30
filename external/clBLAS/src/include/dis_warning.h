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


#ifndef DIS_WARNING_H_
#define DIS_WARNING_H_

#if _MSC_VER

#pragma warning (disable:4204)
#pragma warning (disable:4127)
#define MAY_ALIAS

#else                       /* _MSC_VER */

#define MAY_ALIAS __attribute__((__may_alias__))

#endif


/*
 * Set of macro to mute gcc when we don't need in using some
 * function arguments
 */

#define DUMMY_ARG_USAGE(arg)                            \
do {                                                    \
    (void)arg;                                          \
} while (0)

#define DUMMY_ARGS_USAGE_2(arg1, arg2)                  \
do {                                                    \
    (void)arg1;                                         \
    (void)arg2;                                         \
} while (0)

#define DUMMY_ARGS_USAGE_3(arg1, arg2, arg3)            \
do {                                                    \
    (void)arg1;                                         \
    (void)arg2;                                         \
    (void)arg3;                                         \
} while(0)                                              \

#define DUMMY_ARGS_USAGE_4(arg1, arg2, arg3, arg4)      \
do {                                                    \
    (void)arg1;                                         \
    (void)arg2;                                         \
    (void)arg3;                                         \
    (void)arg4;                                         \
} while(0)                                              \

#endif /* DIS_WARNING_H_ */
