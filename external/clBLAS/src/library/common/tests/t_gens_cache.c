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
 * test generator and cache infrastructure
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <kerngen.h>
#include <kern_cache.h>

enum {
    NR_TEST_PATTERNS = 5,
    KERNELS_PER_PATTERN = 10,
    KCACHE_SIZE_LIMIT = 1048576
};

const char *strcpyImpl =
    "char\n"
    "*strcpy(char *dst, char *src)\n"
    "{\n"
    "   do {\n"
    "       *dst++ = *src++;\n"
    "   } while (*(dst - 1) != 0);\n"
    "}";

static int
testGenFunc(struct KgenContext *ctx)
{
    kgenDeclareFunction(ctx, "char\n"
                             "*strcpy(char *dst, char *src)\n");
    kgenBeginFuncBody(ctx);
    kgenAddStmt(ctx, "char *ret = dst;\n\n");
    kgenBeginBranch(ctx, "do");
    kgenAddStmt(ctx, "*dst = *src;\n"
                     "src++;\n"
                     "dst++;\n");
    kgenEndBranch(ctx, "while (*(dst - 1) != 0)");
    kgenAddBlankLine(ctx);
    kgenAddStmt(ctx, "return ret;\n");

    return kgenEndFuncBody(ctx);
}

static int
kernExtraCmp(const void *extra, const void *extraKey)
{
    unsigned long u1 = *(unsigned long*)extra;
    unsigned long u2 = *(unsigned long*)extraKey;

    return !(u1 == u2);
}


static int
testGen(void)
{
    char buf[4096];
    char name[64];
    int r;
    struct KgenContext *ctx;
    size_t s;

    ctx = createKgenContext(buf, sizeof(buf), true);
    if (ctx == NULL) {
        printf("Context creation failed\n");
        printf("FAIL\n\n");
        return -1;
    }

    printf("Test normal kernel generation\n");
    if (!testGenFunc(ctx)) {
        printf("Generated code:\n\n");
        printf("%s", buf);
        printf("\n\nPASS\n\n");
    }
    else {
        printf("FAIL\n\n");
    }

    printf("Test function name extracting from the generated code\n");
    r = kgenGetLastFuncName(name, sizeof(name), ctx);
    if (r) {
        printf("FAIL\n");
    }
    else {
        if (strcmp((const char*)name, "strcpy")) {
            printf("Extracted names is %s must be strcpy\n", name);
            printf("FAIL\n\n");
            r = -1;
        }
        else {
            printf("PASS\n\n");
        }
    }

    destroyKgenContext(ctx);

    printf("Test source size calculating without actual source "
           "adding to any buffer\n");
    ctx = createKgenContext(NULL, 0, true);
    r = kgenAddStmt(ctx, strcpyImpl);
    if (!r) {
        s = kgenSourceSize(ctx);
        if (s != strlen(strcpyImpl)) {
            r = -1;
        }
    }
    if (r) {
        printf("FAIL\n\n");
    }
    else {
        printf("PASS\n\n");
    }
    destroyKgenContext(ctx);

    ctx = createKgenContext(buf, 5, true);

    if (!r) {
        printf("Test generation with insufficient buffer\n");
        if (testGenFunc(ctx)) {
            printf("PASS\n");
        }
        else {
            printf("FAIL\n");
            r = -1;
        }
    }

    return r;
}

// test case for kache error functionality
static int
errorCacheTestCase(
    const char *msg,
    struct KernelCache *kcache,
    solver_id_t sid,
    SubproblemDim *dims,
    unsigned int nrDims,
    cl_context context,
    cl_device_id device,
    unsigned long extra,
    Kernel *kern)
{
    KernelKey key;
    Kernel* krn1;
    int r;
    bool fail;

    key.device = device;
    key.context = context;
    key.nrDims = nrDims;
    memset(key.subdims, 0, sizeof(key.subdims));
    r = nrDims;
    if (nrDims > MAX_SUBDIMS)
        r = MAX_SUBDIMS;
    memcpy(key.subdims, dims, sizeof(SubproblemDim) * r);

    printf("%s", msg);
    if (kern == NULL) {
        krn1 = findKernel(kcache, sid, &key, &extra);
        fail = (krn1 != NULL);
    }
    else {
        r = addKernelToCache(kcache, sid, kern, &key, kernExtraCmp);
        fail = (r == 0);
    }

    if (fail) {
        printf("FAIL\n");
        r = -1;
    }
    else {
        printf("PASS\n");
        r = 0;
    }

    return r;
}

static int
testCache(cl_context context, cl_device_id device)
{
    int r = 0;
    int i, j;
    unsigned int k;
    const solver_id_t wrongSID = 15;
    struct KernelCache *kcache;
    KernelKey key;
    Kernel *kern[NR_TEST_PATTERNS][KERNELS_PER_PATTERN], *krn1;
    SubproblemDim dims[NR_TEST_PATTERNS][KERNELS_PER_PATTERN][MAX_SUBDIMS];
    unsigned int nrDims[NR_TEST_PATTERNS] = {1, 3, 2, 2, 1};
    unsigned long extra = 7, extra1;

    printf("Testing inserting and normal searching of kernels\n");
    kcache = createKernelCache(10, KCACHE_SIZE_LIMIT);

    key.device = device;
    key.context = context;

    for (i = 0; (i < NR_TEST_PATTERNS) && !r; i++) {
        for (j = 0; (j < KERNELS_PER_PATTERN) && !r; j++) {
            for (k = 0; k < nrDims[i]; k++) {
                dims[i][j][k].x = random() % 1000;
                if (k == 2) {
                    dims[i][j][k].y = SUBDIM_UNUSED;
                    dims[i][j][k].itemX = SUBDIM_UNUSED;
                }
                else {
                    dims[i][j][k].y = random() % 1000;
                    dims[i][j][k].itemX = random() % 1000;
                }
                dims[i][j][k].bwidth = random() % 1000;
                dims[i][j][k].itemY = random() % 1000;
            }

            kern[i][j] = allocKernel();
            kern[i][j]->extra = &extra;
            kern[i][j]->extraSize = sizeof(extra);
            key.nrDims = nrDims[i];
            memset(key.subdims, 0, sizeof(key.subdims));
            memcpy(key.subdims, dims[i][j], sizeof(SubproblemDim) * key.nrDims);
            r = addKernelToCache(kcache, i, kern[i][j], &key, kernExtraCmp);
        }
    }

    if (r) {
        printf("Error at addition to the cache, i = %d, j = %d\n", i, j);
        printf("FAIL\n");
    }
    else {
        // Now try to find each cached kernel
        extra1 = extra;
        for (i = 0; (i < NR_TEST_PATTERNS) && !r; i++) {
            for (j = 0; j < KERNELS_PER_PATTERN; j++) {
                key.nrDims = nrDims[i];
                memset(key.subdims, 0, sizeof(key.subdims));
                memcpy(key.subdims, dims[i][j], sizeof(SubproblemDim) * key.nrDims);
                krn1 = findKernel(kcache, i, &key, &extra1);
                if (krn1 != kern[i][j]) {
                    r = -1;
                    break;
                }
            }
        }
        if (r) {
            printf("First error occurred at pattern %d, kernel %d: ", i, j);
            if (krn1 == NULL) {
                printf("the kernel is not found\n");
            }
            else {
                printf("the kernel mismatch\n");
            }
        }
        else {
            printf("PASS\n");
        }
    }

    // cases for search error functionality
    dims[0][0][0].x = 1001;

    if (!r) {
        r = errorCacheTestCase("Try to search a kernel not being in "
                               "the cache\n",
                               kcache, 0, dims[0][0],
                               nrDims[0], context, device, extra, NULL);
    }

    if (!r) {
        r = errorCacheTestCase("Try To search a kernel with a wrong extra "
                               "information\n", kcache, 0, dims[0][1],
                               nrDims[0], context, device, extra - 2, NULL);
    }

    if (!r) {
        r = errorCacheTestCase("Try to search a kernel with a solver "
                               "ID\n", kcache, wrongSID,
                               dims[0][1], nrDims[0], context, device,
                               extra, NULL);
    }

    if (!r) {
        r = errorCacheTestCase("Try to search a kernel with a wrong number "
                               "of subproblem dimensions\n",
                               kcache, 0, dims[0][1], 500, context, device,
                               extra, NULL);
    }
    if (!r) {
        r = errorCacheTestCase("Try to search a kernel with bad OpenCL context\n",
                               kcache, 0, dims[0][1], 500, (cl_context)-1, device,
                               extra, NULL);
    }
    if (!r) {
        r = errorCacheTestCase("Try to search a kernel with bad OpenCL device\n",
                               kcache, 0, dims[0][1], 500, context,
                               (cl_device_id)-1, extra, NULL);
    }

    // error test cases for inserting to cache
    krn1 = allocKernel();
    krn1->extra = &extra;
    krn1->extraSize = sizeof(extra);

    if (!r) {
        r = errorCacheTestCase("Try to insert a kernel with a wrong solver "
                               "ID\n", kcache, wrongSID,
                               dims[0][0], nrDims[0], context, device,
                               extra, krn1);
    }

    if (!r) {
        r = errorCacheTestCase("Try to insert a kernel with a wrong number "
                               "of subproblem dimensions\n",
                               kcache, 0, dims[0][0],
                               500, context, device, extra, krn1);
    }

    return r;
}


int
main(void)
{
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context context;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs() failed with %d\n", err);
        return 1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs() failed with %d\n", err);
        return 1;
    }
    props[1] = (cl_context_properties)platform;
    context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext() failed with %d\n", err);
        return 1;
    }

    printf("Launch tests for kernel generators\n");
    printf("-----------------------------------------\n");
    if (!testGen()) {
        printf("-----------------------------------------\n\n");
        printf("Launch tests for kernel cache\n");
        printf("-----------------------------------------\n");
        testCache(context, device);
    }

    clReleaseContext(context);
    return 0;
}

