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


#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <devinfo.h>

static cl_ulong closestPowerOf2(cl_ulong x);

static const char L2BENCH_NAME[] = "l2Bench";
static const char *L2BENCH =
    "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |   \n"
    "                               CLK_ADDRESS_NONE            |   \n"
    "                               CLK_FILTER_NEAREST;             \n"

    "__kernel                                                       \n"
    "void l2Bench(                                                  \n"
    "    __read_only image2d_t in,                                  \n"
    "    size_t rounds,                                             \n"
    "    __global float4 *out)                                      \n"
    "{                                                              \n"
    "    int width, height;                                         \n"
    "    size_t gid, nrWorkItems;                                   \n"
    "    size_t pixelsPerWorkItem;                                  \n"
    "    size_t x, y, k, i;                                         \n"
    "    float4 v, sum;                                             \n"

    "    width = get_image_width(in);                               \n"
    "    height = get_image_height(in);                             \n"

    "    gid = get_global_id(0);                                    \n"
    "    nrWorkItems = get_global_size(0);                          \n"

    "    pixelsPerWorkItem = (width * height) / nrWorkItems;        \n"

    "    sum = (float4)(0.0);                                       \n"

    "    for (k = 0; k < rounds; k++) {                             \n"
    "        x = (gid * pixelsPerWorkItem) % width;                 \n"
    "        y = (gid * pixelsPerWorkItem) / width;                 \n"

    "        for (i = 0; i < pixelsPerWorkItem; i++) {              \n"
    "            v = read_imagef(in, sampler, (int2)(x, y));        \n"
    "            sum += v;                                          \n"

    "            x++;                                               \n"
    "            y += x / width;                                    \n"
    "            x %= width;                                        \n"

    "        }                                                      \n"
    "    }                                                          \n"
    "    *out = sum;                                                \n"
    "}                                                              \n";

static const char L1BENCH_NAME[] = "l1Bench";
static const char *L1BENCH =
    "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |   \n"
    "                               CLK_ADDRESS_NONE            |   \n"
    "                               CLK_FILTER_NEAREST;             \n"

    "__kernel                                                       \n"
    "void l1Bench(                                                  \n"
    "    __read_only image2d_t in,                                  \n"
    "    size_t l2Size,                                             \n"
    "    size_t rounds,                                             \n"
    "    __global float4 *out)                                      \n"
    "{                                                              \n"
    "    int width, height;                                         \n"
    "    size_t gid, nrWorkItems;                                   \n"
    "    size_t pixelsPerWorkItem;                                  \n"
    "    size_t x, y, k, i;                                         \n"
    "    float4 v, sum;                                             \n"

    "    width = get_image_width(in);                               \n"
    "    height = get_image_height(in);                             \n"

    "    gid = get_global_id(0);                                    \n"
    "    nrWorkItems = get_global_size(0);                          \n"

    "    pixelsPerWorkItem = (width * height) / nrWorkItems;        \n"

    "    sum = (float4)(0.0);                                       \n"

    "    for (k = 0; k < rounds; k++) {                             \n"
    "        x = (gid * pixelsPerWorkItem) % width;                 \n"
    "        y = (gid * pixelsPerWorkItem) / width;                 \n"

    "        for (i = 0; i < pixelsPerWorkItem - l2Size / sizeof(float4); i++) { \n"
    "            v = read_imagef(in, sampler, (int2)(x, y));        \n"
    "            sum += v;                                          \n"

    "            x++;                                               \n"
    "            y += x / width;                                    \n"
    "            x %= width;                                        \n"

    "        }                                                      \n"
    "    }                                                          \n"
    "    *out = sum;                                                \n"
    "}                                                              \n";

cl_ulong
deviceL2CacheSize(
    cl_device_id device,
    cl_int *error)
{
    const size_t MAX_CACHE_SIZE = 1024 * 1024;
    const size_t MIN_CACHE_SIZE =    1 * 1024;
    const size_t STEP           =    4 * 1024;

    /* Bigger number of rounds increases time measurement precision,
     * but slows the test down.
     */
    const unsigned int ROUNDS = 32;

    /* Repeat each kernel run sereval times for higher reliability. */
    const unsigned int RELIABILITY_ROUNDS = 5;

    cl_int err;

    cl_uint maxComputeUnits;
    cl_bool imageSupport;

    cl_platform_id platform;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_event event;

    cl_float *in;
    size_t width, height;
    const cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_mem imgIn;
    size_t origin[3], region[3];

    cl_float4 out;
    cl_mem bufOut;

    size_t global_work_size, local_work_size;
    cl_ulong start, end, avg;
    cl_long *times;
    cl_double d, max;

    size_t steps;
    size_t i, t;

    /* Collect device properties. */
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl_uint), &maxComputeUnits, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
        sizeof(cl_bool), &imageSupport, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    if (imageSupport == CL_FALSE) {
        if (error != NULL) {
            *error = CL_INVALID_OPERATION;  /* like clCreateImage2D() does */
        }
        return 0;
    }

    steps = (MAX_CACHE_SIZE - MIN_CACHE_SIZE) / STEP;
    times = calloc(steps, sizeof(cl_long));
    if (times == NULL) {
        if (error != NULL) {
            *error = CL_OUT_OF_HOST_MEMORY;
        }
        return 0;
    }

    /* Create necessary OpenCL objects */
    err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
        sizeof(cl_platform_id), &platform, NULL);
    if (err != CL_SUCCESS) {
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    program = clCreateProgramWithSource(ctx, 1, &L2BENCH, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    kernel = clCreateKernel(program, L2BENCH_NAME, &err);
    clReleaseProgram(program);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    /* Main idea of this test is to run one work-item on each compute unit.
     * This will make clear L2 cache hit/miss picture.
     */
    global_work_size = maxComputeUnits;
    local_work_size = 1;

    /* Prepare output buffer */
    bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        sizeof(cl_float4), &out, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    for (t = 0; t < steps; t++) {
        width = (size_t)sqrt((double)(MAX_CACHE_SIZE - t * STEP) / sizeof(cl_float4));
        height = width;

        /* Prepare image buffer */
        in = calloc(width * height, sizeof(cl_float4));
        if (in == NULL) {
            clReleaseMemObject(bufOut);
            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);
            free(times);
            if (error != NULL) {
                *error = CL_OUT_OF_HOST_MEMORY;
            }
            return 0;
        }
        imgIn = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            &format, width, height, 0, in, &err);
        if (err != CL_SUCCESS) {
            free(in);
            clReleaseMemObject(bufOut);
            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);
            free(times);
            if (error != NULL) {
                *error = err;
            }
            return 0;
        }
        origin[0] = origin[1] = origin[2] = 0;
        region[0] = width;
        region[1] = height;
        region[2] = 1;

        avg = 0;
        for (i = 0; i < RELIABILITY_ROUNDS; i++) {
            err = clEnqueueWriteImage(queue, imgIn, CL_TRUE, origin, region,
                0, 0, in, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clEnqueueWriteBuffer(queue, bufOut, CL_TRUE, 0,
                sizeof(cl_float4), &out, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imgIn);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clSetKernelArg(kernel, 1, sizeof(ROUNDS), &ROUNDS);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOut);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                &global_work_size, &local_work_size, 0, NULL, &event);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clWaitForEvents(1, &event);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            start = end = 0UL;
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                sizeof(cl_ulong), &start, NULL);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                sizeof(cl_ulong), &end, NULL);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            clReleaseEvent(event);

            /* NOTE: Sometimes the difference between start and end times
             * can be unexpectedly large - a tens of seconds.
             * This is a wrong behavior.
             */
            //assert(end - start < 10000000000UL);

            avg += end - start;
        }

        times[t] = avg / (width * height);

        clReleaseMemObject(imgIn);
        free(in);
    }
    clReleaseMemObject(bufOut);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    max = 0;
    i = MAX_CACHE_SIZE + 1;
    for (t = 1; t < steps; t++) {
        d = (cl_double)times[t - 1];
        d /= times[t];
        if (d > max) {
            max = d;
            i = MAX_CACHE_SIZE - t * STEP;
        }
    }
    free(times);

    if (i == MAX_CACHE_SIZE + 1)
        return 0;
    return closestPowerOf2(i);
}

cl_ulong
deviceL1CacheSize(
    cl_device_id device,
    cl_ulong l2CacheSize,
    cl_int *error)
{
    const size_t MIN_CACHE_SIZE = 1024;
    const size_t STEP           = 1024;
    size_t L2_SIZE              = (size_t)l2CacheSize;

    /* Bigger number of rounds increases time measurement precision,
     * but slows the test down.
     */
    const unsigned int ROUNDS = 64;

    /* Repeat each kernel run sereval times for higher reliability. */
    const unsigned int RELIABILITY_ROUNDS = 10;

    cl_int err;

    cl_uint maxComputeUnits;
    cl_bool imageSupport;

    cl_platform_id platform;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_event event;

    cl_float *in;
    size_t width, height;
    const cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_mem imgIn;
    size_t origin[3], region[3];

    cl_float4 out;
    cl_mem bufOut;

    size_t global_work_size, local_work_size;
    cl_ulong start, end, avg;
    cl_long *times;
    cl_double d, max;

    size_t steps;
    size_t i, t;

    /* Collect device properties. */
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl_uint), &maxComputeUnits, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
        sizeof(cl_bool), &imageSupport, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    if (imageSupport == CL_FALSE) {
        if (error != NULL) {
            *error = CL_INVALID_OPERATION;  /* like clCreateImage2D() does */
        }
        return 0;
    }

    steps = 1 + (L2_SIZE - MIN_CACHE_SIZE) / STEP;
    times = calloc(steps, sizeof(cl_long));
    if (times == NULL) {
        if (error != NULL) {
            *error = CL_OUT_OF_HOST_MEMORY;
        }
        return 0;
    }

    /* Create necessary OpenCL objects */
    err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
        sizeof(cl_platform_id), &platform, NULL);
    if (err != CL_SUCCESS) {
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    program = clCreateProgramWithSource(ctx, 1, &L1BENCH, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    kernel = clCreateKernel(program, L1BENCH_NAME, &err);
    clReleaseProgram(program);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    /* Main idea of this test is to run one work-item on each compute unit.
     * Image region assigned to one work-item consists of two parts:
     *     - part with size of probable L1 cache
     *     - part with size of L2 cache
     * This makes cache misses in L1 to be misses in L2 as well.
     * It is also assumed, that each Compute Unit has its own L1 cache.
     */
    global_work_size = maxComputeUnits;
    local_work_size = 1;

    /* Prepare output buffer */
    bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        sizeof(cl_float4), &out, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(times);
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }

    for (t = 0; t < steps; t++) {

        width = 64;         /* One image line takes 1KB */
        height = (L2_SIZE - t * STEP + L2_SIZE) * global_work_size /
                        (sizeof(cl_float4) * width);

        /* Prepare image buffer */
        in = calloc(width * height, sizeof(cl_float4));
        if (in == NULL) {
            clReleaseMemObject(bufOut);
            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);
            free(times);
            if (error != NULL) {
                *error = CL_OUT_OF_HOST_MEMORY;
            }
            return 0;
        }
        imgIn = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            &format, width, height, 0, in, &err);
        if (err != CL_SUCCESS) {
            free(in);
            clReleaseMemObject(bufOut);
            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);
            free(times);
            if (error != NULL) {
                *error = err;
            }
            return 0;
        }
        origin[0] = origin[1] = origin[2] = 0;
        region[0] = width;
        region[1] = height;
        region[2] = 1;

        avg = 0;
        for (i = 0; i < RELIABILITY_ROUNDS; i++) {
            err = clEnqueueWriteImage(queue, imgIn, CL_TRUE, origin, region,
                0, 0, in, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clEnqueueWriteBuffer(queue, bufOut, CL_TRUE, 0,
                sizeof(cl_float4), &out, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imgIn);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clSetKernelArg(kernel, 1, sizeof(L2_SIZE), &L2_SIZE);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clSetKernelArg(kernel, 2, sizeof(ROUNDS), &ROUNDS);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOut);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                &global_work_size, &local_work_size, 0, NULL, &event);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clWaitForEvents(1, &event);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            start = end = 0UL;
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                sizeof(cl_ulong), &start, NULL);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                sizeof(cl_ulong), &end, NULL);
            if (err != CL_SUCCESS) {
                clReleaseEvent(event);
                clReleaseMemObject(imgIn);
                free(in);
                clReleaseMemObject(bufOut);
                clReleaseCommandQueue(queue);
                clReleaseContext(ctx);
                free(times);
                if (error != NULL) {
                    *error = err;
                }
                return 0;
            }

            clReleaseEvent(event);

            /* NOTE: Sometimes the difference between start and end times
             * can be unexpectedly large - a tens of seconds.
             * This is a wrong behavior.
             */
            //assert(end - start < 10000000000UL);

            avg += end - start;
        }

        times[t] = avg / ((L2_SIZE - t * STEP) * global_work_size);

        clReleaseMemObject(imgIn);
        free(in);
    }
    clReleaseMemObject(bufOut);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    max = 0;
    i = L2_SIZE + 1;
    for (t = 1; t < steps; t++) {
        d = (cl_double)times[t - 1];
        d /= times[t];
        if (d > max) {
            max = d;
            i = L2_SIZE - t * STEP;
        }
    }
    free(times);

    if (i == L2_SIZE + 1)
        return 0;
    return closestPowerOf2(i);
}

cl_uint
deviceL1CacheAssoc(
    cl_device_id device,
    cl_ulong l1CacheSize,
    cl_int *error)
{
    /* TODO: Implementation needed. */

    (void)device;
    (void)l1CacheSize;

    if (error != NULL) {
        *error = CL_SUCCESS;
    }
    return 32;
}

static cl_ulong
closestPowerOf2(cl_ulong x)
{
    cl_ulong below, above;

    if (x == 0) {
        return 0;
    }
    for (above = 1; above < x; above <<= 1) {
        ; /* just iterate */
    }
    if (above == x) {
        return x;
    }
    below = above >> 1;

    if ((x - below) < (above - x)) {
        return below;
    }
    return above;
}
