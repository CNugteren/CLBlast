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


#include <gtest/gtest.h>
#include <clBLAS.h>

class BasicRoutines : public testing::Test {
protected:
    BasicRoutines() :
        platform(0), device(0), context(NULL), queue(NULL) {
    }

    virtual ~BasicRoutines() {
    }

    virtual void SetUp() {
        cl_int err;
        cl_context_properties props[] = { CL_CONTEXT_PLATFORM, 0, 0 };

        ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(1, &platform, NULL));
        ASSERT_EQ(CL_SUCCESS,
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
        props[1] = (cl_context_properties)platform;
        context = clCreateContext(props, 1, &device, NULL, NULL, &err);
        ASSERT_EQ(CL_SUCCESS, err) << "clCreateContext() failed";
        queue = clCreateCommandQueue(context, device, 0, &err);
        ASSERT_EQ(CL_SUCCESS, err) << "clCreateCommandQueue() failed";
    }

    virtual void TearDown() {
        if (queue != NULL) {
            clReleaseCommandQueue(queue);
        }
        if (context != NULL) {
            clReleaseContext(context);
        }
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

TEST_F(BasicRoutines, UsualCodeFlow) {
    EXPECT_EQ(CL_SUCCESS, clblasSetup());
    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_PREBUILD_KERNELS(context));
    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_CLEANUP_KERNELS(context));
    clblasTeardown();
}

TEST_F(BasicRoutines, DoubleSetup) {
    EXPECT_EQ(CL_SUCCESS, clblasSetup());
    EXPECT_NE(clblasSetup(), CL_SUCCESS);
    clblasTeardown();
}

TEST_F(BasicRoutines, MissedSetup) {
    EXPECT_NE(AMD_clBLAS_PREBUILD_KERNELS(context), CL_SUCCESS);
}

TEST_F(BasicRoutines, BadContext) {
    EXPECT_EQ(CL_SUCCESS, clblasSetup());
    EXPECT_NE(AMD_clBLAS_PREBUILD_KERNELS(NULL), CL_SUCCESS);
    clblasTeardown();
}

TEST_F(BasicRoutines, TwoContexts) {
    cl_int err;
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context anotherContext;

    EXPECT_EQ(CL_SUCCESS, clblasSetup());

    props[1] = (cl_context_properties)platform;
    anotherContext = clCreateContext(props, 1, &device, NULL, NULL, &err);
    ASSERT_EQ(CL_SUCCESS, err) << "Need a context";
    ASSERT_NE(context, anotherContext) << "Contexts must be different";

    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_PREBUILD_KERNELS(context));
    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_PREBUILD_KERNELS(anotherContext));

    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_CLEANUP_KERNELS(context));
    EXPECT_EQ(CL_SUCCESS, AMD_clBLAS_CLEANUP_KERNELS(anotherContext));

    clReleaseContext(context);
    clblasTeardown();
}
