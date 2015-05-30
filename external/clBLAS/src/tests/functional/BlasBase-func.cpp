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
#include <math.h>
#include <clBLAS.h>

#include <common.h>
#include <BlasBase.h>

namespace clMath {

static size_t
imageMaxDimension(cl_context context, int widthHeight)
{
    cl_int err;
    cl_device_id devices[2];
    size_t i, retSize;
    size_t rc = (size_t)-1;
    cl_device_info par;

    par = (widthHeight) ? CL_DEVICE_IMAGE2D_MAX_HEIGHT :
                          CL_DEVICE_IMAGE2D_MAX_WIDTH;

    err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
        sizeof(devices), devices, &retSize);
    if (err == CL_SUCCESS) {
        size_t s;

        retSize /= sizeof(cl_device_id);
        for (i = 0; (i < retSize) && (err == CL_SUCCESS); i++) {
            err = clGetDeviceInfo(devices[i], par, sizeof(s), &s, NULL);
            if (err == CL_SUCCESS) {
                rc = std::min(rc, s);
            }
        }
    }

    if (err != CL_SUCCESS) {
        rc = 0;
    }

    return rc;
}

static size_t
imageMaxWidth(cl_context context)
{
    return imageMaxDimension(context, 0);
}

static size_t
imageMaxHeight(cl_context context)
{
    return imageMaxDimension(context, 1);
}

clblasStatus
BlasBase::addScratchImages(void)
{
    //cl_ulong memSize, allocSize;
    //size_t width, height;
    //clblasStatus status;
    //float scale;

    ///*
    // * get maximum amount of memory each image can takes, not
    // * forgetting that it can be up to three matrices residing
    // * in memory objects
    // */
    //allocSize = maxMemAllocSize();
    //memSize = availGlobalMemSize(0);
    //if (allocSize > memSize / 5) {
    //    allocSize = memSize / 5;
    //    scale = 1.4f;
    //}
    //else {
    //    scale = 1.5f;
    //}

    //height = static_cast<size_t>(sqrt(static_cast<double>(allocSize) / sizeof(cl_float)));
    //width  = height / 4;
    //height = static_cast<size_t>(height / scale);
    //width  = static_cast<size_t>(width * scale);

    //if (height > imageMaxHeight(context_)) {
    //    height = imageMaxHeight(context_);
    //}
    //if (width > imageMaxWidth(context_)) {
    //    width = imageMaxWidth(context_);
    //}

    //imageA_ = clblasAddScratchImage(context_, width, height, &status);
    //if (imageA_) {
    //    imageB_ = clblasAddScratchImage(context_, width, height, &status);
    //}

    //return status;
	return clblasNotImplemented;

}

}   // namespace
