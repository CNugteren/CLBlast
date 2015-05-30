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

#include <common.h>
#include <BlasBase.h>

namespace clMath {

clblasStatus
BlasBase::addScratchImages(void)
{
    //clblasStatus status;

    //// Height must be less than 1024
    //imageA_ = clblasAddScratchImage(context_, 2048, 512, &status);
    //if (imageA_) {
    //    imageB_ = clblasAddScratchImage(context_, 2048, 512, &status);
    //}

    //return status;
	return clblasNotImplemented;
}

}   // namespace
