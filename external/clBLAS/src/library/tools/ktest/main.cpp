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


#define __CL_ENABLE_EXCEPTIONS

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "step.h"
#include "ktest.h"

#include "steps/gemv.h"
#include "steps/symv.h"
#include "steps/gemm.h"
#include "steps/trmm.h"
#include "steps/trsm.h"
#include "steps/syrk.h"
#include "steps/syr2k.h"

#include <init.h>
#include <trace_malloc.h>

clMath::Step* getMasterStep(
    BlasFunctionID funcID,
    std::string platformName,
    std::string deviceName);
clMath::Step* getStep(ListNode *node);
void destroyPatterns(std::vector<clMath::Step*>& patterns);

cl_platform_id
getPlatform(const char *name);

cl_device_id
getDevice(
    cl_platform_id platform,
    const char *name);

int
main(int argc, char *argv[])
{
    clMath::Config cfg;
    cfg.setDefaultConfig("ktest.cfg");
    if (!cfg.parseCommandLine(argc, argv) || !cfg.isSane()) {
        return 1;
    }

    clblasSetup();
    parseEnvImplementation();

    clMath::Step *masterStep = getMasterStep(cfg.blasFunctionID(),
                                          cfg.platform(), cfg.device());
    if (masterStep == NULL) {
        std::cerr << "Function support not implemented yet" << std::endl;
        return 1;
    }

    CLBlasKargs kargs;
    SubproblemDim subdims[MAX_SUBDIMS];

    cfg.kargs(&kargs);
    masterStep->setKargs(kargs);
    masterStep->fixLD();

    ListHead seq;
    listInitHead(&seq);
    bool severalKernels = false;

    /* Single kernel for this function */
    if (cfg.decomposition(subdims)) {
        masterStep->setDecomposition(subdims);
    }
    masterStep->completeDecompositionSingle();

    if (cfg.permitMultiKernels()) {
        masterStep->makeSolutionSequence(&seq,
                                         getPlatform(cfg.platform().c_str()));
        if (listLength(&seq) > 1) {
            severalKernels = true;
        }
    }

    if (severalKernels) {
        std::ofstream fs;
        ListNode *node;

        std::vector<clMath::Step*> steps;

        masterStep->declareVars(NULL);

        for (node = listNodeFirst(&seq); node != &seq; node = node->next) {
            steps.push_back(getStep(node));
        }

        std::string str;

        for (unsigned int i = 0; i < steps.size(); i++) {
            std::stringstream kernelFileName;
            kernelFileName << i << "_" << steps[i]->getBlasFunctionName()
                                       << "_" << cfg.cl();

            steps[i]->setKernelName(kernelFileName.str());
            if (cfg.decomposition(subdims)) {
                steps[i]->setDecomposition(subdims);
            }

            steps[i]->completeDecompositionSingle();

            steps[i]->declareVars(masterStep);

            std::cout << "Generating '" << steps[i]->kernelName()
                                        << "' ..." << std::endl;

            str = steps[i]->generate();
            if (str.empty()) {
                std::cerr << "failed" << std::endl;
                abort();
            }
            fs.open(kernelFileName.str().c_str());
            fs << str;
            fs.close();
        }

        clMath::KTest *ktest = new clMath::KTest(masterStep, &steps, &cfg);

        std::cout << "Generating '" << cfg.cpp() << "' ..." << std::endl;
        str = ktest->generate(cfg.withAccuracy());
        if (str.empty()) {
            std::cerr << "failed" << std::endl;
            abort();
        }
        fs.open(cfg.cpp().c_str());
        fs << str;
        fs.close();

        delete ktest;

        for (std::vector<clMath::Step*>::iterator it = steps.begin();
                it != steps.end(); ++it) {
            delete (*it);
        }
        steps.clear();
    }
    else {

        std::ofstream fs;

        masterStep->setKernelName(cfg.cl());

        std::cout << "Generating '" << masterStep->kernelName()
                                    << "' ..." << std::endl;

        masterStep->declareVars(NULL);

        std::string str;
        str = masterStep->generate();
        if (str.empty()) {
            std::cerr << "failed" << std::endl;
            abort();
        }
        fs.open(cfg.cl().c_str());
        fs << str;
        fs.close();

        clMath::KTest *ktest = new clMath::KTest(masterStep, &cfg);

        std::cout << "Generating '" << cfg.cpp() << "' ..." << std::endl;
        str = ktest->generate(cfg.withAccuracy());
        if (str.empty()) {
            std::cerr << "failed" << std::endl;
            abort();
        }
        fs.open(cfg.cpp().c_str());
        fs << str;
        fs.close();

        delete ktest;
    }

    if (cfg.permitMultiKernels()) {
        masterStep->freeSolutionSequence(&seq);
    }

    delete masterStep;

    return 0;
}

clMath::Step* getMasterStep(
    BlasFunctionID funcID,
    std::string platformName,
    std::string deviceName)
{
    cl_platform_id platformID;
    cl_device_id deviceID;

    platformID = getPlatform(platformName.c_str());
    deviceID = getDevice(platformID, deviceName.c_str());

    switch (funcID) {
    case CLBLAS_GEMV:
        return new clMath::GemvStep(deviceID);
    case CLBLAS_SYMV:
        return new clMath::SymvStep(deviceID);
    case CLBLAS_GEMM:
        return new clMath::GemmStep(deviceID);
    case CLBLAS_TRMM:
        return new clMath::TrmmStep(deviceID);
    case CLBLAS_TRSM:
        return new clMath::TrsmStep(deviceID);
    case CLBLAS_SYRK:
        return new clMath::SyrkStep(deviceID);
    case CLBLAS_SYR2K:
        return new clMath::Syr2kStep(deviceID);
    default:
        return NULL;
    }
}

clMath::Step* getStep(ListNode *node)
{
    switch (clMath::Step::getStepNodeFuncID(node)) {
    case CLBLAS_GEMV:
        return new clMath::GemvStep(node);
    case CLBLAS_SYMV:
        return new clMath::SymvStep(node);
    case CLBLAS_GEMM:
        return new clMath::GemmStep(node);
    case CLBLAS_TRMM:
        return new clMath::TrmmStep(node);
    case CLBLAS_TRSM:
        return new clMath::TrsmStep(node);
    case CLBLAS_SYRK:
        return new clMath::SyrkStep(node);
    case CLBLAS_SYR2K:
        return new clMath::Syr2kStep(node);
    default:
        return NULL;
    }
}


cl_platform_id
getPlatform(const char *name)
{
    cl_int err;
    cl_uint nrPlatforms, i;
    cl_platform_id *list, platform;
    char platformName[64];

    err = clGetPlatformIDs(0, NULL, &nrPlatforms);
    if (err != CL_SUCCESS) {
        return NULL;
    }

    list = (cl_platform_id*)calloc(nrPlatforms, sizeof(*list));
    if (list == NULL) {
        return NULL;
    }

    err = clGetPlatformIDs(nrPlatforms, list, NULL);
    if (err != CL_SUCCESS) {
        free(list);
        return NULL;
    }

    platform = NULL;
    for (i = 0; i < nrPlatforms; i++) {
        err = clGetPlatformInfo(list[i], CL_PLATFORM_NAME,
            sizeof(platformName), platformName, NULL);
        if ((err == CL_SUCCESS) && (strcmp(platformName, name) == 0)) {
            platform = list[i];
            break;
        }
    }

    free(list);
    return platform;
}

cl_device_id
getDevice(
    cl_platform_id platform,
    const char *name)
{

    cl_int err;
    cl_uint nrDevices, i;
    cl_device_id *list, device;
    char deviceName[64];

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nrDevices);
    if (err != CL_SUCCESS) {
        return NULL;
    }
    list = (cl_device_id*)calloc(nrDevices, sizeof(*list));
    if (list == NULL) {
        return NULL;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nrDevices, list, NULL);
    if (err != CL_SUCCESS) {
        free(list);
        return NULL;
    }

    device = NULL;
    for (i = 0; i < nrDevices; i++) {
        err = clGetDeviceInfo(list[i], CL_DEVICE_NAME,
            sizeof(deviceName), deviceName, NULL);
        if ((err == CL_SUCCESS) && (strcmp(deviceName, name) == 0)) {
            device = list[i];
            break;
        }
    }

    free(list);
    return device;
}
