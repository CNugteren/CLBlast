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


#include <sys/types.h>

#include <blas_funcs.h>
#include <solution_seq.h>

#include "config.h"

using namespace clMath;

static const char DEFAULT_PLATFORM_NAME[] = "AMD Accelerated Parallel Processing";

Config::Config() :
    defaultConfig_(""),
    cpp_("ktest.cpp"),
    dataPattern_(RANDOM_MATRIX),
    buildOptions_(""),
	funcID_(CLBLAS_GEMM),
    hasFuncID_(false), hasSubdims_(false),
    skipAccuracy_(false)
{
    setPlatform(DEFAULT_PLATFORM_NAME);
    setDevice("");

    memset(&kargs_, 0, sizeof(kargs_));
    kargs_.kernType = CLBLAS_COMPUTING_KERNEL;
    kargs_.A = kargs_.B = kargs_.C = NULL;
    kargs_.offsetM = kargs_.offsetN = 0;
    kargs_.scimage[0] = kargs_.scimage[1] = NULL;
    kargs_.addrBits = 0;

    kargs_.dtype = TYPE_FLOAT;
    kargs_.order = clblasRowMajor;
    kargs_.side = clblasLeft;
    kargs_.uplo = clblasUpper;
    kargs_.transA = clblasNoTrans;
    kargs_.transB = clblasNoTrans;
    kargs_.diag = clblasNonUnit;
    kargs_.M = kargs_.N = kargs_.K = 0;
    kargs_.lda.matrix = kargs_.ldb.matrix = kargs_.ldc.matrix = 0;
    kargs_.offA = kargs_.offBX = kargs_.offCY = 0;

    memset(&kargs_.alpha, 0, sizeof(kargs_.alpha));
    memset(&kargs_.beta, 0, sizeof(kargs_.beta));

    memset(subdims_, 0, sizeof(subdims_));

    names_[CLBLAS_GEMV] = "gemv";
    names_[CLBLAS_SYMV] = "symv";
    names_[CLBLAS_GEMM] = "gemm";
    names_[CLBLAS_TRMM] = "trmm";
    names_[CLBLAS_TRSM] = "trsm";
    names_[CLBLAS_SYRK] = "syrk";
    names_[CLBLAS_SYR2K] = "syr2k";

    cl_ = names_[funcID_] + ".cl";
}

Config::~Config()
{
    names_.clear();
}

const std::string&
Config::cpp() const
{
    return cpp_;
}

const std::string&
Config::cl() const
{
    return cl_;
}

clMath::KTestMatrixGenerator
Config::dataPattern() const
{
    return dataPattern_;
}

std::string
Config::platform() const
{
    std::string name;
    cl_int err;
    size_t sz;
    char *pname;

    err = clGetPlatformInfo(platform_, CL_PLATFORM_NAME, 0, NULL, &sz);
    if (err != CL_SUCCESS) {
        return "";
    }
    pname = new char[sz + 1];
    err = clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sz, pname, NULL);
    if (err != CL_SUCCESS) {
        delete[] pname;
        return "";
    }
    name = pname;
    delete[] pname;
    return name;
}

std::string
Config::device() const
{
    std::string name;
    cl_int err;
    size_t sz;
    char *dname;

    err = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, NULL, &sz);
    if (err != CL_SUCCESS) {
        return "";
    }
    dname = new char[sz + 1];
    err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sz, dname, NULL);
    if (err != CL_SUCCESS) {
        delete[] dname;
        return "";
    }
    name = dname;
    delete[] dname;
    return name;
}

const std::string&
Config::buildOptions() const
{
    return buildOptions_;
}

void
Config::kargs(CLBlasKargs *kargs) const
{
    cl_int err;

    *kargs = kargs_;
    kargs->addrBits = deviceAddressBits(device_, &err);
}

bool
Config::permitMultiKernels() const
{
    return multiKernel_;
}

bool Config::withAccuracy() const
{
    return !skipAccuracy_;
}

bool
Config::decomposition(SubproblemDim subdims[MAX_SUBDIMS]) const
{
    if (!hasSubdims_)  {
        return false;
    }

    for (int i = 0; i < MAX_SUBDIMS; i++) {
        subdims[i] = subdims_[i];
    }

    subdims[0].itemX = subdims[0].x;
    subdims[0].itemY = subdims[0].y;
    subdims[1].itemX = subdims[1].x;
    subdims[1].itemY = subdims[1].y;

    return true;
}

BlasFunctionID
Config::blasFunctionID() const
{
    return funcID_;
}

void
Config::setDefaultConfig(const std::string& filename)
{
    defaultConfig_ = filename;
}

void
Config::setCpp(const std::string& name)
{
    cpp_ = name;
}

void
Config::setCl(const std::string& name)
{
    cl_ = name;
}

bool
Config::setDataPattern(const std::string& name)
{
    if (strcmp(name.c_str(), "random") == 0) {
        dataPattern_ = clMath::RANDOM_MATRIX;
        return true;
    }
    if (strcmp(name.c_str(), "unit") == 0) {
        dataPattern_ = clMath::UNIT_MATRIX;
        return true;
    }
    if (strcmp(name.c_str(), "sawtooth") == 0) {
        dataPattern_ = clMath::SAWTOOTH_MATRIX;
        return true;
    }
    return false;
}

bool
Config::setPlatform(const std::string& name)
{
    cl_int err;
    cl_uint nrPlatforms;
    cl_platform_id *platforms;
    bool found;
    size_t sz;
    char *pname;

    err = clGetPlatformIDs(0, NULL, &nrPlatforms);
    if ((err != CL_SUCCESS) || (nrPlatforms == 0)) {
        return false;
    }
    platforms = new cl_platform_id[nrPlatforms];
    err = clGetPlatformIDs(nrPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        delete[] platforms;
        return false;
    }

    found = false;
    for (cl_uint i = 0; i < nrPlatforms; i++) {
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &sz);
        if (err != CL_SUCCESS) {
            continue;
        }
        pname = new char[sz + 1];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sz, pname, NULL);
        if (err != CL_SUCCESS) {
            delete[] pname;
            continue;
        }
        if (name.empty()) {
            found = (strcmp(pname, DEFAULT_PLATFORM_NAME) == 0);
        }
        else {
            found = (strcmp(pname, name.c_str()) == 0);
        }
        delete[] pname;
        if (found) {
            platform_ = platforms[i];
            break;
        }
    }

    delete[] platforms;
    return found;
}

bool
Config::setDevice(const std::string& name)
{
    cl_int err;
    cl_uint nrDevices;
    cl_device_id *devices;
    bool found;
    size_t sz;
    char *dname;

    if ((platform_ == NULL) && !setPlatform("")) {
        return false;
    }

    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, NULL, &nrDevices);
    if ((err != CL_SUCCESS) || (nrDevices == 0)) {
        return false;
    }
    devices = new cl_device_id[nrDevices];
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, nrDevices, devices, NULL);
    if (err != CL_SUCCESS) {
        delete[] devices;
        return false;
    }

    found = false;
    for (cl_uint i = 0; i < nrDevices; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &sz);
        if (err != CL_SUCCESS) {
            continue;
        }
        dname = new char[sz + 1];
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sz, dname, NULL);
        if (err != CL_SUCCESS) {
            delete[] dname;
            continue;
        }
        if (name.empty()) {
            found = true;
        }
        else {
            found = (strcmp(dname, name.c_str()) == 0);
        }
        delete[] dname;
        if (found) {
            device_ = devices[i];
            break;
        }
    }

    delete[] devices;
    return found;
}

void
Config::setBuildOptions(const std::string& options)
{
    buildOptions_ = options;
}

bool
Config::setFunction(const std::string& name)
{
    if (name.empty()) {
        return false;
    }
    switch (name.c_str()[0]) {
    case 's':
    case 'S':
        kargs_.dtype = TYPE_FLOAT;
        break;
    case 'd':
    case 'D':
        kargs_.dtype = TYPE_DOUBLE;
        break;
    case 'c':
    case 'C':
        kargs_.dtype = TYPE_COMPLEX_FLOAT;
        break;
    case 'z':
    case 'Z':
        kargs_.dtype = TYPE_COMPLEX_DOUBLE;
        break;
    default:
        return false;
    }

    for (NameMap::iterator it = names_.begin(); it != names_.end(); ++it) {
        if (strcmp(name.substr(1).c_str(), (*it).second.c_str()) == 0) {
            funcID_ = (*it).first;
            setCl((*it).second + ".cl");
            hasFuncID_ = true;
            return true;
        }
    }
    return false;
}

void
Config::setOrder(clblasOrder order)
{
    kargs_.order = order;
}

void
Config::setSide(clblasSide side)
{
    kargs_.side = side;
}

void
Config::setUplo(clblasUplo uplo)
{
    kargs_.uplo = uplo;
}

void
Config::setTransA(clblasTranspose transA)
{
    kargs_.transA = transA;
}

void
Config::setTransB(clblasTranspose transB)
{
    kargs_.transB = transB;
}

void
Config::setDiag(clblasDiag diag)
{
    kargs_.diag = diag;
}

void
Config::setM(size_t M)
{
    kargs_.M = M;
}

void
Config::setN(size_t N)
{
    kargs_.N = N;
}

void
Config::setK(size_t K)
{
    kargs_.K = K;
}

void
Config::setAlpha(ArgMultiplier alpha)
{
    switch (kargs_.dtype) {
    case TYPE_FLOAT:
        kargs_.alpha.argFloat = alpha.argFloat;
        break;
    case TYPE_DOUBLE:
        kargs_.alpha.argDouble = alpha.argDouble;
        break;
    case TYPE_COMPLEX_FLOAT:
        kargs_.alpha.argFloatComplex = alpha.argFloatComplex;
        break;
    case TYPE_COMPLEX_DOUBLE:
        kargs_.alpha.argDoubleComplex = alpha.argDoubleComplex;
        break;
    }
}

void
Config::setBeta(ArgMultiplier beta)
{
    switch (kargs_.dtype) {
    case TYPE_FLOAT:
        kargs_.beta.argFloat = beta.argFloat;
        break;
    case TYPE_DOUBLE:
        kargs_.beta.argDouble = beta.argDouble;
        break;
    case TYPE_COMPLEX_FLOAT:
        kargs_.beta.argFloatComplex = beta.argFloatComplex;
        break;
    case TYPE_COMPLEX_DOUBLE:
        kargs_.beta.argDoubleComplex = beta.argDoubleComplex;
        break;
    }
}

void
Config::setLDA(size_t lda)
{
    kargs_.lda.matrix = lda;
}

void
Config::setLDB(size_t ldb)
{
    kargs_.ldb.matrix = ldb;
}

void
Config::setLDC(size_t ldc)
{
    kargs_.ldc.matrix = ldc;
}

void
Config::setIncX(int incx)
{
    kargs_.ldb.vector = incx;
}

void
Config::setIncY(int incy)
{
    kargs_.ldc.vector = incy;
}

void
Config::setOffA(size_t offA)
{
    kargs_.offA = offA;
}

void
Config::setOffBX(size_t offBX)
{
    kargs_.offBX = offBX;
}

void
Config::setOffCY(size_t offCY)
{
    kargs_.offCY = offCY;
}

void
Config::setMultiKernel(bool multiKernel)
{
    multiKernel_ = multiKernel;
}

void
Config::setSkipAccuracy(void)
{
    skipAccuracy_ = true;
}

void
Config::setDecomposition(
    size_t x0,
    size_t y0,
    size_t bwidth0,
    size_t x1,
    size_t y1,
    size_t bwidth1)
{
    subdims_[0].x = x0;
    subdims_[0].y = y0;
    subdims_[0].bwidth = bwidth0;
    subdims_[1].x = x1;
    subdims_[1].y = y1;
    subdims_[1].bwidth = bwidth1;

    hasSubdims_ = true;
}
