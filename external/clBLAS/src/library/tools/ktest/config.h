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


#ifndef KTEST_CONFIG_H__
#define KTEST_CONFIG_H__

#include <string>
#include <map>
#include <boost/program_options.hpp>

#include <clBLAS.h>
#include <clblas-internal.h>
#include <blas_funcs.h>

#include "ktest-common.h"

namespace po = boost::program_options;

namespace clMath {

typedef std::map<BlasFunctionID, std::string> NameMap;


class Config {
private:
    std::string defaultConfig_;
    std::string cpp_;
    std::string cl_;
    clMath::KTestMatrixGenerator dataPattern_;

    cl_platform_id platform_;
    cl_device_id device_;
    std::string buildOptions_;

	BlasFunctionID funcID_;
	CLBlasKargs kargs_;
    SubproblemDim subdims_[MAX_SUBDIMS];
    bool hasFuncID_;
    bool hasSubdims_;
    bool multiKernel_;
    bool skipAccuracy_;
    po::variables_map vm;

    NameMap names_;

    void setOptDesc(po::options_description& opts, bool useDefaults);
    bool applyOptions(const po::variables_map& vm, bool stopOnError = true);

    bool parseGroupSizeOpt(const std::string& opt);
    bool parseDecompositionOpt(const std::string& opt);
    bool parseArgMultiplier(const std::string& opt, ArgMultiplier& v);

public:
	Config();
	~Config();

    const std::string& cpp() const;
    const std::string& cl() const;
    clMath::KTestMatrixGenerator dataPattern() const;

    std::string platform() const;
    std::string device() const;
    const std::string& buildOptions() const;
    void kargs(CLBlasKargs *kargs) const;
    bool permitMultiKernels() const;
    bool withAccuracy() const;
    bool decomposition(SubproblemDim subdims[MAX_SUBDIMS]) const;
    BlasFunctionID blasFunctionID() const;

    void setDefaultConfig(const std::string& filename);

    void setCpp(const std::string& name);
    void setCl(const std::string& name);
    bool setDataPattern(const std::string& name);

    bool setPlatform(const std::string& name);
    bool setDevice(const std::string& name);
    void setBuildOptions(const std::string& options);

	bool setFunction(const std::string& name);

    void setOrder(clblasOrder order);
	void setSide(clblasSide side);
	void setUplo(clblasUplo uplo);
	void setTransA(clblasTranspose transA);
	void setTransB(clblasTranspose transB);
	void setDiag(clblasDiag diag);
    void setM(size_t M);
    void setN(size_t N);
    void setK(size_t K);
    void setAlpha(ArgMultiplier alpha);
    void setBeta(ArgMultiplier beta);
    void setLDA(size_t lda);
    void setLDB(size_t ldb);
    void setLDC(size_t ldc);
    void setOffA(size_t offA);
    void setOffBX(size_t offBX);
    void setOffCY(size_t offCY);
    void setIncX(int incx);
    void setIncY(int incy);

    void setMultiKernel(bool multiKernel);
    void setSkipAccuracy();
    void setDecomposition(size_t x0, size_t y0, size_t bwidth0,
        size_t x1, size_t y1, size_t bwidth1);

    bool parseCommandLine(int argc, char *argv[]);
    bool loadConfig(const char* filename);
    bool isSane();
};

}   // namespace clMath

#endif	// KTEST_CONFIG_H__
