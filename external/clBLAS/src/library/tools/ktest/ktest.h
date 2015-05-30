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


#ifndef KTEST_KTEST_H__
#define KTEST_KTEST_H__

#include <string>
#include <sstream>

#include "ktest-common.h"
#include "step.h"
#include "config.h"

namespace clMath {

/**
 * @internal
 * @brief Host code generation class
 *
 * Object of this class generate host-side source file that can execute kernels
 * for one or several steps.
 *
 */
class KTest {
private:
    std::string platform_;
    std::string device_;
    std::string kernelSourceFile_;
    std::string buildOptions_;
    KTestMatrixGenerator matrixGen_;
    Step *masterStep_;
    std::vector<Step*> *steps_;
    size_t indent_;
    bool useSeveralKernels_;

    const char* matrixGenName(KTestMatrixGenerator gen);
    const char* vectorGenName(KTestMatrixGenerator gen);

    void typedefs(std::stringstream& ss);
    void declareKTestOptions(std::stringstream& ss);
    void declareBlasOptions(std::stringstream& ss, Step *step);
    void declarePatternVars(std::stringstream& ss, Step *step);
    void generateMain(std::stringstream& ss, bool withAccuracy);

    void loadFile(std::stringstream& ss);

    void randomVector(std::stringstream& ss);
    void unitVector(std::stringstream& ss);
    void sawtoothVector(std::stringstream& ss);
    void compareVectors(std::stringstream& ss);

    void randomMatrix(std::stringstream& ss);
    void unitMatrix(std::stringstream& ss);
    void sawtoothMatrix(std::stringstream& ss);
    void setUpTRSMDiagonal(std::stringstream& ss);
    void compareMatrices(std::stringstream& ss);

    std::string indent();

    void includes(std::stringstream& ss);
    void forwardDeclarations(std::stringstream& ss);
    void declareGranulation(std::stringstream& ss, Step *step);
    void mainInit(std::stringstream& ss);
    void buildKernel(std::stringstream& ss);
    void allocateWriteBuffers(std::stringstream& ss);
    void setKernelArgs(std::stringstream& ss, Step *step);
    void execKernel(std::stringstream& ss);
    void readBuffers(std::stringstream& ss);
    void mainFinish(std::stringstream& ss);
    void auxFunctions(std::stringstream& ss);

    void getPlatform(std::stringstream& ss);
    void getDevice(std::stringstream& ss);
    void createKernel(std::stringstream& ss);
    void printExecTime(std::stringstream& ss);
    void getBufferSizeExpr(Variable *buffer, std::string& size);
public:
    KTest(Step *masterStep, clMath::Config *cfg);
    KTest(Step *masterStep, std::vector<clMath::Step*> *steps, clMath::Config *cfg);

    std::string generate(bool withAccuracy);
};

}   // namespace clMath

#endif  // KTEST_KTEST_H__
