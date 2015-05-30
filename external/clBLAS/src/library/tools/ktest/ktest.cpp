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


#include <blas_funcs.h>
#include "var.h"
#include "ktest.h"
#include "ktest-patterns.h"

using namespace clMath;

KTest::KTest(Step *step, clMath::Config *cfg) :
    platform_(cfg->platform()), device_(cfg->device()),
    kernelSourceFile_(cfg->cl()), buildOptions_(cfg->buildOptions()),
    matrixGen_(cfg->dataPattern()),
    masterStep_(step), indent_(0), useSeveralKernels_(false)
{
}

KTest::KTest(Step *masterStep, std::vector<Step*> *steps, clMath::Config *cfg) :
    platform_(cfg->platform()), device_(cfg->device()),
    kernelSourceFile_(cfg->cl()), buildOptions_(cfg->buildOptions()),
    matrixGen_(cfg->dataPattern()),
    masterStep_(masterStep), steps_(steps), indent_(0), useSeveralKernels_(true)
{
}

std::string
KTest::indent()
{
    std::string str = "";

    for (size_t i = 0; i < indent_; i++) {
        str += " ";
    }
    return str;
}

const char*
KTest::matrixGenName(KTestMatrixGenerator gen)
{
    switch (gen) {
    case RANDOM_MATRIX:
        return "randomMatrix";
    case UNIT_MATRIX:
        return "unitMatrix";
    case SAWTOOTH_MATRIX:
        return "sawtoothMatrix";
    default:
        return NULL;
    }
}

const char*
KTest::vectorGenName(KTestMatrixGenerator gen)
{
    switch (gen) {
    case RANDOM_MATRIX:
        return "randomVector";
    case UNIT_MATRIX:
        return "unitVector";
    case SAWTOOTH_MATRIX:
        return "sawtoothVector";
    default:
        return NULL;
    }
}

std::string
KTest::generate(bool withAccuracy)
{
    std::stringstream ss;
    int level;

    ss << indent() << "#define _CRT_SECURE_NO_WARNINGS" << std::endl;

    ss << std::endl
       << indent() << "#include <assert.h>" << std::endl
       << indent() << "#include <math.h>" << std::endl
       << indent() << "#include <stdio.h>" << std::endl
       << indent() << "#include <stdlib.h>" << std::endl
       << indent() << "#include <string.h>" << std::endl
       << indent() << "#include <time.h>" << std::endl
       << indent() << "#include <string>" << std::endl;
    if (masterStep_->blasFunctionID() == CLBLAS_TRSM) {
        ss << indent() << "#include <math.h>" << std::endl
           << indent() << "#define NANF NAN" << std::endl;
    }

    includes(ss);
    ss << std::endl
       << indent() << "#include \"naive_blas.cpp\"" << std::endl
       << std::endl
       << indent() << "using namespace NaiveBlas;" << std::endl;

    if (masterStep_->blasFunctionID() == CLBLAS_TRSM) {
        setUpTRSMDiagonal(ss);
    }

    level = funcBlasLevel(masterStep_->blasFunctionID());

    switch (matrixGen_) {
    case RANDOM_MATRIX:
        if (level == 2) {
        randomVector(ss);
        }
        randomMatrix(ss);
        break;
    case UNIT_MATRIX:
        if (level == 2) {
        unitVector(ss);
        }
        unitMatrix(ss);
        break;
    case SAWTOOTH_MATRIX:
        if (level == 2) {
            sawtoothVector(ss);
        }
        sawtoothMatrix(ss);
        break;
    default:
        break;
    }

    if (withAccuracy) {
        if (level == 2) {
            compareVectors(ss);
        }
        else {
            compareMatrices(ss);
        }
    }

    declareKTestOptions(ss);
    declareBlasOptions(ss, masterStep_);
    declarePatternVars(ss, masterStep_);

    ss << std::endl
       << indent() << "char* loadFile(const char* path);" << std::endl;
    forwardDeclarations(ss);

    generateMain(ss, withAccuracy);

    loadFile(ss);
    auxFunctions(ss);

    return ss.str();
}

void
KTest::declareKTestOptions(std::stringstream& ss)
{
    ss << std::endl;

    ss << indent() << "const char PLATFORM_NAME[] = \""
       << platform_ << "\";" << std::endl;
    ss << indent() << "const char DEVICE_NAME[] = \""
       << device_ << "\";" << std::endl;
    ss << indent() << "const char BUILD_OPTIONS[] = \""
       << buildOptions_ << "\";" << std::endl;
    ss << indent() << "const char KERNEL_SOURCE[] = \""
       << kernelSourceFile_ << "\";" << std::endl;
}

void
KTest::declareBlasOptions(std::stringstream& ss, Step *step)
{
    ss << std::endl;
    ss << indent() << "const clblasOrder order = "
       << ((step->kargs().order == clblasColumnMajor)
                ? "clblasColumnMajor"
                : "clblasRowMajor")
       << ";" << std::endl;
    ss << indent() << "const clblasSide side = "
       << ((step->kargs().side == clblasRight)
                ? "clblasRight"
                : "clblasLeft")
       << ";" << std::endl;
    ss << indent() << "const clblasUplo uplo = "
       << ((step->kargs().uplo == clblasUpper)
                ? "clblasUpper"
                : "clblasLower")
       << ";" << std::endl;
    ss << indent() << "const clblasTranspose transA = ";
    switch (step->kargs().transA) {
    case clblasNoTrans:
        ss << "clblasNoTrans";
        break;
    case clblasTrans:
        ss << "clblasTrans";
        break;
    case clblasConjTrans:
        ss << "clblasConjTrans";
        break;
    }
    ss << ";" << std::endl;
    ss << indent() << "const clblasTranspose transB = ";
    switch (step->kargs().transB) {
    case clblasNoTrans:
        ss << "clblasNoTrans";
        break;
    case clblasTrans:
        ss << "clblasTrans";
        break;
    case clblasConjTrans:
        ss << "clblasConjTrans";
        break;
    }
    ss << ";" << std::endl;
    ss << indent() << "const clblasDiag diag = "
       << ((step->kargs().diag == clblasUnit)
                ? "clblasUnit"
                : "clblasNonUnit")
       << ";" << std::endl;
}

void
KTest::declarePatternVars(std::stringstream& ss, Step *step)
{
    VarList vars = step->vars();
    ArrayVarList var_arays = step->arrays();

    vars.insert(vars.end(), var_arays.begin(), var_arays.end());

    ss << std::endl;
    for (VarList::const_iterator it = vars.begin(); it != vars.end(); ++it) {
        Variable *var = *it;
        if (step != masterStep_ && var->isBuffer()) {
            // master step buffers are used
            continue;
        }
        ss << indent();
        if (var->constant()) {
            ss << "const ";
        }
        ss << var->type() << " " << var->name();
        if (!var->defaultValue().empty()) {
            ss << " = " << var->defaultValue();
        }
        ss << ";" << std::endl;
    }
}

void
KTest::generateMain(std::stringstream& ss, bool withAccuracy)
{
    ArrayVarList list;
    std::map<unsigned int, const Variable*> kargMap = masterStep_->kargMap();
    std::string size;

    ss << std::endl;
    ss << indent() << "int" << std::endl;
    if (useSeveralKernels_) {
        ss << indent() << "main(int argc, char *argv[])" << std::endl;
    }
    else {
        ss << indent() << "main(void)" << std::endl;
    }

    ss << indent() << "{" << std::endl;
    indent_ += 4;

    ss << std::endl
       << indent() << "char *source;" << std::endl
       << indent() << "cl_ulong start, end;" << std::endl;

    ss << std::endl
       << indent() << "srand((unsigned int)time(NULL));" << std::endl;

    mainInit(ss);

    ss << std::endl;
    list = masterStep_->arrays();
    for (ArrayVarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        ss << indent() << (*it)->name() << " = (" <<(*it)->type() << ")calloc(";
        if ((*it)->isMatrix()) {
            ss << masterStep_->matrixSize((MatrixVariable*)(*it));
        }
        else {
            ss << masterStep_->vectorSize((VectorVariable*)(*it));
        }
        ss << ", "
           << "sizeof(*" << (*it)->name() << "));" << std::endl;
        ss << indent() << "assert(" << (*it)->name() << " != NULL);" << std::endl;
        if ((*it)->copyOf() != NULL) {
            continue;
        }
        if ((*it)->isMatrix()) {
            MatrixVariable *var = (MatrixVariable*)(*it);

            ss << indent() << matrixGenName(matrixGen_) << "(order, "
               << var->rows()->name() << ", "
               << var->columns()->name() << ", " << var->matrixPointer() << ", "
               << var->ld()->name() << ");" << std::endl;
        }
        else {
            VectorVariable *var = (VectorVariable*)(*it);
            ss << indent() << vectorGenName(matrixGen_) << "("
               << var->nElems()->name() << ", "
               << var->vectorPointer() << ", " << var->inc()->name()
               << ");" << std::endl;
        }
    }

    ss << indent() << masterStep_->postRandomCall() << ";" << std::endl;

    for (ArrayVarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        if ((*it)->copyOf() == NULL) {
            continue;
        }
        ss << indent() << "memcpy(" << (*it)->name() << ", "
           << (*it)->copyOf()->name() << ", (";
        if ((*it)->isMatrix()) {
            ss << masterStep_->matrixSize((MatrixVariable*)(*it));
        }
        else {
            ss << masterStep_->vectorSize((VectorVariable*)(*it));
        }
        ss << ") * sizeof(*" << (*it)->copyOf()->name() << "));" << std::endl;
    }

    if (withAccuracy) {
        ss << std::endl
           << indent() << "NaiveBlas::" << masterStep_->naiveCall() << ";"
           << std::endl;
    }

    allocateWriteBuffers(ss);

    if (useSeveralKernels_) {
        for (unsigned int i = 0; i < steps_->size(); i++) {
            Step *step = (*steps_)[i];
            ss << indent() << "{" << std::endl;
            indent_ += 4;

            declareGranulation(ss, step);

            ss << indent() << "const char* kernelName = argc > " << i + 1
                    << " ? argv[" << i + 1 << "] : \""
                    << step->kernelName() << "\";" << std::endl;

            ss << std::endl
                    << indent() << "source = loadFile(kernelName);" << std::endl
                    << indent() << "assert(source != NULL);" << std::endl;
            buildKernel(ss);

            declareBlasOptions(ss, step);
            declarePatternVars(ss, step);
            setKernelArgs(ss, step);


            ss << std::endl
                    << indent() << "start = 0;" << std::endl
                    << indent() << "end = 0;" << std::endl;
            execKernel(ss);
            ss << std::endl
                    << indent() << "printExecTime(end - start);" << std::endl;

            indent_ -= 4;
            ss << indent() << "}" << std::endl;
        }
    }
    else {
        declareGranulation(ss, masterStep_);
        ss << std::endl
                << indent() << "source = loadFile(KERNEL_SOURCE);" << std::endl
                << indent() << "assert(source != NULL);" << std::endl;
        buildKernel(ss);

        setKernelArgs(ss, masterStep_);

        ss << std::endl
                << indent() << "start = 0;" << std::endl
                << indent() << "end = 0;" << std::endl;
        execKernel(ss);
        ss << std::endl
                << indent() << "printExecTime(end - start);" << std::endl;
    }
    if (withAccuracy) {
        readBuffers(ss);

        ss << std::endl
           << indent() << "if (" << masterStep_->compareCall() << ") {"
                                 << std::endl << indent()
           << "    printf(\"Correctness test passed\\n\");" << std::endl
           << indent() << "}" << std::endl
           << indent() << "else {" << std::endl
           << indent() << "    printf(\"Correctness test failed\\n\");"
                       << std::endl
           << indent() << "}" << std::endl
           << indent() << "fflush(stdout);" << std::endl;
    }

    mainFinish(ss);

    ss << std::endl;
    list = masterStep_->arrays();
    for (ArrayVarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        ss << indent() << "free(" << (*it)->name() << ");" << std::endl;
    }
    ss << indent() << "free(source);" << std::endl
       << indent() << "exit(EXIT_SUCCESS);" << std::endl;

    indent_ -= 4;
    ss << indent() << "}" << std::endl;
}

void
KTest::loadFile(std::stringstream& ss)
{
    ss << loadFileCode << std::endl;
}

void
KTest::randomVector(std::stringstream& ss)
{
    ss << randomVectorCode << std::endl;
}

void
KTest::unitVector(std::stringstream& ss)
{
    ss << unitVectorCode << std::endl;
}

void
KTest::sawtoothVector(std::stringstream& ss)
{
    ss << sawtoothVectorCode << std::endl;
}

void
KTest::compareVectors(std::stringstream& ss)
{
    ss << compareVectorsCode << std::endl;
}

void
KTest::randomMatrix(std::stringstream& ss)
{
    ss << randomMatrixCode << std::endl;
}

void
KTest::unitMatrix(std::stringstream& ss)
{
    ss << unitMatrixCode << std::endl;
}

void
KTest::sawtoothMatrix(std::stringstream& ss)
{
    ss << sawtoothMatrixCode << std::endl;
}

void
KTest::setUpTRSMDiagonal(std::stringstream& ss)
{
    ss << setUpTRSMDiagonalCode << std::endl;
}

void
KTest::compareMatrices(std::stringstream& ss)
{
    ss << compareMatricesCode << std::endl;
}



void
KTest::includes(std::stringstream& ss)
{
    ss << std::endl
       << indent() << "#include <CL/cl.h>" << std::endl;
}

void
KTest::forwardDeclarations(std::stringstream& ss)
{
    ss << forwardDeclarationsCode << std::endl;
}

void
KTest::auxFunctions(std::stringstream& ss)
{
    getPlatform(ss);
    getDevice(ss);
    createKernel(ss);
    printExecTime(ss);
}

void
KTest::getPlatform(std::stringstream& ss)
{
    ss << getPlatformCode << std::endl;
}

void
KTest::getDevice(std::stringstream& ss)
{
    ss << getDeviceCode << std::endl;
}

void
KTest::createKernel(std::stringstream& ss)
{
    ss << createKernelCode << std::endl;
}

void
KTest::printExecTime(std::stringstream& ss)
{
    ss << printTimeCode;
}

void
KTest::declareGranulation(std::stringstream& ss, Step *step)
{
    ss << std::endl;
    ss << indent() << "const cl_uint workDim = "
       << step->pgran().wgDim << ";" << std::endl;
    ss << indent() << "const size_t localWorkSize["
       << step->pgran().wgDim
       << "] = { ";
    for (unsigned int i = 0; i < step->pgran().wgDim; i++) {
        if (i != 0) {
            ss << ", ";
        }
        ss << step->pgran().wgSize[i];
    }
    ss << " };" << std::endl;
    ss << indent() << "const size_t globalWorkSize["
       << step->pgran().wgDim
       << "] = { " << step->globalWorkSize() << " };" << std::endl;
}

void
KTest::mainInit(std::stringstream& ss)
{
    ss << std::endl
       << indent() << "cl_int err;" << std::endl
       << indent() << "cl_platform_id platform;" << std::endl
       << indent() << "cl_device_id device;" << std::endl
       << indent() << "cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };" << std::endl
       << indent() << "cl_context context;" << std::endl
       << indent() << "cl_command_queue queue;" << std::endl
       << indent() << "cl_kernel kernel;" << std::endl
       << indent() << "cl_event event;" << std::endl;

    ss << std::endl
       << indent() << "platform = getPlatform(PLATFORM_NAME);" << std::endl
       << indent() << "assert(platform != NULL);" << std::endl
       << indent() << "device = getDevice(platform, DEVICE_NAME);" << std::endl
       << indent() << "assert(device != NULL);" << std::endl
       << indent() << "props[1] = (cl_context_properties)platform;" << std::endl
       << indent() << "context = clCreateContext(props, 1, &device, NULL, NULL, &err);" << std::endl
       << indent() << "assert(context != NULL);" << std::endl
       << indent() << "queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);" << std::endl
       << indent() << "assert(queue != NULL);" << std::endl;
}

void
KTest::buildKernel(std::stringstream& ss)
{
    ss << indent() << "kernel = createKernel(source, context, BUILD_OPTIONS, &err);" << std::endl
       << indent() << "assert(kernel != NULL);" << std::endl;
}

void
KTest::getBufferSizeExpr(Variable *buffer, std::string& size)
{
    ArrayVariableInterface *hostPtr = (ArrayVariableInterface*)(buffer->hostPtr());
    if (hostPtr->isMatrix()) {
        MatrixVariable *ptrVar = (MatrixVariable*)hostPtr;
        if (masterStep_->matrixSize(ptrVar).empty()) {
            size += "sizeof(";
            size += ptrVar->type();
            size += ")";
        }
        else {
            size = "(";
            size += masterStep_->matrixSize(ptrVar);
            size += ") * sizeof(*";
            size += ptrVar->name();
            size += ")";
        }
    }
    else {
        VectorVariable *ptrVar = (VectorVariable*)buffer->hostPtr();
        size = "(";
        size += masterStep_->vectorSize(ptrVar);
        size += ") * sizeof(*";
        size += ptrVar->name();
        size += ")";
    }
}

void
KTest::allocateWriteBuffers(std::stringstream& ss)
{
    VarList list;
    std::string size;

    ss << std::endl;
    list = masterStep_->buffers();
    for (VarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        getBufferSizeExpr(*it, size);
        ss << indent() << (*it)->name() << " = clCreateBuffer(context, "
           << (*it)->flagsStr() << "," << std::endl
           << indent() << "    " << size << ", NULL, &err);" << std::endl;
        ss << indent() << "assert(" << (*it)->name() << " != NULL);" << std::endl;
        if (((*it)->flags() & CL_MEM_READ_WRITE) ||
                            ((*it)->flags() & CL_MEM_READ_ONLY)) {
            ss << indent() << "err = clEnqueueWriteBuffer(queue, "
               << (*it)->name() << ", CL_TRUE, 0," << std::endl
               << indent() << "    " << size << ", "
               << ((Variable*)(*it)->hostPtr())->name() << "," << std::endl
               << indent() << "    0, NULL, NULL);" << std::endl;
            ss << indent() << "assert(err == CL_SUCCESS);" << std::endl;
        }
    }
}

void
KTest::setKernelArgs(std::stringstream& ss, Step *step)
{
    std::map<unsigned int, const Variable*> kargMap = step->kargMap();
    ss << std::endl;
    for (KArgMap::iterator it = kargMap.begin(); it != kargMap.end(); ++it) {
        ss << indent() << "err = clSetKernelArg(kernel, "
                << (*it).first << ", sizeof(" << (*it).second->type() << "), "
                << "&" << (*it).second->name() << ");" << std::endl;
        ss << indent() << "assert(err == CL_SUCCESS);" << std::endl;
    }
}

void
KTest::execKernel(std::stringstream& ss)
{
    ss << std::endl
       << indent() << "event = NULL;" << std::endl
       << indent() << "err = clEnqueueNDRangeKernel(queue, kernel, workDim, NULL," << std::endl
       << indent() << "    globalWorkSize, localWorkSize, 0, NULL, &event);" << std::endl
       << indent() << "assert(err == CL_SUCCESS);" << std::endl
       << indent() << "err = clFinish(queue);" << std::endl
       << indent() << "assert(err == CL_SUCCESS);" << std::endl;

    ss << std::endl
       << indent() << "err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START," << std::endl
       << indent() << "    sizeof(start), &start, NULL);" << std::endl
       << indent() << "err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END," << std::endl
       << indent() << "    sizeof(end), &end, NULL);" << std::endl;
}

void
KTest::readBuffers(std::stringstream& ss)
{
    VarList list;
    std::string size;

    ss << std::endl;
    list = masterStep_->buffers();
    for (VarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        if (((*it)->flags() & CL_MEM_READ_WRITE) ||
                            ((*it)->flags() & CL_MEM_WRITE_ONLY)) {
            getBufferSizeExpr(*it, size);
            ss << indent() << "err = clEnqueueReadBuffer(queue, "
               << (*it)->name() << ", CL_TRUE, 0," << std::endl
               << indent() << "    " << size << ", "
               << ((Variable*)(*it)->hostPtr())->name() << "," << std::endl
               << indent() << "    0, NULL, NULL);" << std::endl;
            ss << indent() << "assert(err == CL_SUCCESS);" << std::endl;
        }
    }
}

void
KTest::mainFinish(std::stringstream& ss)
{
    VarList list;

    ss << std::endl;
    list = masterStep_->buffers();
    for (VarList::const_iterator it = list.begin(); it != list.end(); ++it) {
        ss << indent() << "err = clReleaseMemObject("
           << (*it)->name() << ");" << std::endl;
        ss << indent() << "assert(err == CL_SUCCESS);" << std::endl;
    }
    ss << indent() << "err = clReleaseKernel(kernel);" << std::endl
       << indent() << "assert(err == CL_SUCCESS);" << std::endl
       << indent() << "err = clReleaseCommandQueue(queue);" << std::endl
       << indent() << "assert(err == CL_SUCCESS);" << std::endl
       << indent() << "err = clReleaseContext(context);" << std::endl
       << indent() << "assert(err == CL_SUCCESS);" << std::endl;
}
