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


#include <boost/lexical_cast.hpp>

#include <assert.h>
#include <kerngen.h>
#include <clblas-internal.h>
#include <matrix_dims.h>
#include <solution_seq.h>
#include "step.h"

using namespace clMath;

// This enum reflects CLBlasKargs structure, declared in clblas-internal.h
typedef enum StepKarg {
    KARG_NONE = 0,
    // kernType
    // dtype
    KARG_ORDER,
    KARG_SIDE,
    KARG_UPLO,
    KARG_TRANS_A,
    KARG_TRANS_B,
    KARG_DIAG,
    KARG_M,
    KARG_N,
    KARG_K,
    KARG_ALPHA,
    KARG_A,
    KARG_LDA,
    KARG_B,
    KARG_LDB,
    KARG_BETA,
    KARG_C,
    KARG_LDC,
    // addrBits
    KARG_OFFSET_M,
    KARG_OFFSET_N,
    KARG_OFFSET_K,
    KARG_SCIMAGE_0,
    KARG_SCIMAGE_1,
    KARG_OFF_A,
    KARG_OFF_BX,
    KARG_OFF_CY
} StepKarg;

Step::Step(
    BlasFunctionID funcID,
    cl_device_id device) :
        naiveCall_(""), compareCall_(""), postRandomCall_(""), kernelName_("")
{
    memset(&step_, 0, sizeof(step_));
    memset(&kextra_, 0, sizeof(kextra_));

    step_.funcID = funcID;
    step_.device.id = device;
    identifyDevice(&step_.device);
    step_.args.A = (cl_mem)BUFFER_A;
    step_.args.B = (cl_mem)BUFFER_B;
    step_.args.C = (cl_mem)BUFFER_C;
    if (blasFunctionID() == CLBLAS_SYR2K) {
        kextra_.flags = static_cast<KernelExtraFlags>
                            (kextra_.flags | KEXTRA_SYRK_2K_RANK);
        step_.extraFlags = kextra_.flags;
    }
}

Step::Step(ListNode *node) :
    naiveCall_(""), compareCall_(""), postRandomCall_(""), kernelName_("")
{
    SolutionStep *stepNode;
    memset(&kextra_, 0, sizeof(kextra_));

    stepNode = container_of(node, node, SolutionStep);
    memcpy(&step_, stepNode, sizeof(step_));

    kextra_.dtype = step_.args.dtype;
    kextra_.flags = step_.extraFlags;
    kextra_.kernType = CLBLAS_COMPUTING_KERNEL;
}

Step::~Step()
{
    for (ArrayVarList::iterator it = arrays_.begin(); it != arrays_.end(); ++it) {
        delete (*it);
    }
    for (VarList::iterator it = vars_.begin(); it != vars_.end(); ++it) {
        delete (*it);
    }

    vars_.clear();

    arrays_.clear();

    buffers_.clear();

    kargMap_.clear();
}

BlasFunctionID
Step::getStepNodeFuncID(ListNode *node)
{
    SolutionStep *stepNode;
    stepNode = container_of(node, node, SolutionStep);
    return stepNode->funcID;
}

void
Step::completeDecompositionSingle()
{
    cl_int err;

    kextra_.dtype = kargs().dtype;
    kextra_.kernType = CLBLAS_COMPUTING_KERNEL;
    kextra_.flags = (KernelExtraFlags)(kextra_.flags |
            clblasArgsToKextraFlags(&step_.args, blasFunctionID()));
    if (deviceVendor(device()) == "Advanced Micro Devices, Inc.") {
        kextra_.flags = static_cast<KernelExtraFlags>
            (kextra_.flags | KEXTRA_VENDOR_AMD | KEXTRA_ENABLE_MAD);
    }

    step_.pgran.wfSize = deviceWavefront(device(), &err);

    step_.extraFlags = kextra_.flags;
    step_.patternID = selectPattern(&step_, 0);
    pattern_ = &clblasSolvers[step_.funcID].memPatterns[step_.patternID];

    if (0 == step_.subdims[0].bwidth
            && 0 == step_.subdims[0].bwidth
            && 0 == step_.subdims[0].bwidth) {
        getStepGranulation(&step_);

    }
    else if (pattern_->sops->checkCalcDecomp) {
        pattern_->sops->checkCalcDecomp(&step_.pgran, step_.subdims, 2,
                                        kextra_.dtype, PGRAN_CALC);
    }
    else {
        size_t wgX, wgY;
        size_t x0, y0;
        SolverFlags sflags;

        // Set up granulation for given dimensions

        wgY = step_.subdims[0].y/ step_.subdims[1].y;
        wgX = step_.subdims[0].x/ step_.subdims[1].x;

        x0 = step_.subdims[0].x;
        y0 = step_.subdims[0].y;

        if (funcBlasLevel(blasFunctionID()) == 2) {
            /* Level 2 decomposition size for vectors (dims[0].x) is 1.
             * We have to "restore" it to proceed.
             */
            size_t xBlocks;

            xBlocks = step_.subdims[0].bwidth / step_.subdims[1].bwidth;
            x0 = step_.subdims[1].x * xBlocks;
        }

        /*
         * adjust local size if a subproblem is not divisible
         * between all local threads
         */
        for (; (wgY > 1) && (y0 < wgY); wgY /= 2) { }
        for (; (wgX > 1) && (x0 < wgX); wgX /= 2) { }

        sflags = pattern_->sops->getFlags();
        if (sflags & SF_WSPACE_2D) {
            step_.pgran.wgDim = 2;
            step_.pgran.wgSize[0] = (unsigned int)wgY;
            step_.pgran.wgSize[1] = (unsigned int)wgX;
        }
        else {
            step_.pgran.wgDim = 1;
            step_.pgran.wgSize[0] = (unsigned int)(wgX * wgY);
            step_.pgran.wgSize[1] = 1;
        }

        // fixup work group size in respect with desired work dispatch order
        if ((step_.pgran.wgDim == 2) && pattern_->sops->innerDecompositionAxis) {
            if (pattern_->sops->innerDecompositionAxis(&step_.args) == DECOMP_AXIS_X) {
                unsigned int u;

                u = step_.pgran.wgSize[0];
                step_.pgran.wgSize[0] = step_.pgran.wgSize[1];
                step_.pgran.wgSize[1] = u;
            }
        }

        /* Check that dimensions are bigger than whole problem size */
        if (dimensionsExceedProblemSize(&step_)) {
            getMinimalStepGranulation(&step_);
        }
    }
    detectProblemTails(&step_);
    kextra_.flags = step_.extraFlags;
    if (pattern_->sops->fixupArgs) {
        pattern_->sops->fixupArgs(&step_.args, &step_.subdims[0], &kextra_);
}
    step_.extraFlags = kextra_.flags;
    detectOffsets(&step_);
    kextra_.flags = step_.extraFlags;
    selectVectorization(&step_, &kextra_);
}

void
Step::makeSolutionSequence(ListHead *seq, cl_platform_id platform)
{
    SolutionStep *newStep;

    (void)platform;

    step_.args.A = (cl_mem)BUFFER_A;
    step_.args.B = (cl_mem)BUFFER_B;
    step_.args.C = (cl_mem)BUFFER_C;

    newStep = (SolutionStep*)malloc(sizeof(SolutionStep));
    memcpy(newStep, &step_, sizeof(SolutionStep));
    listAddToTail(seq, &newStep->node);
    decomposeProblemStep(newStep);
}

void
Step::freeSolutionSequence(ListHead *seq)
{
    freeSolutionSeq(seq);
}

std::string
Step::generate()
{
    ssize_t size;
    char *buf;
    std::stringstream ss;

    if ((pattern_->sops == NULL) || (pattern_->sops->genKernel == NULL)) {
        return "";
    }

    ss << "/*" << std::endl;
    for (int i = 0; i < MAX_SUBDIMS; i++) {
        ss << "SubproblemDim[" << i << "]" << std::endl;
        ss << dumpSubdim(step_.subdims + i) << std::endl;
    }
    ss << "PGranularity" << std::endl;
    ss << dumpPgran() << std::endl;
    ss << "CLBLASKernExtra" << std::endl;
    ss << dumpKextra() << std::endl;
    ss << "MemoryPattern" << std::endl;
    ss << dumpMemoryPattern();
    ss << "*/" << std::endl << std::endl;

    size = pattern_->sops->genKernel(NULL, 0, step_.subdims, &step_.pgran,
        static_cast<void*>(&kextra_));
    if (size <= 0) {
        return 0;
    }
    buf = new char[size + 1];
    if (pattern_->sops->genKernel(buf, size, step_.subdims, &step_.pgran,
                static_cast<void*>(&kextra_)) != size) {
        delete[] buf;
        return "";
    }
    ss << buf;

    delete[] buf;
    return ss.str();
}

void
Step::setKargs(const CLBlasKargs& kargs)
{
    step_.args = kargs;
}

const char*
Step::getBlasFunctionName()
{
    switch (blasFunctionID()) {
    case CLBLAS_GEMV:
        return "gemv";
    case CLBLAS_SYMV:
        return "symv";
    case CLBLAS_GEMM:
        return "gemm";
    case CLBLAS_TRMM:
        return "trmm";
    case CLBLAS_TRSM:
        return "trsm";
    case CLBLAS_SYRK:
        return "syrk";
    case CLBLAS_SYR2K:
        return "syr2k";
    default:
        return "";
    }
}

void
Step::setDecomposition(
    const SubproblemDim *subdims)
{
    for (size_t i = 0; i < MAX_SUBDIMS; i++) {
        step_.subdims[i] = subdims[i];
    }
}

Variable*
Step::addVar(
    const std::string& name,
    const std::string& type,
    const std::string& defaultValue)
{
    Variable *var = new Variable(name, type, defaultValue);
    vars_.push_back(var);
    return var;
}

Variable*
Step::addConst(
    const std::string& name,
    const std::string& type,
    const std::string& defaultValue)
{
    Variable *var = addVar(name, type, defaultValue);
    var->setConstant(true);
    return var;
}

Variable*
Step::addVar(
    const std::string& name,
    const std::string& type,
    size_t value)
{
    return addVar(name, type, boost::lexical_cast<std::string>(value));
}

Variable*
Step::addConst(
    const std::string& name,
    const std::string& type,
    size_t value)
{
    return addConst(name, type, boost::lexical_cast<std::string>(value));
}

Variable*
Step::addVar(
    const std::string& name,
    const std::string& type,
    int value)
{
    return addVar(name, type, boost::lexical_cast<std::string>(value));
}

Variable*
Step::addConst(
    const std::string& name,
    const std::string& type,
    int value)
{
    return addConst(name, type, boost::lexical_cast<std::string>(value));
}

MatrixVariable*
Step::addMatrix(
    const std::string& name,
    const std::string& type,
    Variable *rows,
    Variable *columns,
    Variable *ld,
    Variable *off)
{
    MatrixVariable *var = new MatrixVariable(name, type, "NULL");
    var->setMatrixSize(rows, columns, ld, off);
    arrays_.push_back(var);
    return var;
}

VectorVariable*
Step::addVector(
    const std::string& name,
    const std::string& type,
    Variable *N,
    Variable *inc,
    Variable *off)
{
    VectorVariable *var = new VectorVariable(name, type, "NULL");
    var->setVectorSize(N, inc, off);
    arrays_.push_back(var);
    return var;
}

Variable*
Step::addBuffer(
    BufferID bufID,
    const std::string& name,
    const std::string& type,
    cl_mem_flags flags,
    ArrayVariableInterface* hostPtr)
{
    Variable *var = addVar(name, type, "NULL");
    var->setIsBuffer(true);
    var->setFlags(flags);
    var->setHostPtr(hostPtr);
    var->setBufferID(bufID);
    buffers_.push_back(var);
    return var;
}

Variable*
Step::getBuffer(BufferID bufID)
{
    for (VarList::iterator it = buffers_.begin(); it != buffers_.end(); ++it) {
        if ((*it)->getBufID() == bufID) {
            return (*it);
        }
    }
    return NULL;
}


void
Step::setKernelArg(
    unsigned int index,
    const Variable *var)
{
    kargMap_[index] = var;
}

std::string
Step::matrixSize(MatrixVariable *matrix)
{
    std::stringstream size;

    if ((matrix->rows() == NULL) || (matrix->columns() == NULL)) {
        return "";
    }

    if (matrix->off() != NULL) {
        size << matrix->off()->name() << " + ";
    }

    if (matrix->ld() != NULL) {
        size << matrix->ld()->name() << " * ";
    }

    if (step_.args.order == clblasColumnMajor) {
        size << matrix->columns()->name();
    }
    else {
        size << matrix->rows()->name();
    }
    return size.str();
}

std::string
Step::vectorSize(VectorVariable *vector)
{
    std::stringstream size;

    if (vector->nElems() == NULL) {
        return "";
    }

    if (vector->off() != NULL) {
        size << vector->off()->name() << " + ";
    }
    if (vector->inc() == NULL) {
        size << vector->nElems()->name();
    }
    else {
        size << "1 + (" << vector->nElems()->name() << " - 1) * abs("
             << vector->inc()->name() << ")";
    }
    return size.str();
}

void
Step::assignKargs(const StepKargs& map)
{
    CLBlasKargs args;
    KernelArg kargsList[MAX_KERNEL_ARGS];
    Variable *v;

    if ((pattern_->sops == NULL) || (pattern_->sops->assignKargs == NULL)) {
        return;
    }

    memset(&kargsList, KARG_NONE, sizeof(kargsList));

    args.kernType = CLBLAS_COMPUTING_KERNEL;
    args.dtype = TYPE_COMPLEX_DOUBLE;
    args.addrBits = 0;

    args.order = static_cast<clblasOrder>(KARG_ORDER);
    args.side = static_cast<clblasSide>(KARG_SIDE);
    args.uplo = static_cast<clblasUplo>(KARG_UPLO);
    args.transA = static_cast<clblasTranspose>(KARG_TRANS_A);
    args.transB = static_cast<clblasTranspose>(KARG_TRANS_B);
    args.diag = static_cast<clblasDiag>(KARG_DIAG);

    args.M = KARG_M;
    args.N = KARG_N;
    args.K = KARG_K;

    args.lda.matrix = KARG_LDA;
    args.ldb.matrix = KARG_LDB;
    args.ldc.matrix = KARG_LDC;

    args.offsetM = KARG_OFFSET_M;
    args.offsetN = KARG_OFFSET_N;
    args.offsetK = KARG_OFFSET_K;
    args.offA = KARG_OFF_A;
    args.offBX = KARG_OFF_BX;
    args.offCY = KARG_OFF_CY;

    args.A = reinterpret_cast<cl_mem>(KARG_A);
    args.B = reinterpret_cast<cl_mem>(KARG_B);
    args.C = reinterpret_cast<cl_mem>(KARG_C);

    memset(&args.alpha, KARG_ALPHA, sizeof(args.alpha));
    memset(&args.beta, KARG_BETA, sizeof(args.beta));

    args.scimage[0] = reinterpret_cast<cl_mem>(KARG_SCIMAGE_0);
    args.scimage[1] = reinterpret_cast<cl_mem>(KARG_SCIMAGE_1);

    pattern_->sops->assignKargs(kargsList, static_cast<void*>(&args), &kextra_);

    for (unsigned int i = 0; (i < MAX_KERNEL_ARGS) && (kargsList[i].typeSize != 0); i++) {
        switch (static_cast<StepKarg>(kargsList[i].arg.data[0])) {
        case KARG_M:
            v = map.M;
            break;
        case KARG_N:
            v = map.N;
            break;
        case KARG_K:
            v = map.K;
            break;
        case KARG_ALPHA:
            v = map.alpha;
            break;
        case KARG_A:
            v = map.A;
            break;
        case KARG_LDA:
            v = map.lda;
            break;
        case KARG_B:
            v = map.B;
            break;
        case KARG_LDB:
            v = map.ldb;
            break;
        case KARG_BETA:
            v = map.beta;
            break;
        case KARG_C:
            v = map.C;
            break;
        case KARG_LDC:
            v = map.ldc;
            break;
        case KARG_OFFSET_M:
            v = map.offsetM;
            break;
        case KARG_OFFSET_N:
            v = map.offsetN;
            break;
        case KARG_OFFSET_K:
            v = map.offsetK;
            break;
        case KARG_SCIMAGE_0:
            v = map.scimage0;
            break;
        case KARG_SCIMAGE_1:
            v = map.scimage1;
            break;
        case KARG_OFF_A:
            v = map.offA;
            break;
        case KARG_OFF_BX:
            v = map.offBX;
            break;
        case KARG_OFF_CY:
            v = map.offCY;
            break;
        default:
            // KARG_ORDER, KARG_SIDE, KARG_UPLO, KARG_TRANS_A, KARG_TRANS_B,
            // KARG_DIAG
            v = NULL;
            break;
        }
        if (v != NULL) {
            setKernelArg(i, v);
        }
    }
}

std::string
Step::globalWorkSize()
{
    size_t globalWorkSize[MAX_WORK_DIM] = { 0, 0, 0 };
    std::stringstream ss;
    SubproblemDim dims[MAX_SUBDIMS];

    memcpy(dims, step_.subdims, sizeof(dims));

    if (pattern_->sops->calcThreads) {
        pattern_->sops->calcThreads(globalWorkSize, step_.subdims,
                                    &step_.pgran, &step_.args, &kextra_);
    }
    else {
        SubproblemDim globDim;
        const PGranularity *pg;

        pg = (pattern_->nrLevels == 1) ? NULL : &step_.pgran;
        kargsToProbDims(&globDim, blasFunctionID(), &step_.args, false);

        // fixup dimensions in respect with desired work dispatch order
        if ((pgran().wgDim == 2) && pattern_->sops->innerDecompositionAxis) {
            if (pattern_->sops->innerDecompositionAxis(&step_.args) ==
                DECOMP_AXIS_X) {

                /*
                 * these dimensions will not be used more anywhere, so we can
                 * just swap them
                 */
                swapDimXY(&dims[0]);
                swapDimXY(&dims[1]);
                swapDimXY(&globDim);
            }
        }

        calcGlobalThreads(globalWorkSize, dims, pg, globDim.y, globDim.x);
    }

    for (unsigned int i = 0; i < pgran().wgDim; i++) {
        if (i != 0) {
            ss << ", ";
        }
        ss << globalWorkSize[i];
    }

    return ss.str();
}

void
Step::setKernelName(std::string name)
{
    kernelName_ = name;
}

std::string
Step::deviceVendor(cl_device_id device)
{
    cl_int err;
    size_t len;
    char *str;
    std::string vendor = "";

    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &len);
    if (err != CL_SUCCESS) {
        return "";
    }
    str = new char[len + 1];
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, len, str, NULL);
    if (err == CL_SUCCESS) {
        vendor = str;
    }
    delete[] str;
    return vendor;
}
