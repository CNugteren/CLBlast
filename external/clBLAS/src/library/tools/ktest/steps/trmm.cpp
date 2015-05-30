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


#include <sstream>

#include "trmm.h"

using namespace clMath;

TrmmStep::TrmmStep(cl_device_id device) :
    Step(CLBLAS_TRMM, device)
{
}

TrmmStep::TrmmStep(ListNode *node) :
    Step(node)
{
}

void
TrmmStep::declareVars(Step *masterStep)
{
    StepKargs args;
    MatrixVariable *A, *B, *naiveB;

    memset(&args, 0, sizeof(args));
    std::string type = dtypeToString(kargs().dtype);

    args.M = addConst("M", "cl_uint", kargs().M);
    args.N = addConst("N", "cl_uint", kargs().N);
    if (kargs().side == clblasLeft) {
        args.K = args.M;
    }
    else {
        args.K = args.N;
    }

    args.lda = addConst("lda", "cl_uint", kargs().lda.matrix);
    args.ldb = addConst("ldb", "cl_uint", kargs().ldb.matrix);

    args.offA = addVar("offA", "cl_uint", kargs().offA);
    args.offBX = addVar("offB", "cl_uint", kargs().offBX);

    args.alpha = addVar("alpha", type,
        multiplierToString(kargs().dtype, kargs().alpha));

    if (kargs().side == clblasLeft) {
        A = addMatrix("A", type + "*", args.M, args.M, args.lda, args.offA);
    }
    else {
        A = addMatrix("A", type + "*", args.N, args.N, args.lda, args.offA);
    }
    B = addMatrix("B", type + "*", args.M, args.N, args.ldb, args.offBX);
    naiveB = addMatrix("naiveB", type + "*", args.M, args.N, args.ldb, args.offBX);
    naiveB->setCopy(B);

    std::string bufAName, bufBName;
    if (NULL == masterStep) {
        bufAName = "bufA";
        bufBName = "bufB";
    }
    else {
        bufAName = masterStep->getBuffer((BufferID)(long)step_.args.A)->name();
        bufBName = masterStep->getBuffer((BufferID)(long)step_.args.B)->name();
    }

    args.A = addBuffer(BUFFER_A, bufAName, "cl_mem", CL_MEM_READ_ONLY, A);
    args.B = addBuffer(BUFFER_B, bufBName, "cl_mem", CL_MEM_READ_WRITE, B);

    assignKargs(args);

    std::stringstream ss;
    ss << getBlasFunctionName() << "(order, side, uplo, transA, diag, "
       << args.M->name() << ", " << args.N->name() << ", "
       << args.alpha->name() << ", " << A->matrixPointer() << ", "
       << args.lda->name() << ", " << naiveB->matrixPointer() << ", "
       << args.ldb->name() << ")";
    naiveCall_ = ss.str();

    ss.str("");
    ss << "compareMatrices(order, " << args.M->name() << ", " << args.N->name()
       << ", " << B->matrixPointer() << ", " << naiveB->matrixPointer()
       << ", " << args.ldb->name() << ")";
    compareCall_ = ss.str();
}

void
TrmmStep::fixLD()
{
    CLBlasKargs args;

    args = kargs();

    if (args.side == clblasLeft) {
        if (args.lda.matrix < args.M) {
            args.lda.matrix = args.M;
        }
    }
    else {
        if (args.lda.matrix < args.N) {
            args.lda.matrix = args.N;
        }
    }
    if ((args.order == clblasColumnMajor) && (args.ldb.matrix < args.M)) {
        args.ldb.matrix = args.M;
    }
    if ((args.order == clblasRowMajor) && (args.ldb.matrix < args.N)) {
        args.ldb.matrix = args.N;
    }

    // Store original problem size in K, this is used to know it while
    // calculating result by parts using M or N as part size
    if (args.side == clblasLeft) {
        args.K = args.M;
    }
    else {
        args.K = args.N;
    }
    setKargs(args);
}
