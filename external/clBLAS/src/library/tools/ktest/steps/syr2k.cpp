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

#include "syr2k.h"

using namespace clMath;

Syr2kStep::Syr2kStep(cl_device_id device) :
    Step(CLBLAS_SYR2K, device)
{
}

Syr2kStep::Syr2kStep(ListNode *node) :
    Step(node)
{
}

void
Syr2kStep::declareVars(Step *masterStep)
{
    StepKargs args;
    MatrixVariable *A, *B, *C, *naiveC;

    memset(&args, 0, sizeof(args));
    std::string type = dtypeToString(kargs().dtype);

    args.N = addConst("N", "cl_uint", kargs().N);
    args.M = args.N;
    args.K = addConst("K", "cl_uint", kargs().K);

    args.lda = addConst("lda", "cl_uint", kargs().lda.matrix);
    args.ldb = addConst("ldb", "cl_uint", kargs().ldb.matrix);
    args.ldc = addConst("ldc", "cl_uint", kargs().ldc.matrix);

    args.offsetM = addConst("offsetM", "cl_uint", kargs().offsetM);

    args.offA = addVar("offA", "cl_uint", kargs().offA);
    args.offBX = addVar("offB", "cl_uint", kargs().offBX);
    args.offCY = addVar("offC", "cl_uint", kargs().offCY);

    args.alpha = addVar("alpha", type,
        multiplierToString(kargs().dtype, kargs().alpha));
    args.beta = addVar("beta", type,
        multiplierToString(kargs().dtype, kargs().beta));

    if (kargs().transA == clblasNoTrans) {
        A = addMatrix("A", type + "*", args.N, args.K, args.lda, args.offA);
        B = addMatrix("B", type + "*", args.N, args.K, args.lda, args.offBX);
    }
    else {
        A = addMatrix("A", type + "*", args.K, args.N, args.lda, args.offA);
        B = addMatrix("B", type + "*", args.K, args.N, args.lda, args.offBX);
    }
    C = addMatrix("C", type + "*", args.N, args.N, args.ldc, args.offCY);
    naiveC = addMatrix("naiveC", type + "*", args.N, args.N, args.ldc, args.offCY);
    naiveC->setCopy(C);

    std::string bufAName, bufBName, bufCName;
    if (NULL == masterStep) {
        bufAName = "bufA";
        bufBName = "bufB";
        bufCName = "bufC";
    }
    else {
        bufAName = masterStep->getBuffer((BufferID)(long)step_.args.A)->name();
        bufBName = masterStep->getBuffer((BufferID)(long)step_.args.B)->name();
        bufCName = masterStep->getBuffer((BufferID)(long)step_.args.C)->name();
    }

    args.A = addBuffer(BUFFER_A, bufAName, "cl_mem", CL_MEM_READ_ONLY, A);
    args.B = addBuffer(BUFFER_B, bufBName, "cl_mem", CL_MEM_READ_ONLY, B);
    args.C = addBuffer(BUFFER_C, bufCName, "cl_mem", CL_MEM_READ_WRITE, C);

    assignKargs(args);

    std::stringstream ss;
    ss << getBlasFunctionName() << "(order, uplo, transA, "
       << args.N->name() << ", " << args.K->name() << ", "
       << args.alpha->name() << ", " << A->matrixPointer() << ", "
       << args.lda->name() << ", " << B->matrixPointer() << ", "
       << args.ldb->name() << ", " << args.beta->name() << ", "
       << naiveC->matrixPointer() << ", " << args.ldc->name() << ")";
    naiveCall_ = ss.str();

    ss.str("");
    ss << "compareMatrices(order, " << args.N->name() << ", " << args.N->name()
       << ", " << C->matrixPointer() << ", " << naiveC->matrixPointer()
       << ", " << args.ldc->name() << ")";
    compareCall_ = ss.str();
}

void
Syr2kStep::fixLD()
{
    CLBlasKargs args;

    args = kargs();


    if (args.transA == clblasNoTrans) {
        if ((args.order == clblasColumnMajor) && (args.lda.matrix < args.N)) {
            args.lda.matrix = args.N;
        }
        if ((args.order == clblasRowMajor) && (args.lda.matrix < args.K)) {
            args.lda.matrix = args.K;
        }
        if ((args.order == clblasColumnMajor) && (args.ldb.matrix < args.N)) {
            args.ldb.matrix = args.N;
        }
        if ((args.order == clblasRowMajor) && (args.ldb.matrix < args.K)) {
            args.ldb.matrix = args.K;
        }
    }
    else {
        if ((args.order == clblasColumnMajor) && (args.lda.matrix < args.K)) {
            args.lda.matrix = args.K;
        }
        if ((args.order == clblasRowMajor) && (args.lda.matrix < args.N)) {
            args.lda.matrix = args.N;
        }
        if ((args.order == clblasColumnMajor) && (args.ldb.matrix < args.K)) {
            args.ldb.matrix = args.K;
        }
        if ((args.order == clblasRowMajor) && (args.ldb.matrix < args.N)) {
            args.ldb.matrix = args.N;
        }
    }
    if (args.ldc.matrix < args.N) {
        args.ldc.matrix = args.N;
    }

    args.transB = args.transA;
    args.M = args.N;

    setKargs(args);
}

