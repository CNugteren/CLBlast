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

#include "gemv.h"

using namespace clMath;

GemvStep::GemvStep(cl_device_id device) :
    Step(CLBLAS_GEMV, device)
{
}

GemvStep::GemvStep(ListNode *node) :
    Step(node)
{
}

void
GemvStep::declareVars(Step *masterStep)
{
    StepKargs args;
    MatrixVariable *A;
    VectorVariable *X, *Y, *naiveY;

    memset(&args, 0, sizeof(args));
    std::string type = dtypeToString(kargs().dtype);

    args.M = addConst("M", "cl_uint", kargs().M);
    args.N = addConst("N", "cl_uint", kargs().N);

    args.lda = addConst("lda", "cl_uint", kargs().lda.matrix);
    args.ldb = addConst("incx", "cl_int", kargs().ldb.vector);
    args.ldc = addConst("incy", "cl_int", kargs().ldc.vector);

    args.offA = addConst("offA", "cl_uint", kargs().offA);
    args.offBX = addConst("offX", "cl_uint", kargs().offBX);
    args.offCY = addConst("offY", "cl_uint", kargs().offCY);

    args.alpha = addVar("alpha", type,
        multiplierToString(kargs().dtype, kargs().alpha));
    args.beta = addVar("beta", type,
        multiplierToString(kargs().dtype, kargs().beta));

    A = addMatrix("A", type + "*", args.M, args.N, args.lda, args.offA);

    if (kargs().transA == clblasNoTrans) {
        X = addVector("X", type + "*", args.N, args.ldb, args.offBX);
        Y = addVector("Y", type + "*", args.M, args.ldc, args.offCY);
        naiveY = addVector("naiveY", type + "*", args.M, args.ldc, args.offCY);
    }
    else {
        X = addVector("X", type + "*", args.M, args.ldb, args.offBX);
        Y = addVector("Y", type + "*", args.N, args.ldc, args.offCY);
        naiveY = addVector("naiveY", type + "*", args.N, args.ldc, args.offCY);
    }
    naiveY->setCopy(Y);

    std::string bufAName, bufBName, bufCName;
    if (NULL == masterStep) {
        bufAName = "bufA";
        bufBName = "bufX";
        bufCName = "bufY";
    }
    else {
        bufAName = masterStep->getBuffer((BufferID)(long)step_.args.A)->name();
        bufBName = masterStep->getBuffer((BufferID)(long)step_.args.B)->name();
        bufCName = masterStep->getBuffer((BufferID)(long)step_.args.C)->name();
    }
    args.A = addBuffer(BUFFER_A, bufAName, "cl_mem", CL_MEM_READ_ONLY, A);
    args.B = addBuffer(BUFFER_B, bufBName, "cl_mem", CL_MEM_READ_ONLY, X);
    args.C = addBuffer(BUFFER_C, bufCName, "cl_mem", CL_MEM_READ_WRITE, Y);

    assignKargs(args);

    std::stringstream ss;
    ss << getBlasFunctionName() << "(order, transA, "
       << args.M->name() << ", " << args.N->name() << ", "
       << args.alpha->name() << ", " << A->matrixPointer()
       << ", " << args.lda->name() << ", "
       << X->vectorPointer() << ", " << args.ldb->name() << ", "
       << args.beta->name() << ", " << naiveY->vectorPointer()
       << ", " << args.ldc->name() << ")";
    naiveCall_ = ss.str();

    ss.str("");
    if (kargs().transA == clblasNoTrans) {
        ss << "compareVectors(" << args.M->name() << ", "
           << Y->vectorPointer() << ", " << naiveY->vectorPointer()
           << ", " << args.ldc->name() << ")";
    }
    else {
        ss << "compareVectors(" << args.N->name() << ", "
           << Y->vectorPointer() << ", " << naiveY->vectorPointer()
           << ", " << args.ldc->name() << ")";
    }
    compareCall_ = ss.str();
}

void
GemvStep::fixLD()
{
    CLBlasKargs args;

    args = kargs();

    /* M is always number of rows and N is number of columns in gemv */

    if ((args.order == clblasColumnMajor) && (args.lda.matrix < args.M)) {
        args.lda.matrix = args.M;
    }
    if ((args.order == clblasRowMajor) && (args.lda.matrix < args.N)) {
        args.lda.matrix = args.N;
    }

    if (args.ldb.vector == 0) {
        args.ldb.vector = 1;
    }
    if (args.ldc.vector == 0) {
        args.ldc.vector = 1;
    }
    /*
     * store original height of the matrix A
     */
    args.K = (args.transA == clblasNoTrans) ? args.M : args.N;

    setKargs(args);
}
