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


#include <stdio.h>
#include "trsm_kgen.h"

void
genComplexMathOperators(
    struct KgenContext *ctx,
    DataType dtype)
{
    const char *ctype;
    char tmp[1024];

    ctype = dtypeBuiltinType(dtype);
    sprintf(tmp, "%s\ndiv(%s u, %s v)\n", ctype, ctype, ctype);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);
    sprintf(tmp, "return (%s)((u.x * v.x + u.y * v.y) / "
                             "(v.x * v.x + v.y * v.y),"
                             "(u.y * v.x - u.x * v.y) / "
                             "(v.x * v.x + v.y * v.y));\n", ctype);
    kgenAddStmt(ctx, tmp);
    kgenEndFuncBody(ctx);
    kgenAddBlankLine(ctx);

    sprintf(tmp, "%s\nmul(%s u, %s v)\n", ctype, ctype, ctype);
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);
    sprintf(tmp, "return (%s)(u.x * v.x - u.y * v.y, u.x * v.y + u.y * v.x);\n",
            ctype);
    kgenAddStmt(ctx, tmp);
    kgenEndFuncBody(ctx);
    kgenAddBlankLine(ctx);
}

