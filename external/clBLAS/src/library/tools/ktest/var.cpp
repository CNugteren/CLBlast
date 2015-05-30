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


#include "var.h"

using namespace clMath;

struct MemFlags {
    cl_mem_flags flag;
    const char* name;
};

static const struct MemFlags MEM_FLAGS[] = {
    { CL_MEM_READ_WRITE,    "CL_MEM_READ_WRITE" },
    { CL_MEM_WRITE_ONLY,    "CL_MEM_WRITE_ONLY" },
    { CL_MEM_READ_ONLY,     "CL_MEM_READ_ONLY" },
    { CL_MEM_USE_HOST_PTR,  "CL_MEM_USE_HOST_PTR" },
    { CL_MEM_ALLOC_HOST_PTR,"CL_MEM_ALLOC_HOST_PTR" },
    { CL_MEM_COPY_HOST_PTR, "CL_MEM_COPY_HOST_PTR" },
    { 0, NULL }
};

Variable::Variable(
    const std::string& name,
    const std::string& type,
    const std::string& defaultValue)
{
    name_ = name;
    type_ = type;
    defaultValue_ = defaultValue;
    isBuffer_ = false;
    constant_ = false;

    copyOf_ = NULL;

    flags_ = 0;
    hostPtr_ = NULL;
}

Variable::Variable()
{
    Variable("", "");
}

MatrixVariable::MatrixVariable(
    const std::string& name,
    const std::string& type,
    const std::string& defaultValue)
{
    name_ = name;
    type_ = type;
    defaultValue_ = defaultValue;
    isBuffer_ = false;
    constant_ = false;

    copyOf_ = NULL;

    flags_ = 0;
    hostPtr_ = NULL;

    rows_ = NULL;
    columns_ = NULL;
    ld_ = NULL;
    off_ = NULL;
}

VectorVariable::VectorVariable(
    const std::string& name,
    const std::string& type,
    const std::string& defaultValue)
{
    name_ = name;
    type_ = type;
    defaultValue_ = defaultValue;
    isBuffer_ = false;
    constant_ = false;

    copyOf_ = NULL;

    flags_ = 0;
    hostPtr_ = NULL;

    nElems_ = NULL;
    inc_ = NULL;
    off_ = NULL;
}

Variable::~Variable()
{
}

void
Variable::setDefaultValue(const std::string& defaultValue)
{
    defaultValue_ = defaultValue;
}

void
Variable::setConstant(bool constant)
{
    constant_ = constant;
}

void
Variable::setCopy(Variable *copy)
{
    copyOf_ = copy;
}

void
MatrixVariable::setMatrixSize(
    Variable *rows,
    Variable *columns,
    Variable *ld,
    Variable *off)
{
    if ((rows == NULL) || (columns == NULL)) {
        return;
    }
    rows_ = rows;
    columns_ = columns;
    ld_ = ld;
    off_ = off;
    matrixPointer_ = name_;
    if (off != NULL) {
        matrixPointer_ += " + " + off_->name();
}
}

void
VectorVariable::setVectorSize(
    Variable *nElems,
    Variable *inc,
    Variable *off)
{
    if (nElems == NULL) {
        return;
    }
    nElems_ = nElems;
    inc_ = inc;
    off_ = off;
    vectorPointer_ = name_;
    if (off != NULL) {
        vectorPointer_ += " + " + off_->name();
}
}

std::string
Variable::flagsStr() const
{
    std::string str;
    size_t i;

    if (type_ != "cl_mem") {
        return "";
    }
    if (flags_ == 0) {
        return "0";
    }
    for (i = 0; MEM_FLAGS[i].flag != 0; i++) {
        if (flags_ & MEM_FLAGS[i].flag) {
            if (!str.empty()) {
                str += " | ";
            }
            str += MEM_FLAGS[i].name;
        }
    }
    return str;
}

void
Variable::setFlags(cl_mem_flags flags)
{
    if (type_ == "cl_mem") {
        flags_ = flags;
    }
}

void
Variable::setHostPtr(Variable *hostPtr)
{
    if (type_ == "cl_mem") {
        hostPtr_ = hostPtr;
    }
}
