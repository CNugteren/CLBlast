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


#ifndef KTEST_VAR_H__
#define KTEST_VAR_H__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <string>

namespace clMath {

typedef enum BufferID {
    BUFFER_NONE,
    BUFFER_A,
    BUFFER_B,
    BUFFER_C
} BufferID;

/**
 * @internal
 * @brief Variable class
 *
 * Objects of this class store name, type and other attributes of variables
 * necessary for further code generation.
 *
 */

class Variable {
protected:
    std::string name_;
    std::string type_;
    std::string defaultValue_;
    bool constant_;
    bool isBuffer_;
    BufferID bufID_;

    Variable *copyOf_;

    /* Buffer object info */
    cl_mem_flags flags_;
    Variable *hostPtr_;

public:
    Variable(const std::string& name, const std::string& type,
        const std::string& defaultValue = "");
    Variable();
    ~Variable();

    const std::string& name() const         { return name_; }
    const std::string& type() const         { return type_; }

    const std::string& defaultValue() const { return defaultValue_; }
    void setDefaultValue(const std::string& defaultValue);

    bool constant() const                   { return constant_; }
    bool isBuffer() const                   { return isBuffer_; }
    BufferID getBufID() const               { return bufID_; }
    void setConstant(bool constant);
    void setIsBuffer(bool isBuffer)         { isBuffer_ = isBuffer; }

    Variable* copyOf() const                { return copyOf_; }
    void setCopy(Variable *copy);

    void setBufferID(BufferID bufID)        { bufID_ = bufID; }

    cl_mem_flags flags() const              { return flags_; }
    std::string flagsStr() const;
    void setFlags(cl_mem_flags flags);

    Variable* hostPtr() const { return hostPtr_; }
    void setHostPtr(Variable *var);
};

class ArrayVariableInterface : public Variable {
public:
    virtual bool isMatrix() = 0;
    virtual bool isVector() = 0;
    virtual ~ArrayVariableInterface() {}
};

/**
 * @internal
 * @brief Matrix variable class
 *
 * Objects of this class store information about matrix array
 * necessary for further code generation.
 *
 */
class MatrixVariable : public ArrayVariableInterface {
private:
    /* Matrix info */
    Variable *rows_;
    Variable *columns_;
    Variable *ld_;
    Variable *off_;
    std::string matrixPointer_;
public:
    Variable* rows() const                  { return rows_; }
    Variable* columns() const               { return columns_; }
    Variable* ld() const                    { return ld_; }
    Variable* off() const                   { return off_; }

    bool isMatrix()                         { return true; }
    bool isVector()                         { return false; }

    const std::string& matrixPointer() const  { return matrixPointer_; }

    void setMatrixSize(Variable *rows, Variable *columns,
        Variable *ld = NULL, Variable *off = NULL);
    MatrixVariable(const std::string& name, const std::string& type,
        const std::string& defaultValue = "");
    ~MatrixVariable() {};
};

/**
 * @internal
 * @brief Vector variable class
 *
 * Objects of this class store information about vector array
 * necessary for further code generation.
 *
 */
class VectorVariable : public ArrayVariableInterface {
private:
    /* Vector info */
    Variable *nElems_;
    Variable *inc_;
    Variable *off_;
    std::string vectorPointer_;
public:
    Variable* nElems() const                { return nElems_; }
    Variable* inc() const                   { return inc_; }
    Variable* off() const                   { return off_; }

    virtual bool isMatrix()                 { return false; }
    virtual bool isVector()                 { return true; }

    const std::string& vectorPointer() const  { return vectorPointer_; }

    void setVectorSize(Variable *nElems, Variable *inc,
        Variable *off = NULL);
    VectorVariable(const std::string& name, const std::string& type,
        const std::string& defaultValue = "");
};

}   // namespace clMath

#endif  // KTEST_VAR_H__
