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
#include <blas_mempat.h>

#include "step.h"

using namespace clMath;

template <typename T>
struct FlagsDesc {
    T flag;
    const char *desc;
};

static const struct FlagsDesc<KernelExtraFlags> kernelExtraFlagsDesc[] = {
    { KEXTRA_TRANS_A,                   "KEXTRA_TRANS_A" },
    { KEXTRA_CONJUGATE_A,               "KEXTRA_CONJUGATE_A" },
    { KEXTRA_TRANS_B,                   "KEXTRA_TRANS_B" },
    { KEXTRA_CONJUGATE_B,               "KEXTRA_CONJUGATE_B" },
    { KEXTRA_COLUMN_MAJOR,              "KEXTRA_COLUMN_MAJOR" },
    { KEXTRA_UPPER_TRIANG,              "KEXTRA_UPPER_TRIANG" },
    { KEXTRA_SIDE_RIGHT,                "KEXTRA_SIDE_RIGHT" },
    { KEXTRA_UNIT_DIAGONAL,             "KEXTRA_UNIT_DIAGONAL" },
    { KEXTRA_TAILS_M,                   "KEXTRA_TAILS_M" },
    { KEXTRA_TAILS_N,                   "KEXTRA_TAILS_N" },
    { KEXTRA_TAILS_K,                   "KEXTRA_TAILS_K" },
    { KEXTRA_BETA_ZERO,                 "KEXTRA_BETA_ZERO" },
    { KEXTRA_NO_COPY_VEC_A,             "KEXTRA_NO_COPY_VEC_A" },
    { KEXTRA_NO_COPY_VEC_B,             "KEXTRA_NO_COPY_VEC_B" },
    { KEXTRA_NO_COPY_VEC_C,             "KEXTRA_NO_COPY_VEC_C" },
    { KEXTRA_SYRK_SEPARATE_DIAGONAL,    "KEXTRA_SYRK_SEPARATE_DIAGONAL" },
    { KEXTRA_SYRK_EVALUATE_DIAGONAL,    "KEXTRA_SYRK_EVALUATE_DIAGONAL" },
    { KEXTRA_SYRK_2K_RANK,              "KEXTRA_SYRK_2K_RANK" },
    { KEXTRA_INCX_ONE,                  "KEXTRA_INCX_ONE" },
    { KEXTRA_INCY_ONE,                  "KEXTRA_INCY_ONE" },
    { KEXTRA_ENABLE_MAD,                "KEXTRA_ENABLE_MAD" },
    { KEXTRA_VENDOR_AMD,                "KEXTRA_VENDOR_AMD" },

    { static_cast<KernelExtraFlags>(0), NULL }
};

static const struct FlagsDesc<CLMemLevel> memLevelFlagsDesc[] = {
    { CLMEM_LEVEL_LDS,                  "CLMEM_LEVEL_LDS" },
    { CLMEM_LEVEL_L1,                   "CLMEM_LEVEL_L1" },
    { CLMEM_LEVEL_L2,                   "CLMEM_LEVEL_L2" },

    { static_cast<CLMemLevel>(0), NULL }
};

template <typename T>
static void
dumpFlags(std::stringstream& ss, T flags, const struct FlagsDesc<T> *desc)
{
    bool first = true;

    if (flags == static_cast<T>(0)) {
        ss << "-";
        return;
    }

    for (size_t i = 0; desc[i].desc != NULL; i++) {
        if (flags & desc[i].flag) {
            if (!first) {
                ss << " ";
            }
            ss << desc[i].desc;
            flags = static_cast<T>(flags & ~desc[i].flag);
            first = false;
        }
    }
    if (flags != static_cast<T>(0)) {
        if (!first) {
            ss << " ";
        }
        ss << flags;
    }
}

std::string
Step::dtypeToString(DataType dtype)
{
    switch (dtype) {
    case TYPE_FLOAT:            return "cl_float";
    case TYPE_DOUBLE:           return "cl_double";
    case TYPE_COMPLEX_FLOAT:    return "FloatComplex";
    case TYPE_COMPLEX_DOUBLE:   return "DoubleComplex";
    default:                    return "";
    }
}

std::string
Step::multiplierToString(
    DataType dtype,
    ArgMultiplier arg)
{
    std::stringstream ss;

    switch (dtype) {
    case TYPE_FLOAT:
        ss << arg.argFloat;
        break;
    case TYPE_DOUBLE:
        ss << arg.argDouble;
        break;
    case TYPE_COMPLEX_FLOAT:
        ss << "floatComplex(" << arg.argFloatComplex.s[0] << ", "
           << arg.argFloatComplex.s[1] << ")";
        break;
    case TYPE_COMPLEX_DOUBLE:
        ss << "doubleComplex(" << arg.argDoubleComplex.s[0] << ", "
           << arg.argDoubleComplex.s[1] << ")";
        break;
    }
    return ss.str();
}

std::string
Step::dumpSubdim(const SubproblemDim *subdim)
{
    std::stringstream ss;

    if (subdim == NULL) {
        return ss.str();
    }

    ss << "    x      = ";
    if (subdim->x == SUBDIM_UNUSED) {
        ss << "SUBDIM_UNUSED";
    }
    else {
        ss << subdim->x;
    }
    ss << std::endl;

    ss << "    y      = ";
    if (subdim->y == SUBDIM_UNUSED) {
        ss << "SUBDIM_UNUSED";
    }
    else {
        ss << subdim->y;
    }
    ss << std::endl;

    ss << "    bwidth = " << subdim->bwidth << std::endl;

    ss << "    itemX  = ";
    if (subdim->itemX == SUBDIM_UNUSED) {
        ss << "SUBDIM_UNUSED";
    }
    else {
        ss << subdim->itemX;
    }
    ss << std::endl;

    ss << "    itemY  = ";
    if (subdim->itemY == SUBDIM_UNUSED) {
        ss << "SUBDIM_UNUSED";
    }
    else {
        ss << subdim->itemY;
    }
    ss << std::endl;

    return ss.str();
}

std::string
Step::dumpPgran()
{
    std::stringstream ss;
    const PGranularity *pgran = &step_.pgran;

    if (pgran == NULL) {
        return ss.str();
    }

    ss << "    wgDim  = " << pgran->wgDim << std::endl;
    ss << "    wgSize = (";
    for (unsigned int i = 0; i < pgran->wgDim; i++) {
        if (i != 0) {
            ss << ", ";
        }
        ss << pgran->wgSize[i];
    }
    ss << ")" << std::endl;
    ss << "    wfSize = " << pgran->wfSize << std::endl;
    return ss.str();
}

std::string
Step::dumpKextra()
{
    std::stringstream ss;
    const CLBLASKernExtra *kextra = &kextra_;

    if (kextra == NULL) {
        return ss.str();
    }

    ss << "    dtype    = ";
    switch (kextra->dtype) {
    case TYPE_FLOAT:
        ss << "TYPE_FLOAT";
        break;
    case TYPE_DOUBLE:
        ss << "TYPE_DOUBLE";
        break;
    case TYPE_COMPLEX_FLOAT:
        ss << "TYPE_COMPLEX_FLOAT";
        break;
    case TYPE_COMPLEX_DOUBLE:
        ss << "TYPE_COMPLEX_DOUBLE";
        break;
    }
    ss << std::endl;
    ss << "    flags    = ";
    dumpFlags<KernelExtraFlags>(ss, kextra->flags, kernelExtraFlagsDesc);
    ss << std::endl;
    ss << "    kernType = ";
    switch (kextra->kernType) {
    case CLBLAS_COMPUTING_KERNEL:
        ss << "CLBLAS_COMPUTING_KERNEL";
        break;
    case CLBLAS_PREP_A_KERNEL:
        ss << "CLBLAS_PREP_A_KERNEL";
        break;
    case CLBLAS_PREP_B_KERNEL:
        ss << "CLBLAS_PREP_B_KERNEL";
        break;
    default:
        ; // should not be reached
    }
    ss << std::endl;
    // Deprecated data
    ss << "    vecLen   = " << kextra->vecLen << std::endl;
    ss << "    vecLenA  = " << kextra->vecLenA << std::endl;
    ss << "    vecLenB  = " << kextra->vecLenB << std::endl;
    ss << "    vecLenC  = " << kextra->vecLenC << std::endl;
    return ss.str();
}

std::string
Step::dumpMemoryPattern()
{
    std::stringstream ss;
    const MemoryPattern *pattern = pattern_;
    CLBLASMpatExtra *mpatExtra = static_cast<CLBLASMpatExtra*>(pattern->extra);

    if (pattern == NULL) {
        return ss.str();
    }

    ss << "    name     = " << pattern->name << std::endl;
    ss << "    nrLevels = " << pattern->nrLevels << std::endl;
    ss << "    cuLevel  = " << pattern->cuLevel << std::endl;
    ss << "    thLevel  = " << pattern->thLevel << std::endl;

    ss << "    sops";
    if (pattern->sops == NULL) {
        ss << "     = -" << std::endl;
    }
    else {
        ss << std::endl;
        ss << "        genKernel             : "
           << ((pattern->sops->genKernel != NULL) ? "yes" : "no") << std::endl;
        ss << "        assignKargs           : "
           << ((pattern->sops->assignKargs != NULL) ? "yes" : "no") << std::endl;
        ss << "        isFitToLDS            : "
           << ((pattern->sops->isFitToLDS != NULL) ? "yes" : "no") << std::endl;
        ss << "        innerDecompositionAxis: "
           << ((pattern->sops->innerDecompositionAxis != NULL) ? "yes" : "no") << std::endl;
        ss << "        calcThreads           : "
           << ((pattern->sops->calcThreads != NULL) ? "yes" : "no") << std::endl;
        ss << "        imgPackMode           : "
           << ((pattern->sops->imgPackMode != NULL) ? "yes" : "no") << std::endl;
        ss << "        getFlags              : "
           << ((pattern->sops->getFlags != NULL) ? "yes" : "no") << std::endl;
    }

    ss << "    extra" << std::endl;
    ss << "        aMset  = ";
    dumpFlags<CLMemLevel>(ss, static_cast<CLMemLevel>(mpatExtra->aMset),
                        memLevelFlagsDesc);
    ss << std::endl;
    ss << "        bMset  = ";
    dumpFlags<CLMemLevel>(ss, static_cast<CLMemLevel>(mpatExtra->bMset),
                        memLevelFlagsDesc);
    ss << std::endl;
    ss << "        mobjA  = ";
    switch (mpatExtra->mobjA) {
    case CLMEM_GLOBAL_MEMORY:
        ss << "CLMEM_GLOBAL_MEMORY";
        break;
    case CLMEM_LOCAL_MEMORY:
        ss << "CLMEM_LOCAL_MEMORY";
        break;
    case CLMEM_IMAGE:
        ss << "CLMEM_IMAGE";
        break;
    }
    ss << std::endl;
    ss << "        mobjB  = ";
    switch (mpatExtra->mobjB) {
    case CLMEM_GLOBAL_MEMORY:
        ss << "CLMEM_GLOBAL_MEMORY";
        break;
    case CLMEM_LOCAL_MEMORY:
        ss << "CLMEM_LOCAL_MEMORY";
        break;
    case CLMEM_IMAGE:
        ss << "CLMEM_IMAGE";
        break;
    }
    ss << std::endl;
    return ss.str();
}
