################################################################################
 # Copyright 2014 Advanced Micro Devices, Inc.
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
################################################################################

# This pxd file defines all the enums and structs that we plan to use from
# python.  It is used from pyclBLAS.pyx
from libc.stdint cimport intptr_t, uintptr_t

cdef extern from "clBLAS.h":
    # These are base OpenCL enumerations that clBLAS uses
    cdef enum:
        CL_SUCCESS                      = 0
        CL_INVALID_VALUE                = -30
        CL_INVALID_COMMAND_QUEUE        = -36
        CL_INVALID_CONTEXT              = -34
        CL_INVALID_MEM_OBJECT           = -38
        CL_INVALID_DEVICE               = -33
        CL_INVALID_EVENT_WAIT_LIST      = -57
        CL_OUT_OF_RESOURCES             = -5
        CL_OUT_OF_HOST_MEMORY           = -6
        CL_INVALID_OPERATION            = -59
        CL_COMPILER_NOT_AVAILABLE       = -3
        CL_BUILD_PROGRAM_FAILURE        = -11

    cdef enum clblasStatus_:
        clblasSuccess               = CL_SUCCESS
        clblasInvalidValue          = CL_INVALID_VALUE
        clblasInvalidCommandQueue   = CL_INVALID_COMMAND_QUEUE
        clblasInvalidContext        = CL_INVALID_CONTEXT
        clblasInvalidMemObject      = CL_INVALID_MEM_OBJECT
        clblasInvalidDevice         = CL_INVALID_DEVICE
        clblasInvalidEventWaitList  = CL_INVALID_EVENT_WAIT_LIST
        clblasOutOfResources        = CL_OUT_OF_RESOURCES
        clblasOutOfHostMemory       = CL_OUT_OF_HOST_MEMORY
        clblasInvalidOperation      = CL_INVALID_OPERATION
        clblasCompilerNotAvailable  = CL_COMPILER_NOT_AVAILABLE
        clblasBuildProgramFailure   = CL_BUILD_PROGRAM_FAILURE
        clblasNotImplemented        = -1024
        clblasNotInitialized        = -1023
        clblasInvalidMatA
        clblasInvalidMatB
        clblasInvalidMatC
        clblasInvalidVecX
        clblasInvalidVecY
        clblasInvalidDim
        clblasInvalidLeadDimA
        clblasInvalidLeadDimB
        clblasInvalidLeadDimC
        clblasInvalidIncX
        clblasInvalidIncY
        clblasInsufficientMemMatA
        clblasInsufficientMemMatB
        clblasInsufficientMemMatC
        clblasInsufficientMemVecX
        clblasInsufficientMemVecY
    ctypedef clblasStatus_ clblasStatus

    cdef enum clblasOrder_:
        clblasRowMajor             = 0
        clblasColumnMajor          = 1
    ctypedef clblasStatus_ clblasOrder

    cdef enum clblasTranspose_:
        clblasNoTrans             = 0
        clblasTrans               = 1
        clblasConjTrans           = 2
    ctypedef clblasStatus_ clblasTranspose

    ctypedef unsigned int cl_uint
    ctypedef float cl_float
    ctypedef void* cl_mem
    ctypedef void* cl_command_queue
    ctypedef void* cl_event
