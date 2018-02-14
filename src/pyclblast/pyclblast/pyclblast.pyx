
####################################################################################################
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file defines the Python interface to CLBlast. It is inspired by:
# https://github.com/hunse/pyopencl_blas
#
####################################################################################################

import numpy as np
import pyopencl as cl
from pyopencl.array import Array

from libcpp cimport bool

####################################################################################################
# CLBlast and OpenCL data-types
####################################################################################################

cdef extern from "clblast_c.h":

    # Status codes
    ctypedef enum CLBlastStatusCode:
        CLBlastSuccess
        CLBlastOpenCLCompilerNotAvailable
        CLBlastTempBufferAllocFailure
        CLBlastOpenCLOutOfResources
        CLBlastOpenCLOutOfHostMemory
        CLBlastOpenCLBuildProgramFailure
        CLBlastInvalidValue
        CLBlastInvalidCommandQueue
        CLBlastInvalidMemObject
        CLBlastInvalidBinary
        CLBlastInvalidBuildOptions
        CLBlastInvalidProgram
        CLBlastInvalidProgramExecutable
        CLBlastInvalidKernelName
        CLBlastInvalidKernelDefinition
        CLBlastInvalidKernel
        CLBlastInvalidArgIndex
        CLBlastInvalidArgValue
        CLBlastInvalidArgSize
        CLBlastInvalidKernelArgs
        CLBlastInvalidLocalNumDimensions
        CLBlastInvalidLocalThreadsTotal
        CLBlastInvalidLocalThreadsDim
        CLBlastInvalidGlobalOffset
        CLBlastInvalidEventWaitList
        CLBlastInvalidEvent
        CLBlastInvalidOperation
        CLBlastInvalidBufferSize
        CLBlastInvalidGlobalWorkSize
        CLBlastNotImplemented
        CLBlastInvalidMatrixA
        CLBlastInvalidMatrixB
        CLBlastInvalidMatrixC
        CLBlastInvalidVectorX
        CLBlastInvalidVectorY
        CLBlastInvalidDimension
        CLBlastInvalidLeadDimA
        CLBlastInvalidLeadDimB
        CLBlastInvalidLeadDimC
        CLBlastInvalidIncrementX
        CLBlastInvalidIncrementY
        CLBlastInsufficientMemoryA
        CLBlastInsufficientMemoryB
        CLBlastInsufficientMemoryC
        CLBlastInsufficientMemoryX
        CLBlastInsufficientMemoryY
        CLBlastInvalidBatchCount
        CLBlastInvalidOverrideKernel
        CLBlastMissingOverrideParameter
        CLBlastInvalidLocalMemUsage
        CLBlastNoHalfPrecision
        CLBlastNoDoublePrecision
        CLBlastInvalidVectorScalar
        CLBlastInsufficientMemoryScalar
        CLBlastDatabaseError
        CLBlastUnknownError
        CLBlastUnexpectedError

    # OpenCL data-types
    ctypedef float cl_float
    ctypedef double cl_double
    ctypedef unsigned int cl_uint
    ctypedef struct cl_float2:
        cl_float x
        cl_float y
    ctypedef struct cl_double2:
        cl_double x
        cl_double y

    # OpenCL special data-types
    struct _cl_mem:
        pass
    struct _cl_command_queue:
        pass
    struct _cl_event:
        pass
    ctypedef _cl_mem* cl_mem
    ctypedef _cl_command_queue* cl_command_queue
    ctypedef _cl_event* cl_event

    # Matrix layout and transpose types
    ctypedef enum CLBlastLayout:
        CLBlastLayoutRowMajor
        CLBlastLayoutColMajor
    ctypedef enum CLBlastTranspose:
        CLBlastTransposeNo
        CLBlastTransposeYes
        CLBlastTransposeConjugate
    ctypedef enum CLBlastTriangle:
        CLBlastTriangleUpper
        CLBlastTriangleLower
    ctypedef enum CLBlastDiagonal:
        CLBlastDiagonalNonUnit
        CLBlastDiagonalUnit
    ctypedef enum CLBlastSide:
        CLBlastSideLeft
        CLBlastSideRight

    # Precision enum
    ctypedef enum CLBlastPrecision:
        CLBlastPrecisionSingle
        CLBlastPrecisionDouble
        CLBlastPrecisionComplexSingle
        CLBlastPrecisionComplexDouble

# Translates status codes into readable messages
cdef get_status_message(CLBlastStatusCode status):
    if status == CLBlastSuccess:
        return "CLBlastSuccess"
    if status == CLBlastOpenCLCompilerNotAvailable:
        return "CLBlastOpenCLCompilerNotAvailable: CL_COMPILER_NOT_AVAILABLE"
    if status == CLBlastTempBufferAllocFailure:
        return "CLBlastTempBufferAllocFailure: CL_MEM_OBJECT_ALLOCATION_FAILURE"
    if status == CLBlastOpenCLOutOfResources:
        return "CLBlastOpenCLOutOfResources: CL_OUT_OF_RESOURCES"
    if status == CLBlastOpenCLOutOfHostMemory:
        return "CLBlastOpenCLOutOfHostMemory: CL_OUT_OF_HOST_MEMORY"
    if status == CLBlastOpenCLBuildProgramFailure:
        return "CLBlastOpenCLBuildProgramFailure: CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error"
    if status == CLBlastInvalidValue:
        return "CLBlastInvalidValue: CL_INVALID_VALUE"
    if status == CLBlastInvalidCommandQueue:
        return "CLBlastInvalidCommandQueue: CL_INVALID_COMMAND_QUEUE"
    if status == CLBlastInvalidMemObject:
        return "CLBlastInvalidMemObject: CL_INVALID_MEM_OBJECT"
    if status == CLBlastInvalidBinary:
        return "CLBlastInvalidBinary: CL_INVALID_BINARY"
    if status == CLBlastInvalidBuildOptions:
        return "CLBlastInvalidBuildOptions: CL_INVALID_BUILD_OPTIONS"
    if status == CLBlastInvalidProgram:
        return "CLBlastInvalidProgram: CL_INVALID_PROGRAM"
    if status == CLBlastInvalidProgramExecutable:
        return "CLBlastInvalidProgramExecutable: CL_INVALID_PROGRAM_EXECUTABLE"
    if status == CLBlastInvalidKernelName:
        return "CLBlastInvalidKernelName: CL_INVALID_KERNEL_NAME"
    if status == CLBlastInvalidKernelDefinition:
        return "CLBlastInvalidKernelDefinition: CL_INVALID_KERNEL_DEFINITION"
    if status == CLBlastInvalidKernel:
        return "CLBlastInvalidKernel: CL_INVALID_KERNEL"
    if status == CLBlastInvalidArgIndex:
        return "CLBlastInvalidArgIndex: CL_INVALID_ARG_INDEX"
    if status == CLBlastInvalidArgValue:
        return "CLBlastInvalidArgValue: CL_INVALID_ARG_VALUE"
    if status == CLBlastInvalidArgSize:
        return "CLBlastInvalidArgSize: CL_INVALID_ARG_SIZE"
    if status == CLBlastInvalidKernelArgs:
        return "CLBlastInvalidKernelArgs: CL_INVALID_KERNEL_ARGS"
    if status == CLBlastInvalidLocalNumDimensions:
        return "CLBlastInvalidLocalNumDimensions: CL_INVALID_WORK_DIMENSION: Too many thread dimensions"
    if status == CLBlastInvalidLocalThreadsTotal:
        return "CLBlastInvalidLocalThreadsTotal: CL_INVALID_WORK_GROUP_SIZE: Too many threads in total"
    if status == CLBlastInvalidLocalThreadsDim:
        return "CLBlastInvalidLocalThreadsDim: CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension"
    if status == CLBlastInvalidGlobalOffset:
        return "CLBlastInvalidGlobalOffset: CL_INVALID_GLOBAL_OFFSET"
    if status == CLBlastInvalidEventWaitList:
        return "CLBlastInvalidEventWaitList: CL_INVALID_EVENT_WAIT_LIST"
    if status == CLBlastInvalidEvent:
        return "CLBlastInvalidEvent: CL_INVALID_EVENT"
    if status == CLBlastInvalidOperation:
        return "CLBlastInvalidOperation: CL_INVALID_OPERATION"
    if status == CLBlastInvalidBufferSize:
        return "CLBlastInvalidBufferSize: CL_INVALID_BUFFER_SIZE"
    if status == CLBlastInvalidGlobalWorkSize:
        return "CLBlastInvalidGlobalWorkSize: CL_INVALID_GLOBAL_WORK_SIZE"
    if status == CLBlastNotImplemented:
        return "CLBlastNotImplemented: Routine or functionality not implemented yet"
    if status == CLBlastInvalidMatrixA:
        return "CLBlastInvalidMatrixA: Matrix A is not a valid OpenCL buffer"
    if status == CLBlastInvalidMatrixB:
        return "CLBlastInvalidMatrixB: Matrix B is not a valid OpenCL buffer"
    if status == CLBlastInvalidMatrixC:
        return "CLBlastInvalidMatrixC: Matrix C is not a valid OpenCL buffer"
    if status == CLBlastInvalidVectorX:
        return "CLBlastInvalidVectorX: Vector X is not a valid OpenCL buffer"
    if status == CLBlastInvalidVectorY:
        return "CLBlastInvalidVectorY: Vector Y is not a valid OpenCL buffer"
    if status == CLBlastInvalidDimension:
        return "CLBlastInvalidDimension: Dimensions M, N, and K have to be larger than zero"
    if status == CLBlastInvalidLeadDimA:
        return "CLBlastInvalidLeadDimA: LD of A is smaller than the matrix's first dimension"
    if status == CLBlastInvalidLeadDimB:
        return "CLBlastInvalidLeadDimB: LD of B is smaller than the matrix's first dimension"
    if status == CLBlastInvalidLeadDimC:
        return "CLBlastInvalidLeadDimC: LD of C is smaller than the matrix's first dimension"
    if status == CLBlastInvalidIncrementX:
        return "CLBlastInvalidIncrementX: Increment of vector X cannot be zero"
    if status == CLBlastInvalidIncrementY:
        return "CLBlastInvalidIncrementY: Increment of vector Y cannot be zero"
    if status == CLBlastInsufficientMemoryA:
        return "CLBlastInsufficientMemoryA: Matrix A's OpenCL buffer is too small"
    if status == CLBlastInsufficientMemoryB:
        return "CLBlastInsufficientMemoryB: Matrix B's OpenCL buffer is too small"
    if status == CLBlastInsufficientMemoryC:
        return "CLBlastInsufficientMemoryC: Matrix C's OpenCL buffer is too small"
    if status == CLBlastInsufficientMemoryX:
        return "CLBlastInsufficientMemoryX: Vector X's OpenCL buffer is too small"
    if status == CLBlastInsufficientMemoryY:
        return "CLBlastInsufficientMemoryY: Vector Y's OpenCL buffer is too small"
    if status == CLBlastInvalidBatchCount:
        return "CLBlastInvalidBatchCount: The batch count needs to be positive"
    if status == CLBlastInvalidOverrideKernel:
        return "CLBlastInvalidOverrideKernel: Trying to override parameters for an invalid kernel"
    if status == CLBlastMissingOverrideParameter:
        return "CLBlastMissingOverrideParameter: Missing override parameter(s) for the target kernel"
    if status == CLBlastInvalidLocalMemUsage:
        return "CLBlastInvalidLocalMemUsage: Not enough local memory available on this device"
    if status == CLBlastNoHalfPrecision:
        return "CLBlastNoHalfPrecision: Half precision (16-bits) not supported by the device"
    if status == CLBlastNoDoublePrecision:
        return "CLBlastNoDoublePrecision: Double precision (64-bits) not supported by the device"
    if status == CLBlastInvalidVectorScalar:
        return "CLBlastInvalidVectorScalar: The unit-sized vector is not a valid OpenCL buffer"
    if status == CLBlastInsufficientMemoryScalar:
        return "CLBlastInsufficientMemoryScalar: The unit-sized vector's OpenCL buffer is too small"
    if status == CLBlastDatabaseError:
        return "CLBlastDatabaseError: Entry for the device was not found in the database"
    if status == CLBlastUnknownError:
        return "CLBlastUnknownError: A catch-all error code representing an unspecified error"
    if status == CLBlastUnexpectedError:
        return "CLBlastUnexpectedError: A catch-all error code representing an unexpected exception"
    return "PyCLBlast: unrecognized CLBlast status code (code %d)" % status

####################################################################################################
# Generic helpers
####################################################################################################

dtype_size = {np.dtype('float32'): 4,
              np.dtype('float64'): 8,
              np.dtype('complex64'): 8,
              np.dtype('complex128'): 16}

def dtypes_str(dtypes):
    if len(dtypes) == 1:
        return "'%s'" % dtypes[0]
    return "one of %s" % dtypes


def check_dtype(args, dtypes):
    dtype = args[0].dtype
    if not all(arg.dtype == dtype for arg in args):
        raise ValueError("PyCLBlast: All arguments must have the same dtype (%s)" % dtypes_str(dtypes))
    if dtype not in dtypes:
        raise ValueError("PyCLBlast: Data type must be %s" % dtypes_str(dtypes))
    return dtype


def check_array(a, ndim, name):
    if not isinstance(a, Array):
        raise ValueError("PyCLBlast: '%s' must be a PyOpenCL Array" % name)
    if not len(a.shape) == ndim:
        raise ValueError("PyCLBlast: '%s' must have %d dimensions (got %d)" % (name, ndim, len(a.shape)))


def check_matrix(a, name):
    check_array(a, 2, name)


def check_vector(a, name):
    check_array(a, 1, name)


####################################################################################################
# Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def swap(queue, n, x, y, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0):
    dtype = check_dtype([x, y], ["float32", "float64", "complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSswap(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDswap(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCswap(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZswap(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)
    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXswap' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
