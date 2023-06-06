#distutils: language = c++
#cython: binding=True
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

import binascii
import struct
import numpy as np
import pyopencl as cl
from pyopencl.array import Array
from libcpp cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport strdup

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
    ctypedef unsigned short cl_half

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
# Half-precision utility functions
####################################################################################################

def float32_to_float16(float32):
    # Taken from https://gamedev.stackexchange.com/a/28756
    F16_EXPONENT_BITS = 0x1F
    F16_EXPONENT_SHIFT = 10
    F16_EXPONENT_BIAS = 15
    F16_MANTISSA_BITS = 0x3ff
    F16_MANTISSA_SHIFT = (23 - F16_EXPONENT_SHIFT)
    F16_MAX_EXPONENT = (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)

    a = struct.pack('>f', float32)
    b = binascii.hexlify(a)

    f32 = int(b, 16)
    sign = (f32 >> 16) & 0x8000
    exponent = ((f32 >> 23) & 0xff) - 127
    mantissa = f32 & 0x007fffff

    if exponent == 128:
        f16 = sign | F16_MAX_EXPONENT
        if mantissa:
            f16 |= (mantissa & F16_MANTISSA_BITS)
    elif exponent > 15:
        f16 = sign | F16_MAX_EXPONENT
    elif exponent > -15:
        exponent += F16_EXPONENT_BIAS
        mantissa >>= F16_MANTISSA_SHIFT
        f16 = sign | exponent << F16_EXPONENT_SHIFT | mantissa
    else:
        f16 = sign
    return f16

####################################################################################################
# Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHswap(const size_t n, cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def swap(queue, n, x, y, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0):
    """
    xSWAP: Swap two vectors
    """

    dtype = check_dtype([x, y], ["float32", "float64", "complex64", "complex128", "float16"])
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
    elif dtype == np.dtype("float16"):
        err = CLBlastHswap(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXswap' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSscal(const size_t n, const float alpha, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDscal(const size_t n, const double alpha, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCscal(const size_t n, const cl_float2 alpha, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZscal(const size_t n, const cl_double2 alpha, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHscal(const size_t n, const cl_half alpha, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def scal(queue, n, x, x_inc = 1, alpha = 1.0, x_offset = 0):
    """
    xSCAL: Vector scaling
    """

    dtype = check_dtype([x], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSscal(n, <cl_float>alpha, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDscal(n, <cl_double>alpha, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCscal(n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZscal(n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHscal(n, <cl_half>alpha, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXscal' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastScopy(const size_t n, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDcopy(const size_t n, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCcopy(const size_t n, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZcopy(const size_t n, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHcopy(const size_t n, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def copy(queue, n, x, y, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0):
    """
    xCOPY: Vector copy
    """

    dtype = check_dtype([x, y], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastScopy(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDcopy(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCcopy(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZcopy(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHcopy(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXcopy' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSaxpy(const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDaxpy(const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCaxpy(const size_t n, const cl_float2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZaxpy(const size_t n, const cl_double2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHaxpy(const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def axpy(queue, n, x, y, x_inc = 1, y_inc = 1, alpha = 1.0, x_offset = 0, y_offset = 0):
    """
    xAXPY: Vector-times-constant plus vector
    """

    dtype = check_dtype([x, y], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSaxpy(n, <cl_float>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDaxpy(n, <cl_double>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCaxpy(n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZaxpy(n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHaxpy(n, <cl_half>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXaxpy' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Dot product of two vectors: SDOT/DDOT/HDOT
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSdot(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDdot(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHdot(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def dot(queue, n, x, y, dot, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0, dot_offset = 0):
    """
    xDOT: Dot product of two vectors
    """

    dtype = check_dtype([x, y, dot], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(dot, "dot")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem dot_buffer = <cl_mem><size_t>dot.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSdot(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDdot(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHdot(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXdot' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Dot product of two complex vectors: CDOTU/ZDOTU
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCdotu(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZdotu(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def dotu(queue, n, x, y, dot, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0, dot_offset = 0):
    """
    xDOTU: Dot product of two complex vectors
    """

    dtype = check_dtype([x, y, dot], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(dot, "dot")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem dot_buffer = <cl_mem><size_t>dot.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCdotu(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZdotu(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXdotu' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCdotc(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZdotc(const size_t n, cl_mem dot_buffer, const size_t dot_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def dotc(queue, n, x, y, dot, x_inc = 1, y_inc = 1, x_offset = 0, y_offset = 0, dot_offset = 0):
    """
    xDOTC: Dot product of two complex vectors, one conjugated
    """

    dtype = check_dtype([x, y, dot], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(dot, "dot")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem dot_buffer = <cl_mem><size_t>dot.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCdotc(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZdotc(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXdotc' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSnrm2(const size_t n, cl_mem nrm2_buffer, const size_t nrm2_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDnrm2(const size_t n, cl_mem nrm2_buffer, const size_t nrm2_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastScnrm2(const size_t n, cl_mem nrm2_buffer, const size_t nrm2_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDznrm2(const size_t n, cl_mem nrm2_buffer, const size_t nrm2_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHnrm2(const size_t n, cl_mem nrm2_buffer, const size_t nrm2_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def nrm2(queue, n, x, nrm2, x_inc = 1, x_offset = 0, nrm2_offset = 0):
    """
    xNRM2: Euclidian norm of a vector
    """

    dtype = check_dtype([x, nrm2], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(nrm2, "nrm2")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem nrm2_buffer = <cl_mem><size_t>nrm2.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSnrm2(n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDnrm2(n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastScnrm2(n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastDznrm2(n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHnrm2(n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXnrm2' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSasum(const size_t n, cl_mem asum_buffer, const size_t asum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDasum(const size_t n, cl_mem asum_buffer, const size_t asum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastScasum(const size_t n, cl_mem asum_buffer, const size_t asum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDzasum(const size_t n, cl_mem asum_buffer, const size_t asum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHasum(const size_t n, cl_mem asum_buffer, const size_t asum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def asum(queue, n, x, asum, x_inc = 1, x_offset = 0, asum_offset = 0):
    """
    xASUM: Absolute sum of values in a vector
    """

    dtype = check_dtype([x, asum], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(asum, "asum")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem asum_buffer = <cl_mem><size_t>asum.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSasum(n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDasum(n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastScasum(n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastDzasum(n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHasum(n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXasum' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsum(const size_t n, cl_mem sum_buffer, const size_t sum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsum(const size_t n, cl_mem sum_buffer, const size_t sum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastScsum(const size_t n, cl_mem sum_buffer, const size_t sum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDzsum(const size_t n, cl_mem sum_buffer, const size_t sum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsum(const size_t n, cl_mem sum_buffer, const size_t sum_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def sum(queue, n, x, sum, x_inc = 1, x_offset = 0, sum_offset = 0):
    """
    xSUM: Sum of values in a vector (non-BLAS function)
    """

    dtype = check_dtype([x, sum], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(sum, "sum")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem sum_buffer = <cl_mem><size_t>sum.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastScsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastDzsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsum' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastiSamax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiDamax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiCamax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiZamax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiHamax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def amax(queue, n, x, imax, x_inc = 1, x_offset = 0, imax_offset = 0):
    """
    xAMAX: Index of absolute maximum value in a vector
    """

    dtype = check_dtype([x, imax], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(imax, "imax")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem imax_buffer = <cl_mem><size_t>imax.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastiSamax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastiDamax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastiCamax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastiZamax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastiHamax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXamax' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastiSamin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiDamin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiCamin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiZamin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiHamin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def amin(queue, n, x, imin, x_inc = 1, x_offset = 0, imin_offset = 0):
    """
    xAMIN: Index of absolute minimum value in a vector (non-BLAS function)
    """

    dtype = check_dtype([x, imin], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(imin, "imin")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem imin_buffer = <cl_mem><size_t>imin.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastiSamin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastiDamin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastiCamin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastiZamin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastiHamin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXamin' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastiSmax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiDmax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiCmax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiZmax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiHmax(const size_t n, cl_mem imax_buffer, const size_t imax_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def max(queue, n, x, imax, x_inc = 1, x_offset = 0, imax_offset = 0):
    """
    xMAX: Index of maximum value in a vector (non-BLAS function)
    """

    dtype = check_dtype([x, imax], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(imax, "imax")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem imax_buffer = <cl_mem><size_t>imax.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastiSmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastiDmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastiCmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastiZmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastiHmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXmax' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastiSmin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiDmin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiCmin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiZmin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastiHmin(const size_t n, cl_mem imin_buffer, const size_t imin_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def min(queue, n, x, imin, x_inc = 1, x_offset = 0, imin_offset = 0):
    """
    xMIN: Index of minimum value in a vector (non-BLAS function)
    """

    dtype = check_dtype([x, imin], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_matrix(imin, "imin")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem imin_buffer = <cl_mem><size_t>imin.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastiSmin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastiDmin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastiCmin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastiZmin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastiHmin(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXmin' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const float beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const double beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_float2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_double2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_half beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def gemv(queue, m, n, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, a_transp = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xGEMV: General matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSgemv(CLBlastLayoutRowMajor, a_transpose, m, n, <cl_float>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDgemv(CLBlastLayoutRowMajor, a_transpose, m, n, <cl_double>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCgemv(CLBlastLayoutRowMajor, a_transpose, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float2>cl_float2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgemv(CLBlastLayoutRowMajor, a_transpose, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double2>cl_double2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHgemv(CLBlastLayoutRowMajor, a_transpose, m, n, <cl_half>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_half>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgemv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const size_t kl, const size_t ku, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const float beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const size_t kl, const size_t ku, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const double beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const size_t kl, const size_t ku, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_float2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const size_t kl, const size_t ku, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_double2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const size_t m, const size_t n, const size_t kl, const size_t ku, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_half beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def gbmv(queue, m, n, kl, ku, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, a_transp = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xGBMV: General banded matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSgbmv(CLBlastLayoutRowMajor, a_transpose, m, n, kl, ku, <cl_float>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDgbmv(CLBlastLayoutRowMajor, a_transpose, m, n, kl, ku, <cl_double>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCgbmv(CLBlastLayoutRowMajor, a_transpose, m, n, kl, ku, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float2>cl_float2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgbmv(CLBlastLayoutRowMajor, a_transpose, m, n, kl, ku, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double2>cl_double2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHgbmv(CLBlastLayoutRowMajor, a_transpose, m, n, kl, ku, <cl_half>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_half>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgbmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian matrix-vector multiplication: CHEMV/ZHEMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChemv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_float2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhemv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_double2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def hemv(queue, n, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xHEMV: Hermitian matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["complex64", "complex128"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChemv(CLBlastLayoutRowMajor, triangle, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float2>cl_float2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhemv(CLBlastLayoutRowMajor, triangle, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double2>cl_double2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhemv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_float2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_double2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def hbmv(queue, n, k, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xHBMV: Hermitian banded matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["complex64", "complex128"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChbmv(CLBlastLayoutRowMajor, triangle, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float2>cl_float2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhbmv(CLBlastLayoutRowMajor, triangle, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double2>cl_double2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhbmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_float2 alpha, const cl_mem ap_buffer, const size_t ap_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_float2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_double2 alpha, const cl_mem ap_buffer, const size_t ap_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_double2 beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def hpmv(queue, n, ap, x, y, ap_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, ap_offset = 0, x_offset = 0, y_offset = 0):
    """
    xHPMV: Hermitian packed matrix-vector multiplication
    """

    dtype = check_dtype([ap, x, y], ["complex64", "complex128"])
    check_matrix(ap, "ap")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChpmv(CLBlastLayoutRowMajor, triangle, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), ap_buffer, ap_offset, x_buffer, x_offset, x_inc, <cl_float2>cl_float2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhpmv(CLBlastLayoutRowMajor, triangle, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), ap_buffer, ap_offset, x_buffer, x_offset, x_inc, <cl_double2>cl_double2(x=beta.real,y=beta.imag), y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhpmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsymv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const float beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsymv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const double beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsymv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_half beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def symv(queue, n, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xSYMV: Symmetric matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["float32", "float64", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsymv(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsymv(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsymv(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_half>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsymv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const float beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const double beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const size_t k, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_half beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def sbmv(queue, n, k, a, x, y, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, a_offset = 0, x_offset = 0, y_offset = 0):
    """
    xSBMV: Symmetric banded matrix-vector multiplication
    """

    dtype = check_dtype([a, x, y], ["float32", "float64", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsbmv(CLBlastLayoutRowMajor, triangle, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_float>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsbmv(CLBlastLayoutRowMajor, triangle, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_double>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsbmv(CLBlastLayoutRowMajor, triangle, n, k, <cl_half>alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, <cl_half>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsbmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSspmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem ap_buffer, const size_t ap_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const float beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDspmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem ap_buffer, const size_t ap_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const double beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHspmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem ap_buffer, const size_t ap_offset, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_half beta, cl_mem y_buffer, const size_t y_offset, const size_t y_inc,cl_command_queue* queue, cl_event* event)

def spmv(queue, n, ap, x, y, ap_ld, x_inc = 1, y_inc = 1, alpha = 1.0, beta = 0.0, lower_triangle = False, ap_offset = 0, x_offset = 0, y_offset = 0):
    """
    xSPMV: Symmetric packed matrix-vector multiplication
    """

    dtype = check_dtype([ap, x, y], ["float32", "float64", "float16"])
    check_matrix(ap, "ap")
    check_vector(x, "x")
    check_vector(y, "y")

    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSspmv(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, <cl_float>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDspmv(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, <cl_double>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHspmv(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, <cl_half>beta, y_buffer, y_offset, y_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXspmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def trmv(queue, n, a, x, a_ld, x_inc = 1, lower_triangle = False, a_transp = False, unit_diagonal = False, a_offset = 0, x_offset = 0):
    """
    xTRMV: Triangular matrix-vector multiplication
    """

    dtype = check_dtype([a, x], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStrmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtrmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtrmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtrmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHtrmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtrmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const size_t k, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const size_t k, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const size_t k, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const size_t k, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const size_t k, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def tbmv(queue, n, k, a, x, a_ld, x_inc = 1, lower_triangle = False, a_transp = False, unit_diagonal = False, a_offset = 0, x_offset = 0):
    """
    xTBMV: Triangular banded matrix-vector multiplication
    """

    dtype = check_dtype([a, x], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_vector(x, "x")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStbmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtbmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtbmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtbmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHtbmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtbmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem ap_buffer, const size_t ap_offset, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem ap_buffer, const size_t ap_offset, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem ap_buffer, const size_t ap_offset, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem ap_buffer, const size_t ap_offset, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem ap_buffer, const size_t ap_offset, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def tpmv(queue, n, ap, x, ap_ld, x_inc = 1, lower_triangle = False, a_transp = False, unit_diagonal = False, ap_offset = 0, x_offset = 0):
    """
    xTPMV: Triangular packed matrix-vector multiplication
    """

    dtype = check_dtype([ap, x], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(ap, "ap")
    check_vector(x, "x")

    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStpmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtpmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtpmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtpmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHtpmv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtpmv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t n, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem x_buffer, const size_t x_offset, const size_t x_inc,cl_command_queue* queue, cl_event* event)

def trsv(queue, n, a, x, a_ld, x_inc = 1, lower_triangle = False, a_transp = False, unit_diagonal = False, a_offset = 0, x_offset = 0):
    """
    xTRSV: Solves a triangular system of equations
    """

    dtype = check_dtype([a, x], ["float32", "float64", "complex64", "complex128"])
    check_matrix(a, "a")
    check_vector(x, "x")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStrsv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtrsv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtrsv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtrsv(CLBlastLayoutRowMajor, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtrsv' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General rank-1 matrix update: SGER/DGER/HGER
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSger(const CLBlastLayout layout, const size_t m, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDger(const CLBlastLayout layout, const size_t m, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHger(const CLBlastLayout layout, const size_t m, const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def ger(queue, m, n, x, y, a, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, x_offset = 0, y_offset = 0, a_offset = 0):
    """
    xGER: General rank-1 matrix update
    """

    dtype = check_dtype([x, y, a], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSger(CLBlastLayoutRowMajor, m, n, <cl_float>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDger(CLBlastLayoutRowMajor, m, n, <cl_double>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHger(CLBlastLayoutRowMajor, m, n, <cl_half>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXger' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General rank-1 complex matrix update: CGERU/ZGERU
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCgeru(const CLBlastLayout layout, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgeru(const CLBlastLayout layout, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def geru(queue, m, n, x, y, a, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, x_offset = 0, y_offset = 0, a_offset = 0):
    """
    xGERU: General rank-1 complex matrix update
    """

    dtype = check_dtype([x, y, a], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCgeru(CLBlastLayoutRowMajor, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgeru(CLBlastLayoutRowMajor, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgeru' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General rank-1 complex conjugated matrix update: CGERC/ZGERC
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCgerc(const CLBlastLayout layout, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgerc(const CLBlastLayout layout, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def gerc(queue, m, n, x, y, a, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, x_offset = 0, y_offset = 0, a_offset = 0):
    """
    xGERC: General rank-1 complex conjugated matrix update
    """

    dtype = check_dtype([x, y, a], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCgerc(CLBlastLayoutRowMajor, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgerc(CLBlastLayoutRowMajor, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgerc' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian rank-1 matrix update: CHER/ZHER
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCher(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZher(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def her(queue, n, x, a, a_ld, x_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, a_offset = 0):
    """
    xHER: Hermitian rank-1 matrix update
    """

    dtype = check_dtype([x, a], ["complex64", "complex128"])
    check_vector(x, "x")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCher(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZher(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXher' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian packed rank-1 matrix update: CHPR/ZHPR
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChpr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhpr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)

def hpr(queue, n, x, ap, ap_ld, x_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, ap_offset = 0):
    """
    xHPR: Hermitian packed rank-1 matrix update
    """

    dtype = check_dtype([x, ap], ["complex64", "complex128"])
    check_vector(x, "x")
    check_matrix(ap, "ap")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChpr(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhpr(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhpr' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian rank-2 matrix update: CHER2/ZHER2
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCher2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_float2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZher2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_double2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def her2(queue, n, x, y, a, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, y_offset = 0, a_offset = 0):
    """
    xHER2: Hermitian rank-2 matrix update
    """

    dtype = check_dtype([x, y, a], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCher2(CLBlastLayoutRowMajor, triangle, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZher2(CLBlastLayoutRowMajor, triangle, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXher2' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChpr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_float2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_double2 alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)

def hpr2(queue, n, x, y, ap, ap_ld, x_inc = 1, y_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, y_offset = 0, ap_offset = 0):
    """
    xHPR2: Hermitian packed rank-2 matrix update
    """

    dtype = check_dtype([x, y, ap], ["complex64", "complex128"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(ap, "ap")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChpr2(CLBlastLayoutRowMajor, triangle, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhpr2(CLBlastLayoutRowMajor, triangle, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhpr2' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsyr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsyr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsyr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def syr(queue, n, x, a, a_ld, x_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, a_offset = 0):
    """
    xSYR: Symmetric rank-1 matrix update
    """

    dtype = check_dtype([x, a], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsyr(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsyr(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsyr(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsyr' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSspr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDspr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHspr(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)

def spr(queue, n, x, ap, ap_ld, x_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, ap_offset = 0):
    """
    xSPR: Symmetric packed rank-1 matrix update
    """

    dtype = check_dtype([x, ap], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_matrix(ap, "ap")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSspr(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDspr(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHspr(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXspr' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem a_buffer, const size_t a_offset, const size_t a_ld,cl_command_queue* queue, cl_event* event)

def syr2(queue, n, x, y, a, a_ld, x_inc = 1, y_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, y_offset = 0, a_offset = 0):
    """
    xSYR2: Symmetric rank-2 matrix update
    """

    dtype = check_dtype([x, y, a], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(a, "a")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsyr2(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsyr2(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsyr2(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsyr2' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSspr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const float alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDspr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const double alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHspr2(const CLBlastLayout layout, const CLBlastTriangle triangle, const size_t n, const cl_half alpha, const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const cl_mem y_buffer, const size_t y_offset, const size_t y_inc, cl_mem ap_buffer, const size_t ap_offset,cl_command_queue* queue, cl_event* event)

def spr2(queue, n, x, y, ap, ap_ld, x_inc = 1, y_inc = 1, alpha = 1.0, lower_triangle = False, x_offset = 0, y_offset = 0, ap_offset = 0):
    """
    xSPR2: Symmetric packed rank-2 matrix update
    """

    dtype = check_dtype([x, y, ap], ["float32", "float64", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")
    check_matrix(ap, "ap")

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr
    cdef cl_mem ap_buffer = <cl_mem><size_t>ap.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSspr2(CLBlastLayoutRowMajor, triangle, n, <cl_float>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDspr2(CLBlastLayoutRowMajor, triangle, n, <cl_double>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHspr2(CLBlastLayoutRowMajor, triangle, n, <cl_half>alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXspr2' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_half beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def gemm(queue, m, n, k, a, b, c, a_ld, b_ld, c_ld, alpha = 1.0, beta = 0.0, a_transp = False, b_transp = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xGEMM: General matrix-matrix multiplication
    """

    dtype = check_dtype([a, b, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    b_transpose = CLBlastTransposeYes if b_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSgemm(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDgemm(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCgemm(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgemm(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHgemm(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_half>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_half>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgemm' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_half beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def symm(queue, m, n, a, b, c, a_ld, b_ld, c_ld, alpha = 1.0, beta = 0.0, right_side = False, lower_triangle = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xSYMM: Symmetric matrix-matrix multiplication
    """

    dtype = check_dtype([a, b, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    side = CLBlastSideRight if right_side else CLBlastSideLeft
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsymm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_float>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsymm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_double>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCsymm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZsymm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsymm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_half>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_half>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsymm' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastChemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def hemm(queue, m, n, a, b, c, a_ld, b_ld, c_ld, alpha = 1.0, beta = 0.0, right_side = False, lower_triangle = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xHEMM: Hermitian matrix-matrix multiplication
    """

    dtype = check_dtype([a, b, c], ["complex64", "complex128"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    side = CLBlastSideRight if right_side else CLBlastSideLeft
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastChemm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZhemm(CLBlastLayoutRowMajor, side, triangle, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXhemm' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_half beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def syrk(queue, n, k, a, c, a_ld, c_ld, alpha = 1.0, beta = 0.0, lower_triangle = False, a_transp = False, a_offset = 0, c_offset = 0):
    """
    xSYRK: Rank-K update of a symmetric matrix
    """

    dtype = check_dtype([a, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsyrk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsyrk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCsyrk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZsyrk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsyrk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_half>alpha, a_buffer, a_offset, a_ld, <cl_half>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsyrk' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Rank-K update of a hermitian matrix: CHERK/ZHERK
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def herk(queue, n, k, a, c, a_ld, c_ld, alpha = 1.0, beta = 0.0, lower_triangle = False, a_transp = False, a_offset = 0, c_offset = 0):
    """
    xHERK: Rank-K update of a hermitian matrix
    """

    dtype = check_dtype([a, c], ["complex64", "complex128"])
    check_matrix(a, "a")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCherk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZherk(CLBlastLayoutRowMajor, triangle, a_transpose, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXherk' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_half beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def syr2k(queue, n, k, a, b, c, a_ld, b_ld, c_ld, alpha = 1.0, beta = 0.0, lower_triangle = False, ab_transp = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xSYR2K: Rank-2K update of a symmetric matrix
    """

    dtype = check_dtype([a, b, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    ab_transpose = CLBlastTransposeYes if ab_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSsyr2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDsyr2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCsyr2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZsyr2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHsyr2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_half>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_half>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXsyr2k' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastCher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld,cl_command_queue* queue, cl_event* event)

def her2k(queue, n, k, a, b, c, a_ld, b_ld, c_ld, alpha = 1.0, beta = 0.0, lower_triangle = False, ab_transp = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xHER2K: Rank-2K update of a hermitian matrix
    """

    dtype = check_dtype([a, b, c], ["complex64", "complex128"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    ab_transpose = CLBlastTransposeYes if ab_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("complex64"):
        err = CLBlastCher2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_float>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZher2k(CLBlastLayoutRowMajor, triangle, ab_transpose, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, <cl_double>beta, c_buffer, c_offset, c_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXher2k' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)

def trmm(queue, m, n, a, b, a_ld, b_ld, alpha = 1.0, right_side = False, lower_triangle = False, a_transp = False, unit_diagonal = False, a_offset = 0, b_offset = 0):
    """
    xTRMM: Triangular matrix-matrix multiplication
    """

    dtype = check_dtype([a, b], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    side = CLBlastSideRight if right_side else CLBlastSideLeft
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStrmm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_float>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtrmm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_double>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtrmm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtrmm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHtrmm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_half>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtrmm' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastStrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal, const size_t m, const size_t n, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, cl_mem b_buffer, const size_t b_offset, const size_t b_ld,cl_command_queue* queue, cl_event* event)

def trsm(queue, m, n, a, b, a_ld, b_ld, alpha = 1.0, right_side = False, lower_triangle = False, a_transp = False, unit_diagonal = False, a_offset = 0, b_offset = 0):
    """
    xTRSM: Solves a triangular system of equations
    """

    dtype = check_dtype([a, b], ["float32", "float64", "complex64", "complex128"])
    check_matrix(a, "a")
    check_matrix(b, "b")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    side = CLBlastSideRight if right_side else CLBlastSideLeft
    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastStrsm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_float>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDtrsm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_double>alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCtrsm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZtrsm(CLBlastLayoutRowMajor, side, triangle, a_transpose, diagonal, m, n, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXtrsm' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSaxpyBatched(const size_t n, const float *alphas, const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc, cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDaxpyBatched(const size_t n, const double *alphas, const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc, cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCaxpyBatched(const size_t n, const cl_float2 *alphas, const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc, cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZaxpyBatched(const size_t n, const cl_double2 *alphas, const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc, cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHaxpyBatched(const size_t n, const cl_half *alphas, const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc, cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc, const size_t batch_count,cl_command_queue* queue, cl_event* event)

def axpyBatched(queue, n, x, y, alphas, x_offsets, y_offsets, x_inc = 1, y_inc = 1):
    """
    xAXPYBATCHED: Batched version of AXPY
    """

    dtype = check_dtype([x, y], ["float32", "float64", "complex64", "complex128", "float16"])
    check_vector(x, "x")
    check_vector(y, "y")

    if len(x_offsets) != len(y_offsets) != len(alphas):
        raise RuntimeError("PyCLBlast: 'CLBlastXaxpyBatched' failed: length of batch-sized arguments x_offsets, y_offsets, alphas should be equal")
    batch_count = len(x_offsets)

    cdef size_t *x_offsets_c = <size_t *> PyMem_Malloc(batch_count * sizeof(size_t))
    for i in range(batch_count):
        x_offsets_c[i] = x_offsets[i]
    cdef size_t *y_offsets_c = <size_t *> PyMem_Malloc(batch_count * sizeof(size_t))
    for i in range(batch_count):
        y_offsets_c[i] = y_offsets[i]
    cdef void *alphas_c = <void *> PyMem_Malloc(batch_count * sizeof(dtype_size[dtype]))
    for i in range(batch_count):
        if dtype == np.dtype("float32"):
            (<cl_float*>alphas_c)[i] = <cl_float>alphas[i]
        elif dtype == np.dtype("float64"):
            (<cl_double*>alphas_c)[i] = <cl_double>alphas[i]
        elif dtype == np.dtype("complex64"):
            (<cl_float2*>alphas_c)[i] = <cl_float2>cl_float2(x=alphas[i].real,y=alphas[i].imag)
        elif dtype == np.dtype("complex128"):
            (<cl_double2*>alphas_c)[i] = <cl_double2>cl_double2(x=alphas[i].real,y=alphas[i].imag)
        elif dtype == np.dtype("float16"):
            (<cl_half*>alphas_c)[i] = <cl_half>alphas[i]

    cdef cl_mem x_buffer = <cl_mem><size_t>x.base_data.int_ptr
    cdef cl_mem y_buffer = <cl_mem><size_t>y.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSaxpyBatched(n, <cl_float*>alphas_c, x_buffer, x_offsets_c, x_inc, y_buffer, y_offsets_c, y_inc, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDaxpyBatched(n, <cl_double*>alphas_c, x_buffer, x_offsets_c, x_inc, y_buffer, y_offsets_c, y_inc, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCaxpyBatched(n, <cl_float2*>alphas_c, x_buffer, x_offsets_c, x_inc, y_buffer, y_offsets_c, y_inc, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZaxpyBatched(n, <cl_double2*>alphas_c, x_buffer, x_offsets_c, x_inc, y_buffer, y_offsets_c, y_inc, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHaxpyBatched(n, <cl_half*>alphas_c, x_buffer, x_offsets_c, x_inc, y_buffer, y_offsets_c, y_inc, batch_count, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    PyMem_Free(x_offsets_c)
    PyMem_Free(y_offsets_c)
    PyMem_Free(alphas_c)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXaxpyBatched' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const float *alphas, const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld, const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld, const float *betas, cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const double *alphas, const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld, const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld, const double *betas, cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_float2 *alphas, const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld, const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld, const cl_float2 *betas, cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_double2 *alphas, const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld, const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld, const cl_double2 *betas, cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_half *alphas, const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld, const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld, const cl_half *betas, cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld, const size_t batch_count,cl_command_queue* queue, cl_event* event)

def gemmBatched(queue, m, n, k, a, b, c, alphas, betas, a_ld, b_ld, c_ld, a_offsets, b_offsets, c_offsets, a_transp = False, b_transp = False):
    """
    xGEMMBATCHED: Batched version of GEMM
    """

    dtype = check_dtype([a, b, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    if len(a_offsets) != len(b_offsets) != len(c_offsets) != len(alphas) != len(betas):
        raise RuntimeError("PyCLBlast: 'CLBlastXgemmBatched' failed: length of batch-sized arguments a_offsets, b_offsets, c_offsets, alphas, betas should be equal")
    batch_count = len(a_offsets)

    cdef size_t *a_offsets_c = <size_t *> PyMem_Malloc(batch_count * sizeof(size_t))
    for i in range(batch_count):
        a_offsets_c[i] = a_offsets[i]
    cdef size_t *b_offsets_c = <size_t *> PyMem_Malloc(batch_count * sizeof(size_t))
    for i in range(batch_count):
        b_offsets_c[i] = b_offsets[i]
    cdef size_t *c_offsets_c = <size_t *> PyMem_Malloc(batch_count * sizeof(size_t))
    for i in range(batch_count):
        c_offsets_c[i] = c_offsets[i]
    cdef void *alphas_c = <void *> PyMem_Malloc(batch_count * sizeof(dtype_size[dtype]))
    for i in range(batch_count):
        if dtype == np.dtype("float32"):
            (<cl_float*>alphas_c)[i] = <cl_float>alphas[i]
        elif dtype == np.dtype("float64"):
            (<cl_double*>alphas_c)[i] = <cl_double>alphas[i]
        elif dtype == np.dtype("complex64"):
            (<cl_float2*>alphas_c)[i] = <cl_float2>cl_float2(x=alphas[i].real,y=alphas[i].imag)
        elif dtype == np.dtype("complex128"):
            (<cl_double2*>alphas_c)[i] = <cl_double2>cl_double2(x=alphas[i].real,y=alphas[i].imag)
        elif dtype == np.dtype("float16"):
            (<cl_half*>alphas_c)[i] = <cl_half>alphas[i]
    cdef void *betas_c = <void *> PyMem_Malloc(batch_count * sizeof(dtype_size[dtype]))
    for i in range(batch_count):
        if dtype == np.dtype("float32"):
            (<cl_float*>betas_c)[i] = <cl_float>betas[i]
        elif dtype == np.dtype("float64"):
            (<cl_double*>betas_c)[i] = <cl_double>betas[i]
        elif dtype == np.dtype("complex64"):
            (<cl_float2*>betas_c)[i] = <cl_float2>cl_float2(x=betas[i].real,y=betas[i].imag)
        elif dtype == np.dtype("complex128"):
            (<cl_double2*>betas_c)[i] = <cl_double2>cl_double2(x=betas[i].real,y=betas[i].imag)
        elif dtype == np.dtype("float16"):
            (<cl_half*>betas_c)[i] = <cl_half>betas[i]

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    b_transpose = CLBlastTransposeYes if b_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSgemmBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float*>alphas_c, a_buffer, a_offsets_c, a_ld, b_buffer, b_offsets_c, b_ld, <cl_float*>betas_c, c_buffer, c_offsets_c, c_ld, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDgemmBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double*>alphas_c, a_buffer, a_offsets_c, a_ld, b_buffer, b_offsets_c, b_ld, <cl_double*>betas_c, c_buffer, c_offsets_c, c_ld, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCgemmBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float2*>alphas_c, a_buffer, a_offsets_c, a_ld, b_buffer, b_offsets_c, b_ld, <cl_float2*>betas_c, c_buffer, c_offsets_c, c_ld, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgemmBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double2*>alphas_c, a_buffer, a_offsets_c, a_ld, b_buffer, b_offsets_c, b_ld, <cl_double2*>betas_c, c_buffer, c_offsets_c, c_ld, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHgemmBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_half*>alphas_c, a_buffer, a_offsets_c, a_ld, b_buffer, b_offsets_c, b_ld, <cl_half*>betas_c, c_buffer, c_offsets_c, c_ld, batch_count, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    PyMem_Free(a_offsets_c)
    PyMem_Free(b_offsets_c)
    PyMem_Free(c_offsets_c)
    PyMem_Free(alphas_c)
    PyMem_Free(betas_c)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgemmBatched' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# StridedBatched version of GEMM: SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
####################################################################################################

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const float alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const float beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastDgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const double alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const double beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastCgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_float2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const cl_float2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastZgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_double2 alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const cl_double2 beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride, const size_t batch_count,cl_command_queue* queue, cl_event* event)
    CLBlastStatusCode CLBlastHgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose, const size_t m, const size_t n, const size_t k, const cl_half alpha, const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride, const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const cl_half beta, cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride, const size_t batch_count,cl_command_queue* queue, cl_event* event)

def gemmStridedBatched(queue, m, n, k, batch_count, a, b, c, a_ld, b_ld, c_ld, a_stride, b_stride, c_stride, alpha = 1.0, beta = 0.0, a_transp = False, b_transp = False, a_offset = 0, b_offset = 0, c_offset = 0):
    """
    xGEMMSTRIDEDBATCHED: StridedBatched version of GEMM
    """

    dtype = check_dtype([a, b, c], ["float32", "float64", "complex64", "complex128", "float16"])
    check_matrix(a, "a")
    check_matrix(b, "b")
    check_matrix(c, "c")

    cdef cl_mem a_buffer = <cl_mem><size_t>a.base_data.int_ptr
    cdef cl_mem b_buffer = <cl_mem><size_t>b.base_data.int_ptr
    cdef cl_mem c_buffer = <cl_mem><size_t>c.base_data.int_ptr

    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr
    cdef cl_event event = NULL
    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo
    b_transpose = CLBlastTransposeYes if b_transp else CLBlastTransposeNo

    cdef CLBlastStatusCode err
    if dtype == np.dtype("float32"):
        err = CLBlastSgemmStridedBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float>alpha, a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, <cl_float>beta, c_buffer, c_offset, c_ld, c_stride, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float64"):
        err = CLBlastDgemmStridedBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double>alpha, a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, <cl_double>beta, c_buffer, c_offset, c_ld, c_stride, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex64"):
        err = CLBlastCgemmStridedBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_float2>cl_float2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, <cl_float2>cl_float2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, c_stride, batch_count, &command_queue, &event)
    elif dtype == np.dtype("complex128"):
        err = CLBlastZgemmStridedBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_double2>cl_double2(x=alpha.real,y=alpha.imag), a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, <cl_double2>cl_double2(x=beta.real,y=beta.imag), c_buffer, c_offset, c_ld, c_stride, batch_count, &command_queue, &event)
    elif dtype == np.dtype("float16"):
        err = CLBlastHgemmStridedBatched(CLBlastLayoutRowMajor, a_transpose, b_transpose, m, n, k, <cl_half>alpha, a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, <cl_half>beta, c_buffer, c_offset, c_ld, c_stride, batch_count, &command_queue, &event)
    else:
        raise ValueError("PyCLBlast: Unrecognized data-type '%s'" % dtype)

    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'CLBlastXgemmStridedBatched' failed: %s" % get_status_message(err))
    return cl.Event.from_int_ptr(<size_t>event)

####################################################################################################
# Overrides the parameters
####################################################################################################

cdef extern from "clblast_c.h":
    ctypedef struct _cl_device_id:
        pass
    ctypedef _cl_device_id* cl_device_id
    CLBlastStatusCode CLBlastOverrideParameters(const cl_device_id device, const char* kernel_name, const CLBlastPrecision precision, const size_t num_parameters, const char** parameters_names, const size_t* parameters_values)

def override_parameters(device, kernel_name, precision, parameters):
    """
    Override the current parameters for the given kernel, on this device, with this precision.
    """

    cdef cl_device_id device_id = <cl_device_id><size_t>device.int_ptr

    # read the parameters dictionary into names/values arrays, for use in CLBlastOverrideParameters
    cdef size_t n = len(parameters)
    cdef const char **parameter_names = <const char**> PyMem_Malloc(n * sizeof(char*))
    cdef size_t *parameter_values = <size_t*> PyMem_Malloc(n * sizeof(size_t))
    if not (parameter_names or parameter_values):
        raise MemoryError()
    for i, (k, v) in enumerate(parameters.items()):
        parameter_names[i] = strdup(k.encode('ascii'))
        parameter_values[i] = v

    # call the underlying API
    err = CLBlastOverrideParameters(device_id, kernel_name.encode('ascii'), precision, n, parameter_names, parameter_values)
    if err != CLBlastSuccess:
        raise RuntimeError("PyCLBlast: 'OverrideParameters' failed: %s" % get_status_message(err))

    # tidy up:
    PyMem_Free(parameter_names)
    PyMem_Free(parameter_values)

####################################################################################################
