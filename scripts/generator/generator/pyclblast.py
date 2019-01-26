
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import os


NL = os.linesep
SEPARATOR = "####################################################################################################"


def to_np_dtype(flavour):
    return {
        "S": "float32",
        "D": "float64",
        "C": "complex64",
        "Z": "complex128",
        "H": "float16",
    }[flavour.precision_name]


def scalar_cython_conversion(scalar, flavour):
    scalar_type = flavour.alpha_cl if scalar == "alpha" else flavour.beta_cl
    if scalar_type == "float":
        return "<cl_float>" + scalar
    if scalar_type == "double":
        return "<cl_double>" + scalar
    if scalar_type in ["cl_float2", "float2"]:
        return "<cl_float2>cl_float2(x=" + scalar + ".real,y=" + scalar + ".imag)"
    if scalar_type in ["cl_double2", "double2"]:
        return "<cl_double2>cl_double2(x=" + scalar + ".real,y=" + scalar + ".imag)"
    if scalar_type in ["cl_half", "half"]:
        return "<cl_half>" + scalar
    raise RuntimeError("Could not convert flavour '%s:%s'" % (flavour.precision_name, scalar_type))


def generate_pyx(routine):
    result = ""
    if routine.implemented and routine.plain_name() and routine.level in ["1", "2a", "2b", "3"]:
        indent = "    "

        result += SEPARATOR + NL
        result += "# " + routine.description + ": " + routine.short_names() + NL
        result += SEPARATOR + NL
        result += NL

        # Reference C definition
        result += "cdef extern from \"clblast_c.h\":" + NL
        np_dtypes = []
        for flavour in routine.flavours:
            if flavour.precision_name in ["S", "D", "C", "Z", "H"]:
                result += indent + "CLBlastStatusCode CLBlast" + flavour.name + routine.plain_name() + "("
                result += ", ".join(routine.arguments_def_c(flavour)) + ","
                result += "cl_command_queue* queue, cl_event* event)" + NL
                np_dtypes.append(to_np_dtype(flavour))
        result += "" + NL

        # Function definition
        buffers = routine.inputs[:] + routine.outputs[:]
        result += "def " + routine.plain_name() + "(queue, "
        result += ", ".join(routine.arguments_python()) + "):" + NL

        # Documentation
        result += indent + "\"\"\"" + NL
        result += indent + "x" + routine.upper_name() + ": " + routine.description + NL
        result += indent + "\"\"\"" + NL
        result += NL

        # Data types and checks
        result += indent + "dtype = check_dtype([" + ", ".join(buffers) + "], "
        result += "[" + ", ".join(['"%s"' % d for d in np_dtypes]) + "])" + NL
        for buf in buffers:
            if buf in routine.buffers_vector():
                result += indent + "check_vector("
            else:
                result += indent + "check_matrix("
            result += buf + ", \"" + buf + "\")" + NL
        result += NL

        # Buffer transformation
        for buf in buffers:
            result += indent + "cdef cl_mem " + buf + "_buffer = <cl_mem><size_t>" + buf + ".base_data.int_ptr" + NL
        result += NL

        result += indent + "cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr" + NL
        result += indent + "cdef cl_event event = NULL" + NL

        for option in routine.options:
            if option == "a_transpose":
                result += indent + "a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo" + NL
            if option == "b_transpose":
                result += indent + "b_transpose = CLBlastTransposeYes if b_transp else CLBlastTransposeNo" + NL
            if option == "ab_transpose":
                result += indent + "ab_transpose = CLBlastTransposeYes if ab_transp else CLBlastTransposeNo" + NL
            if option == "side":
                result += indent + "side = CLBlastSideRight if right_side else CLBlastSideLeft" + NL
            if option == "triangle":
                result += indent + "triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper" + NL
            if option == "diagonal":
                result += indent + "diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit" + NL

        result += "" + NL
        result += indent + "cdef CLBlastStatusCode err" + NL
        if_prefix = ""
        for flavour in routine.flavours:
            if flavour.precision_name in ["S", "D", "C", "Z", "H"]:
                np_dtype = to_np_dtype(flavour)
                argument_names = [x.
                                  replace("layout", "CLBlastLayoutRowMajor").
                                  replace("alpha", scalar_cython_conversion("alpha", flavour)).
                                  replace("beta", scalar_cython_conversion("beta", flavour))
                                  for x in routine.arguments()]
                result += indent + if_prefix + "if dtype == np.dtype(\"" + np_dtype + "\"):" + NL
                result += indent + indent + "err = CLBlast" + flavour.name + routine.plain_name()
                result += "(" + ", ".join(argument_names) + ", &command_queue, &event)" + NL
                if_prefix = "el"

        result += indent + "else:" + NL
        result += indent + indent + "raise ValueError(\"PyCLBlast: Unrecognized data-type '%s'\" % dtype)" + NL
        result += indent + "if err != CLBlastSuccess:" + NL
        result += indent + indent + "raise RuntimeError(\"PyCLBlast: 'CLBlastX" + routine.plain_name() + "' failed: %s\" % get_status_message(err))" + NL
        result += indent + "return cl.Event.from_int_ptr(<size_t>event)" + NL
        result += NL

    return result
