
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

NL = "\n"
SEPARATOR = "####################################################################################################"


def to_np_dtype(flavour):
    if flavour.precision_name == "S":
        return "float32"
    if flavour.precision_name == "D":
        return "float64"
    if flavour.precision_name == "C":
        return "complex64"
    if flavour.precision_name == "Z":
        return "complex128"
    raise RuntimeError("Could not convert flavour '%s' to numpy" % flavour.precision_name)


def scalar_cython_conversion(scalar, flavour):
    if flavour.precision_name == "S":
        return "<cl_float>" + scalar
    if flavour.precision_name == "D":
        return "<cl_double>" + scalar
    if flavour.precision_name == "C":
        return "<cl_float2>cl_float2(x=" + scalar + ".real,y=" + scalar + ".imag)"
    if flavour.precision_name == "Z":
        return "<cl_double2>cl_double2(x=" + scalar + ".real,y=" + scalar + ".imag)"
    raise RuntimeError("Could not convert flavour '%s'" % flavour.precision_name)


def generate_pyx(routine):
    result = ""
    if routine.implemented and routine.plain_name() in ["swap", "gemm"]:  # TODO: Generalize

        result += SEPARATOR + NL
        result += "# " + routine.description + ": " + routine.short_names() + NL
        result += SEPARATOR + NL
        result += NL

        result += "cdef extern from \"clblast_c.h\":" + NL
        np_dtypes = []
        for flavour in routine.flavours:
            if flavour.precision_name in ["S", "D", "C", "Z"]:
                result += "    CLBlastStatusCode CLBlast" + flavour.name + routine.plain_name() + "("
                result += ", ".join(routine.arguments_def_c(flavour)) + ","
                result += "cl_command_queue* queue, cl_event* event)" + NL
                np_dtypes.append(to_np_dtype(flavour))
        result += "" + NL

        buffers = routine.inputs[:] + routine.outputs[:]
        result += "def " + routine.plain_name() + "(queue, "
        result += ", ".join(routine.arguments_python()) + "):" + NL
        result += "    dtype = check_dtype([" + ", ".join(buffers) + "], "
        result += "[" + ", ".join(['"%s"' % d for d in np_dtypes]) + "])" + NL
        for buf in buffers:
            if buf in routine.buffers_vector():
                result += "    check_vector("
            else:
                result += "    check_matrix("
            result += buf + ", \"" + buf + "\")" + NL
        result += "" + NL

        for buf in buffers:
            result += "    cdef cl_mem " + buf + "_buffer = <cl_mem><size_t>" + buf + ".base_data.int_ptr" + NL
        result += "" + NL

        result += "    cdef cl_command_queue command_queue = <cl_command_queue><size_t>queue.int_ptr" + NL
        result += "    cdef cl_event event = NULL" + NL

        for option in routine.options:
            if option == "a_transpose":
                result += "    a_transpose = CLBlastTransposeYes if a_transp else CLBlastTransposeNo" + NL
            if option == "b_transpose":
                result += "    b_transpose = CLBlastTransposeYes if b_transp else CLBlastTransposeNo" + NL
            if option == "ab_transpose":
                result += "    ab_transpose = CLBlastTransposeYes if ab_transp else CLBlastTransposeNo" + NL
            if option == "side":
                result += "    side = CLBlastSideRight if right_side else CLBlastSideLeft" + NL
            if option == "triangle":
                result += "    triangle = CLBlastTriangleLower if lower_triangle else CLBlastTriangleUpper" + NL
            if option == "diagonal":
                result += "    diagonal = CLBlastDiagonalUnit if unit_diagonal else CLBlastDiagonalNonUnit" + NL

        result += "" + NL
        result += "    cdef CLBlastStatusCode err" + NL
        if_prefix = ""
        for flavour in routine.flavours:
            if flavour.precision_name in ["S", "D", "C", "Z"]:
                np_dtype = to_np_dtype(flavour)
                argument_names = [x.
                                  replace("layout", "CLBlastLayoutRowMajor").
                                  replace("alpha", scalar_cython_conversion("alpha", flavour)).
                                  replace("beta", scalar_cython_conversion("beta", flavour))
                                  for x in routine.arguments()]
                result += "    " + if_prefix + "if dtype == np.dtype(\"" + np_dtype + "\"):" + NL
                result += "        err = CLBlast" + flavour.name + routine.plain_name()
                result += "(" + ", ".join(argument_names) + ", &command_queue, &event)" + NL
                if_prefix = "el"

        result += "    else:" + NL
        result += "        raise ValueError(\"PyCLBlast: Unrecognized data-type '%s'\" % dtype)" + NL
        result += "    if err != CLBlastSuccess:" + NL
        result += "        raise RuntimeError(\"PyCLBlast: 'CLBlastX" + routine.plain_name() + "' failed: %s\" % get_status_message(err))" + NL
        result += "    return cl.Event.from_int_ptr(<size_t>event)" + NL
        result += NL

    return result
