
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

NL = "\n"


def header():
    """Generates the header for the API documentation"""
    result = "CLBlast: API reference" + NL
    result += "================" + NL + NL + NL
    return result


def generate(routine):
    """Generates the API documentation for a given routine"""
    result = ""

    # Routine header
    result += "x" + routine.upper_name() + ": " + routine.description + NL
    result += "-------------" + NL + NL
    result += routine.details + NL + NL

    # Routine API
    result += "C++ API:" + NL
    result += "```" + NL
    result += routine.routine_header_cpp(12, "") + NL
    result += "```" + NL + NL
    result += "C API:" + NL
    result += "```" + NL
    for flavour in routine.flavours:
        result += routine.routine_header_c(flavour, 27, "") + NL
    result += "```" + NL + NL

    # Routine arguments
    result += "Arguments to " + routine.upper_name() + ":" + NL + NL
    for argument in routine.arguments_doc():
        result += "* " + argument + NL
    result += "* `cl_command_queue* queue`: "
    result += "Pointer to an OpenCL command queue associated with a context and device to execute the routine on." + NL
    result += "* `cl_event* event`: "
    result += "Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). "
    result += "This is an optional argument." + NL + NL

    # Routine requirements
    if len(routine.requirements_doc()) > 0:
        result += "Requirements for " + routine.upper_name() + ":" + NL + NL
        for requirement in routine.requirements_doc():
            result += "* " + requirement + NL
        result += NL

    # Routine footer
    result += NL + NL
    return result
