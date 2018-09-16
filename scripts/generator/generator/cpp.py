
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import generator.datatype as datatype
import generator.convert as convert


NL = "\n"
SEPARATOR = "// ================================================================================================="

# Separators for the BLAS levels
LEVEL_SEPARATORS = [
    NL + SEPARATOR + NL + "// BLAS level-1 (vector-vector) routines" + NL + SEPARATOR,
    NL + SEPARATOR + NL + "// BLAS level-2 (matrix-vector) routines" + NL + SEPARATOR,
    NL + SEPARATOR + NL + "// BLAS level-3 (matrix-matrix) routines" + NL + SEPARATOR,
    NL + SEPARATOR + NL + "// Extra non-BLAS routines (level-X)" + NL + SEPARATOR
]

# Names of the level sub-folders
LEVEL_NAMES = ["1", "2", "3", "x"]

# Main header/footer for source files
FOOTER = NL + SEPARATOR + NL
HEADER = NL + SEPARATOR + """
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
""" + SEPARATOR + NL


def clblast_h(routine, cuda=False):
    """The C++ API header (.h)"""
    result = NL + "// " + routine.description + ": " + routine.short_names() + NL
    result += routine.routine_header_cpp(12, " = nullptr", cuda) + ";" + NL
    return result


def clblast_cc(routine, cuda=False):
    """The C++ API implementation (.cpp)"""
    indent1 = " " * (15 + routine.length())
    result = NL + "// " + routine.description + ": " + routine.short_names() + NL
    if routine.implemented:
        result += routine.routine_header_cpp(12, "", cuda, implementation=True) + " {" + NL
        result += "  try {" + NL
        if cuda:
            result += "    const auto context_cpp = Context(context);" + NL
            result += "    const auto device_cpp = Device(device);" + NL
            result += "    auto queue_cpp = Queue(context_cpp, device_cpp);" + NL
        else:
            result += "    auto queue_cpp = Queue(*queue);" + NL
        event = "nullptr" if cuda else "event"
        result += "    auto routine = X" + routine.plain_name() + "<" + routine.template.template + ">(queue_cpp, " + event + ");" + NL
        if routine.batched == 1:
            result += "    " + (NL + "    ").join(routine.batched_transform_to_cpp()) + NL
        if routine.temp_buffer:
            null = "0" if cuda else "nullptr"
            result += "    const auto temp_buffer_provided = temp_buffer != " + null + ";\n"
            result += "    auto temp_buffer_cpp = temp_buffer_provided ? Buffer<T>(temp_buffer) : Buffer<T>(" + null + ");\n"
        result += "    routine.Do" + routine.capitalized_name() + "("
        result += ("," + NL + indent1).join([a for a in routine.arguments_clcudaapi()])
        if routine.temp_buffer:
            result += ",\n" + indent1 + "temp_buffer_cpp, temp_buffer_provided"
        result += ");" + NL
        result += "    return StatusCode::kSuccess;" + NL
        result += "  } catch (...) { return DispatchException(); }" + NL
    else:
        result += routine.routine_header_type_cpp(12, cuda) + " {" + NL
        result += "  return StatusCode::kNotImplemented;" + NL
    result += "}" + NL
    for flavour in routine.flavours:
        indent2 = " " * (34 + routine.length() + len(flavour.template))
        result += "template StatusCode PUBLIC_API " + routine.capitalized_name() + "<" + flavour.template + ">("
        arguments = routine.arguments_type(flavour)
        if cuda:
            arguments = [a.replace("cl_mem", "CUdeviceptr") for a in arguments]
        result += ("," + NL + indent2).join([a for a in arguments])
        result += "," + NL + indent2
        if cuda:
            result += "const CUcontext, const CUdevice"
            if routine.temp_buffer:
                result += ", CUdeviceptr"
        else:
            result += "cl_command_queue*, cl_event*"
            if routine.temp_buffer:
                result += ", cl_mem"
        result += ");" + NL
    return result


def clblast_c_h(routine):
    """The C API header (.h)"""
    result = NL + "// " + routine.description + ": " + routine.short_names() + NL
    for flavour in routine.flavours:
        result += routine.routine_header_c(flavour, 38, " PUBLIC_API") + ";" + NL
    return result


def clblast_c_cc(routine):
    """The C API implementation (.cpp)"""
    result = NL + "// " + routine.name.upper() + NL
    for flavour in routine.flavours:
        template = "<" + flavour.template + ">" if routine.no_scalars() else ""
        indent = " " * (16 + routine.length() + len(template))
        result += routine.routine_header_c(flavour, 27, "") + " {" + NL
        if routine.batched == 1:
            result += "  " + (NL + "  ").join(routine.batched_transform_to_complex(flavour)) + NL
        result += "  try {" + NL
        result += "    return static_cast<CLBlastStatusCode>(" + NL
        result += "      clblast::" + routine.capitalized_name() + template + "("
        result += ("," + NL + indent).join([a for a in routine.arguments_cast(flavour, indent)])
        result += "," + NL + indent + "queue, event)" + NL
        result += "    );" + NL
        result += "  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }" + NL
        result += "}" + NL
    return result


def clblast_netlib_c_h(routine):
    """The Netlib CBLAS API header (.h)"""
    result = NL + "// " + routine.description + ": " + routine.short_names() + NL
    for flavour in routine.flavours:
        if flavour.precision_name in ["S", "D", "C", "Z"]:
            result += routine.routine_header_netlib(flavour, 20, " PUBLIC_API") + ";" + NL
    return result


def clblast_netlib_c_cc(routine):
    """The Netlib CBLAS API implementation (.cpp)"""
    result = NL + "// " + routine.name.upper() + NL
    for flavour in routine.flavours:

        # There is a version available in CBLAS
        if flavour.precision_name in ["S", "D", "C", "Z"]:
            template = "<" + flavour.template + ">" if routine.no_scalars() else ""
            name_postfix = "_sub" if routine.name in routine.routines_scalar_no_return() else ""
            indent = " " * (21 + routine.length() + len(template))
            result += routine.routine_header_netlib(flavour, 9, "") + " {" + NL

            # Initialize OpenCL
            result += "  OPTIONAL_STATIC auto device = get_device();" + NL
            result += "  OPTIONAL_STATIC auto context = clblast::Context(device);" + NL
            result += "  auto queue = clblast::Queue(context, device);" + NL

            # Set alpha and beta
            result += "".join("  " + s + NL for s in routine.scalar_create_cpp(flavour))

            # Copy data structures to the device
            for i, name in enumerate(routine.inputs + routine.outputs):
                result += "  " + routine.set_size(name, routine.buffer_sizes[i]) + NL
            for i, name in enumerate(routine.inputs + routine.outputs):
                buffer_type = routine.get_buffer_type(name, flavour)
                result += "  " + routine.create_buffer(name, buffer_type) + NL
                if name in routine.scalar_buffers_second_non_pointer():
                    result += "  " + buffer_type + " " + name + "_vec[1]; " + name + "_vec[0] = " + name + ";" + NL
            for name in routine.inputs + routine.outputs:
                if name not in routine.scalar_buffers_first():
                    prefix = "" if name in routine.outputs else "const "
                    buffer_type = routine.get_buffer_type(name, flavour)
                    result += "  " + routine.write_buffer(name, prefix + buffer_type) + NL

            # The function call
            result += "  auto queue_cl = queue();" + NL
            result += "  auto s = clblast::" + routine.name.capitalize() + template + "("
            result += ("," + NL + indent).join([a for a in routine.arguments_netlib(flavour, indent)])
            result += "," + NL + indent + "&queue_cl);" + NL

            # Error handling
            result += "  if (s != clblast::StatusCode::kSuccess) {" + NL
            result += "    throw std::runtime_error(\"CLBlast returned with error code \" + clblast::ToString(s));" + NL
            result += "  }" + NL

            # Copy back and clean-up
            for name in routine.outputs:
                if name in routine.scalar_buffers_first() and routine.name not in routine.routines_scalar_no_return():
                    buffer_type = routine.get_buffer_type(name, flavour)
                    result += "  " + buffer_type + " " + name + "[" + name + "_size];" + NL
            for name in routine.outputs:
                buffer_type = routine.get_buffer_type(name, flavour)
                result += "  " + routine.read_buffer(name, buffer_type) + NL
            for name in routine.outputs:
                if name in routine.scalar_buffers_first() and routine.name not in routine.routines_scalar_no_return():
                    result += "  return " + name + "[0]"
                    if flavour.buffer_type in ["float2", "double2"]:
                        if name not in routine.index_buffers():
                            result += ".real()"
                    result += ";" + NL
            result += "}" + NL
    return result


def wrapper_clblas(routine):
    """The wrapper to the reference clBLAS routines (for performance/correctness testing)"""
    result = ""
    if routine.has_tests:
        result += NL + "// Forwards the clBLAS calls for %s" % routine.short_names_tested() + NL
        if routine.no_scalars():
            result += routine.routine_header_wrapper_clblas(routine.template, True, 21) + ";" + NL
        for flavour in routine.flavours:
            result += routine.routine_header_wrapper_clblas(flavour, False, 21) + " {" + NL

            # There is a version available in clBLAS
            if flavour.precision_name in ["S", "D", "C", "Z"]:
                indent = " " * (17 + routine.length())
                arguments = routine.arguments_wrapper_clblas(flavour)
                if routine.scratch:
                    result += "  auto queue = Queue(queues[0]);" + NL
                    result += "  auto context = queue.GetContext();" + NL
                    result += "  auto scratch_buffer = Buffer<" + flavour.template + ">"
                    result += "(context, " + routine.scratch + ");" + NL
                    arguments += ["scratch_buffer()"]
                result += "  return clblas" + flavour.name + routine.name + "("
                result += ("," + NL + indent).join([a for a in arguments])
                result += "," + NL + indent + "num_queues, queues, num_wait_events, wait_events, events);"

            # There is no clBLAS available, forward the call to one of the available functions
            else:  # Half-precision
                indent = " " * (24 + routine.length())

                # Convert to float (note: also integer buffers are stored as half/float)
                for buf in routine.inputs + routine.outputs:
                    result += "  auto " + buf + "_buffer_bis = HalfToFloatBuffer(" + buf + "_buffer, queues[0]);" + NL

                # Call the float routine
                result += "  auto status = clblasX" + routine.name + "("
                result += ("," + NL + indent).join([a for a in routine.arguments_half()])
                result += "," + NL + indent + "num_queues, queues, num_wait_events, wait_events, events);"
                result += NL

                # Convert back to half
                for buf in routine.outputs:
                    result += "  FloatToHalfBuffer(" + buf + "_buffer, " + buf + "_buffer_bis, queues[0]);" + NL
                result += "  return status;"

            # Complete
            result += NL + "}" + NL
    return result


def wrapper_cblas(routine):
    """The wrapper to the reference CBLAS routines (for performance/correctness testing)"""
    result = ""
    if routine.has_tests:
        result += NL + "// Forwards the Netlib BLAS calls for %s" % routine.short_names_tested() + NL
        for flavour in routine.flavours:
            result += routine.routine_header_wrapper_cblas(flavour, 12) + " {" + NL

            # There is a version available in CBLAS
            if flavour.precision_name in ["S", "D", "C", "Z"]:
                indent = " " * (10 + routine.length())
                arguments = routine.arguments_wrapper_cblas(flavour)

                # Complex scalars
                for scalar in routine.scalars:
                    if flavour.is_complex(scalar):
                        result += "  const auto " + scalar + "_array = std::vector<" + flavour.buffer_type[:-1] + ">"
                        result += "{" + scalar + ".real(), " + scalar + ".imag()};" + NL

                # Special case for scalar outputs
                assignment = ""
                postfix, postpostfix = "", ""
                end_of_line = ""
                extra_argument = ""
                for output_buffer in routine.outputs:
                    if output_buffer in routine.scalar_buffers_first():
                        if flavour in [datatype.C, datatype.Z]:
                            postfix += "_sub"
                            indent += "    "
                            extra_argument += "," + NL + indent
                            extra_argument += "reinterpret_cast<return_pointer_" + flavour.buffer_type[:-1] + ">"
                            extra_argument += "(&" + output_buffer + "_buffer[" + output_buffer + "_offset])"
                        elif output_buffer in routine.index_buffers():
                            assignment = "reinterpret_cast<int*>(&" + output_buffer + "_buffer[0])[" + output_buffer + "_offset] = static_cast<int>("
                            postpostfix = ")"
                            indent += " " * (len(assignment) + 1)
                        else:
                            assignment = output_buffer + "_buffer[" + output_buffer + "_offset]"
                            if flavour.name in ["Sc", "Dz"]:
                                assignment += ".real("
                                end_of_line += ")"
                            else:
                                assignment += " = "
                            indent += " " * len(assignment)

                result += "  " + assignment + "cblas_" + flavour.name.lower() + routine.name + postfix + "("
                result += ("," + NL + indent).join([a for a in arguments])
                result += extra_argument + end_of_line + ")" + postpostfix + ";" + NL

            # There is no CBLAS available, forward the call to one of the available functions
            else:  # Half-precision
                indent = " " * (9 + routine.length())

                # Convert to float (note: also integer buffers are stored as half/float)
                for buf in routine.inputs + routine.outputs:
                    result += "  auto " + buf + "_buffer_bis = HalfToFloatBuffer(" + buf + "_buffer);" + NL

                # Call the float routine
                result += "  cblasX" + routine.name + "("
                result += ("," + NL + indent).join([a for a in routine.arguments_half()])
                result += ");" + NL

                # Convert back to half
                for buf in routine.outputs:
                    result += "  FloatToHalfBuffer(" + buf + "_buffer, " + buf + "_buffer_bis);" + NL

            # Complete
            result += "}" + NL
    return result


def wrapper_cublas(routine):
    """The wrapper to the reference cuBLAS routines (for performance/correctness testing)"""
    result = ""
    if routine.has_tests:
        result += NL + "// Forwards the cuBLAS calls for %s" % routine.short_names_tested() + NL
        if routine.no_scalars():
            result += routine.routine_header_wrapper_cublas(routine.template, True, 23) + ";" + NL
        for flavour in routine.flavours:
            result += routine.routine_header_wrapper_cublas(flavour, False, 23) + " {" + NL

            # There is a version available in cuBLAS
            if flavour.precision_name in ["S", "D", "C", "Z"]:
                indent = " " * (24 + routine.length())
                arguments = routine.arguments_wrapper_cublas(flavour)

                # Handles row-major
                if routine.has_layout():
                    result += "  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }" + NL

                # Complex scalars
                for scalar in routine.scalars:
                    if flavour.is_complex(scalar):
                        cuda_complex = "cuDoubleComplex" if flavour.precision_name == "Z" else "cuComplex"
                        result += "  " + cuda_complex + " " + scalar + "_cuda;" + NL
                        result += "  " + scalar + "_cuda.x = " + scalar + ".real();" + NL
                        result += "  " + scalar + "_cuda.y = " + scalar + ".imag();" + NL

                # Calls the cuBLAS routine
                result += "  auto status = cublas" + flavour.name_cublas() + routine.name + "(handle, "
                result += ("," + NL + indent).join([a for a in arguments]) + ");" + NL
                result += "  cudaDeviceSynchronize();" + NL
                result += "  return status;"

            # There is no cuBLAS available, forward the call to one of the available functions
            else:  # Half-precision
                result += "  return CUBLAS_STATUS_NOT_SUPPORTED;"
            #     indent = " " * (24 + routine.length())

            #     # Convert to float (note: also integer buffers are stored as half/float)
            #     for buf in routine.inputs + routine.outputs:
            #         result += "  auto " + buf + "_buffer_bis = HalfToFloatBuffer(" + buf + "_buffer, queues[0]);" + NL

            #     # Call the float routine
            #     result += "  return cublasX" + routine.name + "(handle,"
            #     result += ("," + NL + indent).join([a for a in routine.arguments_half()]) + ");" + NL
            #     result += "  cudaDeviceSynchronize();" + NL
            #     result += "  return status;"

            #     # Convert back to half
            #     for buf in routine.outputs:
            #         result += "  FloatToHalfBuffer(" + buf + "_buffer, " + buf + "_buffer_bis, queues[0]);" + NL
            #     result += "  return status;"

            # Complete
            result += NL + "}" + NL
    return result


def performance_test(routine, level_string):
    """Generates the body of a performance test for a specific routine"""
    result = ""
    result += "#include \"test/performance/client.hpp\"" + NL
    result += "#include \"test/routines/level" + level_string + "/x" + routine.lowercase_name() + ".hpp\"" + NL + NL
    result += "// Main function (not within the clblast namespace)" + NL
    result += "int main(int argc, char *argv[]) {" + NL
    result += "  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);" + NL
    default = convert.precision_to_full_name(routine.flavours[0].precision_name)
    result += "  switch(clblast::GetPrecision(command_line_args, clblast::Precision::k" + default + ")) {" + NL
    for precision in ["H", "S", "D", "C", "Z"]:
        result += "    case clblast::Precision::k" + convert.precision_to_full_name(precision) + ":"
        found = False
        for flavour in routine.flavours:
            if flavour.precision_name == precision:
                extra_template_argument = "0, " if routine.name == "gemm" and routine.batched == 0 else ""
                result += NL + "      clblast::RunClient<clblast::TestX" + routine.plain_name()
                result += flavour.test_template(extra_template_argument)
                result += ">(argc, argv); break;" + NL
                found = True
        if not found:
            result += " throw std::runtime_error(\"Unsupported precision mode\");" + NL
    result += "  }" + NL
    result += "  return 0;" + NL
    result += "}" + NL
    return result


def correctness_test(routine, level_string):
    """Generates the body of a correctness test for a specific routine"""
    result = ""
    result += "#include \"test/correctness/testblas.hpp\"" + NL
    result += "#include \"test/routines/level" + level_string + "/x" + routine.lowercase_name() + ".hpp\"" + NL + NL
    result += "// Main function (not within the clblast namespace)" + NL
    result += "int main(int argc, char *argv[]) {" + NL
    result += "  auto errors = size_t{0};" + NL
    not_first = "false"
    extra_template_arguments = ["1, ", "2, "] if routine.name == "gemm" and routine.batched == 0 else [""]
    for extra_template_argument in extra_template_arguments:
        for flavour in routine.flavours:
            result += "  errors += clblast::RunTests<clblast::TestX" + routine.plain_name()
            result += flavour.test_template(extra_template_argument)
            result += ">(argc, argv, " + not_first + ", \"" + flavour.name + routine.upper_name() + "\");" + NL
            not_first = "true"
    result += "  if (errors > 0) { return 1; } else { return 0; }" + NL
    result += "}" + NL
    return result
