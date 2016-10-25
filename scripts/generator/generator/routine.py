
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from itertools import chain

import generator.convert as convert


class Routine:
    """Class holding routine-specific information (e.g. name, which arguments, which precisions)"""
    def __init__(self, implemented, has_tests, level, name, template, flavours, sizes, options,
                 inputs, outputs, buffer_sizes, scalars, scratch,
                 description, details, requirements):
        self.implemented = implemented
        self.has_tests = has_tests
        self.level = level
        self.name = name
        self.template = template
        self.flavours = flavours
        self.sizes = sizes
        self.options = options
        self.inputs = inputs
        self.outputs = outputs
        self.buffer_sizes = buffer_sizes
        self.scalars = scalars
        self.scratch = scratch  # Scratch buffer (e.g. for xDOT)
        self.description = description
        self.details = details
        self.requirements = requirements

    @staticmethod
    def scalar_buffers_first():
        """List of scalar buffers"""
        return ["dot", "nrm2", "asum", "sum", "imax", "imin"]

    @staticmethod
    def scalar_buffers_second():
        """List of scalar buffers"""
        return ["sa", "sb", "sc", "ss", "sd1", "sd2", "sx1", "sy1", "sparam"]

    @staticmethod
    def other_scalars():
        """List of scalars other than alpha and beta"""
        return ["cos", "sin"]

    @staticmethod
    def index_buffers():
        """List of buffers with unsigned int type"""
        return ["imax", "imin"]

    @staticmethod
    def postfix(name):
        """Retrieves the postfix for a buffer"""
        return "inc" if (name in ["x", "y"]) else "ld"

    @staticmethod
    def buffers_vector():
        """Distinguish between vectors and matrices"""
        return ["x", "y"]

    @staticmethod
    def buffers_matrix():
        """Distinguish between vectors and matrices"""
        return ["a", "b", "c", "ap"]

    @staticmethod
    def set_size(name, size):
        """Sets the size of a buffer"""
        return "const auto " + name + "_size = " + size + ";"

    @staticmethod
    def create_buffer(name, template):
        """Creates a new CLCudaAPI buffer"""
        return "auto " + name + "_buffer = clblast::Buffer<" + template + ">(context, " + name + "_size);"

    @staticmethod
    def write_buffer(name, template):
        """Writes to a CLCudaAPI buffer"""
        data_structure = "reinterpret_cast<" + template + "*>(" + name + ")"
        return name + "_buffer.Write(queue, " + name + "_size, " + data_structure + ");"

    @staticmethod
    def read_buffer(name, template):
        """Reads from a CLCudaAPI buffer"""
        data_structure = "reinterpret_cast<" + template + "*>(" + name + ")"
        return name + "_buffer.Read(queue, " + name + "_size, " + data_structure + ");"

    def non_index_inputs(self):
        """Lists of input/output buffers not index (integer)"""
        buffers = self.inputs[:]  # make a copy
        for i in self.index_buffers():
            if i in buffers:
                buffers.remove(i)
        return buffers

    def non_index_outputs(self):
        """Lists of input/output buffers not index (integer)"""
        buffers = self.outputs[:]  # make a copy
        for i in self.index_buffers():
            if i in buffers:
                buffers.remove(i)
        return buffers

    def buffers_without_ld_inc(self):
        """List of buffers without 'inc' or 'ld'"""
        return self.scalar_buffers_first() + self.scalar_buffers_second() + ["ap"]

    def length(self):
        """Retrieves the number of characters in the routine's name"""
        return len(self.name)

    def no_scalars(self):
        """Determines whether or not this routine has scalar arguments (alpha/beta)"""
        return self.scalars == []

    def short_names(self):
        """Returns the upper-case names of these routines (all flavours)"""
        return "/".join([f.name + self.name.upper() for f in self.flavours])

    def short_names_tested(self):
        """As above, but excludes some"""
        names = [f.name + self.name.upper() for f in self.flavours]
        if "H" + self.name.upper() in names:
            names.remove("H" + self.name.upper())
        return "/".join(names)

    def buffers_first(self):
        """Determines which buffers go first (between alpha and beta) and which ones go after"""
        if self.level == "2b":
            return ["x", "y"]
        return ["ap", "a", "b", "x"]

    def buffers_second(self):
        if self.level == "2b":
            return ["ap", "a", "b", "c"]
        return ["y", "c"]

    def buffer(self, name):
        """Retrieves a variable name for a specific input/output vector/matrix (e.g. 'x')"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer"]
            b = [name + "_offset"]
            c = [name + "_" + self.postfix(name)] if (name not in self.buffers_without_ld_inc()) else []
            return [", ".join(a + b + c)]
        return []

    def buffer_bis(self, name):
        """As above but with a '_bis' suffix for the buffer name"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer_bis"]
            b = [name + "_offset"]
            c = [name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + b + c)]
        return []

    def buffer_zero_offset(self, name):
        """As above, but with an offset value of zero"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer()"]
            b = ["0"]
            c = [name + "_" + self.postfix(name)] if (name not in self.buffers_without_ld_inc()) else []
            return [", ".join(a + b + c)]
        return []

    def buffer_def(self, name):
        """As above but with data-types"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            a = [prefix + "cl_mem " + name + "_buffer"]
            b = ["const size_t " + name + "_offset"]
            c = ["const size_t " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + b + c)]
        return []

    def buffer_def_wrapper_cl(self, name, flavour):
        """As above but with data-types"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            a = [prefix + "Buffer<" + flavour.buffer_type + ">& " + name + "_buffer"]
            b = ["const size_t " + name + "_offset"]
            c = ["const size_t " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + b + c)]
        return []

    def buffer_def_vector(self, name, flavour):
        """As above but as vectors"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            a = [prefix + "std::vector<" + flavour.buffer_type + ">& " + name + "_buffer"]
            b = ["const size_t " + name + "_offset"]
            c = ["const size_t " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + b + c)]
        return []

    def buffer_def_pointer(self, name, flavour):
        """As above but as plain C pointer"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            data_type = "void" if flavour.is_non_standard() else flavour.buffer_type
            a = [prefix + data_type + "* " + name + ""]
            c = ["const int " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + c)]
        return []

    def buffer_clcudaapi(self, name):
        """As above but with CLCudaAPI buffers"""
        if name in self.inputs or name in self.outputs:
            buffer_type = "unsigned int" if (name in self.index_buffers()) else self.template.buffer_type
            a = ["Buffer<" + buffer_type + ">(" + name + "_buffer)"]
            b = [name + "_offset"]
            c = [name + "_" + self.postfix(name)] if (name not in self.buffers_without_ld_inc()) else []
            return [", ".join(a + b + c)]
        return []

    def buffer_wrapper_clblas(self, name):
        """As above but with a static cast for clBLAS wrapper"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer()"]
            b = [name + "_offset"]
            c = []
            if name in ["x", "y"]:
                c = ["static_cast<int>(" + name + "_" + self.postfix(name) + ")"]
            elif name in ["a", "b", "c"]:
                c = [name + "_" + self.postfix(name)]
            return [", ".join(a + b + c)]
        return []

    def buffer_wrapper_cblas(self, name, flavour):
        """As above but with a static cast for CBLAS wrapper"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            if name == "sy1":
                a = [name + "_buffer[" + name + "_offset]"]
            elif flavour.precision_name in ["C", "Z"]:
                a = ["reinterpret_cast<" + prefix + flavour.buffer_type[:-1] + "*>" +
                     "(&" + name + "_buffer[" + name + "_offset])"]
            else:
                a = ["&" + name + "_buffer[" + name + "_offset]"]
            c = []
            if name in ["x", "y"]:
                c = ["static_cast<int>(" + name + "_" + self.postfix(name) + ")"]
            elif name in ["a", "b", "c"]:
                c = [name + "_" + self.postfix(name)]
            return [", ".join(a + c)]
        return []

    def buffer_type(self, name):
        """As above, but only data-types"""
        prefix = "const " if (name in self.inputs) else ""
        if (name in self.inputs) or (name in self.outputs):
            a = [prefix + "cl_mem"]
            b = ["const size_t"]
            c = ["const size_t"] if (name not in self.buffers_without_ld_inc()) else []
            return [", ".join(a + b + c)]
        return []

    def buffer_doc(self, name):
        """Retrieves the documentation of the buffers"""
        prefix = "const " if (name in self.inputs) else ""
        inout = "input" if (name in self.inputs) else "output"
        if (name in self.inputs) or (name in self.outputs):
            math_name = name.upper() + " matrix" if (name in self.buffers_matrix()) else name + " vector"
            inc_ld_description = "Leading dimension " if (name in self.buffers_matrix()) else "Stride/increment "
            a = ["`" + prefix + "cl_mem " + name + "_buffer`: OpenCL buffer to store the " + inout + " " + math_name + "."]
            b = ["`const size_t " + name + "_offset`: The offset in elements from the start of the " + inout + " " + math_name + "."]
            if name not in self.buffers_without_ld_inc():
                c = ["`const size_t " + name + "_" + self.postfix(name) + "`: " +
                     inc_ld_description + "of the " + inout + " " + math_name + ". This value must be greater than 0."]
            else:
                c = []
            return a + b + c
        return []

    def scalar(self, name):
        """Retrieves the name of a scalar (alpha/beta)"""
        if name in self.scalars:
            return [name]
        return []

    def scalar_cpp(self, name):
        """As above, but with _cpp as a suffix"""
        if name in self.scalars:
            return [name + "_cpp"]
        return []

    def scalar_half_to_float(self, name):
        """As above, but converts from float to half"""
        if name in self.scalars:
            return ["HalfToFloat(" + name + ")"]
        return []

    def scalar_use(self, name, flavour):
        """Retrieves the use of a scalar (alpha/beta)"""
        if name in self.scalars:
            if name == "alpha":
                return [flavour.use_alpha()]
            elif name == "beta":
                return [flavour.use_beta()]
            return [name]
        return []

    def scalar_use_wrapper(self, name, flavour):
        """As above, but for the clBLAS wrapper"""
        if name in self.scalars:
            if name == "alpha":
                return [flavour.use_alpha_opencl()]
            elif name == "beta":
                return [flavour.use_beta_opencl()]
            return [name]
        return []

    def scalar_use_wrapper_cblas(self, name, flavour):
        """As above, but for the CBLAS wrapper"""
        if name in self.scalars:
            if flavour.is_complex(name):
                return [name + "_array.data()"]
            return [name]
        return []

    def scalar_def(self, name, flavour):
        """Retrieves the definition of a scalar (alpha/beta)"""
        if name in self.scalars:
            if name == "alpha":
                return ["const " + flavour.alpha_cl + " " + name]
            return ["const " + flavour.beta_cl + " " + name]
        return []

    def scalar_def_plain(self, name, flavour):
        """As above, but without 'cl_' prefix"""
        if name in self.scalars:
            if name == "alpha":
                return ["const " + flavour.alpha_cpp + " " + name]
            return ["const " + flavour.beta_cpp + " " + name]
        return []

    def scalar_def_void(self, name, flavour):
        """Retrieves the definition of a scalar (alpha/beta) but make it a void pointer in case of non-standard types"""
        if name in self.scalars:
            if name == "alpha":
                data_type = "void*" if flavour.is_complex("alpha") else flavour.alpha_cpp
                return ["const " + data_type + " " + name]
            data_type = "void*" if flavour.is_complex("beta") else flavour.beta_cpp
            return ["const " + data_type + " " + name]
        return []

    def scalar_type(self, name, flavour):
        """Retrieves the type of a scalar (alpha/beta)"""
        if name in self.scalars:
            if name == "alpha":
                return ["const " + flavour.alpha_cpp]
            return ["const " + flavour.beta_cpp]
        return []

    def scalar_doc(self, name):
        """Retrieves the documentation of a scalar"""
        if name in self.scalars:
            if name == "alpha":
                return ["`const " + self.template.alpha_cpp + " " + name + "`: Input scalar constant."]
            return ["`const " + self.template.beta_cpp + " " + name + "`: Input scalar constant."]
        return []

    def scalar_create_cpp(self, flavour):
        """Creates a C++ version of a scalar based on a void*"""
        result = []
        for name in self.scalars:
            if name == "alpha":
                result.append("const auto alpha_cpp = " + flavour.use_alpha_clblast() + ";")
            elif name == "beta":
                result.append("const auto beta_cpp = " + flavour.use_beta_clblast() + ";")
        return result

    def sizes_list(self):
        """Retrieves a list of comma-separated sizes (m, n, k)"""
        if self.sizes:
            return [", ".join([s for s in self.sizes])]
        return []

    def sizes_def(self):
        """Retrieves the definition of the sizes (m,n,k)"""
        if self.sizes:
            return [", ".join(["const size_t " + s for s in self.sizes])]
        return []

    def sizes_def_netlib(self):
        """Retrieves the definition of the sizes (m,n,k) for the CBLAS API"""
        if self.sizes:
            return [", ".join(["const int " + s for s in self.sizes])]
        return []

    def sizes_type(self):
        """Retrieves the types of the sizes (m,n,k)"""
        if self.sizes:
            return [", ".join(["const size_t" for s in self.sizes])]
        return []

    def sizes_doc(self):
        """# Retrieves the documentation of the sizes"""
        if self.sizes:
            definitions = ["`const size_t " + s + "`: Integer size argument. This value must be positive." for s in self.sizes]
            return definitions
        return []

    def options_list(self):
        """Retrieves a list of options"""
        if self.options:
            return [", ".join(self.options)]
        return []

    def options_cast(self, indent):
        """As above, but now casted to CLBlast data-types"""
        if self.options:
            options = ["static_cast<clblast::" + convert.option_to_clblast(o) + ">(" + o + ")" for o in self.options]
            return [(",\n" + indent).join(options)]
        return []

    def options_def(self):
        """Retrieves the definitions of the options (layout, transpose, side, etc.)"""
        if self.options:
            definitions = ["const " + convert.option_to_clblast(o) + " " + o for o in self.options]
            return [", ".join(definitions)]
        return []

    def options_def_c(self):
        """As above, but now for the C API"""
        if self.options:
            definitions = ["const CLBlast" + convert.option_to_clblast(o) + " " + o for o in self.options]
            return [", ".join(definitions)]
        return []

    def options_def_wrapper_clblas(self):
        """As above, but now using clBLAS data-types"""
        if self.options:
            definitions = ["const " + convert.option_to_clblas(o) + " " + o for o in self.options]
            return [", ".join(definitions)]
        return []

    def options_def_wrapper_cblas(self):
        """As above, but now using CBLAS data-types"""
        if self.options:
            definitions = ["const " + convert.option_to_cblas(o) + " " + o for o in self.options]
            return [", ".join(definitions)]
        return []

    def options_type(self):
        """Retrieves the types of the options (layout, transpose, side, etc.)"""
        if self.options:
            definitions = ["const " + convert.option_to_clblast(o) for o in self.options]
            return [", ".join(definitions)]
        return []

    def options_doc(self):
        """Retrieves the documentation of the options"""
        if self.options:
            definitions = ["`const " + convert.option_to_clblast(o) + " " + o + "`: " + convert.option_to_documentation(o) for o in self.options]
            return definitions
        return []

    def arguments(self):
        """Retrieves a combination of all the argument names (no types)"""
        return (self.options_list() + self.sizes_list() +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_first()])) +
                self.scalar("alpha") +
                list(chain(*[self.buffer(b) for b in self.buffers_first()])) +
                self.scalar("beta") +
                list(chain(*[self.buffer(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar(s) for s in self.other_scalars()])))

    def arguments_half(self):
        """As above, but with conversions from half to float"""
        return (self.options_list() + self.sizes_list() +
                list(chain(*[self.buffer_bis(b) for b in self.scalar_buffers_first()])) +
                self.scalar_half_to_float("alpha") +
                list(chain(*[self.buffer_bis(b) for b in self.buffers_first()])) +
                self.scalar_half_to_float("beta") +
                list(chain(*[self.buffer_bis(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_bis(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar(s) for s in self.other_scalars()])))

    def arguments_clcudaapi(self):
        """Retrieves a combination of all the argument names, with CLCudaAPI casts"""
        return (self.options_list() + self.sizes_list() +
                list(chain(*[self.buffer_clcudaapi(b) for b in self.scalar_buffers_first()])) +
                self.scalar("alpha") +
                list(chain(*[self.buffer_clcudaapi(b) for b in self.buffers_first()])) +
                self.scalar("beta") +
                list(chain(*[self.buffer_clcudaapi(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_clcudaapi(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar(s) for s in self.other_scalars()])))

    def arguments_cast(self, flavour, indent):
        """As above, but with CLBlast casts"""
        return (self.options_cast(indent) + self.sizes_list() +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_first()])) +
                self.scalar_use("alpha", flavour) +
                list(chain(*[self.buffer(b) for b in self.buffers_first()])) +
                self.scalar_use("beta", flavour) +
                list(chain(*[self.buffer(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use(s, flavour) for s in self.other_scalars()])))

    def arguments_netlib(self, flavour, indent):
        """As above, but for the Netlib CBLAS API"""
        return (self.options_cast(indent) + self.sizes_list() +
                list(chain(*[self.buffer_zero_offset(b) for b in self.scalar_buffers_first()])) +
                self.scalar_cpp("alpha") +
                list(chain(*[self.buffer_zero_offset(b) for b in self.buffers_first()])) +
                self.scalar_cpp("beta") +
                list(chain(*[self.buffer_zero_offset(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_zero_offset(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar(s) for s in self.other_scalars()])))

    def arguments_wrapper_clblas(self, flavour):
        """As above, but for the clBLAS wrapper"""
        return (self.options_list() + self.sizes_list() +
                list(chain(*[self.buffer_wrapper_clblas(b) for b in self.scalar_buffers_first()])) +
                self.scalar_use_wrapper("alpha", flavour) +
                list(chain(*[self.buffer_wrapper_clblas(b) for b in self.buffers_first()])) +
                self.scalar_use_wrapper("beta", flavour) +
                list(chain(*[self.buffer_wrapper_clblas(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_wrapper_clblas(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use_wrapper(s, flavour) for s in self.other_scalars()])))

    def arguments_wrapper_cblas(self, flavour):
        """As above, but for the CBLAS wrapper"""
        return (self.options_list() + self.sizes_list() +
                self.scalar_use_wrapper_cblas("alpha", flavour) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.buffers_first()])) +
                self.scalar_use_wrapper_cblas("beta", flavour) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use_wrapper_cblas(s, flavour) for s in self.other_scalars()])))

    def arguments_def(self, flavour):
        """Retrieves a combination of all the argument definitions"""
        return (self.options_def() + self.sizes_def() +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_first()])) +
                self.scalar_def("alpha", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_first()])) +
                self.scalar_def("beta", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])))

    def arguments_def_netlib(self, flavour):
        """As above, but for the Netlib CBLAS API"""
        return (self.options_def_c() + self.sizes_def_netlib() +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.scalar_buffers_first()])) +
                self.scalar_def_void("alpha", flavour) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.buffers_first()])) +
                self.scalar_def_void("beta", flavour) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])))

    def arguments_def_c(self, flavour):
        """As above, but for the C API"""
        return (self.options_def_c() + self.sizes_def() +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_first()])) +
                self.scalar_def("alpha", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_first()])) +
                self.scalar_def("beta", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])))

    def arguments_def_wrapper_clblas(self, flavour):
        """As above, but clBLAS wrapper plain data-types"""
        return (self.options_def_wrapper_clblas() + self.sizes_def() +
                list(chain(*[self.buffer_def_wrapper_cl(b, flavour) for b in self.scalar_buffers_first()])) +
                self.scalar_def_plain("alpha", flavour) +
                list(chain(*[self.buffer_def_wrapper_cl(b, flavour) for b in self.buffers_first()])) +
                self.scalar_def_plain("beta", flavour) +
                list(chain(*[self.buffer_def_wrapper_cl(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def_wrapper_cl(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def_plain(s, flavour) for s in self.other_scalars()])))

    def arguments_def_wrapper_cblas(self, flavour):
        """As above, but CBLAS wrapper plain data-types"""
        return (self.options_def_wrapper_cblas() + self.sizes_def() +
                list(chain(*[self.buffer_def_vector(b, flavour) for b in self.scalar_buffers_first()])) +
                self.scalar_def_plain("alpha", flavour) +
                list(chain(*[self.buffer_def_vector(b, flavour) for b in self.buffers_first()])) +
                self.scalar_def_plain("beta", flavour) +
                list(chain(*[self.buffer_def_vector(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def_vector(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def_plain(s, flavour) for s in self.other_scalars()])))

    def arguments_type(self, flavour):
        """Retrieves a combination of all the argument types"""
        return (self.options_type() + self.sizes_type() +
                list(chain(*[self.buffer_type(b) for b in self.scalar_buffers_first()])) +
                self.scalar_type("alpha", flavour) +
                list(chain(*[self.buffer_type(b) for b in self.buffers_first()])) +
                self.scalar_type("beta", flavour) +
                list(chain(*[self.buffer_type(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_type(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_type(s, flavour) for s in self.other_scalars()])))

    def arguments_doc(self):
        """Retrieves a combination of all the argument types"""
        return (self.options_doc() + self.sizes_doc() +
                list(chain(*[self.buffer_doc(b) for b in self.scalar_buffers_first()])) +
                list(chain(*[self.buffer_doc(b) for b in self.scalar_buffers_first()])) +
                self.scalar_doc("alpha") +
                list(chain(*[self.buffer_doc(b) for b in self.buffers_first()])) +
                self.scalar_doc("beta") +
                list(chain(*[self.buffer_doc(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_doc(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_doc(s) for s in self.other_scalars()])))

    def requirements_doc(self):
        """Retrieves a list of routine requirements for documentation"""
        return self.requirements

    def routine_header_cpp(self, spaces, default_event):
        """Retrieves the C++ templated definition for a routine"""
        indent = " " * (spaces + self.length())
        result = "template <" + self.template.name + ">\n"
        result += "StatusCode " + self.name.capitalize() + "("
        result += (",\n" + indent).join([a for a in self.arguments_def(self.template)])
        result += ",\n" + indent + "cl_command_queue* queue, cl_event* event" + default_event + ")"
        return result

    def routine_header_type_cpp(self, spaces):
        """As above, but now without variable names"""
        indent = " " * (spaces + self.length())
        result = "template <" + self.template.name + ">\n"
        result += "StatusCode " + self.name.capitalize() + "("
        result += (",\n" + indent).join([a for a in self.arguments_type(self.template)])
        result += ",\n" + indent + "cl_command_queue*, cl_event*)"
        return result

    def routine_header_c(self, flavour, spaces, extra_qualifier):
        """As above, but now for C"""
        indent = " " * (spaces + self.length())
        result = "CLBlastStatusCode" + extra_qualifier + " CLBlast" + flavour.name + self.name + "("
        result += (",\n" + indent).join([a for a in self.arguments_def_c(flavour)])
        result += ",\n" + indent + "cl_command_queue* queue, cl_event* event)"
        return result

    def routine_header_netlib(self, flavour, spaces, extra_qualifier):
        """As above, but now for the original Netlib CBLAS API"""
        indent = " " * (spaces + self.length())
        result = "void" + extra_qualifier + " cblas_" + flavour.name.lower() + self.name + "("
        result += (",\n" + indent).join([a for a in self.arguments_def_netlib(flavour)]) + ")"
        return result

    def routine_header_wrapper_clblas(self, flavour, def_only, spaces):
        """As above, but now for the clBLAS wrapper"""
        template = "<" + flavour.template + ">" if self.no_scalars() and not def_only else ""
        indent = " " * (spaces + self.length() + len(template))
        result = ""
        if self.no_scalars():
            result += "template <"
            if def_only:
                result += flavour.name
            result += ">\n"
        result += "clblasStatus clblasX" + self.name + template + "("
        result += (",\n" + indent).join([a for a in self.arguments_def_wrapper_clblas(flavour)])
        result += ",\n" + indent + "cl_uint num_queues, cl_command_queue *queues"
        result += ",\n" + indent + "cl_uint num_wait_events, const cl_event *wait_events, cl_event *events)"
        return result

    def routine_header_wrapper_cblas(self, flavour, spaces):
        """As above, but now for the CBLAS wrapper"""
        indent = " " * (spaces + self.length())
        result = "void cblasX" + self.name + "("
        result += (",\n" + indent).join([a for a in self.arguments_def_wrapper_cblas(flavour)]) + ")"
        return result
