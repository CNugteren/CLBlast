
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from itertools import chain

import generator.convert as convert


class Routine:
    """Class holding routine-specific information (e.g. name, which arguments, which precisions)"""
    def __init__(self, implemented, has_tests, batched_strided, temp_buffer, level, name, template, flavours, sizes, options,
                 inputs, outputs, buffer_sizes, scalars, scratch,
                 description, details, requirements):
        self.implemented = implemented
        self.has_tests = has_tests
        self.batched = batched_strided
        self.temp_buffer = temp_buffer
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

    def lowercase_name(self):
        postfix = "strided" if self.batched == 2 else ""
        postfix += "batched" if self.batched != 0 else ""
        return self.name + postfix

    def plain_name(self):
        postfix = "Strided" if self.batched == 2 else ""
        postfix += "Batched" if self.batched != 0 else ""
        return self.name + postfix

    def capitalized_name(self):
        postfix = "Strided" if self.batched == 2 else ""
        postfix += "Batched" if self.batched != 0 else ""
        return self.name.capitalize() + postfix

    def upper_name(self):
        postfix = "STRIDED" if self.batched == 2 else ""
        postfix += "BATCHED" if self.batched != 0 else ""
        return self.name.upper() + postfix

    def b_star(self):
        return "*" if self.batched == 1 else ""

    def b_s(self):
        return "s" if self.batched == 1 else ""

    def batch_count_def(self):
        return ["const size_t batch_count"] if self.batched != 0 else []

    def batch_count_list(self):
        return ["batch_count"] if self.batched != 0 else []

    def batch_count_type(self):
        return ["const size_t"] if self.batched != 0 else []

    def batch_count_doc(self):
        return ["`const size_t batch_count`: Number of batches. This value must be positive."] if self.batched != 0 else []

    def batched_transform_to_cpp(self):
        result = []
        for scalar in self.scalars:
            result.append("auto " + scalar + "s_cpp = std::vector<T>();")
        for buffer_name in self.inputs + self.outputs:
            result.append("auto " + buffer_name + "_offsets_cpp = std::vector<size_t>();")
        result.append("for (auto batch = size_t{0}; batch < batch_count; ++batch) {")
        for scalar in self.scalars:
            result.append("  " + scalar + "s_cpp.push_back(" + scalar + "s[batch]);")
        for buffer_name in self.inputs + self.outputs:
            result.append("  " + buffer_name + "_offsets_cpp.push_back(" + buffer_name + "_offsets[batch]);")
        result.append("}")
        return result

    def batched_transform_to_complex(self, flavour):
        result = []
        for scalar in self.scalars:
            result.append("auto " + scalar + "s_cpp = std::vector<" + flavour.buffer_type + ">();")
        result.append("for (auto batch = size_t{0}; batch < batch_count; ++batch) {")
        for scalar in self.scalars:
            content = scalar
            if scalar == "alpha":
                content = flavour.use_alpha(postfix="s[batch]")
            elif scalar == "beta":
                content = flavour.use_beta(postfix="s[batch]")
            result.append("  " + scalar + "s_cpp.push_back(" + content + ");")
        result.append("}")
        return result

    @staticmethod
    def scalar_buffers_first():
        """List of scalar buffers"""
        return ["dot", "nrm2", "asum", "sum", "imax", "imin"]

    @staticmethod
    def scalar_buffers_second():
        """List of scalar buffers"""
        return ["sa", "sb", "sc", "ss", "sd1", "sd2", "sx1", "sy1", "sparam"]

    @staticmethod
    def scalar_buffers_second_non_pointer():
        """As above, but these ones are not passed as pointers but as scalars instead"""
        return ["sy1"]

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
        return "inc" if (name in ["x", "y", "z"]) else "ld"

    @staticmethod
    def buffers_vector():
        """Distinguish between vectors and matrices"""
        return ["x", "y", "z"]

    @staticmethod
    def buffers_matrix():
        """Distinguish between vectors and matrices"""
        return ["a", "b", "c", "ap"]

    @staticmethod
    def buffers_tensor():
        """Distinguish between vectors and matrices and tensors"""
        return ["im", "col", "kernel", "result"]

    @staticmethod
    def routines_scalar_no_return():
        return ["dotu", "dotc"]

    @staticmethod
    def set_size(name, size):
        """Sets the size of a buffer"""
        return "const auto " + name + "_size = " + size + ";"

    @staticmethod
    def create_buffer(name, template):
        """Creates a new CLCudaAPI buffer"""
        return "auto " + name + "_buffer = clblast::Buffer<" + template + ">(context, " + name + "_size);"

    def write_buffer(self, name, template):
        """Writes to a CLCudaAPI buffer"""
        postfix = ""
        if name in self.scalar_buffers_second_non_pointer():
            postfix = "_vec"
        data_structure = "reinterpret_cast<" + template + "*>(" + name + postfix + ")"
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
        return self.scalar_buffers_first() + self.scalar_buffers_second() + ["ap", "im", "col", "kernel", "result"]

    def get_buffer_type(self, name, flavour):
        if name in self.index_buffers():
            return "int"
        return flavour.buffer_type

    def length(self):
        """Retrieves the number of characters in the routine's name"""
        return len(self.capitalized_name())

    def no_scalars(self):
        """Determines whether or not this routine has scalar arguments (alpha/beta)"""
        return self.scalars == [] or self.name in ["im2col", "col2im", "convgemm"]

    def has_layout(self):
        """Determines whether the layout is an argument"""
        return "layout" in self.options

    def short_names(self):
        """Returns the upper-case names of these routines (all flavours)"""
        return "/".join([f.name + self.upper_name() for f in self.flavours])

    def short_names_tested(self):
        """As above, but excludes some"""
        names = [f.name + self.upper_name() for f in self.flavours]
        if "H" + self.upper_name() in names:
            names.remove("H" + self.upper_name())
        return "/".join(names)

    def buffers_first(self):
        """Determines which buffers go first (between alpha and beta) and which ones go after"""
        if self.level == "2b" or self.name == "had":
            return ["x", "y"]
        extra_buffer = "col" if self.name == "col2im" else "im"
        return ["ap", "a", "b", "x", extra_buffer, "kernel"]

    def buffers_second(self):
        if self.level == "2b" or self.name == "had":
            return ["z", "ap", "a", "b", "c"]
        extra_buffer = "im" if self.name == "col2im" else "col"
        return ["y", "c", extra_buffer, "result"]

    def buffer(self, name):
        """Retrieves a variable name for a specific input/output vector/matrix (e.g. 'x')"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer"]
            b = [name + "_offset" + self.b_s()]
            c = [name + "_" + self.postfix(name)] if (name not in self.buffers_without_ld_inc()) else []
            if self.batched == 2:
                c += [name + "_stride"]
            return [", ".join(a + b + c)]
        return []

    def buffer_bis(self, name):
        """As above but with a '_bis' suffix for the buffer name"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer_bis"]
            b = [name + "_offset"]
            c = [name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            if self.batched == 2:
                c += [name + "_stride"]
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
            b = ["const size_t " + self.b_star() + name + "_offset" + self.b_s()]
            c = ["const size_t " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            if self.batched == 2:
                c += ["const size_t " + name + "_stride"]
            return [", ".join(a + b + c)]
        return []

    def buffer_def_wrapper_cl(self, name, flavour):
        """As above but for OpenCL"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            a = [prefix + "Buffer<" + flavour.buffer_type + ">& " + name + "_buffer"]
            b = ["const size_t " + name + "_offset"]
            c = ["const size_t " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + b + c)]
        return []

    def buffer_def_wrapper_cuda(self, name, flavour):
        """As above but for CUDA"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            a = [prefix + flavour.buffer_type + "* " + name + "_buffer"]
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
            pointer = "" if name in self.scalar_buffers_second_non_pointer() else "*"
            a = [prefix + data_type + pointer + " " + name + ""]
            c = ["const int " + name + "_" + self.postfix(name)] if name not in self.buffers_without_ld_inc() else []
            return [", ".join(a + c)]
        return []

    def buffer_clcudaapi(self, name):
        """As above but with CLCudaAPI buffers"""
        if name in self.inputs or name in self.outputs:
            buffer_type = "unsigned int" if (name in self.index_buffers()) else self.template.buffer_type
            a = ["Buffer<" + buffer_type + ">(" + name + "_buffer)"]
            b = [name + "_offsets_cpp"] if self.batched == 1 else [name + "_offset"]
            c = [name + "_" + self.postfix(name)] if (name not in self.buffers_without_ld_inc()) else []
            if self.batched == 2:
                c += [name + "_stride"]
            return [", ".join(a + b + c)]
        return []

    def buffer_wrapper_clblas(self, name):
        """As above but with a static cast for clBLAS wrapper"""
        if name in self.inputs or name in self.outputs:
            a = [name + "_buffer()"]
            b = [name + "_offset"]
            c = []
            if name in ["x", "y", "z"]:
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
            if name in ["x", "y", "z", "a", "b", "c"]:
                c = ["static_cast<int>(" + name + "_" + self.postfix(name) + ")"]
            return [", ".join(a + c)]
        return []

    def buffer_wrapper_cublas(self, name, flavour):
        """As above but for cuBLAS the wrapper"""
        prefix = "const " if name in self.inputs else ""
        if name in self.inputs or name in self.outputs:
            if name in self.index_buffers():
                a = ["reinterpret_cast<int*>(&" + name + "_buffer[" + name + "_offset])"]
            elif name in self.outputs and flavour.name in ["Sc", "Dz"]:
                dtype = "float" if flavour.name == "Sc" else "double"
                a = ["reinterpret_cast<" + dtype + "*>(&" + name + "_buffer[" + name + "_offset])"]
            elif flavour.precision_name in ["C", "Z"]:
                cuda_complex = "cuDoubleComplex" if flavour.precision_name == "Z" else "cuComplex"
                a = ["reinterpret_cast<" + prefix + cuda_complex + "*>" +
                     "(&" + name + "_buffer[" + name + "_offset])"]
            else:
                a = ["&" + name + "_buffer[" + name + "_offset]"]
            c = []
            if name in ["x", "y", "z"]:
                c = ["static_cast<int>(" + name + "_" + self.postfix(name) + ")"]
            elif name in ["a", "b", "c"]:
                c = [name + "_" + self.postfix(name)]
            result = [", ".join(a + c)]
            if self.name == "trmm" and name == "a":
                result *= 2
            return result
        return []

    def buffer_type(self, name):
        """As above, but only data-types"""
        prefix = "const " if (name in self.inputs) else ""
        if (name in self.inputs) or (name in self.outputs):
            a = [prefix + "cl_mem"]
            b = ["const size_t" + self.b_star()]
            c = ["const size_t"] if (name not in self.buffers_without_ld_inc()) else []
            if self.batched == 2:
                c += ["const size_t"]
            return [", ".join(a + b + c)]
        return []

    def buffer_doc(self, name):
        """Retrieves the documentation of the buffers"""
        prefix = "const " if (name in self.inputs) else ""
        inout = "input" if (name in self.inputs) else "output"
        if (name in self.inputs) or (name in self.outputs):
            math_name = name.upper() + " matrix" if (name in self.buffers_matrix()) else name + " tensor" if (name in self.buffers_tensor()) else name + " vector"
            inc_ld_description = "Leading dimension " if (name in self.buffers_matrix()) else "Stride/increment "
            a = ["`" + prefix + "cl_mem " + name + "_buffer`: OpenCL buffer to store the " + inout + " " + math_name + "."]
            b = ["`const size_t " + self.b_star() + name + "_offset" + self.b_s() + "`: The offset" + self.b_s() + " in elements from the start of the " + inout + " " + math_name + "."]
            c = []
            if name not in self.buffers_without_ld_inc():
                c = ["`const size_t " + name + "_" + self.postfix(name) + "`: " +
                     inc_ld_description + "of the " + inout + " " + math_name + ". This value must be greater than 0."]
            if self.batched == 2:
                c += ["`const size_t " + name + "_stride`: The (fixed) stride between two batches of the " + name.upper() + " matrix."]
            return a + b + c
        return []

    def scalar(self, name):
        """Retrieves the name of a scalar (alpha/beta)"""
        if name in self.scalars:
            if self.batched == 1:
                return [name + "s_cpp"]
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
                if self.batched == 1:
                    return ["alphas_cpp.data()"]
                return [flavour.use_alpha()]
            elif name == "beta":
                if self.batched == 1:
                    return ["betas_cpp.data()"]
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

    def scalar_use_wrapper_cublas(self, name, flavour):
        """As above, but for the cuBLAS wrapper"""
        if name in self.scalars:
            if flavour.is_complex(name):
                return ["&" + name + "_cuda"]
            return ["&" + name]
        return []

    def scalar_def(self, name, flavour):
        """Retrieves the definition of a scalar (alpha/beta)"""
        if name in self.scalars:
            if name == "alpha":
                return ["const " + flavour.alpha_cl + " " + self.b_star() + name + self.b_s()]
            return ["const " + flavour.beta_cl + " " + self.b_star() + name + self.b_s()]
        return []

    def scalar_def_plain(self, name, flavour):
        """As above, but without 'cl_' prefix"""
        if name in self.scalars:
            if name == "alpha":
                return ["const " + flavour.alpha_cpp + " " + self.b_star() + name + self.b_s()]
            return ["const " + flavour.beta_cpp + " " + self.b_star() + name + self.b_s()]
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
                return ["const " + flavour.alpha_cpp + self.b_star()]
            return ["const " + flavour.beta_cpp + self.b_star()]
        return []

    def scalar_doc(self, name):
        """Retrieves the documentation of a scalar"""
        if name in self.scalars:
            if name == "alpha":
                return ["`const " + self.template.alpha_cpp + " " + self.b_star() + name + self.b_s() + "`: Input scalar constant" + self.b_s() + "."]
            return ["`const " + self.template.beta_cpp + " " + self.b_star() + name + self.b_s() + "`: Input scalar constant" + self.b_s() + "."]
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

    def sizes_list_as_int(self):
        """Retrieves a list of comma-separated sizes (m, n, k) cast to integers"""
        if self.sizes:
            return [", ".join(["static_cast<int>(" + s + ")" for s in self.sizes])]
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

    def options_list_no_layout(self):
        """Retrieves a list of options"""
        options = self.options[:]
        if "layout" in options:
            options.remove("layout")
        if options:
            return [", ".join(options)]
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

    def options_def_wrapper_cublas(self):
        """As above, but now using cuBLAS data-types"""
        if self.options:
            definitions = ["const " + convert.option_to_cublas(o) + " " + o for o in self.options]
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
                list(chain(*[self.scalar(s) for s in self.other_scalars()])) +
                self.batch_count_list())

    def arguments_cast(self, flavour, indent):
        """As above, but with CLBlast casts"""
        return (self.options_cast(indent) + self.sizes_list() +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_first()])) +
                self.scalar_use("alpha", flavour) +
                list(chain(*[self.buffer(b) for b in self.buffers_first()])) +
                self.scalar_use("beta", flavour) +
                list(chain(*[self.buffer(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use(s, flavour) for s in self.other_scalars()])) +
                self.batch_count_list())

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
        return (self.options_list() + self.sizes_list_as_int() +
                self.scalar_use_wrapper_cblas("alpha", flavour) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.buffers_first()])) +
                self.scalar_use_wrapper_cblas("beta", flavour) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_wrapper_cblas(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use_wrapper_cblas(s, flavour) for s in self.other_scalars()])))

    def arguments_wrapper_cublas(self, flavour):
        """As above, but for the cuBLAS wrapper"""
        return (self.options_list_no_layout() + self.sizes_list_as_int() +
                self.scalar_use_wrapper_cublas("alpha", flavour) +
                list(chain(*[self.buffer_wrapper_cublas(b, flavour) for b in self.buffers_first()])) +
                self.scalar_use_wrapper_cublas("beta", flavour) +
                list(chain(*[self.buffer_wrapper_cublas(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_wrapper_cublas(b, flavour) for b in self.scalar_buffers_first()])) +
                list(chain(*[self.buffer_wrapper_cublas(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_use_wrapper_cublas(s, flavour) for s in self.other_scalars()])))

    def arguments_def(self, flavour):
        """Retrieves a combination of all the argument definitions"""
        return (self.options_def() + self.sizes_def() +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_first()])) +
                self.scalar_def("alpha", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_first()])) +
                self.scalar_def("beta", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])) +
                self.batch_count_def())

    def arguments_def_netlib(self, flavour):
        """As above, but for the Netlib CBLAS API"""
        result=(self.options_def_c() + self.sizes_def_netlib() +
                self.scalar_def_void("alpha", flavour) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.buffers_first()])) +
                self.scalar_def_void("beta", flavour) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])))
        if self.name in self.routines_scalar_no_return():
            result += list(chain(*[self.buffer_def_pointer(b, flavour) for b in self.scalar_buffers_first()]))
        result += self.batch_count_def()
        return result

    def arguments_def_c(self, flavour):
        """As above, but for the C API"""
        return (self.options_def_c() + self.sizes_def() +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_first()])) +
                self.scalar_def("alpha", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_first()])) +
                self.scalar_def("beta", flavour) +
                list(chain(*[self.buffer_def(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_def(s, flavour) for s in self.other_scalars()])) +
                self.batch_count_def())

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

    def arguments_def_wrapper_cublas(self, flavour):
        """As above, but cuBLAS wrapper plain data-types"""
        return (self.options_def_wrapper_cublas() + self.sizes_def() +
                list(chain(*[self.buffer_def_wrapper_cuda(b, flavour) for b in self.scalar_buffers_first()])) +
                self.scalar_def_plain("alpha", flavour) +
                list(chain(*[self.buffer_def_wrapper_cuda(b, flavour) for b in self.buffers_first()])) +
                self.scalar_def_plain("beta", flavour) +
                list(chain(*[self.buffer_def_wrapper_cuda(b, flavour) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_def_wrapper_cuda(b, flavour) for b in self.scalar_buffers_second()])) +
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
                list(chain(*[self.scalar_type(s, flavour) for s in self.other_scalars()])) +
                self.batch_count_type())

    def arguments_doc(self):
        """Retrieves a combination of all the argument types"""
        return (self.options_doc() + self.sizes_doc() +
                list(chain(*[self.buffer_doc(b) for b in self.scalar_buffers_first()])) +
                self.scalar_doc("alpha") +
                list(chain(*[self.buffer_doc(b) for b in self.buffers_first()])) +
                self.scalar_doc("beta") +
                list(chain(*[self.buffer_doc(b) for b in self.buffers_second()])) +
                list(chain(*[self.buffer_doc(b) for b in self.scalar_buffers_second()])) +
                list(chain(*[self.scalar_doc(s) for s in self.other_scalars()])) +
                self.batch_count_doc())

    def arguments_python(self):
        """Arguments for the Python wrapper pyclblast"""
        result = list()
        result.extend(self.sizes)
        buffers = self.inputs + self.outputs
        result.extend(buffers[:])
        for buf in buffers:
            if buf in self.buffers_matrix():
                result.append(buf + "_ld")
        for buf in buffers:
            if buf in self.buffers_vector():
                result.append(buf + "_inc = 1")
        for scalar in self.scalars:
            default = "1.0" if scalar == "alpha" else "0.0"
            result.append(scalar + " = " + default)
        for option in self.options:
            if option == "a_transpose":
                result.append("a_transp = False")
            if option == "b_transpose":
                result.append("b_transp = False")
            if option == "ab_transpose":
                result.append("ab_transp = False")
            if option == "side":
                result.append("right_side = False")
            if option == "triangle":
                result.append("lower_triangle = False")
            if option == "diagonal":
                result.append("unit_diagonal = False")
        for buf in buffers:
            result.append(buf + "_offset = 0")
        return result

    def requirements_doc(self):
        """Retrieves a list of routine requirements for documentation"""
        return self.requirements

    def routine_header_cpp(self, spaces, default_event, cuda=False, implementation=False):
        """Retrieves the C++ templated definition for a routine"""
        indent = " " * (spaces + self.length())
        arguments = self.arguments_def(self.template)
        mem_type = "cl_mem"
        if cuda:
            arguments = [a.replace(mem_type, "CUdeviceptr") for a in arguments]
            mem_type = "CUdeviceptr"
        result = "template <" + self.template.name + ">\n"
        result += "StatusCode " + self.capitalized_name() + "("
        result += (",\n" + indent).join([a for a in arguments])
        result += ",\n" + indent
        if cuda:
            result += "const CUcontext context, const CUdevice device"
        else:
            result += "cl_command_queue* queue, cl_event* event" + default_event
        if self.temp_buffer:
            result += ",\n" + indent + mem_type + " temp_buffer"
            if not implementation:
                result += " = 0" if cuda else " = nullptr"
        result += ")"
        return result

    def routine_header_type_cpp(self, spaces, cuda=False):
        """As above, but now without variable names"""
        indent = " " * (spaces + self.length())
        arguments = self.arguments_type(self.template)
        if cuda:
            arguments = [a.replace("cl_mem", "CUdeviceptr") for a in arguments]
        result = "template <" + self.template.name + ">\n"
        result += "StatusCode " + self.capitalized_name() + "("
        result += (",\n" + indent).join([a for a in arguments])
        result += ",\n" + indent
        if cuda:
            result += "const CUcontext, const CUdevice"
        else:
            result += "cl_command_queue*, cl_event*"
        result += ")"
        return result

    def routine_header_c(self, flavour, spaces, extra_qualifier):
        """As above, but now for C"""
        indent = " " * (spaces + self.length())
        result = "CLBlastStatusCode" + extra_qualifier + " CLBlast" + flavour.name + self.plain_name() + "("
        result += (",\n" + indent).join([a for a in self.arguments_def_c(flavour)])
        result += ",\n" + indent + "cl_command_queue* queue, cl_event* event)"
        return result

    def routine_header_netlib(self, flavour, spaces, extra_qualifier):
        """As above, but now for the original Netlib CBLAS API"""
        return_type = "void"
        for output in self.outputs:
            if output in self.index_buffers():
                return_type = "int"
                break
            if output in self.scalar_buffers_first() and self.name not in self.routines_scalar_no_return():
                return_type = flavour.buffer_type.replace("2", "")
                break
        indent = " " * (spaces + len(return_type) + self.length())
        routine_name = self.name
        if self.name in self.routines_scalar_no_return():
            routine_name += "_sub"
            indent += "    "
        if self.batched != 0:
            routine_name += "batched"
        result = return_type + extra_qualifier + " cblas_" + flavour.name.lower() + routine_name + "("
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

    def routine_header_wrapper_cublas(self, flavour, def_only, spaces):
        """As above, but now for the cuBLAS wrapper"""
        template = "<" + flavour.template + ">" if self.no_scalars() and not def_only else ""
        indent = " " * (spaces + self.length() + len(template))
        result = ""
        if self.no_scalars():
            result += "template <"
            if def_only:
                result += flavour.name
            result += ">\n"
        result += "cublasStatus_t cublasX" + self.name + template + "(cublasHandle_t handle, "
        result += (",\n" + indent).join([a for a in self.arguments_def_wrapper_cublas(flavour)]) + ")"
        return result
