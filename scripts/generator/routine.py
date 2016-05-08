#!/usr/bin/env python

# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file contains the 'Routine' class, used in the generator script to generate the CLBlast API
# interface and implementation.
#
# ==================================================================================================

# System modules
from itertools import chain

# Translates an option name to a CLBlast data-type
def OptionToCLBlast(x):
	return {
	    'layout': "Layout",
	    'a_transpose': "Transpose",
	    'b_transpose': "Transpose",
	    'ab_transpose': "Transpose",
	    'side': "Side",
	    'triangle': "Triangle",
	    'diagonal': "Diagonal",
	}[x]

# As above, but for clBLAS data-types
def OptionToWrapperCL(x):
	return {
	    'layout': "clblasOrder",
	    'a_transpose': "clblasTranspose",
	    'b_transpose': "clblasTranspose",
	    'ab_transpose': "clblasTranspose",
	    'side': "clblasSide",
	    'triangle': "clblasUplo",
	    'diagonal': "clblasDiag",
	}[x]

# As above, but for CBLAS data-types
def OptionToWrapperC(x):
	return {
	    'layout': "CBLAS_ORDER",
	    'a_transpose': "CBLAS_TRANSPOSE",
	    'b_transpose': "CBLAS_TRANSPOSE",
	    'ab_transpose': "CBLAS_TRANSPOSE",
	    'side': "CBLAS_SIDE",
	    'triangle': "CBLAS_UPLO",
	    'diagonal': "CBLAS_DIAG",
	}[x]

# Translates an option name to a documentation string
def OptionToDoc(x):
	return {
	    'layout': "Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.",
	    'a_transpose': "Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
	    'b_transpose': "Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
	    'ab_transpose': "Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
	    'side': "The horizontal position of the triangular matrix, either `Side::kLeft` (141) or `Side::kRight` (142).",
	    'triangle': "The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).",
	    'diagonal': "The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for a non-unit values on the diagonal or `Diagonal::kUnit` (132) for a unit values on the diagonal.",
	}[x]

# ==================================================================================================

# Class holding routine-specific information (e.g. name, which arguments, which precisions)
class Routine():
	def __init__(self, implemented, has_tests, level, name, template, flavours, sizes, options,
	             inputs, outputs, scalars, scratch, description, details, requirements):
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
		self.scalars = scalars
		self.scratch = scratch # Scratch buffer (e.g. for xDOT)
		self.description = description
		self.details = details
		self.requirements = requirements

	# List of scalar buffers
	def ScalarBuffersFirst(self):
		return ["dot","nrm2","asum","sum","imax","imin"]
	def ScalarBuffersSecond(self):
		return ["sa","sb","sc","ss","sd1","sd2","sx1","sy1","sparam"]

	# List of scalars other than alpha and beta
	def OtherScalars(self):
		return ["cos","sin"]

	# List of buffers with unsigned int type
	def IndexBuffers(self):
		return ["imax","imin"]

	# List of buffers without 'inc' or 'ld'
	def BuffersWithoutLdInc(self):
		return self.ScalarBuffersFirst() + self.ScalarBuffersSecond() + ["ap"]

	# Retrieves the number of characters in the routine's name
	def Length(self):
		return len(self.name)

	# Retrieves the postfix for a buffer
	def Postfix(self, name):
		return "inc" if (name in ["x","y"]) else "ld"

	# Determines whether or not this routine has scalar arguments (alpha/beta)
	def NoScalars(self):
		return self.scalars == []

	# Returns the upper-case names of these routines (all flavours)
	def ShortNames(self):
		return "/".join([f.name+self.name.upper() for f in self.flavours])

	# Determines which buffers go first (between alpha and beta) and which ones go after
	def BuffersFirst(self):
		if self.level == "2b":
			return ["x","y"]
		return ["ap","a","b","x"]
	def BuffersSecond(self):
		if self.level == "2b":
			return ["ap","a","b","c"]
		return ["y","c"]

	# Distinguish between vectors and matrices
	def BuffersVector(self):
		return ["x","y"]
	def BuffersMatrix(self):
		return ["a","b","c","ap"]

	# ==============================================================================================

	# Retrieves a variable name for a specific input/output vector/matrix (e.g. 'x')
	def Buffer(self, name):
		if (name in self.inputs) or (name in self.outputs):
			a = [name+"_buffer"]
			b = [name+"_offset"]
			c = [name+"_"+self.Postfix(name)] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with data-types
	def BufferDef(self, name):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			a = [prefix+"cl_mem "+name+"_buffer"]
			b = ["const size_t "+name+"_offset"]
			c = ["const size_t "+name+"_"+self.Postfix(name)] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# As above but as vectors
	def BufferDefVector(self, name, flavour):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			a = [prefix+"std::vector<"+flavour.buffertype+">& "+name+"_buffer"]
			b = ["const size_t "+name+"_offset"]
			c = ["const size_t "+name+"_"+self.Postfix(name)] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with Claduc buffers
	def BufferCladuc(self, name):
		if (name in self.inputs) or (name in self.outputs):
			buffertype = "unsigned int" if (name in self.IndexBuffers()) else self.template.buffertype
			a = ["Buffer<"+buffertype+">("+name+"_buffer)"]
			b = [name+"_offset"]
			c = [name+"_"+self.Postfix(name)] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with a static cast for clBLAS wrapper
	def BufferWrapperCL(self, name):
		if (name in self.inputs) or (name in self.outputs):
			a = [name+"_buffer"]
			b = [name+"_offset"]
			c = []
			if (name in ["x","y"]):
				c = ["static_cast<int>("+name+"_"+self.Postfix(name)+")"]
			elif (name in ["a","b","c"]):
				c = [name+"_"+self.Postfix(name)]
			return [", ".join(a+b+c)]
		return []

	# As above but with a static cast for CBLAS wrapper
	def BufferWrapperC(self, name, flavour):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			if name == "sy1":
				a = [name+"_buffer["+name+"_offset]"]
			elif flavour.precision_name in ["C","Z"]:
				a = ["reinterpret_cast<"+prefix+flavour.buffertype[:-1]+"*>(&"+name+"_buffer["+name+"_offset])"]
			else:
				a = ["&"+name+"_buffer["+name+"_offset]"]
			c = []
			if (name in ["x","y"]):
				c = ["static_cast<int>("+name+"_"+self.Postfix(name)+")"]
			elif (name in ["a","b","c"]):
				c = [name+"_"+self.Postfix(name)]
			return [", ".join(a+c)]
		return []

	# As above, but only data-types
	def BufferType(self, name):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			a = [prefix+"cl_mem"]
			b = ["const size_t"]
			c = ["const size_t"] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# Retrieves the documentation of the buffers
	def BufferDoc(self, name):
		prefix = "const " if (name in self.inputs) else ""
		inout = "input" if (name in self.inputs) else "output"
		if (name in self.inputs) or (name in self.outputs):
			math_name = name.upper()+" matrix" if (name in self.BuffersMatrix()) else name+" vector"
			incld_description = "Leading dimension " if (name in self.BuffersMatrix()) else "Stride/increment "
			a = ["`"+prefix+"cl_mem "+name+"_buffer`: OpenCL buffer to store the "+inout+" "+math_name+"."]
			b = ["`const size_t "+name+"_offset`: The offset in elements from the start of the "+inout+" "+math_name+"."]
			c = ["`const size_t "+name+"_"+self.Postfix(name)+"`: "+incld_description+"of the "+inout+" "+math_name+"."] if (name not in self.BuffersWithoutLdInc()) else []
			return a+b+c
		return []

	# ==============================================================================================

	# Retrieves the name of a scalar (alpha/beta)
	def Scalar(self, name):
		if (name in self.scalars):
			return [name]
		return []

	# Retrieves the use of a scalar (alpha/beta)
	def ScalarUse(self, name, flavour):
		if name in self.scalars:
			if name == "alpha":
				return [flavour.UseAlpha()]
			elif name == "beta":
				return [flavour.UseBeta()]
			return [name]
		return []

	# Retrieves the use of a scalar (alpha/beta)
	def ScalarUseWrapper(self, name, flavour):
		if name in self.scalars:
			if name == "alpha":
				return [flavour.UseAlphaCL()]
			elif name == "beta":
				return [flavour.UseBetaCL()]
			return [name]
		return []

	# Retrieves the use of a scalar for CBLAS (alpha/beta)
	def ScalarUseWrapperC(self, name, flavour):
		if name in self.scalars:
			if flavour.IsComplex(name):
				return [name+"_array.data()"]
			return [name]
		return []

	# Retrieves the definition of a scalar (alpha/beta)
	def ScalarDef(self, name, flavour):
		if name in self.scalars:
			if name == "alpha":
				return ["const "+flavour.alpha_cl+" "+name]
			return ["const "+flavour.beta_cl+" "+name]
		return []

	# As above, but without 'cl_' prefix
	def ScalarDefPlain(self, name, flavour):
		if name in self.scalars:
			if name == "alpha":
				return ["const "+flavour.alpha_cpp+" "+name]
			return ["const "+flavour.beta_cpp+" "+name]
		return []

	# Retrieves the type of a scalar (alpha/beta)
	def ScalarType(self, name, flavour):
		if name in self.scalars:
			if name == "alpha":
				return ["const "+flavour.alpha_cpp]
			return ["const "+flavour.beta_cpp]
		return []

	# Retrieves the documentation of a scalar
	def ScalarDoc(self, name):
		if name in self.scalars:
			if name == "alpha":
				return ["`const "+self.template.alpha_cpp+" "+name+"`: Input scalar constant."]
			return ["`const "+self.template.beta_cpp+" "+name+"`: Input scalar constant."]
		return []

	# ==============================================================================================

	# Retrieves a list of comma-separated sizes (m, n, k)
	def Sizes(self):
		if self.sizes:
			return [", ".join([s for s in self.sizes])]
		return []

	# Retrieves the definition of the sizes (m,n,k)
	def SizesDef(self):
		if self.sizes:
			return [", ".join(["const size_t "+s for s in self.sizes])]
		return []

	# Retrieves the types of the sizes (m,n,k)
	def SizesType(self):
		if self.sizes:
			return [", ".join(["const size_t" for s in self.sizes])]
		return []

	# Retrieves the documentation of the sizes
	def SizesDoc(self):
		if self.sizes:
			definitions = ["`const size_t "+s+"`: Integer size argument." for s in self.sizes]
			return definitions
		return []

	# ==============================================================================================

	# Retrieves a list of options
	def Options(self):
		if self.options:
			return [", ".join(self.options)]
		return []

	# As above, but now casted to CLBlast data-types
	def OptionsCast(self, indent):
		if self.options:
			options = ["static_cast<clblast::"+OptionToCLBlast(o)+">("+o+")" for o in self.options]
			return [(",\n"+indent).join(options)]
		return []

	# Retrieves the definitions of the options (layout, transpose, side, etc.)
	def OptionsDef(self):
		if self.options:
			definitions = ["const "+OptionToCLBlast(o)+" "+o for o in self.options]
			return [", ".join(definitions)]
		return []

	# As above, but now using clBLAS data-types
	def OptionsDefWrapperCL(self):
		if self.options:
			definitions = ["const "+OptionToWrapperCL(o)+" "+o for o in self.options]
			return [", ".join(definitions)]
		return []

	# As above, but now using CBLAS data-types
	def OptionsDefWrapperC(self):
		if self.options:
			definitions = ["const "+OptionToWrapperC(o)+" "+o for o in self.options]
			return [", ".join(definitions)]
		return []

	# Retrieves the types of the options (layout, transpose, side, etc.)
	def OptionsType(self):
		if self.options:
			definitions = ["const "+OptionToCLBlast(o) for o in self.options]
			return [", ".join(definitions)]
		return []

	# Retrieves the documentation of the options
	def OptionsDoc(self):
		if self.options:
			definitions = ["`const "+OptionToCLBlast(o)+"`: "+OptionToDoc(o) for o in self.options]
			return definitions
		return []

	# ==============================================================================================

	# Retrieves a combination of all the argument names, with Claduc casts
	def ArgumentsCladuc(self, flavour, indent):
		return (self.Options() + self.Sizes() +
		        list(chain(*[self.BufferCladuc(b) for b in self.ScalarBuffersFirst()])) +
		        self.Scalar("alpha") +
		        list(chain(*[self.BufferCladuc(b) for b in self.BuffersFirst()])) +
		        self.Scalar("beta") +
		        list(chain(*[self.BufferCladuc(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferCladuc(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.Scalar(s) for s in self.OtherScalars()])))

	# Retrieves a combination of all the argument names, with CLBlast casts
	def ArgumentsCast(self, flavour, indent):
		return (self.OptionsCast(indent) + self.Sizes() +
		        list(chain(*[self.Buffer(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarUse("alpha", flavour) +
		        list(chain(*[self.Buffer(b) for b in self.BuffersFirst()])) +
		        self.ScalarUse("beta", flavour) +
		        list(chain(*[self.Buffer(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.Buffer(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarUse(s, flavour) for s in self.OtherScalars()])))

	# As above, but for the clBLAS wrapper
	def ArgumentsWrapperCL(self, flavour):
		return (self.Options() + self.Sizes() +
		        list(chain(*[self.BufferWrapperCL(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarUseWrapper("alpha", flavour) +
		        list(chain(*[self.BufferWrapperCL(b) for b in self.BuffersFirst()])) +
		        self.ScalarUseWrapper("beta", flavour) +
		        list(chain(*[self.BufferWrapperCL(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferWrapperCL(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarUseWrapper(s, flavour) for s in self.OtherScalars()])))

	# As above, but for the CBLAS wrapper
	def ArgumentsWrapperC(self, flavour):
		return (self.Options() + self.Sizes() +
		        self.ScalarUseWrapperC("alpha", flavour) +
		        list(chain(*[self.BufferWrapperC(b, flavour) for b in self.BuffersFirst()])) +
		        self.ScalarUseWrapperC("beta", flavour) +
		        list(chain(*[self.BufferWrapperC(b, flavour) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferWrapperC(b, flavour) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarUseWrapperC(s, flavour) for s in self.OtherScalars()])))

	# Retrieves a combination of all the argument definitions
	def ArgumentsDef(self, flavour):
		return (self.OptionsDef() + self.SizesDef() +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarDef("alpha", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersFirst()])) +
		        self.ScalarDef("beta", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarDef(s, flavour) for s in self.OtherScalars()])))

	# As above, but clBLAS wrapper plain datatypes
	def ArgumentsDefWrapperCL(self, flavour):
		return (self.OptionsDefWrapperCL() + self.SizesDef() +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarDefPlain("alpha", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersFirst()])) +
		        self.ScalarDefPlain("beta", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarDefPlain(s, flavour) for s in self.OtherScalars()])))

	# As above, but CBLAS wrapper plain datatypes
	def ArgumentsDefWrapperC(self, flavour):
		return (self.OptionsDefWrapperC() + self.SizesDef() +
		        list(chain(*[self.BufferDefVector(b, flavour) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarDefPlain("alpha", flavour) +
		        list(chain(*[self.BufferDefVector(b, flavour) for b in self.BuffersFirst()])) +
		        self.ScalarDefPlain("beta", flavour) +
		        list(chain(*[self.BufferDefVector(b, flavour) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferDefVector(b, flavour) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarDefPlain(s, flavour) for s in self.OtherScalars()])))
	
	# Retrieves a combination of all the argument types
	def ArgumentsType(self, flavour):
		return (self.OptionsType() + self.SizesType() +
		        list(chain(*[self.BufferType(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarType("alpha", flavour) +
		        list(chain(*[self.BufferType(b) for b in self.BuffersFirst()])) +
		        self.ScalarType("beta", flavour) +
		        list(chain(*[self.BufferType(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferType(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarType(s, flavour) for s in self.OtherScalars()])))
	
	# Retrieves a combination of all the argument types
	def ArgumentsDoc(self):
		return (self.OptionsDoc() + self.SizesDoc() +
		        list(chain(*[self.BufferDoc(b) for b in self.ScalarBuffersFirst()])) +
		        list(chain(*[self.BufferDoc(b) for b in self.ScalarBuffersFirst()])) +
		        self.ScalarDoc("alpha") +
		        list(chain(*[self.BufferDoc(b) for b in self.BuffersFirst()])) +
		        self.ScalarDoc("beta") +
		        list(chain(*[self.BufferDoc(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.BufferDoc(b) for b in self.ScalarBuffersSecond()])) +
		        list(chain(*[self.ScalarDoc(s) for s in self.OtherScalars()])))

	# ==============================================================================================

	# Retrieves a list of routine requirements for documentation
	def RequirementsDoc(self):
		return []

	# ==============================================================================================

	# Retrieves the C++ templated definition for a routine
	def RoutineHeaderCPP(self, spaces, default_event):
		indent = " "*(spaces + self.Length())
		result = "template <"+self.template.name+">\n"
		result += "StatusCode "+self.name.capitalize()+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDef(self.template)])
		result += ",\n"+indent+"cl_command_queue* queue, cl_event* event"+default_event+")"
		return result

	# As above, but now without variable names
	def RoutineHeaderTypeCPP(self, spaces):
		indent = " "*(spaces + self.Length())
		result = "template <"+self.template.name+">\n"
		result += "StatusCode "+self.name.capitalize()+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsType(self.template)])
		result += ",\n"+indent+"cl_command_queue*, cl_event*)"
		return result

	# As above, but now for C
	def RoutineHeaderC(self, flavour, spaces, extra_qualifier):
		indent = " "*(spaces + self.Length())
		result = "StatusCode"+extra_qualifier+" CLBlast"+flavour.name+self.name+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDef(flavour)])
		result += ",\n"+indent+"cl_command_queue* queue, cl_event* event)"
		return result

	# As above, but now for the clBLAS wrapper
	def RoutineHeaderWrapperCL(self, flavour, def_only, spaces):
		template = "<"+flavour.template+">" if self.NoScalars() and not def_only else ""
		indent = " "*(spaces + self.Length() + len(template))
		result = ""
		if self.NoScalars():
			result += "template <"
			if def_only:
				result += flavour.name
			result += ">\n"
		result += "clblasStatus clblasX"+self.name+template+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDefWrapperCL(flavour)])
		result += ",\n"+indent+"cl_uint num_queues, cl_command_queue *queues"
		result += ",\n"+indent+"cl_uint num_wait_events, const cl_event *wait_events, cl_event *events)"
		return result

	# As above, but now for the CBLAS wrapper
	def RoutineHeaderWrapperC(self, flavour, def_only, spaces):
		indent = " "*(spaces + self.Length())
		result = "void cblasX"+self.name+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDefWrapperC(flavour)])+")"
		return result

# ==================================================================================================
