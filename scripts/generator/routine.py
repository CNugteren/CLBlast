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
def OptionToWrapper(x):
	return {
	    'layout': "clblasOrder",
	    'a_transpose': "clblasTranspose",
	    'b_transpose': "clblasTranspose",
	    'ab_transpose': "clblasTranspose",
	    'side': "clblasSide",
	    'triangle': "clblasUplo",
	    'diagonal': "clblasDiag",
	}[x]

# ==================================================================================================

# Class holding routine-specific information (e.g. name, which arguments, which precisions)
class Routine():
	def __init__(self, implemented, level, name, template, flavours, sizes, options,
	             inputs, outputs, scalars, scratch, description):
		self.implemented = implemented
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

	# List of scalar buffers
	def ScalarBuffers(self):
		return ["SA","SB","C","S","dot"]

	# List of buffers without 'inc' or 'ld'
	def BuffersWithoutLdInc(self):
		return self.ScalarBuffers() + ["ap"]

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

	# As above but with Claduc buffers
	def BufferCladuc(self, name):
		if (name in self.inputs) or (name in self.outputs):
			a = ["Buffer<"+self.template.buffertype+">("+name+"_buffer)"]
			b = [name+"_offset"]
			c = [name+"_"+self.Postfix(name)] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with a static cast for clBLAS wrapper
	def BufferWrapper(self, name):
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

	# As above, but only data-types
	def BufferType(self, name):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			a = [prefix+"cl_mem"]
			b = ["const size_t"]
			c = ["const size_t"] if (name not in self.BuffersWithoutLdInc()) else []
			return [", ".join(a+b+c)]
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
	def OptionsDefWrapper(self):
		if self.options:
			definitions = ["const "+OptionToWrapper(o)+" "+o for o in self.options]
			return [", ".join(definitions)]
		return []

	# Retrieves the types of the options (layout, transpose, side, etc.)
	def OptionsType(self):
		if self.options:
			definitions = ["const "+OptionToCLBlast(o) for o in self.options]
			return [", ".join(definitions)]
		return []

	# ==============================================================================================

	# Retrieves a combination of all the argument names, with Claduc casts
	def ArgumentsCladuc(self, flavour, indent):
		return (self.Options() + self.Sizes() +
		        list(chain(*[self.BufferCladuc(b) for b in self.ScalarBuffers()])) +
		        self.Scalar("alpha") +
		        list(chain(*[self.BufferCladuc(b) for b in self.BuffersFirst()])) +
		        self.Scalar("beta") +
		        list(chain(*[self.BufferCladuc(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.Scalar(s) for s in ["C","S"]])))

	# Retrieves a combination of all the argument names, with CLBlast casts
	def ArgumentsCast(self, flavour, indent):
		return (self.OptionsCast(indent) + self.Sizes() +
		        list(chain(*[self.Buffer(b) for b in self.ScalarBuffers()])) +
		        self.ScalarUse("alpha", flavour) +
		        list(chain(*[self.Buffer(b) for b in self.BuffersFirst()])) +
		        self.ScalarUse("beta", flavour) +
		        list(chain(*[self.Buffer(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.ScalarUse(s, flavour) for s in ["C","S"]])))

	# As above, but for the clBLAS wrapper
	def ArgumentsWrapper(self, flavour):
		return (self.Options() + self.Sizes() +
		        list(chain(*[self.BufferWrapper(b) for b in self.ScalarBuffers()])) +
		        self.ScalarUseWrapper("alpha", flavour) +
		        list(chain(*[self.BufferWrapper(b) for b in self.BuffersFirst()])) +
		        self.ScalarUseWrapper("beta", flavour) +
		        list(chain(*[self.BufferWrapper(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.ScalarUseWrapper(s, flavour) for s in ["C","S"]])))

	# Retrieves a combination of all the argument definitions
	def ArgumentsDef(self, flavour):
		return (self.OptionsDef() + self.SizesDef() +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffers()])) +
		        self.ScalarDef("alpha", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersFirst()])) +
		        self.ScalarDef("beta", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.ScalarDef(s, flavour) for s in ["C","S"]])))

	# As above, but clBLAS wrapper plain datatypes
	def ArgumentsDefWrapper(self, flavour):
		return (self.OptionsDefWrapper() + self.SizesDef() +
		        list(chain(*[self.BufferDef(b) for b in self.ScalarBuffers()])) +
		        self.ScalarDefPlain("alpha", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersFirst()])) +
		        self.ScalarDefPlain("beta", flavour) +
		        list(chain(*[self.BufferDef(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.ScalarDefPlain(s, flavour) for s in ["C","S"]])))
	
	# Retrieves a combination of all the argument types
	def ArgumentsType(self, flavour):
		return (self.OptionsType() + self.SizesType() +
		        list(chain(*[self.BufferType(b) for b in self.ScalarBuffers()])) +
		        self.ScalarType("alpha", flavour) +
		        list(chain(*[self.BufferType(b) for b in self.BuffersFirst()])) +
		        self.ScalarType("beta", flavour) +
		        list(chain(*[self.BufferType(b) for b in self.BuffersSecond()])) +
		        list(chain(*[self.ScalarType(s, flavour) for s in ["C","S"]])))


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
	def RoutineHeaderWrapper(self, flavour, def_only, spaces):
		template = "<"+flavour.template+">" if self.NoScalars() and not def_only else ""
		indent = " "*(spaces + self.Length() + len(template))
		result = ""
		if self.NoScalars():
			result += "template <"
			if def_only:
				result += flavour.name
			result += ">\n"
		result += "clblasStatus clblasX"+self.name+template+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDefWrapper(flavour)])
		result += ",\n"+indent+"cl_uint num_queues, cl_command_queue *queues"
		result += ",\n"+indent+"cl_uint num_wait_events, const cl_event *wait_events, cl_event *events)"
		return result

# ==================================================================================================
