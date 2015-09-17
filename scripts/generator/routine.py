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

	# ==============================================================================================

	# Retrieves a variable name for a specific input/output vector/matrix (e.g. 'x')
	def Buffer(self, name):
		if (name in self.inputs) or (name in self.outputs):
			a = [name+"_buffer"]
			b = [name+"_offset"]
			c = [name+"_"+self.Postfix(name)] if (name not in ["dot"]) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with data-types
	def BufferDef(self, name):
		prefix = "const " if (name in self.inputs) else ""
		if (name in self.inputs) or (name in self.outputs):
			a = [prefix+"cl_mem "+name+"_buffer"]
			b = ["const size_t "+name+"_offset"]
			c = ["const size_t "+name+"_"+self.Postfix(name)] if (name not in ["dot"]) else []
			return [", ".join(a+b+c)]
		return []

	# As above but with Claduc buffers
	def BufferCladuc(self, name):
		if (name in self.inputs) or (name in self.outputs):
			a = ["Buffer<"+self.template.buffertype+">("+name+"_buffer)"]
			b = [name+"_offset"]
			c = [name+"_"+self.Postfix(name)] if (name not in ["dot"]) else []
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
			c = ["const size_t"] if (name not in ["dot"]) else []
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
		if ((name == "alpha") and (name in self.scalars)):
			return [flavour.UseAlpha()]
		elif ((name == "beta") and (name in self.scalars)):
			return [flavour.UseBeta()]
		return []

	# Retrieves the use of a scalar (alpha/beta)
	def ScalarUseWrapper(self, name, flavour):
		if ((name == "alpha") and (name in self.scalars)):
			return [flavour.UseAlphaCL()]
		elif ((name == "beta") and (name in self.scalars)):
			return [flavour.UseBetaCL()]
		return []

	# Retrieves the definition of a scalar (alpha/beta)
	def ScalarDef(self, name, flavour):
		if ((name == "alpha") and (name in self.scalars)):
			return ["const "+flavour.alpha_cl+" "+name]
		elif ((name == "beta") and (name in self.scalars)):
			return ["const "+flavour.beta_cl+" "+name]
		return []

	# As above, but without 'cl_' prefix
	def ScalarDefPlain(self, name, flavour):
		if ((name == "alpha") and (name in self.scalars)):
			return ["const "+flavour.alpha_cpp+" "+name]
		elif ((name == "beta") and (name in self.scalars)):
			return ["const "+flavour.beta_cpp+" "+name]
		return []

	# Retrieves the type of a scalar (alpha/beta)
	def ScalarType(self, name, flavour):
		if ((name == "alpha") and (name in self.scalars)):
			return ["const "+flavour.alpha_cpp]
		elif ((name == "beta") and (name in self.scalars)):
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
		return (self.Options() + self.Sizes() + self.BufferCladuc("dot") +
		        self.Scalar("alpha") +
		        self.BufferCladuc("a") + self.BufferCladuc("b") + self.BufferCladuc("x") +
		        self.Scalar("beta") + self.BufferCladuc("y") + self.BufferCladuc("c"))

	# Retrieves a combination of all the argument names, with CLBlast casts
	def ArgumentsCast(self, flavour, indent):
		return (self.OptionsCast(indent) + self.Sizes() + self.Buffer("dot") +
		        self.ScalarUse("alpha", flavour) +
		        self.Buffer("a") + self.Buffer("b") + self.Buffer("x") +
		        self.ScalarUse("beta", flavour) + self.Buffer("y") + self.Buffer("c"))

	# As above, but for the clBLAS wrapper
	def ArgumentsWrapper(self, flavour):
		return (self.Options() + self.Sizes() + self.BufferWrapper("dot") +
		        self.ScalarUseWrapper("alpha", flavour) +
		        self.BufferWrapper("a") + self.BufferWrapper("b") + self.BufferWrapper("x") +
		        self.ScalarUseWrapper("beta", flavour) + self.BufferWrapper("y") + self.BufferWrapper("c"))

	# Retrieves a combination of all the argument definitions
	def ArgumentsDef(self, flavour):
		return (self.OptionsDef() + self.SizesDef() + self.BufferDef("dot") +
		        self.ScalarDef("alpha", flavour) +
		        self.BufferDef("a") + self.BufferDef("b") + self.BufferDef("x") +
		        self.ScalarDef("beta", flavour) + self.BufferDef("y") + self.BufferDef("c"))

	# As above, but clBLAS wrapper plain datatypes
	def ArgumentsDefWrapper(self, flavour):
		return (self.OptionsDefWrapper() + self.SizesDef() + self.BufferDef("dot") +
		        self.ScalarDefPlain("alpha", flavour) +
		        self.BufferDef("a") + self.BufferDef("b") + self.BufferDef("x") +
		        self.ScalarDefPlain("beta", flavour) + self.BufferDef("y") + self.BufferDef("c"))
	
	# Retrieves a combination of all the argument types
	def ArgumentsType(self, flavour):
		return (self.OptionsType() + self.SizesType() + self.BufferType("dot") +
		        self.ScalarType("alpha", flavour) +
		        self.BufferType("a") + self.BufferType("b") + self.BufferType("x") +
		        self.ScalarType("beta", flavour) + self.BufferType("y") + self.BufferType("c"))


	# ==============================================================================================

	# Retrieves the C++ templated definition for a routine
	def RoutineHeaderCPP(self, spaces):
		indent = " "*(spaces + self.Length())
		result = "template <"+self.template.name+">\n"
		result += "StatusCode "+self.name.capitalize()+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsDef(self.template)])
		result += ",\n"+indent+"cl_command_queue* queue, cl_event* event)"
		return result

	# As above, but now without variable names
	def RoutineHeaderTypeCPP(self, spaces):
		indent = " "*(spaces + self.Length())
		result = "template <"+self.template.name+">\n"
		result += "StatusCode "+self.name.capitalize()+"("
		result += (",\n"+indent).join([a for a in self.ArgumentsType(self.template)])
		result += ",\n"+indent+"cl_command_queue* queue, cl_event* event)"
		return result

	# As above, but now for C
	def RoutineHeaderC(self, flavour, spaces):
		indent = " "*(spaces + self.Length())
		result = "StatusCode CLBlast"+flavour.name+self.name+"("
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
