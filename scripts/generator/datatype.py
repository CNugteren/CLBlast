#!/usr/bin/env python

# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file contains the 'DataType' class, used in the generator script to generate the CLBlast API
# interface and implementation.
#
# ==================================================================================================

# Short-hands for data-types
FLT = "float"
DBL = "double"
FLT2 = "float2"
DBL2 = "double2"
F2CL = "cl_float2"
D2CL = "cl_double2"

# Structure holding data-type and precision information
class DataType():
	def __init__(self, precision_name, name, template, scalars, buffertype):
		self.precision_name = precision_name
		self.name = name
		self.template = template
		self.alpha_cpp = scalars[0]
		self.beta_cpp = scalars[1]
		self.alpha_cl = scalars[2]
		self.beta_cl = scalars[3]
		self.buffertype = buffertype

	# Outputs the name of the data-type (alpha/beta), possibly transforming into the right type
	def UseAlpha(self):
		if self.alpha_cpp in [FLT2, DBL2]:
			return self.alpha_cpp+"{alpha.s[0], alpha.s[1]}"
		return "alpha"
	def UseBeta(self):
		if self.beta_cpp in [FLT2, DBL2]:
			return self.beta_cpp+"{beta.s[0], beta.s[1]}"
		return "beta"

	# As above, but the transformation is in the opposite direction
	def UseAlphaCL(self):
		if self.alpha_cpp in [FLT2, DBL2]:
			return self.alpha_cl+"{{alpha.real(), alpha.imag()}}"
		return "alpha"
	def UseBetaCL(self):
		if self.beta_cpp in [FLT2, DBL2]:
			return self.beta_cl+"{{beta.real(), beta.imag()}}"
		return "beta"

	# Returns the template as used in the correctness/performance tests
	def TestTemplate(self):
		if self.buffertype != self.beta_cpp:
			return "<"+self.buffertype+","+self.beta_cpp+">, "+self.buffertype+", "+self.beta_cpp
		return "<"+self.buffertype+">, "+self.buffertype+", "+self.beta_cpp


# ==================================================================================================
