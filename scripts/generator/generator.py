#!/usr/bin/env python

# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This script automatically generates the bodies of the following files, creating the full CLBlast
# API interface and implementation (C, C++, and clBLAS wrapper):
#    clblast.h
#    clblast.cc
#    clblast_c.h
#    clblast_c.cc
#    wrapper_clblas.h
# It also generates the main functions for the correctness and performance tests as found in
#    test/correctness/routines/levelX/xYYYY.cc
#    test/performance/routines/levelX/xYYYY.cc
#
# ==================================================================================================

# System modules
import sys
import os.path

# Local files
from routine import Routine
from datatype import DataType, FLT, DBL, FLT2, DBL2, F2CL, D2CL

# ==================================================================================================

# Regular data-types
S = DataType("S", FLT,  [FLT,  FLT,  FLT,  FLT],  FLT ) # single (32)
D = DataType("D", DBL,  [DBL,  DBL,  DBL,  DBL],  DBL ) # double (64)
C = DataType("C", FLT2, [FLT2, FLT2, F2CL, F2CL], FLT2) # single-complex (3232)
Z = DataType("Z", DBL2, [DBL2, DBL2, D2CL, D2CL], DBL2) # double-complex (6464)

# Special cases
Css = DataType("C", FLT,          [FLT,  FLT,  FLT,  FLT], FLT2) # As C, but with constants from S
Zdd = DataType("Z", DBL,          [DBL,  DBL,  DBL,  DBL], DBL2) # As Z, but with constants from D
Ccs = DataType("C", FLT2+","+FLT, [FLT2, FLT,  F2CL, FLT], FLT2) # As C, but with one constant from S
Zzd = DataType("Z", DBL2+","+DBL, [DBL2, DBL,  D2CL, DBL], DBL2) # As Z, but with one constant from D

# C++ template data-types
T = DataType("typename T", "T", ["T", "T", "T", "T"], "T") # regular routine
Tc = DataType("typename T", "std::complex<T>,T", ["T", "T", "T", "T"], "std::complex<T>") # for herk
TU = DataType("typename T, typename U", "T,U", ["T", "U", "T", "U"], "T") # for her2k

# ==================================================================================================

# Populates a list of routines
routines = [
[ # Level 1: vector-vector
  Routine(False, "1", "rotg",  T,  [S,D],     [], [], [], ["sa","sb","sc","ss"], [], False, "Generate givens plane rotation"),
  Routine(False, "1", "rotmg", T,  [S,D],     [], [], [], ["sd1","sd2","sx1","sy1","sparam"], [], False, "Generate modified givens plane rotation"),
  Routine(False, "1", "rot",   T,  [S,D],     ["n"], [], [], ["x","y"], ["cos","sin"], False, "Apply givens plane rotation"),
  Routine(False, "1", "rotm",  T,  [S,D],     ["n"], [], [], ["x","y","sparam"], [], False, "Apply modified givens plane rotation"),
  Routine(True,  "1", "swap",  T,  [S,D,C,Z], ["n"], [], [], ["x","y"], [], False, "Swap two vectors"),
  Routine(True,  "1", "scal",  T,  [S,D,C,Z], ["n"], [], [], ["x"], ["alpha"], False, "Vector scaling"),
  Routine(True,  "1", "copy",  T,  [S,D,C,Z], ["n"], [], ["x"], ["y"], [], False, "Vector copy"),
  Routine(True,  "1", "axpy",  T,  [S,D,C,Z], ["n"], [], ["x"], ["y"], ["alpha"], False, "Vector-times-constant plus vector"),
  Routine(True,  "1", "dot",   T,  [S,D],     ["n"], [], ["x","y"], ["dot"], [], True, "Dot product of two vectors"),
  Routine(True,  "1", "dotu",  T,  [C,Z],     ["n"], [], ["x","y"], ["dot"], [], True, "Dot product of two complex vectors"),
  Routine(True,  "1", "dotc",  T,  [C,Z],     ["n"], [], ["x","y"], ["dot"], [], True, "Dot product of two complex vectors, one conjugated"),
],
[ # Level 2: matrix-vector
  Routine(True,  "2a", "gemv",  T,  [S,D,C,Z], ["m","n"], ["layout","a_transpose"], ["a","x"], ["y"], ["alpha","beta"], False, "General matrix-vector multiplication"),
  Routine(True,  "2a", "gbmv",  T,  [S,D,C,Z], ["m","n","kl","ku"], ["layout","a_transpose"], ["a","x"], ["y"], ["alpha","beta"], False, "General banded matrix-vector multiplication"),
  Routine(True,  "2a", "hemv",  T,  [C,Z],     ["n"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], False, "Hermitian matrix-vector multiplication"),
  Routine(True,  "2a", "hbmv",  T,  [C,Z],     ["n","k"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], False, "Hermitian banded matrix-vector multiplication"),
  Routine(True,  "2a", "hpmv",  T,  [C,Z],     ["n"], ["layout","triangle"], ["ap","x"], ["y"], ["alpha","beta"], False, "Hermitian packed matrix-vector multiplication"),
  Routine(True,  "2a", "symv",  T,  [S,D],     ["n"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], False, "Symmetric matrix-vector multiplication"),
  Routine(True,  "2a", "sbmv",  T,  [S,D],     ["n","k"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], False, "Symmetric banded matrix-vector multiplication"),
  Routine(True,  "2a", "spmv",  T,  [S,D],     ["n"], ["layout","triangle"], ["ap","x"], ["y"], ["alpha","beta"], False, "Symmetric packed matrix-vector multiplication"),
  Routine(True,  "2a", "trmv",  T,  [S,D,C,Z], ["n"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], True, "Triangular matrix-vector multiplication"),
  Routine(True,  "2a", "tbmv",  T,  [S,D,C,Z], ["n","k"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], True, "Triangular banded matrix-vector multiplication"),
  Routine(True,  "2a", "tpmv",  T,  [S,D,C,Z], ["n"], ["layout","triangle","a_transpose","diagonal"], ["ap"], ["x"], [], True, "Triangular packed matrix-vector multiplication"),
  Routine(False, "2a", "trsv",  T,  [S,D,C,Z], ["n"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], False, "Solves a triangular system of equations"),
  Routine(False, "2a", "tbsv",  T,  [S,D,C,Z], ["n","k"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], False, "Solves a banded triangular system of equations"),
  Routine(False, "2a", "tpsv",  T,  [S,D,C,Z], ["n"], ["layout","triangle","a_transpose","diagonal"], ["ap"], ["x"], [], False, "Solves a packed triangular system of equations"),
  # Level 2: matrix update
  Routine(True,  "2b", "ger",   T,  [S,D],     ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], False, "General rank-1 matrix update"),
  Routine(True,  "2b", "geru",  T,  [C,Z],     ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], False, "General rank-1 complex matrix update"),
  Routine(True,  "2b", "gerc",  T,  [C,Z],     ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], False, "General rank-1 complex conjugated matrix update"),
  Routine(True,  "2b", "her",   Tc, [Css,Zdd], ["n"], ["layout","triangle"], ["x"], ["a"], ["alpha"], False, "Hermitian rank-1 matrix update"),
  Routine(True,  "2b", "hpr",   Tc, [Css,Zdd], ["n"], ["layout","triangle"], ["x"], ["ap"], ["alpha"], False, "Hermitian packed rank-1 matrix update"),
  Routine(True,  "2b", "her2",  T,  [C,Z],     ["n"], ["layout","triangle"], ["x","y"], ["a"], ["alpha"], False, "Hermitian rank-2 matrix update"),
  Routine(True,  "2b", "hpr2",  T,  [C,Z],     ["n"], ["layout","triangle"], ["x","y"], ["ap"], ["alpha"], False, "Hermitian packed rank-2 matrix update"),
  Routine(True,  "2b", "syr",   T,  [S,D],     ["n"], ["layout","triangle"], ["x"], ["a"], ["alpha"], False, "Symmetric rank-1 matrix update"),
  Routine(True,  "2b", "spr",   T,  [S,D],     ["n"], ["layout","triangle"], ["x"], ["ap"], ["alpha"], False, "Symmetric packed rank-1 matrix update"),
  Routine(True,  "2b", "syr2",  T,  [S,D],     ["n"], ["layout","triangle"], ["x","y"], ["a"], ["alpha"], False, "Symmetric rank-2 matrix update"),
  Routine(True,  "2b", "spr2",  T,  [S,D],     ["n"], ["layout","triangle"], ["x","y"], ["ap"], ["alpha"], False, "Symmetric packed rank-2 matrix update"),
],
[ # Level 3: matrix-matrix
  Routine(True,  "3", "gemm",  T,  [S,D,C,Z], ["m","n","k"], ["layout","a_transpose","b_transpose"], ["a","b"], ["c"], ["alpha","beta"], False, "General matrix-matrix multiplication"),
  Routine(True,  "3", "symm",  T,  [S,D,C,Z], ["m","n"], ["layout","side","triangle"], ["a","b"], ["c"], ["alpha","beta"], False, "Symmetric matrix-matrix multiplication"),
  Routine(True,  "3", "hemm",  T,  [C,Z],     ["m","n"], ["layout","side","triangle"], ["a","b"], ["c"], ["alpha","beta"], False, "Hermitian matrix-matrix multiplication"),
  Routine(True,  "3", "syrk",  T,  [S,D,C,Z], ["n","k"], ["layout","triangle","a_transpose"], ["a"], ["c"], ["alpha","beta"], False, "Rank-K update of a symmetric matrix"),
  Routine(True,  "3", "herk",  Tc, [Css,Zdd], ["n","k"], ["layout","triangle","a_transpose"], ["a"], ["c"], ["alpha","beta"], False, "Rank-K update of a hermitian matrix"),
  Routine(True,  "3", "syr2k", T,  [S,D,C,Z], ["n","k"], ["layout","triangle","ab_transpose"], ["a","b"], ["c"], ["alpha","beta"], False, "Rank-2K update of a symmetric matrix"),
  Routine(True,  "3", "her2k", TU, [Ccs,Zzd], ["n","k"], ["layout","triangle","ab_transpose"], ["a","b"], ["c"], ["alpha","beta"], False, "Rank-2K update of a hermitian matrix"),
  Routine(True,  "3", "trmm",  T,  [S,D,C,Z], ["m","n"], ["layout","side","triangle","a_transpose","diagonal"], ["a"], ["b"], ["alpha"], False, "Triangular matrix-matrix multiplication"),
  Routine(False, "3", "trsm",  T,  [S,D,C,Z], ["m","n"], ["layout","side","triangle","a_transpose","diagonal"], ["a"], ["b"], ["alpha"], False, "Solves a triangular system of equations"),
]]

# ==================================================================================================
# Translates an option name to a CLBlast data-type
def PrecisionToFullName(x):
	return {
		'H': "Half",
		'S': "Single",
		'D': "Double",
		'C': "ComplexSingle",
		'Z': "ComplexDouble",
	}[x]

# ==================================================================================================
# Separators for the BLAS levels
separators = ["""
// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================""",
"""
// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================""",
"""
// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// ================================================================================================="""]

# Main header/footer for source files
header = """
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================
"""
footer = """
// =================================================================================================
"""

# ==================================================================================================

# The C++ API header (.h)
def clblast_h(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		result += routine.RoutineHeaderCPP(12, " = nullptr")+";\n"
	return result

# The C++ API implementation (.cc)
def clblast_cc(routines):
	result = ""
	for routine in routines:
		indent1 = " "*(20 + routine.Length())
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		if routine.implemented:
			result += routine.RoutineHeaderCPP(12, "")+" {\n"
			result += "  auto queue_cpp = Queue(*queue);\n"
			result += "  auto event_cpp = Event(event);\n"
			result += "  auto routine = X"+routine.name+"<"+routine.template.template+">(queue_cpp, event_cpp);\n"
			result += "  auto status = routine.SetUp();\n"
			result += "  if (status != StatusCode::kSuccess) { return status; }\n"
			result += "  return routine.Do"+routine.name.capitalize()+"("
			result += (",\n"+indent1).join([a for a in routine.ArgumentsCladuc(routine.template, indent1)])
			result += ");\n"
		else:
			result += routine.RoutineHeaderTypeCPP(12)+" {\n"
			result += "  return StatusCode::kNotImplemented;\n"
		result += "}\n"
		for flavour in routine.flavours:
			indent2 = " "*(34 + routine.Length() + len(flavour.template))
			result += "template StatusCode PUBLIC_API "+routine.name.capitalize()+"<"+flavour.template+">("
			result += (",\n"+indent2).join([a for a in routine.ArgumentsType(flavour)])
			result += ",\n"+indent2+"cl_command_queue*, cl_event*);\n"
	return result

# ==================================================================================================

# The C API header (.h)
def clblast_c_h(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		for flavour in routine.flavours:
			result += routine.RoutineHeaderC(flavour, 31, " PUBLIC_API")+";\n"
	return result

# The C API implementation (.cc)
def clblast_c_cc(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.name.upper()+"\n"
		for flavour in routine.flavours:
			template = "<"+flavour.template+">" if routine.NoScalars() else ""
			indent = " "*(26 + routine.Length() + len(template))
			result += routine.RoutineHeaderC(flavour, 20, "")+" {\n"
			result += "  auto status = clblast::"+routine.name.capitalize()+template+"("
			result += (",\n"+indent).join([a for a in routine.ArgumentsCast(flavour, indent)])
			result += ",\n"+indent+"queue, event);"
			result += "\n  return static_cast<StatusCode>(status);\n}\n"
	return result

# ==================================================================================================

# The wrapper to the reference clBLAS routines (for performance/correctness testing)
def wrapper_clblas(routines):
	result = ""
	for routine in routines:
		result += "\n// Forwards the clBLAS calls for %s\n" % (routine.ShortNames())
		if routine.NoScalars():
			result += routine.RoutineHeaderWrapper(routine.template, True, 21)+";\n"
		for flavour in routine.flavours:
			indent = " "*(17 + routine.Length())
			result += routine.RoutineHeaderWrapper(flavour, False, 21)+" {\n"
			arguments = routine.ArgumentsWrapper(flavour)
			if routine.scratch:
				result += "  auto queue = Queue(queues[0]);\n"
				result += "  auto context = queue.GetContext();\n"
				result += "  auto scratch_buffer = Buffer<"+flavour.template+">(context, n*x_inc + x_offset);\n"
				arguments += ["scratch_buffer()"]
			result += "  return clblas"+flavour.name+routine.name+"("
			result += (",\n"+indent).join([a for a in arguments])
			result += ",\n"+indent+"num_queues, queues, num_wait_events, wait_events, events);"
			result += "\n}\n"
	return result

# ==================================================================================================

# Checks for the number of command-line arguments
if len(sys.argv) != 2:
	print "[ERROR] Usage: generator.py <root_of_clblast>"
	sys.exit()

# Parses the command-line arguments
path_clblast = sys.argv[1]
files = [
  path_clblast+"/include/clblast.h",
  path_clblast+"/src/clblast.cc",
  path_clblast+"/include/clblast_c.h",
  path_clblast+"/src/clblast_c.cc",
  path_clblast+"/test/wrapper_clblas.h",
]
header_lines = [84, 64, 93, 22, 22]
footer_lines = [6, 3, 9, 2, 6]

# Checks whether the command-line arguments are valid; exists otherwise
for f in files:
	if not os.path.isfile(f):
		print "[ERROR] The path '"+path_clblast+"' does not point to the root of the CLBlast library"
		sys.exit()

# ==================================================================================================

# Iterates over all files to output
for i in xrange(0,len(files)):

	# Stores the header and the footer of the original file
	with open(files[i]) as f:
		original = f.readlines()
	file_header = original[:header_lines[i]]
	file_footer = original[-footer_lines[i]:]

	# Re-writes the body of the file
	with open(files[i], "w") as f:
		body = ""
		for level in [1,2,3]:
			body += separators[level-1]+"\n"
			if i == 0:
				body += clblast_h(routines[level-1])
			if i == 1:
				body += clblast_cc(routines[level-1])
			if i == 2:
				body += clblast_c_h(routines[level-1])
			if i == 3:
				body += clblast_c_cc(routines[level-1])
			if i == 4:
				body += wrapper_clblas(routines[level-1])
		f.write("".join(file_header))
		f.write(body)
		f.write("".join(file_footer))

# ==================================================================================================

# Outputs all the correctness-test implementations
for level in [1,2,3]:
	for routine in routines[level-1]:
		filename = path_clblast+"/test/correctness/routines/level"+str(level)+"/x"+routine.name+".cc"
		with open(filename, "w") as f:
			body = ""
			body += "#include \"correctness/testblas.h\"\n"
			body += "#include \"routines/level"+str(level)+"/x"+routine.name+".h\"\n\n"
			body += "// Shortcuts to the clblast namespace\n"
			body += "using float2 = clblast::float2;\n"
			body += "using double2 = clblast::double2;\n\n"
			body += "// Main function (not within the clblast namespace)\n"
			body += "int main(int argc, char *argv[]) {\n"
			not_first = "false"
			for flavour in routine.flavours:
				body += "  clblast::RunTests<clblast::TestX"+routine.name+flavour.TestTemplate()
				body += ">(argc, argv, "+not_first+", \""+flavour.name+routine.name.upper()+"\");\n"
				not_first = "true"
			body += "  return 0;\n"
			body += "}\n"
			f.write(header+"\n")
			f.write(body)
			f.write(footer)

# Outputs all the performance-test implementations
for level in [1,2,3]:
	for routine in routines[level-1]:
		filename = path_clblast+"/test/performance/routines/level"+str(level)+"/x"+routine.name+".cc"
		with open(filename, "w") as f:
			body = ""
			body += "#include \"performance/client.h\"\n"
			body += "#include \"routines/level"+str(level)+"/x"+routine.name+".h\"\n\n"
			body += "// Shortcuts to the clblast namespace\n"
			body += "using float2 = clblast::float2;\n"
			body += "using double2 = clblast::double2;\n\n"
			body += "// Main function (not within the clblast namespace)\n"
			body += "int main(int argc, char *argv[]) {\n"
			default = PrecisionToFullName(routine.flavours[0].name)
			body += "  switch(clblast::GetPrecision(argc, argv, clblast::Precision::k"+default+")) {\n"
			for precision in ["H","S","D","C","Z"]:
				body += "    case clblast::Precision::k"+PrecisionToFullName(precision)+":"
				found = False
				for flavour in routine.flavours:
					if flavour.name == precision:
						body += "\n      clblast::RunClient<clblast::TestX"+routine.name+flavour.TestTemplate()
						body += ">(argc, argv); break;\n"
						found = True
				if not found:
					body += " throw std::runtime_error(\"Unsupported precision mode\");\n"
			body += "  }\n"
			body += "  return 0;\n"
			body += "}\n"
			f.write(header+"\n")
			f.write(body)
			f.write(footer)

# ==================================================================================================
