# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################


# Locate an Netlib implementation.
# Pre-built binaries for windows can be found at http://icl.cs.utk.edu/lapack-for-windows/lapack/
#
# Defines the following variables:
#
#   Netlib_FOUND - Found the OPENCL framework
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   Netlib_LIBRARIES - libNetlib
#
# Accepts the following variables as input:
#
#   Netlib_ROOT - (as a CMake or environment variable)
#                The root directory of where Netlib libraries are found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether FindNetlib should search for
#                              64bit or 32bit libs
#
#   Netlib_COMPILERS - Prioritized list of compiler flavors that this find package should search for when
#                             looking for libraries.  The user could have multiple flavors of Netlib installed
#                             and setting this before calling FindPackage will alter order searched
#-----------------------
# Example Usage:
#
#    find_package(Netlib REQUIRED)
#    include_directories(${Netlib_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${Netlib_LIBRARIES})
#
#-----------------------

#TODO:  Extend this to use Netlib_FIND_COMPONENTS, Netlib_FIND_REQUIRED, Netlib_FIND_QUIETLY
include( FindPackageHandleStandardArgs )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

# This is a prioritized list of Netlib compiler versions that this FindModule looks for
if( NOT DEFINED Netlib_COMPILERS )
	set( Netlib_COMPILERS minGW intel )
endif( )

# Debug print statements
#message( "Netlib_LIBRARY_PATH_SUFFIXES: ${Netlib_LIBRARY_PATH_SUFFIXES}" )
#message( "ENV{Netlib_ROOT}: $ENV{Netlib_ROOT}" )
#message( "Netlib_FIND_COMPONENTS: ${Netlib_FIND_COMPONENTS}" )
#message( "Netlib_FIND_REQUIRED: ${Netlib_FIND_REQUIRED}" )

# If the user does not set which components to find, then default to all components
if( NOT Netlib_FIND_COMPONENTS )
	set( Netlib_FIND_COMPONENTS BLAS )
endif( )

# The library name available from Netlib has different names for 64bit and 32bit libs
if( LIB64 )
	set( Netlib_BLAS_LIBNAME libblas )
#	set( Netlib_BLAS_LIBNAME BLAS )  Even though the download is named BLAS, the linker expects the .dll to be called libblas.dll
else( )
	set( Netlib_BLAS_LIBNAME libblas )
endif( )

list( FIND Netlib_FIND_COMPONENTS BLAS contains_BLAS )
if( NOT contains_BLAS EQUAL -1 )
	# Find and set the location of main Netlib lib file
	find_library( Netlib_BLAS_LIBRARY
		NAMES ${Netlib_BLAS_LIBNAME}
		HINTS
			${Netlib_ROOT}
			ENV Netlib_ROOT
		PATHS
			/usr/lib
			/usr/local/lib
		DOC "Netlib dynamic library path"
		PATH_SUFFIXES lib
	)
	mark_as_advanced( Netlib_BLAS_LIBRARY )

	FIND_PACKAGE_HANDLE_STANDARD_ARGS( NETLIB DEFAULT_MSG Netlib_BLAS_LIBRARY )
endif( )

if( NETLIB_FOUND )
	list( APPEND Netlib_LIBRARIES ${Netlib_BLAS_LIBRARY} )
else( )
	if( NOT Netlib_FIND_QUIETLY )
		message( WARNING "FindNetlib could not find the Netlib library" )
		message( STATUS "Did you remember to set the Netlib_ROOT environment variable?" )
	endif( )
endif()
