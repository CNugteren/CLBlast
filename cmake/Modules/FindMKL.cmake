
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================
#
# Defines the following variables:
#   MKL_FOUND          Boolean holding whether or not the Intel MKL BLAS library was found
#   MKL_INCLUDE_DIRS   The Intel MKL BLAS include directory
#   MKL_LIBRARIES      The Intel MKL BLAS library
#
# In case MKL is not installed in the default directory, set the MKL_ROOT variable to point to
# the root of MKL, such that 'mkl_cblas.h' can be found in $MKL_ROOT/include. This can either be
# done using an environmental variable (e.g. export MKL_ROOT=/path/to/MKL) or using a CMake
# variable (e.g. cmake -DMKL_ROOT=/path/to/MKL ..).
#
# ==================================================================================================

# Sets the possible install locations
set(MKL_HINTS
  ${MKL_ROOT}
  $ENV{MKL_ROOT}
)
set(MKL_PATHS
  /usr
  /usr/local
  /usr/local/opt
  /usr/local/mkl
  /opt/intel
  /opt/intel/mkl
)

# Finds the include directories
find_path(MKL_INCLUDE_DIRS
  NAMES mkl_cblas.h
  HINTS ${MKL_HINTS}
  PATH_SUFFIXES
    include inc include/x86_64 include/x64
  PATHS ${MKL_PATHS}
  DOC "Intel MKL CBLAS include header mkl_cblas.h"
)
mark_as_advanced(MKL_INCLUDE_DIRS)

# Finds the libraries
set(MKL_LIB_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import lib/intel64)
find_library(MKL_LIBRARIES_LP64 NAMES mkl_intel_lp64 HINTS ${MKL_HINTS} PATH_SUFFIXES ${MKL_LIB_SUFFIXES} PATHS ${MKL_PATHS} DOC "Intel MKL lp64 library")
find_library(MKL_LIBRARIES_THREAD NAMES mkl_intel_thread HINTS ${MKL_HINTS} PATH_SUFFIXES ${MKL_LIB_SUFFIXES} PATHS ${MKL_PATHS} DOC "Intel MKL thread library")
find_library(MKL_LIBRARIES_CORE NAMES mkl_core HINTS ${MKL_HINTS} PATH_SUFFIXES ${MKL_LIB_SUFFIXES} PATHS ${MKL_PATHS} DOC "Intel MKL core library")
find_library(MKL_LIBRARIES_OMP NAMES iomp5 HINTS ${MKL_HINTS} PATH_SUFFIXES ${MKL_LIB_SUFFIXES} PATHS ${MKL_PATHS} DOC "Intel OpenMP library")
set(MKL_LIBRARIES ${MKL_LIBRARIES_LP64} ${MKL_LIBRARIES_THREAD} ${MKL_LIBRARIES_CORE} ${MKL_LIBRARIES_OMP})
mark_as_advanced(MKL_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT MKL_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'mkl_cblas.h', install MKL or set MKL_ROOT")
endif()
if(NOT MKL_LIBRARIES)
    message(STATUS "Could NOT find the Intel MKL BLAS library, install it or set MKL_ROOT")
endif()

# Determines whether or not MKL was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIRS MKL_LIBRARIES)

# ==================================================================================================
