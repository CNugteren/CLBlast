
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
#   CLBLAS_FOUND          Boolean holding whether or not the clBLAS library was found
#   CLBLAS_INCLUDE_DIRS   The clBLAS include directory
#   CLBLAS_LIBRARIES      The clBLAS library
#
# In case clBLAS is not installed in the default directory, set the CLBLAS_ROOT variable to point to
# the root of clBLAS, such that 'clBLAS.h' can be found in $CLBLAS_ROOT/include. This can either be
# done using an environmental variable (e.g. export CLBLAS_ROOT=/path/to/clBLAS) or using a CMake
# variable (e.g. cmake -DCLBLAS_ROOT=/path/to/clBLAS ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CLBLAS_HINTS
  ${CLBLAS_ROOT}
  $ENV{CLBLAS_ROOT}
)
set(CLBLAS_PATHS
  /usr
  /usr/local
)

# Finds the include directories
find_path(CLBLAS_INCLUDE_DIRS
  NAMES clBLAS.h
  HINTS ${CLBLAS_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CLBLAS_PATHS}
  DOC "clBLAS include header clBLAS.h"
)
mark_as_advanced(CLBLAS_INCLUDE_DIRS)

# Finds the library
find_library(CLBLAS_LIBRARIES
  NAMES clBLAS
  HINTS ${CLBLAS_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CLBLAS_PATHS}
  DOC "clBLAS library"
)
mark_as_advanced(CLBLAS_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CLBLAS_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'clBLAS.h', install clBLAS or set CLBLAS_ROOT")
endif()
if(NOT CLBLAS_LIBRARIES)
    message(STATUS "Could NOT find clBLAS library, install it or set CLBLAS_ROOT")
endif()

# Determines whether or not clBLAS was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(clBLAS DEFAULT_MSG CLBLAS_INCLUDE_DIRS CLBLAS_LIBRARIES)

# ==================================================================================================
