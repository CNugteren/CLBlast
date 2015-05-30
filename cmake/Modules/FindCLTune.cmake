
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
#   CLTUNE_FOUND          Boolean holding whether or not the CLTune library was found
#   CLTUNE_INCLUDE_DIRS   The CLTune include directory
#   CLTUNE_LIBRARIES      The CLTune library
#
# In case CLTune is not installed in the default directory, set the CLTUNE_ROOT variable to point to
# the root of CLTune, such that 'cltune.h' can be found in $CLTUNE_ROOT/include. This can either be
# done using an environmental variable (e.g. export CLTUNE_ROOT=/path/to/cltune) or using a CMake
# variable (e.g. cmake -DCLTUNE_ROOT=/path/to/cltune ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CLTUNE_HINTS
  ${CLTUNE_ROOT}
  $ENV{CLTUNE_ROOT}
)
set(CLTUNE_PATHS
  /usr
  /usr/local
)

# Finds the include directories
find_path(CLTUNE_INCLUDE_DIRS
  NAMES cltune.h
  HINTS ${CLTUNE_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CLTUNE_PATHS}
  DOC "CLTune include header cltune.h"
)
mark_as_advanced(CLTUNE_INCLUDE_DIRS)

# Finds the library
find_library(CLTUNE_LIBRARIES
  NAMES cltune
  HINTS ${CLTUNE_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32
  PATHS ${CLTUNE_PATHS}
  DOC "CLTune library"
)
mark_as_advanced(CLTUNE_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CLTUNE_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'cltune.h', install CLTune or set CLTUNE_ROOT")
endif()
if(NOT CLTUNE_LIBRARIES)
    message(STATUS "Could NOT find CLTune library, install it or set CLTUNE_ROOT")
endif()

# Determines whether or not CLTune was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLTune DEFAULT_MSG CLTUNE_INCLUDE_DIRS CLTUNE_LIBRARIES)

# ==================================================================================================
