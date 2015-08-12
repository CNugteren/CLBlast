
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================

# CMake project details
cmake_minimum_required(VERSION 2.8.10)
project("clblast" CXX)
set(clblast_VERSION_MAJOR 0)
set(clblast_VERSION_MINOR 3)
set(clblast_VERSION_PATCH 0)

# Options and their default values
option(SAMPLES "Enable compilation of the examples" OFF)
option(TUNERS "Enable compilation of the tuners" OFF)
option(TESTS "Enable compilation of the performance and correctness tests" OFF)

# ==================================================================================================

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH false) # Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_BUILD_WITH_INSTALL_RPATH false) # When building, don't use the install RPATH already
set(CMAKE_INSTALL_RPATH "") # The RPATH to be used when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH false) # Don't add the automatically determined parts

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# C++ compiler settings
set(FLAGS "-O3 -std=c++11")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
    set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(FLAGS "${FLAGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
  set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
  set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Requires OpenCL. It is found through the included "FindOpenCL.cmake" in CMAKE_MODULE_PATH.
find_package(OpenCL REQUIRED)

# Locates the CLTune library in case the tuners need to be compiled. "FindCLTune.cmake" is included.
if(TUNERS)
  find_package(CLTune)
  if(NOT CLTUNE_FOUND)
    message(STATUS "Could NOT find CLTune, disabling the compilation of the tuners")
    set(TUNERS OFF)
  endif()
endif()

# ==================================================================================================

# Includes directories: CLBlast and OpenCL
include_directories(${clblast_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS})

# ==================================================================================================

# Sets the supported routines and the used kernels. New routines and kernels should be added here.
set(KERNELS copy pad transpose padtranspose xaxpy xgemv xgemm)
set(SAMPLE_PROGRAMS sgemm)
set(LEVEL1_ROUTINES xaxpy)
set(LEVEL2_ROUTINES xgemv)
set(LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm)
set(ROUTINES ${LEVEL1_ROUTINES} ${LEVEL2_ROUTINES} ${LEVEL3_ROUTINES})

# ==================================================================================================

# Gathers all source-files
set(SOURCES src/clblast.cc src/database.cc src/routine.cc src/utilities.cc src/clblast_c.cc)
foreach(ROUTINE ${LEVEL1_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level1/${ROUTINE}.cc)
endforeach()
foreach(ROUTINE ${LEVEL2_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level2/${ROUTINE}.cc)
endforeach()
foreach(ROUTINE ${LEVEL3_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level3/${ROUTINE}.cc)
endforeach()

# Creates and links the library
add_library(clblast SHARED ${SOURCES})
target_link_libraries(clblast ${OPENCL_LIBRARIES})

# Installs the library
install(TARGETS clblast DESTINATION lib)
install(FILES include/clblast.h DESTINATION include)
install(FILES include/clblast_c.h DESTINATION include)

# ==================================================================================================

# This section contains all the code related to the examples
if(SAMPLES)

  # Adds sample programs
  foreach(SAMPLE ${SAMPLE_PROGRAMS})
    add_executable(sample_${SAMPLE} samples/${SAMPLE}.cc)
    target_link_libraries(sample_${SAMPLE} clblast ${OPENCL_LIBRARIES})
    install(TARGETS sample_${SAMPLE} DESTINATION bin)
  endforeach()

endif()

# ==================================================================================================

# This section contains all the code related to the tuners. These tuners require the presence of
# the CLTune library (not included as part of the source).
if(TUNERS)

  # Includes CLTune
  include_directories(${CLTUNE_INCLUDE_DIRS})

  # Creates the common tuner objects (requires CMake 2.8.8)
  add_library(tuners_common OBJECT src/tuning/tuning.cc)

  # Adds tuning executables
  foreach(KERNEL ${KERNELS})
    add_executable(tuner_${KERNEL} $<TARGET_OBJECTS:tuners_common> src/tuning/${KERNEL}.cc)
    target_link_libraries(tuner_${KERNEL} clblast ${CLTUNE_LIBRARIES} ${OPENCL_LIBRARIES})
    install(TARGETS tuner_${KERNEL} DESTINATION bin)
  endforeach()

endif()

# ==================================================================================================

# Down from here is all test (performance and correctness) related. Note that these tests require
# the compilation of the clBLAS library to act as a reference.
if(TESTS)

  # Enables clBLAS to provide as reference for the tests (source-code is shipped with the project).
  # This subproject uses specific flags to reduce the amount of warnings.
  set(CMAKE_CXX_FLAGS_CLBLAST ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "-O3")
  add_subdirectory(external/clBLAS/src)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_CLBLAST})

  # Adds new include directories for the reference clBLAS
  include_directories(${clblast_SOURCE_DIR}/test ${clBLAS_SOURCE_DIR})

  # Creates the common correctness-tests objects (requires CMake 2.8.8)
  add_library(test_correctness_common OBJECT
              test/correctness/tester.cc test/correctness/testblas.cc)

  # Compiles the correctness-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level1/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level2/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level3/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(test_${ROUTINE} clBLAS clblast ${OPENCL_LIBRARIES})
    install(TARGETS test_${ROUTINE} DESTINATION bin)
  endforeach()

  # Creates the common performance-tests objects (requires CMake 2.8.8)
  add_library(test_performance_common OBJECT test/performance/client.cc)

  # Compiles the performance-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level1/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level2/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level3/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(client_${ROUTINE} clBLAS clblast ${OPENCL_LIBRARIES})
    install(TARGETS client_${ROUTINE} DESTINATION bin)
  endforeach()

endif()
# ==================================================================================================
