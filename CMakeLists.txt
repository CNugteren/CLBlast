
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
project("clblast" C CXX)
set(clblast_VERSION_MAJOR 0)
set(clblast_VERSION_MINOR 6)
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
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(FLAGS "/Ox")
  set(FLAGS "${FLAGS} /wd4715")
else ()
  set(FLAGS "-O3 -std=c++11")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
      set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(FLAGS "${FLAGS} -Wall -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
    set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
    set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# C compiler settings (for the sample)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CFLAGS "/Ox")
else ()
  set(CFLAGS "-O3 -std=c99")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")

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

# Locates the reference BLAS libraries in case the tests need to be compiled. The "FindclBLAS.cmake"
# and "FindCBLAS.cmake" are included.
if(TESTS)
  find_package(clBLAS)
  find_package(CBLAS)
  if(NOT CLBLAS_FOUND AND NOT CBLAS_FOUND)
    message(STATUS "Could NOT find clBLAS nor a CPU BLAS, disabling the compilation of the tests")
    set(TESTS OFF)
  endif()
endif()

# ==================================================================================================

# Includes directories: CLBlast and OpenCL
include_directories(${clblast_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS})

# ==================================================================================================

# Sets the supported routines and the used kernels. New routines and kernels should be added here.
set(KERNELS copy pad transpose padtranspose xaxpy xdot xger xgemm xgemv)
set(SAMPLE_PROGRAMS_CPP sgemm)
set(SAMPLE_PROGRAMS_C sgemm)
set(LEVEL1_ROUTINES xswap xscal xcopy xaxpy xdot xdotu xdotc xnrm2)
set(LEVEL2_ROUTINES xgemv xgbmv xhemv xhbmv xhpmv xsymv xsbmv xspmv xtrmv xtbmv xtpmv
                    xger xgeru xgerc xher xhpr xher2 xhpr2 xsyr xspr xsyr2 xspr2)
set(LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm)
set(ROUTINES ${LEVEL1_ROUTINES} ${LEVEL2_ROUTINES} ${LEVEL3_ROUTINES})
set(PRECISIONS 32 64 3232 6464)

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

# Sets a default platform ($DEVICEPLATFORM) and device ($DEFAULT_DEVICE) to run tuners and tests on
set(DEVICEPLATFORM )
if(DEFINED ENV{DEFAULT_DEVICE})
  set(DEVICEPLATFORM ${DEVICEPLATFORM} -device $ENV{DEFAULT_DEVICE})
endif()
if(DEFINED ENV{DEFAULT_PLATFORM})
  set(DEVICEPLATFORM ${DEVICEPLATFORM} -platform $ENV{DEFAULT_PLATFORM})
endif()

# ==================================================================================================

# This section contains all the code related to the examples
if(SAMPLES)

  # Downloads the cl.hpp file from Khronos
  file(DOWNLOAD https://www.khronos.org/registry/cl/api/1.1/cl.hpp ${clblast_SOURCE_DIR}/samples/cl.hpp)

  # Adds sample programs (C++)
  foreach(SAMPLE ${SAMPLE_PROGRAMS_CPP})
    add_executable(clblast_sample_${SAMPLE} samples/${SAMPLE}.cc)
    target_link_libraries(clblast_sample_${SAMPLE} clblast ${OPENCL_LIBRARIES})
    install(TARGETS clblast_sample_${SAMPLE} DESTINATION bin)
  endforeach()

  # Adds sample programs (C)
  foreach(SAMPLE ${SAMPLE_PROGRAMS_C})
    add_executable(clblast_sample_${SAMPLE}_c samples/${SAMPLE}.c)
    target_link_libraries(clblast_sample_${SAMPLE}_c clblast ${OPENCL_LIBRARIES})
    install(TARGETS clblast_sample_${SAMPLE}_c DESTINATION bin)
  endforeach()

endif()

# ==================================================================================================

# This section contains all the code related to the tuners. These tuners require the presence of
# the CLTune library (not included as part of the source).
if(TUNERS)

  # Includes CLTune
  include_directories(${CLTUNE_INCLUDE_DIRS})

  # Adds tuning executables
  foreach(KERNEL ${KERNELS})
    add_executable(clblast_tuner_${KERNEL} src/tuning/${KERNEL}.cc)
    target_link_libraries(clblast_tuner_${KERNEL} clblast ${CLTUNE_LIBRARIES} ${OPENCL_LIBRARIES})
    install(TARGETS clblast_tuner_${KERNEL} DESTINATION bin)
  endforeach()

  # Adds 'alltuners' target: runs all tuners for all precisions
  set(ALLTUNERS )
  set(ALLTUNERSDEPENDS )
  foreach(KERNEL ${KERNELS})
    foreach(PRECISION ${PRECISIONS})
      set(ALLTUNERS ${ALLTUNERS} COMMAND clblast_tuner_${KERNEL} -precision ${PRECISION} ${DEVICEPLATFORM})
    endforeach()
    set(ALLTUNERSDEPENDS clblast_tuner_${KERNEL})
  endforeach()
  add_custom_target(alltuners ${ALLTUNERS} DEPENDS ${ALLTUNERSDEPENDS})

endif()

# ==================================================================================================

# Down from here is all test (performance and correctness) related. Note that these tests require
# the presence of clBLAS and/or a BLAS library to act as a reference.
if(TESTS)

  # Sets the specifics for the reference BLAS libraries
  set(REF_INCLUDES )
  set(REF_LIBRARIES )
  if(CLBLAS_FOUND)
    set(REF_INCLUDES ${REF_INCLUDES} ${CLBLAS_INCLUDE_DIRS})
    set(REF_LIBRARIES ${REF_LIBRARIES} ${CLBLAS_LIBRARIES})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      add_definitions(" /DCLBLAST_REF_CLBLAS")
    else()
      add_definitions(" -DCLBLAST_REF_CLBLAS")
    endif()
  endif()
  if(CBLAS_FOUND)
    set(REF_INCLUDES ${REF_INCLUDES} ${CBLAS_INCLUDE_DIRS})
    set(REF_LIBRARIES ${REF_LIBRARIES} ${CBLAS_LIBRARIES})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      add_definitions(" /DCLBLAST_REF_CBLAS")
    else()
      add_definitions(" -DCLBLAST_REF_CBLAS")
    endif()
  endif()

  # Sets the include directories
  include_directories(${clblast_SOURCE_DIR}/test ${REF_INCLUDES})

  # Creates the common correctness-tests objects (requires CMake 2.8.8)
  add_library(test_correctness_common OBJECT
              test/correctness/tester.cc test/correctness/testblas.cc)

  # Compiles the correctness-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(clblast_test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level1/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(clblast_test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level2/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(clblast_test_${ROUTINE} $<TARGET_OBJECTS:test_correctness_common>
                   test/correctness/routines/level3/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(clblast_test_${ROUTINE} clblast ${REF_LIBRARIES} ${OPENCL_LIBRARIES})
    install(TARGETS clblast_test_${ROUTINE} DESTINATION bin)
  endforeach()

  # Adds 'alltests' target: runs all tests
  set(ALLTESTS )
  set(ALLTESTSDEPENDS )
  foreach(ROUTINE ${ROUTINES})
    set(ALLTESTS ${ALLTESTS} COMMAND clblast_test_${ROUTINE} ${DEVICEPLATFORM})
    set(ALLTESTSDEPENDS clblast_test_${ROUTINE})
  endforeach()
  add_custom_target(alltests ${ALLTESTS} DEPENDS ${ALLTESTSDEPENDS})

  # Creates the common performance-tests objects (requires CMake 2.8.8)
  add_library(test_performance_common OBJECT test/performance/client.cc)

  # Compiles the performance-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(clblast_client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level1/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(clblast_client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level2/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(clblast_client_${ROUTINE} $<TARGET_OBJECTS:test_performance_common>
                   test/performance/routines/level3/${ROUTINE}.cc)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(clblast_client_${ROUTINE} clblast ${REF_LIBRARIES} ${OPENCL_LIBRARIES})
    install(TARGETS clblast_client_${ROUTINE} DESTINATION bin)
  endforeach()

endif()
# ==================================================================================================
