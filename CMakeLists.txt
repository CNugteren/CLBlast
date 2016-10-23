
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================

cmake_minimum_required(VERSION 2.8.11)

# Overrides for MSVC static runtime
set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c_flag_overrides.cmake)
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx_flag_overrides.cmake)

# CMake project details
project("clblast" C CXX)
set(clblast_VERSION_MAJOR 0)
set(clblast_VERSION_MINOR 9)
set(clblast_VERSION_PATCH 0)

# Options and their default values
option(BUILD_SHARED_LIBS "Build a shared (ON) or static library (OFF)" ON)
option(SAMPLES "Enable compilation of the examples" OFF)
option(TUNERS "Enable compilation of the tuners" OFF)
option(CLIENTS "Enable compilation of the clients to test and compare performance" OFF)
option(TESTS "Enable compilation of the correctness tests" OFF)

# Compile in verbose mode with additional diagnostic messages
option(VERBOSE "Compile in verbose mode for additional diagnostic messages" OFF)
if(VERBOSE)
  message("-- Building in verbose mode")
  add_definitions(-DVERBOSE)
endif()

# ==================================================================================================

# RPATH settings
set(CMAKE_MACOSX_RPATH 1)

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif(MSVC)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# DLL Settings
if(MSVC)
  if(BUILD_SHARED_LIBS)
    add_definitions(" /DCLBLAST_DLL")
  endif()
endif(MSVC)

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ox")
  set(FLAGS "${FLAGS} /wd4715")
else()
  set(FLAGS "-std=c++11")
  if(VERBOSE)
    set(FLAGS "${FLAGS} -O1 -g")
  else()
    set(FLAGS "${FLAGS} -O3")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
      set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0.0)
      # GCC does not support attributes on template arguments
      # in particular we hit this with the alignment attributes on cl_XXX types
      # which are then used to instantiate various templates in CLBlast
      set(FLAGS "${FLAGS} -Wno-ignored-attributes")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
    set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
    set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
    set(FLAGS "${FLAGS} -Wno-deprecated-declarations")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# C compiler settings (for the sample)
if(MSVC)
  set(CFLAGS "/Ox")
else()
  set(CFLAGS "-O3 -std=c99")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${clblast_SOURCE_DIR}/cmake/Modules/")

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
if(CLIENTS OR TESTS)
  find_package(clBLAS)
  find_package(CBLAS)
  if(NOT CLBLAS_FOUND AND NOT CBLAS_FOUND)
    if(TESTS)
      message(STATUS "Could NOT find clBLAS nor a CPU BLAS, disabling the compilation of the tests")
      set(TESTS OFF)
    endif()
    if(CLIENTS)
      message(STATUS "Could NOT find clBLAS nor a CPU BLAS, head-to-head performance comparison not supported in the clients")
    endif()
  endif()
endif()

# ==================================================================================================

# Sets the supported routines and the used kernels. New routines and kernels should be added here.
set(KERNELS copy_fast copy_pad transpose_fast transpose_pad xaxpy xdot xger
            xgemm xgemm_direct xgemv)
set(SAMPLE_PROGRAMS_CPP sgemm)
set(SAMPLE_PROGRAMS_C sasum dgemv sgemm haxpy cache)
set(LEVEL1_ROUTINES xswap xscal xcopy xaxpy xdot xdotu xdotc xnrm2 xasum xamax)
set(LEVEL2_ROUTINES xgemv xgbmv xhemv xhbmv xhpmv xsymv xsbmv xspmv xtrmv xtbmv xtpmv
                    xger xgeru xgerc xher xhpr xher2 xhpr2 xsyr xspr xsyr2 xspr2)
set(LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm)
set(LEVELX_ROUTINES xomatcopy)
set(ROUTINES ${LEVEL1_ROUTINES} ${LEVEL2_ROUTINES} ${LEVEL3_ROUTINES} ${LEVELX_ROUTINES})
set(PRECISIONS 32 64 3232 6464 16)

# ==================================================================================================

# Gathers all source-files
set(SOURCES
  src/database/database.cpp
  src/routines/common.cpp
  src/utilities/clblast_exceptions.cpp
  src/utilities/utilities.cpp
  src/cache.cpp
  src/clblast.cpp
  src/clblast_c.cpp
  src/routine.cpp
)
foreach(ROUTINE ${LEVEL1_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level1/${ROUTINE}.cpp)
endforeach()
foreach(ROUTINE ${LEVEL2_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level2/${ROUTINE}.cpp)
endforeach()
foreach(ROUTINE ${LEVEL3_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level3/${ROUTINE}.cpp)
endforeach()
foreach(ROUTINE ${LEVELX_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/levelx/${ROUTINE}.cpp)
endforeach()

# Creates and links the library
if(BUILD_SHARED_LIBS)
  add_library(clblast SHARED ${SOURCES})
else(BUILD_SHARED_LIBS)
  add_library(clblast STATIC ${SOURCES})
endif()

target_link_libraries(clblast ${OPENCL_LIBRARIES})

# Includes directories: CLBlast and OpenCL
target_include_directories(clblast PUBLIC
                           $<BUILD_INTERFACE:${clblast_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${clblast_SOURCE_DIR}/src>
                           $<INSTALL_INTERFACE:include>
                           ${OPENCL_INCLUDE_DIRS})

# Sets the proper __declspec(dllexport) keyword for Visual Studio when the library is built
if(MSVC)
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(clblast PRIVATE COMPILING_DLL=1) # requires at least CMake 2.8.11
  endif()
endif()

# Installs the library
install(TARGETS clblast EXPORT CLBlast DESTINATION lib)
install(FILES include/clblast.h DESTINATION include)
install(FILES include/clblast_c.h DESTINATION include)
install(FILES include/clblast_half.h DESTINATION include)

# Installs the config for find_package in dependent projects
install(EXPORT CLBlast DESTINATION lib/cmake/CLBLast FILE CLBlastConfig.cmake)

# Install pkg-config file on Linux
if(UNIX)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/clblast.pc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/clblast.pc" @ONLY IMMEDIATE)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/clblast.pc
            DESTINATION lib/pkgconfig)
endif()

# ==================================================================================================

# Sets a default platform ($CLBLAST_PLATFORM) and device ($CLBLAST_DEVICE) to run tuners and tests
set(DEVICEPLATFORM )
if(DEFINED ENV{CLBLAST_DEVICE})
  set(DEVICEPLATFORM ${DEVICEPLATFORM} -device $ENV{CLBLAST_DEVICE})
endif()
if(DEFINED ENV{CLBLAST_PLATFORM})
  set(DEVICEPLATFORM ${DEVICEPLATFORM} -platform $ENV{CLBLAST_PLATFORM})
endif()

# Optionally also provides other options to the tests such as -full_test ($CLBLAST_TEST_ARGUMENTS)
set(TEST_ARGUMENTS )
if(DEFINED ENV{CLBLAST_TEST_ARGUMENTS})
  set(TEST_ARGUMENTS $ENV{CLBLAST_TEST_ARGUMENTS})
endif()

# ==================================================================================================

# This section contains all the code related to the examples
if(SAMPLES)

  # Downloads the cl.hpp file from Khronos
  file(DOWNLOAD https://www.khronos.org/registry/cl/api/1.1/cl.hpp ${clblast_SOURCE_DIR}/samples/cl.hpp)

  # Adds sample programs (C++)
  foreach(SAMPLE ${SAMPLE_PROGRAMS_CPP})
    add_executable(clblast_sample_${SAMPLE} samples/${SAMPLE}.cpp)
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

  # Visual Studio requires the sources of non-exported objects/libraries
  set(TUNERS_COMMON )
  if(MSVC)
    set(TUNERS_COMMON ${TUNERS_COMMON} src/utilities/utilities.cpp)
  endif()

  # Adds tuning executables
  foreach(KERNEL ${KERNELS})
    add_executable(clblast_tuner_${KERNEL} ${TUNERS_COMMON} src/tuning/kernels/${KERNEL}.cpp)
    target_link_libraries(clblast_tuner_${KERNEL} clblast ${CLTUNE_LIBRARIES} ${OPENCL_LIBRARIES})
    target_include_directories(clblast_tuner_${KERNEL} PUBLIC ${CLTUNE_INCLUDE_DIRS})
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

# Section for the tests: common part for both performance ('CLIENTS') and correctness ('TESTS')
if(CLIENTS OR TESTS)

  # Sets the specifics for the reference BLAS libraries
  set(REF_INCLUDES )
  set(REF_LIBRARIES )
  if(CLBLAS_FOUND)
    find_package(Threads)
    set(REF_LIBRARIES ${REF_LIBRARIES} ${CLBLAS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    set(REF_INCLUDES ${REF_INCLUDES} ${CLBLAS_INCLUDE_DIRS})
    if(MSVC)
      add_definitions(" /DCLBLAST_REF_CLBLAS")
    else()
      add_definitions(" -DCLBLAST_REF_CLBLAS")
    endif()
  endif()
  if(CBLAS_FOUND)
    set(REF_INCLUDES ${REF_INCLUDES} ${CBLAS_INCLUDE_DIRS})
    set(REF_LIBRARIES ${REF_LIBRARIES} ${CBLAS_LIBRARIES})
    if(MSVC)
      add_definitions(" /DCLBLAST_REF_CBLAS")
    else()
      add_definitions(" -DCLBLAST_REF_CBLAS")
    endif()
  endif()

endif()

# ==================================================================================================

# Section for the performance tests (i.e. the client). These compare against optionally a reference
# library, either clBLAS or a CPU BLAS.
if(CLIENTS)

  # Visual Studio requires the sources of non-exported objects/libraries
  set(CLIENTS_COMMON )
  if(MSVC)
    set(CLIENTS_COMMON ${CLIENTS_COMMON} src/utilities/utilities.cpp test/performance/client.cpp)
  else()
    # Creates the common performance-tests objects (requires CMake 2.8.8)
    add_library(test_performance_common OBJECT test/performance/client.cpp)

    # Adds CLBlast's interface include paths because we can't link to CLBlast here
    target_include_directories(test_performance_common PRIVATE
                               $<TARGET_PROPERTY:clblast,INTERFACE_INCLUDE_DIRECTORIES>
                               ${clblast_SOURCE_DIR} ${REF_INCLUDES})
    set(CLIENTS_COMMON ${CLIENTS_COMMON} $<TARGET_OBJECTS:test_performance_common>)
  endif()

  # Compiles the performance-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(clblast_client_${ROUTINE} ${CLIENTS_COMMON}
                   test/performance/routines/level1/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(clblast_client_${ROUTINE} ${CLIENTS_COMMON}
                   test/performance/routines/level2/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(clblast_client_${ROUTINE} ${CLIENTS_COMMON}
                   test/performance/routines/level3/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVELX_ROUTINES})
    add_executable(clblast_client_${ROUTINE} ${CLIENTS_COMMON}
                   test/performance/routines/levelx/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(clblast_client_${ROUTINE} clblast ${REF_LIBRARIES} ${OPENCL_LIBRARIES})
    target_include_directories(clblast_client_${ROUTINE} PUBLIC ${clblast_SOURCE_DIR} ${REF_INCLUDES})
    install(TARGETS clblast_client_${ROUTINE} DESTINATION bin)
  endforeach()

endif()

# ==================================================================================================

# Section for the correctness tests. Note that these tests require the presence of clBLAS and/or a
# CPU BLAS library to act as a reference.
if(TESTS)
  enable_testing()

  # Visual Studio requires the sources of non-exported objects/libraries
  set(TESTS_COMMON )
  if(MSVC)
    set(TESTS_COMMON ${TESTS_COMMON} src/utilities/utilities.cpp
        test/correctness/tester.cpp test/correctness/testblas.cpp)
  else()
    # Creates the common correctness-tests objects (requires CMake 2.8.8)
    add_library(test_correctness_common OBJECT
                test/correctness/tester.cpp test/correctness/testblas.cpp)
    target_include_directories(test_correctness_common PUBLIC
                               $<TARGET_PROPERTY:clblast,INTERFACE_INCLUDE_DIRECTORIES>
                               ${clblast_SOURCE_DIR} ${REF_INCLUDES})
    set(TESTS_COMMON ${TESTS_COMMON} $<TARGET_OBJECTS:test_correctness_common>)
  endif()

  # Compiles the correctness-tests
  foreach(ROUTINE ${LEVEL1_ROUTINES})
    add_executable(clblast_test_${ROUTINE} ${TESTS_COMMON}
                   test/correctness/routines/level1/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVEL2_ROUTINES})
    add_executable(clblast_test_${ROUTINE} ${TESTS_COMMON}
                   test/correctness/routines/level2/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVEL3_ROUTINES})
    add_executable(clblast_test_${ROUTINE} ${TESTS_COMMON}
                   test/correctness/routines/level3/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${LEVELX_ROUTINES})
    add_executable(clblast_test_${ROUTINE} ${TESTS_COMMON}
                   test/correctness/routines/levelx/${ROUTINE}.cpp)
  endforeach()
  foreach(ROUTINE ${ROUTINES})
    target_link_libraries(clblast_test_${ROUTINE} clblast ${REF_LIBRARIES} ${OPENCL_LIBRARIES})
    install(TARGETS clblast_test_${ROUTINE} DESTINATION bin)
    target_include_directories(clblast_test_${ROUTINE} PUBLIC ${clblast_SOURCE_DIR} ${REF_INCLUDES})
    add_test(clblast_test_${ROUTINE} clblast_test_${ROUTINE} ${DEVICEPLATFORM} ${TEST_ARGUMENTS})
  endforeach()

  # Adds 'alltests' target: runs all tests
  set(ALLTESTS )
  set(ALLTESTSDEPENDS )
  foreach(ROUTINE ${ROUTINES})
    set(ALLTESTS ${ALLTESTS} COMMAND clblast_test_${ROUTINE} ${DEVICEPLATFORM} ${TEST_ARGUMENTS})
    set(ALLTESTSDEPENDS clblast_test_${ROUTINE})
  endforeach()
  add_custom_target(alltests ${ALLTESTS} DEPENDS ${ALLTESTSDEPENDS})

endif()

# ==================================================================================================
