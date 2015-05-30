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
cmake_minimum_required(VERSION 2.6)
project(clblas.samples)

# Configure
set(AMDAPPSDKROOT $ENV{AMDAPPSDKROOT}
    CACHE FILEPATH "ATI Stream SDK root path")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

if(TARGET_PLATFORM EQUAL 32 OR TARGET_PLATFORM EQUAL 64)
    set(TARGET_PLATFORM ${TARGET_PLATFORM} CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
else()
    if(CMAKE_SIZEOF_VOID_P MATCHES 8)
        set(TARGET_PLATFORM "64" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    else()
        set(TARGET_PLATFORM "32" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    endif()
endif()
message(STATUS "Target platform: ${TARGET_PLATFORM}-bit")
if(TARGET_PLATFORM EQUAL 32)
    set(_arch "x86" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)
else()
    set(_arch "x86_64" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
endif()

# Find OpenCL SDK
find_path(OPENCL_INCLUDE_DIRS CL/cl.h OpenCL/cl.h
    HINTS
        $ENV{AMDAPPSDKROOT}/include
        ${AMDAPPSDKROOT}/include
)
find_library(OPENCL_LIBRARIES OpenCL
    HINTS
        $ENV{AMDAPPSDKROOT}/lib/${_arch}
        ${AMDAPPSDKROOT}/lib/${_arch}
)
if(OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES)
    message(STATUS "Found OpenCL: ${OPENCL_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find OpenCL SDK")
endif()
mark_as_advanced(OPENCL_INCLUDE_DIRS OPENCL_LIBRARIES)

# Turn on maximum compiler verbosity
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-pedantic -Wall -Wextra
        -D_POSIX_C_SOURCE=199309L -D_XOPEN_SOURCE=500
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wstrict-prototypes" CACHE STRING
        "Default CFLAGS" FORCE)
    # Don't use -rpath.
    set(CMAKE_SKIP_RPATH ON CACHE BOOL "Skip RPATH" FORCE)

    set(CMAKE_C_FLAGS "-m${TARGET_PLATFORM} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-m${TARGET_PLATFORM} ${CMAKE_CXX_FLAGS}")

    if(TARGET_PLATFORM EQUAL 32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-builtin")
    endif()
endif()
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(SGEMV_SAMPLE_SRC example_sgemv.c)
set(SSYMV_SAMPLE_SRC example_ssymv.c)
set(SGEMM_SAMPLE_SRC example_sgemm.c)
set(STRMM_SAMPLE_SRC example_strmm.c)
set(STRSM_SAMPLE_SRC example_strsm.c)
set(SSYRK_SAMPLE_SRC example_ssyrk.c)
set(SSYR2K_SAMPLE_SRC example_ssyr2k.c)
set(STRMV_SAMPLE_SRC example_strmv.c)
set(STRSV_SAMPLE_SRC example_strsv.c)
set(SGER_SAMPLE_SRC example_sger.c)
set(SSYR_SAMPLE_SRC example_ssyr.c)
set(SSYR2_SAMPLE_SRC example_ssyr2.c)
set(SSYMM_SAMPLE_SRC example_ssymm.c)
set(CHER_SAMPLE_SRC example_cher.c)
set(CHEMM_SAMPLE_SRC example_chemm.cpp)
set(CHERK_SAMPLE_SRC example_cherk.cpp)
set(STPMV_SAMPLE_SRC example_stpmv.c)
set(CHPMV_SAMPLE_SRC example_chpmv.c)
set(STPSV_SAMPLE_SRC example_stpsv.c)
set(SSPMV_SAMPLE_SRC example_sspmv.c)
set(SSPR_SAMPLE_SRC example_sspr.c)
set(CHPR_SAMPLE_SRC example_chpr.c)
set(SSPR2_SAMPLE_SRC example_sspr2.c)
set(ZHPR2_SAMPLE_SRC example_zhpr2.c)
set(SGBMV_SAMPLE_SRC example_sgbmv.c)
set(STBMV_SAMPLE_SRC example_stbmv.c)
set(SSBMV_SAMPLE_SRC example_ssbmv.c)
set(CHBMV_SAMPLE_SRC example_chbmv.c)
set(STBSV_SAMPLE_SRC example_stbsv.c)
set(CHER2K_SAMPLE_SRC example_cher2k.c)
set(SSWAP_SAMPLE_SRC example_sswap.c)
set(SSCAL_SAMPLE_SRC example_sscal.c)
set(CSSCAL_SAMPLE_SRC example_csscal.c)
set(SCOPY_SAMPLE_SRC example_scopy.c)
set(SAXPY_SAMPLE_SRC example_saxpy.c)
set(SDOT_SAMPLE_SRC example_sdot.c)
set(SROTG_SAMPLE_SRC example_srotg.c)
set(SROTMG_SAMPLE_SRC example_srotmg.c)
set(SROT_SAMPLE_SRC example_srot.c)
set(SROTM_SAMPLE_SRC example_srotm.c)
set(iSAMAX_SAMPLE_SRC example_isamax.c)
set(SNRM2_SAMPLE_SRC example_snrm2.c)
set(SASUM_SAMPLE_SRC example_sasum.c)

include_directories( ${OPENCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../include )

# Set the OpenCL library include path depending on target platform
if( TARGET_PLATFORM EQUAL 64 )
    if( WIN32 )
	    link_directories( ${ATI_STREAM_SDK_ROOT}/lib/x86_64/ ${PROJECT_SOURCE_DIR}/../lib64/import )
    elseif( UNIX )
	    link_directories( ${ATI_STREAM_SDK_ROOT}/lib/x86_64/ ${PROJECT_SOURCE_DIR}/../lib64 )
    endif()
else()
    if( WIN32 )
	    link_directories( ${ATI_STREAM_SDK_ROOT}/lib/x86/ ${PROJECT_SOURCE_DIR}/../lib32/import )
    elseif( UNIX )
	    link_directories( ${ATI_STREAM_SDK_ROOT}/lib/x86/ ${PROJECT_SOURCE_DIR}/../lib32 )
    endif()
endif()

add_executable(example_sgemv ${SGEMV_SAMPLE_SRC})
target_link_libraries(example_sgemv ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssymv ${SSYMV_SAMPLE_SRC})
target_link_libraries(example_ssymv ${OPENCL_LIBRARIES} clblas)

add_executable(example_sgemm ${SGEMM_SAMPLE_SRC})
target_link_libraries(example_sgemm ${OPENCL_LIBRARIES} clblas)

add_executable(example_strmm ${STRMM_SAMPLE_SRC})
target_link_libraries(example_strmm ${OPENCL_LIBRARIES} clblas)

add_executable(example_strsm ${STRSM_SAMPLE_SRC})
target_link_libraries(example_strsm ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssyrk ${SSYRK_SAMPLE_SRC})
target_link_libraries(example_ssyrk ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssyr2k ${SSYR2K_SAMPLE_SRC})
target_link_libraries(example_ssyr2k ${OPENCL_LIBRARIES} clblas)

add_executable(example_strmv ${STRMV_SAMPLE_SRC})
target_link_libraries(example_strmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_strsv ${STRSV_SAMPLE_SRC})
target_link_libraries(example_strsv ${OPENCL_LIBRARIES} clblas)

add_executable(example_sger ${SGER_SAMPLE_SRC})
target_link_libraries(example_sger ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssyr ${SSYR_SAMPLE_SRC})
target_link_libraries(example_ssyr ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssyr2 ${SSYR2_SAMPLE_SRC})
target_link_libraries(example_ssyr2 ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssymm ${SSYMM_SAMPLE_SRC})
target_link_libraries(example_ssymm ${OPENCL_LIBRARIES} clblas)

add_executable(example_cher ${CHER_SAMPLE_SRC})
target_link_libraries(example_cher ${OPENCL_LIBRARIES} clblas)

add_executable(example_chemm ${CHEMM_SAMPLE_SRC})
target_link_libraries(example_chemm ${OPENCL_LIBRARIES} clblas)

add_executable(example_cherk ${CHERK_SAMPLE_SRC})
target_link_libraries(example_cherk ${OPENCL_LIBRARIES} clblas)

add_executable(example_stpmv ${STPMV_SAMPLE_SRC})
target_link_libraries(example_stpmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_chpmv ${CHPMV_SAMPLE_SRC})
target_link_libraries(example_chpmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_stpsv ${STPSV_SAMPLE_SRC})
target_link_libraries(example_stpsv ${OPENCL_LIBRARIES} clblas)

add_executable(example_sspmv ${SSPMV_SAMPLE_SRC})
target_link_libraries(example_sspmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_sspr ${SSPR_SAMPLE_SRC})
target_link_libraries(example_sspr ${OPENCL_LIBRARIES} clblas)

add_executable(example_chpr ${CHPR_SAMPLE_SRC})
target_link_libraries(example_chpr ${OPENCL_LIBRARIES} clblas)

add_executable(example_sspr2 ${SSPR2_SAMPLE_SRC})
target_link_libraries(example_sspr2 ${OPENCL_LIBRARIES} clblas)

add_executable(example_zhpr2 ${ZHPR2_SAMPLE_SRC})
target_link_libraries(example_zhpr2 ${OPENCL_LIBRARIES} clblas)

add_executable(example_sgbmv ${SGBMV_SAMPLE_SRC})
target_link_libraries(example_sgbmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_stbmv ${STBMV_SAMPLE_SRC})
target_link_libraries(example_stbmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_ssbmv ${SSBMV_SAMPLE_SRC})
target_link_libraries(example_ssbmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_chbmv ${CHBMV_SAMPLE_SRC})
target_link_libraries(example_chbmv ${OPENCL_LIBRARIES} clblas)

add_executable(example_stbsv ${STBSV_SAMPLE_SRC})
target_link_libraries(example_stbsv ${OPENCL_LIBRARIES} clblas)

add_executable(example_cher2k ${CHER2K_SAMPLE_SRC})
target_link_libraries(example_cher2k ${OPENCL_LIBRARIES} clblas)

add_executable(example_sswap ${SSWAP_SAMPLE_SRC})
target_link_libraries(example_sswap ${OPENCL_LIBRARIES} clblas)

add_executable(example_sscal ${SSCAL_SAMPLE_SRC})
target_link_libraries(example_sscal ${OPENCL_LIBRARIES} clblas)

add_executable(example_csscal ${CSSCAL_SAMPLE_SRC})
target_link_libraries(example_csscal ${OPENCL_LIBRARIES} clblas)

add_executable(example_scopy ${SCOPY_SAMPLE_SRC})
target_link_libraries(example_scopy ${OPENCL_LIBRARIES} clblas)

add_executable(example_saxpy ${SAXPY_SAMPLE_SRC})
target_link_libraries(example_saxpy ${OPENCL_LIBRARIES} clblas)

add_executable(example_sdot ${SDOT_SAMPLE_SRC})
target_link_libraries(example_sdot ${OPENCL_LIBRARIES} clblas)

add_executable(example_srotg ${SROTG_SAMPLE_SRC})
target_link_libraries(example_srotg ${OPENCL_LIBRARIES} clblas)

add_executable(example_srotmg ${SROTMG_SAMPLE_SRC})
target_link_libraries(example_srotmg ${OPENCL_LIBRARIES} clblas)

add_executable(example_srot ${SROT_SAMPLE_SRC})
target_link_libraries(example_srot ${OPENCL_LIBRARIES} clblas)

add_executable(example_srotm ${SROTM_SAMPLE_SRC})
target_link_libraries(example_srotm ${OPENCL_LIBRARIES} clblas)

add_executable(example_isamax ${iSAMAX_SAMPLE_SRC})
target_link_libraries(example_isamax ${OPENCL_LIBRARIES} clblas)

add_executable(example_snrm2 ${SNRM2_SAMPLE_SRC})
target_link_libraries(example_snrm2 ${OPENCL_LIBRARIES} clblas)

add_executable(example_sasum ${SASUM_SAMPLE_SRC})
target_link_libraries(example_sasum ${OPENCL_LIBRARIES} clblas)
