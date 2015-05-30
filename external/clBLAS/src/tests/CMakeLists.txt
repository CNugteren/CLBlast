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

set(SRC_COMMON
    cmdline.c
    common.cpp
    clBLAS-wrapper.cpp
    BlasBase.cpp
)

set(SRC_COMMON_TIMER
    timer.c
)

# group of sources with reference implementation stuff
set (SRC_COMMON_REFIMPL
    blas.c
    blas-cblas.c
    blas-wrapper.cpp
)

set(SRC_CORR
    correctness/blas-lapack.c
    correctness/BlasBase-corr.cpp
    correctness/corr-gemm.cpp
    correctness/corr-trmm.cpp
    correctness/corr-trsm.cpp
    correctness/corr-gemv.cpp
    correctness/corr-symv.cpp
    correctness/corr-spmv.cpp
    correctness/corr-syr2k.cpp
    correctness/corr-syrk.cpp
    correctness/corr-trmv.cpp
    correctness/corr-tpmv.cpp
    correctness/corr-trsv.cpp
	correctness/corr-symm.cpp
	correctness/corr-gemm2.cpp
	correctness/corr-ger.cpp
	correctness/corr-gerc.cpp
    correctness/corr-her.cpp
	correctness/corr-her2.cpp
	correctness/corr-syr.cpp
    correctness/corr-spr.cpp
	correctness/corr-syr2.cpp
	correctness/corr-hemv.cpp
	correctness/corr-hpmv.cpp
	correctness/corr-hemm.cpp
	correctness/corr-herk.cpp
    correctness/corr-tpsv.cpp
    correctness/corr-hpr.cpp
	correctness/corr-hpr2.cpp
	correctness/corr-spr2.cpp
	correctness/corr-gbmv.cpp
    correctness/corr-hbmv.cpp
	correctness/corr-tbmv.cpp
    correctness/corr-tbsv.cpp
    correctness/corr-sbmv.cpp
    correctness/corr-her2k.cpp
    correctness/corr-scal.cpp
	correctness/corr-swap.cpp
	correctness/corr-copy.cpp
    correctness/corr-axpy.cpp
	correctness/corr-dot.cpp
    correctness/corr-dotc.cpp
    correctness/corr-rotg.cpp
    correctness/corr-rotm.cpp
	correctness/corr-rot.cpp
    correctness/corr-rotmg.cpp
    correctness/corr-nrm2.cpp
    correctness/corr-asum.cpp
    correctness/corr-iamax.cpp
    correctness/test-correctness.cpp
    correctness/tcase-filter.cpp
)

set(SRC_PERF
    performance/PerformanceRecorder.cpp
    performance/PerformanceTest.cpp
    performance/TrxmPerformanceTest.cpp
    performance/BlasBase-perf.cpp
    performance/perf-gemm.cpp
    performance/perf-gemm2.cpp
    performance/perf-gemv.cpp
    performance/perf-syr2k.cpp
    performance/perf-syrk.cpp
    performance/perf-symv.cpp
    performance/perf-spmv.cpp
    performance/perf-trmm.cpp
    performance/perf-trsm.cpp
    performance/perf-trmv.cpp
    performance/perf-tpmv.cpp
    performance/perf-trsv.cpp
    performance/perf-symm.cpp
    performance/perf-ger.cpp
    performance/perf-gerc.cpp
    performance/perf-syr.cpp
    performance/perf-spr.cpp
	performance/perf-her.cpp
	performance/perf-her2.cpp
    performance/perf-syr2.cpp
	performance/perf-hemm.cpp
    performance/perf-hemv.cpp
    performance/perf-hpmv.cpp
    performance/perf-herk.cpp
	performance/perf-tpsv.cpp
    performance/perf-hpr.cpp
	performance/perf-hpr2.cpp
    performance/perf-spr2.cpp
    performance/perf-sbmv.cpp
    performance/perf-gbmv.cpp
    performance/perf-hbmv.cpp
    performance/perf-tbmv.cpp
    performance/perf-tbsv.cpp
    performance/perf-her2k.cpp
    performance/perf-scal.cpp
    performance/perf-swap.cpp
	performance/perf-copy.cpp
    performance/perf-axpy.cpp
	performance/perf-dot.cpp
    performance/perf-dotc.cpp
    performance/perf-rotg.cpp
    performance/perf-rotm.cpp
	performance/perf-rot.cpp
    performance/perf-rotmg.cpp
    performance/perf-nrm2.cpp
    performance/perf-asum.cpp
    performance/perf-iamax.cpp
	performance/test-performance.cpp
)

set(SRC_FUNC
   functional/func-error.cpp
   functional/func-event.cpp
   functional/func-thread.cpp
   functional/func-queue.cpp
   #functional/func-images.cpp
   functional/test-functional.cpp
   functional/BlasBase-func.cpp
)

set(TESTS_HEADERS
    ${clBLAS_SOURCE_DIR}/clBLAS.h
    ${clBLAS_SOURCE_DIR}/clBLAS-complex.h
	${clBLAS_SOURCE_DIR}/include/cltypes.h
	${clBLAS_SOURCE_DIR}/include/defbool.h
    include/blas-internal.h
    include/blas-cblas.h
    include/blas-wrapper.h
    include/clBLAS-wrapper.h
    include/cmdline.h
    include/BlasBase.h
    include/common.h
    include/BlasBase.h
    include/gemm.h
    include/trmm.h
    include/tpmv.h
    include/trsm.h
    include/gemv.h
    include/symv.h
    include/spmv.h
    include/syr2k.h
    include/syrk.h
    include/trmv.h
    include/trsv.h
	include/symm.h
	include/ger.h
    include/gerc.h
    include/syr.h
    include/spr.h
    include/her.h
	include/her2.h
    include/syr2.h
	include/hemv.h
	include/hpmv.h
	include/hemm.h
	include/herk.h
    include/tpsv.h
    include/hpr.h
	include/hpr2.h
    include/spr2.h
    include/gbmv.h
    include/hbmv.h
    include/tbmv.h
    include/tbsv.h
	include/copy.h
    include/sbmv.h
    include/dot.h
    include/dotc.h
    include/her2k.h
    include/scal.h
	include/swap.h
    include/axpy.h
    include/rotg.h
    include/rotm.h
	include/rot.h
    include/asum.h
    include/rotmg.h
    include/nrm2.h
    include/iamax.h
	include/blas-math.h
    include/blas-random.h
    include/matrix.h
    include/timer.h
)

set(CORR_HEADERS
    correctness/blas-lapack.h
    correctness/trsm-delta.h
    correctness/tcase-filter.h
    correctness/delta.h
	correctness/trsv-delta.h
)

set(PERF_HEADERS
    performance/PerformanceTest.h
    performance/PerformanceRecorder.h
)

set(FUNC_HEADERS
   functional/func.h
)

# Setup Visual Studio file tabs
source_group(correctness FILES ${SRC_CORR} ${CORR_HEADERS})
source_group(performance FILES ${SRC_PERF} ${PERF_HEADERS})
source_group(functional  FILES ${SRC_FUNC} ${FUNC_HEADERS})

# FIXME: it's a temporary solution to workaround segfault in clGetProgramInfo()
# at paramVal = CL_PROGRAM_BINARIES and several devices in the context
add_definitions( -DTEST_WITH_SINGLE_DEVICE )

# vs11 needs std::tuples compiled with 10 parameters by default
# NOTE: this assumes that googletest is compiled with the same preprocessor macro; they must match
if( MSVC11 )
	add_definitions( "/D_VARIADIC_MAX=10" )
endif()

# Having problems on build server, compiling gtest headers with -pedantic; disabling detection of long long
# http://code.google.com/p/googletest/issues/detail?id=334
if( CMAKE_COMPILER_IS_GNUCXX )
	add_definitions( -Wno-long-long )
endif( )

if( CMAKE_Fortran_COMPILER_ID STREQUAL "PGI" )
	message( STATUS "Detected PGI Fortran compiler." )

	# By default, -Mipa=fast is used, and this does not mix well with the cl compiler
	string( REPLACE "-Mipa=fast" "" CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} )
	
	# In windows, dynamically link to the C runtime, and tell fortran linker to not include default main subroutine
	if( WIN32 )
		set( CMAKE_EXE_LINKER_FLAGS "-Bdynamic -Mnostartup ${CMAKE_EXE_LINKER_FLAGS}" )
	endif( )
endif( )

# Library with functions for time measurement. In Windows they are included automatically
if(UNIX)
    if(NOT APPLE)
        set(TIME_LIBRARY "rt")
    endif()
    set(THREAD_LIBRARY "pthread")
endif()

# This logic supports the build server, if it compiles the runtime seperately from the test programs
# It stitches together a path to a previously built static library, based on our 'make install' logic
# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
set( runtime.library "${CMAKE_INSTALL_PREFIX}/lib${SUFFIX_LIB}" )

if( WIN32 )
	set( runtime.library "${runtime.library}/import/clBLAS${CMAKE_STATIC_LIBRARY_SUFFIX}" )
else( )
	set( runtime.library "${runtime.library}/${CMAKE_SHARED_LIBRARY_PREFIX}clBLAS${CMAKE_SHARED_LIBRARY_SUFFIX}" )
endif( )

if( GTEST_FOUND )
    if( CORR_TEST_WITH_ACML AND ACML_FOUND )
		include_directories(${OPENCL_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
			${clBLAS_SOURCE_DIR} ${ACML_INCLUDE_DIRS}
			${clBLAS_SOURCE_DIR}/tests/include  ${clBLAS_SOURCE_DIR}/include)

		add_definitions(-DCORR_TEST_WITH_ACML)
	    
	    add_executable(test-correctness ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
									    ${CORR_HEADERS} ${TESTS_HEADERS})
        set_target_properties( test-correctness PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

	    add_executable(test-medium ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
							      ${CORR_HEADERS} ${TESTS_HEADERS})
	    set_target_properties(test-medium PROPERTIES COMPILE_DEFINITIONS MEDIUM_TESTS)
        set_target_properties( test-medium PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

	    add_executable(test-short ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
							      ${CORR_HEADERS} ${TESTS_HEADERS})
	    set_target_properties(test-short PROPERTIES COMPILE_DEFINITIONS SHORT_TESTS)
        set_target_properties( test-short PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

	    # The build server builds the library with gcc 4.1.2 to support Red Hat 5.5, but the test programs must be built with 
	    # gcc > 4.3.2 to support ACML.  
	    # If the runtime is being built by the project, use it, otherwise link to a runtime library specified in the install prefix
	    if( BUILD_RUNTIME )
		    target_link_libraries(test-correctness ${ACML_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
		    target_link_libraries(test-medium ${ACML_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
		    target_link_libraries(test-short ${ACML_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
	    else( )
		    target_link_libraries(test-correctness
			    ${ACML_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library})
		    target_link_libraries(test-medium
			    ${ACML_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library})
		    target_link_libraries(test-short
			    ${ACML_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library})
	    endif( )
	else( )
		#Link against the netlib reference library
		include_directories(${OPENCL_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
			${clBLAS_SOURCE_DIR} ${clBLAS_SOURCE_DIR}/tests/include ${clBLAS_SOURCE_DIR}/include})

		add_executable(test-correctness ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
						${CORR_HEADERS} ${TESTS_HEADERS})
        set_target_properties( test-correctness PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

		add_executable(test-medium ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
					  ${CORR_HEADERS} ${TESTS_HEADERS})
		set_target_properties( test-medium PROPERTIES COMPILE_DEFINITIONS MEDIUM_TESTS )
        set_target_properties( test-medium PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

		add_executable(test-short ${SRC_CORR} ${SRC_COMMON} ${SRC_COMMON_REFIMPL}
					  ${CORR_HEADERS} ${TESTS_HEADERS})
		set_target_properties( test-short PROPERTIES COMPILE_DEFINITIONS SHORT_TESTS )
        set_target_properties( test-short PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

		if( NOT CORR_TEST_WITH_ACML AND NOT WIN32 AND NOT APPLE)
			set_target_properties( test-correctness PROPERTIES LINKER_LANGUAGE Fortran )
			set_target_properties( test-medium PROPERTIES LINKER_LANGUAGE Fortran )
			set_target_properties( test-short PROPERTIES LINKER_LANGUAGE Fortran )
		endif( )
		
	    if( BUILD_RUNTIME )
			if( NETLIB_FOUND )
				target_link_libraries(test-correctness ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
				target_link_libraries(test-medium ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
				target_link_libraries(test-short ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
			else( )
				target_link_libraries(test-correctness ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
				target_link_libraries(test-medium ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
				target_link_libraries(test-short ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} clBLAS)
			endif( )
		else( )
			if( NETLIB_FOUND )
				target_link_libraries(test-correctness ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
				target_link_libraries(test-medium ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
				target_link_libraries(test-short ${Netlib_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
			else( )
				target_link_libraries(test-correctness ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
				target_link_libraries(test-medium ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
				target_link_libraries(test-short ${BLAS_LIBRARIES} ${GTEST_LIBRARIES} ${OPENCL_LIBRARIES} ${runtime.library} )
			endif( )
		endif( )
    endif( )
    
    set_property( TARGET test-correctness PROPERTY FOLDER "Test")
    set_property( TARGET test-medium PROPERTY FOLDER "Test")
    set_property( TARGET test-short PROPERTY FOLDER "Test")

    # CPack configuration; include the executable into the package
    install( TARGETS test-correctness test-medium test-short
            RUNTIME DESTINATION bin${SUFFIX_BIN}
            LIBRARY DESTINATION lib${SUFFIX_LIB}
            ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
            )
    
    get_target_property( testLocation test-correctness LOCATION )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/copyTestDependencies.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/copyTestDependencies.cmake"
        @ONLY
    )

    # Register script at run at install time to analyze the executable and copy dependencies into package
    install( SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/copyTestDependencies.cmake")
 
	if( ACML_FOUND )
		include_directories(${OPENCL_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
			${clBLAS_SOURCE_DIR} ${clBLAS_SOURCE_DIR}/tests/include  ${clBLAS_SOURCE_DIR}/include)

		add_definitions(-DPERF_TEST_WITH_ACML)
		include_directories(${ACML_INCLUDE_DIRS})
		add_executable(test-performance ${SRC_PERF} ${SRC_COMMON}
			${SRC_COMMON_TIMER} ${PERF_HEADERS} ${TESTS_HEADERS}
			${SRC_COMMON_REFIMPL})
		target_link_libraries(test-performance ${ACML_LIBRARIES})
        set_target_properties( test-performance PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )


		if( BUILD_RUNTIME )
			target_link_libraries(test-performance ${GTEST_LIBRARIES} ${TIME_LIBRARY} clBLAS)
		else()
			target_link_libraries( test-performance ${GTEST_LIBRARIES} ${TIME_LIBRARY} ${OPENCL_LIBRARIES} ${runtime.library} )
		endif()

        set_property( TARGET test-performance PROPERTY FOLDER "Test")

        # CPack configuration; include the executable into the package
        install( TARGETS test-performance
                RUNTIME DESTINATION bin${SUFFIX_BIN}
                LIBRARY DESTINATION lib${SUFFIX_LIB}
                ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
                )
	endif()

	include_directories(${OPENCL_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
		${clBLAS_SOURCE_DIR} ${clBLAS_SOURCE_DIR}/tests/include ${clBLAS_SOURCE_DIR}/include )

	add_executable(test-functional ${SRC_FUNC} ${SRC_COMMON} ${SRC_COMMON_TIMER}
								  ${FUNC_HEADERS} ${TESTS_HEADERS})
								  
    set_target_properties( test-functional PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
	if( BUILD_RUNTIME )
		target_link_libraries(test-functional ${GTEST_LIBRARIES} ${TIME_LIBRARY} ${THREAD_LIBRARY} clBLAS )
	else()
		target_link_libraries(test-functional ${GTEST_LIBRARIES} ${TIME_LIBRARY} ${THREAD_LIBRARY} ${OPENCL_LIBRARIES} ${runtime.library} )
	endif()

    set_property( TARGET test-functional PROPERTY FOLDER "Test")

    # CPack configuration; include the executable into the package
    install( TARGETS test-functional
            RUNTIME DESTINATION bin${SUFFIX_BIN}
            LIBRARY DESTINATION lib${SUFFIX_LIB}
            ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
            )
endif()
