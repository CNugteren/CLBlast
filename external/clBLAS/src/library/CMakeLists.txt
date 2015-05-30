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

set(SRC_BLAS
    blas/init.c
    blas/impl.c
    blas/scimage.c
    blas/xgemv.c
    blas/xsymv.c
    blas/xgemm.c
    blas/xtrmm.c
    blas/xtrsm.c
    blas/xsyrk.c
    blas/xsyr2k.c
    blas/xtrmv.c
    blas/xtrsv.c
    blas/xsymm.c
	blas/xgemm2.c
    blas/xger.c
	blas/xsyr.c
	blas/xsyr2.c
	blas/xher.c
	blas/xher2.c
	blas/xhemv.c
	blas/xhemm.c
	blas/xherk.c
	blas/xhpmv.c
	blas/xspmv.c
	blas/xgbmv.c
	blas/xtbmv.c
	blas/xshbmv.c
	blas/xtbsv.c
	blas/xher2k.c
	blas/xswap.c
	blas/xscal.c
	blas/xcopy.c
	blas/xaxpy.c
	blas/xdot.c
	blas/xrotg.c
	blas/xrotmg.c
	blas/xrot.c
	blas/xrotm.c
    blas/ixamax.c
	blas/xnrm2.c
    blas/xasum.c
)

set(SRC_BLAS_HEADERS
    blas/include/blas_funcs.h
    blas/include/matrix_dims.h
    blas/include/matrix_props.h
    blas/include/blas_mempat.h
    blas/include/clblas-internal.h
    blas/include/solution_seq.h
    blas/include/events.h
)

set(SRC_BLAS_GENERIC
    blas/generic/common.c
    blas/generic/blas_funcs.c
    blas/generic/events.c
    blas/generic/matrix_props.c
    blas/generic/matrix_dims.c
    blas/generic/kdump.c
    blas/generic/solution_assert.c
    blas/generic/solution_seq.c
    blas/generic/solution_seq_make.c
    blas/generic/problem_iter.c
    blas/generic/kernel_extra.c)

set(SRC_BLAS_GENS
    blas/gens/gen_init.c
    blas/gens/blas_kgen.c
	blas/gens/blas_subgroup.c
    blas/gens/gen_helper.c
    blas/gens/tilemul.c
    blas/gens/fetch.c
    blas/gens/tile.c
    blas/gens/tile_iter.c
    blas/gens/decomposition.c
    blas/gens/gemv.c
    blas/gens/symv.c
    blas/gens/gemm.c
    blas/gens/trmm.c
    blas/gens/trsm.c
    blas/gens/syrxk.c
    blas/gens/trxm_common.c
    blas/gens/trsm_kgen.c
    blas/gens/xxmv_common.c
    blas/gens/legacy/blas_kgen_legacy.c
    blas/gens/legacy/gen_helper_legacy.c
    blas/gens/legacy/trxm_common_legacy.c
    blas/gens/legacy/trsm_kgen_legacy.c
    blas/gens/legacy/blkmul.c
    blas/gens/legacy/gemm_lds.c
    blas/gens/legacy/gemm_img.c
    blas/gens/legacy/trmm_lds.c
    blas/gens/legacy/trmm_img.c
    blas/gens/legacy/trsm_lds.c
    blas/gens/legacy/trsm_img.c
    blas/gens/legacy/trsm_cached_lds.c

	blas/gens/trmv_reg.cpp
	blas/gens/ger_lds.cpp
	blas/gens/trsv_trtri.cpp
	blas/gens/trsv_gemv.cpp
	blas/gens/kprintf.cpp
	blas/gens/syr_lds.cpp
	blas/gens/her_lds.cpp
	blas/gens/syr2_lds.cpp
	blas/gens/her2_lds.cpp
	blas/gens/symm_cached.cpp
	blas/gens/gemm_cached.cpp
	blas/gens/gemm_tail_cached.cpp
	blas/gens/gbmv.cpp
	blas/gens/tuned_numbers.c
	blas/gens/swap_reg.cpp
    blas/gens/scal_reg.cpp
    blas/gens/copy_reg.cpp
    blas/gens/axpy_reg.cpp
    blas/gens/dot.cpp
    blas/gens/reduction.cpp
    blas/gens/rotg_reg.cpp
    blas/gens/rotmg_reg.cpp
    blas/gens/rotm_reg.cpp
    blas/gens/iamax.cpp
    blas/gens/nrm2.cpp
    blas/gens/asum.cpp
)

set (SRC_CL_TEMPLATES
    gemm.cl
    gemm_helper.cl
    gbmv.cl
    ger.cl
    her.cl
    symm_helper.cl
    syr2_her2.cl
    syr_her.cl
    trsv.cl
    her2.cl
    symm.cl
    syr2.cl
    syr.cl
    trmv.cl
    trsv_gemv.cl
    swap.cl
    scal.cl
    copy.cl
    axpy.cl
    dot.cl
    reduction.cl
    rotg.cl
    rotmg.cl
    rotm.cl
    iamax.cl
    nrm2.cl
    asum.cl
)

set(SRC_BLAS_GENERIC_HEADERS
    blas/generic/solution_assert.h
    blas/generic/problem_iter.h
)

set(SRC_BLAS_GENS_HEADERS
    blas/gens/fetch.h
    blas/gens/blas_kgen.h
	blas/gens/blas_subgroup.h
    blas/gens/gen_helper.h
    blas/gens/init.h
    blas/gens/trxm_common.h
    blas/gens/trsm_kgen.h
    blas/gens/xxmv_common.h
    blas/gens/tile.h
    blas/gens/tile_iter.h
    blas/gens/tuned_numbers.h
)

set(SRC_COMMON
    common/list.c
    common/clkern.c
    common/kern_cache.c
    common/kerngen_core.c
    common/kgen_basic.c
    common/kgen_loop_helper.c
    common/kgen_guard.c
    common/misc.c
    common/devinfo.c
    common/devinfo-cache.c
    common/mutex.c
    common/trace_malloc.c
)

set(SRC_COMMON_GENS
    common/gens/dblock_kgen.c
)

set(SRC_TOOLS
    tools/tune/toolslib.c
    tools/tune/fileio.c
    tools/tune/dimension.c
    tools/tune/storage_init.c
    tools/tune/storage_io.c
    tools/tune/storage_data.c
)

set(CLBLAS_SOURCES
    ${SRC_COMMON} ${SRC_COMMON_GENS} ${SRC_BLAS} ${SRC_BLAS_GENERIC}
    ${SRC_BLAS_GENS} ${SRC_TOOLS} ../clBLAS.def
)
set(GLOBAL_HEADERS
    ${clBLAS_SOURCE_DIR}/clBLAS.h
    ${clBLAS_SOURCE_DIR}/clBLAS-complex.h 
    ${clBLAS_SOURCE_DIR}/include/clkern.h
    ${clBLAS_SOURCE_DIR}/include/cltypes.h
    ${clBLAS_SOURCE_DIR}/include/dblock_kgen.h
    ${clBLAS_SOURCE_DIR}/include/defbool.h
    ${clBLAS_SOURCE_DIR}/include/devinfo.h
    ${clBLAS_SOURCE_DIR}/include/dis_warning.h
    ${clBLAS_SOURCE_DIR}/include/kern_cache.h
    ${clBLAS_SOURCE_DIR}/include/kernel_extra.h
    ${clBLAS_SOURCE_DIR}/include/kerngen.h
    ${clBLAS_SOURCE_DIR}/include/list.h
    ${clBLAS_SOURCE_DIR}/include/mempat.h
    ${clBLAS_SOURCE_DIR}/include/msvc.h
    ${clBLAS_SOURCE_DIR}/include/mutex.h
    ${clBLAS_SOURCE_DIR}/include/solver.h
)

source_group(common FILES ${SRC_COMMON})
source_group(common\\gens FILES ${SRC_COMMON_GENS})
source_group(blas FILES ${SRC_BLAS})
source_group(blas\\include FILES ${SRC_BLAS_HEADERS})
source_group(blas\\generic FILES ${SRC_BLAS_GENERIC})
source_group(blas\\gens FILES ${SRC_BLAS_GENS}
    ${SRC_BLAS_GENS_HEADERS})

include_directories(${OPENCL_INCLUDE_DIRS}
    ${clBLAS_SOURCE_DIR}
    ${clBLAS_SOURCE_DIR}/include
    ${clBLAS_SOURCE_DIR}/library/blas/include
    ${clBLAS_SOURCE_DIR}/library/tools/tune
    ${clBLAS_BINARY_DIR}/include
)

option( BLAS_DUMP_CLBLAS_KERNELS "Force the library to dump OpenCL kernels to disk" OFF )
if( BLAS_DUMP_CLBLAS_KERNELS )
    add_definitions( -DDUMP_CLBLAS_KERNELS )
endif()

option( BLAS_KEEP_KERNEL_SOURCES "Prevent the library from stripping source from kernels" OFF )
if( BLAS_KEEP_KERNEL_SOURCES )
    add_definitions( -DKEEP_CLBLAS_KERNEL_SOURCES )
endif()

option( BLAS_TRACE_MALLOC "Simple functionality to track memory leaks" OFF )
if( BLAS_TRACE_MALLOC )
    add_definitions( -DTRACE_MALLOC )
endif()

option( BLAS_PRINT_BUILD_ERRORS "Enable printing of OpenCL compiler errors on stdout" ON )
if( BLAS_PRINT_BUILD_ERRORS )
    add_definitions( -DPRINT_BUILD_ERRORS )
endif()

include( ExternalProject )
ExternalProject_Add( tplgen
    URL "${PROJECT_SOURCE_DIR}/library/tools/tplgen"
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property( tplgen binary_dir )

set( tplgenBinaryDir "" )
if( CMAKE_COMPILER_IS_GNUCXX )
    set( tplgenBinaryDir ${binary_dir} )
else()
    set( tplgenBinaryDir "${binary_dir}/${CMAKE_CFG_INTDIR}" )
endif()

add_custom_target( GENERATE_CLT
    COMMAND ${tplgenBinaryDir}/tplgen -o ${clBLAS_BINARY_DIR}/include ${SRC_CL_TEMPLATES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/library/blas/gens/clTemplates
)

add_dependencies( GENERATE_CLT tplgen )

if( CMAKE_COMPILER_IS_GNUCC )
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/clBLAS.pc.in
                    ${CMAKE_CURRENT_BINARY_DIR}/clBLAS.pc @ONLY )

    install( FILES ${CMAKE_CURRENT_BINARY_DIR}/clBLAS.pc
             DESTINATION lib${SUFFIX_LIB}/pkgconfig )
endif( )

add_library(clBLAS ${CLBLAS_SOURCES} ${GLOBAL_HEADERS} ${SRC_BLAS_HEADERS} ${SRC_BLAS_GENS_HEADERS})
add_dependencies(clBLAS GENERATE_CLT)
set_target_properties(clBLAS PROPERTIES VERSION ${clBLAS_VERSION})
set_target_properties(clBLAS PROPERTIES SOVERSION ${clBLAS_SOVERSION})
set_target_properties( clBLAS PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
target_link_libraries(clBLAS ${OPENCL_LIBRARIES} ${MATH_LIBRARY})

# CPack configuration; include the executable into the package
install( TARGETS clBLAS
                EXPORT Library
                RUNTIME DESTINATION bin${SUFFIX_BIN}
                LIBRARY DESTINATION lib${SUFFIX_LIB}
		ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
		)

# For debug builds, include the debug runtimes into the package for testing on non-developer machines
set( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP true )
set( CMAKE_INSTALL_DEBUG_LIBRARIES true )
set( CMAKE_INSTALL_DEBUG_LIBRARIES_ONLY true )

if( WIN32 )
    set( CLBLAS_RUNTIME_DESTINATION bin${SUFFIX_BIN} )
else( )
    set( CLBLAS_RUNTIME_DESTINATION lib${SUFFIX_LIB} )
endif( )

include( InstallRequiredSystemLibraries )

# Install necessary runtime files for debug builds
install(    PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            CONFIGURATIONS Debug
            DESTINATION ${CLBLAS_RUNTIME_DESTINATION} )

# Install all *.pdb files for debug builds
install(    DIRECTORY ${PROJECT_BINARY_DIR}/staging/
            DESTINATION ${CLBLAS_RUNTIME_DESTINATION}
            OPTIONAL
            CONFIGURATIONS Debug
            FILES_MATCHING PATTERN "*.pdb" )

# Install a snapshot of the source as it was for this build; useful for the .pdb's
install(    DIRECTORY ${PROJECT_SOURCE_DIR}
            DESTINATION ${CLBLAS_RUNTIME_DESTINATION}
            OPTIONAL
            CONFIGURATIONS Debug )
