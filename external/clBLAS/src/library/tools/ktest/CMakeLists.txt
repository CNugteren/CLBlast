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


set(KTEST_SRC
    config.cpp
    config-cmdline.cpp
    ktest.cpp
    main.cpp
    step.cpp
    step-dump.cpp
    var.cpp
    steps/gemv.cpp
    steps/symv.cpp
    steps/gemm.cpp
    steps/syrk.cpp
    steps/syr2k.cpp
    steps/trmm.cpp
    steps/trsm.cpp
)

set(KTEST_EXTERNAL_SRC
    ../../common/kgen_basic.c
    ../../common/clkern.c
    ../../common/devinfo.c
    ../../common/kern_cache.c
    ../../common/kerngen_core.c
    ../../common/kgen_guard.c
    ../../common/kgen_loop_helper.c
    ../../common/list.c
    ../../common/misc.c
    ../../common/mutex.c
    ../../common/trace_malloc.c
    ../../common/gens/dblock_kgen.c
    ../../blas/impl.c
    ../../blas/scimage.c
    ../../blas/generic/blas_funcs.c
    ../../blas/generic/common.c
    ../../blas/generic/events.c
    ../../blas/generic/kernel_extra.c
    ../../blas/generic/matrix_dims.c
    ../../blas/generic/matrix_props.c
    ../../blas/generic/problem_iter.c
    ../../blas/generic/solution_seq_make.c
    ../../blas/generic/solution_seq.c
    ../../blas/generic/kdump.c
    ../../blas/gens/tile.c
    ../../blas/gens/tile_iter.c
    ../../blas/gens/blas_subgroup.c
    ../../blas/gens/decomposition.c
    ../../blas/gens/blas_kgen.c
    ../../blas/gens/fetch.c
    ../../blas/gens/gemm.c
    ../../blas/gens/gemv.c
    ../../blas/gens/gen_helper.c
    ../../blas/gens/gen_init.c
    ../../blas/gens/symv.c
    ../../blas/gens/syrxk.c
    ../../blas/gens/tilemul.c
    ../../blas/gens/trmm.c
    ../../blas/gens/trsm.c
    ../../blas/gens/trsm_kgen.c
    ../../blas/gens/trxm_common.c
    ../../blas/gens/xxmv_common.c
    ../../blas/gens/legacy/blkmul.c
    ../../blas/gens/legacy/gemm_img.c
    ../../blas/gens/legacy/gemm_lds.c
    ../../blas/gens/legacy/trmm_img.c
    ../../blas/gens/legacy/trmm_lds.c
    ../../blas/gens/legacy/trsm_img.c
    ../../blas/gens/legacy/trsm_lds.c
    ../../blas/gens/legacy/trsm_cached_lds.c
    ../../blas/gens/legacy/blas_kgen_legacy.c
    ../../blas/gens/legacy/gen_helper_legacy.c
    ../../blas/gens/legacy/trxm_common_legacy.c
    ../../blas/gens/legacy/trsm_kgen_legacy.c
    ../../blas/init.c
    ../tune/fileio.c
    ../tune/toolslib.c
    ../tune/storage_init.c
    ../tune/storage_data.c
    ../tune/storage_io.c
    ../tune/dimension.c
    
	../../blas/gens/trmv_reg.cpp
	../../blas/gens/ger_lds.cpp
	../../blas/gens/trsv_trtri.cpp
	../../blas/gens/trsv_gemv.cpp
	../../blas/gens/kprintf.cpp
	../../blas/gens/syr_lds.cpp
	../../blas/gens/symm_cached.cpp
	../../blas/gens/gemm_cached.cpp
	../../blas/gens/gemm_tail_cached.cpp
	../../blas/gens/syr2_lds.cpp
	../../blas/gens/her_lds.cpp
	../../blas/gens/her2_lds.cpp
	../../blas/gens/gbmv.cpp
	../../blas/gens/tuned_numbers.c
	../../blas/gens/swap_reg.cpp
    ../../blas/gens/scal_reg.cpp
    ../../blas/gens/copy_reg.cpp
    ../../blas/gens/axpy_reg.cpp
    ../../blas/gens/dot.cpp
    ../../blas/gens/reduction.cpp
    ../../blas/gens/rotg_reg.cpp
    ../../blas/gens/rotmg_reg.cpp
    ../../blas/gens/rotm_reg.cpp
    ../../blas/gens/iamax.cpp
    ../../blas/gens/nrm2.cpp
    ../../blas/gens/asum.cpp
)

include_directories(
    ${OPENCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${clBLAS_SOURCE_DIR}
    ${clBLAS_SOURCE_DIR}/include
    ${clBLAS_SOURCE_DIR}/library/blas/include
    ${clBLAS_SOURCE_DIR}/library/blas/gens
    ${clBLAS_SOURCE_DIR}/library/tools/tune
    ${PROJECT_BINARY_DIR}/include
)

#setup Visual studio tabs
source_group(\\ FILES ${KTEST_SRC})

add_executable(make-ktest ${KTEST_SRC} ${KTEST_EXTERNAL_SRC})
add_dependencies(make-ktest GENERATE_CLT)
target_link_libraries(make-ktest ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${MATH_LIBRARY})
set_target_properties( make-ktest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# CPack configuration; include the executable into the package
install( TARGETS make-ktest
		RUNTIME DESTINATION bin${SUFFIX_BIN}
		LIBRARY DESTINATION lib${SUFFIX_LIB}
		ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
		)
