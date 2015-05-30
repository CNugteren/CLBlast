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


set(SRC_TILEMUL
    ../tilemul.c
    ../fetch.c
    ${clBLAS_SOURCE_DIR}/library/common/kerngen_core.c
    ${clBLAS_SOURCE_DIR}/library/common/kgen_basic.c
    ${clBLAS_SOURCE_DIR}/library/common/kgen_loop_helper.c
    ${clBLAS_SOURCE_DIR}/library/common/misc.c
    ${clBLAS_SOURCE_DIR}/library/blas/gens/blas_kgen.c
    ${clBLAS_SOURCE_DIR}/library/blas/gens/tile.c
    ${clBLAS_SOURCE_DIR}/library/blas/gens/tile_iter.c
    ${clBLAS_SOURCE_DIR}/library/blas/gens/gen_helper.c
    ${clBLAS_SOURCE_DIR}/library/blas/generic/blas_funcs.c
    ${clBLAS_SOURCE_DIR}/library/blas/generic/matrix_dims.c
    ${clBLAS_SOURCE_DIR}/library/blas/generic/matrix_props.c
    ${clBLAS_SOURCE_DIR}/library/common/gens/dblock_kgen.c
    ${clBLAS_SOURCE_DIR}/library/common/kgen_guard.c
    ${clBLAS_SOURCE_DIR}/library/common/list.c
    ${clBLAS_SOURCE_DIR}/library/common/mutex.c 
    ${clBLAS_SOURCE_DIR}/library/common/trace_malloc.c
    t_tilemul.c
)

include_directories(${OPENCL_INCLUDE_DIRS} ${clBLAS_SOURCE_DIR} ${clBLAS_SOURCE_DIR}/include
                    ${clBLAS_SOURCE_DIR}/library/blas/include ${clBLAS_SOURCE_DIR}/library/blas/gens)

add_executable(t_tilemul ${SRC_TILEMUL})
target_link_libraries(t_tilemul ${OPENCL_LIBRARIES})
set_target_properties( t_tilemul PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# CPack configuration; include the executable into the package
install( TARGETS t_tilemul
		RUNTIME DESTINATION bin${SUFFIX_BIN}
		LIBRARY DESTINATION lib${SUFFIX_LIB}
		ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
		)
