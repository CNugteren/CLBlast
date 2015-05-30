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
    ../list.c
    ../clkern.c
    ../kern_cache.c
    ../kerngen_core.c
    ../kgen_basic.c
    ../kgen_loop_helper.c
    ../kgen_guard.c
    ../misc.c
    ../gens/dblock_kgen.c
    ../devinfo.c
    ../devinfo-cache.c
    ../mutex.c
    ../trace_malloc.c
)

set(SRC_DBLOCK_KGEN
    ${SRC_COMMON}
    t_dblock_kgen.c
)

set(SRC_GENS_CACHE
    ${SRC_COMMON}
    t_gens_cache.c
)

include_directories(${OPENCL_INCLUDE_DIRS} ${clBLAS_SOURCE_DIR} ${clBLAS_SOURCE_DIR}/include ${clBLAS_SOURCE_DIR}/src/blas/include)

add_executable(t_dblock_kgen ${SRC_DBLOCK_KGEN})
target_link_libraries(t_dblock_kgen ${OPENCL_LIBRARIES} ${MATH_LIBRARY})
set_target_properties( t_dblock_kgen PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

add_executable(t_gens_cache ${SRC_GENS_CACHE})
target_link_libraries(t_gens_cache ${OPENCL_LIBRARIES} ${MATH_LIBRARY})
set_target_properties( t_gens_cache PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# CPack configuration; include the executable into the package
install( TARGETS t_dblock_kgen t_gens_cache
		RUNTIME DESTINATION bin${SUFFIX_BIN}
		LIBRARY DESTINATION lib${SUFFIX_LIB}
		ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
		)
