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

set(CLIENT_SRC client.cpp stdafx.cpp statisticalTimer.cpp)
set(CLIENT_HEADER
    stdafx.h
    targetver.h
    statisticalTimer.h
    clfunc_common.hpp
    clfunc_xgemm.hpp
    clfunc_xgemv.hpp
    clfunc_xsymv.hpp
    clfunc_xtrmm.hpp
    clfunc_xtrsm.hpp
    clfunc_xsyrk.hpp
    clfunc_xsyr2k.hpp
	clfunc_xhemm.hpp
	clfunc_xsymm.hpp
	clfunc_xherk.hpp
	clfunc_xher2k.hpp)

set(WRAPPER_SRC testPerfWrapper.cpp)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# Having problems on build server, compiling gtest headers with -pedantic; disabling detection of long long
# http://code.google.com/p/googletest/issues/detail?id=334
if( CMAKE_COMPILER_IS_GNUCXX )
	add_definitions( -Wno-long-long )
endif( )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${OPENCL_INCLUDE_DIRS}
    ${clBLAS_SOURCE_DIR}
    ${clBLAS_SOURCE_DIR}/include
    ${clBLAS_SOURCE_DIR}/tests/include
    .)

add_executable(client ${CLIENT_SRC} ${CLIENT_HEADER})
target_link_libraries(client ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} clBLAS)
set_target_properties( client PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

add_executable(testPerfWrapper ${WRAPPER_SRC})
target_link_libraries(testPerfWrapper ${Boost_LIBRARIES})
set_target_properties( testPerfWrapper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# CPack configuration; include the executable into the package
install( TARGETS client testPerfWrapper
		RUNTIME DESTINATION bin${SUFFIX_BIN}
		LIBRARY DESTINATION lib${SUFFIX_LIB}
		ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
		)
