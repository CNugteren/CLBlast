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

set(SGEMV_SAMPLE_SRC example_sgemv.c)
set(SSYMV_SAMPLE_SRC example_ssymv.c)
set(SGEMM_SAMPLE_SRC example_sgemm.c)
set(STRMM_SAMPLE_SRC example_strmm.c)
set(STRSM_SAMPLE_SRC example_strsm.c)
set(SSYRK_SAMPLE_SRC example_ssyrk.c)
set(SSYR2K_SAMPLE_SRC example_ssyr2k.c)

set(STRMV_SAMPLE_SRC example_strmv.c) # Addition: for STRMV
set(DTRMV_SAMPLE_SRC example_dtrmv.c) # Addition: for STRMV
set(STRSV_SAMPLE_SRC example_strsv.c) # Addition: for STRSV
set(SGER_SAMPLE_SRC example_sger.c)   # Addition: for SGER
set(SSYR_SAMPLE_SRC example_ssyr.c)   # Addition: for SSYR
set(SSYR2_SAMPLE_SRC example_ssyr2.c)   # Addition: for SSYR2
set(CHER_SAMPLE_SRC example_cher.c)
set(ZHEMV_SAMPLE_SRC example_zhemv.cpp)
set(ZHER2_SAMPLE_SRC example_zher2.c)
set(CHERK_SAMPLE_SRC example_cherk.cpp)
set(SSYMM_SAMPLE_SRC example_ssymm.c)
set(CHEMM_SAMPLE_SRC example_chemm.cpp)

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

set(VERSION_SAMPLE_SRC clBlasVersion.c)

include_directories(${OPENCL_INCLUDE_DIRS} ${clBLAS_SOURCE_DIR})

add_executable(example_sgemv ${SGEMV_SAMPLE_SRC})
target_link_libraries(example_sgemv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sgemv PROPERTY FOLDER "Samples")

add_executable(example_ssymv ${SSYMV_SAMPLE_SRC})
target_link_libraries(example_ssymv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssymv PROPERTY FOLDER "Samples")

add_executable(example_sgemm ${SGEMM_SAMPLE_SRC})
target_link_libraries(example_sgemm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sgemm PROPERTY FOLDER "Samples")

add_executable(example_strmm ${STRMM_SAMPLE_SRC})
target_link_libraries(example_strmm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_strmm PROPERTY FOLDER "Samples")

add_executable(example_strsm ${STRSM_SAMPLE_SRC})
target_link_libraries(example_strsm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_strsm PROPERTY FOLDER "Samples")

add_executable(example_ssyrk ${SSYRK_SAMPLE_SRC})
target_link_libraries(example_ssyrk ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssyrk PROPERTY FOLDER "Samples")

add_executable(example_ssyr2k ${SSYR2K_SAMPLE_SRC})
target_link_libraries(example_ssyr2k ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssyr2k PROPERTY FOLDER "Samples")

add_executable(version ${VERSION_SAMPLE_SRC})
target_link_libraries(version ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET version PROPERTY FOLDER "Samples")

# Addition - for samples
add_executable(example_strmv ${STRMV_SAMPLE_SRC})
target_link_libraries(example_strmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_strmv PROPERTY FOLDER "Samples")

add_executable(example_dtrmv ${DTRMV_SAMPLE_SRC})
target_link_libraries(example_dtrmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_dtrmv PROPERTY FOLDER "Samples")

add_executable(example_strsv ${STRSV_SAMPLE_SRC})
target_link_libraries(example_strsv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_strsv PROPERTY FOLDER "Samples")

add_executable(example_sger ${SGER_SAMPLE_SRC})
target_link_libraries(example_sger ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sger PROPERTY FOLDER "Samples")

add_executable(example_cher ${CHER_SAMPLE_SRC})
target_link_libraries(example_cher ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_cher PROPERTY FOLDER "Samples")

add_executable(example_ssyr ${SSYR_SAMPLE_SRC})
target_link_libraries(example_ssyr ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssyr PROPERTY FOLDER "Samples")

add_executable(example_ssyr2 ${SSYR2_SAMPLE_SRC})
target_link_libraries(example_ssyr2 ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssyr2 PROPERTY FOLDER "Samples")

add_executable(example_zhemv ${ZHEMV_SAMPLE_SRC})
target_link_libraries(example_zhemv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_zhemv PROPERTY FOLDER "Samples")

add_executable(example_zher2 ${ZHER2_SAMPLE_SRC})
target_link_libraries(example_zher2 ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_zher2 PROPERTY FOLDER "Samples")

add_executable(example_cherk ${CHERK_SAMPLE_SRC})
target_link_libraries(example_cherk ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_cherk PROPERTY FOLDER "Samples")

add_executable(example_ssymm ${SSYMM_SAMPLE_SRC})
target_link_libraries(example_ssymm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssymm PROPERTY FOLDER "Samples")

add_executable(example_chemm ${CHEMM_SAMPLE_SRC})
target_link_libraries(example_chemm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_chemm PROPERTY FOLDER "Samples")

add_executable(example_stpmv ${STPMV_SAMPLE_SRC})
target_link_libraries(example_stpmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_stpmv PROPERTY FOLDER "Samples")

add_executable(example_chpmv ${CHPMV_SAMPLE_SRC})
target_link_libraries(example_chpmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_chpmv PROPERTY FOLDER "Samples")

add_executable(example_stpsv ${STPSV_SAMPLE_SRC})
target_link_libraries(example_stpsv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_stpsv PROPERTY FOLDER "Samples")

add_executable(example_sspmv ${SSPMV_SAMPLE_SRC})
target_link_libraries(example_sspmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sspmv PROPERTY FOLDER "Samples")

add_executable(example_sspr ${SSPR_SAMPLE_SRC})
target_link_libraries(example_sspr ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sspr PROPERTY FOLDER "Samples")

add_executable(example_chpr ${CHPR_SAMPLE_SRC})
target_link_libraries(example_chpr ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_chpr PROPERTY FOLDER "Samples")

add_executable(example_sspr2 ${SSPR2_SAMPLE_SRC})
target_link_libraries(example_sspr2 ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sspr2 PROPERTY FOLDER "Samples")

add_executable(example_zhpr2 ${ZHPR2_SAMPLE_SRC})
target_link_libraries(example_zhpr2 ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_zhpr2 PROPERTY FOLDER "Samples")

add_executable(example_sgbmv ${SGBMV_SAMPLE_SRC})
target_link_libraries(example_sgbmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sgbmv PROPERTY FOLDER "Samples")

add_executable(example_stbmv ${STBMV_SAMPLE_SRC})
target_link_libraries(example_stbmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_stbmv PROPERTY FOLDER "Samples")

add_executable(example_ssbmv ${SSBMV_SAMPLE_SRC})
target_link_libraries(example_ssbmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_ssbmv PROPERTY FOLDER "Samples")

add_executable(example_chbmv ${CHBMV_SAMPLE_SRC})
target_link_libraries(example_chbmv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_chbmv PROPERTY FOLDER "Samples")

add_executable(example_stbsv ${STBSV_SAMPLE_SRC})
target_link_libraries(example_stbsv ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_stbsv PROPERTY FOLDER "Samples")

add_executable(example_cher2k ${CHER2K_SAMPLE_SRC})
target_link_libraries(example_cher2k ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_cher2k PROPERTY FOLDER "Samples")

add_executable(example_sswap ${SSWAP_SAMPLE_SRC})
target_link_libraries(example_sswap ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sswap PROPERTY FOLDER "Samples")

add_executable(example_sscal ${SSCAL_SAMPLE_SRC})
target_link_libraries(example_sscal ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sscal PROPERTY FOLDER "Samples")

add_executable(example_csscal ${CSSCAL_SAMPLE_SRC})
target_link_libraries(example_csscal ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_csscal PROPERTY FOLDER "Samples")

add_executable(example_scopy ${SCOPY_SAMPLE_SRC})
target_link_libraries(example_scopy ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_scopy PROPERTY FOLDER "Samples")

add_executable(example_saxpy ${SAXPY_SAMPLE_SRC})
target_link_libraries(example_saxpy ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_saxpy PROPERTY FOLDER "Samples")

add_executable(example_sdot ${SDOT_SAMPLE_SRC})
target_link_libraries(example_sdot ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sdot PROPERTY FOLDER "Samples")

add_executable(example_srotg ${SROTG_SAMPLE_SRC})
target_link_libraries(example_srotg ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_srotg PROPERTY FOLDER "Samples")

add_executable(example_srotmg ${SROTMG_SAMPLE_SRC})
target_link_libraries(example_srotmg ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_srotmg PROPERTY FOLDER "Samples")

add_executable(example_srot ${SROT_SAMPLE_SRC})
target_link_libraries(example_srot ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_srot PROPERTY FOLDER "Samples")

add_executable(example_srotm ${SROTM_SAMPLE_SRC})
target_link_libraries(example_srotm ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_srotm PROPERTY FOLDER "Samples")

add_executable(example_isamax ${iSAMAX_SAMPLE_SRC})
target_link_libraries(example_isamax ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_isamax PROPERTY FOLDER "Samples")

add_executable(example_snrm2 ${SNRM2_SAMPLE_SRC})
target_link_libraries(example_snrm2 ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_snrm2 PROPERTY FOLDER "Samples")

add_executable(example_sasum ${SASUM_SAMPLE_SRC})
target_link_libraries(example_sasum ${OPENCL_LIBRARIES} clBLAS)
set_property( TARGET example_sasum PROPERTY FOLDER "Samples")

# CPack configuration; include the executable into the package
install( TARGETS example_sgemm example_sgemv example_ssymv example_ssyrk
         example_ssyr2k example_strmm example_strsm 
         example_strmv example_strsv example_sger example_cher example_ssyr 
         example_ssyr2 example_cherk example_ssymm example_chemm
         example_stpmv example_chpmv example_stpsv example_sspmv example_sspr example_chpr
         example_sspr2 example_zhpr2 
         example_sgbmv example_stbmv example_ssbmv example_chbmv example_stbsv
         example_cher2k
         example_sswap example_sscal example_csscal example_scopy example_saxpy example_sdot
         example_srotg example_srotmg example_srot example_srotm
         example_snrm2 example_sasum example_isamax

         version
        RUNTIME DESTINATION bin${SUFFIX_BIN}
        LIBRARY DESTINATION lib${SUFFIX_LIB}
        ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
        )

configure_file( "${PROJECT_SOURCE_DIR}/samples/CMakeLists.pack"
		"${PROJECT_BINARY_DIR}/samples/CMakeLists.txt" COPYONLY )
		
install(FILES
            example_sgemv.c
            example_ssymv.c 
            example_sgemm.c
            example_strmm.c
            example_strsm.c
            example_ssyrk.c
            example_ssyr2k.c
			example_strmv.c
			example_strsv.c
			example_sger.c 
			example_ssyr.c 
			example_ssyr2.c
			example_ssymm.c
			example_cher.c 
            example_chemm.cpp
            example_cherk.cpp
            example_ssymm.c
            example_chemm.cpp
            example_stpmv.c
            example_chpmv.c
            example_stpsv.c
    	    example_sspmv.c
    	    example_sspr.c
    	    example_chpr.c
    	    example_sspr2.c
    	    example_zhpr2.c
    	    example_sgbmv.c
    	    example_stbmv.c
    	    example_ssbmv.c
    	    example_chbmv.c
    	    example_stbsv.c
    	    example_cher2k.c
    	    example_sswap.c
            example_sscal.c
            example_scopy.c
            example_csscal.c
            example_saxpy.c
            example_sdot.c
            example_srotg.c
            example_srotmg.c
            example_srot.c
            example_srotm.c
            example_isamax.c
            example_snrm2.c
            example_sasum.c

            clBlasVersion.c
            ${PROJECT_BINARY_DIR}/samples/CMakeLists.txt

        DESTINATION
		    samples )
