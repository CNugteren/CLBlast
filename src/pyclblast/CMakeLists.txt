cmake_minimum_required(VERSION 3.20)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Find python and numpy
find_package(
  Python3
  REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

# Run the cython compiler
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "./src"
  OUTPUT_VARIABLE Cython_SOURCE_DIR)
find_program(CYTHON "cython")
add_custom_command(
  OUTPUT "${Cython_SOURCE_DIR}/pyclblast.cpp"
  DEPENDS "${Cython_SOURCE_DIR}/pyclblast.pyx"
  VERBATIM
  COMMAND "${CYTHON}" -3 "${Cython_SOURCE_DIR}/pyclblast.pyx"
          --output-file "${Cython_SOURCE_DIR}/pyclblast.cpp")


# Add module target
Python3_add_library(pyclblast MODULE WITH_SOABI
                    "${Cython_SOURCE_DIR}/pyclblast.cpp")

# Numpy libraries - NOTE: clean NPY_LIBRARIES cache (may fail for venv)
cmake_path(GET Python3_NumPy_INCLUDE_DIRS PARENT_PATH Python3_NumPy_CORE_DIR)
unset(NPY_LIBRARIES CACHE)
find_library(NPY_LIBRARIES
  NAMES npymath
  PATHS ${Python3_NumPy_CORE_DIR}
  PATH_SUFFIXES lib
  DOC "Numpy math library"
  REQUIRED
  NO_DEFAULT_PATH)
target_link_libraries(pyclblast PRIVATE ${NPY_LIBRARIES})
target_include_directories(pyclblast PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

# CLBlast library
set(CLBLAST_HINTS
  ${CLBLAST_ROOT}
  $ENV{CLBLAST_ROOT}
)
find_package(CLBlast CONFIG REQUIRED HINTS ${CLBLAST_HINTS})
target_link_libraries(pyclblast PRIVATE clblast)

install(TARGETS pyclblast DESTINATION .)

# In windows pyclblast cannot find the dll, even on path.
# Probably related to change in 3.8, that loads dll only for trusted location
# see https://stackoverflow.com/questions/41365446/how-to-resolve-importerror-dll-load-failed-on-python
# One workaround is to copy the dll to the same dir as the module.
# TODO: add python version check
if (WIN32)
  cmake_path(APPEND CLBlast_DIR "../../../bin" OUTPUT_VARIABLE CLBlast_BINDIR)
  cmake_path(SET CLBlast_BINDIR NORMALIZE "${CLBlast_BINDIR}")
  unset(CLBlast_SHARED_LIBPATH CACHE)
  find_file(CLBlast_SHARED_LIBPATH
    NAMES clblast.dll
    PATHS ${CLBlast_BINDIR}
    DOC "CLBlast shared library"
    REQUIRED)

  # copy dll to build
  install(FILES ${CLBlast_SHARED_LIBPATH} DESTINATION .)
endif()
