cmake_minimum_required(VERSION 3.20)
project(aegis VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AEGIS_WERROR "Treat compiler warnings as errors" OFF)
option(AEGIS_NATIVE "Tune for the build machine's CPU (non-portable binaries)" OFF)
option(AEGIS_PYTHON "Build the Python extension module (needs pybind11)" ON)
set(AEGIS_SANITIZE "" CACHE STRING "Sanitizer to enable: address, thread, undefined, or empty")

if(AEGIS_SANITIZE)
  add_compile_options(-fsanitize=${AEGIS_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${AEGIS_SANITIZE})
endif()

set(AEGIS_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)
if(AEGIS_WERROR)
  list(APPEND AEGIS_WARNINGS -Werror)
endif()

if(AEGIS_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  else()
    add_compile_options(-mcpu=native)
  endif()
endif()

find_package(Threads REQUIRED)

# ---- core library -----------------------------------------------------------
add_library(aegis_core STATIC
  core/src/audit_log.cpp
  core/src/features.cpp
  core/src/mlp.cpp
  core/src/pipeline.cpp
  core/src/scorer.cpp
  core/src/sha256.cpp
)
target_include_directories(aegis_core PUBLIC core/include)
target_link_libraries(aegis_core PUBLIC Threads::Threads)
target_compile_options(aegis_core PRIVATE ${AEGIS_WARNINGS})
# Position-independent so the same archive links into the Python shared module.
set_target_properties(aegis_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- tests ------------------------------------------------------------------
enable_testing()
add_executable(aegis_tests
  core/tests/test_main.cpp
  core/tests/test_audit.cpp
  core/tests/test_containers.cpp
  core/tests/test_crypto.cpp
  core/tests/test_features.cpp
  core/tests/test_mlp.cpp
  core/tests/test_scorer.cpp
)
target_link_libraries(aegis_tests PRIVATE aegis_core)
target_compile_options(aegis_tests PRIVATE ${AEGIS_WARNINGS})
add_test(NAME core_tests COMMAND aegis_tests)

# ---- benchmark --------------------------------------------------------------
add_executable(aegis_bench core/bench/bench_main.cpp)
target_link_libraries(aegis_bench PRIVATE aegis_core)
target_compile_options(aegis_bench PRIVATE ${AEGIS_WARNINGS})

# ---- Python bindings --------------------------------------------------------
if(AEGIS_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(
      # get_cmake_dir() rather than `-m pybind11 --cmakedir`: the CLI quotes
      # paths that contain spaces, which CMake would then treat literally.
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PYBIND11_RC)
    if(PYBIND11_RC EQUAL 0)
      find_package(pybind11 CONFIG REQUIRED)
      pybind11_add_module(_core core/bindings/aegis_py.cpp)
      target_link_libraries(_core PRIVATE aegis_core)
      target_compile_options(_core PRIVATE ${AEGIS_WARNINGS})
      # Drop the compiled module straight into the Python package.
      set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/aegis"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/aegis"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/aegis")
    else()
      message(WARNING "pybind11 not found for ${Python_EXECUTABLE}; skipping Python module")
    endif()
  endif()
endif()
