# Percorso di build SEPARATO per i test (i motori si compilano con il make puro).
# Tutta la logica di test e' C compilato dal compilatore C; il C++ e' confinato
# ai file *_gtest.cc (solo macro TEST + dichiarazioni extern "C").
#
#   cmake -S tests -B tests/build && cmake --build tests/build -j
#   ctest --test-dir tests/build --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(moty_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# gtest: sistema se presente, altrimenti FetchContent (tag fissato, shallow)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

find_package(OpenMP)
enable_testing()
include(GoogleTest)
set(ENGDIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# opzionale: -march per abilitare i kernel AVX512/i8mm nei test locali
set(MOTY_TEST_MARCH "" CACHE STRING "march per i sorgenti C dei test (es. native)")

# libmoty-hw (M1): la stessa implementazione unica dei kernel che usano i
# motori. MOTY_TEST_MARCH si applica anche a hw.c: i test esercitano il
# tier SIMD reale della macchina.
add_library(moty_core STATIC ../nn/alloc.c ../nn/quant.c)
target_include_directories(moty_core PRIVATE ${ENGDIR})
if(OpenMP_C_FOUND)
  target_link_libraries(moty_core PRIVATE OpenMP::OpenMP_C)
endif()
if(MOTY_TEST_MARCH)
  set_source_files_properties(../nn/alloc.c ../nn/quant.c PROPERTIES COMPILE_OPTIONS "-march=${MOTY_TEST_MARCH}")
endif()
add_library(moty_hw STATIC ../hw/hw.c)
target_include_directories(moty_hw PRIVATE ${ENGDIR})
if(OpenMP_C_FOUND)
  target_link_libraries(moty_hw PRIVATE OpenMP::OpenMP_C)
endif()
if(MOTY_TEST_MARCH)
  set_source_files_properties(../hw/hw.c PROPERTIES COMPILE_OPTIONS "-march=${MOTY_TEST_MARCH}")
endif()

function(moty_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${ENGDIR})
  target_link_libraries(${name} PRIVATE GTest::gtest_main m moty_hw moty_core)
  if(OpenMP_C_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(MOTY_TEST_MARCH)
    foreach(src ${ARGN})
      if(src MATCHES "\\.c$")
        set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-march=${MOTY_TEST_MARCH}")
      endif()
    endforeach()
  endif()
  gtest_discover_tests(${name})
endfunction()

moty_test(test_headers headers_tests.c headers_gtest.cc)
moty_test(test_simd    simd_tests.c    simd_gtest.cc)
moty_test(test_scratch scratch_tests.c scratch_gtest.cc)
moty_test(test_kernels kernels_tests.c kernels_gtest.cc)
moty_test(test_conv    conv_tests.c    conv_gtest.cc)
moty_test(test_attn    attn_tests.c    attn_gtest.cc)
moty_test(test_prof    prof_tests.c    prof_gtest.cc ${CMAKE_CURRENT_BINARY_DIR}/prof_tests_on.c)
moty_test(test_moe     moe_tests.c     moe_gtest.cc)
moty_test(test_mla     mla_tests.c     mla_gtest.cc)
moty_test(test_qwen    qwen_tests.c    qwen_gtest.cc)
moty_test(test_gemma   gemma_tests.c   gemma_gtest.cc)
moty_test(test_glm     glm_tests.c     glm_gtest.cc)

# prof_tests_on.c: clone di prof_tests.c compilato CON la strumentazione
# (P4) — verifica che le stesse macro accumulino quando MOTY_PROF è
# attivo. Il define sta DENTRO il file generato: niente COMPILE_OPTIONS
# che possano essere sovrascritti dal loop MOTY_TEST_MARCH in moty_test().
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/prof_tests_on.c CONTENT "#define MOTY_PROF 1
#define prof_tests_main prof_tests_on_main
#include \"${CMAKE_CURRENT_SOURCE_DIR}/prof_tests.c\"
")

# utility C pura (non un test ctest): parita' del tokenizer contro un corpus HF
add_executable(tok_oracle tok_oracle.c)
target_include_directories(tok_oracle PRIVATE ${ENGDIR})
target_link_libraries(tok_oracle PRIVATE m)

# utility C pura (non un test ctest): micro-bench dei percorsi caldi
add_executable(bench bench.c)
target_include_directories(bench PRIVATE ${ENGDIR})
target_link_libraries(bench PRIVATE m moty_hw moty_core)
if(OpenMP_C_FOUND)
  target_link_libraries(bench PRIVATE OpenMP::OpenMP_C)
endif()
if(MOTY_TEST_MARCH)
  set_source_files_properties(bench.c PROPERTIES COMPILE_OPTIONS "-march=${MOTY_TEST_MARCH}")
endif()
