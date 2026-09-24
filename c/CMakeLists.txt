# moty — build CMake di riferimento (M6).
#
#   cmake -B build && cmake --build build -j        # tutto: librerie + motori
#   cmake -B build -DMOTY_ARCH=native && ...        # tier SIMD di QUESTA macchina
#   cmake -B build -DMOTY_BUILD_TESTS=ON && ... && ctest --test-dir build
#
# Il Makefile in c/ resta il percorso portabile-esotico (ARMv7/PPC64 cross,
# tier portable-v4/sve2, docker): le stesse fonti, le stesse librerie
# (libmoty-hw → libmoty-nn → libmoty-runtime). Le piattaforme desktop
# (Linux/macOS/MinGW) sono coperte qui; se serve un flag che solo il
# Makefile conosce, usare il Makefile.
cmake_minimum_required(VERSION 3.24)
project(moty LANGUAGES C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(MOTY_BUILD_TESTS "Costruisci la suite gtest (richiede C++)" ON)
option(MOTY_PLUGINS "Abilita i motori out-of-tree (MOTY_PLUGIN=...)" OFF)
set(MOTY_ARCH "" CACHE STRING "march (vuoto: portable; 'native': questa macchina)")

# MOTY_THREADPOOL: no OpenMP runtime (a clang toolchain without libomp);
# the parallel loops run on moty's own pthread pool (nn/par.c)
# MOTY_LIBRARY_ONLY: the embeddable library (libmoty, api/moty.h) is what the
# consumer wants — the pthread pool is then the default (no libomp needed)
option(MOTY_LIBRARY_ONLY "build for libmoty consumers (pthread pool by default, no tests)" OFF)
if(MOTY_LIBRARY_ONLY)
  set(_moty_tp_default ON)
else()
  set(_moty_tp_default OFF)
endif()
option(MOTY_THREADPOOL "pthread pool instead of OpenMP" ${_moty_tp_default})
if(MOTY_THREADPOOL)
  set(OpenMP_C_FOUND FALSE)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  add_compile_definitions(MOTY_THREADPOOL)
  link_libraries(Threads::Threads)
else()
  find_package(OpenMP)
endif()
if(OpenMP_C_FOUND)
  add_compile_options(${OpenMP_C_FLAGS})   # le pragma omp vivono nelle librerie
endif()

# ---- flag per piattaforma (specchio del Makefile, percorsi desktop) ----
if(APPLE)
  set(MOTY_CFLAGS "")
  # Apple clang senza libomp: single-thread (come il Makefile)
  execute_process(COMMAND brew --prefix libomp RESULT_VARIABLE _brew ERROR_QUIET OUTPUT_QUIET)
  if(_brew EQUAL 0)
    # libomp presente: usalo se trovato
    find_library(MOTY_LIBOMP omp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(MOTY_LIBOMP)
      set(MOTY_OMPC -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include)
      list(APPEND MOTY_CFLAGS ${MOTY_OMPC})
    endif()
  endif()
elseif(MINGW OR MSYS)
  if(NOT MOTY_ARCH)
    set(MOTY_ARCH x86-64-v3)
  endif()
  set(MOTY_CFLAGS -D_FILE_OFFSET_BITS=64)
else()
  # Linux/BSD: default portable v3 come il Makefile 'portable'; MOTY_ARCH=native per il tier massimo
  if(NOT MOTY_ARCH)
    set(MOTY_ARCH x86-64-v3)
  endif()
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function)
if(MOTY_ARCH)
  add_compile_options(-march=${MOTY_ARCH})
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- librerie (stesso ordine del Makefile) ----
add_library(moty_hw STATIC hw/hw.c)
target_link_libraries(moty_hw PUBLIC ${MOTY_OMPC})

set(MOTY_NN_SRCS nn/par.c nn/alloc.c nn/quant.c nn/mat.c nn/matmul.c nn/norm.c nn/rope.c
                 nn/attn_kernels.c nn/sample.c nn/attn.c nn/conv.c nn/ffn.c
                 nn/moe.c nn/head.c runtime/moecache.c io/tier.c)
add_library(moty_nn STATIC ${MOTY_NN_SRCS})
find_package(Threads REQUIRED)       # nn/alloc.c (registries, the pool) uses pthreads in every build
target_link_libraries(moty_nn PUBLIC moty_hw Threads::Threads)

add_library(moty_rt STATIC runtime/env.c runtime/kvcache.c api/api.c)
target_link_libraries(moty_rt PUBLIC moty_nn moty_hw)

# ---- motori: ogni engines/X.c con main rinominato, come nel Makefile ----
function(moty_engine name)
  # binario standalone: il SUO main (come nel Makefile)
  add_executable(${name} engines/${name}.c)
  target_link_libraries(${name} PRIVATE moty_rt moty_nn moty_hw m)
  if(OpenMP_C_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
  endif()
  # oggetto per il dispatcher moty: main rinominato (come -Dmain=X_main)
  add_library(${name}_obj OBJECT engines/${name}.c)
  target_compile_definitions(${name}_obj PRIVATE main=${name}_main)
  set_target_properties(${name}_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# glm e' escluso dai triplet ARMv7 (vedi Makefile); qui compila ovunque su desktop
foreach(e lfm2 qwenmoe qwen olmoe gemma glm)
  moty_engine(${e})
endforeach()

# ---- libmoty: the embeddable library (api/moty.h, docs/api.md) ----
# one archive with the three layers, the lfm2 and qwen engines and the
# registry that picks one by config.json model_type
add_library(moty_lib_engines OBJECT engines/lfm2.c engines/qwen.c)
target_compile_definitions(moty_lib_engines PRIVATE MOTY_NO_MAIN)       # no main in the library
set_target_properties(moty_lib_engines PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(moty_lib STATIC ${MOTY_NN_SRCS} hw/hw.c runtime/env.c runtime/kvcache.c api/api.c api/registry.c
            $<TARGET_OBJECTS:moty_lib_engines>)
set_target_properties(moty_lib PROPERTIES OUTPUT_NAME moty POSITION_INDEPENDENT_CODE ON)
target_include_directories(moty_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api> $<INSTALL_INTERFACE:include>)
if(OpenMP_C_FOUND)
  target_link_libraries(moty_lib PUBLIC OpenMP::OpenMP_C)
endif()
target_link_libraries(moty_lib PUBLIC m Threads::Threads)
install(TARGETS moty_lib ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES api/moty.h DESTINATION include)

add_executable(moty moty.c
  $<TARGET_OBJECTS:lfm2_obj> $<TARGET_OBJECTS:qwenmoe_obj>
  $<TARGET_OBJECTS:qwen_obj> $<TARGET_OBJECTS:olmoe_obj>
  $<TARGET_OBJECTS:gemma_obj> $<TARGET_OBJECTS:glm_obj>)
target_link_libraries(moty PRIVATE moty_rt moty_nn moty_hw m)
if(OpenMP_C_FOUND)
  target_link_libraries(moty PRIVATE OpenMP::OpenMP_C)
endif()
if(MOTY_PLUGINS)
  target_compile_definitions(moty PRIVATE MOTY_PLUGINS)
  target_link_libraries(moty PRIVATE ${CMAKE_DL_LIBS})
  add_library(moty_sample_plugin MODULE plugins/sample.c)
  set_target_properties(moty_sample_plugin PROPERTIES PREFIX "")
  target_compile_definitions(moty_sample_plugin PRIVATE)
endif()

# ---- test (la suite riusa i target se costruita dal padre) ----
if(MOTY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# a library-only consumer builds just moty_lib: `cmake --install` must not
# require the CLI
if(NOT MOTY_LIBRARY_ONLY)
  install(TARGETS moty RUNTIME DESTINATION bin)
endif()
