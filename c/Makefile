
# --- Target detection ---
# Ask the compiler what it actually targets (rofl0r's note on #129): a gcc/clang
# toolchain reports its target triple via `-dumpmachine`, e.g.
#   x86_64-w64-mingw32   x86_64-pc-cygwin   x86_64-unknown-linux-gnu
#   arm64-apple-darwin   powerpc64le-unknown-linux-gnu
# The triple follows the toolchain, not the host shell that `uname` reports, so
# it is correct under a native-Windows shell (no `uname` on PATH) AND when
# cross-compiling (e.g. make CC=x86_64-w64-mingw32-gcc on Linux). CC is chosen
# per-target below, so probe with the user's CC if they set one, else gcc
# (present on Linux and mingw, and a clang shim on macOS).
DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)

MINGW  := $(findstring mingw,$(TRIPLET))
CYGWIN := $(findstring cygwin,$(TRIPLET))
DARWIN := $(findstring darwin,$(TRIPLET))
LINUX  := $(findstring linux,$(TRIPLET))
IS_WIN := $(MINGW)$(CYGWIN)

# Fallbacks for the rare toolchain that does not answer -dumpmachine: keep the
# #129 signal (OS=Windows_NT, set in every Windows shell) for Windows, then
# `uname` for everything else, so no host regresses.
ifeq ($(TRIPLET),)
IS_WIN := $(if $(filter Windows_NT,$(OS)),1,)
ifeq ($(IS_WIN),)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
DARWIN  := $(findstring Darwin,$(UNAME_S))
endif
endif

TARGET_CPU := $(firstword $(subst -, ,$(TRIPLET)))
ifeq ($(TARGET_CPU),)
TARGET_CPU := $(UNAME_M)
endif
X86_64  := $(filter x86_64 amd64,$(TARGET_CPU))
AARCH64 := $(filter aarch64 arm64,$(TARGET_CPU))
PPC64   := $(filter powerpc64% ppc64%,$(TARGET_CPU))
# 32-bit ARM (ARMv7-A: Cortex-A7/A9/A15/A17; ARMv8-A core in AArch32: A53/A72).
# Lista esplicita: arm64/aarch64 passano da AARCH64, 'arm' e' il triplet comune
# (arm-linux-gnueabihf). Senza questa branca il triplet cade nel ramo x86-64
# (-march=native e' invalido in cross + nessun -mfpu -> __ARM_NEON indefinito
# -> path scalare silente, vedi #WP1.4).
ARMV7   := $(filter arm armv7l armv6l armv5tel,$(TARGET_CPU))

ifneq (,$(DARWIN))
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
# `brew --prefix libomp` can print the formula's prospective path even when it
# is not installed, so verify both artifacts before adding unusable flags.
ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -I. -O2 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# Opt-in: ARCH=native appends -mcpu=native (arm64 clang uses -mcpu, not -march),
# which unlocks the i8mm SMMLA int8/int4 dot kernels in glm.c. ARCH unset ->
# no -mcpu, default build byte-identical. Apple clang knows apple-m4 / native.
ifneq ($(ARCH),)
CFLAGS += -mcpu=$(ARCH)
endif
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics - tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (portable binary with AVX2). For max speed on THIS
# machine use ARCH=native: on AVX-VNNI CPUs (Intel Alder Lake+, Meteor Lake+)
# it also unlocks the 128-bit VPDPBUSD int8/int4 dot kernel (dot_i8i8/dot_i4i8),
# which the x86-64-v3 baseline does not define. The #ifdef guards in glm.c mean
# a v3 build simply compiles out the VNNI path - safe on any x86-64.
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -I. -D_FILE_OFFSET_BITS=64 -O2 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# -lpsapi: compat.h calls GetProcessMemoryInfo (rss_gb). It's linked via
# #pragma comment(lib,"psapi.lib") for MSVC, but MinGW gcc ignores that pragma
# (warns), so link psapi explicitly or the build fails undefined-reference on
# modern GCC (e.g. 16.x under UCRT). Harmless where the pragma also resolves it.
LDFLAGS = -lm -fopenmp -static -lpsapi
EXE     = .exe
else
ifneq (,$(PPC64))
# --- Linux PowerPC (POWER8/POWER9/POWER10) ---
# PowerPC GCC uses -mcpu, not -march. ARCH=native works on gcc >= 4.7.
# The AVX2/NEON kernels fall back to the portable scalar C path
# (validated token-exact vs the transformers oracle on a POWER8 S824).
CC      = gcc
ARCH   ?= native
CFLAGS  = -I. -O2 -mcpu=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
else
ifneq (,$(ARMV7))
# --- Linux 32-bit ARM (ARMv7-A: Cortex-A7/A9/A15/A17; ARMv8-A in AArch32) ---
# NEON e' baseline su tutti i Cortex-A con VFPv4. -mfloat-abi omesso: il default
# del toolchain decide (hard per gnueabihf, soft per gnueabi) -> funziona in ogni
# caso (le chiamate FP sono slow in soft, ma i kernel NEON compilano e girano).
# ARCH override: ARCH=armv7ve -mfpu via FPU=... ; ARCH=armv8-a per A53/A72 in
# 32-bit (lunghezza 128-bit NEON, niente dotprod/i8mm - servono ARMv8.2+ in AArch64).
CC      = gcc
ARCH   ?= armv7-a
FPU    ?= neon-vfpv4
CFLAGS  = -I. -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -mfpu=$(FPU) -munaligned-access -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
else
# --- Linux / *BSD x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
# -pthread: Linux lo tira dentro via -fopenmp, i *BSD no e le pthread_* non risolvono (#219).
ARCH   ?= native
CFLAGS  = -I. -O2 -march=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
endif
endif
endif

# Motori da compilare con `make all`. glm.c e' AArch64-only (intrinseci NEON
# vaddvq a 64-bit + MoE 744B: nessun target Cortex-A 32-bit lo userebbe); sui
# triplet ARMV7 lo escludiamo e `make glm` dara' errore esplicito se richiesto.
ALL_ENGS = glm olmoe qwen qwenmoe lfm2 gemma moty
ifneq (,$(ARMV7))
ALL_ENGS = olmoe qwen qwenmoe lfm2 gemma
endif
all: $(foreach e,$(ALL_ENGS),$e$(EXE))

# phony aliases → '<name>.exe' on Windows (so 'make glm' works on every platform)
ifneq ($(EXE),)
# alias utili solo quando EXE non vuoto (su Linux i target file si chiamano gia' cosi')
glm: glm$(EXE)
olmoe: olmoe$(EXE)
qwen: qwen$(EXE)
qwenmoe: qwenmoe$(EXE)
lfm2: lfm2$(EXE)
gemma: gemma$(EXE)
moty: moty$(EXE)
endif

# ---- libmoty-hw (M1): ONE compiled implementation of the hw kernels ----
# Same CFLAGS as the engines: the -march tier hw.c selects is the tier the
# whole program runs. Engines link the archive; hw.h holds only prototypes
# + legacy name macros (docs/modularization-plan.md, docs/symbol-map.md).
# ---- libmoty-nn core (M2): allocatori/arena + quantizzazione, una copia ----
NN_CORE_SRCS = nn/alloc.c nn/quant.c
NN_CORE_DEPS = nn/nn_alloc.h nn/nn_quant.h
libmoty-nn.a: $(NN_CORE_SRCS) $(NN_CORE_DEPS) libmoty-hw.a
	$(CC) $(CFLAGS) -c nn/alloc.c -o nn/alloc.o
	$(CC) $(CFLAGS) -c nn/quant.c -o nn/quant.o
	ar rcs $@ nn/alloc.o nn/quant.o

HW_SRCS = hw/hw.c
HW_DEPS = hw/hw.h hw/hw_impl.h hw/hw_avx512.h hw/hw_avx2.h hw/hw_neon.h \
          hw/hw_sve2.h hw/hw_vsx.h hw/hw_scalar.h hw/hw_quant.h \
          hw/hw_deltanet.h hw/hw_backend.h
libmoty-hw.a: $(HW_SRCS) $(HW_DEPS)
	$(CC) $(CFLAGS) -c $(HW_SRCS) -o hw/hw.o
	ar rcs $@ hw/hw.o

# ---- source layout (v2): engines/ nn/ hw/ runtime/ io/ tok/ util/ ----
# All includes are repo-c-relative ("nn/nn.h", "hw/hw.h", ...), so every
# compile adds -I. A shared deps list keeps the .o rules honest.
NN_CORE   = nn/nn.h nn/nn_alloc.h nn/nn_norm.h nn/nn_sample.h nn/nn_gemm.h hw/hw.h
RUNTIME_H = runtime/runtime.h runtime/rt_model_load.h runtime/rt_kv_cache.h runtime/rt_gen_loop.h runtime/rt_env_cfg.h
IO_H      = io/st.h io/gguf.h io/tier.h
TOK_H     = tok/tok.h tok/tok_unicode.h
UTIL_H    = util/json.h util/compat.h util/prof.h

glm$(EXE): libmoty-nn.a libmoty-hw.a engines/glm.c io/st.h io/uring.h util/json.h $(TOK_H) util/compat.h util/grammar.h
	$(CC) $(CFLAGS) engines/glm.c -o glm$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

olmoe$(EXE): libmoty-nn.a libmoty-hw.a engines/olmoe.c $(NN_CORE) runtime/moe.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) engines/olmoe.c -o olmoe$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

qwen$(EXE): libmoty-nn.a libmoty-hw.a engines/qwen.c $(NN_CORE) nn/nn_rope.h nn/nn_deltanet.h runtime/runtime.h $(IO_H) util/stw.h engines/qwen_train.h $(UTIL_H)
	$(CC) $(CFLAGS) engines/qwen.c -o qwen$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

qwenmoe$(EXE): libmoty-nn.a libmoty-hw.a engines/qwenmoe.c $(NN_CORE) nn/nn_rope.h nn/nn_deltanet.h runtime/moe.h runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) engines/qwenmoe.c -o qwenmoe$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

lfm2$(EXE): libmoty-nn.a libmoty-hw.a engines/lfm2.c $(NN_CORE) nn/nn_rope.h nn/nn_attn.h nn/nn_conv.h nn/nn_ffn.h nn/nn_moe_sigmoid.h runtime/moe.h runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) engines/lfm2.c -o lfm2$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

# ---- Unified multi-model binary ----
# Each engines/model_X.c is compiled to model_X.o with its main() renamed.
# The .o files are linked with moty.o (the dispatcher). To add a model:
#   1. Add a compile rule: $(CC) $(CFLAGS) -Dmain=model_X_main -c engines/model_X.c
#   2. Add model_X.o to the moty link line
#   3. Add {"arch", model_X_main} to the models[] table in moty.c
# See docs/adding-a-model.md for the full recipe (config, hooks, layers, tests).

lfm2.o: libmoty-nn.a libmoty-hw.a engines/lfm2.c $(NN_CORE) nn/nn_rope.h nn/nn_attn.h nn/nn_conv.h nn/nn_ffn.h nn/nn_moe_sigmoid.h runtime/moe.h runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) -Dmain=lfm2_main -c engines/lfm2.c -o lfm2.o

qwenmoe.o: libmoty-nn.a libmoty-hw.a engines/qwenmoe.c $(NN_CORE) nn/nn_rope.h nn/nn_deltanet.h runtime/moe.h runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) -Dmain=qwenmoe_main -c engines/qwenmoe.c -o qwenmoe.o

gemma.o: libmoty-nn.a libmoty-hw.a engines/gemma.c $(NN_CORE) runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) -Dmain=gemma_main -c engines/gemma.c -o gemma.o

olmoe.o: libmoty-nn.a libmoty-hw.a engines/olmoe.c $(NN_CORE) runtime/moe.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) -Dmain=olmoe_main -c engines/olmoe.c -o olmoe.o

qwen.o: libmoty-nn.a libmoty-hw.a engines/qwen.c $(NN_CORE) nn/nn_rope.h nn/nn_deltanet.h runtime/runtime.h $(IO_H) util/stw.h engines/qwen_train.h $(UTIL_H)
	$(CC) $(CFLAGS) -Dmain=qwen_main -c engines/qwen.c -o qwen.o

glm.o: libmoty-nn.a libmoty-hw.a engines/glm.c io/st.h io/uring.h util/json.h $(TOK_H) util/compat.h util/grammar.h
	$(CC) $(CFLAGS) -Dmain=glm_main -c engines/glm.c -o glm.o

moty$(EXE): libmoty-nn.a libmoty-hw.a moty.c lfm2.o qwenmoe.o gemma.o olmoe.o qwen.o glm.o
	$(CC) $(CFLAGS) moty.c lfm2.o qwenmoe.o gemma.o olmoe.o qwen.o glm.o -o moty$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

gemma$(EXE): libmoty-nn.a libmoty-hw.a engines/gemma.c $(NN_CORE) runtime/runtime.h $(IO_H) $(TOK_H) $(UTIL_H)
	$(CC) $(CFLAGS) engines/gemma.c -o gemma$(EXE) $(LDFLAGS) libmoty-hw.a libmoty-nn.a

# Use a baseline that matches the compiler target. macOS already targets a
# portable baseline when ARCH is empty; forcing the x86 value there breaks
# Apple Silicon. Unknown targets use native rather than an invalid x86 flag.
ifneq (,$(DARWIN))
PORTABLE_ARCH =
else ifneq (,$(AARCH64))
PORTABLE_ARCH = armv8-a
else ifneq (,$(ARMV7))
PORTABLE_ARCH = armv7-a
else ifneq (,$(PPC64))
PORTABLE_ARCH = power8
else ifneq (,$(X86_64))
PORTABLE_ARCH = x86-64-v3
else
PORTABLE_ARCH = native
endif

portable:
	$(MAKE) clean >/dev/null 2>&1 || true
	$(MAKE) all ARCH=$(PORTABLE_ARCH)
	$(MAKE) clean >/dev/null 2>&1 || true

# x86 AVX-512 tier of the portable binary: x86-64-v4 = AVX512F/BW/CD/DQ/VL
# (dot_f32 avx512f) ma NON AVX512-VNNI (int8 resta sul tier AVX2; VNNI solo
# con ARCH=native su CPU che la supporta).
portable-v4:
	$(MAKE) all ARCH=x86-64-v4

# aarch64 SVE2 tier of the portable binary: armv9-a implica SVE2 (svdot per gli
# int8/int4) + SVE1 (dot_f32/dot_f32i8/DeltaNet scalabili). Come portable-v4 e'
# specifico di un'architettura: su x86 usare `portable`/`portable-v4`, su POWER
# `portable`. Per core specifici: `make ARCH=neoverse-v2` (Grace/Cobalt).
portable-sve2:
	$(MAKE) all ARCH=armv9-a

# test suite compilata con -march=native: esercita i kernel SIMD reali
TESTS_BUILD_NATIVE ?= tests/build-native
test-native:
	cmake -S tests -B $(TESTS_BUILD_NATIVE) -DMOTY_TEST_MARCH=native
	cmake --build $(TESTS_BUILD_NATIVE) -j
	ctest --test-dir $(TESTS_BUILD_NATIVE) --output-on-failure

# GoogleTest suite: separate CMake build path (engines stay on plain make).
# First configure downloads a pinned gtest via FetchContent unless a system
# GTest is found.
TESTS_BUILD ?= tests/build
test:
	cmake -S tests -B $(TESTS_BUILD)
	cmake --build $(TESTS_BUILD) -j
	ctest --test-dir $(TESTS_BUILD) --output-on-failure

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f libmoty-hw.a libmoty-nn.a hw/hw.o nn/alloc.o nn/quant.o moty lfm2.o qwenmoe.o gemma.o olmoe.o qwen.o glm.o glm glm.exe olmoe olmoe.exe qwen qwen.exe gemma gemma.exe
	rm -rf $(TESTS_BUILD)
	rm -rf $(TESTS_BUILD_NATIVE)

.PHONY: all glm olmoe qwen qwenmoe gemma portable portable-v4 portable-sve2 test test-native check clean
