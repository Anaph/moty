/* hw.c — the single compiled implementation of the hw contract (M1).
 * Compiled ONCE per build with the engine CFLAGS: the -march tier this
 * TU selects is the tier the whole program runs (compile-time dispatch,
 * no CPUID at runtime — see docs/modularization-plan.md). */
#include "hw_impl.h"
