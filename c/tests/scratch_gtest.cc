// Glue gtest per scratch_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int sa_alignment(void);
int sa_stability(void);
int sa_growth_preserves(void);
int sa_reset(void);
int sa_two_arenas(void);
int sa_free_reuse(void);
int sa_kernel_pattern(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { EXPECT_EQ(0, fn()); }

C_TEST(Scratch, Alignment64,            sa_alignment)
C_TEST(Scratch, PointerStability,       sa_stability)
C_TEST(Scratch, GrowthPreservesContent, sa_growth_preserves)
C_TEST(Scratch, ResetKeepsCapacity,     sa_reset)
C_TEST(Scratch, TwoArenasIndependent,   sa_two_arenas)
C_TEST(Scratch, FreeAndReuse,           sa_free_reuse)
C_TEST(Scratch, KernelPhasePattern,     sa_kernel_pattern)
