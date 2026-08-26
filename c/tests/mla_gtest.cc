// Glue gtest per mla_tests.c.
#include <gtest/gtest.h>

extern "C" {
int mla_absorb_eq_reconstruct(void);
int mla_encode_correctness(void);
int mla_value_accum_correctness(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(MlaEquivalence, AbsorbEqReconstruct, mla_absorb_eq_reconstruct)
C_TEST(MlaEncode,     RmsnormAndRope,      mla_encode_correctness)
C_TEST(MlaValue,      AccumMatchesRef,     mla_value_accum_correctness)
