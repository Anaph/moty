// Glue gtest per attn_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int at_causality_f32(void);
int at_causality_kv8(void);
int at_vs_reference(void);
int at_fused_vs_f32(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { EXPECT_EQ(0, fn()); }

C_TEST(AttnGQA, BatchEqualsIncrementalF32, at_causality_f32)
C_TEST(AttnGQA, BatchEqualsIncrementalKv8, at_causality_kv8)
C_TEST(AttnGQA, MatchesSerialReference,    at_vs_reference)
C_TEST(AttnGQA, FusedI4gMatchesF32,        at_fused_vs_f32)
