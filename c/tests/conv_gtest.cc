// Glue gtest per conv_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int cv_causality(void);
int cv_fused_vs_legacy(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { EXPECT_EQ(0, fn()); }

C_TEST(ConvFused, BatchEqualsPerToken, cv_causality)
C_TEST(ConvFused, VnniMatchesLegacy,  cv_fused_vs_legacy)
