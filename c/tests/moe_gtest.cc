// Glue gtest per moe_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int moe_topk_ref(void);
int moe_cache_lfru(void);
int moe_cache_cap1(void);
int moe_cache_pin(void);
int moe_route_bias_test(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(MoeTopk, BitIdenticalNaive, moe_topk_ref)
C_TEST(MoeCache, LfruEvictionReuse, moe_cache_lfru)
C_TEST(MoeCache, CapClampToOne,     moe_cache_cap1)
C_TEST(MoeCache, PinNeverEvicted,   moe_cache_pin)
C_TEST(MoeRoute, BiasToResident,    moe_route_bias_test)
