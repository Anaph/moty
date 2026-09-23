// Glue gtest per q4r4_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int q4_f16_roundtrip(void);
int q4_quant_g32_exact(void);
int q4_pack_noclip(void);
int q4_gemm_int_exact(void);
int q4_matmul_driver(void);
}

TEST(Q4R4, F16RoundTrip)      { EXPECT_EQ(q4_f16_roundtrip(), 0); }
TEST(Q4R4, QuantG32Exact)     { EXPECT_EQ(q4_quant_g32_exact(), 0); }
TEST(Q4R4, PackNoClip)        { EXPECT_EQ(q4_pack_noclip(), 0); }
TEST(Q4R4, GemmIntegerExact)  { EXPECT_EQ(q4_gemm_int_exact(), 0); }
TEST(Q4R4, MatmulDriver)      { EXPECT_EQ(q4_matmul_driver(), 0); }
