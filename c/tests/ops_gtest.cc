// Glue gtest per ops_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int op_rmsnorm(void);
int op_silu_mul(void);
int op_softmax(void);
int op_axpy_add(void);
int op_shortconv(void);
int op_rope_table(void);
}

TEST(HwOps, RmsNorm)     { EXPECT_EQ(op_rmsnorm(), 0); }
TEST(HwOps, SiluMul)     { EXPECT_EQ(op_silu_mul(), 0); }
TEST(HwOps, Softmax)     { EXPECT_EQ(op_softmax(), 0); }
TEST(HwOps, AxpyAdd)     { EXPECT_EQ(op_axpy_add(), 0); }
TEST(HwOps, ShortConv)   { EXPECT_EQ(op_shortconv(), 0); }
TEST(HwOps, RopeTable)   { EXPECT_EQ(op_rope_table(), 0); }
