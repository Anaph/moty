// Glue gtest per kernels_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int kt_dot_i4g8p(void);
int kt_dot_i4i8p(void);
int kt_grouped_batch(void);
int kt_argmax(void);
int kt_dist(void);
int kt_conv_causality(void);
int kt_report(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(KernelsI4G8p,   GroupedVnniVsDequantRef, kt_dot_i4g8p)
C_TEST(KernelsI4I8p,  PermutedBitExact,        kt_dot_i4i8p)
C_TEST(KernelsGrouped, BatchInvariance,        kt_grouped_batch)
C_TEST(KernelsArgmax,  ParallelEqualsSerial,   kt_argmax)
C_TEST(KernelsDist,    DistributionCoherent,   kt_dist)
C_TEST(KernelsConv,    CausalityRequiresEngine, kt_conv_causality)
C_TEST(KernelsInfo,    ReportBackend,          kt_report)
