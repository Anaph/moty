// Glue gtest per head_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int hd_build(void);
int hd_full_equivalence(void);
int hd_shortlist(void);
}

TEST(HeadTopK, Build)            { EXPECT_EQ(hd_build(), 0); }
TEST(HeadTopK, FullEquivalence)  { EXPECT_EQ(hd_full_equivalence(), 0); }
TEST(HeadTopK, Shortlist)        { EXPECT_EQ(hd_shortlist(), 0); }
