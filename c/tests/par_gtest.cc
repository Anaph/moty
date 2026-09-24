// Glue gtest per par_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int par_static_split(void);
int par_dynamic_chunks(void);
int par_team_resize(void);
int par_nested(void);
int par_sleep_wakeup(void);
int par_q4r4_threads_bitexact(void);
}

TEST(Par, StaticSplit)          { EXPECT_EQ(par_static_split(), 0); }
TEST(Par, DynamicChunks)        { EXPECT_EQ(par_dynamic_chunks(), 0); }
TEST(Par, TeamResize)           { EXPECT_EQ(par_team_resize(), 0); }
TEST(Par, Nested)               { EXPECT_EQ(par_nested(), 0); }
TEST(Par, SleepWakeup)          { EXPECT_EQ(par_sleep_wakeup(), 0); }
TEST(Par, Q4R4ThreadsBitExact)  { EXPECT_EQ(par_q4r4_threads_bitexact(), 0); }
