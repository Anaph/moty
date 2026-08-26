// Glue gtest per prof_tests.c: il file viene compilato due volte (senza/ con
// COLIBRI_PROF) e rinominato via -Dprof_tests_main. Solo macro TEST qui.
#include <gtest/gtest.h>

extern "C" {
int prof_tests_main(void);
int prof_tests_on_main(void);
}

TEST(Prof, MacrosOffDisappear) { EXPECT_EQ(0, prof_tests_main()); }
TEST(Prof, MacrosOnAccumulate) { EXPECT_EQ(0, prof_tests_on_main()); }
