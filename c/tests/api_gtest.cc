// Glue gtest per api_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int ap_lifecycle(void);
int ap_continuation(void);
int ap_abort_callback(void);
int ap_abort_thread(void);
int ap_inject(void);
int ap_errors(void);
int ap_cycles(void);
}

TEST(Api, Lifecycle)       { EXPECT_EQ(ap_lifecycle(), 0); }
TEST(Api, Continuation)    { EXPECT_EQ(ap_continuation(), 0); }
TEST(Api, AbortCallback)   { EXPECT_EQ(ap_abort_callback(), 0); }
TEST(Api, AbortThread)     { EXPECT_EQ(ap_abort_thread(), 0); }
TEST(Api, Inject)          { EXPECT_EQ(ap_inject(), 0); }
TEST(Api, Errors)          { EXPECT_EQ(ap_errors(), 0); }
TEST(Api, Cycles)          { EXPECT_EQ(ap_cycles(), 0); }
