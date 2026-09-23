// Glue gtest per lfm2_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int lt_cfg_dense(void);
int lt_load_small_no_mats(void);
int lt_ref_f32(void);
int lt_ref_q8(void);
int lt_kv8_alloc(void);
int lt_fused_bitexact(void);
int lt_packed_roundtrip(void);
}

TEST(Lfm2, ConfigDenseHf)       { EXPECT_EQ(lt_cfg_dense(), 0); }
TEST(Lfm2, LoadSmallNoMats)     { EXPECT_EQ(lt_load_small_no_mats(), 0); }
TEST(Lfm2, RefF32PrefillDecode) { EXPECT_EQ(lt_ref_f32(), 0); }
TEST(Lfm2, RefQ8)               { EXPECT_EQ(lt_ref_q8(), 0); }
TEST(Lfm2, Kv8ScaleArrays)      { EXPECT_EQ(lt_kv8_alloc(), 0); }
TEST(Lfm2, FusedProjectionsBitExact) { EXPECT_EQ(lt_fused_bitexact(), 0); }
TEST(Lfm2, PackedContainerRoundTrip) { EXPECT_EQ(lt_packed_roundtrip(), 0); }
