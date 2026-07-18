// RED: "Real time" must restore x1 pacing (playback contract).
// Repro: slider x16 -> key 1 ("Real time (1)") -> scene keeps running x16
// forever: set_real_time never touched time_scale_ (1D/2D overrides are
// no-ops, base only flips stepping_). Contract: after set_real_time(),
// time_scale() == 1 on EVERY director.

#include <gtest/gtest.h>

import ses.scenario.tunneling1d_director;
import ses.scenario.doubleslit2d_director;

namespace {

// The user's repro family: 1D scenes, where set_real_time was a pure no-op.
TEST(TimeScaleRealTime, Line1dFamilyRestoresUnitScale) {
    ses_shell::Tunneling1DDirector d;
    d.set_time_scale(16);
    ASSERT_EQ(d.time_scale(), 16);
    d.set_real_time();
    EXPECT_EQ(d.time_scale(), 1);
}

// Leaf-local time_scale_ copy (derives ScenarioDirector directly).
TEST(TimeScaleRealTime, Lattice2dFamilyRestoresUnitScale) {
    ses_shell::DoubleSlit2DDirector d;
    d.set_time_scale(16);
    ASSERT_EQ(d.time_scale(), 16);
    d.set_real_time();
    EXPECT_EQ(d.time_scale(), 1);
}

}  // namespace
