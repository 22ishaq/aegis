#include <vector>

#include "aegis/pipeline.hpp"
#include "aegis/scorer.hpp"
#include "check.hpp"
#include "helpers.hpp"

using namespace aegis;
using testing::txn;

namespace {

constexpr int64_t kT0 = 1'700'000'000'000;
constexpr int64_t kMin = 60'000;
constexpr int64_t kHour = 3'600'000;

// A model that is almost always quiet, so these tests isolate the rule layer.
Scorer quiet_scorer() {
    std::vector<float> mean(kNumFeatures, 0.0f), stddev(kNumFeatures, 1.0f);
    Mlp::Layer L;
    L.in = kNumFeatures;
    L.out = 1;
    L.w.assign(kNumFeatures, 0.0f);
    L.b = {-8.0f};  // sigmoid(-8) ~ 0.0003
    return Scorer(Mlp(mean, stddev, {L}, 0.5f, 0.9f));
}

}  // namespace

TEST(scorer_rejects_wrong_feature_count) { CHECK_THROWS(Scorer(testing::make_mlp({5, 4, 1}))); }

TEST(normal_spending_is_allowed) {
    Scorer s = quiet_scorer();
    for (int i = 0; i < 20; ++i) {
        const auto r = s.score(txn(1, kT0 + i * 5 * kHour, 3000 + i * 10));
        CHECK(r.decision == Decision::Allow);
        CHECK(r.rule_mask == 0);
        CHECK(!r.has_attribution);
    }
}

TEST(rule_card_testing) {
    Scorer s = quiet_scorer();
    s.score(txn(1, kT0, 4500));
    ScoreResult r;
    for (int i = 0; i < 5; ++i) r = s.score(txn(1, kT0 + kHour + i * kMin, 100, 51.5, -0.12, 9, 700 + i));
    CHECK(r.rule_mask & kRuleCardTesting);
    CHECK(r.decision == Decision::Block);  // severity 0.90 >= block threshold 0.9
    CHECK(r.has_attribution);
}

TEST(rule_impossible_travel) {
    Scorer s = quiet_scorer();
    s.score(txn(1, kT0, 5000, 51.5074, -0.1278));
    const auto r = s.score(txn(1, kT0 + 40 * kMin, 5000, 6.5244, 3.3792, 1, 100, 566));  // Lagos
    CHECK(r.rule_mask & kRuleImpossibleTravel);
    CHECK(r.decision == Decision::Block);
}

TEST(legitimate_flight_is_not_impossible_travel) {
    Scorer s = quiet_scorer();
    s.score(txn(1, kT0, 5000, 51.5074, -0.1278));
    // London -> New York in 9 hours is ~620 km/h: fast, but physically possible.
    const auto r = s.score(txn(1, kT0 + 9 * kHour, 5000, 40.7128, -74.0060, 1, 101, 840));
    CHECK(!(r.rule_mask & kRuleImpossibleTravel));
}

TEST(rule_structuring) {
    Scorer s = quiet_scorer();
    const auto first = s.score(txn(1, kT0, 950000, 51.5, -0.12, 1, 42, 826, 4829, Channel::Transfer));
    CHECK(!(first.rule_mask & kRuleStructuring));  // a single $9,500 transfer is not a pattern
    const auto second = s.score(txn(1, kT0 + 3 * kHour, 975000, 51.5, -0.12, 1, 42, 826, 4829, Channel::Transfer));
    CHECK(second.rule_mask & kRuleStructuring);
    CHECK(second.decision != Decision::Allow);
}

TEST(rule_account_takeover) {
    Scorer s = quiet_scorer();
    for (int i = 0; i < 30; ++i) s.score(txn(1, kT0 + i * 8 * kHour, 2000 + (i % 7) * 300));
    // New device, new country, ~50x the usual amount.
    const auto r = s.score(txn(1, kT0 + 31 * 8 * kHour, 120000, 55.7558, 37.6173, 777, 900, 643, 5732,
                               Channel::Ecommerce));
    CHECK(r.rule_mask & kRuleAccountTakeover);
}

TEST(pipeline_processes_everything_in_order) {
    Scorer s = quiet_scorer();
    std::vector<Transaction> txns;
    for (int i = 0; i < 50'000; ++i) txns.push_back(txn(1 + (i % 500), kT0 + i * 1000, 1500 + (i % 13) * 100));
    std::vector<ScoreResult> flagged;
    const auto rep = run_pipeline(s, txns, PipelineConfig{1024, 0.0}, &flagged);
    CHECK(rep.processed == txns.size());
    CHECK(rep.service_ns.count() == txns.size());
    CHECK(rep.throughput > 0);
    CHECK(s.engine().num_accounts() == 500);
}
