#include <cmath>
#include <random>
#include <vector>

#include "aegis/features.hpp"
#include "aegis/stats.hpp"
#include "check.hpp"
#include "helpers.hpp"

using namespace aegis;
using testing::txn;

constexpr int64_t kT0 = 1'700'000'000'000;  // Nov 2023, in ms
constexpr int64_t kMin = 60'000;
constexpr int64_t kHour = 3'600'000;

TEST(welford_matches_two_pass) {
    std::mt19937_64 rng(1);
    std::lognormal_distribution<double> d(3.0, 1.0);
    std::vector<double> xs(10'000);
    Welford w;
    for (double& x : xs) {
        x = d(rng);
        w.add(x);
    }
    double mean = 0;
    for (double x : xs) mean += x;
    mean /= xs.size();
    double ss = 0;
    for (double x : xs) ss += (x - mean) * (x - mean);
    CHECK_NEAR(w.mean, mean, 1e-9 * mean);
    CHECK_NEAR(w.variance(), ss / (xs.size() - 1), 1e-6 * ss / xs.size());
}

TEST(haversine_known_distances) {
    CHECK_NEAR(haversine_km(51.5074, -0.1278, 48.8566, 2.3522), 343.5, 2.0);     // London - Paris
    CHECK_NEAR(haversine_km(40.7128, -74.0060, 51.5074, -0.1278), 5570, 15.0);  // New York - London
    CHECK_NEAR(haversine_km(10, 20, 10, 20), 0.0, 1e-9);
}

TEST(first_transaction_has_no_history) {
    FeatureEngine fe;
    const auto r = fe.process(txn(1, kT0, 2500));
    CHECK(r.s.first_txn);
    CHECK(r.x[kNewDevice] == 1.0f && r.x[kNewMerchant] == 1.0f && r.x[kNewCountry] == 1.0f);
    CHECK(r.x[kTxn1h] == 0.0f);  // a transaction must not count itself
    CHECK(r.x[kLogHistoryLen] == 0.0f);
    CHECK(r.x[kAmountZ] == 0.0f);
    CHECK(r.x[kLogKmFromLast] == 0.0f);
    CHECK_NEAR(r.x[kLogAmount], std::log1p(25.0), 1e-6);
}

TEST(second_transaction_sees_decayed_history) {
    FeatureEngine fe;
    fe.process(txn(1, kT0, 2500));
    const auto r = fe.process(txn(1, kT0 + 30 * kMin, 4000));
    CHECK(!r.s.first_txn);
    CHECK(r.x[kNewDevice] == 0.0f && r.x[kNewMerchant] == 0.0f && r.x[kNewCountry] == 0.0f);
    CHECK_NEAR(r.x[kTxn1h], std::exp(-0.5), 1e-6);  // 1 txn, 30 min ago, tau = 1h
    CHECK_NEAR(r.x[kLogSecsSinceLast], std::log1p(1800.0), 1e-5);
    CHECK_NEAR(r.x[kLogHistoryLen], std::log1p(1.0), 1e-6);
    CHECK_NEAR(r.x[kLogAmt24h], std::log1p(25.0 * std::exp(-1800.0 / 86400.0)), 1e-5);
}

TEST(accounts_are_isolated) {
    FeatureEngine fe;
    for (int i = 0; i < 5; ++i) fe.process(txn(1, kT0 + i * kMin, 1000));
    const auto r = fe.process(txn(2, kT0 + 6 * kMin, 1000));
    CHECK(r.s.first_txn);
    CHECK(fe.num_accounts() == 2);
}

TEST(impossible_travel_signal) {
    FeatureEngine fe;
    fe.process(txn(1, kT0, 5000, 51.5074, -0.1278));                // London
    const auto r = fe.process(txn(1, kT0 + kHour, 5000, -33.8688, 151.2093, 1, 100, 36));  // Sydney 1h later
    CHECK(r.s.km_from_last > 16'000);
    CHECK(r.s.speed_kmh > 15'000);
    CHECK(r.x[kNewCountry] == 1.0f);
}

TEST(amount_zscore_flags_outlier) {
    FeatureEngine fe;
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int64_t> cents(2000, 4000);
    for (int i = 0; i < 50; ++i) fe.process(txn(1, kT0 + i * 6 * kHour, cents(rng)));
    const auto normal = fe.process(txn(1, kT0 + 51 * 6 * kHour, 3000));
    const auto huge = fe.process(txn(1, kT0 + 52 * 6 * kHour, 250000));
    CHECK(std::fabs(normal.x[kAmountZ]) < 1.5f);
    CHECK(huge.x[kAmountZ] > 5.0f);
}

TEST(out_of_order_timestamps_do_not_inflate_counters) {
    FeatureEngine fe;
    fe.process(txn(1, kT0 + kHour, 1000));
    const auto r = fe.process(txn(1, kT0, 1000));  // arrives late
    CHECK(r.s.secs_since_last == 0.0);
    CHECK_NEAR(r.x[kTxn1h], 1.0, 1e-6);
}

TEST(card_testing_counter) {
    FeatureEngine fe;
    for (int i = 0; i < 4; ++i) fe.process(txn(1, kT0 + i * kMin, 150, 51.5, -0.12, 1, 500 + i));
    const auto r = fe.process(txn(1, kT0 + 5 * kMin, 120, 51.5, -0.12, 1, 600));
    // 4 prior small txns, 2-5 minutes old, tau = 10 min:
    // e^-0.5 + e^-0.4 + e^-0.3 + e^-0.2 = 2.84
    CHECK_NEAR(r.s.small_10m, std::exp(-0.5) + std::exp(-0.4) + std::exp(-0.3) + std::exp(-0.2), 1e-9);
    CHECK(r.x[kNewMerchant1h] > 3.0f);
}

TEST(validate_rejects_bad_input) {
    CHECK(validate(txn(1, kT0, 100)) == nullptr);
    CHECK(validate(txn(0, kT0, 100)) != nullptr);
    CHECK(validate(txn(1, kT0, 0)) != nullptr);
    CHECK(validate(txn(1, kT0, 100, 95.0, 0.0)) != nullptr);
    CHECK(validate(txn(1, kT0, 100, NAN, 0.0)) != nullptr);
    CHECK(validate(txn(1, -5, 100)) != nullptr);
}
