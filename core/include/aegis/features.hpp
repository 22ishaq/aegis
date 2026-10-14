// Streaming feature engine: turns a raw transaction into a FeatureVector that
// describes how *unusual* it is for this account, in O(1) time per event.
//
// Point-in-time correctness: features are computed from the account's state
// BEFORE the current transaction is folded in. A feature must only use
// information that existed at the moment of the payment; otherwise the model
// trains on data from the future ("label leakage") and looks brilliant offline
// but fails in production. Because the same C++ code computes features for
// training (via the Python bindings) and for live scoring, the two can never
// drift apart ("training/serving skew").
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "aegis/flat_map.hpp"
#include "aegis/stats.hpp"
#include "aegis/types.hpp"

namespace aegis {

// Tiny fixed-capacity set of recently seen ids with round-robin eviction.
// A linear scan over 8-16 integers is faster than any hash set at this size
// (it is one or two cache lines and the compiler vectorises the compare).
template <typename T, std::size_t N>
struct RecentSet {
    std::array<T, N> items{};
    uint8_t count = 0;
    uint8_t next = 0;

    bool contains(T v) const noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (items[i] == v) return true;
        }
        return false;
    }

    void insert(T v) noexcept {
        if (contains(v)) return;
        items[next] = v;
        next = static_cast<uint8_t>((next + 1) % N);
        if (count < N) ++count;
    }
};

// Everything we remember about one account: a few hundred bytes, no history.
struct AccountState {
    int64_t last_ts_ms = 0;
    double last_lat = 0.0;
    double last_lon = 0.0;
    Welford log_amount;              // distribution of log(amount)
    double txn_1h = 0.0;             // decayed counters, all as of last_ts_ms
    double txn_24h = 0.0;
    double amt_24h = 0.0;
    double small_10m = 0.0;
    double new_merchant_1h = 0.0;
    double near_threshold_24h = 0.0;
    RecentSet<uint64_t, 8> devices;
    RecentSet<uint64_t, 16> merchants;
    RecentSet<uint16_t, 4> countries;
};

// Raw (un-logged, un-normalised) quantities the rule engine reasons about.
// Rules are written in business units ("faster than 900 km/h") so a compliance
// officer can read and audit them; the model consumes the transformed features.
struct Signals {
    double amount_usd = 0.0;
    double amount_z = 0.0;
    double km_from_last = 0.0;
    double speed_kmh = 0.0;
    double secs_since_last = 0.0;
    double txn_1h = 0.0;
    double small_10m = 0.0;
    double near_threshold_24h = 0.0;  // prior near-threshold txns, decayed
    double mcc_risk = 0.0;
    bool first_txn = false;
    bool new_device = false;
    bool new_country = false;
};

struct FeatureResult {
    FeatureVector x{};
    Signals s;
};

// Prior fraud risk of a merchant category (0 = benign, 1 = very risky).
double mcc_risk(uint16_t mcc) noexcept;

class FeatureEngine {
public:
    explicit FeatureEngine(std::size_t expected_accounts = 1 << 16);

    // Compute point-in-time features for `t`, then update the account state.
    // Precondition: validate(t) == nullptr.
    FeatureResult process(const Transaction& t);

    std::size_t num_accounts() const noexcept { return accounts_.size(); }
    const AccountState* account(uint64_t id) const noexcept { return accounts_.find(id); }
    void reset() { accounts_.clear(); }

private:
    FlatMap<AccountState> accounts_;
};

}  // namespace aegis
