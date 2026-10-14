// Core value types shared by every stage of the Aegis pipeline.
//
// Design rule: everything on the hot path is a plain, trivially-copyable struct.
// No strings, no heap pointers, no virtual functions. That lets us move
// transactions through a lock-free ring buffer with a single memcpy and keeps
// each one inside two cache lines.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace aegis {

enum class Channel : uint8_t {
    CardPresent = 0,  // chip / contactless at a physical terminal
    Ecommerce = 1,    // card-not-present online purchase
    Transfer = 2,     // account-to-account bank transfer
    Atm = 3,          // cash withdrawal
};

// One payment event. Money is stored as integer cents: floating point cannot
// represent 0.10 exactly, and rounding drift in a ledger is a real bug class.
struct Transaction {
    uint64_t txn_id = 0;
    uint64_t account_id = 0;  // must be non-zero (0 is the hash-map empty marker)
    uint64_t merchant_id = 0;
    uint64_t device_id = 0;
    int64_t timestamp_ms = 0;  // Unix epoch, milliseconds, UTC
    int64_t amount_cents = 0;
    double lat = 0.0;
    double lon = 0.0;
    uint16_t mcc = 0;      // ISO 18245 merchant category code
    uint16_t country = 0;  // ISO 3166-1 numeric
    Channel channel = Channel::CardPresent;
};
static_assert(std::is_trivially_copyable_v<Transaction>);

// Returns nullptr when the transaction is well-formed, otherwise a static
// description of the first problem found. Validation happens once at the edge
// (API / bindings) so the hot path can trust its inputs.
inline const char* validate(const Transaction& t) noexcept {
    if (t.account_id == 0) return "account_id must be non-zero";
    if (t.amount_cents <= 0) return "amount_cents must be positive";
    if (t.amount_cents > 100'000'000'00LL) return "amount_cents exceeds 100M limit";
    if (!(t.lat >= -90.0 && t.lat <= 90.0)) return "lat out of range";  // also rejects NaN
    if (!(t.lon >= -180.0 && t.lon <= 180.0)) return "lon out of range";
    if (static_cast<uint8_t>(t.channel) > 3) return "unknown channel";
    if (t.timestamp_ms < 0) return "timestamp_ms must be non-negative";
    return nullptr;
}

// ---------------------------------------------------------------------------
// Feature schema. The order here IS the model's input contract: the Python
// trainer reads these names through the bindings, so training and serving can
// never silently disagree about which column means what.
// ---------------------------------------------------------------------------
enum Feature : std::size_t {
    kLogAmount = 0,       // log1p(amount in USD)
    kAmountZ,             // z-score of log amount vs this account's history
    kTxn1h,               // exponentially-decayed count of prior txns, tau = 1h
    kTxn24h,              // same, tau = 24h
    kLogAmt24h,           // log1p(decayed USD spend, tau = 24h)
    kLogSecsSinceLast,    // log1p(seconds since previous txn)
    kLogKmFromLast,       // log1p(great-circle km from previous txn location)
    kLogSpeedKmh,         // log1p(implied travel speed from previous txn)
    kNewDevice,           // 1 if device not among the account's recent devices
    kNewMerchant,         // 1 if merchant not among recent merchants
    kNewCountry,          // 1 if country not among recent countries
    kHourSin,             // local hour of day, encoded on the unit circle
    kHourCos,
    kMccRisk,             // prior risk of the merchant category, 0..1
    kCardNotPresent,      // 1 for ecommerce / transfer
    kRoundAmount,         // 1 for whole-hundred amounts >= $100
    kNearThreshold,       // 1 for $9,000-$9,999.99 (just under reporting limit)
    kLogHistoryLen,       // log1p(number of prior txns): thin files are riskier
    kSmallTxn10m,         // decayed count of prior sub-$5 txns, tau = 10min
    kNewMerchant1h,       // decayed count of prior first-seen merchants, tau = 1h
    kNumFeatureIds,
};

inline constexpr std::size_t kNumFeatures = kNumFeatureIds;
using FeatureVector = std::array<float, kNumFeatures>;

inline constexpr std::array<std::string_view, kNumFeatures> kFeatureNames = {
    "log_amount",        "amount_z",         "txn_1h",          "txn_24h",
    "log_amt_24h",       "log_secs_since_last", "log_km_from_last", "log_speed_kmh",
    "new_device",        "new_merchant",     "new_country",     "hour_sin",
    "hour_cos",          "mcc_risk",         "card_not_present", "round_amount",
    "near_threshold",    "log_history_len",  "small_txn_10m",   "new_merchant_1h",
};

enum class Decision : uint8_t { Allow = 0, Review = 1, Block = 2 };

inline constexpr std::string_view to_string(Decision d) noexcept {
    switch (d) {
        case Decision::Allow: return "ALLOW";
        case Decision::Review: return "REVIEW";
        case Decision::Block: return "BLOCK";
    }
    return "UNKNOWN";
}

}  // namespace aegis
