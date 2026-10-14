#include "aegis/features.hpp"

#include <algorithm>
#include <cmath>

namespace aegis {

namespace {

constexpr double kTau10m = 600.0;
constexpr double kTau1h = 3600.0;
constexpr double kTau24h = 86400.0;
constexpr double kNoHistorySeconds = 30.0 * 86400.0;  // "a long time" for a first txn
constexpr double kMinDtForSpeedS = 60.0;              // GPS/clock jitter floor
constexpr double kMaxSpeedKmh = 20000.0;              // clamp absurd values
constexpr double kMinLogStd = 0.1;                    // ~10% spread floor in log space

}  // namespace

double mcc_risk(uint16_t mcc) noexcept {
    switch (mcc) {
        case 7995: return 0.90;  // betting / casino gambling
        case 6051: return 0.90;  // quasi-cash: crypto, money orders
        case 4829: return 0.80;  // wire transfers
        case 5967: return 0.70;  // direct marketing, inbound telesales
        case 5944: return 0.60;  // jewellery
        case 5816: return 0.60;  // digital games / in-game currency
        case 5732: return 0.50;  // electronics
        case 6011: return 0.50;  // ATM cash
        case 5999: return 0.30;  // misc retail
        case 5311: return 0.15;  // department stores
        case 5541: return 0.10;  // fuel
        case 5411: return 0.05;  // groceries
        case 5812: return 0.05;  // restaurants
        case 4111: return 0.05;  // commuter transport
        case 4900: return 0.05;  // utilities
        default: return 0.20;
    }
}

FeatureEngine::FeatureEngine(std::size_t expected_accounts) : accounts_(expected_accounts) {}

FeatureResult FeatureEngine::process(const Transaction& t) {
    bool inserted = false;
    AccountState& s = accounts_.get_or_insert(t.account_id, inserted);
    const bool first = s.log_amount.n == 0;

    const double usd = static_cast<double>(t.amount_cents) / 100.0;
    const double log_amt = std::log1p(usd);

    // Time since the previous transaction. Out-of-order events clamp to 0 so a
    // late-arriving record can never make counters grow backwards.
    const double dt_s =
        first ? kNoHistorySeconds : std::max(0.0, static_cast<double>(t.timestamp_ms - s.last_ts_ms) / 1000.0);

    // Bring every decayed counter forward to "now". These are the values
    // describing PRIOR activity, which is what the features must see.
    const double txn_1h = decay(s.txn_1h, dt_s, kTau1h);
    const double txn_24h = decay(s.txn_24h, dt_s, kTau24h);
    const double amt_24h = decay(s.amt_24h, dt_s, kTau24h);
    const double small_10m = decay(s.small_10m, dt_s, kTau10m);
    const double new_merchant_1h = decay(s.new_merchant_1h, dt_s, kTau1h);
    const double near_thr_24h = decay(s.near_threshold_24h, dt_s, kTau24h);

    // Geography: how far did the card "travel" since its last use, and how fast?
    double km = 0.0;
    double speed = 0.0;
    if (!first) {
        km = haversine_km(s.last_lat, s.last_lon, t.lat, t.lon);
        speed = std::min(kMaxSpeedKmh, km / (std::max(dt_s, kMinDtForSpeedS) / 3600.0));
    }

    const bool new_device = !s.devices.contains(t.device_id);
    const bool new_merchant = !s.merchants.contains(t.merchant_id);
    const bool new_country = !s.countries.contains(t.country);

    // How many standard deviations is this amount from the account's norm?
    double z = 0.0;
    if (s.log_amount.n >= 3) {
        const double sd = std::max(s.log_amount.stddev(), kMinLogStd);
        z = std::clamp((log_amt - s.log_amount.mean) / sd, -10.0, 10.0);
    }

    // Local solar hour from UTC + longitude (15 degrees = 1 hour). Encoding it as
    // (sin, cos) tells the model that 23:00 and 01:00 are neighbours.
    const double utc_hours = std::fmod(static_cast<double>(t.timestamp_ms) / 3.6e6, 24.0);
    const double local_hour = std::fmod(utc_hours + t.lon / 15.0 + 48.0, 24.0);
    const double angle = 2.0 * kPi * local_hour / 24.0;

    const double risk = mcc_risk(t.mcc);
    const bool cnp = t.channel == Channel::Ecommerce || t.channel == Channel::Transfer;
    const bool round_amt = t.amount_cents >= 10000 && t.amount_cents % 10000 == 0;
    const bool near_thr = t.amount_cents >= 900000 && t.amount_cents < 1000000;

    FeatureResult r;
    FeatureVector& x = r.x;
    x[kLogAmount] = static_cast<float>(log_amt);
    x[kAmountZ] = static_cast<float>(z);
    x[kTxn1h] = static_cast<float>(txn_1h);
    x[kTxn24h] = static_cast<float>(txn_24h);
    x[kLogAmt24h] = static_cast<float>(std::log1p(amt_24h));
    x[kLogSecsSinceLast] = static_cast<float>(std::log1p(dt_s));
    x[kLogKmFromLast] = static_cast<float>(std::log1p(km));
    x[kLogSpeedKmh] = static_cast<float>(std::log1p(speed));
    x[kNewDevice] = new_device ? 1.0f : 0.0f;
    x[kNewMerchant] = new_merchant ? 1.0f : 0.0f;
    x[kNewCountry] = new_country ? 1.0f : 0.0f;
    x[kHourSin] = static_cast<float>(std::sin(angle));
    x[kHourCos] = static_cast<float>(std::cos(angle));
    x[kMccRisk] = static_cast<float>(risk);
    x[kCardNotPresent] = cnp ? 1.0f : 0.0f;
    x[kRoundAmount] = round_amt ? 1.0f : 0.0f;
    x[kNearThreshold] = near_thr ? 1.0f : 0.0f;
    x[kLogHistoryLen] = static_cast<float>(std::log1p(static_cast<double>(s.log_amount.n)));
    x[kSmallTxn10m] = static_cast<float>(small_10m);
    x[kNewMerchant1h] = static_cast<float>(new_merchant_1h);

    Signals& sig = r.s;
    sig.amount_usd = usd;
    sig.amount_z = z;
    sig.km_from_last = km;
    sig.speed_kmh = speed;
    sig.secs_since_last = dt_s;
    sig.txn_1h = txn_1h;
    sig.small_10m = small_10m;
    sig.near_threshold_24h = near_thr_24h;
    sig.mcc_risk = risk;
    sig.first_txn = first;
    sig.new_device = new_device;
    sig.new_country = new_country;

    // Only now fold the current transaction into the account's state.
    s.txn_1h = txn_1h + 1.0;
    s.txn_24h = txn_24h + 1.0;
    s.amt_24h = amt_24h + usd;
    s.small_10m = small_10m + (usd < 5.0 ? 1.0 : 0.0);
    s.new_merchant_1h = new_merchant_1h + (new_merchant ? 1.0 : 0.0);
    s.near_threshold_24h = near_thr_24h + (near_thr ? 1.0 : 0.0);
    s.log_amount.add(log_amt);
    s.devices.insert(t.device_id);
    s.merchants.insert(t.merchant_id);
    s.countries.insert(t.country);
    s.last_ts_ms = std::max(s.last_ts_ms, t.timestamp_ms);
    s.last_lat = t.lat;
    s.last_lon = t.lon;
    return r;
}

}  // namespace aegis
