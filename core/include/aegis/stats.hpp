// Streaming statistics: every function here is O(1) time and O(1) memory per
// event, which is what lets the feature engine keep up with millions of
// transactions per second without storing raw history.
#pragma once

#include <cmath>
#include <cstdint>

namespace aegis {

// Welford's online algorithm for mean and variance.
//
// The textbook formula var = E[x^2] - E[x]^2 subtracts two large, nearly equal
// numbers and loses precision ("catastrophic cancellation"). Welford instead
// tracks the running mean and the sum of squared deviations from it (m2),
// updating both incrementally. It is numerically stable and needs no history.
struct Welford {
    uint32_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x) noexcept {
        ++n;
        const double delta = x - mean;
        mean += delta / n;
        m2 += delta * (x - mean);  // note: uses the *updated* mean
    }

    double variance() const noexcept { return n > 1 ? m2 / (n - 1) : 0.0; }  // sample variance
    double stddev() const noexcept { return std::sqrt(variance()); }
};

// Exponentially-decayed counter: an O(1) stand-in for "how many events in the
// last tau seconds". Instead of storing timestamps for a sliding window, we
// keep one number and shrink it by exp(-dt/tau) as time passes. An event that
// happened tau seconds ago contributes e^-1 ~ 0.37; one from 3*tau ago ~ 0.05.
inline double decay(double value, double dt_seconds, double tau_seconds) noexcept {
    if (dt_seconds <= 0.0) return value;
    return value * std::exp(-dt_seconds / tau_seconds);
}

inline constexpr double kEarthRadiusKm = 6371.0088;  // IUGG mean radius
inline constexpr double kPi = 3.14159265358979323846;

// Great-circle distance between two lat/lon points (haversine formula).
// Accurate to ~0.5% (it assumes a spherical Earth), which is plenty for
// detecting a card used in London and Lagos twenty minutes apart.
inline double haversine_km(double lat1, double lon1, double lat2, double lon2) noexcept {
    constexpr double kDeg = kPi / 180.0;
    const double dlat = (lat2 - lat1) * kDeg;
    const double dlon = (lon2 - lon1) * kDeg;
    const double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
                     std::cos(lat1 * kDeg) * std::cos(lat2 * kDeg) * std::sin(dlon / 2) * std::sin(dlon / 2);
    return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(std::fmin(1.0, a)));
}

}  // namespace aegis
