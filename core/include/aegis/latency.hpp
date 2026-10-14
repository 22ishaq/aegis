// Log-linear latency histogram (the idea behind HdrHistogram).
//
// Averages hide the slow requests that users actually notice, so latency is
// reported as percentiles (p50, p99, p99.9). Storing every sample to sort later
// costs memory and time; instead each sample increments one of ~2,000 counters.
// Every power-of-two range [2^k, 2^(k+1)) is split into 32 equal sub-buckets,
// so any reported value is within ~3% of the true one, from 1 ns to centuries.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace aegis {

class LatencyHistogram {
public:
    static constexpr int kSubBits = 5;
    static constexpr uint64_t kSub = 1ULL << kSubBits;  // 32 sub-buckets per octave

    void record(uint64_t v) noexcept {
        ++counts_[index_of(v)];
        ++count_;
        sum_ += static_cast<double>(v);
        min_ = std::min(min_, v);
        max_ = std::max(max_, v);
    }

    void merge(const LatencyHistogram& o) noexcept {
        for (std::size_t i = 0; i < counts_.size(); ++i) counts_[i] += o.counts_[i];
        count_ += o.count_;
        sum_ += o.sum_;
        min_ = std::min(min_, o.min_);
        max_ = std::max(max_, o.max_);
    }

    // p in [0, 100]. Returns the midpoint of the bucket holding that rank.
    uint64_t percentile(double p) const noexcept {
        if (count_ == 0) return 0;
        if (p >= 100.0) return max_;  // the extremes are tracked exactly
        if (p <= 0.0) return min_;
        const double clamped = std::clamp(p, 0.0, 100.0);
        auto rank = static_cast<uint64_t>(clamped / 100.0 * static_cast<double>(count_));
        rank = std::clamp<uint64_t>(rank, 1, count_);
        uint64_t seen = 0;
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            seen += counts_[i];
            if (seen >= rank) return std::clamp(midpoint_of(i), min_, max_);
        }
        return max_;
    }

    uint64_t count() const noexcept { return count_; }
    uint64_t min() const noexcept { return count_ ? min_ : 0; }
    uint64_t max() const noexcept { return max_; }
    double mean() const noexcept { return count_ ? sum_ / static_cast<double>(count_) : 0.0; }

private:
    static std::size_t index_of(uint64_t v) noexcept {
        if (v < kSub) return static_cast<std::size_t>(v);
        const int msb = 63 - std::countl_zero(v);
        const int e = msb - kSubBits + 1;               // which octave (>= 1)
        const uint64_t mantissa = v >> (msb - kSubBits);  // top 6 bits: [32, 63]
        return static_cast<std::size_t>(e) * kSub + static_cast<std::size_t>(mantissa - kSub);
    }

    static uint64_t midpoint_of(std::size_t idx) noexcept {
        if (idx < kSub) return idx;
        const uint64_t e = idx / kSub;
        const uint64_t mantissa = idx % kSub + kSub;
        const uint64_t low = mantissa << (e - 1);
        const uint64_t width = 1ULL << (e - 1);
        return low + width / 2;
    }

    std::array<uint64_t, 64 * kSub> counts_{};
    uint64_t count_ = 0;
    double sum_ = 0.0;
    uint64_t min_ = std::numeric_limits<uint64_t>::max();
    uint64_t max_ = 0;
};

}  // namespace aegis
