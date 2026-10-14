#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "aegis/mlp.hpp"
#include "aegis/types.hpp"

namespace testing {

// Deterministic random MLP with the given layer widths, e.g. {20, 16, 1}.
inline aegis::Mlp make_mlp(const std::vector<uint32_t>& widths, uint64_t seed = 42, float review = 0.5f,
                           float block = 0.9f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::uniform_real_distribution<float> pos(0.5f, 2.0f);
    const std::size_t n = widths.front();
    std::vector<float> mean(n), stddev(n);
    for (std::size_t i = 0; i < n; ++i) {
        mean[i] = normal(rng);
        stddev[i] = pos(rng);
    }
    std::vector<aegis::Mlp::Layer> layers;
    for (std::size_t l = 0; l + 1 < widths.size(); ++l) {
        aegis::Mlp::Layer L;
        L.in = widths[l];
        L.out = widths[l + 1];
        L.w.resize(std::size_t(L.in) * L.out);
        L.b.resize(L.out);
        for (float& w : L.w) w = normal(rng);
        for (float& b : L.b) b = normal(rng) * 0.1f;
        layers.push_back(std::move(L));
    }
    return aegis::Mlp(mean, stddev, layers, review, block);
}

inline aegis::Transaction txn(uint64_t account, int64_t ts_ms, int64_t cents, double lat = 51.5074,
                              double lon = -0.1278, uint64_t device = 1, uint64_t merchant = 100,
                              uint16_t country = 826, uint16_t mcc = 5411,
                              aegis::Channel ch = aegis::Channel::CardPresent) {
    static uint64_t next_id = 1;
    aegis::Transaction t;
    t.txn_id = next_id++;
    t.account_id = account;
    t.merchant_id = merchant;
    t.device_id = device;
    t.timestamp_ms = ts_ms;
    t.amount_cents = cents;
    t.lat = lat;
    t.lon = lon;
    t.mcc = mcc;
    t.country = country;
    t.channel = ch;
    return t;
}

inline std::string temp_path(const std::string& name) {
    const auto dir = std::filesystem::temp_directory_path() / "aegis_tests";
    std::filesystem::create_directories(dir);
    const auto p = dir / name;
    std::filesystem::remove(p);
    std::filesystem::remove(p.string() + ".sig");
    return p.string();
}

}  // namespace testing
