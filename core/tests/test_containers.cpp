// Ring buffer, hash map and latency histogram.
#include <random>
#include <thread>
#include <unordered_map>

#include "aegis/flat_map.hpp"
#include "aegis/latency.hpp"
#include "aegis/spsc_ring.hpp"
#include "check.hpp"

TEST(ring_rejects_non_power_of_two) {
    CHECK_THROWS(aegis::SpscRing<int>(100));
    CHECK_THROWS(aegis::SpscRing<int>(1));
}

TEST(ring_fifo_and_full_empty) {
    aegis::SpscRing<int> r(4);
    int v = 0;
    CHECK(!r.try_pop(v));
    for (int i = 0; i < 4; ++i) CHECK(r.try_push(i));
    CHECK(!r.try_push(99));  // full
    for (int i = 0; i < 4; ++i) {
        CHECK(r.try_pop(v));
        CHECK(v == i);
    }
    CHECK(!r.try_pop(v));
    // Indices keep growing past the capacity; wrap-around must still be FIFO.
    for (int round = 0; round < 1000; ++round) {
        CHECK(r.try_push(round));
        CHECK(r.try_pop(v) && v == round);
    }
}

TEST(ring_two_thread_stress) {
    // One producer, one consumer, a tiny ring to force constant contention.
    // Any lost, duplicated or reordered element breaks the sequence check.
    constexpr uint64_t kN = 2'000'000;
    aegis::SpscRing<uint64_t> r(64);
    std::thread producer([&] {
        for (uint64_t i = 0; i < kN; ++i) {
            while (!r.try_push(i)) std::this_thread::yield();
        }
    });
    uint64_t expected = 0;
    bool in_order = true;
    uint64_t v = 0;
    while (expected < kN) {
        if (!r.try_pop(v)) continue;
        if (v != expected) in_order = false;
        ++expected;
    }
    producer.join();
    CHECK(in_order);
    CHECK(r.size_approx() == 0);
}

TEST(flat_map_matches_unordered_map) {
    aegis::FlatMap<uint64_t> m(8);  // start tiny to force many resizes
    std::unordered_map<uint64_t, uint64_t> ref;
    std::mt19937_64 rng(7);
    for (int i = 0; i < 200'000; ++i) {
        const uint64_t k = (rng() % 50'000) + 1;
        bool inserted = false;
        uint64_t& v = m.get_or_insert(k, inserted);
        CHECK(inserted == (ref.find(k) == ref.end()));
        v += k;
        ref[k] += k;
    }
    CHECK(m.size() == ref.size());
    bool all_equal = true;
    for (const auto& [k, v] : ref) {
        const uint64_t* got = m.find(k);
        if (!got || *got != v) all_equal = false;
    }
    CHECK(all_equal);
    CHECK(m.find(999'999) == nullptr);
    bool ins = false;
    CHECK_THROWS(m.get_or_insert(0, ins));
}

TEST(latency_histogram_percentiles) {
    aegis::LatencyHistogram h;
    for (uint64_t v = 1; v <= 100'000; ++v) h.record(v);
    CHECK(h.count() == 100'000);
    CHECK_NEAR(h.percentile(50), 50'000, 50'000 * 0.035);
    CHECK_NEAR(h.percentile(99), 99'000, 99'000 * 0.035);
    CHECK_NEAR(h.mean(), 50'000.5, 1e-6);
    CHECK(h.min() == 1 && h.max() == 100'000);
    CHECK(h.percentile(100) == 100'000);
}
