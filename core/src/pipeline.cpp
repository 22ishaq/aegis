#include "aegis/pipeline.hpp"

#include <chrono>
#include <thread>

#include "aegis/spsc_ring.hpp"

namespace aegis {

namespace {

using Clock = std::chrono::steady_clock;

inline uint64_t now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

struct Envelope {
    Transaction txn;
    uint64_t enqueue_ns;
};

}  // namespace

PipelineReport run_pipeline(Scorer& scorer, const std::vector<Transaction>& txns, const PipelineConfig& cfg,
                            std::vector<ScoreResult>* flagged_out) {
    SpscRing<Envelope> ring(cfg.ring_capacity);
    PipelineReport report;
    const std::size_t n = txns.size();
    const uint64_t start = now_ns();

    std::thread producer([&] {
        const double ns_per_event = cfg.target_rate > 0.0 ? 1e9 / cfg.target_rate : 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (ns_per_event > 0.0) {
                // Busy-wait to the scheduled arrival time. sleep() is far too
                // coarse (tens of microseconds) to pace a million events/sec.
                const uint64_t due = start + static_cast<uint64_t>(static_cast<double>(i) * ns_per_event);
                while (now_ns() < due) {
                }
            }
            Envelope e{txns[i], now_ns()};
            while (!ring.try_push(e)) std::this_thread::yield();  // backpressure
        }
    });

    Envelope e{};
    std::size_t done = 0;
    while (done < n) {
        if (!ring.try_pop(e)) continue;  // spin: the consumer core is dedicated
        const uint64_t t0 = now_ns();
        const ScoreResult r = scorer.score(e.txn);
        const uint64_t t1 = now_ns();
        report.service_ns.record(t1 - t0);
        report.end_to_end_ns.record(t1 - e.enqueue_ns);
        if (r.decision == Decision::Review) ++report.reviews;
        if (r.decision == Decision::Block) ++report.blocks;
        if (flagged_out && r.decision != Decision::Allow) flagged_out->push_back(r);
        ++done;
    }
    producer.join();

    report.processed = done;
    report.seconds = static_cast<double>(now_ns() - start) / 1e9;
    report.throughput = report.seconds > 0 ? static_cast<double>(done) / report.seconds : 0.0;
    return report;
}

}  // namespace aegis
