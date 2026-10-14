// Two-thread streaming pipeline: an ingest thread feeds transactions through
// the lock-free ring to a scoring thread, which records latency percentiles.
//
//   [ingest thread] --SpscRing--> [scoring thread: features -> model -> rules]
//
// This is the shape of a real low-latency system: network/parsing work on one
// core, a single-threaded, cache-hot decision loop on another. The scoring
// thread owns all account state, so it needs no locks at all. To scale out,
// shard accounts across N such pipelines by hash(account_id).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aegis/latency.hpp"
#include "aegis/scorer.hpp"
#include "aegis/types.hpp"

namespace aegis {

struct PipelineConfig {
    std::size_t ring_capacity = 1 << 16;
    // Arrival rate in events/second for the ingest thread. 0 = push as fast
    // as possible, which measures maximum throughput. A finite rate measures
    // latency under a realistic load instead of under a permanent backlog.
    double target_rate = 0.0;
};

struct PipelineReport {
    uint64_t processed = 0;
    uint64_t reviews = 0;
    uint64_t blocks = 0;
    double seconds = 0.0;
    double throughput = 0.0;           // events per second
    LatencyHistogram service_ns;       // time spent inside Scorer::score
    LatencyHistogram end_to_end_ns;    // enqueue -> decision, includes queueing
};

PipelineReport run_pipeline(Scorer& scorer, const std::vector<Transaction>& txns, const PipelineConfig& cfg,
                            std::vector<ScoreResult>* flagged_out = nullptr);

}  // namespace aegis
