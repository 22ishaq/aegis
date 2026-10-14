// Aegis micro- and macro-benchmarks.
//
//   aegis_bench [--model models/aegis.model] [--n 2000000] [--json reports/bench.json]
//
// Each stage is timed in isolation and then end to end, so a regression can be
// traced to the component that caused it. Numbers depend on the machine; the
// JSON output records the compiler and core count next to every result.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "aegis/features.hpp"
#include "aegis/mlp.hpp"
#include "aegis/pipeline.hpp"
#include "aegis/scorer.hpp"
#include "aegis/sha256.hpp"

using namespace aegis;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Synthetic stream: accounts with a home city, a usual device and spend level,
// and ~1% anomalous events, enough to exercise every code path realistically.
std::vector<Transaction> synth(std::size_t n, std::size_t accounts, uint64_t seed) {
    struct City {
        double lat, lon;
        uint16_t country;
    };
    static constexpr City kCities[] = {
        {51.5074, -0.1278, 826}, {40.7128, -74.0060, 840}, {48.8566, 2.3522, 250}, {35.6762, 139.6503, 392},
        {1.3521, 103.8198, 702}, {-33.8688, 151.2093, 36}, {52.5200, 13.4050, 276}, {19.0760, 72.8777, 356},
        {6.5244, 3.3792, 566},   {-23.5505, -46.6333, 76},
    };
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick_acct(1, accounts);
    std::uniform_int_distribution<int> pick_city(0, 9);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.6);
    static constexpr uint16_t kMccs[] = {5411, 5812, 5541, 4111, 5311, 5999, 5732, 7995};

    std::vector<Transaction> out(n);
    const int64_t t0 = 1'700'000'000'000;
    const int64_t span_ms = 30LL * 86'400'000;
    for (std::size_t i = 0; i < n; ++i) {
        const uint64_t a = pick_acct(rng);
        const City& home = kCities[mix64(a) % 10];
        const bool anomaly = u01(rng) < 0.01;
        const City& c = anomaly ? kCities[pick_city(rng)] : home;
        const double base = 2.5 + static_cast<double>(mix64(a * 7) % 100) / 50.0;  // per-account spend level
        Transaction& t = out[i];
        t.txn_id = i + 1;
        t.account_id = a;
        t.merchant_id = a * 31 + (anomaly ? 10'000 + rng() % 1000 : rng() % 20);
        t.device_id = anomaly ? rng() : a * 10 + 1;
        t.timestamp_ms = t0 + static_cast<int64_t>(static_cast<double>(i) / static_cast<double>(n) * span_ms);
        t.amount_cents = std::max<int64_t>(1, static_cast<int64_t>(std::exp(base + noise(rng) + (anomaly ? 2.0 : 0.0)) * 100));
        t.lat = c.lat + noise(rng) * 0.05;
        t.lon = c.lon + noise(rng) * 0.05;
        t.country = c.country;
        t.mcc = kMccs[rng() % 8];
        t.channel = static_cast<Channel>(rng() % 3);
    }
    return out;
}

Mlp load_or_make_model(const std::string& path) {
    if (!path.empty() && std::filesystem::exists(path)) {
        std::printf("model: %s\n", path.c_str());
        return Mlp::load(path);
    }
    std::printf("model: random 20-32-16-1 (no trained model found; speed is unaffected)\n");
    std::mt19937_64 rng(1);
    std::normal_distribution<float> d(0.0f, 0.3f);
    std::vector<uint32_t> widths = {kNumFeatures, 32, 16, 1};
    std::vector<Mlp::Layer> layers;
    for (std::size_t l = 0; l + 1 < widths.size(); ++l) {
        Mlp::Layer L;
        L.in = widths[l];
        L.out = widths[l + 1];
        L.w.resize(std::size_t(L.in) * L.out);
        L.b.assign(L.out, 0.0f);
        for (float& w : L.w) w = d(rng);
        layers.push_back(std::move(L));
    }
    layers.back().b[0] = -6.0f;  // keep the alert rate realistic (~1%)
    return Mlp(std::vector<float>(kNumFeatures, 0.0f), std::vector<float>(kNumFeatures, 1.0f), layers, 0.5f, 0.9f);
}

struct Json {
    std::string body;
    void add(const std::string& k, double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.6g", v);
        body += (body.empty() ? "" : ",\n") + std::string("  \"") + k + "\": " + buf;
    }
    void add(const std::string& k, const std::string& v) {
        body += (body.empty() ? "" : ",\n") + std::string("  \"") + k + "\": \"" + v + "\"";
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/aegis.model";
    std::string json_path;
    std::size_t n = 2'000'000;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--n")) n = std::stoul(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--json")) json_path = argv[i + 1];
    }

    Json json;
#if defined(__clang__)
    const std::string compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
    const std::string compiler = "gcc " __VERSION__;
#else
    const std::string compiler = "unknown";
#endif
    json.add("compiler", compiler);
    json.add("hardware_threads", std::thread::hardware_concurrency());
    json.add("events", static_cast<double>(n));
    std::printf("Aegis benchmark | %s | %u hw threads | %zu events\n\n", compiler.c_str(),
                std::thread::hardware_concurrency(), n);

    const Mlp model = load_or_make_model(model_path);
    const std::vector<Transaction> txns = synth(n, 100'000, 7);

    // 1. SHA-256 throughput (audit log + model integrity).
    {
        std::vector<uint8_t> buf(64 << 20, 0xAB);
        const auto t0 = Clock::now();
        const Digest d = Sha256::hash(buf.data(), buf.size());
        const double s = seconds_since(t0);
        std::printf("%-34s %10.1f MB/s   (digest %02x%02x...)\n", "sha256", 64.0 / s, d[0], d[1]);
        json.add("sha256_mb_per_s", 64.0 / s);
    }

    // 2. Feature engine alone.
    {
        FeatureEngine fe(100'000);
        double sink = 0;
        const auto t0 = Clock::now();
        for (const auto& t : txns) sink += fe.process(t).x[0];
        const double s = seconds_since(t0);
        std::printf("%-34s %10.1f ns/txn  (%.2f M txn/s)%s\n", "feature engine", s * 1e9 / n, n / s / 1e6,
                    sink == 42 ? " " : "");
        json.add("features_ns_per_txn", s * 1e9 / n);
    }

    // 3. Model inference alone.
    {
        FeatureEngine fe(100'000);
        std::vector<FeatureVector> xs;
        xs.reserve(n);
        for (const auto& t : txns) xs.push_back(fe.process(t).x);
        double sink = 0;
        const auto t0 = Clock::now();
        for (const auto& x : xs) sink += model.predict_proba(x.data());
        const double s = seconds_since(t0);
        std::printf("%-34s %10.1f ns/inference%s\n", "mlp forward pass", s * 1e9 / n, sink < 0 ? " " : "");
        json.add("mlp_ns_per_inference", s * 1e9 / n);

        float attr[kNumFeatures];
        const std::size_t m = std::min<std::size_t>(n, 500'000);
        const auto t1 = Clock::now();
        for (std::size_t i = 0; i < m; ++i) sink += model.attribute(xs[i].data(), attr);
        const double s1 = seconds_since(t1);
        std::printf("%-34s %10.1f ns/explanation\n", "mlp forward + backward (explain)", s1 * 1e9 / m);
        json.add("explain_ns", s1 * 1e9 / m);
    }

    // 4. Full scorer, single thread.
    {
        Scorer scorer(model, 100'000);
        std::size_t flagged = 0;
        const auto t0 = Clock::now();
        for (const auto& t : txns) flagged += scorer.score(t).decision != Decision::Allow;
        const double s = seconds_since(t0);
        std::printf("%-34s %10.1f ns/txn  (%.2f M txn/s, %.2f%% flagged)\n", "scorer (features+model+rules)",
                    s * 1e9 / n, n / s / 1e6, 100.0 * flagged / n);
        json.add("scorer_ns_per_txn", s * 1e9 / n);
        json.add("scorer_flag_rate", static_cast<double>(flagged) / n);
    }

    // 5. Two-thread pipeline at full speed (throughput).
    {
        Scorer scorer(model, 100'000);
        const auto rep = run_pipeline(scorer, txns, PipelineConfig{1 << 16, 0.0});
        std::printf("%-34s %10.2f M txn/s\n", "pipeline, saturated", rep.throughput / 1e6);
        std::printf("%-34s p50 %6llu ns | p99 %6llu ns | p99.9 %6llu ns\n", "  service time",
                    (unsigned long long)rep.service_ns.percentile(50), (unsigned long long)rep.service_ns.percentile(99),
                    (unsigned long long)rep.service_ns.percentile(99.9));
        json.add("pipeline_throughput", rep.throughput);
        json.add("service_p50_ns", static_cast<double>(rep.service_ns.percentile(50)));
        json.add("service_p99_ns", static_cast<double>(rep.service_ns.percentile(99)));
        json.add("service_p999_ns", static_cast<double>(rep.service_ns.percentile(99.9)));
    }

    // 6. Paced pipeline: realistic load, end-to-end latency including the queue hop.
    {
        const double rate = 1'000'000.0;
        const std::size_t m = std::min<std::size_t>(n, 2'000'000);
        std::vector<Transaction> sub(txns.begin(), txns.begin() + static_cast<std::ptrdiff_t>(m));
        Scorer scorer(model, 100'000);
        const auto rep = run_pipeline(scorer, sub, PipelineConfig{1 << 16, rate});
        std::printf("%-34s p50 %6llu ns | p99 %6llu ns | p99.9 %6llu ns\n", "pipeline @ 1M txn/s, end-to-end",
                    (unsigned long long)rep.end_to_end_ns.percentile(50),
                    (unsigned long long)rep.end_to_end_ns.percentile(99),
                    (unsigned long long)rep.end_to_end_ns.percentile(99.9));
        json.add("paced_rate", rate);
        json.add("e2e_p50_ns", static_cast<double>(rep.end_to_end_ns.percentile(50)));
        json.add("e2e_p99_ns", static_cast<double>(rep.end_to_end_ns.percentile(99)));
        json.add("e2e_p999_ns", static_cast<double>(rep.end_to_end_ns.percentile(99.9)));
    }

    if (!json_path.empty()) {
        std::ofstream f(json_path);
        f << "{\n" << json.body << "\n}\n";
        std::printf("\nwrote %s\n", json_path.c_str());
    }
    return 0;
}
