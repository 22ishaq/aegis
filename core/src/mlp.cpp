#include "aegis/mlp.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

namespace aegis {

static_assert(std::endian::native == std::endian::little, "model format assumes a little-endian host");

namespace {

constexpr char kMagic[8] = {'A', 'E', 'G', 'I', 'S', 'M', 'D', 'L'};
constexpr uint32_t kVersion = 1;

// Bounds-checked sequential reader over an untrusted byte buffer. A model file
// is input like any other: never trust a length field without checking it.
class Reader {
public:
    Reader(const uint8_t* p, std::size_t n) : p_(p), n_(n) {}

    template <typename T>
    T read() {
        T v;
        need(sizeof(T));
        std::memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::vector<float> read_floats(std::size_t count) {
        need(count * sizeof(float));
        std::vector<float> v(count);
        std::memcpy(v.data(), p_ + pos_, count * sizeof(float));
        pos_ += count * sizeof(float);
        return v;
    }

    void read_bytes(void* dst, std::size_t count) {
        need(count);
        std::memcpy(dst, p_ + pos_, count);
        pos_ += count;
    }

    std::size_t pos() const { return pos_; }

private:
    void need(std::size_t k) const {
        if (k > n_ - pos_) throw ModelIntegrityError("model file truncated");
    }
    const uint8_t* p_;
    std::size_t n_;
    std::size_t pos_ = 0;
};

template <typename T>
void put(std::vector<uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

void put_floats(std::vector<uint8_t>& out, const std::vector<float>& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(float));
}

bool all_finite(const std::vector<float>& v) {
    for (float f : v) {
        if (!std::isfinite(f)) return false;
    }
    return true;
}

// Dot product with 8 independent accumulators.
//
// A single running sum `s += a[i] * b[i]` cannot be vectorised by the compiler:
// SIMD would add the terms in a different order, and float addition is not
// associative, so the result would change. Compilers only do it under
// -ffast-math, which also breaks NaN/Inf handling everywhere. Here each of the
// 8 lanes is its own sum, so the compiler can map them onto two 4-wide NEON
// (or one 8-wide AVX) registers with *exactly* the semantics written, then we
// combine the lanes once at the end. Same trick BLAS libraries use.
inline float dot(const float* a, const float* b, uint32_t n) noexcept {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; ++k) acc[k] += a[i + k] * b[i + k];
    }
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

Mlp::Mlp(std::vector<float> mean, std::vector<float> stddev, std::vector<Layer> layers, float review_threshold,
         float block_threshold)
    : mean_(std::move(mean)),
      std_(std::move(stddev)),
      layers_(std::move(layers)),
      review_threshold_(review_threshold),
      block_threshold_(block_threshold) {
    validate_and_prepare();
    const std::vector<uint8_t> bytes = to_bytes();
    std::memcpy(fingerprint_.data(), bytes.data() + bytes.size() - 32, 32);
}

void Mlp::validate_and_prepare() {
    const std::size_t n = mean_.size();
    if (n == 0 || n > kMaxWidth) throw std::invalid_argument("bad feature count");
    if (std_.size() != n) throw std::invalid_argument("mean/std size mismatch");
    if (layers_.empty() || layers_.size() > kMaxLayers) throw std::invalid_argument("bad layer count");
    if (!all_finite(mean_) || !all_finite(std_)) throw std::invalid_argument("non-finite normalisation stats");
    std::size_t expected_in = n;
    for (const Layer& L : layers_) {
        if (L.in != expected_in) throw std::invalid_argument("layer input width does not chain");
        if (L.out == 0 || L.out > kMaxWidth) throw std::invalid_argument("bad layer width");
        if (L.w.size() != std::size_t(L.in) * L.out || L.b.size() != L.out)
            throw std::invalid_argument("weight shape mismatch");
        // NaN/Inf weights would make every score NaN, which compares false
        // against every threshold: a poisoned model could silently ALLOW all.
        if (!all_finite(L.w) || !all_finite(L.b)) throw std::invalid_argument("non-finite weights");
        expected_in = L.out;
    }
    if (layers_.back().out != 1) throw std::invalid_argument("final layer must have one output");
    if (!(review_threshold_ > 0.0f && review_threshold_ <= block_threshold_ && block_threshold_ < 1.0f))
        throw std::invalid_argument("thresholds must satisfy 0 < review <= block < 1");
    inv_std_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(std_[i] > 0.0f)) throw std::invalid_argument("std-dev must be positive");
        inv_std_[i] = 1.0f / std_[i];
    }
}

std::vector<uint8_t> Mlp::to_bytes() const {
    std::vector<uint8_t> out;
    out.insert(out.end(), kMagic, kMagic + 8);
    put(out, kVersion);
    put(out, static_cast<uint32_t>(mean_.size()));
    put(out, static_cast<uint32_t>(layers_.size()));
    put(out, review_threshold_);
    put(out, block_threshold_);
    for (const Layer& L : layers_) {
        put(out, L.in);
        put(out, L.out);
    }
    put_floats(out, mean_);
    put_floats(out, std_);
    for (const Layer& L : layers_) {
        put_floats(out, L.w);
        put_floats(out, L.b);
    }
    const Digest d = Sha256::hash(out.data(), out.size());
    out.insert(out.end(), d.begin(), d.end());
    return out;
}

Mlp Mlp::from_bytes(const uint8_t* data, std::size_t len) {
    if (len < 8 + 32) throw ModelIntegrityError("model file too small");

    // Verify the checksum BEFORE parsing anything else: don't feed attacker-
    // controlled bytes to the parser unless they are at least self-consistent.
    const Digest actual = Sha256::hash(data, len - 32);
    if (!constant_time_equal(actual.data(), data + len - 32, 32))
        throw ModelIntegrityError("model checksum mismatch: file is corrupted or was modified");

    Reader r(data, len - 32);
    char magic[8];
    r.read_bytes(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) throw ModelIntegrityError("not an Aegis model file");
    if (r.read<uint32_t>() != kVersion) throw ModelIntegrityError("unsupported model version");
    const uint32_t n_features = r.read<uint32_t>();
    const uint32_t n_layers = r.read<uint32_t>();
    if (n_features == 0 || n_features > kMaxWidth || n_layers == 0 || n_layers > kMaxLayers)
        throw ModelIntegrityError("model dimensions out of range");

    Mlp m;
    m.review_threshold_ = r.read<float>();
    m.block_threshold_ = r.read<float>();
    m.layers_.resize(n_layers);
    for (Layer& L : m.layers_) {
        L.in = r.read<uint32_t>();
        L.out = r.read<uint32_t>();
        if (L.in == 0 || L.in > kMaxWidth || L.out == 0 || L.out > kMaxWidth)
            throw ModelIntegrityError("layer width out of range");
    }
    m.mean_ = r.read_floats(n_features);
    m.std_ = r.read_floats(n_features);
    for (Layer& L : m.layers_) {
        L.w = r.read_floats(std::size_t(L.in) * L.out);
        L.b = r.read_floats(L.out);
    }
    if (r.pos() != len - 32) throw ModelIntegrityError("trailing bytes in model file");

    try {
        m.validate_and_prepare();
    } catch (const std::invalid_argument& e) {
        throw ModelIntegrityError(std::string("invalid model: ") + e.what());
    }
    std::memcpy(m.fingerprint_.data(), data + len - 32, 32);
    return m;
}

Mlp Mlp::load(const std::string& path, const std::optional<std::string>& hmac_key) {
    const std::vector<uint8_t> bytes = read_file(path);
    if (hmac_key) {
        std::vector<uint8_t> sig_hex = read_file(path + ".sig");
        while (!sig_hex.empty() && (sig_hex.back() == '\n' || sig_hex.back() == '\r' || sig_hex.back() == ' '))
            sig_hex.pop_back();
        std::vector<uint8_t> expected;
        try {
            expected = from_hex(std::string_view(reinterpret_cast<const char*>(sig_hex.data()), sig_hex.size()));
        } catch (const std::invalid_argument&) {
            throw ModelIntegrityError("model signature file is malformed");
        }
        const Digest actual = hmac_sha256(reinterpret_cast<const uint8_t*>(hmac_key->data()), hmac_key->size(),
                                          bytes.data(), bytes.size());
        if (expected.size() != actual.size() || !constant_time_equal(expected.data(), actual.data(), actual.size()))
            throw ModelIntegrityError("model signature does not verify: refusing to load an untrusted model");
    }
    return from_bytes(bytes.data(), bytes.size());
}

void Mlp::save(const std::string& path, const std::optional<std::string>& hmac_key) const {
    const std::vector<uint8_t> bytes = to_bytes();
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + path);
        f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (hmac_key) {
        const Digest sig = hmac_sha256(reinterpret_cast<const uint8_t*>(hmac_key->data()), hmac_key->size(),
                                       bytes.data(), bytes.size());
        std::ofstream f(path + ".sig", std::ios::trunc);
        f << to_hex(sig) << '\n';
    }
}

float Mlp::forward(const float* x, float* z_out, float* hidden) const noexcept {
    // Deliberately uninitialised: every element read below is written first,
    // and zero-filling 2 KB per call would be measurable on the hot path.
    std::array<float, kMaxWidth> a;
    std::array<float, kMaxWidth> b;
    const std::size_t n = mean_.size();
    for (std::size_t i = 0; i < n; ++i) a[i] = (x[i] - mean_[i]) * inv_std_[i];
    if (z_out) std::memcpy(z_out, a.data(), n * sizeof(float));

    float* cur = a.data();
    float* nxt = b.data();
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        const float* w = L.w.data();
        for (uint32_t o = 0; o < L.out; ++o) {
            nxt[o] = dot(w + std::size_t(o) * L.in, cur, L.in) + L.b[o];
        }
        if (l + 1 < layers_.size()) {
            for (uint32_t o = 0; o < L.out; ++o) nxt[o] = nxt[o] > 0.0f ? nxt[o] : 0.0f;  // ReLU
            if (hidden) std::memcpy(hidden + l * kMaxWidth, nxt, L.out * sizeof(float));
        }
        std::swap(cur, nxt);
    }
    return cur[0];
}

float Mlp::logit(const float* x) const noexcept { return forward(x, nullptr, nullptr); }

float Mlp::predict_proba(const float* x) const noexcept {
    const float z = logit(x);
    return 1.0f / (1.0f + std::exp(-z));
}

float Mlp::attribute(const float* x, float* out) const noexcept {
    std::array<float, kMaxWidth> z{};
    std::array<float, kMaxWidth * kMaxLayers> hidden{};
    const float result = forward(x, z.data(), hidden.data());

    // Backpropagation for a single output: start with d(logit)/d(logit) = 1 and
    // push the gradient back through each layer: multiply by W^T, then zero the
    // entries whose ReLU was inactive (gradient of max(0, u) is 0 when u <= 0).
    std::array<float, kMaxWidth> g{};
    std::array<float, kMaxWidth> g_prev{};
    g[0] = 1.0f;
    for (std::size_t l = layers_.size(); l-- > 0;) {
        const Layer& L = layers_[l];
        std::fill(g_prev.begin(), g_prev.begin() + L.in, 0.0f);
        for (uint32_t o = 0; o < L.out; ++o) {
            const float go = g[o];
            if (go == 0.0f) continue;
            const float* row = L.w.data() + std::size_t(o) * L.in;
            for (uint32_t i = 0; i < L.in; ++i) g_prev[i] += row[i] * go;
        }
        if (l > 0) {
            const float* h = hidden.data() + (l - 1) * kMaxWidth;
            for (uint32_t i = 0; i < L.in; ++i) {
                if (h[i] <= 0.0f) g_prev[i] = 0.0f;
            }
        }
        std::swap(g, g_prev);
    }
    // g now holds d(logit)/d(z) for the standardised input z.
    for (std::size_t i = 0; i < mean_.size(); ++i) out[i] = g[i] * z[i];
    return result;
}

}  // namespace aegis
