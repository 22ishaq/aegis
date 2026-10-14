// Multi-layer perceptron inference + the signed model file format.
//
// The network is trained in Python (python/aegis/model.py, NumPy only) and
// exported to a small binary file. This class loads it, refuses it if it has
// been corrupted or tampered with, and runs the forward pass in a few hundred
// nanoseconds with no allocation.
//
// Architecture: x -> standardise -> [Dense -> ReLU] * (L-1) -> Dense -> sigmoid
//
// File layout (little-endian):
//   char[8]  magic "AEGISMDL"
//   u32      version (=1)
//   u32      n_features
//   u32      n_layers
//   f32      review_threshold        operating points chosen on validation data
//   f32      block_threshold
//   u32[2]   (in, out) per layer
//   f32[n]   feature means           standardisation statistics from training
//   f32[n]   feature std-devs
//   per layer: f32[out*in] weights (row-major), f32[out] biases
//   u8[32]   SHA-256 of all preceding bytes   (integrity)
//
// Optionally, `<file>.sig` holds a hex HMAC-SHA256 of the whole file under a
// deployment secret (authenticity). See docs/THREAT_MODEL.md: "model supply chain".
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "aegis/sha256.hpp"

namespace aegis {

class ModelIntegrityError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Mlp {
public:
    static constexpr std::size_t kMaxWidth = 256;
    static constexpr std::size_t kMaxLayers = 8;

    struct Layer {
        uint32_t in = 0;
        uint32_t out = 0;
        std::vector<float> w;  // out x in, row-major: w[o * in + i]
        std::vector<float> b;  // out
    };

    Mlp() = default;
    // Validates shapes and values; throws std::invalid_argument on any problem.
    Mlp(std::vector<float> mean, std::vector<float> stddev, std::vector<Layer> layers, float review_threshold,
        float block_threshold);

    // Parses and verifies the embedded SHA-256. Throws ModelIntegrityError.
    static Mlp from_bytes(const uint8_t* data, std::size_t len);
    std::vector<uint8_t> to_bytes() const;

    // Loads a model file. If `hmac_key` is set, `<path>.sig` must exist and match.
    static Mlp load(const std::string& path, const std::optional<std::string>& hmac_key = std::nullopt);
    // Writes the model, and `<path>.sig` when a key is given.
    void save(const std::string& path, const std::optional<std::string>& hmac_key = std::nullopt) const;

    // x points at n_features() raw (unstandardised) feature values.
    float logit(const float* x) const noexcept;
    float predict_proba(const float* x) const noexcept;

    // Gradient x input attribution: out[i] ~ how much feature i pushed the logit
    // up (+) or down (-) relative to an "average" transaction (the training
    // mean). For a linear model this decomposition is exact; for a ReLU network
    // it is the exact linearisation around the current input. Returns the logit.
    float attribute(const float* x, float* out) const noexcept;

    std::size_t n_features() const noexcept { return mean_.size(); }
    const std::vector<float>& mean() const noexcept { return mean_; }
    const std::vector<float>& stddev() const noexcept { return std_; }
    std::size_t n_layers() const noexcept { return layers_.size(); }
    const std::vector<Layer>& layers() const noexcept { return layers_; }
    float review_threshold() const noexcept { return review_threshold_; }
    float block_threshold() const noexcept { return block_threshold_; }
    // SHA-256 of the serialised model: a stable version id written into every
    // audit record, so each decision is traceable to the exact model that made it.
    const Digest& fingerprint() const noexcept { return fingerprint_; }

private:
    void validate_and_prepare();
    // Runs the network; if `hidden` is non-null, stores each hidden layer's
    // post-ReLU activations (kMaxWidth floats per layer) for backpropagation.
    float forward(const float* x, float* z_out, float* hidden) const noexcept;

    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<float> inv_std_;
    std::vector<Layer> layers_;
    float review_threshold_ = 0.5f;
    float block_threshold_ = 0.9f;
    Digest fingerprint_{};
};

}  // namespace aegis
