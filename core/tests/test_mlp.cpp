#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>

#include "aegis/mlp.hpp"
#include "check.hpp"
#include "helpers.hpp"

using aegis::Mlp;
using aegis::ModelIntegrityError;

namespace {

Mlp linear_model() {
    // logit = 0.5 + 2*z0 - 1*z1 + 0.25*z2, with z = (x - mean) / std
    Mlp::Layer L{3, 1, {2.0f, -1.0f, 0.25f}, {0.5f}};
    return Mlp({1.0f, 0.0f, -1.0f}, {2.0f, 1.0f, 0.5f}, {L}, 0.5f, 0.9f);
}

}  // namespace

TEST(linear_model_forward_is_exact) {
    const Mlp m = linear_model();
    const float x[3] = {3.0f, 1.0f, 0.0f};  // z = {1, 1, 2}
    const float expected = 0.5f + 2.0f - 1.0f + 0.5f;
    CHECK_NEAR(m.logit(x), expected, 1e-6);
    CHECK_NEAR(m.predict_proba(x), 1.0 / (1.0 + std::exp(-expected)), 1e-6);
}

TEST(linear_attribution_decomposes_logit_exactly) {
    const Mlp m = linear_model();
    const float x[3] = {3.0f, 1.0f, 0.0f};
    float a[3];
    const float logit = m.attribute(x, a);
    CHECK_NEAR(a[0], 2.0, 1e-6);
    CHECK_NEAR(a[1], -1.0, 1e-6);
    CHECK_NEAR(a[2], 0.5, 1e-6);
    CHECK_NEAR(0.5f + a[0] + a[1] + a[2], logit, 1e-6);  // bias + contributions = logit
}

TEST(relu_attribution_matches_finite_differences) {
    // attribution_i = (d logit / d z_i) * z_i  with  z_i = (x_i - mean_i) / std_i.
    // By the chain rule that equals (d logit / d x_i) * (x_i - mean_i), and the
    // raw-space gradient can be measured numerically with central differences.
    // This checks the hand-written backward pass against calculus.
    const Mlp m = testing::make_mlp({6, 16, 8, 1}, 11);
    const float x[6] = {0.3f, -1.2f, 2.0f, 0.7f, -0.4f, 1.1f};
    float a[6];
    m.attribute(x, a);
    const float eps = 1e-3f;
    for (int i = 0; i < 6; ++i) {
        float xp[6], xm[6];
        std::copy(x, x + 6, xp);
        std::copy(x, x + 6, xm);
        xp[i] += eps;
        xm[i] -= eps;
        const double grad_x = (static_cast<double>(m.logit(xp)) - m.logit(xm)) / (2.0 * eps);
        const double expected = grad_x * (x[i] - m.mean()[i]);
        CHECK_NEAR(a[i], expected, 1e-2 * (1.0 + std::fabs(expected)));
    }
}

TEST(serialisation_roundtrip_preserves_predictions_and_fingerprint) {
    const Mlp m = testing::make_mlp({20, 32, 16, 1}, 5, 0.3f, 0.8f);
    const std::vector<uint8_t> bytes = m.to_bytes();
    const Mlp back = Mlp::from_bytes(bytes.data(), bytes.size());
    CHECK(back.fingerprint() == m.fingerprint());
    CHECK_NEAR(back.review_threshold(), 0.3, 1e-7);
    CHECK_NEAR(back.block_threshold(), 0.8, 1e-7);
    std::mt19937_64 rng(9);
    std::normal_distribution<float> d(0.0f, 2.0f);
    float worst = 0.0f;
    for (int k = 0; k < 200; ++k) {
        float x[20];
        for (float& v : x) v = d(rng);
        worst = std::max(worst, std::fabs(m.logit(x) - back.logit(x)));
    }
    CHECK(worst == 0.0f);  // bit-identical weights -> bit-identical outputs
}

TEST(tampered_model_is_rejected) {
    const Mlp m = testing::make_mlp({20, 8, 1});
    std::vector<uint8_t> bytes = m.to_bytes();
    // Flip one bit in the weights: the checksum must catch it.
    std::vector<uint8_t> flipped = bytes;
    flipped[bytes.size() / 2] ^= 0x01;
    CHECK_THROWS(Mlp::from_bytes(flipped.data(), flipped.size()));
    // Truncation.
    CHECK_THROWS(Mlp::from_bytes(bytes.data(), bytes.size() - 1));
    CHECK_THROWS(Mlp::from_bytes(bytes.data(), 10));
}

TEST(non_finite_weights_are_rejected) {
    Mlp::Layer L{2, 1, {1.0f, std::numeric_limits<float>::quiet_NaN()}, {0.0f}};
    CHECK_THROWS(Mlp({0.0f, 0.0f}, {1.0f, 1.0f}, {L}, 0.5f, 0.9f));
    Mlp::Layer ok{2, 1, {1.0f, 1.0f}, {0.0f}};
    CHECK_THROWS(Mlp({0.0f, 0.0f}, {1.0f, 0.0f}, {ok}, 0.5f, 0.9f));  // zero std
    CHECK_THROWS(Mlp({0.0f, 0.0f}, {1.0f, 1.0f}, {ok}, 0.9f, 0.5f));  // review > block
}

TEST(signed_model_requires_correct_key) {
    const std::string path = testing::temp_path("model.aegis");
    const Mlp m = testing::make_mlp({20, 8, 1});
    m.save(path, std::string("correct-horse-battery-staple"));
    const Mlp ok = Mlp::load(path, std::string("correct-horse-battery-staple"));
    CHECK(ok.fingerprint() == m.fingerprint());
    CHECK_THROWS(Mlp::load(path, std::string("wrong-key-wrong-key")));
    // Unsigned load still verifies the embedded checksum.
    CHECK(Mlp::load(path).fingerprint() == m.fingerprint());
    // Corrupt the signature file.
    {
        std::ofstream sig(path + ".sig", std::ios::trunc);
        sig << std::string(64, '0') << '\n';
    }
    CHECK_THROWS(Mlp::load(path, std::string("correct-horse-battery-staple")));
}
