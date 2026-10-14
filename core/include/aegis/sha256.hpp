// SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104), implemented from scratch.
//
// Why write our own instead of linking OpenSSL? Two reasons: the core has zero
// third-party dependencies, and implementing it is the best way to understand
// what a hash function actually does. It is verified against the official NIST
// and RFC 4231 test vectors (core/tests) and cross-checked against Python's
// hashlib on random inputs (tests/test_crypto_parity.py).
//
// In production you would still prefer a vetted library (BoringSSL, libsodium)
// with constant-time, hardware-accelerated code paths; see docs/THREAT_MODEL.md.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aegis {

using Digest = std::array<uint8_t, 32>;

class Sha256 {
public:
    Sha256() noexcept { reset(); }

    void reset() noexcept;
    void update(const uint8_t* data, std::size_t len) noexcept;
    void update(std::string_view s) noexcept {
        update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }
    Digest finalize() noexcept;  // also resets the object

    static Digest hash(const uint8_t* data, std::size_t len) noexcept {
        Sha256 h;
        h.update(data, len);
        return h.finalize();
    }
    static Digest hash(std::string_view s) noexcept {
        return hash(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

private:
    void compress(const uint8_t* block) noexcept;

    std::array<uint32_t, 8> h_{};
    std::array<uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    uint64_t total_len_ = 0;
};

// Keyed hash: proves a message was produced by someone holding `key`.
// A plain hash only detects accidental corruption: an attacker who edits the
// data can simply recompute it. They cannot recompute an HMAC without the key.
Digest hmac_sha256(const uint8_t* key, std::size_t key_len, const uint8_t* msg, std::size_t msg_len) noexcept;

inline Digest hmac_sha256(std::string_view key, std::string_view msg) noexcept {
    return hmac_sha256(reinterpret_cast<const uint8_t*>(key.data()), key.size(),
                       reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
}

// Compares in time independent of *where* the first difference is. A normal
// early-exit memcmp leaks, through timing, how many leading bytes of a guessed
// MAC were correct, letting an attacker forge one byte at a time.
bool constant_time_equal(const uint8_t* a, const uint8_t* b, std::size_t len) noexcept;

std::string to_hex(const uint8_t* data, std::size_t len);
inline std::string to_hex(const Digest& d) { return to_hex(d.data(), d.size()); }

// Throws std::invalid_argument on odd length or non-hex characters.
std::vector<uint8_t> from_hex(std::string_view hex);

}  // namespace aegis
