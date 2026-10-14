#include "aegis/sha256.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace aegis {

namespace {

// First 32 bits of the fractional parts of the cube roots of the first 64
// primes. "Nothing-up-my-sleeve" numbers: anyone can re-derive them, so the
// designers could not have hidden a backdoor in the choice of constants.
constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr uint32_t rotr(uint32_t x, int n) noexcept { return (x >> n) | (x << (32 - n)); }

}  // namespace

void Sha256::reset() noexcept {
    // Initial hash value: fractional parts of the square roots of the first 8 primes.
    h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    buf_len_ = 0;
    total_len_ = 0;
}

void Sha256::compress(const uint8_t* block) noexcept {
    // 1. Message schedule: expand the 16 input words into 64.
    std::array<uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
               (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);  // big-endian
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    // 2. 64 rounds mixing the schedule into eight working variables.
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);            // "choose": e picks bits from f or g
        const uint32_t t1 = h + S1 + ch + kK[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);  // "majority" vote of a, b, c
        const uint32_t t2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // 3. Davies-Meyer feed-forward: add the result back into the running state.
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::update(const uint8_t* data, std::size_t len) noexcept {
    total_len_ += len;
    if (buf_len_ > 0) {
        const std::size_t take = std::min(len, 64 - buf_len_);
        std::memcpy(buf_.data() + buf_len_, data, take);
        buf_len_ += take;
        data += take;
        len -= take;
        if (buf_len_ < 64) return;
        compress(buf_.data());
        buf_len_ = 0;
    }
    while (len >= 64) {
        compress(data);
        data += 64;
        len -= 64;
    }
    if (len > 0) {
        std::memcpy(buf_.data(), data, len);
        buf_len_ = len;
    }
}

Digest Sha256::finalize() noexcept {
    // Padding: a single 1 bit, zeros, then the message length in bits as a
    // 64-bit big-endian integer, so the total is a multiple of 512 bits.
    // Encoding the length prevents two different messages from padding to
    // the same bytes.
    const uint64_t bit_len = total_len_ * 8;
    const uint8_t one = 0x80;
    update(&one, 1);
    const uint8_t zero = 0;
    while (buf_len_ != 56) update(&zero, 1);
    std::array<uint8_t, 8> len_be{};
    for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    update(len_be.data(), 8);

    Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<uint8_t>(h_[i] >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(h_[i] >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(h_[i] >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(h_[i]);
    }
    reset();
    return out;
}

Digest hmac_sha256(const uint8_t* key, std::size_t key_len, const uint8_t* msg, std::size_t msg_len) noexcept {
    // HMAC(K, m) = H((K' ^ opad) || H((K' ^ ipad) || m)), where K' is the key
    // hashed if longer than one block, then zero-padded to the 64-byte block.
    // The nested structure defeats length-extension attacks that break the
    // naive construction H(key || message).
    std::array<uint8_t, 64> k{};
    if (key_len > 64) {
        const Digest kd = Sha256::hash(key, key_len);
        std::memcpy(k.data(), kd.data(), kd.size());
    } else if (key_len > 0) {
        std::memcpy(k.data(), key, key_len);
    }

    std::array<uint8_t, 64> ipad{}, opad{};
    for (int i = 0; i < 64; ++i) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    Sha256 inner;
    inner.update(ipad.data(), ipad.size());
    inner.update(msg, msg_len);
    const Digest inner_digest = inner.finalize();

    Sha256 outer;
    outer.update(opad.data(), opad.size());
    outer.update(inner_digest.data(), inner_digest.size());
    return outer.finalize();
}

bool constant_time_equal(const uint8_t* a, const uint8_t* b, std::size_t len) noexcept {
    // OR together the XOR of every byte pair; never branch on secret data.
    volatile uint8_t diff = 0;
    for (std::size_t i = 0; i < len; ++i) diff = diff | static_cast<uint8_t>(a[i] ^ b[i]);
    return diff == 0;
}

std::string to_hex(const uint8_t* data, std::size_t len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(len * 2, '\0');
    for (std::size_t i = 0; i < len; ++i) {
        out[2 * i] = kHex[data[i] >> 4];
        out[2 * i + 1] = kHex[data[i] & 0x0f];
    }
    return out;
}

std::vector<uint8_t> from_hex(std::string_view hex) {
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    if (hex.size() % 2 != 0) throw std::invalid_argument("hex string has odd length");
    std::vector<uint8_t> out(hex.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const int hi = nibble(hex[2 * i]);
        const int lo = nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) throw std::invalid_argument("invalid hex character");
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

}  // namespace aegis
