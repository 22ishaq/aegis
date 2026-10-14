// SHA-256 against NIST FIPS 180-4 examples; HMAC-SHA256 against RFC 4231.
#include <algorithm>
#include <string>

#include "aegis/sha256.hpp"
#include "check.hpp"

using aegis::hmac_sha256;
using aegis::Sha256;
using aegis::to_hex;

TEST(sha256_nist_vectors) {
    CHECK(to_hex(Sha256::hash("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(to_hex(Sha256::hash("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(to_hex(Sha256::hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(to_hex(Sha256::hash("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopq"
                              "rlmnopqrsmnopqrstnopqrstu")) ==
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
}

TEST(sha256_million_a_streamed_in_odd_chunks) {
    // Feeding data in awkward chunk sizes exercises the internal buffering.
    Sha256 h;
    const std::string chunk(997, 'a');
    std::size_t left = 1'000'000;
    while (left > 0) {
        const std::size_t n = std::min(left, chunk.size());
        h.update(std::string_view(chunk).substr(0, n));
        left -= n;
    }
    CHECK(to_hex(h.finalize()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(hmac_sha256_rfc4231) {
    // Case 1
    CHECK(to_hex(hmac_sha256(std::string(20, '\x0b'), "Hi There")) ==
          "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    // Case 2: key shorter than the block
    CHECK(to_hex(hmac_sha256("Jefe", "what do ya want for nothing?")) ==
          "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    // Case 3
    CHECK(to_hex(hmac_sha256(std::string(20, '\xaa'), std::string(50, '\xdd'))) ==
          "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
    // Case 4
    std::string key4;
    for (int i = 1; i <= 25; ++i) key4.push_back(static_cast<char>(i));
    CHECK(to_hex(hmac_sha256(key4, std::string(50, '\xcd'))) ==
          "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
    // Case 6: key longer than the block must be hashed first
    CHECK(to_hex(hmac_sha256(std::string(131, '\xaa'), "Test Using Larger Than Block-Size Key - Hash Key First")) ==
          "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    // Case 7
    CHECK(to_hex(hmac_sha256(std::string(131, '\xaa'),
                             "This is a test using a larger than block-size key and a larger than block-size data. "
                             "The key needs to be hashed before being used by the HMAC algorithm.")) ==
          "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
}

TEST(hex_roundtrip_and_rejection) {
    const auto d = Sha256::hash("aegis");
    const auto back = aegis::from_hex(to_hex(d));
    CHECK(back.size() == 32 && std::equal(back.begin(), back.end(), d.begin()));
    CHECK_THROWS(aegis::from_hex("abc"));
    CHECK_THROWS(aegis::from_hex("zz"));
}

TEST(constant_time_equal_basic) {
    const uint8_t a[4] = {1, 2, 3, 4};
    const uint8_t b[4] = {1, 2, 3, 5};
    CHECK(aegis::constant_time_equal(a, a, 4));
    CHECK(!aegis::constant_time_equal(a, b, 4));
}
