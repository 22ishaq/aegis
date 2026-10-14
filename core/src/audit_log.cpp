#include "aegis/audit_log.hpp"

#include <unistd.h>  // fsync

#include <charconv>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace aegis {

namespace {

constexpr std::size_t kMaxPayload = 64 * 1024;

std::vector<std::string> split_tabs(std::string_view line, std::size_t max_fields) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (out.size() + 1 < max_fields) {
        const std::size_t tab = line.find('\t', start);
        if (tab == std::string_view::npos) break;
        out.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    out.emplace_back(line.substr(start));  // payload is everything after the 4th tab
    return out;
}

template <typename T>
bool parse_int(const std::string& s, T& out) {
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc() && ptr == s.data() + s.size();
}

std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    if (!f) return lines;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

}  // namespace

Digest AuditLog::record_mac(std::string_view key, uint64_t seq, int64_t ts, const Digest& prev,
                            std::string_view payload) noexcept {
    // HMAC over a length-unambiguous encoding: fixed-width integers, fixed-width
    // prev MAC, then the payload. Variable-width fields first could let two
    // different records serialise to the same bytes.
    std::string msg;
    msg.reserve(8 + 8 + 32 + payload.size());
    for (int i = 0; i < 8; ++i) msg.push_back(static_cast<char>(seq >> (8 * i)));
    for (int i = 0; i < 8; ++i) msg.push_back(static_cast<char>(static_cast<uint64_t>(ts) >> (8 * i)));
    msg.append(reinterpret_cast<const char*>(prev.data()), prev.size());
    msg.append(payload);
    return hmac_sha256(key, msg);
}

AuditLog::AuditLog(std::string path, std::string key, bool fsync_each_append)
    : path_(std::move(path)), key_(std::move(key)), fsync_each_(fsync_each_append) {
    if (key_.size() < 16) throw std::invalid_argument("audit key must be at least 16 bytes");
    if (path_.empty()) return;

    const AuditVerifyResult existing = verify_file(path_, key_);
    if (!existing.ok) {
        throw std::runtime_error("audit log " + path_ + " failed verification at entry " +
                                 std::to_string(existing.first_bad_seq) + ": " + existing.reason +
                                 " - refusing to append to a tampered chain");
    }
    next_seq_ = existing.entries;
    if (existing.entries > 0) {
        const std::vector<uint8_t> h = from_hex(existing.head_hex);
        std::memcpy(head_.data(), h.data(), head_.size());
    }
    file_ = std::fopen(path_.c_str(), "a");
    if (!file_) throw std::runtime_error("cannot open audit log " + path_);
}

AuditLog::~AuditLog() {
    if (file_) std::fclose(file_);
}

std::string AuditLog::append(std::string_view payload, int64_t timestamp_ms) {
    if (payload.size() > kMaxPayload) throw std::invalid_argument("audit payload too large");
    if (payload.find_first_of("\n\r\t") != std::string_view::npos)
        throw std::invalid_argument("audit payload must be a single line without tabs");

    std::lock_guard<std::mutex> lock(mu_);
    const Digest mac = record_mac(key_, next_seq_, timestamp_ms, head_, payload);
    std::string line = std::to_string(next_seq_) + '\t' + std::to_string(timestamp_ms) + '\t' + to_hex(head_) +
                       '\t' + to_hex(mac) + '\t' + std::string(payload);

    if (file_) {
        line.push_back('\n');
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() || std::fflush(file_) != 0)
            throw std::runtime_error("audit log write failed");
        // fsync asks the OS to push the bytes to the storage device, so an
        // acknowledged audit record survives a power cut, not just a crash.
        if (fsync_each_ && ::fsync(::fileno(file_)) != 0) throw std::runtime_error("audit log fsync failed");
    } else {
        memory_lines_.push_back(std::move(line));
    }
    head_ = mac;
    ++next_seq_;
    return to_hex(mac);
}

AuditVerifyResult AuditLog::verify_lines(const std::vector<std::string>& lines, std::string_view key) {
    AuditVerifyResult r;
    Digest prev{};  // genesis link is all zeros
    for (std::size_t i = 0; i < lines.size(); ++i) {
        auto fail = [&](std::string why) {
            r.ok = false;
            r.first_bad_seq = static_cast<int64_t>(i);
            r.reason = std::move(why);
            r.entries = i;
            r.head_hex = to_hex(prev);
            return r;
        };
        const std::vector<std::string> f = split_tabs(lines[i], 5);
        if (f.size() != 5) return fail("malformed line");
        uint64_t seq = 0;
        int64_t ts = 0;
        if (!parse_int(f[0], seq) || !parse_int(f[1], ts)) return fail("malformed sequence or timestamp");
        if (seq != i) return fail("sequence gap or reordering (expected " + std::to_string(i) + ")");
        if (f[2] != to_hex(prev)) return fail("prev link does not match previous entry");
        std::vector<uint8_t> mac;
        try {
            mac = from_hex(f[3]);
        } catch (const std::invalid_argument&) {
            return fail("malformed mac");
        }
        const Digest expected = record_mac(key, seq, ts, prev, f[4]);
        if (mac.size() != expected.size() || !constant_time_equal(mac.data(), expected.data(), expected.size()))
            return fail("mac mismatch: entry content was modified");
        std::memcpy(prev.data(), mac.data(), prev.size());
    }
    r.entries = lines.size();
    r.head_hex = to_hex(prev);
    return r;
}

AuditVerifyResult AuditLog::verify_file(const std::string& path, std::string_view key) {
    return verify_lines(read_lines(path), key);
}

AuditVerifyResult AuditLog::verify() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (path_.empty()) return verify_lines(memory_lines_, key_);
    return verify_file(path_, key_);
}

std::string AuditLog::head_hex() const {
    std::lock_guard<std::mutex> lock(mu_);
    return to_hex(head_);
}

uint64_t AuditLog::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return next_seq_;
}

}  // namespace aegis
