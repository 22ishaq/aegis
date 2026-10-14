// Tamper-evident, append-only audit log (a keyed hash chain).
//
// Financial regulators require that decisions be reconstructable and that the
// record of them cannot be quietly rewritten. Each entry stores the MAC of the
// previous entry, and its own MAC covers that link:
//
//   mac_i = HMAC(key, seq_i || ts_i || mac_{i-1} || payload_i)
//
// Changing, deleting, inserting, or reordering any entry breaks every link
// after it, and the attacker cannot repair the chain without the key. This is
// the same idea that underlies Git commits, Certificate Transparency and
// blockchains, minus the consensus.
//
// Known limit: someone who can delete the *newest* entries (truncation) leaves
// a shorter chain that still verifies. The defence is to periodically publish
// ("anchor") the head MAC somewhere the attacker cannot reach, then check
// later that the chain still extends that anchor. See docs/THREAT_MODEL.md.
//
// On-disk format: one line per entry, tab-separated:
//   seq \t timestamp_ms \t prev_mac_hex \t mac_hex \t payload
#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "aegis/sha256.hpp"

namespace aegis {

struct AuditVerifyResult {
    bool ok = true;
    uint64_t entries = 0;        // number of entries checked
    int64_t first_bad_seq = -1;  // line index of the first broken entry
    std::string reason;
    std::string head_hex;        // MAC of the last valid entry
};

class AuditLog {
public:
    // Opens (or creates) the log at `path`; an empty path keeps it in memory.
    // Fails closed: if an existing file does not verify, the constructor throws
    // rather than extend a chain that has already been tampered with.
    AuditLog(std::string path, std::string key, bool fsync_each_append = true);
    ~AuditLog();

    AuditLog(const AuditLog&) = delete;
    AuditLog& operator=(const AuditLog&) = delete;

    // Appends a record and returns its MAC as hex. Thread-safe.
    // The payload must be a single line (canonical JSON from the caller).
    std::string append(std::string_view payload, int64_t timestamp_ms);

    // Re-reads the whole log (from disk if file-backed) and checks every link.
    AuditVerifyResult verify() const;
    static AuditVerifyResult verify_file(const std::string& path, std::string_view key);
    static AuditVerifyResult verify_lines(const std::vector<std::string>& lines, std::string_view key);

    std::string head_hex() const;
    uint64_t size() const;

    static Digest record_mac(std::string_view key, uint64_t seq, int64_t ts, const Digest& prev,
                             std::string_view payload) noexcept;

private:
    mutable std::mutex mu_;
    std::string path_;
    std::string key_;
    bool fsync_each_;
    std::FILE* file_ = nullptr;
    std::vector<std::string> memory_lines_;  // used when path_ is empty
    uint64_t next_seq_ = 0;
    Digest head_{};
};

}  // namespace aegis
