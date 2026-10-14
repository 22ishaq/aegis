#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "aegis/audit_log.hpp"
#include "check.hpp"
#include "helpers.hpp"

using aegis::AuditLog;

namespace {

const std::string kKey = "test-audit-key-0123456789";

std::vector<std::string> read_all(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l)) lines.push_back(l);
    return lines;
}

void write_all(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream f(path, std::ios::trunc);
    for (const auto& l : lines) f << l << '\n';
}

std::string make_log(const std::string& name, int n) {
    const std::string path = testing::temp_path(name);
    AuditLog log(path, kKey, /*fsync_each_append=*/false);
    for (int i = 0; i < n; ++i) {
        log.append("{\"alert\":" + std::to_string(i) + ",\"decision\":\"REVIEW\"}", 1'700'000'000'000 + i);
    }
    return path;
}

}  // namespace

TEST(audit_in_memory_chain_verifies) {
    AuditLog log("", kKey);
    std::string last;
    for (int i = 0; i < 100; ++i) last = log.append("{\"i\":" + std::to_string(i) + "}", i);
    const auto r = log.verify();
    CHECK(r.ok);
    CHECK(r.entries == 100);
    CHECK(r.head_hex == last);
    CHECK(log.head_hex() == last);
}

TEST(audit_file_chain_verifies_and_resumes) {
    const std::string path = make_log("resume.log", 20);
    {
        AuditLog reopened(path, kKey, false);  // must pick up seq 20 and the head MAC
        CHECK(reopened.size() == 20);
        reopened.append("{\"after\":\"restart\"}", 42);
    }
    const auto r = AuditLog::verify_file(path, kKey);
    CHECK(r.ok);
    CHECK(r.entries == 21);
}

TEST(audit_detects_modified_payload) {
    const std::string path = make_log("modified.log", 50);
    auto lines = read_all(path);
    const auto pos = lines[20].find("REVIEW");
    lines[20].replace(pos, 6, "ALLOW!");  // an insider "downgrades" a decision
    write_all(path, lines);
    const auto r = AuditLog::verify_file(path, kKey);
    CHECK(!r.ok);
    CHECK(r.first_bad_seq == 20);
}

TEST(audit_detects_deleted_entry) {
    const std::string path = make_log("deleted.log", 50);
    auto lines = read_all(path);
    lines.erase(lines.begin() + 10);
    write_all(path, lines);
    const auto r = AuditLog::verify_file(path, kKey);
    CHECK(!r.ok);
    CHECK(r.first_bad_seq == 10);
}

TEST(audit_detects_reordering) {
    const std::string path = make_log("reordered.log", 50);
    auto lines = read_all(path);
    std::swap(lines[30], lines[31]);
    write_all(path, lines);
    const auto r = AuditLog::verify_file(path, kKey);
    CHECK(!r.ok);
    CHECK(r.first_bad_seq == 30);
}

TEST(audit_detects_forged_chain_without_key) {
    // An attacker rewrites the whole log consistently, but with their own key.
    const std::string path = testing::temp_path("forged.log");
    {
        AuditLog forged(path, "attacker-key-0123456789", false);
        forged.append("{\"alert\":0,\"decision\":\"ALLOW\"}", 1);
    }
    const auto r = AuditLog::verify_file(path, kKey);
    CHECK(!r.ok);
    CHECK(r.first_bad_seq == 0);
}

TEST(audit_refuses_to_extend_tampered_log) {
    const std::string path = make_log("failclosed.log", 10);
    auto lines = read_all(path);
    lines[5].back() = 'X';
    write_all(path, lines);
    CHECK_THROWS(AuditLog(path, kKey, false));
}

TEST(audit_rejects_multiline_payload_and_short_key) {
    AuditLog log("", kKey);
    CHECK_THROWS(log.append("line1\nline2", 0));
    CHECK_THROWS(log.append("a\tb", 0));
    CHECK_THROWS(AuditLog("", "short"));
}
