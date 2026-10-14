"""Demo: an insider edits the audit log to hide a BLOCK decision.

    make demo-tamper
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from aegis import _core  # noqa: E402

KEY = "demo-audit-key-0123456789"


def main() -> None:
    path = str(Path(tempfile.mkdtemp()) / "audit.log")
    log = _core.AuditLog(path, KEY, False)
    for i in range(8):
        decision = "BLOCK" if i == 5 else "REVIEW"
        log.append(f'{{"alert":"ALR-{i}","decision":"{decision}","account":4471}}', 1_790_000_000_000 + i)
    print(f"1. Wrote 8 decisions. Chain verifies: {_core.AuditLog.verify_file(path, KEY)['ok']}")
    print(f"   head MAC (publish this somewhere the insider can't reach): {log.head[:24]}...")

    lines = open(path).read().splitlines()
    print("\n2. Insider rewrites entry 5 from BLOCK to ALLOW:")
    print("   before:", lines[5].split("\t")[-1])
    lines[5] = lines[5].replace("BLOCK", "ALLOW")
    print("   after: ", lines[5].split("\t")[-1])
    open(path, "w").write("\n".join(lines) + "\n")

    r = _core.AuditLog.verify_file(path, KEY)
    print(f"\n3. Verification: ok={r['ok']}  first_bad_seq={r['first_bad_seq']}  reason={r['reason']!r}")

    print("\n4. Recomputing a plain SHA-256 would not help the insider: entries are HMAC'd with a key they")
    print("   don't hold. And the service fails closed - it refuses to append to a broken chain:")
    try:
        _core.AuditLog(path, KEY, False)
    except RuntimeError as e:
        print(f"   RuntimeError: {e}")


if __name__ == "__main__":
    main()
