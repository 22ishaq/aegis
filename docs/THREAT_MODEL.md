# Threat model

Method: STRIDE per component, with each threat mapped to a control in the code and a test that proves it.
"Residual" lists what is *not* covered, because an honest threat model says where it stops.

## Assets

1. **Decisions:** the correctness of ALLOW/REVIEW/BLOCK.
2. **The audit trail:** evidence of what was decided, when, and by which model.
3. **The model:** its integrity (no poisoning or substitution).
4. **Customer data:** transaction history and memos.
5. **Money:** the LLM API budget.

## Actors

| Actor | Capability |
|---|---|
| External fraudster | Controls transaction fields they submit, including free-text memos |
| Credential thief | Holds a leaked viewer or analyst API key |
| Malicious insider | Write access to files on the server (logs, model directory) |
| Compromised LLM output | Model influenced by injected text, or simply wrong |

## Threats and controls

| # | STRIDE | Threat | Control | Proof |
|---|---|---|---|---|
| T1 | Tampering | Insider edits an audit entry (e.g. BLOCK→ALLOW) | HMAC chain: every entry's MAC covers the previous MAC | `audit_detects_modified_payload` |
| T2 | Tampering | Insider deletes or reorders entries | Sequence numbers + prev-link verification | `audit_detects_deleted_entry`, `audit_detects_reordering` |
| T3 | Tampering | Insider rewrites the whole chain consistently | MACs need the secret key; a plain hash would not be enough | `audit_detects_forged_chain_without_key` |
| T4 | Tampering | Service keeps writing after tampering, burying the evidence | Fail closed: refuses to open a broken chain | `audit_refuses_to_extend_tampered_log`, `test_server_refuses_tampered_audit_log` |
| T5 | Tampering | Swapped or poisoned model file | SHA-256 checksum, then HMAC signature, verified *before* parsing | `tampered_model_is_rejected`, `signed_model_requires_correct_key` |
| T6 | Tampering | Model with NaN weights (NaN fails every `>=` threshold check, so everything is ALLOWed) | Finiteness validation at load | `non_finite_weights_are_rejected` |
| T7 | Tampering | Malformed model triggers out-of-bounds reads | Bounds-checked reader, exact-length check, dimension limits | `tampered_model_is_rejected` (truncation) |
| T8 | Spoofing | Request without credentials | API key on every `/v1` route, WebSocket auth on the first message | `test_authentication_and_roles`, `test_websocket_requires_auth` |
| T9 | Info disclosure | Timing attack to recover a key or MAC byte by byte | `hmac.compare_digest` / `constant_time_equal`, every key compared | code review: `security.py`, `sha256.cpp` |
| T10 | Elevation | Viewer key used to score or investigate | Role hierarchy (viewer < analyst) | `test_authentication_and_roles` |
| T11 | Tampering | **Prompt injection** in a memo steers the LLM's verdict | Normalise → strip invisible characters → unescapable delimiters → detect → deterministic policy override | `test_guard.py`, `test_hijacked_model_is_contained` |
| T12 | Elevation | Hijacked LLM reads other customers' data | Tools are read-only and scoped to one account | `test_hijacked_model_is_contained` (refused call) |
| T13 | Tampering | LLM returns malformed or out-of-policy output | JSON schema at the API, plus `validate_report` and enum/range checks locally | `test_malformed_reports_are_rejected` |
| T14 | DoS | "Denial of wallet": a script or leaked key loops the paid LLM endpoint | Per-key token bucket on `/investigate` (burst 3, then 1 per 10 s); 8-turn cap per investigation | `test_investigation_is_rate_limited` |
| T15 | DoS | Huge request bodies / slow WebSocket consumers | 64 KB body cap; bounded per-subscriber queues that drop events instead of buffering | `test_oversized_body_rejected` |
| T16 | Tampering | XSS through attacker-controlled memo text in the dashboard | `textContent` only, never `innerHTML`; strict CSP (`script-src 'self'`) | code review: `app.js`, `test_security_headers` |
| T17 | Tampering | Clickjacking the analyst console | `X-Frame-Options: DENY`, `frame-ancestors 'none'` | `test_security_headers` |
| T18 | Tampering | Bad input (NaN coordinates, negative amounts, unknown fields) | Strict Pydantic schema + C++ `validate()` | `test_input_validation`, `validate_rejects_bad_input` |
| T19 | Info disclosure | Stack traces in error responses | Generic 500 handler; details only in server logs | code review: `app.py` |
| T20 | Info disclosure | API keys leaked through URLs/logs | WebSocket key sent as a message, never a query string; keys only via env | code review |
| T21 | Tampering | Deserialising an untrusted dataset executes code | `np.load(..., allow_pickle=False)`, fixed-width string arrays | code review: `datagen.py` |
| T22 | Repudiation | "The model made that decision, not us" / "which model?" | Model SHA-256 fingerprint in every audit record | `engine.py` |
| T23 | Info disclosure | Customer free text copied into the audit trail | Memos stored as SHA-256 hashes in audit records | `engine.py` |

## Prompt injection in detail

Defence is layered because no single layer is reliable:

| Layer | Stops | Fails when |
|---|---|---|
| Unicode NFKC + invisible-character stripping | Fullwidth letters, zero-width splitting, bidi tricks | Novel encodings |
| `<untrusted_data>` wrapper with `<`/`>` replaced | Breaking out of the delimiter | The model ignores the convention |
| Pattern detector | Known phrasings ("ignore previous instructions", fake `SYSTEM:`) | Paraphrased or translated attacks |
| System prompt | Most casual attempts | Determined attackers (prompts are not a security boundary) |
| **Least-privilege tools** | Data exfiltration across customers | Holds even if the model is fully hijacked |
| **Deterministic policy** | Dismissing hard-rule, BLOCKed or injection-bearing alerts | Holds even if the model is fully hijacked |

The last two layers don't depend on the model behaving, which is why they are the real security boundary.
The first four lower the rate of attacks that get through; the last two limit what a successful one can do.

## Residual risks (not covered)

- **Key management.** Keys come from environment variables. Production needs a KMS/HSM, rotation, and
  asymmetric signatures (Ed25519) so verifiers cannot forge.
- **Truncation of the audit log.** Deleting the *newest* entries leaves a valid shorter chain. The mitigation is
  to anchor the head MAC periodically somewhere the insider cannot write (a WORM bucket, a transparency log, a
  separate system) and check later that the chain still extends that anchor.
- **Transport security.** The server binds to loopback and has no TLS. Production terminates TLS at a proxy.
- **Adversarial ML.** A patient fraudster can probe the model to learn its boundary (staying below the velocity
  thresholds, for example). Mitigations: rate-limit score feedback, rotate models, and monitor for "just under
  the threshold" behaviour.
- **Hand-rolled crypto.** SHA-256 and HMAC here pass the official vectors and cross-check against OpenSSL, but
  they are not hardened against every side channel. Production should use a vetted library; this implementation
  exists for learning and for zero dependencies.
- **Single process.** No HA. A crash loses the in-memory alert queue (the audit chain on disk survives).
