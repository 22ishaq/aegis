# Architecture

## The life of one payment

1. **Ingest.** A payment arrives at `POST /v1/transactions/score` (or from the demo stream). Pydantic rejects
   unknown fields, NaN/Inf and out-of-range values; `aegis::validate` checks the C++ invariants again at the
   binding boundary.
2. **Features** (`core/src/features.cpp`). The engine looks up the account's `AccountState` (320 bytes) in an
   open-addressing hash map. It brings the decayed counters forward to "now", computes 20 features from the state
   as it was *before* this payment, then folds the payment into the state.
3. **Model** (`core/src/mlp.cpp`). The features are standardised and run through a 20→32→16→1 ReLU network
   loaded from a signed file, giving a fraud probability.
4. **Rules** (`core/src/scorer.cpp`). Six typology rules are evaluated on raw signals in business units
   (km/h, dollars, counts).
5. **Decision.** `risk = max(model, highest severity among fired rules)`. BLOCK if risk ≥ block threshold, REVIEW
   if risk ≥ review threshold *or* a hard rule fired, else ALLOW. Both thresholds are stored in the model file.
6. **Explain** (flagged payments only). A backward pass computes gradient × input per feature; the top
   positive contributors become reason codes.
7. **Record.** Flagged payments become alerts. A canonical-JSON summary is appended to the HMAC hash chain; it
   includes the model's SHA-256 fingerprint, so every decision traces back to the exact model that made it.
8. **Investigate** (on demand). The injection guard screens the memo, the agent gathers evidence through
   case-scoped tools, the report is schema-validated, deterministic policy has the final word, and the report's
   hash is appended to the chain.

Steps 2 to 5 take about 290 ns on one core.

## Components

| Component | Language | File(s) | Responsibility |
|---|---|---|---|
| Types & validation | C++ | `types.hpp` | `Transaction` (72 bytes, trivially copyable), feature schema |
| Ring buffer | C++ | `spsc_ring.hpp` | Lock-free hand-off between the ingest and scoring threads |
| Hash map | C++ | `flat_map.hpp` | Per-account state, open addressing + linear probing |
| Streaming stats | C++ | `stats.hpp` | Welford, exponential decay, haversine |
| Feature engine | C++ | `features.*` | Point-in-time features, O(1) per event |
| MLP | C++ | `mlp.*` | Signed model format, inference, attribution |
| Scorer | C++ | `scorer.*` | Rules + model → decision |
| Crypto | C++ | `sha256.*` | SHA-256, HMAC-SHA256, constant-time compare |
| Audit log | C++ | `audit_log.*` | Keyed hash chain, fail-closed verification |
| Pipeline | C++ | `pipeline.*` | Two-thread streaming harness with latency histograms |
| Bindings | C++ | `bindings/aegis_py.cpp` | NumPy batch APIs that release the GIL |
| Data simulator | Python | `datagen.py` | 5 fraud typologies, hard negatives, injection memos |
| Model training | Python | `model.py`, `train.py` | NumPy MLP, Adam, time-split evaluation, signing |
| Engine facade | Python | `engine.py` | Thread-safe scoring, explanations, case store, audit records |
| Investigator | Python | `agent/` | Guard, tools, Claude loop, offline fallback, policy |
| Service | Python | `service/` | FastAPI, roles, rate limits, headers, dashboard |

## Concurrency model

The C++ pipeline uses **one writer per piece of state**:

- The ingest thread owns the ring buffer's `head`; the scoring thread owns its `tail`. Each publishes with a
  release store and observes the other with an acquire load, so neither needs a lock.
- The scoring thread exclusively owns the `FeatureEngine` and every `AccountState`. No locks are needed on the
  hot path.
- To scale out, run N pipelines and route each payment to `hash(account_id) % N`. Each account's state lives in
  exactly one place, so pipelines never coordinate. This is the same partitioning Kafka uses.

The Python service calls the scorer under a single `threading.Lock`. It is fast enough (≈1 µs including the
Python→C++ call) that the lock is never the bottleneck at demo rates. The FastAPI endpoints for slow work (LLM
investigations, fsync on audit writes) run off the event loop through `asyncio.to_thread`.

## Memory

- `AccountState` is 320 bytes. Each map slot adds an 8-byte key, and the map keeps load below 0.7, so the cost is
  about 470 bytes per account: ~470 MB per million accounts, held in RAM.
- No per-transaction history is stored by the engine. That is the point of the decayed counters.
- The Python case store is bounded (5,000 alerts, 50 history rows per account), so it cannot grow without limit.

## File formats

**Model (`models/aegis.model`)**, little-endian:

```
"AEGISMDL" | u32 version | u32 n_features | u32 n_layers | f32 review_thr | f32 block_thr
| (u32 in, u32 out) × n_layers | f32 mean[n] | f32 std[n] | per layer: f32 W[out·in], f32 b[out]
| u8 sha256[32]  over everything above
```

`models/aegis.model.sig` holds a hex HMAC-SHA256 of the whole file. The loader checks the checksum first, then
parses with bounds checks, then validates shapes, finiteness and threshold ordering.

**Audit log (`data/audit.log`)**, one entry per line:

```
seq \t timestamp_ms \t prev_mac_hex \t mac_hex \t canonical_json_payload
mac = HMAC(key, le64(seq) || le64(ts) || prev_mac || payload)
```

## API

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/healthz` | none | Liveness |
| GET | `/v1/meta` | viewer | Model fingerprint, thresholds, features, rules |
| POST | `/v1/transactions/score` | analyst | Score one payment |
| GET | `/v1/alerts` | viewer | Alert queue |
| GET | `/v1/alerts/{id}` | viewer | Full case: transaction, explanation, frozen history |
| POST | `/v1/alerts/{id}/investigate` | analyst | Run the investigator (rate limited: 3 burst, 1 per 10 s) |
| GET | `/v1/audit/verify` | viewer | Re-verify the entire chain |
| GET | `/v1/stats` | viewer | Counters and scoring latency percentiles |
| WS | `/v1/stream` | viewer | Live scored events; authenticate with the first message |

OpenAPI docs are served at `/docs`.

## Design decisions

**C++ core, Python around it.** The per-event path has to be fast and predictable, with no garbage collector
and no interpreter. Everything around it (training, the API, the LLM) changes often and benefits from Python's
ecosystem. pybind11 joins them, and the *same* feature code runs in both training and serving.

**`risk = max(model, rules)`.** A rule can raise a score but never hide what the model saw. Rules give
regulators line-by-line auditability and cover known typologies from day one. The model catches combinations no
one wrote a rule for. The cost is a few more false positives, which the training report shows honestly
(hybrid PR-AUC 0.977 vs model-only 0.980).

**Decayed counters instead of sliding windows.** An exact 1-hour sliding window needs every timestamp in the
window; a decayed counter needs one double. The decayed version also has no cliff edge: an event 61 minutes ago
still counts a little.

**A hand-written agent loop.** Every step of the loop is visible and testable with a fake client, and it is the
natural place to enforce scope, turn limits and validation. The SDK's tool runner would also work.

**Deterministic policy after the LLM.** An LLM can be persuaded; an `if` statement cannot. The LLM adds value in
summarising evidence and drafting the SAR narrative, while the irreversible calls stay with code and humans.

**Fail closed.** A model with a bad signature is not loaded, and a broken audit chain stops the service from
starting. For a security control, "keep running in a degraded way" is the wrong default.
