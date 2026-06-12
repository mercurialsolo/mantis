# Computer Plane — Profile Snapshot Pipeline

**Status:** Proposed
**Owner:** TBD
**Tracks issue:** [#699](https://github.com/mercurialsolo/mantis/issues/699) (Phase 2 of [#696](https://github.com/mercurialsolo/mantis/issues/696))
**Gates:** any code under `src/mantis_agent/observability/profile_snapshotter.py`

> This doc is the **explicit gate** issue #699 names before any
> profile-snapshot code can land. Phase 2 host backends (E2B
> Desktop, Daytona) cannot mount the `osworld-data` Modal Volume, so
> the per-profile Chrome user-data-dir has to round-trip through an
> object store. Getting this round-trip wrong corrupts cookies,
> drops sessions, or — worst — silently loads a stale profile while
> the operator thinks it's current. The spec exists so we lock the
> contract before writing the code.

## Summary

A **profile snapshot** is a content-addressed, integrity-verified
archive of one Chrome user-data-dir. The snapshotter takes one when
a CUA run ends, uploads it to a canonical object store, and atomically
flips a per-profile `latest` pointer. The next run for that
`(tenant_id, profile_id)` resolves the pointer, downloads the
snapshot, and untars it into the host's local Chrome path before
Chrome starts.

Today (Phase 1 / 1.5) profile bytes live on `modal.Volume("osworld-data")`
and reuse comes free — every Modal function in the same app sees the
same volume. Phase 2 ships hosts that **don't** mount that volume
(E2B Desktop, Daytona). The snapshot pipeline is the substitute.

This doc covers:

1. The on-disk format and naming.
2. The object-store layout and atomic-pointer protocol.
3. Hot vs cold snapshot modes (and why cold is the default).
4. Chrome SQLite WAL consistency — the dominant correctness risk.
5. Per-profile lock semantics (TTL + renewal + reaper).
6. Cross-region behaviour and stickiness.
7. Failure modes and how each fails-safe.
8. Operations: metrics, alerts, manual override.

## Goals

1. **Drop-in replacement** for the Modal Volume reuse the brain
   already depends on. A run-then-rerun for the same `(tenant_id,
   profile_id)` produces identical Chrome state regardless of which
   host the second run lands on.
2. **No silent loss.** A snapshot upload that can't be verified is
   rejected before the `latest` pointer flips; the previous snapshot
   stays canonical.
3. **No silent corruption.** Chrome's SQLite databases (`Cookies`,
   `Local Storage`, `IndexedDB`, the History stack) round-trip without
   torn reads or write-ahead-log drift. The cold-mode default
   guarantees this; hot mode is an opt-in for callers who explicitly
   accept the risk.
4. **Hostile-tenant safe.** A misbehaving tenant cannot corrupt
   another tenant's `latest` pointer, exhaust shared storage, or
   read another tenant's bytes.
5. **Operator visibility.** A human triaging "why is this run using
   stale data?" can list snapshots, see their sha256, compare to the
   `latest` pointer, and roll back to any prior snapshot via a single
   pointer write.

## Non-goals

- A general-purpose backup product. Snapshots are CUA-tenant-scoped
  artefacts; no SLA on durability beyond the host's object-store
  defaults.
- Multi-region replication. We pin `(tenant_id, profile_id)` to a
  region for the warm window; cross-region failover is a separate
  product decision (see §6).
- Encryption at rest beyond the bucket's default. Customer-side
  encryption (BYOK) is a follow-up.
- Schema migration for old Chrome profile formats. The brain pins a
  Chrome major version per deploy; mismatched snapshots either load
  cleanly or fail loud (we don't attempt a migration).

## 1. On-disk format

### Per-profile directory

A profile snapshot archives **exactly one Chrome user-data-dir**,
identified canonically by:

```
(tenant_id, profile_id, chrome_major_version)
```

`chrome_major_version` is captured because Chrome between major
versions occasionally rewrites SQLite schemas (notably `Cookies`),
and a v126 snapshot loaded into v131 can silently lose rows. The
snapshotter records the version that wrote it; the loader refuses
mismatches.

The archived directory tree is the Chrome user-data-dir as Chrome
writes it. We do not whitelist subpaths — Chrome's own contract is
that the whole directory is the unit of consistency.

### Archive format

```
profile-<sha256-prefix>.tar.zst
```

- **tar (POSIX ustar)** — single-file, streaming-friendly, preserves
  permissions + timestamps. Required because Chrome cares about
  some symlink + permission specifics in `Singleton*` files.
- **zstd** at level 9 (compress) / streaming (decompress). Level 9
  is the empirical sweet spot for Chrome profiles (~2.5–4× smaller
  than uncompressed; ~4× faster compress than gzip-9). Level 3 is
  too lossy; level 19+ is too slow at the ~1 GB sizes typical for a
  CUA profile.
- **No encryption at the archive level.** The bucket policy handles
  at-rest; BYOK is future.

### Manifest

Each snapshot ships with a sibling manifest:

```
profile-<sha256-prefix>.manifest.json
```

```json
{
  "version": 1,
  "schema": "computer-plane.profile-snapshot",
  "tenant_id": "acme",
  "profile_id": "alice",
  "chrome_major_version": 131,
  "archive_sha256": "1a2b3c…",
  "archive_size_bytes": 482919334,
  "uncompressed_size_bytes": 1_412_339_201,
  "captured_at_ms": 1733456512000,
  "captured_by": {
    "host": "modal|e2b|daytona",
    "host_run_id": "fc-01KTVJM035…",
    "writer_version": "snapshotter-0.1.0"
  },
  "mode": "cold",
  "chrome_uptime_seconds_at_capture": 0,
  "predecessor_sha256": "8d9e7f…",
  "notes": ""
}
```

- `version: 1` lets us rev the manifest in place without breaking
  readers.
- `archive_sha256` is the **uncompressed-source** sha256 (the
  `sha256(open(archive_path, "rb"))` value before `latest` flips, NOT
  the inner tree hash). The loader recomputes after download and
  compares; a mismatch refuses the load.
- `predecessor_sha256` is the snapshot this one supersedes. Used by
  the operator-side "history of profile X" view.
- `mode: "cold" | "hot"` records whether Chrome was running at
  capture time. The loader can refuse hot snapshots when the policy
  says so.

### Naming + sha prefix

We use the first 12 chars of the archive sha256 as the on-disk
filename prefix. Collision space is negligible at our scale (<1e6
snapshots per tenant), and short names keep S3 console paging readable.

## 2. Object-store layout

The canonical store is **S3** (or any S3-compatible — R2, Tigris,
MinIO). The layout is bucket-relative:

```
snapshots/<tenant_id>/<profile_id>/<chrome_major>/
    profile-1a2b3c4d5e6f.tar.zst
    profile-1a2b3c4d5e6f.manifest.json
    profile-8d9e7f6a5b4c.tar.zst
    profile-8d9e7f6a5b4c.manifest.json
    ...
    latest.json
```

### `latest.json` — the atomic pointer

```json
{
  "version": 1,
  "active_sha256_prefix": "1a2b3c4d5e6f",
  "active_archive_key": "snapshots/acme/alice/131/profile-1a2b3c4d5e6f.tar.zst",
  "active_manifest_key": "snapshots/acme/alice/131/profile-1a2b3c4d5e6f.manifest.json",
  "flipped_at_ms": 1733456515000,
  "flipped_from_sha256_prefix": "8d9e7f6a5b4c"
}
```

**Atomic pointer flip protocol:**

1. Writer uploads `profile-<new>.tar.zst` and `profile-<new>.manifest.json`
   to the per-profile prefix. Both PUTs must succeed before the
   pointer changes.
2. Writer fetches the current `latest.json` (if any), notes the
   sha256 prefix.
3. Writer constructs a new `latest.json` with `flipped_from_sha256_prefix`
   set to the current pointer's `active_sha256_prefix` (or empty
   string on first snapshot).
4. Writer PUTs `latest.json` with **S3 If-None-Match: \***  for the
   first-time case, or **conditional-on-ETag** of the previously
   fetched `latest.json` otherwise. If the conditional PUT fails,
   the writer **does not retry blindly** — it re-reads, recomputes
   `flipped_from`, and retries with the new ETag. Up to 3 attempts.
5. If all 3 attempts fail, the new snapshot stays in the prefix
   (visible via the bucket listing) but the pointer didn't flip.
   The writer reports this to the brain so it can decide whether
   to fail the run or retry on the next op.

**Why conditional PUT and not a write-the-pointer-first scheme?**
Because two writers racing on the same `(tenant, profile)` is real —
two concurrent runs against the same profile_id are a 409 on the API
today, but the snapshotter also fires from a reaper path and from a
manual operator rollback. Conditional PUT is the only mechanism the
S3 API gives us for compare-and-swap that doesn't require a separate
locks bucket.

**Idempotency.** A retry that uploads the same `archive_sha256` is
a content-addressed no-op: the archive is already at its canonical
key, the manifest is identical, only the pointer flip happens (if
it hadn't already). Writers can retry the whole sequence safely.

### Why not `latest.tar.zst` directly?

A direct content-addressed pointer (e.g. symlink or a "pointer
archive" inside the prefix) lacks the metadata the loader needs:
the predecessor sha for rollback history, the captured_at_ms for
TTL decisions, the mode for cold-vs-hot policy. A separate
`latest.json` keeps the pointer and the archive decoupled — we can
upload many snapshots without re-uploading metadata, and we can
flip the pointer between snapshots with one small write.

## 3. Hot vs cold snapshot modes

### Cold mode (default)

1. Chrome exits gracefully (`SIGTERM`, wait up to 8s for cleanup,
   then `SIGKILL`).
2. Snapshotter `fsync`s the profile directory recursively.
3. tar + zstd the directory.
4. Upload, flip pointer.

Cold mode guarantees Chrome has rewritten its WAL pages and dropped
the locks the SQLite databases keep open. The archive is a
consistent point-in-time of the profile.

**This is the only mode we ship in the first cut of Phase 2.** A
caller asking for hot mode gets a clear `NotImplementedError` until
we explicitly ship it.

### Hot mode (deferred; documented for completeness)

1. Send Chrome `SIGSTOP` (freezes the process; sockets stay open,
   file descriptors stay valid).
2. Force SQLite WAL checkpoint via CDP for each `Cookies`,
   `Local Storage`, `IndexedDB`, `History` database. We need CDP
   here because Chrome holds the SQLite handles; an external
   `sqlite3` command would race the WAL.
3. `fsync` the profile directory.
4. tar + zstd.
5. Send `SIGCONT`.

Even with the CDP-driven checkpoint, hot mode carries inherent risk:
a SQLite write transaction in flight at `SIGSTOP` time is rolled
back by the checkpoint, which can drop the last few cookies set or
IndexedDB writes. We document the risk; we don't paper over it.

**When would we ever ship hot mode?**

- Long-running profiles that can't tolerate a Chrome restart (some
  paid-API flows that lose session state on a fresh browser).
- Watch-mode CI plans where the snapshotter fires every N minutes
  and the cost of stopping Chrome is unacceptable.

Neither is on the Phase 2 critical path.

### Edge cases the snapshotter must handle

- **Chrome crashed.** The profile dir is present but Chrome wrote a
  crash sentinel (`Singleton*`, `LOCK` files). Cold-mode snapshot is
  still valid; mark `notes: "chrome-crashed-before-capture"` on the
  manifest. The loader treats this as a soft warning, not a refusal.
- **Profile is empty.** First run for `(tenant, profile)`. Snapshot
  the empty directory; manifest carries
  `uncompressed_size_bytes: <few KB of bare Chrome files>`. The
  loader sees this as "no prior state" and Chrome boots fresh.
- **Profile exceeds bucket size limit.** S3 has a 5 TiB single-PUT
  cap; profiles can balloon to 10s of GB (history + cache). The
  snapshotter **rejects** profiles > 8 GB before tar; the brain
  surfaces this as a tenant-visible error so they can prune.
- **Chrome lockfile (`SingletonLock`) present without Chrome
  running.** Delete it before tar. Chrome will re-create it on next
  boot.

## 4. Chrome SQLite WAL consistency — the dominant correctness risk

Chrome's `Cookies`, `Local Storage`, `IndexedDB`, and `History`
databases all use SQLite in WAL mode. WAL means:

- The main database file is read-only during writes.
- New writes go to a `<db>.sqlite-wal` file.
- Checkpoints flush WAL → main file.

If you snapshot during writes (or before a checkpoint), three
failure modes appear:

1. **Torn read.** You capture the main file mid-write. SQLite's
   journal isn't replayed on load → data corruption.
2. **WAL drift.** You capture the WAL file separately from the main
   file. SQLite refuses to load (timestamps don't match).
3. **Stale checkpoint.** The WAL hasn't been checkpointed in a while;
   your archive has the WAL but not the main-file post-checkpoint
   state.

**Cold mode dodges all three** by waiting for Chrome to exit (Chrome
checkpoints its WAL on shutdown).

**Hot mode requires the CDP-driven `Network.clearCookies` →
`WAL_CHECKPOINT(FULL)` dance.** Even then, a write in flight at
checkpoint time can be lost. The hot-mode spec deliberately
defers shipping until we have a proven test harness that exercises
this — the docs above list the harness as a precondition.

The loader, on its side, **validates** the SQLite databases before
declaring the load successful:

```
sqlite3 Cookies "PRAGMA integrity_check"
sqlite3 "Local Storage/leveldb" ... [leveldb has its own integrity check]
```

A failure on integrity check refuses the load and falls back to
"fresh profile" with a WARNING-level log line so the operator can
triage. The brain proceeds with a fresh profile rather than
crashing the run.

## 5. Per-profile lock semantics

The current `tenant_lock_path(tenant, profile_id)` lockfile (#342)
lives on Modal Volume and dies with the executor that wrote it.
Phase 2 needs more:

- The lock has to outlive the host that holds it (a Daytona
  sandbox can die mid-run and leave a lockfile no Daytona observer
  will clean up).
- The lock has to survive cross-host transitions (a snapshot
  written from Modal, loaded by E2B, both need to know the
  profile is "in use").
- A reaper has to be able to forcibly release a lock without
  guessing.

### Storage

A separate **lock object** lives next to the snapshot pointer:

```
snapshots/<tenant>/<profile>/<chrome_major>/lock.json
```

```json
{
  "version": 1,
  "holder_run_id": "20260612_001523_aabbccdd",
  "holder_host": "modal|e2b|daytona",
  "holder_host_run_id": "fc-01KTVJM035…",
  "acquired_at_ms": 1733456500000,
  "renewed_at_ms": 1733456590000,
  "expires_at_ms": 1733456900000,
  "renewal_count": 17
}
```

### Acquire (compare-and-swap)

1. Read `lock.json`. If absent → write a new lock with `If-None-Match: *`.
   Success → owned.
2. If present and `expires_at_ms < now`, treat as expired. Write a
   new lock with `If-Match: <ETag>` of the expired lock. Success →
   owned (we forcibly took it).
3. If present and `expires_at_ms >= now`, return 409 with the
   `holder_*` fields so the brain can surface them.

### Renewal

The holder renews every 60s by writing a new `lock.json` with
`renewed_at_ms = now`, `expires_at_ms = now + TTL`, and
`renewal_count++`. Renewal uses `If-Match: <ETag>` of the current
lock — if someone else stole the lock (because we missed too many
renewals), the renewal fails and the holder must consider the
profile dirty and either re-acquire (paying the snapshot-reload
cost) or fail the run.

**Default TTL: 5 minutes.** Default renewal interval: 60 seconds.
Both tunable per-tenant.

### Reaper

A scheduled function (Modal cron or Daytona scheduled task — host's
choice) sweeps every 5 minutes:

1. List `lock.json` files across all profiles.
2. For each: read `expires_at_ms`. If `< now - grace_period_seconds`
   (default 60s), forcibly delete the lock.
3. Log the eviction at WARNING level with `holder_run_id` so the
   operator can grep "which run lost its lock."

The reaper **never** terminates running workloads — it only releases
locks. The orphaned host (if any) discovers it lost the lock on its
next renewal attempt.

### Why an object-store lock, not Redis / Modal Dict?

Modal Dict couples us to Modal. Redis adds an operational
dependency every Phase 2 host (E2B, Daytona) has to plumb to. The
object-store CAS is already free with the snapshot pipeline; the
lock is just one more small object. The trade-off is round-trip
latency (50–200ms per renew) which is fine at 60s intervals.

## 6. Cross-region behaviour

A Chrome profile loaded in a different region than it was captured
in surfaces a real issue: cookies set by CF Turnstile / DataDome /
similar carry an implicit IP fingerprint, and a sudden region
change forces re-challenges (sometimes outright bans).

The snapshotter records the **capture region** on the manifest:

```json
"captured_in": {
  "host_region": "us-east-1",
  "egress_ip_geo": {"country": "US", "region": "NY"}
}
```

The loader, on the receiving end, surfaces a WARNING when:

- `current host_region != snapshot host_region`, AND
- The snapshot is < 24h old (older snapshots are stale enough that
  the cookies probably need rotation anyway).

The loader does **not** refuse the load — that's a tenant-policy
decision. We just make sure operators see it.

### Pinning

For the warm window after a snapshot is written, the orchestrator
should prefer routing `(tenant_id, profile_id)` to the same host
region. This is a router optimisation, not a snapshotter
correctness concern; documented here so the snapshotter's
`captured_in` field is the canonical source for the router's
stickiness decision.

## 7. Failure modes and what each fails-safe to

| Failure | What happens | Fails-safe to |
|---|---|---|
| Snapshot archive PUT fails | Snapshotter retries 3× with backoff; on persistent failure, marks the run's terminal status with `snapshot_failed: true` but does NOT block the run. Previous snapshot stays canonical. | Stale-but-coherent profile on next load. |
| Manifest PUT succeeds but archive PUT failed | Snapshotter detects the mismatch on its own listing-after-write check; deletes the orphaned manifest. | Previous snapshot stays canonical. |
| Pointer flip fails (conditional PUT race) | Snapshotter retries up to 3× with re-read. On persistent failure, the snapshot stays in the bucket but the pointer doesn't flip. | Operator can flip the pointer manually via a CLI; meanwhile the previous snapshot stays canonical. |
| Loader can't download archive | Falls back to fresh profile; brain logs at WARNING; run proceeds. | Fresh Chrome. The cost is whatever the plan does on a logged-out / unverified state. |
| sha256 mismatch on download | Refuses the load; falls back to fresh profile. | Same as above; never silently loads corrupt bytes. |
| SQLite integrity check fails | Refuses the load; falls back to fresh profile. | Same. |
| Lock present but holder is dead | Reaper sweeps within 5 minutes. Until then: 409 on acquire. | Brain surfaces 409 to caller; caller retries after the TTL. |
| Lock renewal fails (lost the lock) | Holder forcibly aborts its run with `lock_lost: true` on the terminal status. | Operator sees the eviction in events log; run is retryable. |
| Chrome > 8 GB profile | Snapshotter refuses to tar; brain surfaces `profile_too_large` to caller. | Tenant must prune (instructions in the error). No silent partial snapshot. |
| Chrome major version mismatch | Loader refuses; falls back to fresh profile; logs at WARNING. | Fresh Chrome. Operator visibility on the version pin. |
| S3 region unavailable | Snapshotter and loader both fail loud after retries. | Run can't start (loader) or run finishes but profile isn't persisted (snapshotter). Documented degradation. |

Common thread: **the snapshotter never silently produces something
the loader will silently load wrong.** Every silent path produces
a fresh Chrome; every recovery path runs through operator-visible
warnings.

## 8. Operations

### Metrics (one prometheus block, emitted by both snapshotter and loader)

```
mantis_profile_snapshot_total{outcome="success|skipped|failed", host="..."}
mantis_profile_snapshot_archive_size_bytes{tenant_id="...", profile_id="..."}
mantis_profile_snapshot_capture_seconds{mode="cold|hot"}
mantis_profile_snapshot_upload_seconds
mantis_profile_snapshot_pointer_flip_seconds
mantis_profile_load_total{outcome="success|fresh_fallback|integrity_failed|sha_mismatch|version_mismatch|too_large", host="..."}
mantis_profile_load_download_seconds
mantis_profile_load_extract_seconds
mantis_profile_lock_acquire_total{outcome="acquired|forced_expired|conflict_409"}
mantis_profile_lock_evict_total{reason="ttl_expired|operator_force"}
mantis_profile_archive_size_bytes_histogram
mantis_profile_uncompressed_size_bytes_histogram
```

### Alerts

- `mantis_profile_snapshot_total{outcome="failed"}` rate > 1/min for
  10 minutes → page; something systemic is broken (auth, region,
  bucket policy).
- `mantis_profile_load_total{outcome="integrity_failed"}` >0 across
  any 5-minute window → page; we just refused to load a snapshot
  the writer thought it produced cleanly.
- `mantis_profile_load_total{outcome="sha_mismatch"}` >0 → page;
  someone's tampering with the archive bytes between upload and
  download (or the bucket is broken).

### Operator surfaces

CLI:

```
mantis profile-snapshots list <tenant_id>/<profile_id>
  → lists all snapshots under the prefix, sorted by captured_at_ms,
    with sha prefix + size + mode + chrome_major.

mantis profile-snapshots show <tenant_id>/<profile_id> [--sha <prefix>]
  → pretty-prints the manifest. Default: latest.json.

mantis profile-snapshots rollback <tenant_id>/<profile_id> --sha <prefix>
  → flips latest.json to point at the named snapshot. Requires
    --confirm to actually write. Logs the rollback in the events log.

mantis profile-snapshots delete <tenant_id>/<profile_id> --sha <prefix>
  → removes one historical snapshot. Cannot delete the current
    latest (you have to rollback first). Cannot delete the only
    snapshot (the prefix has to retain at least one).

mantis profile-snapshots force-unlock <tenant_id>/<profile_id>
  → admin override. Writes an empty lock.json with
    `holder_run_id: "operator-force"` so the reaper sweeps it next
    pass. Requires confirm + logs the reason.
```

API:

```
GET  /v1/profile-snapshots/{tenant_id}/{profile_id}
GET  /v1/profile-snapshots/{tenant_id}/{profile_id}/{sha_prefix}
POST /v1/profile-snapshots/{tenant_id}/{profile_id}/rollback
DELETE /v1/profile-snapshots/{tenant_id}/{profile_id}/{sha_prefix}
POST /v1/profile-snapshots/{tenant_id}/{profile_id}/force-unlock
```

All tenant-gated; `force-unlock` and `delete` additionally
admin-gated.

## 9. Implementation milestones

The snapshotter ships as `src/mantis_agent/observability/profile_snapshotter.py`
in three milestones; each milestone is a separate PR.

**M1 — Read path only.** The loader, with no writer side. Reads
`latest.json`, downloads the archive, verifies sha + integrity,
extracts into the local profile path. The writer is a stub that
prints what it would do. This lets us validate the contract end-to-end
by manually populating snapshots for a couple of test tenants, then
running brains that load them. No correctness risk to existing tenants.

**M2 — Write path with cold-mode default.** The full writer: capture,
tar+zstd, upload, conditional pointer flip. Hot mode raises
`NotImplementedError` with a pointer to this doc. M2 is the first
milestone that lets a Phase 2 host (E2B, Daytona) actually replace
Modal Volume reuse.

**M3 — Operator surfaces.** CLI commands + HTTP routes + metrics +
alerts. Optional polish — nothing on the critical CUA path depends
on it; ships when the first non-Modal host has 1+ paying tenant
relying on snapshot reuse.

The acceptance criterion for the gate (this doc) lifting is:

- M1 reviewed and approved.
- A worked example walks through a snapshot round-trip for one
  test tenant + profile, using S3 and a local Daytona-like
  sandbox. The example lives in `docs/getting-started/`.

## 10. Test plan

This is a correctness-critical surface; the test plan is unusually
detailed. Each item is implementable against a local S3 (MinIO) or
the real bucket via env var.

### Unit tests (`tests/test_profile_snapshotter_*.py`)

- Manifest schema round-trips (every required field; rejects v0).
- Pointer flip with conditional PUT; race with two writers; either
  succeeds and the loser sees the ETag mismatch and retries.
- Pointer rollback (operator-side flip back to a prior sha).
- sha256 mismatch refuses load.
- Chrome major version mismatch refuses load.
- Profile too large rejects on the writer side.
- SQLite integrity check failure refuses load.
- Cold mode snapshot of a running Chrome → snapshotter waits for
  exit (or `SIGKILL`s after 8s); manifest reflects the wait.
- Hot mode raises `NotImplementedError` with the doc link.

### Lock unit tests (`tests/test_profile_lock_*.py`)

- Acquire on empty: writes lock; succeeds.
- Acquire on held: 409.
- Acquire on expired: takes it.
- Renewal: ETag stays consistent through N renewals.
- Renewal after eviction: fails; holder gets `lock_lost`.
- Reaper sweep: evicts past-grace locks; leaves fresh ones alone.

### Integration test (one) — full round-trip

A single integration test that:

1. Materialises a non-empty Chrome profile with known cookies +
   localStorage + IndexedDB.
2. Snapshots it (cold-mode default).
3. Verifies the archive in MinIO + the pointer's
   `active_sha256_prefix`.
4. Wipes the local profile.
5. Loads via the loader.
6. Spawns Chrome, reads the cookies + localStorage + IndexedDB.
7. Asserts they match the originals.

This is the canonical "does the spec work end-to-end" gate. If
this test breaks, none of the lower-level tests matter.

### Soak test (manual; documented, not in CI)

Run the integration test 100 times back-to-back on a real S3
bucket; record:

- Mean + p99 capture seconds.
- Mean + p99 round-trip wall-clock (snapshot → load).
- Failure rate (must be 0).
- Manifest size distribution.

Document the soak-test results in this file's appendix before M2
ships. Re-run the soak on every Chrome major version bump.

## 11. Open questions

- **GC of historical snapshots.** This doc doesn't yet specify how
  we age out old snapshots — keep every snapshot forever? Keep
  last N? Last 7 days? Needs a tenant-policy field. Suggested
  default: keep the 5 newest, plus any tagged for rollback in the
  last 30 days. Defer the decision to a follow-up doc; default to
  "no GC" until then so we never silently lose state.
- **Encryption at rest (BYOK).** Per-tenant KMS key support is a
  real customer ask. Defer to a follow-up doc — adds a small layer
  to the loader (KMS decrypt) and writer (KMS encrypt). The wire
  contract here doesn't change.
- **Cross-host profile portability.** A profile captured on Modal
  loaded on E2B Desktop should work the same; the integration test
  enforces this by varying the host. Are there host-specific paths
  inside Chrome's profile that need rewriting on load? Investigation
  open during M1.

## References

- [#699](https://github.com/mercurialsolo/mantis/issues/699) — Phase 2 issue
- [#696](https://github.com/mercurialsolo/mantis/issues/696) — Computer Plane umbrella
- [`docs/reference/computer-plane.md`](computer-plane.md) — Phase 0/1/1.5 spec
- [`docs/reference/compute-client.md`](compute-client.md) — unified ComputeClient contract
- [`feedback_no_customer_names_in_tracked_files`](https://github.com/mercurialsolo/mantis) (memory) — when documenting tenant examples, use neutral names (`acme`, `alice`); never real customer brand names
