"""
promotion.py

Verified promotion of exactly-certified discoveries into discoveries.json.

Pipeline:  numerical candidate -> exact certificate construction ->
independent verify_certificate -> certified staging record (run-local,
append-only certified.jsonl written by workers) -> ATOMIC promotion into
discoveries.json by a single controlled merger.

Guarantees:
  * only exact, independently re-verified certificates are promoted
    (verify_certificate is re-run immediately before writing);
  * exact coefficient data only (floats / silent rationalization rejected
    by verify_certificate);
  * atomic replace (tmp file + fsync + os.replace) under an exclusive lock
    file, so concurrent promoters cannot corrupt the store;
  * idempotent: deduplicated by a canonical EXACT projective identifier
    (sha256 of the sorted canonical rational line tuples — exact arithmetic,
    never floating coordinates; arrangements are NOT identified across
    general GL/PGL images);
  * existing entries are preserved; entries carrying certificates are
    re-verified on merge, legacy entries (no certificate — the historical
    RL-era records) are preserved untouched and labeled
    'legacy_unverified_by_promoter';
  * nontrivial campaign exports enforce d1 >= 2 (baseline classes require
    allow_baseline=True; the mathematical library itself remains
    d1 = 0/1 capable).
"""

import hashlib
import json
import os
import time

from arrangement import LineArrangement, ProjectiveLine
from certificates import verify_certificate, certificate_to_json, \
    certificate_from_json

SCHEMA_VERSION = "discovery-2.0"


def canonical_discovery_id(arr: LineArrangement) -> str:
    """Exact canonical identifier: sha256 over the sorted canonical rational
    line tuples (ProjectiveLine canonicalizes by the first nonzero
    coordinate, so per-line scaling and line order are quotiented; general
    GL/PGL images are deliberately NOT identified)."""
    key = repr(tuple(sorted(str(l.coords) for l in arr.lines)))
    return hashlib.sha256(key.encode()).hexdigest()


def certificate_hash(cert_json: dict) -> str:
    return hashlib.sha256(
        json.dumps(cert_json, sort_keys=True).encode()).hexdigest()


def build_discovery_entry(cert, run_id, engine=None, pair_class=None,
                          search_params=None, lattice_hash=None):
    """Build a schema-2.0 entry from an IN-MEMORY certificate dict.
    verify_certificate is the caller's responsibility at staging time; the
    promoter re-verifies regardless."""
    from penalized_saito import (FUNCTIONAL_VERSION, DEFAULT_LAMBDA,
                                 DEFAULT_BETA, runtime_provenance)
    lines = [ProjectiveLine(*c) for c in cert["lines"]]
    arr = LineArrangement(lines)
    d1, d2 = cert["d1"], cert["d2"]
    cj = certificate_to_json(cert)
    sp_ = search_params or {}
    prov = runtime_provenance(".")
    entry = {
        # legacy-compatible fields (readers of the historical schema)
        "n": len(arr),
        "exponents": [1, d1, d2],
        "b2": arr.b2(),
        "max_mult": arr.max_multiplicity(),
        "mult_profile": sorted(arr.multiplicities(), reverse=True),
        "n_pts": arr.n_intersection_points(),
        "lines": [str(l) for l in arr.lines],
        "source": sp_.get("source", "swap_promotion"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # schema-2.0 exact-certificate fields
        "schema_version": SCHEMA_VERSION,
        "discovery_id": canonical_discovery_id(arr),
        "field": cj.get("field", "QQ"),
        "certificate": cj,
        "certificate_hash": certificate_hash(cj),
        "verification_status": "verified_exact",
        "target_pair": [d1, d2],
        "pair_class": pair_class or ("nontrivial" if d1 >= 2 else
                                     "baseline_near_pencil" if d1 == 1
                                     else "baseline_pencil"),
        "run_id": run_id,
        "engine": engine,
        "functional_version": FUNCTIONAL_VERSION,
        "search_lambda": (sp_.get("lambda") if sp_.get("lambda") is not None
                          else "unknown_pre_audit"),
        "search_beta": (sp_.get("beta") if sp_.get("beta") is not None
                        else "unknown_pre_audit"),
        "search_field": sp_.get("field") or "real",
        "code_commit": prov["code_commit"],
        "source_content_hash": prov["source_content_hash"],
        "lattice_hash": lattice_hash,
    }
    return entry


class _FileLock:
    """Exclusive lock via O_CREAT|O_EXCL lock file (portable, NFS-tolerant
    enough for single-host promoters); stale locks older than `stale_s` are
    broken."""

    def __init__(self, path, timeout=60.0, stale_s=600.0):
        self.lock_path = path + ".lock"
        self.timeout = timeout
        self.stale_s = stale_s

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock_path) \
                            > self.stale_s:
                        os.unlink(self.lock_path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(f"lock {self.lock_path} busy")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass


def _load_store(path):
    if not os.path.exists(path):
        return {"arrangements": [], "index": {}}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("arrangements", [])
    data.setdefault("index", {})
    return data


def _atomic_write(data, path):
    """tmp write + fsync + os.replace + PARENT-DIRECTORY fsync (crash
    durability of the rename itself)."""
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".",
                      os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def load_verified_discoveries(path, reverify=True):
    """STRICT loader for the verified store: returns only entries with a
    supported schema version, verification_status == 'verified_exact', a
    certificate present, and (when reverify) verify_certificate passing.
    Everything else is returned separately as rejects — never counted as a
    discovery."""
    data = _load_store(path)
    ok, rejects = [], []
    for r in data["arrangements"]:
        if r.get("schema_version") != SCHEMA_VERSION:
            rejects.append((r, "unsupported_schema"))
            continue
        if r.get("verification_status") != "verified_exact":
            rejects.append((r, "not_verified_exact"))
            continue
        if "certificate" not in r:
            rejects.append((r, "missing_certificate"))
            continue
        if reverify:
            try:
                if not verify_certificate(
                        certificate_from_json(r["certificate"])):
                    rejects.append((r, "failed_reverification"))
                    continue
            except Exception as ex:
                rejects.append((r, f"certificate_error({ex})"))
                continue
        ok.append(r)
    return ok, rejects


def migrate_legacy_store(path, out_legacy=None, out_quarantine=None,
                         dry_run=False):
    """Recoverable migration enforcing the contract that discoveries.json
    holds ONLY exactly-verified schema-2 entries.

    1. checksum-addressed backup of the current file;
    2. partition into verified schema-2 / legacy-unverified / malformed;
    3. verified entries stay; legacy candidates move to legacy_candidates
       .json (original data + provenance + original index + migration
       timestamp/reason); malformed entries go to a quarantine report;
    4. atomic writes throughout; restartable (idempotent on a migrated
       store); the backup is never deleted.
    Returns a summary dict.
    """
    import hashlib
    import time as _t
    out_legacy = out_legacy or os.path.join(
        os.path.dirname(os.path.abspath(path)), "legacy_candidates.json")
    out_quarantine = out_quarantine or os.path.join(
        os.path.dirname(os.path.abspath(path)),
        "legacy_quarantine_report.json")
    if not os.path.exists(path):
        return {"status": "no_store", "path": path}
    raw_bytes = open(path, "rb").read()
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    backup = path + f".backup.{checksum[:16]}"
    if not dry_run and not os.path.exists(backup):
        with open(backup, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())
    data = _load_store(path)
    stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    verified, legacy, malformed = [], [], []
    for idx, r in enumerate(data["arrangements"]):
        try:
            if r.get("schema_version") == SCHEMA_VERSION and \
                    r.get("verification_status") == "verified_exact" and \
                    "certificate" in r:
                if verify_certificate(
                        certificate_from_json(r["certificate"])):
                    verified.append(r)
                else:
                    r2 = dict(r)
                    r2["_quarantine_reason"] = "failed_reverification"
                    r2["_original_index"] = idx
                    malformed.append(r2)
            elif isinstance(r, dict) and "lines" in r:
                r2 = dict(r)
                r2["_migration"] = {
                    "timestamp": stamp, "original_index": idx,
                    "reason": "legacy_unverified_by_promoter",
                    "source_checksum": checksum,
                }
                legacy.append(r2)
            else:
                malformed.append({"_original_index": idx,
                                  "_quarantine_reason": "malformed",
                                  "record": r})
        except Exception as ex:
            malformed.append({"_original_index": idx,
                              "_quarantine_reason": f"error({ex})",
                              "record": str(r)[:2000]})
    summary = {
        "source_checksum": checksum, "backup": backup,
        "n_total": len(data["arrangements"]), "n_verified": len(verified),
        "n_legacy": len(legacy), "n_malformed": len(malformed),
        "dry_run": dry_run, "timestamp": stamp,
        "already_migrated": (len(legacy) == 0 and len(malformed) == 0),
    }
    if dry_run:
        return summary
    with _FileLock(path):
        # merge legacy candidates into any existing legacy store (restart-
        # safe: dedup by original data key)
        legacy_store = _load_store(out_legacy) if os.path.exists(out_legacy) \
            else {"arrangements": [], "index": {}}
        known = {json.dumps(r.get("lines"), sort_keys=True)
                 for r in legacy_store["arrangements"]}
        for r in legacy:
            k = json.dumps(r.get("lines"), sort_keys=True)
            if k not in known:
                legacy_store["arrangements"].append(r)
                known.add(k)
        _atomic_write(legacy_store, out_legacy)
        if malformed:
            _atomic_write({"quarantined": malformed,
                           "source_checksum": checksum,
                           "timestamp": stamp}, out_quarantine)
        new_index = {}
        for i, r in enumerate(verified):
            legacy_key = f"{tuple(sorted(r['lines']))}|" \
                         f"{tuple(r.get('exponents', []))}"
            new_index[legacy_key] = i
        _atomic_write({"arrangements": verified, "index": new_index,
                       "store_meta": {"contract": "verified_exact_only",
                                      "migrated": stamp,
                                      "source_checksum": checksum}}, path)
    return summary


def promote(entries, path, allow_baseline=False, reverify_existing=True):
    """Atomically merge schema-2.0 entries into the discovery store.

    Returns {'promoted': k, 'duplicates': d, 'rejected': [(id, reason)]}.
    Every entry is independently re-verified here (certificate parsed from
    its JSON form and verify_certificate re-run from scratch); duplicates
    (same discovery_id or legacy canonical key) are skipped; existing
    records are never dropped.  With reverify_existing, previously promoted
    schema-2.0 entries are re-checked and flagged (never silently removed)
    if their certificate fails.
    """
    rejected, accepted = [], []
    for e in entries:
        did = e.get("discovery_id", "?")
        if e.get("schema_version") != SCHEMA_VERSION:
            rejected.append((did, "wrong_schema"))
            continue
        d1, d2 = e["target_pair"]
        if d1 < 2 and not allow_baseline:
            rejected.append((did, f"baseline_pair_class_d1={d1}"))
            continue
        try:
            cert = certificate_from_json(e["certificate"])
        except Exception as ex:
            rejected.append((did, f"certificate_parse_error({ex})"))
            continue
        if certificate_hash(e["certificate"]) != e.get("certificate_hash"):
            rejected.append((did, "certificate_hash_mismatch"))
            continue
        if not verify_certificate(cert):
            rejected.append((did, "certificate_failed_reverification"))
            continue
        # exact identifier recomputed from the certificate, never trusted
        arr = LineArrangement([ProjectiveLine(*c) for c in cert["lines"]])
        if canonical_discovery_id(arr) != did:
            rejected.append((did, "discovery_id_mismatch"))
            continue
        accepted.append(e)

    promoted = duplicates = 0
    with _FileLock(path):
        data = _load_store(path)
        known_ids = {r.get("discovery_id") for r in data["arrangements"]
                     if r.get("discovery_id")}
        if reverify_existing:
            for r in data["arrangements"]:
                if r.get("schema_version") == SCHEMA_VERSION and \
                        r.get("verification_status") == "verified_exact":
                    try:
                        ok = verify_certificate(
                            certificate_from_json(r["certificate"]))
                    except Exception:
                        ok = False
                    if not ok:
                        r["verification_status"] = \
                            "FAILED_REVERIFICATION_ON_MERGE"
                elif "schema_version" not in r:
                    r.setdefault("verification_status",
                                 "legacy_unverified_by_promoter")
        for e in accepted:
            legacy_key = f"{tuple(sorted(e['lines']))}|{tuple(e['exponents'])}"
            if e["discovery_id"] in known_ids or \
                    legacy_key in data["index"]:
                duplicates += 1
                continue
            data["index"][legacy_key] = len(data["arrangements"])
            data["arrangements"].append(e)
            known_ids.add(e["discovery_id"])
            promoted += 1
        if promoted or reverify_existing:
            _atomic_write(data, path)
    return {"promoted": promoted, "duplicates": duplicates,
            "rejected": rejected}


def promote_from_staging(cells_dir, path, allow_baseline=False,
                         run_id="swap_campaign"):
    """Merge every cells/*/certified.jsonl staging record into the store.

    Staging records carry the certificate file reference; the certificate
    JSON is loaded and re-verified here."""
    import glob
    entries = []
    for cpath in sorted(glob.glob(os.path.join(cells_dir, "*",
                                               "certified.jsonl"))):
        cell_dir = os.path.dirname(cpath)
        with open(cpath) as f:
            for line in f:
                rec = json.loads(line)
                cert_file = os.path.join(cell_dir, rec["certificate_file"])
                with open(cert_file) as cf:
                    cj = json.load(cf)
                cert = certificate_from_json(cj)
                entry = build_discovery_entry(
                    cert, run_id=run_id, engine=rec.get("engine"),
                    search_params={"lambda": rec.get("lambda"),
                                   "beta": rec.get("beta"),
                                   "field": rec.get("optimization_field",
                                                    "real"),
                                   "source": f"swap_{rec.get('engine')}"},
                    lattice_hash=rec.get("lattice_hash"))
                entries.append(entry)
    return promote(entries, path, allow_baseline=allow_baseline)
