"""Offline, restartable approval matching and count-only FDA corpus sizing."""

import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import zipfile

from src.preprocessing.chunking import chunk_documents, token_count
from src.preprocessing.normalization import normalize_record, strings
from src.utils.config import canonical, digest, utcnow

GIB = 1024**3
SCHEMA_VERSION = 1
MAX_VALUE_CHARS = 64 * 1024**2


class Paused(RuntimeError):
    """A resource allowance was reached; committed checkpoints remain usable."""


class JsonStream:
    """Decode individual JSON values from a bounded incremental text buffer."""

    def __init__(self, stream, block_size=65536):
        """Initialize a reader without reading the whole file."""
        self.stream, self.block_size = stream, block_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        """Discard consumed text and append one block."""
        self.buffer = self.buffer[self.position:]
        self.position = 0
        block = self.stream.read(self.block_size)
        self.eof = not block
        self.buffer += block
        if len(self.buffer) > MAX_VALUE_CHARS:
            raise ValueError("One JSON value exceeds the 64 Mi-character safety limit")

    def peek(self):
        """Return the next non-whitespace character, or an empty string at EOF."""
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, character):
        """Consume an expected JSON delimiter or reject malformed input."""
        if self.peek() != character:
            raise ValueError(f"Expected JSON delimiter {character!r}")
        self.position += 1

    def value(self):
        """Read one complete value, extending the buffer as necessary."""
        if not self.peek():
            raise ValueError("Truncated JSON value")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # Numeric literals can finish at a block boundary before their
                # final digits arrive. Objects/strings have explicit endings.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("Invalid or truncated JSON") from error
                self.fill()


def iter_results(path, block_size=65536):
    """Yield results records while validating the full FDA JSON wrapper and ZIP CRC."""
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.endswith(".json")]
        if len(members) != 1:
            raise ValueError("Expected exactly one JSON member per FDA ZIP")
        with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig") as stream:
            reader = JsonStream(stream, block_size)
            reader.expect("{")
            found = False
            keys = set()
            while reader.peek() != "}":
                key = reader.value()
                if not isinstance(key, str) or key in keys:
                    raise ValueError("Invalid or duplicate top-level JSON key")
                keys.add(key)
                reader.expect(":")
                if key == "results":
                    found = True
                    reader.expect("[")
                    if reader.peek() != "]":
                        while True:
                            yield reader.value()
                            if reader.peek() == "]":
                                break
                            reader.expect(",")
                            if reader.peek() == "]":
                                raise ValueError("Trailing comma in results array")
                    reader.expect("]")
                else:
                    reader.value()
                if reader.peek() == "}":
                    break
                reader.expect(",")
                if reader.peek() == "}":
                    raise ValueError("Trailing comma in JSON object")
            reader.expect("}")
            if not found or reader.peek():
                raise ValueError("Missing results array or trailing JSON content")


def normalize_application(value):
    """Normalize an explicit NDA/ANDA/BLA identifier without guessing its prefix."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(ANDA|NDA|BLA)[\s-]*(\d{1,6})", value.strip().upper())
    return match[1] + match[2].zfill(6) if match else None


def version_rank(record):
    """Order versions numerically, then by effective date, then canonical source hash."""
    version = str(record.get("version", ""))
    number = int(version) if re.fullmatch(r"\d{1,9}", version) else -1
    effective = record.get("effective_time", "")
    effective = effective if isinstance(effective, str) and re.fullmatch(r"\d{8}", effective) else ""
    return number, effective, digest(record)


def peak_rss_bytes():
    """Read native process peak resident memory on Windows, Linux or macOS."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "page", "page_peak"
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.WinDLL("psapi", use_last_error=True)
        api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        api.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not api.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


class ResourceGuard:
    """Enforce analysis-only disk allowance and a resident-memory stop threshold."""

    def __init__(self, folder, max_bytes=2 * GIB, min_free=80 * GIB, max_ram=4 * GIB):
        """Set independent limits; existing source downloads do not count as new storage."""
        self.folder, self.max_bytes = Path(folder), max_bytes
        self.min_free, self.max_ram = min_free, max_ram

    def usage(self):
        """Measure all analysis artifacts, including SQLite rollback journals."""
        return sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())

    def check(self):
        """Pause before continuing when memory, disk or free-space limits are reached."""
        if self.usage() >= self.max_bytes:
            raise Paused("Analysis storage allowance reached (default 2 GiB)")
        if shutil.disk_usage(self.folder).free < self.min_free:
            raise Paused("Destination disk has less than the required free-space reserve")
        if peak_rss_bytes() >= self.max_ram:
            raise Paused("Process peak RAM reached the configured stop threshold")


def open_database(path, fingerprint, max_bytes=2 * GIB):
    """Open compact analysis state and reject incompatible source/configuration resumes."""
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=TRUNCATE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    page_size = db.execute("PRAGMA page_size").fetchone()[0]
    # Leave space for a rollback journal as large as the database and for reports.
    maximum = max(32, (max_bytes - 16 * 1024**2) // (2 * page_size))
    db.execute(f"PRAGMA max_page_count={maximum}")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (
            source TEXT PRIMARY KEY, records INTEGER NOT NULL, complete INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS approvals (
            application TEXT PRIMARY KEY, approved INTEGER NOT NULL, tentative INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS labels (
            source TEXT NOT NULL, ordinal INTEGER NOT NULL, source_hash TEXT NOT NULL,
            set_id TEXT, category TEXT NOT NULL, matched_applications TEXT NOT NULL,
            PRIMARY KEY(source, ordinal)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS labels_hash ON labels(source_hash);
        CREATE INDEX IF NOT EXISTS labels_set ON labels(set_id);
        CREATE TABLE IF NOT EXISTS selected (
            set_id TEXT PRIMARY KEY, source TEXT NOT NULL, ordinal INTEGER NOT NULL,
            source_hash TEXT NOT NULL, version INTEGER NOT NULL, effective TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS sized (
            set_id TEXT PRIMARY KEY, passages INTEGER NOT NULL, embedding_tokens INTEGER NOT NULL,
            passage_text_tokens INTEGER NOT NULL, passage_jsonl_bytes INTEGER NOT NULL,
            embedding_text_bytes INTEGER NOT NULL, normalized_jsonl_bytes INTEGER NOT NULL,
            maximum_input_tokens INTEGER NOT NULL, overlong_inputs INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS inputs (
            hash BLOB PRIMARY KEY, tokens INTEGER NOT NULL, bytes INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS sections (
            section TEXT PRIMARY KEY, passages INTEGER NOT NULL, embedding_tokens INTEGER NOT NULL
        ) WITHOUT ROWID;
    """)
    expected = canonical({"schema": SCHEMA_VERSION, "fingerprint": fingerprint})
    previous = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if previous and previous[0] != expected:
        db.close()
        raise ValueError("Sources or processing configuration changed; use a new analysis directory")
    db.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (expected,))
    db.commit()
    return db


def completed(db, source):
    """Return the committed record cursor and completion flag for a partition/stage."""
    return db.execute("SELECT records, complete FROM progress WHERE source=?", (source,)).fetchone() or (0, 0)


def checkpoint(db, source, records, complete=False):
    """Commit the cursor in the same transaction as its derived records."""
    db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?)", (source, records, int(complete)))
    db.commit()


def build_approvals(db, path, expected_records, guard, emit=print):
    """Build an approval lookup, retaining tentative-only applications for reporting."""
    source = "approval:" + path.name
    cursor, done = completed(db, source)
    if done:
        return
    total, last = 0, time.monotonic()
    for total, record in enumerate(iter_results(path), 1):
        if total <= cursor:
            continue
        if not isinstance(record, dict):
            raise ValueError("Malformed Drugs@FDA record")
        application = normalize_application(record.get("application_number"))
        if application:
            submissions = record.get("submissions", [])
            if not isinstance(submissions, list):
                raise ValueError("Malformed Drugs@FDA submissions")
            statuses = {s.get("submission_status") for s in submissions if isinstance(s, dict)}
            db.execute("""INSERT INTO approvals VALUES (?, ?, ?) ON CONFLICT(application) DO UPDATE SET
                approved=MAX(approved, excluded.approved), tentative=MAX(tentative, excluded.tentative)
            """, (application, int("AP" in statuses), int("TA" in statuses)))
        if total % 250 == 0:
            guard.check()
            checkpoint(db, source, total)
        if time.monotonic() - last >= 10:
            emit(f"Approvals: {total:,}/{expected_records:,}", flush=True)
            last = time.monotonic()
    if total != expected_records:
        raise ValueError(f"Approval count mismatch: observed {total}, expected {expected_records}")
    checkpoint(db, source, total, True)


def match_applications(db, record):
    """Classify labels by explicit application matches, never by drug-name similarity."""
    if not isinstance(record, dict) or not isinstance(record.get("openfda", {}), dict):
        return "malformed_record", []
    values = strings(record.get("openfda", {}).get("application_number"))
    if not values:
        return "missing_application", []
    applications = sorted({app for value in values if (app := normalize_application(value))})
    if not applications:
        return "invalid_application", []
    rows = [(app, db.execute("SELECT approved, tentative FROM approvals WHERE application=?", (app,))
             .fetchone()) for app in applications]
    matched = [app for app, row in rows if row and row[0]]
    if matched:
        return "matched", matched
    if any(row and row[1] for _, row in rows):
        return "tentative_without_approval", []
    if any(row for _, row in rows):
        return "application_without_approval", []
    return "unmatched_application", []


def match_partition(db, path, expected_records, guard, emit=print, stop_after=None):
    """Audit every label and keep only locations/identities of newest matched label sets."""
    source = "label:" + path.name
    cursor, done = completed(db, source)
    if done:
        return
    total, last = 0, time.monotonic()
    for total, record in enumerate(iter_results(path), 1):
        if total <= cursor:
            continue
        category, applications = match_applications(db, record)
        record_hash = digest(record)
        set_id = record.get("set_id") if isinstance(record, dict) else None
        set_id = set_id if isinstance(set_id, str) and set_id.strip() else None
        db.execute("INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?)", (
            path.name, total, record_hash, set_id, category, canonical(applications),
        ))
        if category == "matched" and set_id:
            version, effective, _ = version_rank(record)
            db.execute("""INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(set_id) DO UPDATE SET source=excluded.source, ordinal=excluded.ordinal,
                source_hash=excluded.source_hash, version=excluded.version, effective=excluded.effective
                WHERE (excluded.version, excluded.effective, excluded.source_hash) >
                      (selected.version, selected.effective, selected.source_hash)
            """, (set_id, path.name, total, record_hash, version, effective))
        if total % 100 == 0:
            guard.check()
            checkpoint(db, source, total)
        if time.monotonic() - last >= 10:
            emit(f"Matching {path.name}: {total:,}/{expected_records:,}", flush=True)
            last = time.monotonic()
        if stop_after is not None and total >= stop_after:
            checkpoint(db, source, total)
            raise Paused("Requested trial checkpoint reached; rerun to resume")
    if total != expected_records:
        raise ValueError(f"Label count mismatch in {path.name}: {total} != {expected_records}")
    checkpoint(db, source, total, True)


def size_record(db, record, source_hash, snapshot_id, chunking, guard, commit=True):
    """Count exact existing-pipeline inputs without saving documents, passages or vectors."""
    document = normalize_record({
        "record": record, "kind": "fda_full", "provenance": {"body_sha256": source_hash},
    }, snapshot_id)
    chunks = chunk_documents([document], **chunking)
    totals = dict.fromkeys(("passages", "tokens", "text_tokens", "jsonl_bytes", "text_bytes", "long"), 0)
    maximum, sections = 0, {}
    for number, chunk in enumerate(chunks, 1):
        text = chunk.embedding_text.encode("utf-8")
        tokens = token_count(chunk.embedding_text, chunking["encoding"])
        db.execute("INSERT OR IGNORE INTO inputs VALUES (?, ?, ?)", (hashlib.sha256(text).digest(), tokens,
                                                                    len(text)))
        totals["passages"] += 1
        totals["tokens"] += tokens
        totals["text_tokens"] += chunk.token_count
        totals["jsonl_bytes"] += len(canonical(chunk.to_dict()).encode("utf-8")) + 1
        totals["text_bytes"] += len(text)
        totals["long"] += int(tokens > 8191)
        maximum = max(maximum, tokens)
        section = sections.setdefault(chunk.section, [0, 0])
        section[0] += 1
        section[1] += tokens
        if number % 128 == 0:
            guard.check()
    for section, (passages, tokens) in sections.items():
        db.execute("""INSERT INTO sections VALUES (?, ?, ?) ON CONFLICT(section) DO UPDATE SET
            passages=passages+excluded.passages, embedding_tokens=embedding_tokens+excluded.embedding_tokens
        """, (section, passages, tokens))
    db.execute("INSERT INTO sized VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        document["set_id"], totals["passages"], totals["tokens"], totals["text_tokens"],
        totals["jsonl_bytes"], totals["text_bytes"], len(canonical(document).encode("utf-8")) + 1,
        maximum, totals["long"],
    ))
    guard.check()
    if commit:
        db.commit()


def size_partition(db, path, expected_records, source_hash, snapshot_id, chunking, guard, emit=print,
                   stop_after=None):
    """Replay one compressed partition, sizing only selected, not-yet-counted label sets."""
    source = "size:" + path.name
    _, done = completed(db, source)
    if done:
        return
    # At most one source partition's selection (20,000 records), never the full corpus.
    wanted = {row[0]: row[1:] for row in db.execute("""SELECT s.ordinal, s.set_id, s.source_hash
        FROM selected s LEFT JOIN sized z ON s.set_id=z.set_id
        WHERE s.source=? AND z.set_id IS NULL""", (path.name,))}
    total, processed, last = 0, 0, time.monotonic()
    last_commit = last
    for total, record in enumerate(iter_results(path), 1):
        if total not in wanted:
            continue
        set_id, expected_hash = wanted[total]
        if digest(record) != expected_hash or record.get("set_id") != set_id:
            raise ValueError("Selected record identity changed during sizing")
        size_record(db, record, source_hash, snapshot_id, chunking, guard, commit=False)
        processed += 1
        if processed % 25 == 0 or time.monotonic() - last_commit >= 5:
            db.commit()
            last_commit = time.monotonic()
        if time.monotonic() - last >= 10:
            count = db.execute("SELECT COALESCE(SUM(passages),0) FROM sized").fetchone()[0]
            emit(f"Sizing {path.name}: {processed:,}/{len(wanted):,} pending sets; "
                 f"{count:,} total passages", flush=True)
            last = time.monotonic()
        if stop_after is not None and processed >= stop_after:
            db.commit()
            raise Paused("Requested sizing trial reached; rerun to resume")
    if total != expected_records:
        raise ValueError(f"Sizing source count mismatch: {total} != {expected_records}")
    checkpoint(db, source, total, True)


def baseline_comparison(db, records):
    """Compare all original full-label records with approval policy and downloaded identities."""
    results, seen = [], set()
    for envelope in records:
        if envelope["kind"] != "fda_full":
            continue
        record = envelope["record"]
        identity = digest(record)
        if identity in seen:
            continue
        seen.add(identity)
        category, applications = match_applications(db, record)
        exact = db.execute("SELECT 1 FROM labels WHERE source_hash=? LIMIT 1", (identity,)).fetchone()
        selected = db.execute("SELECT source_hash FROM selected WHERE set_id=?", (record.get("set_id"),))
        selected = selected.fetchone()
        exists = db.execute("SELECT 1 FROM labels WHERE set_id=? LIMIT 1", (record.get("set_id"),)).fetchone()
        results.append({
            "set_id": record.get("set_id"), "source_record_hash": identity,
            "approval_category": category, "matched_applications": applications,
            "identical_record_in_download": bool(exact), "set_in_download": bool(exists),
            "identical_to_selected_record": bool(selected and selected[0] == identity),
        })
    return {"records": len(results), "approval_matched_records": sum(
        row["approval_category"] == "matched" for row in results
    ), "details": results}


def analysis_report(db, source_report, config, fingerprint, guard, status, error=None):
    """Distinguish exact corpus counts and storage lower bounds from unmeasured server RAM."""
    categories = dict(db.execute("SELECT category, COUNT(*) FROM labels GROUP BY category"))
    totals = db.execute("""SELECT COUNT(*), COALESCE(SUM(passages),0), COALESCE(SUM(embedding_tokens),0),
        COALESCE(SUM(passage_text_tokens),0), COALESCE(SUM(passage_jsonl_bytes),0),
        COALESCE(SUM(embedding_text_bytes),0), COALESCE(SUM(normalized_jsonl_bytes),0),
        COALESCE(MAX(maximum_input_tokens),0), COALESCE(SUM(overlong_inputs),0) FROM sized""").fetchone()
    unique = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(bytes),0) FROM inputs"
    ).fetchone()
    dimensions = config["embedding"]["dimensions"]
    vectors, cache = totals[1] * dimensions * 4, unique[0] * dimensions * 4
    selected = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    matched_sets = db.execute(
        "SELECT COUNT(DISTINCT set_id) FROM labels WHERE category='matched'"
    ).fetchone()[0]
    complete_matching = sum(row[0] for row in db.execute(
        "SELECT complete FROM progress WHERE source LIKE 'label:%'"
    ))
    return {
        "generated_at": utcnow(), "status": status, "error": error, "fingerprint": fingerprint,
        "scope": "Every approval-matched label set in all downloaded partitions; newest matched version",
        "approval_rule": "Application-number match with at least one AP submission; not current-market proof",
        "configuration": config, "source": source_report,
        "matching": {
            "scanned_records": sum(categories.values()), "categories": categories,
            "unique_source_records": db.execute(
                "SELECT COUNT(DISTINCT source_hash) FROM labels"
            ).fetchone()[0],
            "unique_label_sets": db.execute("SELECT COUNT(DISTINCT set_id) FROM labels").fetchone()[0],
            "matched_unique_label_sets": matched_sets, "selected_records": selected,
            "matched_missing_set_id": db.execute(
                "SELECT COUNT(*) FROM labels WHERE category='matched' AND set_id IS NULL"
            ).fetchone()[0],
            "matched_duplicate_or_superseded_records": categories.get("matched", 0) - matched_sets
            - db.execute(
                "SELECT COUNT(*) FROM labels WHERE category='matched' AND set_id IS NULL"
            ).fetchone()[0],
            "completed_label_partitions": complete_matching,
            "qualifying_approval_applications": db.execute(
                "SELECT COUNT(*) FROM approvals WHERE approved=1"
            ).fetchone()[0],
        },
        "sizing": dict(zip((
            "sized_label_records", "passages", "embedding_input_tokens", "passage_text_tokens",
            "passage_jsonl_bytes_if_written", "embedding_text_bytes", "normalized_jsonl_bytes_if_written",
            "maximum_embedding_input_tokens", "embedding_inputs_exceeding_8191_tokens",
        ), totals, strict=True)) | {
            "unique_embedding_inputs": unique[0], "unique_embedding_input_tokens": unique[1],
            "unique_embedding_text_bytes": unique[2], "raw_float32_vector_bytes": vectors,
            "raw_float32_cache_payload_bytes": cache,
            "passages_saved": False, "embeddings_requested": 0, "index_built": False,
        },
        "storage_scenarios": [
            {"chroma_vector_multiplier": factor,
             "active_and_replacement_plus_shared_cache_and_sources_bytes": int(
                 2 * (factor * vectors + totals[4] + totals[5]) + 1.3 * cache
                 + source_report["compressed_bytes"]
             )}
            for factor in (2, 4, 8)
        ],
        "storage_caveat": "Multipliers are assumptions, not measured Chroma/BM25 overhead or upper bounds",
        "resources": {"analysis_artifact_bytes": guard.usage(), "analysis_peak_rss_bytes": peak_rss_bytes(),
                      "additional_storage_limit_bytes": guard.max_bytes},
        "server_decision": {
            "hosting_budget_eur_per_month": 50, "recommended_ram_gib": None,
            "reason": "Full-pipeline RAM, latency and index overhead require a representative benchmark",
            "raw_vectors_only_gib": round(vectors / GIB, 3),
            "remaining_work": ["Cost-bounded embedding/index benchmark", "Compare current hosting quotes"],
        },
        "sections": [dict(zip(("section", "passages", "embedding_tokens"), row, strict=True))
                     for row in db.execute("SELECT * FROM sections ORDER BY section")],
    }


def write_readable_report(path, report):
    """Write a compact plain-text report alongside the machine-readable result."""
    match, size = report["matching"], report["sizing"]
    lines = [
        "FDA approval-matched corpus sizing", f"Status: {report['status']}",
        f"Scanned label records: {match['scanned_records']:,}",
        f"Matched label records: {match['categories'].get('matched', 0):,}",
        f"Selected newest matched label sets: {match['selected_records']:,}",
        f"Sized label sets: {size['sized_label_records']:,}", f"Passages: {size['passages']:,}",
        f"Embedding-input tokens: {size['embedding_input_tokens']:,}",
        f"Unique embedding-input tokens: {size['unique_embedding_input_tokens']:,}",
        f"Passage JSONL if written: {size['passage_jsonl_bytes_if_written'] / GIB:.3f} GiB",
        f"Raw vectors: {size['raw_float32_vector_bytes'] / GIB:.3f} GiB",
        f"Raw cache payload: {size['raw_float32_cache_payload_bytes'] / GIB:.3f} GiB",
        f"Analysis peak RAM: {report['resources']['analysis_peak_rss_bytes'] / 1024**2:.1f} MiB",
        f"Analysis artifacts: {report['resources']['analysis_artifact_bytes'] / 1024**2:.1f} MiB",
        "", "These are sizing results, not a measured full-index RAM requirement.",
        "Chroma/HNSW, BM25, runtime objects, reranker and rebuild headroom still need benchmarking.",
        "Server budget: <= EUR 50/month; select the cheapest plan that passes that benchmark.",
        "No passages, embeddings or new index were saved. No embedding API requests were made.",
    ]
    if report.get("error"):
        lines.append("Stopped: " + report["error"])
    for category, count in sorted(match["categories"].items()):
        lines.append(f"{category}: {count:,}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
