"""Bind curated query intents to frozen FDA evidence without running a retriever.

Rule-located spans are draft annotations. Independent human review is required
before a benchmark can be labeled finalized. No model-generated judgments.
"""

import re
from collections import Counter

from src.utils.config import ROOT, digest, read_json, write_json

DIFFICULTY = {
    "easy": "Explicit drug identity, direct wording, one evidence section.",
    "medium": "Paraphrase, abbreviation, lexical variation or multiple attributes within a topic.",
    "hard": "Safety qualifiers, multiple constraints or evidence across sections; no clinical inference.",
}


def make_benchmark(snapshot, specs_path=None):
    specs = read_json(specs_path or ROOT / "data/benchmarks/query_specs.json")
    documents = [d for d in snapshot["documents"] if d["kind"] == "fda_full"]
    queries = []
    dev_drugs = {"aspirin", "metformin", "sertraline", "levothyroxine"}
    for i, spec in enumerate(specs, 1):
        query = {
            "query_id": f"Q{i:03d}",
            **spec,
            "partition": "dev" if spec["drug"] in dev_drugs or spec.get("dev") else "test",
            "judgment_status": "draft_rule_located",
            "reviewer": None,
            "annotation_method": "curated intent; deterministic FDA evidence lookup; no retrieval predictions",
            "qrels": {},
            "evidence": [],
            "judgments_exhaustive": False,
        }
        if spec["answerable"]:
            for doc in documents:
                identity = " ".join(value for values in doc["metadata"].values() for value in values)
                # Do not judge combination products as single-ingredient equivalents automatically.
                generic = " ".join(doc["metadata"]["generic_name"])
                if not re.search(r"\b" + re.escape(spec["drug"]) + r"\b", identity, re.IGNORECASE):
                    continue
                if "," in generic or " AND " in generic:
                    continue
                for target in spec["targets"]:
                    for section in target["sections"]:
                        text = doc["sections"].get(section, "")
                        matches = [re.search(pattern, text, re.IGNORECASE) for pattern in target["patterns"]]
                        if not matches or not all(matches):
                            continue
                        unit = doc["document_id"] + ":" + section
                        query["qrels"][unit] = 2
                        for match in matches:
                            start, end = max(0, match.start() - 90), min(len(text), match.end() + 180)
                            query["evidence"].append(
                                {
                                    "unit_id": unit,
                                    "document_id": doc["document_id"],
                                    "set_id": doc["set_id"],
                                    "section": section,
                                    "start": start,
                                    "end": end,
                                    "text": text[start:end],
                                    "source_record_hash": doc["source_record_hash"],
                                    "source_body_sha256": doc["provenance"]["body_sha256"],
                                    "source_url": doc["authority_url"],
                                    "effective_time": doc["effective_time"],
                                }
                            )
            if not query["qrels"]:
                raise ValueError(f"No FDA evidence for {query['query_id']}: {query['text']}")
        queries.append(query)
    benchmark = {
        "version": "0.1-draft",
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "corpus_document_ids": sorted(d["document_id"] for d in documents),
        "difficulty_definitions": DIFFICULTY,
        "status": "requires_independent_human_review",
        "evaluation_unit": "document-section; results deduplicated by document and section",
        "limitations": [
            "Draft span rules establish source presence, not independent expert relevance.",
            "Qrels are incomplete: unjudged sections are treated as nonrelevant for provisional scoring.",
            "Section-level scores do not establish that a retrieved chunk contains the answer.",
            "Brand/generic variation is limited to aliases actually present; no unverified ontology maps.",
        ],
        "queries": queries,
    }
    benchmark["benchmark_id"] = digest(benchmark)[:20]
    validate_benchmark(benchmark, documents, require_review=False)
    write_json(ROOT / "data/benchmarks/benchmark.json", benchmark)
    return benchmark


def validate_benchmark(benchmark, documents, require_review=True):
    queries = benchmark["queries"]
    if len(queries) != 80 or Counter(q["difficulty"] for q in queries) != {
        "easy": 25,
        "medium": 30,
        "hard": 25,
    }:
        raise ValueError("Benchmark must contain exactly 25 easy, 30 medium, 25 hard queries")
    if len({q["query_id"] for q in queries}) != 80 or len({q["text"] for q in queries}) != 80:
        raise ValueError("Duplicate benchmark query IDs or text")
    if not any(not q["answerable"] for q in queries):
        raise ValueError("Unsupported queries required")
    docmap = {d["document_id"]: d for d in documents}
    if set(benchmark["corpus_document_ids"]) != set(docmap):
        raise ValueError("Benchmark corpus mismatch")
    for query in queries:
        if query["partition"] not in {"dev", "test"}:
            raise ValueError("Invalid partition")
        if any(grade > 0 for grade in query["qrels"].values()) != query["answerable"]:
            raise ValueError("Answerability/qrels mismatch")
        if any(grade not in (0, 1, 2) for grade in query["qrels"].values()):
            raise ValueError("Judgment grades must be 0, 1, or 2")
        for unit in query["qrels"]:
            document_id, section = unit.split(":", 1)
            if document_id not in docmap or section not in docmap[document_id]["sections"]:
                raise ValueError("Judgment refers to an absent source document-section")
        positive_units = {key for key, grade in query["qrels"].items() if grade > 0}
        if not positive_units.issubset({e["unit_id"] for e in query["evidence"]}):
            raise ValueError("Every positive relevance judgment needs source evidence")
        for evidence in query["evidence"]:
            doc = docmap[evidence["document_id"]]
            text = doc["sections"][evidence["section"]]
            if not 0 <= evidence["start"] < evidence["end"] <= len(text):
                raise ValueError("Evidence offsets fall outside source section")
            if evidence["unit_id"] != evidence["document_id"] + ":" + evidence["section"]:
                raise ValueError("Evidence unit identity mismatch")
            if text[evidence["start"] : evidence["end"]] != evidence["text"]:
                raise ValueError("Evidence span no longer matches frozen source")
            if doc["source_record_hash"] != evidence["source_record_hash"]:
                raise ValueError("Evidence source hash mismatch")
            if (
                evidence["source_body_sha256"] != doc["provenance"]["body_sha256"]
                or evidence["set_id"] != doc["set_id"]
            ):
                raise ValueError("Evidence source provenance mismatch")
        if require_review and (
            query["judgment_status"] != "reviewed"
            or not query.get("reviewer")
            or not query.get("reviewed_at")
            or not query.get("review_notes")
        ):
            raise ValueError(
                "Independent human relevance review is required; use --provisional only for draft experiments"
            )
    if require_review and benchmark.get("status") != "reviewed":
        raise ValueError("Benchmark status must be reviewed after adjudication")
    drugs_dev = {q["drug"] for q in queries if q["partition"] == "dev" and q["answerable"]}
    drugs_test = {q["drug"] for q in queries if q["partition"] == "test" and q["answerable"]}
    if drugs_dev & drugs_test:
        raise ValueError("Drug groups overlap development/test partitions")
    return True


def evidence_coverage(results, query, k):
    evidence = query["evidence"]
    if not evidence:
        return None
    covered = sum(
        any(
            r.chunk.document_id == e["document_id"]
            and r.chunk.section == e["section"]
            and r.chunk.start <= e["start"]
            and r.chunk.end >= e["end"]
            for r in results[:k]
        )
        for e in evidence
    )
    return covered / len(evidence)
