"""A small source-based section lookup comparison; no clinical relevance claims."""

from src.ingestion.coverage import record_covers_name
from src.ingestion.data_loader import load_snapshot
from src.ingestion.integrity import verify_snapshot
from src.preprocessing.normalization import normalize_record
from src.retrieval.search import SearchSession
from src.utils.config import ROOT, digest, read_json, utcnow, write_json

METHODS = ("keyword", "bm25", "dense", "hybrid_rrf", "hybrid_normalized", "hybrid_rerank")
TASKS = (
    ("Uses", "Find the uses section in a label containing {drug}.", ("indications_and_usage",)),
    ("Dosage", "Find dosage instructions in a label containing {drug}.", ("dosage_and_administration",)),
    (
        "Warnings",
        "Find warnings and precautions in a label containing {drug}.",
        ("warnings", "warnings_and_cautions", "boxed_warning"),
    ),
)


def make_questions(snapshot, names):
    """Choose references from FDA identity and section fields before running retrieval."""
    questions = []
    for i, name in enumerate(names):
        topic, template, sections = TASKS[i % len(TASKS)]
        references = []
        for envelope in snapshot["records"]:
            if envelope["kind"] != "fda_full" or not record_covers_name(envelope["record"], name):
                continue
            doc = normalize_record(envelope, snapshot["manifest"]["snapshot_id"])
            for section in sections:
                if doc["sections"].get(section):
                    references.append(
                        {
                            "unit_id": doc["document_id"] + ":" + section,
                            "set_id": doc["set_id"],
                            "source_record_hash": doc["source_record_hash"],
                            "source_body_sha256": doc["provenance"]["body_sha256"],
                            "source_url": doc["authority_url"],
                        }
                    )
        if not references:
            raise ValueError(f"No source-backed {topic.lower()} section for {name}")
        questions.append(
            {"drug": name, "topic": topic, "text": template.format(drug=name), "references": references}
        )
    return questions


def check_results(results, question):
    expected = {row["unit_id"] for row in question["references"]}
    first_rank = next(
        (i for i, row in enumerate(results[:10], 1) if row["document_id"] + ":" + row["section"] in expected),
        None,
    )
    return {
        "found_first": first_rank == 1,
        "found_in_10": first_rank is not None,
        "first_matching_rank": first_rank,
    }


def compare_methods():
    index_path = ROOT / read_json(ROOT / "results/search_latest.json")["run_path"]
    index = read_json(index_path)
    snapshot = load_snapshot(index["snapshot_id"])
    integrity = verify_snapshot(snapshot)
    questions = make_questions(snapshot, index["configuration"]["ingestion"]["lookup_names"])
    chunks = read_json(index_path.parent / "chunks.json")
    indexed_units = {row["document_id"] + ":" + row["section"] for row in chunks}
    if any(ref["unit_id"] not in indexed_units for q in questions for ref in q["references"]):
        raise ValueError("A reference section is absent from the saved index")
    source_hash = digest((ROOT / "src/evaluation/section_check.py").read_text(encoding="utf-8"))
    identifier = digest([index, questions, METHODS, source_hash])
    output = ROOT / "reports/expanded_60/method_comparison.json"
    report = read_json(output) if output.exists() else {}
    if report.get("comparison_id") != identifier:
        report = {
            "comparison_id": identifier,
            "status": "running",
            "snapshot_id": index["snapshot_id"],
            "index_run_id": index["run_id"],
            "created_at": utcnow(),
            "corpus_drug_names": len(questions),
            "corpus_labels": len({row["set_id"] for row in chunks}),
            "corpus_chunks": len(chunks),
            "reference_status": "automatic_source_identity_and_section_matches",
            "questions": questions,
            "snapshot_integrity": integrity,
            "settings": index["selection"],
            "methods": [],
            "limitations": [
                "One direct section-finding question per drug; 20 uses, 20 dosage, 20 warnings.",
                "An expected match identifies a source label and section, not a medically sufficient passage.",
                "References are automatic source checks, without independent relevance review.",
                "Existing settings were reused without tuning; all questions are descriptive checks, not an untouched test set.",
                "Timing excludes model loading and API question embedding; no clinical correctness claim follows.",
            ],
        }
    completed = {row["method"] for row in report["methods"]}
    if report.get("status") == "complete" and len(completed) == len(METHODS):
        print("Using the saved 60-drug comparison.", flush=True)
        return report
    for method in METHODS:
        if method in completed:
            continue
        print(f"Comparing {method}: {len(questions)} questions", flush=True)
        session = SearchSession(index_path, method=method)
        if method == "dense":
            session.retriever.store.embedder.encode([q["text"] for q in questions])
        session.search(questions[0]["text"], k=10)  # Warm the method before timing.
        rows = []
        for i, question in enumerate(questions, 1):
            response = session.search(question["text"], k=10)
            rows.append(
                {
                    "drug": question["drug"],
                    "topic": question["topic"],
                    **check_results(response["results"], question),
                    "latency_ms": response["latency_ms"],
                    "results": [
                        {key: row[key] for key in ("document_id", "set_id", "section", "chunk_id")}
                        for row in response["results"]
                    ],
                }
            )
            if i % 10 == 0:
                print(f"  {method}: {i}/{len(questions)}", flush=True)
        report["methods"].append(
            {
                "method": method,
                "questions": len(rows),
                "found_first": sum(row["found_first"] for row in rows),
                "found_in_10": sum(row["found_in_10"] for row in rows),
                "average_ms": sum(row["latency_ms"] for row in rows) / len(rows),
                "per_question": rows,
            }
        )
        write_json(output, report)
    report["status"] = "complete"
    report["completed_at"] = utcnow()
    report["source_code_hash"] = source_hash
    write_json(output, report)
    print("Saved the complete 60-drug comparison.", flush=True)
    return report
