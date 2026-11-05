"""Deterministic app-output grading; citation validity is distinct from factual support."""

from pydantic import ValidationError

from src.api.schemas import AnswerResponse, SearchResponse
from src.evaluation.section_check import covers_reference
from src.services.answer_service import ANSWER_SECTIONS
from src.utils.config import digest

GRADER_VERSION = "app-autograder-v1"


def traceable(row, documents, snapshot_id):
    """Match a passage/window against the exact saved document version and text."""
    doc = documents.get(row["document_id"])
    if doc is None:
        return False
    text = doc["sections"].get(row["section"], "")
    return (
        doc["kind"] == "fda_full"
        and row["snapshot_id"] == doc["snapshot_id"] == snapshot_id
        and row["set_id"] == doc["set_id"]
        and row["label_id"] == doc["label_id"]
        and row["effective_time"] == doc["effective_time"]
        and row["source_record_hash"] == doc["source_record_hash"]
        and row["source_body_sha256"] == doc["provenance"]["body_sha256"]
        and row["source_url"] == doc["authority_url"]
        and 0 <= row["start"] < row["end"] <= len(text)
        and text[row["start"]:row["end"]] == row["text"]
    )


def coverage(rows, requirements):
    """Credit a requirement once if any full acceptable source span is covered."""
    found = [item.id for item in requirements
             if any(covers_reference(rows, ref.model_dump()) for ref in item.references)]
    return {
        "required": len(requirements), "found": len(found), "matched": found,
        "coverage": len(found) / len(requirements) if requirements else None,
    }


def grade_response(case, response, documents, snapshot_id, mode, k):
    """Grade real filtered results and answer outputs; never infer semantic correctness."""
    checks = []

    def check(name, passed, reason):
        checks.append({"name": name, "passed": bool(passed), "reason": reason})

    schema = AnswerResponse if mode == "answer" else SearchResponse
    try:
        body = schema.model_validate(response).model_dump(mode="json")
    except (ValidationError, TypeError):
        check("response_schema", False, "Response does not match the public app schema.")
        return {"checks": checks, "retrieval": None, "cited_evidence": None,
                "status": "failed", "semantic": {"status": "not_run"}}
    check("response_schema", True, "Public response schema is valid.")
    check("request_identity", body["query"] == case.question and body["snapshot_id"] == snapshot_id,
          "Question and source snapshot must match the evaluated case.")
    hits = body["results"]
    check("result_contract", len(hits) <= k and body["no_match"] == (not hits)
          and [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1)),
          "Results must respect k, ranks and no-match status.")
    valid_hits = [hit for hit in hits if traceable(hit, documents, snapshot_id)]
    check("retrieval_traceability", len(valid_hits) == len(hits),
          "Every retrieved passage must match the saved source text, version and provenance.")
    retrieval = coverage(valid_hits, case.requirements)
    retrieval["first_matching_rank"] = next(
        (rank for rank in range(1, len(valid_hits) + 1)
         if coverage(valid_hits[:rank], case.requirements)["found"]), None,
    )
    if case.requirements:
        check("required_evidence", retrieval["coverage"] == 1,
              "All authored evidence requirements must be represented within the app's returned k passages.")
    elif case.category in {"unknown_drug", "out_of_scope"}:
        check("negative_retrieval", body["no_match"], "Unsupported drugs/topics should return no match.")
    cited = None
    semantic = {"status": "not_applicable"} if mode == "search" else {"status": "not_run"}
    error = False
    if mode == "answer":
        citations = body["citations"]
        valid_citations = [row for row in citations if traceable(row, documents, snapshot_id)]
        check("citation_traceability", len(valid_citations) == len(citations),
              "Citation windows must match exact saved source text and provenance.")
        ids = [row["source_id"] for row in citations]
        used = {source for statement in body["statements"] for source in statement["source_ids"]}
        check("citation_integrity", len(set(ids)) == len(ids) and set(ids) == used
              and all(source.strip() for source in ids)
              and all(statement["text"].strip() and statement["source_ids"]
                      for statement in body["statements"]),
              "Each statement needs valid citation IDs; cited windows must be unique and used.")
        check("audience", body["audience"] == case.audience, "The requested audience must be preserved.")
        generated = body["answer_status"] == "generated"
        abstained = body["answer_status"] == "insufficient_evidence"
        error = body["answer_status"] == "unavailable"
        consistent = (
            generated and body["insufficient_evidence"] is False and bool(body["answer"])
            and bool(body["statements"]) and sum(s["section"] == "answer" for s in body["statements"]) == 1
            or abstained and body["insufficient_evidence"] is True and body["answer"] is None
            and not body["statements"] and not citations and bool(body["evidence_note"])
            or error and body["insufficient_evidence"] is None and body["answer"] is None
            and not body["statements"] and not citations
        )
        check("answer_contract", consistent, "Answer content and evidence status must agree.")
        if generated:
            paragraphs = []
            for section, title in ANSWER_SECTIONS.items():
                statements = [item for item in body["statements"] if item["section"] == section]
                if statements:
                    paragraphs.append(title + "\n" + "\n".join(
                        item["text"] + " " + " ".join(f"[{source}]"
                                                       for source in dict.fromkeys(item["source_ids"]))
                        for item in statements
                    ))
            check("rendered_answer", body["answer"] == "\n\n".join(paragraphs),
                  "The displayed answer must contain exactly its graded, cited statements.")
        check("expected_outcome", generated if case.expected_outcome == "answer" else abstained,
              "Expected a supported answer." if case.requirements else "Expected explicit abstention.")
        cited = coverage(valid_citations, case.requirements)
    failed = any(not item["passed"] for item in checks)
    status = "error" if error else "failed" if failed else "passed"
    if status == "passed" and mode == "answer" and case.expected_outcome == "answer":
        status = "review_required"
    return {"checks": checks, "retrieval": retrieval, "cited_evidence": cited,
            "status": status, "semantic": semantic}


def apply_judgment(grade, judgment):
    """Keep service errors, deterministic failures and semantic failures distinct."""
    grade["semantic"] = {"status": "graded", **judgment}
    result = judgment["result"]
    passed = (
        result["relevant"] and result["appropriate_abstention"] and not result["critical_errors"]
        and all(item["supported"] for item in result["statements"])
        and all(item["satisfied"] for item in result["requirements"])
    )
    if grade["status"] != "error":
        grade["status"] = "passed" if passed and all(c["passed"] for c in grade["checks"]) else "failed"
    return grade


def summarize(rows):
    """Keep retrieval coverage, answer judgments, failures and pending reviews separate."""
    counts = {status: sum(row["grade"]["status"] == status for row in rows)
              for status in ("passed", "failed", "error", "review_required")}
    positives = [row for row in rows if row["expected_outcome"] == "answer"]
    retrieval = [row["grade"].get("retrieval") or {} for row in positives]
    values = [item.get("coverage") or 0 for item in retrieval]

    def tokens(usage, name):
        value = usage.get(name, 0) if isinstance(usage, dict) else 0
        return value if isinstance(value, int) and value >= 0 else 0

    usage = {
        "captured_embedding_tokens": sum(tokens(row.get("usage"), "new_embedding_tokens") for row in rows),
        "captured_answer_input_tokens": sum(tokens((row.get("usage") or {}).get("answer"), "input_tokens")
                                            for row in rows),
        "captured_answer_output_tokens": sum(tokens((row.get("usage") or {}).get("answer"), "output_tokens")
                                             for row in rows),
    }
    for name in ("input_tokens", "output_tokens"):
        usage["new_judge_" + name] = sum(
            tokens(row["grade"]["semantic"].get("usage"), name) for row in rows
            if not row["grade"]["semantic"].get("cache_hit", False)
        )
    return {
        "cases": len(rows), **counts,
        "answerable_cases": len(positives),
        "mean_evidence_coverage": sum(values) / len(values) if values else None,
        "evidence_hit_rate_at_k": sum(value > 0 for value in values) / len(values) if values else None,
        "complete_evidence_rate_at_k": sum(value == 1 for value in values) / len(values) if values else None,
        "mean_reciprocal_rank": sum(1 / item["first_matching_rank"]
                                    for item in retrieval if item.get("first_matching_rank"))
                                / len(values) if values else None,
        "usage": usage,
        "semantic_graded": sum(row["grade"]["semantic"]["status"] == "graded" for row in rows),
        "critical_errors": sum(len(row["grade"]["semantic"].get("result", {}).get("critical_errors", []))
                               for row in rows),
        "average_ms": sum(row["elapsed_ms"] for row in rows) / len(rows) if rows else None,
    }


def compare_reports(baseline, current):
    """Compare identical cases and grading rules; reject changes that confound the comparison."""
    for field in ("dataset_hash", "snapshot_id", "mode", "k", "case_ids", "grading_hash", "judge_config"):
        if baseline[field] != current[field]:
            raise ValueError(f"Reports are not comparable: {field} differs")
    if baseline["status"] != "complete" or current["status"] != "complete":
        raise ValueError("Only complete runs can be compared")
    before = {row["case_id"]: row for row in baseline["cases"]}
    after = {row["case_id"]: row for row in current["cases"]}
    if (set(before) != set(baseline["case_ids"]) or set(after) != set(before)
            or len(before) != len(baseline["cases"]) or len(after) != len(current["cases"])):
        raise ValueError("Report cases are incomplete or duplicated")

    def semantic_scores(result):
        statements, requirements = result["statements"], result["requirements"]
        return (
            sum(item["supported"] for item in statements) / len(statements) if statements else 1,
            sum(item["satisfied"] for item in requirements) / len(requirements) if requirements else 1,
            int(result["relevant"]), int(result["appropriate_abstention"]),
        )

    regressions, improvements, changed = [], [], []
    for case_id in before:
        old, new = before[case_id]["grade"], after[case_id]["grade"]
        semantic_dropped = False
        if old["semantic"]["status"] == new["semantic"]["status"] == "graded":
            if old["semantic"]["model"] != new["semantic"]["model"]:
                raise ValueError("Reports are not comparable: resolved judge models differ")
            semantic_dropped = any(
                new_value < old_value for old_value, new_value in zip(
                    semantic_scores(old["semantic"]["result"]),
                    semantic_scores(new["semantic"]["result"]), strict=True,
                )
            )
        old_coverage = (old.get("retrieval") or {}).get("coverage")
        new_coverage = (new.get("retrieval") or {}).get("coverage")
        coverage_dropped = (old_coverage is not None and new_coverage is not None
                            and new_coverage < old_coverage)
        old_failures = {item["name"] for item in old["checks"] if not item["passed"]}
        new_failures = {item["name"] for item in new["checks"] if not item["passed"]}
        critical_increased = (
            len(new["semantic"].get("result", {}).get("critical_errors", []))
            > len(old["semantic"].get("result", {}).get("critical_errors", []))
        )
        if (old["status"] == "passed" and new["status"] != "passed" or coverage_dropped
                or new_failures - old_failures or critical_increased or semantic_dropped):
            regressions.append(case_id)
        if old["status"] in {"failed", "error"} and new["status"] == "passed":
            improvements.append(case_id)
        if digest(old) != digest(new):
            changed.append(case_id)
    return {"regressions": regressions, "improvements": improvements, "changed_cases": changed,
            "baseline": summarize(baseline["cases"]), "current": summarize(current["cases"])}
