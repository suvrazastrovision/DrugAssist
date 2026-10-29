"""Sixty section lookups and ten source-anchored challenges; no clinical validation."""

import re

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

# Written against the saved FDA sections before running the challenge. Quotes are evidence anchors,
# not generated answers or independent clinical relevance judgments. Equivalent wording may be missed.
HARD_TASKS = (
    {
        "drug": "prednisone", "topic": "Vocabulary",
        "text": "My body has become used to long-term prednisone. Find the label evidence explaining "
                "why suddenly quitting is unsafe, and what someone should tell medical staff and do "
                "when a new illness with fever develops.",
        "requirements": (
            ("Avoid abrupt stopping without supervision", "information_for_patients",
             ("Patients should be warned not to discontinue the use of corticosteroids abruptly "
             "or without medical supervision.")),
            (
                "Adrenal dependence, informing staff and seeking advice for acute illness",
                "information_for_patients",
                (
                    "As prolonged use may cause adrenal insufficiency and make patients dependent "
                    "on corticosteroids, they should advise any medical attendants that they are "
                    "taking corticosteroids and they should seek medical advice at once should "
                    "they develop an acute illness including fever or other signs of infection."
                ),
            ),
        ),
    },
    {
        "drug": "gabapentin", "topic": "Vocabulary",
        "text": "For gabapentin, find instructions for someone whose kidneys remove medicine very slowly "
                "and whose blood is cleaned by a machine. Include both the very-low-kidney-function "
                "daily-dose rule and the extra dosing associated with a four-hour filtering session.",
        "requirements": (
            ("Proportional reduction below creatinine clearance 15", "dosage_and_administration",
             ("For patients with creatinine clearance <15 mL/min, reduce daily dose in proportion to "
             "creatinine clearance (e.g., patients with a creatinine clearance of 7.5 mL/min should receive "
             "one-half the daily dose that patients with a creatinine clearance of 15 mL/min receive).")),
            (
                "Maintenance and supplemental post-hemodialysis dosing",
                "dosage_and_administration",
                (
                    "Patients on hemodialysis should receive maintenance doses based on estimates "
                    "of creatinine clearance as indicated in the upper portion of the table and a "
                    "supplemental post-hemodialysis dose administered after each 4 hours of "
                    "hemodialysis as indicated in the lower portion of the table."
                ),
            ),
        ),
    },
    {
        "drug": "levothyroxine", "topic": "Vocabulary",
        "text": "For a morning levothyroxine tablet, find the timing rules around breakfast, calcium or "
                "iron supplements, and a hair-and-nail vitamin that can interfere with thyroid blood tests. "
                "Retrieve all three instructions rather than just the thyroid condition it treats.",
        "requirements": (
            ("Empty stomach before breakfast", "dosage_and_administration",
             ("Administer once daily, preferably on an empty stomach, one-half to one hour before "
             "breakfast with a full glass of water.")),
            ("Separate calcium and iron binders by four hours", "drug_interactions",
             ("Phosphate Binders (e.g., calcium carbonate, ferrous sulfate, sevelamer, lanthanum) "
             "Phosphate binders may bind to levothyroxine. Administer levothyroxine sodium tablets "
             "at least 4 hours apart from these agents.")),
            ("Stop biotin before thyroid laboratory assessment", "dosage_and_administration",
             ("Advise patients to stop biotin and biotin-containing supplements at least 2 days "
             "before assessing TSH and/or T4 levels.")),
        ),
    },
    {
        "drug": "apixaban", "topic": "Constraints",
        "text": "Find apixaban evidence distinguishing the reduced dose for nonvalvular atrial "
                "fibrillation when age, weight and creatinine criteria coexist, from the interaction "
                "rule for someone already taking 2.5 mg twice daily who also receives a combined "
                "P-gp and strong CYP3A4 inhibitor.",
        "requirements": (
            ("At least two age, weight and creatinine criteria", "dosage_and_administration",
             ("In patients with at least 2 of the following characteristics: age greater than or equal to "
             "80 years, body weight less than or equal to 60 kg, or serum creatinine greater than or equal "
             "to 1.5 mg/dL, the recommended dose is 2.5 mg orally twice daily.")),
            ("Avoid combined inhibitors when already taking the lower dose", "drug_interactions", (
                ("For patients receiving apixaban at a dose of 2.5 mg twice daily, avoid coadministration "
                "with combined P-gp and strong CYP3A4 inhibitors"),
                ("For patients receiving ELIQUIS at a dose of 2.5 mg twice daily, avoid coadministration "
                "with combined P-gp and strong CYP3A4 inhibitors"),
            )),
        ),
    },
    {
        "drug": "metformin", "topic": "Constraints",
        "text": "In immediate-release metformin labeling, distinguish starting treatment with an eGFR "
                "of 38 from managing an existing prescription around iodinated contrast. Find the "
                "renal initiation limit, circumstances requiring a temporary hold, and kidney-check "
                "and restart conditions after imaging.",
        "requirements": (
            ("Do not initiate at eGFR 30 to 45", "dosage_and_administration",
             ("Initiation of metformin hydrochloride tablets in patients with an eGFR between "
             "30 – 45 mL/minute/1.73 m 2 is not recommended.")),
            ("Contrast-related hold and qualifying conditions", "dosage_and_administration",
             ("Discontinue metformin hydrochloride tablets at the time of, or prior to, an iodinated "
             "contrast imaging procedure in patients with an eGFR between 30 and 60 mL/min/1.73 m 2 ; "
             "in patients with a history of liver disease, alcoholism, or heart failure; or in patients "
             "who will be administered intra-arterial iodinated contrast.")),
            ("Recheck after 48 hours and restart only if stable", "dosage_and_administration",
             ("Re-evaluate eGFR 48 hours after the imaging procedure; restart metformin hydrochloride "
             "tablets if renal function is stable.")),
        ),
    },
    {
        "drug": "sertraline", "topic": "Constraints",
        "text": "Locate sertraline label evidence that distinguishes mild from moderate or severe "
                "liver impairment and gives the waiting periods for switching to and from an MAOI "
                "antidepressant. Include both switching directions, not only a general serotonin "
                "syndrome warning.",
        "requirements": (
            ("Half dosage for mild hepatic impairment", "dosage_and_administration",
             ("Both the recommended starting dosage and therapeutic range in patients with mild "
             "hepatic impairment (Child Pugh scores 5 or 6) are half the recommended daily dosage")),
            ("Moderate and severe hepatic impairment not recommended", "dosage_and_administration",
             ("The use of sertraline hydrochloride tablets in patients with moderate (Child Pugh scores "
             "7 to 9) or severe hepatic impairment (Child Pugh scores 10 to 15) is not recommended")),
            ("Fourteen-day interval in both MAOI switching directions", "dosage_and_administration",
             ("At least 14 days must elapse between discontinuation of a monoamine oxidase inhibitor "
             "(MAOI) antidepressant and initiation of sertraline hydrochloride tablets. In addition, "
             "at least 14 days must elapse after stopping sertraline hydrochloride tablets before "
             "starting an MAOI antidepressant")),
        ),
    },
    {
        "drug": "warfarin", "topic": "Multi-section",
        "text": "Find warfarin evidence explaining why a therapeutic INR does not guarantee freedom "
                "from bleeding, together with the monitoring instruction when antibiotics are started "
                "or stopped. Retrieve both the residual-risk statement and the drug-change action, "
                "rather than a generic bleeding side-effect list.",
        "requirements": (
            ("Therapeutic INR does not eliminate bleeding risk", "warnings_and_cautions",
             "However, maintenance of INR in the therapeutic range does not eliminate the risk of bleeding."),
            ("More frequent monitoring with short-term interacting medicines", "drug_interactions",
             ("More frequent INR monitoring should be performed when starting or stopping other drugs, "
             "including botanicals, or when changing dosages of other drugs, including drugs intended "
             "for short-term use (e.g., antibiotics, antifungals, corticosteroids)")),
        ),
    },
    {
        "drug": "atorvastatin", "topic": "Multi-section",
        "text": "Find atorvastatin evidence linking clarithromycin to a dose ceiling and describing "
                "the muscle-safety action when CK is markedly raised or myopathy is suspected. "
                "Retrieve the interaction restriction and the warning action, not cholesterol-lowering "
                "indications or a general list of muscle symptoms.",
        "requirements": (
            ("Clarithromycin-associated 20 mg ceiling", "drug_interactions",
             "In patients taking clarithromycin or itraconazole, do not exceed atorvastatin 20 mg"),
            ("Discontinue with markedly elevated CK or suspected myopathy", "warnings_and_cautions",
             ("Discontinue atorvastatin calcium if markedly elevated CK levels occur or myopathy "
             "is diagnosed or suspected.")),
        ),
    },
    {
        "drug": "clopidogrel", "topic": "Near-match",
        "text": "For clopidogrel and acid-suppressing medicines, locate evidence addressing whether "
                "taking omeprazole twelve hours apart removes the interaction, and whether pantoprazole "
                "has the same effect. Retrieve the avoidance instruction, the separated-dose finding "
                "and the comparison with other proton-pump inhibitors.",
        "requirements": (
            ("Avoid omeprazole or esomeprazole", "drug_interactions",
             "Avoid concomitant use of clopidogrel with omeprazole or esomeprazole."),
            ("Reduced activity even twelve hours apart", "drug_interactions",
             ("In clinical studies, omeprazole was shown to reduce significantly the antiplatelet activity "
             "of clopidogrel when given concomitantly or 12 hours apart.")),
            ("Other named PPIs had less effect, not an absence-of-risk claim", "drug_interactions",
             ("Dexlansoprazole, lansoprazole, and pantoprazole had less effect on the antiplatelet activity "
             "of clopidogrel than did omeprazole or esomeprazole")),
        ),
    },
    {
        "drug": "ibuprofen", "topic": "Near-match",
        "text": "For the acetaminophen plus ibuprofen dual-action caplets, locate directions for "
                "people aged twelve and over, the maximum in one day, and the acetaminophen-related "
                "warning about taking another product with the same ingredient. Exclude instructions "
                "for plain ibuprofen tablets.",
        "requirements": (
            ("Combination-product age, caplet count and interval", "dosage_and_administration",
             "Adults and children 12 years and over: take 2 caplets every 8 hours while symptoms persist"),
            ("Combination-product maximum daily amount", "dosage_and_administration",
             "do not take more than 6 caplets in 24 hours, unless directed by a doctor"),
            ("Acetaminophen duplication and liver damage warning", "warnings",
             ("Acetaminophen liver damage warning: This product contains acetaminophen. Severe liver "
             "damage may occur if you take: • with other drugs containing acetaminophen")),
        ),
    },
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
            {"id": f"baseline_{i + 1:02d}", "group": "baseline", "drug": name, "topic": topic,
             "text": template.format(drug=name), "references": references}
        )
    return questions


def make_hard_questions(snapshot):
    """Locate every exact occurrence in identity-matched labels; fail if evidence is missing."""
    questions = []
    for i, task in enumerate(HARD_TASKS, 1):
        question = {key: task[key] for key in ("drug", "topic", "text")}
        question.update(id=f"hard_{i:02d}", group="hard", requirements=[], references=[])
        for j, (description, section, quotes) in enumerate(task["requirements"], 1):
            requirement_id = f"hard_{i:02d}_r{j}"
            question["requirements"].append({"id": requirement_id, "description": description})
            found = []
            for envelope in snapshot["records"]:
                if envelope["kind"] != "fda_full" or not record_covers_name(envelope["record"], task["drug"]):
                    continue
                doc = normalize_record(envelope, snapshot["manifest"]["snapshot_id"])
                text = doc["sections"].get(section, "")
                for quote in (quotes,) if isinstance(quotes, str) else quotes:
                    for match in re.finditer(re.escape(quote), text):
                        found.append({
                            "requirement_id": requirement_id,
                            "unit_id": doc["document_id"] + ":" + section,
                            "document_id": doc["document_id"], "section": section,
                            "set_id": doc["set_id"], "start": match.start(), "end": match.end(),
                            "quote": quote, "source_record_hash": doc["source_record_hash"],
                            "source_body_sha256": doc["provenance"]["body_sha256"],
                            "source_url": doc["authority_url"],
                        })
            if not found:
                raise ValueError(f"Missing saved evidence for {requirement_id}: {description}")
            question["references"].extend(found)
        questions.append(question)
    return questions


def covers_reference(results, reference):
    """Allow contiguous coverage across overlapping chunks, but never bridge a missing-text gap."""
    intervals = sorted(
        (row["start"], row["end"]) for row in results
        if row["document_id"] == reference["document_id"] and row["section"] == reference["section"]
    )
    cursor = reference["start"]
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= reference["end"]:
            return True
    return False


def check_results(results, question):
    """Measure section hits or complete evidence-span coverage for one question."""
    if question.get("group") == "hard":
        first_ranks = {}
        for rank in range(1, min(10, len(results)) + 1):
            for reference in question["references"]:
                required = reference["requirement_id"]
                if required not in first_ranks and covers_reference(results[:rank], reference):
                    first_ranks[required] = rank
        first = min(first_ranks.values(), default=None)
        total = len(question["requirements"])
        return {
            "found_first": first == 1, "found_in_10": first is not None, "first_matching_rank": first,
            "requirements_found": len(first_ranks), "requirements_total": total,
            "evidence_coverage": len(first_ranks) / total,
            "all_evidence_found": len(first_ranks) == total,
            "matched_requirements": sorted(first_ranks),
        }
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


def summarize(rows):
    """Aggregate hit counts, latency, and applicable hard-question coverage metrics."""
    summary = {
        "questions": len(rows), "found_first": sum(row["found_first"] for row in rows),
        "found_in_10": sum(row["found_in_10"] for row in rows),
        "average_ms": sum(row["latency_ms"] for row in rows) / len(rows),
    }
    if rows[0]["group"] == "hard":
        summary.update(
            evidence_coverage=sum(row["evidence_coverage"] for row in rows) / len(rows),
            complete_in_10=sum(row["all_evidence_found"] for row in rows),
        )
    return summary


def compare_methods(chroma_path=None):
    """Evaluate saved retrieval methods against baseline and hard source references."""
    index_path = ROOT / read_json(ROOT / "results/search_latest.json")["run_path"].replace("\\", "/")
    index = read_json(index_path)
    snapshot = load_snapshot(index["snapshot_id"])
    integrity = verify_snapshot(snapshot)
    names = index["configuration"]["ingestion"]["lookup_names"]
    questions = make_questions(snapshot, names) + make_hard_questions(snapshot)
    chunks = read_json(index_path.parent / "chunks.json")
    indexed_units = {row["document_id"] + ":" + row["section"] for row in chunks}
    if any(ref["unit_id"] not in indexed_units for q in questions for ref in q["references"]):
        raise ValueError("A reference section is absent from the saved index")
    if any(not covers_reference(chunks, ref) for q in questions if q["group"] == "hard"
           for ref in q["references"]):
        raise ValueError("A hard reference span is not completely represented in the passage index")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
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
            "corpus_drug_names": len(names),
            "question_groups": {"baseline": len(names), "hard": len(HARD_TASKS)},
            "corpus_labels": len({row["set_id"] for row in chunks}),
            "corpus_chunks": len(chunks),
            "reference_status": "baseline_automatic_sections; hard_repository_reviewed_exact_source_spans",
            "questions": questions,
            "snapshot_integrity": integrity,
            "settings": index["selection"],
            "methods": [],
            "limitations": [
                "Baseline: 60 original section-finding questions; 20 uses, 20 dosage, 20 warnings.",
                "Challenge: 10 authored questions and 25 required evidence items, frozen before retrieval.",
                "Baseline hits identify a drug section, not a medically sufficient passage.",
                (
                    "Hard hits require full coverage of selected source spans; duplicate chunks "
                    "earn no extra credit."
                ),
                (
                    "Exact spans are repository-reviewed anchors, not exhaustive relevance "
                    "judgments or independent clinical review."
                ),
                "Equivalent wording or other valid evidence can be missed by this strict span-based scorer.",
                (
                    "Designed difficulty is not independently validated; ten questions are a small "
                    "diagnostic stress test."
                ),
                (
                    "Existing settings were reused without tuning; all questions are descriptive "
                    "checks, not an untouched test set."
                ),
                (
                    "Timing excludes model loading and API question embedding; no clinical "
                    "correctness claim follows."
                ),
                (
                    "All methods use unfiltered top-10 retrieval without the app parser, score "
                    "cutoff or answer writer."
                ),
            ],
        }
    completed = {row["method"] for row in report["methods"]}
    if report.get("status") == "complete" and len(completed) == len(METHODS):
        print("Using the saved 60-baseline / 10-hard comparison.", flush=True)
        return report
    for method in METHODS:
        if method in completed:
            continue
        print(f"Comparing {method}: {len(questions)} questions", flush=True)
        session = SearchSession(index_path, method=method, chroma_path=chroma_path)
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
                    "id": question["id"], "group": question["group"],
                    **check_results(response["results"], question),
                    "latency_ms": response["latency_ms"],
                    "results": [
                        {key: row[key] for key in (
                            "rank", "score", "document_id", "set_id", "section", "chunk_id", "start", "end",
                        )}
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
                "groups": {group: summarize([row for row in rows if row["group"] == group])
                           for group in ("baseline", "hard")},
                "per_question": rows,
            }
        )
        write_json(output, report)
        if hasattr(session.embedder, "close"):
            session.embedder.close()
    report["status"] = "complete"
    report["completed_at"] = utcnow()
    report["source_code_hash"] = source_hash
    write_json(output, report)
    print("Saved the complete 60-baseline / 10-hard comparison.", flush=True)
    return report
