"""Evaluate the real app services, regrade saved responses, and compare frozen runs."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from fastapi.encoders import jsonable_encoder

from src.evaluation.dataset import EvalDataset, starter_dataset, verify_dataset
from src.evaluation.grading import apply_judgment, compare_reports, grade_response, summarize
from src.evaluation.judge import OpenAIJudge
from src.ingestion.data_loader import load_snapshot
from src.ingestion.integrity import verify_snapshot
from src.preprocessing.normalization import normalize_record
from src.utils.config import CODE_ROOT, ROOT, digest, read_json, utcnow, write_json

DEFAULT_DIRECTORY = ROOT / "results/private/evaluation"


class EvaluationInputError(ValueError):
    """A safe, actionable input error that does not include provider or credential data."""


def implementation_hash(folder):
    """Record source identity without inspecting credentials, caches or user data."""
    return digest({str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(folder.rglob("*.py"))})


def load_sources(snapshot_id):
    """Verify raw bodies and normalized source lineage before using gold evidence."""
    snapshot = load_snapshot(snapshot_id)
    verify_snapshot(snapshot)
    expected = {doc["document_id"]: doc for envelope in snapshot["records"]
                for doc in [normalize_record(envelope, snapshot_id)]}
    if any(doc != expected.get(doc["document_id"]) for doc in snapshot["documents"]):
        raise ValueError("Normalized documents differ from the frozen source records")
    return snapshot


def evaluate(dataset, documents, collect, *, mode="search", k=5, case_ids=None,
             judge=None, metadata=None, output=None):
    """Collect or replay bounded cases, saving each completed case before continuing.

    collect(case) returns a captured record containing response, elapsed_ms, usage,
    and optionally a redacted error code. No app/model is initialized by this function.
    """
    verify_dataset(dataset, documents)
    if mode not in {"search", "answer"} or not 1 <= k <= 20:
        raise ValueError("Evaluation requires search/answer mode and k between 1 and 20")
    if judge and mode != "answer":
        raise ValueError("Semantic judging requires answer mode")
    selected = case_ids if case_ids is not None else [case.id for case in dataset.cases]
    cases = {case.id: case for case in dataset.cases}
    if not selected or len(set(selected)) != len(selected) or any(item not in cases for item in selected):
        raise ValueError("Select unique, existing case identifiers")
    docs = {doc["document_id"]: doc for doc in documents}
    grading_sources = ("dataset.py", "grading.py", "judge.py", "runner.py")
    report = {
        "schema_version": 1, "created_at": utcnow(), "status": "running",
        "dataset_hash": dataset.fingerprint, "dataset_review_status": dataset.review_status,
        "snapshot_id": dataset.snapshot_id, "mode": mode, "k": k, "case_ids": selected,
        "grading_hash": digest({name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in grading_sources}),
        "judge_config": judge.configuration if judge else None,
        "implementation_hash": implementation_hash(CODE_ROOT / "src"),
        "metadata": metadata or {}, "cases": [],
        "limitations": [
            dataset.notes,
            "Diagnostic scores do not establish medical correctness or clinical suitability.",
            "Exact evidence spans are not exhaustive judgments of all valid supporting evidence.",
            "A model judge can be wrong; calibrate against human review before using it as a release gate.",
            "Without a semantic judge, generated answers remain review_required "
            "even if structural checks pass.",
        ],
    }

    def checkpoint():
        report["summary"] = summarize(report["cases"])
        if output is not None:
            write_json(output, report)

    checkpoint()
    for case_id in selected:
        case = cases[case_id]
        started = time.perf_counter()
        try:
            captured = collect(case)
        except Exception:  # noqa: BLE001 -- Never write provider exceptions, keys or private response bodies.
            captured = {"response": None, "error": "app_unavailable",
                        "elapsed_ms": (time.perf_counter() - started) * 1000, "usage": None}
        response = jsonable_encoder(captured.get("response"))
        if captured.get("error"):
            grade = {"status": "error", "checks": [], "retrieval": None, "cited_evidence": None,
                     "semantic": {"status": "not_run"}, "error": "app_unavailable"}
        else:
            grade = grade_response(case, response, docs, dataset.snapshot_id, mode, k)
        if judge and grade["status"] != "error":
            integrity_checks = {"response_schema", "request_identity", "result_contract",
                                "retrieval_traceability", "citation_traceability", "citation_integrity",
                                "answer_contract", "rendered_answer", "audience"}
            invalid = any(not item["passed"] for item in grade["checks"] if item["name"] in integrity_checks)
            if invalid:
                grade["semantic"] = {"status": "blocked_invalid_evidence"}
            else:
                try:
                    apply_judgment(grade, judge.grade(case, response))
                except Exception:  # noqa: BLE001 -- Judge/service failures are not answer-quality failures.
                    grade["semantic"] = {"status": "error", "error": "judge_unavailable_or_invalid"}
                    grade["status"] = "error"
        report["cases"].append({
            "case_id": case.id, "question": case.question, "split": case.split, "category": case.category,
            "expected_outcome": case.expected_outcome,
            "elapsed_ms": captured.get("elapsed_ms", 0), "usage": captured.get("usage"),
            "response": response, "grade": grade,
        })
        checkpoint()
    report["status"] = "complete"
    report["completed_at"] = utcnow()
    checkpoint()
    if output is not None:
        write_markdown(Path(output).with_suffix(".md"), report)
    return report


def write_markdown(path, report):
    """Write a compact report keeping failures and missing semantic grades visible."""
    summary = report["summary"]
    lines = ["# App evaluation", "", f"Snapshot: `{report['snapshot_id']}`. Mode: {report['mode']}. "
             f"k={report['k']}. Dataset review: {report['dataset_review_status']}.", "",
             f"Cases: {summary['cases']}; passed: {summary['passed']}; failed: {summary['failed']}; "
             f"errors: {summary['error']}; review required: {summary['review_required']}.", "",
             f"Semantic grades: {summary['semantic_graded']}; critical errors: {summary['critical_errors']}.",
             "", "| Case | Status | Evidence coverage | Failed checks |", "|---|---|---|---|"]
    for row in report["cases"]:
        grade = row["grade"]
        coverage = grade["retrieval"]["coverage"] if grade["retrieval"] else None
        score = f"{coverage:.0%}" if coverage is not None else "n/a"
        failures = ", ".join(item["name"] for item in grade["checks"] if not item["passed"]) or "—"
        lines.append(f"| {row['case_id']} | {grade['status']} | {score} | {failures} |")
    lines.extend(["", "## Cases needing attention", ""])
    for row in report["cases"]:
        grade = row["grade"]
        if grade["status"] == "passed":
            continue
        lines.extend([f"### {row['case_id']}", "", f"Status: {grade['status']}.", ""])
        for item in grade["checks"]:
            if not item["passed"]:
                lines.append(f"- {item['name']}: {item['reason']}")
        semantic = grade["semantic"]
        if semantic["status"] == "graded":
            result = semantic["result"]
            lines.append(f"- Judge: {result['reason']}")
            for item in result["statements"]:
                if not item["supported"]:
                    lines.append(f"- Unsupported statement {item['index']}: {item['reason']}")
            for item in result["requirements"]:
                if not item["satisfied"]:
                    lines.append(f"- Missing requirement {item['id']}: {item['reason']}")
            lines.extend(f"- Critical: {error}" for error in result["critical_errors"])
        else:
            lines.append(f"- Semantic assessment: {semantic['status']}")
        lines.append("")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _new_path(path):
    """Do not overwrite a dataset, captured run, or previous grading report."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise EvaluationInputError("--output must use a .json extension; Markdown is written separately")
    if path.exists() or path.with_suffix(".md").exists():
        raise EvaluationInputError("Output already exists; choose another --output path")
    return path


def _output_path(path):
    stamp = utcnow().replace(":", "-").replace("+", "_")
    return _new_path(path or DEFAULT_DIRECTORY / f"run_{stamp}.json")


def _judge(args):
    if not args.judge_model:
        return None
    if not args.allow_api:
        raise EvaluationInputError("--judge-model requires --allow-api; model grading can incur charges")
    return OpenAIJudge({"model": args.judge_model}, DEFAULT_DIRECTORY / "judge_cache")


def _capture_service(mode, k):
    """Use the same builders as /search and /answer, including the app parser and cutoff."""
    from src.services.answer_service import OpenAIAnswerWriter, build_answer_service
    from src.services.search_service import build_search_service
    from src.utils.config import load_config

    search = build_search_service()
    try:
        answer = build_answer_service(search) if mode == "answer" else None
        writer = answer.writer if answer else None
        metadata = {
            "api": load_config()["api"], "answer": answer.settings.model_dump() if answer else None,
            "index": search.session.run,
            "answer_prompt_hash": None,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("chromadb", "numpy", "pydantic", "sentence-transformers")},
        }
        if answer:
            from src.services.answer_service import INSTRUCTIONS

            metadata["answer_prompt_hash"] = digest(INSTRUCTIONS)

        def collect(case):
            started = time.perf_counter()
            if isinstance(writer, OpenAIAnswerWriter):
                writer.last_usage = None
            embedder = search.session.embedder
            before = getattr(embedder, "usage_tokens", 0)
            response = (answer.answer(case.question, k, case.audience) if answer
                        else search.search(case.question, k))
            return {
                "response": response, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "usage": {"new_embedding_tokens": getattr(embedder, "usage_tokens", 0) - before,
                          "answer": getattr(writer, "last_usage", None)},
            }

        return search, collect, metadata
    except Exception:
        search.close()
        raise


def build_parser():
    """Keep data preparation, paid collection and offline regrading explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create 30 draft, source-pinned cases offline")
    prepare.add_argument("--snapshot", help="Defaults to the active search index's snapshot")
    prepare.add_argument("--output", default=str(DEFAULT_DIRECTORY / "dataset.json"))
    for name in ("run", "grade"):
        command = commands.add_parser(
            name, help="Collect app outputs" if name == "run" else "Regrade saved outputs",
        )
        command.add_argument("--dataset", default=str(DEFAULT_DIRECTORY / "dataset.json"))
        command.add_argument("--output")
        command.add_argument("--judge-model", help="Optional semantic judge; explicitly select a model")
        command.add_argument("--allow-api", action="store_true", help="Allow potentially charged model calls")
        if name == "run":
            command.add_argument("--mode", choices=("search", "answer"), default="search")
            command.add_argument("--k", type=int, choices=range(1, 21), default=5)
            command.add_argument("--split", choices=("development", "regression", "holdout"))
            command.add_argument("--limit", type=int, help="Limit the number of evaluated cases")
        else:
            command.add_argument("--responses", required=True, help="A saved run JSON from this runner")
    compare = commands.add_parser("compare", help="Compare equivalent runs; exit 1 for regressions")
    compare.add_argument("baseline")
    compare.add_argument("current")
    return parser


def main(argv=None):
    """Exit 1 on failures/errors, 2 on ungraded answers or invalid input, otherwise 0."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            try:
                comparison = compare_reports(read_json(args.baseline), read_json(args.current))
            except ValueError as exc:
                raise EvaluationInputError(str(exc)) from None
            print(json.dumps(comparison, indent=2))
            if comparison["regressions"] or comparison["current"]["error"]:
                return 1
            return 2 if comparison["current"]["review_required"] else 0
        if args.command == "prepare":
            output = _new_path(args.output)
            snapshot_id = args.snapshot
            if snapshot_id is None:
                pointer = read_json(ROOT / "results/search_latest.json")
                index_path = ROOT / pointer["run_path"].replace("\\", "/")
                snapshot_id = read_json(index_path)["snapshot_id"]
            snapshot = load_sources(snapshot_id)
            dataset = starter_dataset(snapshot)
            verify_dataset(dataset, snapshot["documents"])
            write_json(output, dataset.model_dump())
            print(f"Prepared {len(dataset.cases)} draft cases: {output}")
            return 0
        if args.command == "run" and not args.allow_api:
            raise EvaluationInputError(
                "run requires --allow-api; query embeddings and answers may incur charges",
            )
        judge = _judge(args)
        if args.command == "run" and judge and args.mode != "answer":
            raise EvaluationInputError("--judge-model requires --mode answer")
        dataset = EvalDataset.model_validate(read_json(args.dataset))
        output = _output_path(args.output)
        if args.allow_api:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        snapshot = load_sources(dataset.snapshot_id)
        verify_dataset(dataset, snapshot["documents"])
        if args.command == "grade":
            saved = read_json(args.responses)
            if saved["dataset_hash"] != dataset.fingerprint or saved["snapshot_id"] != dataset.snapshot_id:
                raise EvaluationInputError("Saved responses do not match this dataset/snapshot")
            records = {row["case_id"]: row for row in saved["cases"]}
            if (len(records) != len(saved["cases"]) or set(records) != set(saved["case_ids"])
                    or saved["status"] != "complete"):
                raise EvaluationInputError(
                    "Saved run must contain all selected cases exactly once and be complete",
                )
            report = evaluate(
                dataset, snapshot["documents"], lambda case: {
                    **records[case.id],
                    "error": "app_unavailable" if records[case.id]["response"] is None else None,
                }, mode=saved["mode"], k=saved["k"], case_ids=saved["case_ids"], judge=judge,
                metadata={"collection": saved["metadata"], "parent_report_hash": digest(saved),
                          "capture_implementation_hash": saved["implementation_hash"],
                          "capture_created_at": saved["created_at"]},
                output=output,
            )
        else:
            selected = [case.id for case in dataset.cases if not args.split or case.split == args.split]
            if args.limit is not None:
                if args.limit < 1:
                    raise EvaluationInputError("--limit must be positive")
                selected = selected[:args.limit]
            if not selected:
                raise EvaluationInputError("No cases match the selected split")
            service, collect, metadata = _capture_service(args.mode, args.k)
            try:
                if service.session.snapshot_id != dataset.snapshot_id:
                    raise EvaluationInputError(
                        "The active app index differs from the evaluation dataset snapshot",
                    )
                report = evaluate(dataset, snapshot["documents"], collect, mode=args.mode, k=args.k,
                                  case_ids=selected, judge=judge, metadata=metadata, output=output)
            finally:
                service.close()
        print(json.dumps(report["summary"], indent=2))
        print(f"Saved JSON and Markdown reports: {output}")
        if report["summary"]["failed"] or report["summary"]["error"]:
            return 1
        return 2 if report["summary"]["review_required"] else 0
    except EvaluationInputError as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, RuntimeError):
        # Source/provider exceptions may include secrets; explain supported commands without echoing them.
        print("Evaluation could not run. Check arguments, dataset/source integrity, active snapshot, "
              "output path, and explicit --allow-api. Use 'fda-rag evaluate --help'.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
