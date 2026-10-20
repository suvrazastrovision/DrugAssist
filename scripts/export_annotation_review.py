"""Prepare concrete review materials; never assign reviewer identities or approval."""

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    benchmark = json.loads((ROOT / "data/benchmarks/benchmark.json").read_text(encoding="utf-8"))
    output = ROOT / "reports/annotation_review"
    output.mkdir(parents=True, exist_ok=True)
    query_rows, judgment_rows = [], []
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent FDA evidence review</title>',
        "<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:30px auto;padding:20px}article{border-top:2px solid #9aa;padding:20px 0}blockquote{background:#eef3fa;padding:15px}code{font-size:12px}h1,h2{color:#164368}</style>",
        "<h1>Independent annotation review packet</h1><p>All grades and spans below are drafts. Read the full frozen source sections, verify every query constraint, identify missing relevant evidence, and adjudicate disagreements. No human approval has been recorded.</p>",
        "<p>See ANNOTATION_PROTOCOL.md. This packet is a review aid; static excerpts cannot replace full-label assessment.</p>",
    ]
    for query in benchmark["queries"]:
        query_rows.append(
            {
                k: query.get(k, "")
                for k in [
                    "query_id",
                    "text",
                    "difficulty",
                    "partition",
                    "answerable",
                    "judgment_status",
                    "reviewer",
                    "reviewed_at",
                    "review_notes",
                ]
            }
        )
        body += [
            f"<article><h2>{query['query_id']}: {html.escape(query['text'])}</h2>",
            f"<p>Difficulty: {query['difficulty']}; partition: {query['partition']}; draft answerable: {query['answerable']}</p>",
        ]
        if not query["answerable"]:
            body.append(
                "<p>Draft unsupported rationale: " + html.escape(query["unsupported_reason"]) + "</p>"
            )
        for unit, grade in query["qrels"].items():
            judgment_rows.append(
                {
                    "query_id": query["query_id"],
                    "unit_id": unit,
                    "draft_grade": grade,
                    "reviewed_grade": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
        for evidence in query["evidence"]:
            body.append(
                f'<p>Label set <code>{html.escape(evidence["set_id"])}</code>; section <strong>{html.escape(evidence["section"])}</strong>; effective date {evidence["effective_time"]}; offsets {evidence["start"]}–{evidence["end"]}. <a href="{html.escape(evidence["source_url"], quote=True)}">Live FDA reference</a> (may now resolve a newer version).</p>'
            )
            body.append("<blockquote>" + html.escape(evidence["text"]) + "</blockquote>")
            body.append(
                "<p>Frozen source record SHA-256: <code>" + evidence["source_record_hash"] + "</code></p>"
            )
        body.append("</article>")
    body.append("</html>")
    (output / "annotation_review.html").write_text("\n".join(body), encoding="utf-8")
    for name, rows in [("queries.csv", query_rows), ("judgments.csv", judgment_rows)]:
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        "Exported review packet for", len(query_rows), "queries and", len(judgment_rows), "draft source units"
    )


if __name__ == "__main__":
    main()
