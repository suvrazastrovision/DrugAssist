import pytest

from src.indexing.chroma_store import ChromaStore
from src.retrieval.search import SearchSession, format_results, run_search
from src.utils.config import digest, read_json, write_json


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "hybrid_rerank"),
        (["--method", "bm25"], "bm25"),
        (["--method", "selected"], "selected"),
        (["--run", "saved-run.json"], "selected"),
    ],
)
def test_cli_method_defaults_and_overrides(options, expected, monkeypatch):
    import sys

    from src import cli
    from src.retrieval import search

    captured = []
    monkeypatch.setattr(cli, "load_config", lambda path=None: {"api": {"method": "hybrid_rerank"}})
    monkeypatch.setattr(search, "run_search", lambda *args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["fda-rag", "search", "metformin", *options])
    cli.main()
    assert captured[0][1] == expected


@pytest.fixture
def saved_run(chunks, tmp_path):
    run = {
        "run_id": "test-search-run",
        "snapshot_id": "test-snapshot",
        "provisional": True,
        "selection": {
            "preferred_on_dev": "bm25",
            "bm25": {"k1": 1.5, "b": 0.75},
            "chunking": {"encoding": "cl100k_base"},
            "embedding": {"corpus_hash": digest(sorted(c.chunk_id for c in chunks))},
        },
    }
    write_json(tmp_path / "run.json", run)
    write_json(tmp_path / "chunks.json", [c.to_dict() for c in chunks])
    return tmp_path / "run.json"


def test_search_reuses_saved_corpus_and_returns_complete_source_text(saved_run):
    session = SearchSession(saved_run)
    response = session.search("safety qualifier", k=2, filters={"section": "warnings"})
    assert response["method"] == "bm25" and response["snapshot_id"] == "test-snapshot"
    assert len(response["results"]) == 2
    assert all(row["section"] == "warnings" for row in response["results"])
    assert all(
        row["text"] and row["authority_url"] and row["source_record_hash"] for row in response["results"]
    )
    assert "not correctness probabilities" in response["score_note"]


def test_active_expanded_index_is_used_without_claiming_new_benchmark_scores(
    saved_run, tmp_path, monkeypatch
):
    from src.retrieval import search

    descriptor = read_json(saved_run)
    descriptor.update(
        default_method="bm25",
        artifact_type="search_index",
        evaluation_status="expanded_corpus_not_benchmarked",
        coverage={"covered_names": 60},
    )
    del descriptor["selection"]["preferred_on_dev"]
    write_json(saved_run, descriptor)
    write_json(tmp_path / "results/search_latest.json", {"run_path": "run.json"})
    write_json(tmp_path / "results/latest.json", {"run_path": "old-unavailable-run.json"})
    monkeypatch.setattr(search, "ROOT", tmp_path)
    result = SearchSession().search("safety")
    assert result["corpus_drug_names"] == 60
    assert "has not been benchmarked" in format_results(result)
    assert "Verified named-drug coverage: 60" in format_results(result)


def test_unknown_label_filter_is_not_silently_relaxed(saved_run):
    response = SearchSession(saved_run, "keyword").search("safety", filters={"set_id": "missing"})
    assert response["results"] == []
    assert "No filters were relaxed" in format_results(response)


@pytest.mark.parametrize("question,k", [("", 5), ("   ", 5), ("safety", 0), ("safety", 101)])
def test_bad_query_inputs_fail_clearly(saved_run, question, k):
    with pytest.raises(ValueError):
        SearchSession(saved_run).search(question, k)


def test_snapshot_mismatch_is_rejected(saved_run):
    with pytest.raises(ValueError, match="snapshot"):
        SearchSession(saved_run, snapshot_id="another-snapshot")


def test_corrupted_chunk_text_is_rejected(saved_run):
    from src.utils.config import read_json

    inventory = read_json(saved_run.parent / "chunks.json")
    inventory[0]["text"] = "modified source text"
    write_json(saved_run.parent / "chunks.json", inventory)
    with pytest.raises(ValueError, match="inconsistent"):
        SearchSession(saved_run)


def test_interactive_questions_and_quit(saved_run, monkeypatch, capsys):
    answers = iter(["safety qualifier", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    run_search(run_path=saved_run, method="keyword", k=1)
    text = capsys.readouterr().out
    assert "Ready: keyword" in text and "Passage:" in text and "Source:" in text


def test_json_output_is_parseable_and_preserves_text(saved_run, capsys):
    import json

    run_search("safety", run_path=saved_run, as_json=True, k=1)
    output = json.loads(capsys.readouterr().out)
    assert output["question"] == "safety" and output["results"][0]["text"]


def test_open_existing_does_not_create_a_missing_index(chunks, tmp_path):
    class Embedder:
        fingerprint = "test-search-embedding"

    location = tmp_path / "missing-index"
    with pytest.raises(FileNotFoundError, match="index is missing"):
        ChromaStore(chunks, Embedder(), path=location, create=False)
    assert not location.exists()


def test_open_existing_does_not_create_another_collection(chunks, tmp_path):
    class Embedder:
        fingerprint = "test-search-embedding"

    original = ChromaStore(chunks, Embedder(), path=tmp_path)
    before = len(original.client.list_collections())
    with pytest.raises(FileNotFoundError, match="collection is missing"):
        ChromaStore(chunks[:1], Embedder(), path=tmp_path, create=False)
    assert len(original.client.list_collections()) == before
