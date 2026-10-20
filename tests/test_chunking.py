import pytest

from src.preprocessing.chunking import chunk_documents, spans, token_count


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "section", "semantic"])
def test_exact_offsets_coverage_and_limits(document, strategy):
    chunks = chunk_documents([document], strategy, size=32, overlap=4)
    assert chunks == chunk_documents([document], strategy, size=32, overlap=4)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for section, text in document["sections"].items():
        coverage = [False] * len(text)
        for c in [c for c in chunks if c.section == section]:
            assert text[c.start : c.end] == c.text
            assert token_count(c.text) <= 32
            coverage[c.start : c.end] = [True] * (c.end - c.start)
            assert c.set_id == document["set_id"] and c.source_record_hash
        assert all(covered or char.isspace() for char, covered in zip(text, coverage, strict=True))


def test_unicode_bytes_do_not_corrupt_offsets():
    text = "Do not change ≥ 2.5 µg/kg or β blockers. 日本語の文字。 " * 15
    parts = list(spans(text, size=24, overlap=4))
    assert parts[-1][1] == len(text)
    assert all("\ufffd" not in text[a:b] and token_count(text[a:b]) <= 24 for a, b in parts)


@pytest.mark.parametrize("size,overlap", [(16, 16), (16, -1), (0, 0)])
def test_invalid_chunk_configuration(size, overlap):
    with pytest.raises(ValueError):
        list(spans("text", size=size, overlap=overlap))


def test_chunk_id_changes_when_source_changes(document):
    first = chunk_documents([document], size=32, overlap=4)
    document["document_id"] = "other-source-record"
    second = chunk_documents([document], size=32, overlap=4)
    assert {c.chunk_id for c in first}.isdisjoint({c.chunk_id for c in second})
