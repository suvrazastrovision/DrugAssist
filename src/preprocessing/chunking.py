"""Offset-preserving, model-token-aware segmentation alternatives.

All strategies stay inside a label section; cross-section safety context is retained
through parent metadata, rather than silently joining unrelated passages.
"""

import re
from dataclasses import asdict, dataclass

import tiktoken

from src.utils.config import digest


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    snapshot_id: str
    set_id: str
    section: str
    start: int
    end: int
    text: str
    identity: str
    token_count: int
    kind: str
    source_record_hash: str
    source_body_sha256: str
    effective_time: str
    authority_url: str
    strategy: str
    size: int
    overlap: int

    @property
    def embedding_text(self):
        return f"{self.identity}\nSection: {self.section}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def token_count(text, encoding="cl100k_base"):
    return len(tiktoken.get_encoding(encoding).encode(text, disallowed_special=()))


def _token_char_ends(text, encoding):
    # Map token byte ends to character positions, never splitting a UTF-8 character.
    encoded = text.encode("utf-8")
    byte_to_char, byte_count = {0: 0}, 0
    for i, char in enumerate(text, 1):
        byte_count += len(char.encode("utf-8"))
        byte_to_char[byte_count] = i
    byte_pos, ends = 0, []
    for token in encoding.encode(text, disallowed_special=()):
        byte_pos += len(encoding.decode_single_token_bytes(token))
        if byte_pos == len(encoded) or (encoded[byte_pos] & 0xC0) != 0x80:
            ends.append(byte_to_char[byte_pos])
    return sorted(set(ends))


def spans(text, strategy="section", size=256, overlap=32, encoding="cl100k_base"):
    if strategy not in {"fixed", "recursive", "section", "semantic"}:
        raise ValueError("Unknown chunking strategy")
    if size < 8 or not 0 <= overlap < size:
        raise ValueError("Require size >= 8 and 0 <= overlap < size")
    codec = tiktoken.get_encoding(encoding)
    # Semantic here means linguistic sentence/paragraph boundaries, not an embedding change detector.
    preferred = []
    if strategy in {"section", "recursive"}:
        preferred = [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n", text)]
    if strategy == "semantic":
        preferred = [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text)]
    paragraphs = [m.end() for m in re.finditer(r"\n\s*\n", text)] if strategy == "section" else []
    if strategy == "recursive":
        preferred += [m.end() for m in re.finditer(r"\n|;\s|,\s", text)]
    start = 0
    while start < len(text):
        # Tokenize a bounded prefix; a cl100k token contains at least one byte.
        prefix = text[start : start + size * 16]
        tokens = codec.encode(prefix, disallowed_special=())[:size]
        byte_prefix = b"".join(codec.decode_single_token_bytes(t) for t in tokens)
        end = start + len(byte_prefix.decode("utf-8", errors="ignore"))
        if end <= start:
            end = start + 1
        if end < len(text) and preferred:
            aligned = [p for p in preferred if start < p <= end and p >= start + (end - start) // 2]
            paragraph_aligned = [
                p for p in paragraphs if start < p <= end and p >= start + (end - start) // 2
            ]
            if paragraph_aligned:
                aligned = paragraph_aligned
            if aligned:
                end = max(aligned)
        if text[start:end].strip():
            yield start, end
        if end >= len(text):
            break
        if overlap:
            tokens = codec.encode(text[start:end], disallowed_special=())
            tail = b"".join(codec.decode_single_token_bytes(t) for t in tokens[-overlap:])
            new_start = end - len(tail.decode("utf-8", errors="ignore"))
            start = new_start if new_start > start else end
        else:
            start = end


def chunk_documents(documents, strategy="section", size=256, overlap=32, encoding="cl100k_base"):
    chunks = []
    for doc in documents:
        names = doc["metadata"]
        identity = " | ".join(
            dict.fromkeys(names["brand_name"] + names["generic_name"] + names["substance_name"])
        )
        identity = identity or f"Label set {doc['set_id']}"
        for section, text in sorted(doc["sections"].items()):
            for start, end in spans(text, strategy, size, overlap, encoding):
                value = text[start:end]
                chunk_id = digest(
                    [doc["document_id"], section, start, end, value, strategy, size, overlap, encoding]
                )[:40]
                chunks.append(
                    Chunk(
                        chunk_id,
                        doc["document_id"],
                        doc["snapshot_id"],
                        doc["set_id"],
                        section,
                        start,
                        end,
                        value,
                        identity,
                        token_count(value, encoding),
                        doc["kind"],
                        doc["source_record_hash"],
                        doc["provenance"]["body_sha256"],
                        doc["effective_time"] or "",
                        doc["authority_url"],
                        strategy,
                        size,
                        overlap,
                    )
                )
    return chunks
