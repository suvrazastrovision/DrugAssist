# Database schema

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

## Storage overview

| Store | Purpose | Location |
|---|---|---|
| Chroma | Passage metadata and vector retrieval | `data/chroma/` |
| Embedding cache | Reuse document/question vectors | `data/cache/embeddings.sqlite3` |
| Accounts | Users and login sessions | `data/auth/users.sqlite3` |

- These are separate databases, with **no SQL foreign keys between them**.
- Answer writing reuses saved evidence; it adds no tables or local answer history.

![DrugAssist database schema: Chroma tables, vector-index files, embedding cache, and separate account tables](database-schema.svg)

[Open the full-size diagram](database-schema.svg).

## Diagram key

| Mark | Meaning |
|---|---|
| `PK` | Primary key; multiple marked columns form a composite key |
| `FK` | Foreign-key declaration matching an existing table |
| `UQ` | Unique column or column combination |
| `NN` | Explicit `NOT NULL` declaration |
| Double bar | Exactly one parent in a valid relationship |
| Circle and bar | Zero or one parent; nullable foreign key |
| Circle and crow's foot | Zero or many children |
| Solid connector | Matching SQL foreign-key declaration |
| Dashed connector | Application/storage link, not a matching SQL foreign key |

- Core tables only; cardinalities describe valid records.
- Chroma's declared foreign keys do not guarantee enforcement on every connection.
- The vector-file group is not a table and has no cardinality markers.

## Chroma: searchable evidence

- Database: `data/chroma/chroma.sqlite3`.
- Vector files: `data/chroma/<vector-segment-id>/`.
- Active: **8,601 passages, 118 label sets, 60 drug names**.
- Embeddings: `text-embedding-3-small`, 1,536 dimensions; cosine HNSW.
- Historical collections may differ.

| Component | Role |
|---|---|
| `tenants`, `databases` | Chroma namespaces, separate from user accounts |
| `collections`, `collection_metadata` | Named record groups, configuration and fingerprints |
| `segments`, `segment_metadata` | Storage segments and attributes |
| `embeddings` | Record IDs, segment membership and update sequence; **no vector column** |
| `embedding_metadata` | Scalar attributes and stored document text |
| HNSW files | Numerical vector-search index |
| `embeddings_queue` | Internal update log; can contain vector BLOBs |

### Schema details A and B

- **A:** `segments.collection` references `collection(id)`, while the table is named `collections`; intended link shown dashed.
- **B:** `embeddings.segment_id` has no SQL foreign key to `segments.id`; directory links are application-managed.
- Other internals handle arrays, FTS and maintenance. Chroma's trigram FTS is separate from Python BM25.

### Passage mapping

These are **Chroma record API fields**, not a separate SQL `chunks` table.

| Field | App value |
|---|---|
| `id` | Deterministic `chunk_id`; internally `embeddings.embedding_id` |
| `embedding` | Vector for `embedding_text` |
| `document` | Identity + section heading + original passage text |
| `metadata` | The 16 scalar attributes below |

```text
chunk_id, document_id, snapshot_id, set_id, section,
start, end, token_count, kind, source_record_hash,
source_body_sha256, effective_time, authority_url,
strategy, size, overlap
```

- Document text uses `chroma:document`; separate `text`/`identity` fields remain in JSON.
- Offsets: Python characters, `section_text[start:end]`, end exclusive.
- IDs identify label sets (`set_id`), documents (`document_id`) or passages/splitting settings (`chunk_id`).

| Saved file | Role |
|---|---|
| `data/processed/<snapshot-id>/documents.json` | Complete normalized sections |
| `results/search_indexes/<index-id>/chunks.json` | Passage inventory |
| `results/search_latest.json` | Active index pointer |

File/database links use identifiers and fingerprints, not SQL foreign keys.

## Embedding cache

- Table: `vectors(key TEXT PRIMARY KEY, vector BLOB)`.
- Key: SHA-256 of canonical JSON `[embedding_fingerprint, exact_input_text]`.
- Value: `float32` vector; dimensions/finite values checked.
- Shared document/question cache; no separate model, drug or chunk columns.

## Accounts and sessions

- Default: `data/auth/users.sqlite3`; override with `DRUGASSIST_AUTH_DB`.
- `sessions.user_id -> users.id`: one user, many sessions; foreign keys enabled, delete cascades.
- Unique: `email`, `token_hash`. Hashes: Argon2 passwords, SHA-256 session tokens.
- Creation/expiry: Unix seconds; session user ID and expiry indexed.
- No search-history table or user-to-FDA foreign key.

## Editable relationship diagram

<details>
<summary>Show Mermaid source</summary>

The SVG above provides the table-grid layout and crow's-foot notation.

```mermaid
flowchart TB
    subgraph Chroma["Chroma retrieval store: data/chroma/chroma.sqlite3"]
        Tenants["tenants<br/>id TEXT PK"]
        Databases["databases<br/>id TEXT PK<br/>tenant_id TEXT NN FK<br/>name TEXT NN<br/>UQ: tenant_id + name"]
        Collections["collections<br/>id TEXT PK<br/>database_id TEXT NN FK<br/>name TEXT NN<br/>dimension INTEGER<br/>config_json_str TEXT<br/>schema_str TEXT<br/>UQ: name + database_id"]
        CollectionMeta["collection_metadata<br/>collection_id TEXT PK, FK<br/>key TEXT NN PK<br/>str_value / int_value / float_value / bool_value"]
        Segments["segments<br/>id TEXT PK<br/>type TEXT NN<br/>scope TEXT NN<br/>collection TEXT NN - see A"]
        SegmentMeta["segment_metadata<br/>segment_id TEXT PK, FK<br/>key TEXT NN PK<br/>str_value / int_value / float_value / bool_value"]
        Records["embeddings<br/>id INTEGER PK<br/>segment_id TEXT NN - see B<br/>embedding_id TEXT NN = chunk_id<br/>seq_id BLOB NN<br/>created_at TIMESTAMP NN<br/>UQ: segment_id + embedding_id"]
        RecordMeta["embedding_metadata<br/>id INTEGER PK, FK<br/>key TEXT NN PK<br/>string_value / int_value / float_value / bool_value"]
        Queue["embeddings_queue - separate internal log<br/>seq_id INTEGER PK<br/>id TEXT NN<br/>vector BLOB<br/>operation / topic / encoding / metadata / created_at"]
        Tenants -->|"FK: tenant_id"| Databases
        Databases -->|"FK: database_id"| Collections
        Collections -->|"FK: collection_id"| CollectionMeta
        Collections -.->|"A: intended collection relationship"| Segments
        Segments -->|"FK: segment_id"| SegmentMeta
        Segments -.->|"B: application segment link"| Records
        Records -->|"FK: id"| RecordMeta
    end

    Files["HNSW files<br/>data/chroma/vector-segment-id/<br/>data_level0.bin / header.bin / length.bin<br/>link_lists.bin / index_metadata.pickle"]
    Segments -.->|"B: vector-segment directory"| Files

    subgraph Cache["Separate cache: data/cache/embeddings.sqlite3"]
        Vectors["vectors<br/>key TEXT PK<br/>vector BLOB<br/>Key: digest of fingerprint + exact text<br/>Value: float32 vector"]
    end

    subgraph Auth["Separate accounts: data/auth/users.sqlite3"]
        Users["users<br/>id INTEGER PK<br/>email TEXT NN UQ<br/>password_hash TEXT NN<br/>created_at INTEGER NN"]
        Sessions["sessions<br/>id INTEGER PK<br/>user_id INTEGER NN FK<br/>token_hash TEXT NN UQ<br/>created_at INTEGER NN<br/>expires_at INTEGER NN"]
        Users -->|"FK: user_id; 0..many sessions; ON DELETE CASCADE"| Sessions
    end

    classDef core fill:#eef6ff,stroke:#2b628e,color:#18374b;
    classDef metadata fill:#ecf8f4,stroke:#198374,color:#18374b;
    classDef storage fill:#fff5e7,stroke:#b87927,color:#18374b;
    classDef account fill:#f3effa,stroke:#8060aa,color:#18374b;
    class Tenants,Databases,Collections,Segments,Records core;
    class CollectionMeta,SegmentMeta,RecordMeta metadata;
    class Queue,Files,Vectors storage;
    class Users,Sessions account;
```

</details>

## Verification and scope

- Schema audit: **20 October 2026**, Chroma **1.5.9**, on-disk schemas and active index.
- Inspection used read-only SQLite connections; no account rows or secrets were read.
- Chroma internals may change after migrations.
- Traceability and correct storage do not establish medical relevance or clinical correctness.

Implementation: [Chroma](../src/indexing/chroma_store.py), [chunk offsets](../src/preprocessing/chunking.py), [cache](../src/embeddings/cache.py), [auth schema](../src/auth/database.py), [sessions](../src/auth/service.py), [fingerprints](../src/utils/config.py).
