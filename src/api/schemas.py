"""Validated HTTP inputs and source-traceable outputs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from src.query_parser import ParsedQuery
from src.services.answer_service import AnswerStatement


class Credentials(BaseModel):
    """Validated account credentials with bounded email and password fields."""

    model_config = ConfigDict(extra="forbid")
    email: EmailStr = Field(max_length=254)
    password: str = Field(strict=True, min_length=8, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def trim_email(cls, value):
        """Strip surrounding whitespace from a submitted email string."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def not_blank_password(cls, value):
        """Reject passwords containing only whitespace."""
        if not value.strip():
            raise ValueError("Enter a nonempty password")
        return value


class UserResponse(BaseModel):
    """Public account details that exclude passwords and session tokens."""

    id: int
    email: str
    created_at: datetime


class LoginResponse(BaseModel):
    """A bearer token and its expiration time for a successful login."""

    session_token: str
    token_type: Literal["bearer"]
    expires_at: datetime


class SearchRequest(BaseModel):
    """A nonempty question and a bounded number of requested results."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(strict=True, min_length=1, max_length=2000)
    k: int = Field(default=5, strict=True, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value):
        """Reject queries containing only whitespace."""
        if not value.strip():
            raise ValueError("Enter a nonempty query")
        return value


class ThresholdInfo(BaseModel):
    """The active method's experimental relevance cutoff."""

    method: str
    min_score: float = Field(allow_inf_nan=False)
    status: Literal["experimental"]


class SearchHit(BaseModel):
    """A ranked label passage with offsets and source provenance."""

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    drug_name: str
    label_id: str | None
    set_id: str
    document_id: str
    chunk_id: str
    snapshot_id: str
    section: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    effective_time: str | None
    source_url: str | None
    source_record_hash: str
    source_body_sha256: str


class SearchResponse(BaseModel):
    """Retrieved evidence, applied filters, and explicit no-match information."""

    query: str
    parsed: ParsedQuery
    no_match: bool
    no_match_reason: str | None
    method: str
    snapshot_id: str
    applied_filters: dict[str, str | list[str]]
    score_kind: str
    score_note: str
    threshold: ThresholdInfo
    results: list[SearchHit]
    disclaimer: str


class AnswerRequest(SearchRequest):
    """A search request with patient or clinician wording preferences."""

    audience: Literal["patient", "clinician"] = "patient"


class AnswerCitation(SearchHit):
    """A cited source window with its identifier and context completeness."""

    source_id: str
    context_partial: bool


class AnswerResponse(SearchResponse):
    """A written answer, checked citation identifiers, and retrieval evidence."""

    answer: str | None
    statements: list[AnswerStatement]
    citations: list[AnswerCitation]
    audience: Literal["patient", "clinician"]
    answer_model: str
    answer_status: Literal["generated", "insufficient_evidence", "unavailable"]
    insufficient_evidence: bool | None
    evidence_note: str | None
    answer_note: str


class HealthResponse(BaseModel):
    """Index connectivity and document counts reported by the health endpoint."""

    status: Literal["ready", "unavailable"]
    chromadb: Literal["connected", "unavailable"]
    indexed_documents: int | None = Field(ge=0)
    indexed_chunks: int | None = Field(ge=0)
