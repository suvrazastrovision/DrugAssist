"""Frozen, versioned evaluation cases anchored to exact saved FDA label text."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluation.section_check import make_hard_questions
from src.utils.config import digest


class StrictModel(BaseModel):
    """Reject unexpected fields and coercions in evaluation artifacts."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Reference(StrictModel):
    """One acceptable occurrence of the evidence for a required fact."""

    document_id: str
    set_id: str
    section: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    source_record_hash: str
    source_body_sha256: str


class Requirement(StrictModel):
    """An expected fact and alternative source occurrences supporting it."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    references: list[Reference] = Field(min_length=1)


class EvalCase(StrictModel):
    """An end-to-end question; abstention expectations are authored explicitly."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    split: Literal["development", "regression", "holdout"]
    category: str
    audience: Literal["patient", "clinician"] = "patient"
    expected_outcome: Literal["answer", "abstain"]
    requirements: list[Requirement]

    @model_validator(mode="after")
    def valid_requirements(self):
        """Positive cases need evidence; negative cases must not require an answer."""
        if bool(self.requirements) != (self.expected_outcome == "answer"):
            raise ValueError("Answer cases require evidence; abstention cases require no facts")
        if len({item.id for item in self.requirements}) != len(self.requirements):
            raise ValueError("Requirement identifiers must be unique within a case")
        if not self.question.strip():
            raise ValueError("Questions cannot be blank")
        return self


class EvalDataset(StrictModel):
    """A source-pinned dataset; a draft is never advertised as clinical validation."""

    schema_version: Literal[1] = 1
    snapshot_id: str = Field(min_length=1)
    review_status: Literal["draft", "human_reviewed"] = "draft"
    notes: str
    cases: list[EvalCase] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_cases(self):
        """Stable case IDs are needed to join saved responses and compare runs."""
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Case identifiers must be unique")
        return self

    @property
    def fingerprint(self):
        """Hash questions, source anchors, expectations, splits and review status."""
        return digest(self.model_dump())


# These are diagnostic paraphrases of the existing evidence anchors, not held-out tests.
DIRECT_QUESTIONS = (
    (0, 0, "What does the prednisone label say about abruptly stopping corticosteroids?"),
    (0, 1, "What should patients on prolonged prednisone tell medical attendants and do for a new illness?"),
    (1, 0, "What gabapentin dosing rule is stated for creatinine clearance below 15 mL/min?"),
    (1, 1, "What does the gabapentin label say about supplemental dosing after hemodialysis?"),
    (2, 0, "When does the levothyroxine label say to take the tablet relative to breakfast?"),
    (2, 1, "How far apart should levothyroxine and calcium or iron binders be taken according to the label?"),
    (2, 2, "What does the levothyroxine label say about biotin before thyroid laboratory testing?"),
    (3, 0, "Which age, weight and creatinine criteria determine the reduced apixaban dose in the label?"),
    (3, 1, "What is the apixaban interaction rule for combined inhibitors at the 2.5 mg twice-daily dose?"),
    (4, 0, "What does the metformin label say about initiation at an eGFR between 30 and 45?"),
    (4, 2, "What kidney check and restart condition does the metformin label specify after contrast?"),
    (5, 0, "What does the sertraline label say about dosage in mild hepatic impairment?"),
    (5, 1, "What does the sertraline label say about moderate or severe hepatic impairment?"),
    (6, 0, "Does the warfarin label say a therapeutic INR eliminates bleeding risk?"),
    (8, 0, "What does the clopidogrel label say about omeprazole or esomeprazole?"),
)
NEGATIVE_QUESTIONS = (
    ("out_of_scope", "What will the weather in Berlin be tomorrow?"),
    ("out_of_scope", "What is the current stock price of the manufacturer of metformin?"),
    ("unknown_drug", "What are the warnings for ZZZ_EVALUATION_FICTIONAL_8472?"),
    ("unavailable_information", "What is my metformin prescription number at my local pharmacy?"),
    ("unavailable_information", "What is the exact number of metformin tablets in my cupboard right now?"),
)


def starter_dataset(snapshot):
    """Create 30 draft cases without generating facts or calling an external model.

    Ten challenges and fifteen focused questions reuse the existing 25 source-anchored
    requirements. Five negative cases exercise abstention. They are not 30 independent
    clinical judgments, and none is represented as an untouched holdout.
    """
    questions = make_hard_questions(snapshot)
    cases = []
    for question in questions:
        requirements = []
        for item in question["requirements"]:
            fields = Reference.model_fields
            references = [Reference(**{key: ref[key] for key in fields})
                          for ref in question["references"] if ref["requirement_id"] == item["id"]]
            requirements.append(Requirement(**item, references=references))
        cases.append(EvalCase(
            id=question["id"], question=question["text"], split="regression",
            category=question["topic"], audience="clinician", expected_outcome="answer",
            requirements=requirements,
        ))
    for number, (task, requirement, text) in enumerate(DIRECT_QUESTIONS, 1):
        cases.append(EvalCase(
            id=f"focused_{number:02d}", question=text, split="development", category="focused_fact",
            expected_outcome="answer",
            requirements=[cases[task].requirements[requirement].model_copy(deep=True)],
        ))
    for number, (category, text) in enumerate(NEGATIVE_QUESTIONS, 1):
        cases.append(EvalCase(
            id=f"negative_{number:02d}", question=text, split="development", category=category,
            expected_outcome="abstain", requirements=[],
        ))
    return EvalDataset(
        snapshot_id=snapshot["manifest"]["snapshot_id"], cases=cases,
        notes="Draft diagnostic cases reusing existing label anchors; human review is required. "
              "Focused cases share evidence with regression cases. No independent holdout or "
              "clinical validation is included. Add separately authored, reviewed holdout cases "
              "before release.",
    )


def verify_dataset(dataset, documents):
    """Reject changed snapshots, fabricated quotes, bad offsets and mismatched source hashes."""
    docs = {doc["document_id"]: doc for doc in documents}
    for case in dataset.cases:
        for requirement in case.requirements:
            for ref in requirement.references:
                doc = docs.get(ref.document_id)
                if doc is None:
                    raise ValueError(f"Reference document missing for {case.id}")
                section = doc["sections"].get(ref.section, "")
                valid = (
                    doc["kind"] == "fda_full" and doc["snapshot_id"] == dataset.snapshot_id
                    and doc["set_id"] == ref.set_id
                    and doc["source_record_hash"] == ref.source_record_hash
                    and doc["provenance"]["body_sha256"] == ref.source_body_sha256
                    and 0 <= ref.start < ref.end <= len(section)
                    and section[ref.start:ref.end] == ref.quote
                )
                if not valid:
                    raise ValueError(f"Source anchor mismatch for {case.id}/{requirement.id}")
