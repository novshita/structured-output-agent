"""All Pydantic models (JobPosting, Salary, tool responses, ExtractionResult)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from pydantic_core import PydanticCustomError

ErrorType = Literal[
    "json_parse",
    "schema_missing_field",
    "schema_type",
    "schema_enum",
    "cross_field",
    "tool_response_invalid",
    "max_retries_exceeded",
    "llm_error",
]


class Salary(BaseModel):
    min: Optional[int] = Field(None, ge=0)
    max: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, pattern=r"^[A-Z]{3}$")  # ISO 4217
    period: Optional[Literal["hour", "month", "year"]] = None

    @model_validator(mode="after")
    def min_not_above_max(self) -> "Salary":
        if self.min is not None and self.max is not None and self.min > self.max:
            # Custom error type so the classifier can tag it as `cross_field`.
            raise PydanticCustomError(
                "cross_field",
                "salary.min ({min}) must be <= salary.max ({max})",
                {"min": self.min, "max": self.max},
            )
        return self


class JobPosting(BaseModel):
    title: str = Field(min_length=2)
    company: str = Field(min_length=1)
    location: Optional[str] = None
    work_mode: Literal["onsite", "hybrid", "remote", "unspecified"]
    employment_type: Literal["full_time", "part_time", "contract", "internship", "unspecified"]
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unspecified"]
    salary: Optional[Salary] = None
    required_skills: list[str] = Field(default_factory=list)
    years_experience_min: Optional[int] = Field(None, ge=0, le=40)
    apply_url: Optional[HttpUrl] = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("required_skills")
    @classmethod
    def dedupe_skills(cls, v: list[str]) -> list[str]:
        return sorted({s.strip().lower() for s in v if s.strip()})


# --- Tool I/O -------------------------------------------------------------


class ToolCall(BaseModel):
    """What the model emits when it wants to call a tool."""

    name: str
    arguments: dict = Field(default_factory=dict)


class CurrencyNormalization(BaseModel):
    """Response of the `normalize_currency` tool."""

    model_config = ConfigDict(extra="forbid")

    raw: str
    code: Optional[str] = Field(None, pattern=r"^[A-Z]{3}$")
    recognized: bool

    @model_validator(mode="after")
    def code_iff_recognized(self) -> "CurrencyNormalization":
        if self.recognized != (self.code is not None):
            raise ValueError("`code` must be set if and only if `recognized` is true")
        return self


# --- Results --------------------------------------------------------------


class CallRecord(BaseModel):
    """One LLM call made while extracting a document."""

    attempt_no: int
    raw_output: Optional[str]
    error_type: Optional[ErrorType] = None
    error_detail: Optional[str] = None
    success: bool = False
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class ExtractionResult(BaseModel):
    run_id: str
    ok: bool
    posting: Optional[JobPosting] = None
    attempts: int
    last_error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    calls: list[CallRecord] = Field(default_factory=list)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def latency_ms(self) -> int:
        return sum(c.latency_ms for c in self.calls)
