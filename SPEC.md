# SPEC: Structured Output Agent

> Goal: make an LLM reliable, not random. Turn messy job postings into strictly validated JSON, retry intelligently on failure, and measure how well it works.

 **Language:** Python 3.11+ | **Core libs:** Pydantic v2, an LLM SDK (Anthropic or OpenAI), SQLite, pytest

---

## 1. Problem statement

LLMs return free-form text. Downstream code needs typed, validated data. This agent takes raw, messy input (job postings pasted from anywhere) and returns a `JobPosting` object that is guaranteed to match a schema, or a clear, logged failure.

## 2. Goals

- G1. Enforce a Pydantic schema on every LLM output.
- G2. On parse or validation failure, retry by feeding the exact error back to the model (bounded retries).
- G3. Validate tool responses as well as final outputs.
- G4. Log every validation failure (input, raw output, error, attempt number) to SQLite.
- G5. Measure reliability with a repeatable evaluation and publish the numbers.

## 3. Non-goals

- No UI beyond a CLI (a tiny Streamlit demo is optional stretch).
- No fine-tuning, no RAG, no multi-agent logic.
- No web scraping. Inputs are text files.

## 4. Architecture

```
raw text ──► Extractor ──► LLM call ──► Parser ──► Validator (Pydantic)
                 ▲                                      │
                 │            error message             │ fail
                 └────────── Retry controller ◄─────────┘
                                   │ (max N)
                                   ▼
                     Failure logger (SQLite) ──► DeadLetter result

Tool call path:  LLM ──► tool ──► Tool response validator ──► back to LLM
```

Components:

| Component | Responsibility |
|---|---|
| `llm.py` | Thin provider wrapper: `complete(messages) -> str`. Keeps the agent provider-agnostic. |
| `schemas.py` | All Pydantic models. |
| `extractor.py` | Builds the prompt, calls the LLM, parses JSON, validates. |
| `retry.py` | Retry loop with error feedback, max attempts, optional backoff. |
| `tools.py` | Example tools plus Pydantic models for their responses. |
| `logger.py` | SQLite logging of attempts and failures. |
| `cli.py` | `extract <file>` and `eval <dir>` commands. |

## 5. Data schema (starting point, adjust as needed)

```python
from pydantic import BaseModel, Field, field_validator, HttpUrl
from typing import Literal, Optional

class Salary(BaseModel):
    min: Optional[int] = Field(None, ge=0)
    max: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, min_length=3, max_length=3)  # ISO 4217
    period: Optional[Literal["hour", "month", "year"]] = None

class JobPosting(BaseModel):
    title: str = Field(min_length=2)
    company: str
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
    def dedupe_skills(cls, v):
        return sorted({s.strip().lower() for s in v if s.strip()})
```

Cross-field rule to add: if `salary.min` and `salary.max` are both present, `min <= max`.

## 6. Functional requirements

- **FR1.** `extract(text) -> ExtractionResult` where result is either `success(JobPosting, attempts)` or `failure(last_error, attempts)`.
- **FR2.** Strip markdown code fences and surrounding prose before JSON parsing.
- **FR3.** Retry up to `MAX_RETRIES` (default 3). Each retry prompt includes the previous raw output and the exact Pydantic error text.
- **FR4.** Never retry infinitely. After the cap, return a failure result and log it.
- **FR5.** At least one tool (for example `normalize_currency` or `lookup_company`) whose response is validated by its own Pydantic model. Malformed tool output triggers a retry or a graceful error.
- **FR6.** SQLite table `attempts(id, run_id, input_hash, attempt_no, raw_output, error_type, error_detail, success, latency_ms, tokens_in, tokens_out, created_at)`.
- **FR7.** Config via environment variables: `LLM_PROVIDER`, `LLM_MODEL`, `MAX_RETRIES`, API key.
- **FR8.** CLI: `python -m agent extract samples/01.txt` and `python -m agent eval samples/`.

## 7. Error taxonomy (log each as `error_type`)

| Type | Example |
|---|---|
| `json_parse` | Output isn't valid JSON |
| `schema_missing_field` | Required field absent |
| `schema_type` | String where int expected |
| `schema_enum` | Value outside allowed literals |
| `cross_field` | salary min > max |
| `tool_response_invalid` | Tool returned malformed data |
| `max_retries_exceeded` | Gave up |

## 8. Evaluation plan (this is the portfolio centerpiece)

- Dataset: 20-30 real job postings in `samples/`, varied on purpose (messy formatting, missing salary, non-English fragments, multiple locations, very short and very long).
- Hand-label ground truth for 10+ of them in `samples/labels/*.json`.
- Metrics:
  - **First-try validity rate** (schema passes on attempt 1)
  - **Final validity rate** (passes within retries)
  - **Mean attempts per success**
  - **Field-level accuracy** vs. labels (title, company, work_mode, salary)
  - **Failure breakdown** by error type
  - **Avg latency and tokens per document**
- Baseline comparison: run once with no schema enforcement and no retries, then with the full agent. Report the delta.
- Output: `eval/results.md` generated by the `eval` command.

## 9. Testing

- Unit tests (no LLM calls, use a fake LLM that returns scripted outputs):
  - fenced JSON parses; prose-wrapped JSON parses
  - each error type is detected and classified
  - retry stops at `MAX_RETRIES`
  - retry prompt contains the previous error
  - tool response validation rejects bad data
- One integration test marked `@pytest.mark.live`, skipped by default.

## 10. Repo structure

```
structured-output-agent/
├── README.md
├── SPEC.md
├── pyproject.toml
├── .env.example
├── src/agent/
│   ├── __init__.py
│   ├── llm.py
│   ├── schemas.py
│   ├── extractor.py
│   ├── retry.py
│   ├── tools.py
│   ├── logger.py
│   └── cli.py
├── samples/            # input postings
│   └── labels/         # ground-truth JSON
├── eval/
│   └── results.md
└── tests/
```

## 11. Seven-day milestones

| Day | Deliverable | Done when |
|---|---|---|
| 1 | Repo, schemas, 20 samples collected, LLM wrapper | `llm.complete("hi")` works; schema unit-tested |
| 2 | Basic extractor (no retry) | `extract` returns a validated object for a clean sample |
| 3 | Retry controller with error feedback | Scripted-failure tests pass; cap enforced |
| 4 | Tool + response validation, SQLite logging | Failures appear in DB with correct error types |
| 5 | Eval harness, labels, baseline vs. full run | `eval/results.md` generated |
| 6 | README, architecture diagram, demo GIF | A stranger can run it in 5 minutes |
| 7 | Buffer, cleanup, publish, write-up | Pushed to GitHub; short post drafted |

## 12. Acceptance criteria

- [ ] All schema violations produce a retry or a logged failure. Nothing unvalidated escapes.
- [ ] Retries never exceed `MAX_RETRIES`.
- [ ] Final validity rate is measured on 20+ documents, with baseline comparison.
- [ ] Failure breakdown by error type is published.
- [ ] Unit tests pass without network access.
- [ ] README includes: problem, architecture diagram, quick start, results table, limitations.
- [ ] No API keys committed.

## 13. Stretch goals (only if Day 7 buffer is free)

- Native structured-output / tool-use mode from the provider vs. prompt-only JSON, compared in the eval.
- Streamlit demo with a side-by-side raw vs. validated view.
- Async batch processing with a concurrency limit.
- Confidence calibration: does `confidence` correlate with actual accuracy?

## 14. README outline (fill in on Day 6)

1. One-line pitch and the headline metric ("first-try 7x% → final 9x%")
2. Architecture diagram
3. Quick start (install, set key, run one command)
4. How retry-with-feedback works (short)
5. Evaluation results table and failure breakdown
6. Design decisions and limitations
7. What I'd do next

## 15. Risks

| Risk | Mitigation |
|---|---|
| Scope creep into RAG or multi-agent | Non-goals list above; park ideas in a `IDEAS.md` |
| Too few messy samples, so results look perfect | Deliberately include hard cases |
| Provider rate limits or cost | Cache responses by input hash during development |
| Flaky live tests | Fake LLM for unit tests |
