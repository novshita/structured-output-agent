# Structured Output Agent

**Makes an LLM reliable instead of random: turns messy job postings into strictly validated JSON, retries with the exact error when output is invalid, and measures how well that works.**

> Headline metric: first-try validity **TBD%** → final validity **TBD%** (after retry-with-feedback), measured on 20+ real postings. See [Evaluation](#evaluation-results).

> **Status:** 🚧 In development. Extractor, retry loop, tool validation, SQLite logging and the eval harness are implemented and unit-tested; the sample dataset and live eval run are next (see [SPEC.md](SPEC.md)).

---

## The problem

LLMs return free-form text, but downstream code needs typed, validated data. Give an LLM a job posting and ask for JSON, and sometimes you get it. Other times it's wrapped in markdown fences, missing a field, `"remote-ish"` instead of `"remote"`, or a salary where `min > max`.

This agent sits between the model and your code. For any raw job posting it returns either:

- a `JobPosting` object **guaranteed** to match the schema, or
- a clear failure result, **logged** with the raw output, the error type, and the attempt number.

Unvalidated data never gets through.

## Architecture

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

| Module | Responsibility |
|---|---|
| [`llm.py`](src/agent/llm.py) | Thin provider wrapper, `complete(messages) -> str`. Keeps the agent provider-agnostic (Anthropic or OpenAI). |
| [`schemas.py`](src/agent/schemas.py) | All Pydantic models: `JobPosting`, `Salary`, tool responses, `ExtractionResult`. |
| [`extractor.py`](src/agent/extractor.py) | Builds the prompt, calls the LLM, strips fences/prose, parses JSON, validates. |
| [`retry.py`](src/agent/retry.py) | Bounded retry loop that feeds the exact error back to the model. |
| [`tools.py`](src/agent/tools.py) | Example tools (e.g. `normalize_currency`) whose responses are validated by their own Pydantic models. |
| [`logger.py`](src/agent/logger.py) | Logs every attempt to SQLite. |
| [`cli.py`](src/agent/cli.py) | `extract` and `eval` commands. |

### Output schema (abridged)

```python
class JobPosting(BaseModel):
    title: str                      # min 2 chars
    company: str
    location: Optional[str]
    work_mode: Literal["onsite", "hybrid", "remote", "unspecified"]
    employment_type: Literal["full_time", "part_time", "contract", "internship", "unspecified"]
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unspecified"]
    salary: Optional[Salary]        # min <= max enforced
    required_skills: list[str]      # trimmed, lowercased, deduped, sorted
    years_experience_min: Optional[int]   # 0–40
    apply_url: Optional[HttpUrl]
    confidence: float               # 0–1
```

The full schema is in [SPEC.md §5](SPEC.md#5-data-schema-starting-point-adjust-as-needed).

## Quick start

Requires Python 3.11+.

```bash
# 1. Install
git clone <repo-url> structured-output-agent
cd structured-output-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",openai" to use OpenAI instead of Anthropic

# 2. Configure
cp .env.example .env               # then add your API key

# 3. Run
python -m agent extract samples/01.txt    # extract one posting
python -m agent eval samples/             # run the full evaluation -> eval/results.md
```

### Configuration

All configuration comes from environment variables (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-5-5` | Model ID for the chosen provider |
| `MAX_RETRIES` | `3` | Hard cap on retry attempts |
| `AGENT_DB` | `agent.db` | SQLite file for the `attempts` log |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | none | Provider API key |

### Running tests

```bash
pytest               # unit tests only; uses a scripted fake LLM, no network needed
pytest -m live       # the integration test against a real provider (needs an API key)
```

## How retry-with-feedback works

1. The extractor asks the model for JSON that matches the `JobPosting` schema.
2. The parser strips markdown code fences and any prose around the JSON, then parses it.
3. Pydantic validates the result, including cross-field rules such as `salary.min <= salary.max`.
4. **If parsing or validation fails**, the retry controller builds a new prompt that includes the model's **previous raw output** and the **exact Pydantic error text**, and asks it to fix only what is wrong.
5. This repeats up to `MAX_RETRIES` times. It never loops forever. After the cap, the agent returns a failure result and logs it as `max_retries_exceeded`.

Every attempt, successful or not, is written to the SQLite `attempts` table:

```
attempts(id, run_id, input_hash, attempt_no, raw_output, error_type,
         error_detail, success, latency_ms, tokens_in, tokens_out, created_at)
```

### Error taxonomy

| `error_type` | Meaning |
|---|---|
| `json_parse` | Output isn't valid JSON |
| `schema_missing_field` | A required field is absent |
| `schema_type` | Wrong type (e.g. string where int expected) |
| `schema_enum` | Value outside the allowed literals |
| `cross_field` | A cross-field rule failed (e.g. salary min > max) |
| `tool_response_invalid` | A tool returned malformed data (or the tool call itself was malformed) |
| `llm_error` | The provider call failed (network, API error, refusal) |
| `max_retries_exceeded` | Gave up after the retry cap |

## Evaluation results

> ⏳ Not yet run. These tables are filled in from [`eval/results.md`](eval/results.md), which `python -m agent eval samples/` generates.

**Dataset:** 20+ real job postings in `samples/`, chosen to be varied and hard: messy formatting, missing salary, non-English fragments, multiple locations, very short and very long. 10+ of them have hand-labelled ground truth in `samples/labels/`.

**Baseline vs. full agent**

| Metric | Baseline (no schema enforcement, no retries) | Full agent | Δ |
|---|---|---|---|
| First-try validity rate | TBD | TBD | TBD |
| Final validity rate | TBD | TBD | TBD |
| Mean attempts per success | n/a | TBD | n/a |
| Avg latency per doc | TBD | TBD | TBD |
| Avg tokens per doc | TBD | TBD | TBD |

**Field-level accuracy (vs. labels)**

| Field | Accuracy |
|---|---|
| `title` | TBD |
| `company` | TBD |
| `work_mode` | TBD |
| `salary` | TBD |

**Failure breakdown**

| `error_type` | Count | % of failed attempts |
|---|---|---|
| `json_parse` | TBD | TBD |
| `schema_missing_field` | TBD | TBD |
| `schema_type` | TBD | TBD |
| `schema_enum` | TBD | TBD |
| `cross_field` | TBD | TBD |
| `tool_response_invalid` | TBD | TBD |
| `max_retries_exceeded` | TBD | TBD |

## Design decisions and limitations

**Decisions**

- **Pydantic is the source of truth.** The schema defines both the prompt contract and the validation. Nothing reaches the caller without passing `model_validate`.
- **The exact error goes back to the model.** Sending Pydantic's own error text, rather than a vague "please fix it", tells the model precisely which field broke and why.
- **Bounded retries.** A hard `MAX_RETRIES` cap keeps cost and latency predictable, and failures become data instead of hangs.
- **Provider-agnostic LLM wrapper.** One `complete()` function means the provider can be swapped, and unit tests can use a scripted fake LLM with no network calls.
- **Tool outputs are validated too.** Tools are also an untrusted boundary, so each tool response has its own Pydantic model.

**Limitations**

- Valid does not mean correct. The schema guarantees shape, not truth, so a well-formed but wrong `company` still passes. The field-level accuracy numbers measure this gap.
- `confidence` is self-reported by the model and not yet calibrated.
- The dataset is small (20–30 postings), so treat the percentages as directional.
- Inputs are plain text files only. There is no scraping and no HTML or PDF parsing.

## What I'd do next

- Compare the provider's native structured-output / tool-use mode against prompt-only JSON in the eval.
- Check confidence calibration: does `confidence` track actual field accuracy?
- Async batch processing with a concurrency limit.
- A small Streamlit demo showing raw and validated output side by side.

## Project layout

```
structured-output-agent/
├── README.md
├── SPEC.md              # full project spec and 7-day plan
├── pyproject.toml
├── .env.example
├── src/agent/           # llm, schemas, extractor, retry, tools, logger, cli
├── samples/             # input postings (.txt)
│   └── labels/          # hand-labelled ground truth (.json)
├── eval/
│   └── results.md       # generated by `python -m agent eval`
└── tests/               # unit tests (fake LLM) + one @pytest.mark.live test
```
