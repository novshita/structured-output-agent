"""Builds the prompt, calls the LLM, parses JSON, validates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Literal, Optional, Union

from pydantic import ValidationError

from .llm import LLM, get_llm
from .logger import AttemptLogger
from .schemas import ErrorType, ExtractionResult, JobPosting, ToolCall
from .tools import TOOLS, Tool, describe_tools

Mode = Literal["full", "baseline"]

SYSTEM_PROMPT = """\
You extract structured data from job postings.

Reply with a single JSON object that validates against this JSON Schema. \
No prose, no markdown, no comments.

{schema}

Rules:
- Use "unspecified" for work_mode, employment_type or seniority when the posting does not say.
- Use null for any optional field the posting does not state. Do not guess salaries or URLs.
- salary.currency must be an ISO 4217 code (e.g. "USD"). salary.period is "hour", "month" or "year".
- Express salary amounts as plain integers in the stated currency (e.g. 120k -> 120000).
- A single salary figure: "up to X" sets only max, "from X" sets only min, a plain "X" sets both.
- If several salary ranges are listed (e.g. per location), use the first one.
- Copy the job title as written, without the company name or location.
- confidence is your 0-1 estimate that every field you filled in is correct.

Tools: if you need one, reply with ONLY {{"tool_call": {{"name": <tool>, "arguments": {{...}}}}}} \
and you will receive the result. Available tools:
{tools}
"""

BASELINE_SYSTEM_PROMPT = """\
Extract the key details of this job posting as JSON with the fields: title, company, location, \
work_mode, employment_type, seniority, salary (min, max, currency, period), required_skills, \
years_experience_min, apply_url, confidence.
"""

USER_TEMPLATE = "Job posting:\n<posting>\n{text}\n</posting>"

RETRY_TEMPLATE = """\
Your previous output (above) was rejected.

Error type: {error_type}
Error:
{error_detail}

Fix exactly what the error describes and reply with ONLY the corrected JSON object."""


class JSONParseError(ValueError):
    pass


# --- Parsing & validation --------------------------------------------------

_FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def parse_json(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating code fences and surrounding prose."""
    s = text.strip()
    for candidate in [s, *_FENCE.findall(s)]:
        try:
            obj = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(s, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise JSONParseError(f"No JSON object found in output: {s[:200]!r}")


def classify_validation_error(e: ValidationError) -> ErrorType:
    """Map a Pydantic error to the taxonomy, using the first error reported."""
    t = e.errors()[0]["type"]
    if t == "missing":
        return "schema_missing_field"
    if t in ("literal_error", "enum"):
        return "schema_enum"
    if t == "cross_field":
        return "cross_field"
    return "schema_type"


@dataclass
class Parsed:
    kind: Literal["ok", "tool", "error"]
    posting: Optional[JobPosting] = None
    tool_call: Optional[ToolCall] = None
    error_type: Optional[ErrorType] = None
    error_detail: Optional[str] = None


def interpret(raw: str, allow_tools: bool = True) -> Parsed:
    """Turn one raw model output into a validated posting, a tool call, or a classified error."""
    try:
        data = parse_json(raw)
    except JSONParseError as e:
        return Parsed("error", error_type="json_parse", error_detail=str(e))
    if allow_tools and set(data) == {"tool_call"}:
        try:
            return Parsed("tool", tool_call=ToolCall.model_validate(data["tool_call"]))
        except ValidationError as e:
            return Parsed("error", error_type="tool_response_invalid",
                          error_detail=f"malformed tool_call: {e}")
    try:
        return Parsed("ok", posting=JobPosting.model_validate(data))
    except ValidationError as e:
        return Parsed("error", error_type=classify_validation_error(e), error_detail=str(e))


# --- Prompts ---------------------------------------------------------------


def build_system_prompt(mode: Mode = "full", tools: dict[str, Tool] = TOOLS) -> str:
    if mode == "baseline":
        return BASELINE_SYSTEM_PROMPT
    schema = json.dumps(JobPosting.model_json_schema(), indent=2)
    return SYSTEM_PROMPT.format(schema=schema, tools=describe_tools(tools))


def build_retry_prompt(error_type: str, error_detail: str) -> str:
    return RETRY_TEMPLATE.format(error_type=error_type, error_detail=error_detail)


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Entry point -----------------------------------------------------------


def extract(
    text: str,
    llm: Optional[LLM] = None,
    *,
    mode: Mode = "full",
    max_retries: Optional[int] = None,
    logger: Union[AttemptLogger, None, Literal[False]] = None,
    tools: dict[str, Tool] = TOOLS,
    backoff_s: float = 0.0,
) -> ExtractionResult:
    """Extract a validated JobPosting from raw text, retrying with error feedback.

    mode="baseline" disables retries and tools and uses a schema-free prompt; output is still
    validated afterwards so it can be measured. Pass logger=False to skip SQLite logging.
    """
    from .retry import run_with_retries

    llm = llm or get_llm()
    if max_retries is None:
        max_retries = int(os.getenv("MAX_RETRIES", "3"))
    if mode == "baseline":
        max_retries, tools = 0, {}
    if logger is None:
        logger = AttemptLogger()

    return run_with_retries(
        llm,
        system=build_system_prompt(mode, tools),
        user_prompt=USER_TEMPLATE.format(text=text),
        max_retries=max_retries,
        tools=tools,
        logger=logger or None,
        run_id=uuid.uuid4().hex[:12],
        input_hash=input_hash(text),
        backoff_s=backoff_s,
    )
