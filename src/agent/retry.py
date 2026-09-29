"""Retry loop with error feedback, max attempts, optional backoff."""

from __future__ import annotations

import json
import time
from typing import Optional

from .extractor import build_retry_prompt, interpret
from .llm import LLM, LLMError
from .logger import AttemptLogger
from .schemas import CallRecord, ExtractionResult
from .tools import Tool, ToolError, run_tool

# Successful tool calls don't consume retries, but are capped so a model can't loop on them.
MAX_TOOL_CALLS = 3


def run_with_retries(
    llm: LLM,
    *,
    system: str,
    user_prompt: str,
    max_retries: int,
    tools: dict[str, Tool],
    logger: Optional[AttemptLogger],
    run_id: str,
    input_hash: str,
    backoff_s: float = 0.0,
) -> ExtractionResult:
    """Call the LLM until it yields a valid JobPosting or `max_retries` failures follow the first try.

    Every call is recorded; a failed run ends with a `max_retries_exceeded` record.
    """
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    calls: list[CallRecord] = []
    failures = 0
    tool_calls = 0
    last_error: Optional[str] = None

    def record(rec: CallRecord) -> None:
        calls.append(rec)
        if logger:
            logger.log(run_id, input_hash, rec)

    while failures <= max_retries:
        attempt_no = len(calls) + 1
        try:
            resp = llm.complete(messages, system)
        except LLMError as e:
            last_error = str(e)
            record(CallRecord(attempt_no=attempt_no, raw_output=None, error_type="llm_error",
                              error_detail=last_error))
            failures += 1
            _sleep(backoff_s, failures)
            continue

        base = dict(attempt_no=attempt_no, raw_output=resp.text, latency_ms=resp.latency_ms,
                    tokens_in=resp.tokens_in, tokens_out=resp.tokens_out)
        parsed = interpret(resp.text, allow_tools=bool(tools))
        messages.append({"role": "assistant", "content": resp.text or "(empty response)"})

        if parsed.kind == "ok":
            record(CallRecord(**base, success=True))
            return ExtractionResult(run_id=run_id, ok=True, posting=parsed.posting,
                                    attempts=attempt_no, calls=calls)

        if parsed.kind == "tool":
            call = parsed.tool_call
            try:
                if tool_calls >= MAX_TOOL_CALLS:
                    raise ToolError(f"tool call limit ({MAX_TOOL_CALLS}) reached; return the final JSON now",
                                    "tool_response_invalid")
                tool_calls += 1
                result = run_tool(call.name, call.arguments, tools)
            except ToolError as e:
                parsed.error_type, parsed.error_detail = e.error_type, str(e)
            else:
                record(CallRecord(**base, error_detail=f"tool_call:{call.name}"))
                messages.append({"role": "user", "content":
                                 f"TOOL_RESULT {call.name}: {json.dumps(result.model_dump())}"})
                continue

        # Parse, validation, or tool failure: feed the exact error back.
        last_error = f"{parsed.error_type}: {parsed.error_detail}"
        record(CallRecord(**base, error_type=parsed.error_type, error_detail=parsed.error_detail))
        failures += 1
        if failures <= max_retries:
            messages.append({"role": "user",
                             "content": build_retry_prompt(parsed.error_type, parsed.error_detail)})
            _sleep(backoff_s, failures)

    record(CallRecord(attempt_no=len(calls), raw_output=None, error_type="max_retries_exceeded",
                      error_detail=last_error))
    return ExtractionResult(run_id=run_id, ok=False, attempts=len(calls) - 1, last_error=last_error,
                            error_type="max_retries_exceeded", calls=calls)


def _sleep(backoff_s: float, failures: int) -> None:
    if backoff_s > 0:
        time.sleep(backoff_s * 2 ** (failures - 1))
