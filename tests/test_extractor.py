import json

import pytest

from agent.extractor import JSONParseError, extract, interpret, parse_json
from agent.llm import FakeLLM
from agent.logger import AttemptLogger

from .conftest import as_json


# --- Parsing -----------------------------------------------------------------

def test_fenced_json_parses(valid_json):
    assert parse_json(f"```json\n{valid_json}\n```")["company"] == "Acme Corp"


def test_prose_wrapped_json_parses(valid_json):
    text = f"Sure! Here is the data:\n{valid_json}\nLet me know if you need anything else {{:"
    assert parse_json(text)["title"] == "Senior Backend Engineer"


def test_no_json_raises():
    with pytest.raises(JSONParseError):
        parse_json("I couldn't find a job posting here.")


# --- Error classification -------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("not json at all", "json_parse"),
    (as_json(company=...), "schema_missing_field"),
    (as_json(years_experience_min="five"), "schema_type"),
    (as_json(work_mode="remote-ish"), "schema_enum"),
    (as_json(salary={"min": 90000, "max": 70000, "currency": "EUR", "period": "year"}), "cross_field"),
    ('{"tool_call": {"arguments": {}}}', "tool_response_invalid"),
])
def test_each_error_type_is_classified(raw, expected):
    parsed = interpret(raw)
    assert parsed.kind == "error"
    assert parsed.error_type == expected


# --- Retry loop -------------------------------------------------------------------

def test_clean_output_succeeds_first_try(valid_json):
    res = extract("posting", FakeLLM([valid_json]), logger=False)
    assert res.ok and res.attempts == 1
    assert res.posting.company == "Acme Corp"


def test_retry_prompt_contains_previous_error_and_output(valid_json):
    bad = as_json(work_mode="remote-ish")
    llm = FakeLLM([bad, valid_json])
    res = extract("posting", llm, logger=False)
    assert res.ok and res.attempts == 2
    second_prompt = llm.calls[1]
    assert second_prompt[-2] == {"role": "assistant", "content": bad}
    assert "schema_enum" in second_prompt[-1]["content"]
    assert "remote-ish" in second_prompt[-1]["content"]


def test_retry_stops_at_max_retries():
    llm = FakeLLM(["nope"] * 10)
    res = extract("posting", llm, max_retries=3, logger=False)
    assert not res.ok
    assert len(llm.calls) == 4  # first try + 3 retries
    assert res.error_type == "max_retries_exceeded"
    assert res.calls[-1].error_type == "max_retries_exceeded"


def test_zero_retries_means_single_call():
    llm = FakeLLM(["nope", "nope"])
    assert not extract("posting", llm, max_retries=0, logger=False).ok
    assert len(llm.calls) == 1


def test_baseline_never_retries(valid_json):
    llm = FakeLLM(["nope", valid_json])
    res = extract("posting", llm, mode="baseline", max_retries=5, logger=False)
    assert not res.ok and len(llm.calls) == 1


# --- Tools ------------------------------------------------------------------------

def _tool_call(raw):
    return json.dumps({"tool_call": {"name": "normalize_currency", "arguments": {"raw": raw}}})


def test_tool_result_is_fed_back(valid_json):
    llm = FakeLLM([_tool_call("€"), valid_json])
    res = extract("posting", llm, logger=False)
    assert res.ok and res.attempts == 2
    tool_msg = llm.calls[1][-1]["content"]
    assert tool_msg.startswith("TOOL_RESULT normalize_currency")
    assert '"code": "EUR"' in tool_msg


def test_invalid_tool_response_is_rejected_and_retried(valid_json):
    from agent.schemas import CurrencyNormalization
    from agent.tools import Tool

    broken = {"normalize_currency": Tool(
        name="normalize_currency", description="broken", parameters={"raw": "x"},
        fn=lambda raw: {"raw": raw, "code": "dollars", "recognized": True},
        response_model=CurrencyNormalization,
    )}
    llm = FakeLLM([_tool_call("$"), valid_json])
    res = extract("posting", llm, tools=broken, logger=False)
    assert res.ok
    assert res.calls[0].error_type == "tool_response_invalid"
    assert "tool_response_invalid" in llm.calls[1][-1]["content"]


def test_unknown_tool_is_rejected(valid_json):
    call = json.dumps({"tool_call": {"name": "lookup_salary", "arguments": {}}})
    res = extract("posting", FakeLLM([call, valid_json]), logger=False)
    assert res.ok and res.calls[0].error_type == "tool_response_invalid"


def test_tool_calls_are_capped():
    llm = FakeLLM([_tool_call("$")] * 20)
    res = extract("posting", llm, max_retries=1, logger=False)
    assert not res.ok
    assert len(llm.calls) == 5  # 3 allowed tool calls + 2 over-limit failures


# --- Logging ----------------------------------------------------------------------

def test_attempts_are_logged_to_sqlite(tmp_path, valid_json):
    logger = AttemptLogger(tmp_path / "t.db")
    res = extract("posting", FakeLLM(["nope", as_json(work_mode="x"), valid_json]), logger=logger)
    rows = logger.rows(res.run_id)
    assert [r["error_type"] for r in rows] == ["json_parse", "schema_enum", None]
    assert [r["success"] for r in rows] == [0, 0, 1]
    assert [r["attempt_no"] for r in rows] == [1, 2, 3]
    assert rows[0]["raw_output"] == "nope"


def test_failure_is_logged_as_max_retries_exceeded(tmp_path):
    logger = AttemptLogger(tmp_path / "t.db")
    res = extract("posting", FakeLLM(["nope"] * 3), max_retries=1, logger=logger)
    assert [r["error_type"] for r in logger.rows(res.run_id)] == [
        "json_parse", "json_parse", "max_retries_exceeded"]
